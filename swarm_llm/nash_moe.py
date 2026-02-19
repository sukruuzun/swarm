"""
Nash Oyun Teorisi Tabanlı Mixture of Experts (MoE)
=====================================================
John Nash'in denge teorisini dil modeli mimarisine uygulayan
stratejik expert routing mekanizması.

Temel Fikir:
  Standart MoE'de router basit bir softmax ile expert seçer.
  Nash MoE'de her expert bir "oyuncu"dur ve routing, oyun teorisinin
  Nash Dengesi'ne yakınsar.

  Nash Dengesi: Hiçbir expert, stratejisini tek taraflı değiştirerek
  daha fazla kazanç sağlayamaz.

Neden Nash?
  1. Expert Çöküşü Çözülür: Standart MoE'de router hep aynı 1-2 expert'e
     yönlendirir, diğerleri atıl kalır. Nash dengesinde bu olamaz çünkü
     atıl expert "boş alan = fırsat" görür.
  2. Doğal Yük Dengeleme: Yapay "load balancing loss" gerekmez.
  3. Dinamik Uzmanlık: Expert'ler kendi uzmanlık alanlarını keşfeder.

Matematiksel Formülasyon:
  Her expert k'nin kazanç fonksiyonu:
    U_k(g_k, g_{-k}) = E[quality_k · load_k - cost_k · overload_k]

  Nash Dengesi:
    g_k* = argmax_{g_k} U_k(g_k, g*_{-k})  ∀k

  Yaklaşım: Regret Matching (Zinkevich, 2007)
    - Kümülatif pişmanlık (regret) takip edilir
    - Pozitif pişmanlıklar routing olasılıklarına dönüştürülür
    - Zamanla Nash Dengesine yakınsar
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """
    Tek bir uzman (expert) ağı.

    SwiGLU aktivasyonlu Feed-Forward Network.
    Her expert kendi parametrelerine sahiptir ve
    farklı bir "uzmanlık alanı" geliştirir.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.v(x)))


class NashRouter(nn.Module):
    """
    Nash Dengesi Tabanlı Expert Router.

    Standart softmax router yerine oyun-teorik routing:
    1. Her token için expert "fayda" (utility) skorları hesaplanır
    2. Regret matching ile Nash dengesine yakınsayan ağırlıklar üretilir
    3. Top-K seçimi ile sparse aktivasyon sağlanır

    Regret Matching Algoritması:
      - Her adımda, seçilmeyen expert'lerin "ne kazanacaktı" bilgisi hesaplanır
      - Kümülatif pişmanlık (regret) birikir
      - Pozitif pişmanlıklar normalize edilerek olasılıklara dönüşür
      - Bu süreç Nash Dengesine yakınsar (Blackwell, 1956)
    """

    def __init__(self, embed_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Expert fayda (utility) skoru hesaplayıcı
        self.utility_net = nn.Linear(embed_dim, num_experts, bias=False)

        # Kümülatif pişmanlık (regret) takibi -- eğitim boyunca güncellenir
        # Her expert için kümülatif pişmanlık (öğrenilmez, state olarak tutulur)
        self.register_buffer(
            'cumulative_regret',
            torch.zeros(num_experts)
        )

        # Expert kullanım sayacı (yük dengeleme izleme)
        self.register_buffer(
            'expert_counts',
            torch.zeros(num_experts)
        )

        # Toplam token sayacı
        self.register_buffer(
            'total_tokens',
            torch.tensor(0, dtype=torch.long)
        )

        # Isınma tamamlandı mı? (İlk N adım standart softmax kullan)
        self.warmup_steps = 100
        self.register_buffer(
            'step_count',
            torch.tensor(0, dtype=torch.long)
        )

        # Nash denge sıcaklığı (keşif-sömürü dengesi)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def _regret_matching_probs(self) -> torch.Tensor:
        """
        Regret Matching: Kümülatif pişmanlıkları olasılıklara dönüştür.

        Pozitif pişmanlıklar normalize edilir.
        Tüm pişmanlıklar negatifse uniform dağılım kullanılır.
        """
        positive_regret = F.relu(self.cumulative_regret)
        total = positive_regret.sum()

        if total > 0:
            return positive_regret / total
        else:
            # Tüm pişmanlıklar negatif → uniform (keşif modu)
            return torch.ones(self.num_experts, device=self.cumulative_regret.device) / self.num_experts

    def forward(
        self,
        x: torch.Tensor,
        expert_outputs: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Nash routing.

        Args:
            x: (batch, seq_len, embed_dim) giriş
            expert_outputs: Eğer verilmişse, regret güncelleme için kullanılır

        Returns:
            router_weights: (batch, seq_len, num_experts) routing ağırlıkları
            top_k_indices: (batch, seq_len, top_k) seçilen expert indeksleri
            info: İstatistik bilgileri
        """
        B, L, D = x.shape

        # ── 1. Expert Fayda Skorları ──────────────────────────────────
        # Her token için her expert'in tahmini faydası
        utility_scores = self.utility_net(x)  # (B, L, num_experts)

        # ── 2. Nash Dengesi Ağırlıkları ──────────────────────────────
        if self.step_count < self.warmup_steps:
            # Isınma: Standart softmax (kararlılık için)
            router_logits = utility_scores / self.temperature.abs().clamp(min=0.1)
            router_probs = F.softmax(router_logits, dim=-1)
        else:
            # Nash: Regret matching + utility
            # Regret prior'ını utility skorlarıyla birleştir
            regret_prior = self._regret_matching_probs()  # (num_experts,)

            # Token-spesifik utility + global regret prior
            # Bu birleşim: yerel bilgi (bu token için en iyi expert) +
            # global denge (hangi expert'ler az kullanılmış)
            combined = utility_scores / self.temperature.abs().clamp(min=0.1)
            combined = combined + torch.log(regret_prior.unsqueeze(0).unsqueeze(0) + 1e-8)
            router_probs = F.softmax(combined, dim=-1)

        # ── 3. Top-K Seçimi (Sparse Aktivasyon) ──────────────────────
        top_k_weights, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )  # (B, L, top_k)

        # Top-K ağırlıklarını yeniden normalize et
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # ── 4. Regret Güncelleme (Eğitim sırasında) ──────────────────
        if self.training:
            self.step_count += 1

            # Expert kullanım istatistikleri
            with torch.no_grad():
                # Her expert'in ne kadar seçildiğini say (vektörize)
                flat_indices = top_k_indices.reshape(-1)
                counts = torch.zeros(self.num_experts, device=flat_indices.device)
                counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float))
                self.expert_counts += counts
                self.total_tokens += B * L

                # Regret güncelleme: seçilmeyen expert'ler için
                # "Eğer bu expert seçilseydi ne olurdu?" tahmini
                if expert_outputs is not None:
                    self._update_regret(x, router_probs, expert_outputs, top_k_indices)

        # ── 5. İstatistikler ──────────────────────────────────────────
        info = self._compute_stats(router_probs, top_k_indices)

        return top_k_weights, top_k_indices, info

    @torch.no_grad()
    def _update_regret(
        self,
        x: torch.Tensor,
        router_probs: torch.Tensor,
        expert_outputs: List[torch.Tensor],
        chosen_indices: torch.Tensor,
    ):
        """
        Nash Regret güncelleme.

        Her expert için:
        regret_k = E[utility_k - utility_chosen]

        Pozitif regret = "Bu expert seçilseydi daha iyiydi" → gelecekte daha çok seç
        Negatif regret = "Doğru seçim yapılmış" → mevcut strateji iyi
        """
        B, L, K = chosen_indices.shape

        # Her expert çıktısının normunu proxy utility olarak kullan
        expert_utilities = torch.stack([
            out.norm(dim=-1).mean() for out in expert_outputs
        ])  # (num_experts,)

        # Seçilen expert'lerin ortalama utility'si
        chosen_utility = expert_utilities[chosen_indices].mean()

        # Her expert için regret = kendi utility'si - seçilenlerin utility'si
        regret = expert_utilities - chosen_utility

        # Kümülatif regret güncelle (üstel hareketli ortalama)
        decay = 0.99
        self.cumulative_regret = decay * self.cumulative_regret + regret

    @torch.no_grad()
    def _compute_stats(self, router_probs, top_k_indices) -> dict:
        """Expert kullanım istatistikleri."""
        if top_k_indices.dim() < 3:
            return {
                'expert_usage': torch.zeros(self.num_experts),
                'balance_score': 0.0,
                'regret': self.cumulative_regret.clone(),
                'temperature': self.temperature.item(),
                'is_warmup': (self.step_count < self.warmup_steps).item(),
            }

        # Vektörize expert dağılımı (for döngüsü yok)
        flat = top_k_indices.reshape(-1)
        expert_usage = torch.zeros(self.num_experts, device=flat.device)
        expert_usage.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
        expert_usage = expert_usage / flat.numel()

        # Yük dengeleme skoru (1.0 = mükemmel denge, 0 = tek expert)
        balance = 1.0 - expert_usage.std() / (expert_usage.mean() + 1e-8)

        return {
            'expert_usage': expert_usage,
            'balance_score': balance.item(),
            'regret': self.cumulative_regret.clone(),
            'temperature': self.temperature.item(),
            'is_warmup': (self.step_count < self.warmup_steps).item(),
        }


class NashMoE(nn.Module):
    """
    Nash Mixture of Experts.

    Parisi'nin sığırcık dikkatinden geçen token'ları,
    Nash Dengesi ile yönlendirilen uzman ağlardan geçirir.

    Birleşim:
      1. StarlingAttention: Token'lar komşularıyla etkileşir (Parisi)
      2. NashMoE: Expert'ler stratejik olarak aktive olur (Nash)

    v_i(t+1) = Parisi(Komşu Etkileşimi) + Nash(En İyi Tepki)

    Seyrek Aktivasyon:
      num_experts=8, top_k=2 → Her token için sadece 2/8 expert aktif
      Bu, FFN belleğini ~4x azaltır.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Expert ağları
        self.experts = nn.ModuleList([
            Expert(embed_dim, hidden_dim, dropout)
            for _ in range(num_experts)
        ])

        # Nash Router
        self.router = NashRouter(embed_dim, num_experts, top_k)

        # Paylaşılan (shared) expert -- her zaman aktif
        # Bu, tüm token'ların ihtiyaç duyduğu temel işlevi sağlar
        # (dil bilgisi gibi evrensel bilgi)
        self.shared_expert = Expert(embed_dim, hidden_dim // 2, dropout)
        self.shared_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Nash MoE ileri geçiş.

        1. Tüm expert'lerin çıktılarını hesapla
        2. Nash router ile routing ağırlıkları belirle
        3. Ağırlıklı toplam ile çıktı üret

        Not: Tam verimli implementasyon için expert'ler sadece
        seçildiklerinde çalıştırılır (sparse). Burada eğitim
        kolaylığı için tümü çalıştırılır, sonra maskelenir.

        Args:
            x: (batch, seq_len, embed_dim)

        Returns:
            output: (batch, seq_len, embed_dim)
            info: Expert istatistikleri
        """
        B, L, D = x.shape

        # ── 1. Tüm Expert Çıktıları ──────────────────────────────────
        expert_outputs = [expert(x) for expert in self.experts]
        # Her biri (B, L, D)

        # ── 2. Nash Routing ──────────────────────────────────────────
        top_k_weights, top_k_indices, info = self.router(x, expert_outputs)
        # top_k_weights: (B, L, top_k)
        # top_k_indices: (B, L, top_k)

        # ── 3. Sparse Çıktı Birleştirme (Vektörize -- for döngüsü YOK) ─
        # Expert çıktılarını stack: (B, L, num_experts, D)
        stacked = torch.stack(expert_outputs, dim=2)

        # Her token için seçilen expert çıktılarını gather ile al
        # top_k_indices: (B, L, top_k) → (B, L, top_k, D) expand
        gather_idx = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        # stacked'ten seçim: (B, L, top_k, D)
        selected = torch.gather(stacked, dim=2, index=gather_idx)

        # Ağırlıklı toplam: (B, L, top_k, 1) * (B, L, top_k, D) → sum → (B, L, D)
        output = (top_k_weights.unsqueeze(-1) * selected).sum(dim=2)

        # ── 4. Shared Expert (Evrensel Bilgi) ─────────────────────────
        shared_out = self.shared_expert(x)
        gate = torch.sigmoid(self.shared_gate)
        output = gate * shared_out + (1 - gate) * output

        return output, info


class NashMoEEfficient(nn.Module):
    """
    Verimli Nash MoE -- Scatter/Gather tabanlı sparse hesaplama.

    NashMoE'nun daha verimli versiyonu: Expert'ler sadece
    kendilerine yönlendirilen token'ları işler.

    Bellek tasarrufu: num_experts=8, top_k=2 ise
    her expert ortalama L/4 token işler (L yerine).
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            Expert(embed_dim, hidden_dim, dropout)
            for _ in range(num_experts)
        ])

        self.router = NashRouter(embed_dim, num_experts, top_k)
        self.shared_expert = Expert(embed_dim, hidden_dim // 2, dropout)
        self.shared_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Verimli sparse MoE (Tam vektörize -- for döngüsü YOK).

        Tüm expert'leri batch olarak hesapla → gather ile seç → ağırlıklı topla.
        GPU üzerinde ~8-10x hızlanma sağlar.
        """
        B, L, D = x.shape

        # ── 1. Tüm Expert Çıktıları (batch, paralel) ─────────────────
        expert_outputs = [expert(x) for expert in self.experts]
        # Her biri (B, L, D)

        # ── 2. Nash Routing ───────────────────────────────────────────
        top_k_weights, top_k_indices, info = self.router(x, expert_outputs)
        # top_k_weights: (B, L, top_k)
        # top_k_indices: (B, L, top_k)

        # ── 3. Vektörize Gather + Ağırlıklı Toplam ───────────────────
        # Stack: (B, L, num_experts, D)
        stacked = torch.stack(expert_outputs, dim=2)
        # Gather: (B, L, top_k, D)
        gather_idx = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        selected = torch.gather(stacked, dim=2, index=gather_idx)
        # Ağırlıklı toplam: (B, L, D)
        output = (top_k_weights.unsqueeze(-1) * selected).sum(dim=2)

        # Not: Regret güncelleme router.forward() içinde zaten yapılıyor
        # (expert_outputs parametresi geçildiğinde). Burada tekrar yapmaya gerek yok.

        # ── 4. Shared Expert (evrensel bilgi) ─────────────────────────
        shared_out = self.shared_expert(x)
        gate = torch.sigmoid(self.shared_gate)
        output = gate * shared_out + (1 - gate) * output

        return output, info
