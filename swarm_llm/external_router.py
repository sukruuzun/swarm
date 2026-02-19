"""
External Parisi-Nash Router (Dışsal Yönlendirici)
==================================================
Eğitilmiş dev modelleri (Llama, Qwen vb.) "dilimleyip" yönetmek için:
router, hangi blokların (layer grupları) bu forward'da RAM'e yüklenip
çalıştırılacağına karar verir.

TEORİK TEMELLER:
================

1. NASH DENGESİ (Cumulative Regret → Logit Düzeltmesi)
   ────────────────────────────────────────────────────
   Her blok bir "oyuncu". Eğer bir blok çok az seçiliyorsa, regret birikir.
   Biriken regret, o bloğun logit'ine BONUS olarak eklenir → daha fazla
   seçilir → denge sağlanır. Hiçbir bloğun "ben daha çok seçilseydim
   daha iyi olurdu" dememesi = Nash dengesi.

   Formül: adjusted_logits = gate(x)/T + α × cumulative_regret

2. PARİSİ TAVLAMA (Simulated Annealing)
   ─────────────────────────────────────
   Temperature yüksek başlar (keşif), zamanla düşer (sömürü).
   Erken dönemde router tüm blokları keşfeder, sonra en iyilere yakınsar.
   
   Formül: T(t) = T_initial × (T_min/T_initial)^(t/T_anneal)
   
   Bu, Parisi'nin spin-glass modelindeki soğutma çizelgesinin analojisi.

3. SIĞIRCIK SÜRÜ ZEKASI (Boid Rules)
   ──────────────────────────────────
   Sığırcık kuşları 3 kurala uyar: ayrılma, hizalama, bağlılık.
   Router karşılığı:
   - Ayrılma: Az kullanılan bloklara exploration noise → sürü dağılır
   - Hizalama: Tüm blokların kullanımı dengeye çekilir (load balancing)  
   - Bağlılık: Block usage EMA ile sürü hafızası → anlık dalgalanma önlenir

Kullanım:
  - MoE-fication: Dense 70B → N blok (örn. 80 layer → 8 expert x 10 layer)
  - Her forward'da router sadece top_k blok seçer → RAM'de ~2B parametre
  - External Router çok küçük (embed_dim → 64 → num_blocks), hızlı
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExternalParisiNashRouter(nn.Module):
    """
    Parisi-Nash-Sığırcık tabanlı blok seçici.
    
    Teori → Kod Bağlantıları:
    ─────────────────────────
    • cumulative_regret → logit'lere eklenir (Nash dengesi)
    • current_temperature → otomatik soğuma (Parisi tavlama)
    • exploration_noise → az kullanılan bloklara keşif bonusu (Sığırcık ayrılma)
    • block_usage → EMA ile sürü hafızası (Sığırcık bağlılık)
    • aux_loss → load balancing (Sığırcık hizalama)
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        top_k: int = 2,
        router_hidden: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.top_k = top_k

        # MLP Gate: Embedding → Blok logitleri
        self.gate = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, router_hidden, bias=False),
            nn.GELU(),
            nn.Linear(router_hidden, num_blocks, bias=False),
        )

        # ── PARİSİ TAVLAMA ──
        self.register_buffer("current_temperature", torch.tensor(2.0))
        self.register_buffer("initial_temperature", torch.tensor(2.0))
        self.register_buffer("min_temperature", torch.tensor(0.3))
        self.register_buffer("anneal_steps", torch.tensor(500, dtype=torch.long))
        
        # ── NASH DENGESİ ──
        self.register_buffer("cumulative_regret", torch.zeros(num_blocks))
        self.register_buffer("regret_weight", torch.tensor(0.5))  # α: regret etkisi
        
        # ── SIĞIRCIK SÜRÜ ZEKASI ──
        self.register_buffer("block_usage", torch.zeros(num_blocks))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

    def set_temperature(self, t: float):
        self.current_temperature.fill_(t)

    def _anneal_temperature(self):
        """PARİSİ TAVLAMA: Temperature'ı otomatik soğut.
        
        T(t) = T_initial × (T_min / T_initial) ^ (t / anneal_steps)
        t=0 → T=T_initial (keşif modu)
        t=anneal_steps → T=T_min (sömürü modu)
        """
        if self.step_count >= self.anneal_steps:
            self.current_temperature.fill_(self.min_temperature.item())
            return
        
        ratio = self.step_count.float() / self.anneal_steps.float()
        T_new = self.initial_temperature * (
            self.min_temperature / self.initial_temperature
        ) ** ratio
        self.current_temperature.fill_(T_new.item())

    def _compute_exploration_noise(self):
        """SIĞIRCIK AYRIŞMA: Az kullanılan bloklara keşif bonusu.
        
        Sığırcık kuşları sürüde birbirinden uzaklaşma eğilimi gösterir
        (separation rule). Router karşılığı: Kullanım ortalamasının
        altında kalan bloklar bonus alır → sürü genişler.
        
        noise_i = max(0, mean(usage) - usage_i)
        """
        if self.block_usage.sum() < 1e-8:
            return torch.zeros_like(self.block_usage)
        
        mean_usage = self.block_usage.mean()
        # Az kullanılan bloklar pozitif noise alır (keşfe teşvik)
        noise = (mean_usage - self.block_usage).clamp(min=0)
        # Normalize et ki çok büyük olmasın
        if noise.max() > 1e-8:
            noise = noise / noise.max() * 0.5
        return noise

    def forward(
        self,
        x: torch.Tensor,
        pool_input: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, embed_dim) veya (B, embed_dim)
            pool_input: True ise x'i (B,L,D) → (1,1,D) pool edip tek karar üretir

        Returns:
            probs:    (B, L, num_blocks) veya (1, 1, num_blocks)
            indices:  (B, L, top_k) veya (1, 1, top_k)
            aux_loss: Load balancing loss (skaler)
            weights:  (B, L, top_k) seçilen blokların ağırlıkları (normalize)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, D) → (B, 1, D)
        if pool_input and x.size(1) > 1:
            x = x.mean(dim=(0, 1), keepdim=True).expand(1, 1, x.size(-1))

        # ── 1. MLP GATE: Ham logitler ──
        T = self.current_temperature.clamp(min=0.1)
        logits = self.gate(x) / T

        # ── 2. NASH DENGESİ: Regret düzeltmesi ──
        # Biriken pişmanlık logitlere eklenir → az seçilen bloklar avantaj kazanır
        # Bu, hiçbir bloğun "daha çok seçilseydim daha iyi olurdu" dememesini sağlar
        if not self.training:
            # İnference'da regret'i kullanarak dengeyi koru
            regret_bonus = self.cumulative_regret * self.regret_weight
            logits = logits + regret_bonus.unsqueeze(0).unsqueeze(0)
        
        # ── 3. SIĞIRCIK KEŞIF: Exploration noise ──
        if not self.training:
            exploration = self._compute_exploration_noise()
            logits = logits + exploration.unsqueeze(0).unsqueeze(0)

        probs = F.softmax(logits, dim=-1)

        _, indices = torch.topk(probs, self.top_k, dim=-1)
        # Seçilen blokların ağırlıkları (normalize)
        top_probs = torch.gather(probs, -1, indices)
        weights = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # ── 4. SIĞIRCIK HIZALAMA: Load balancing loss ──
        # Tüm blokların kullanımı dengeye çekilir (alignment rule)
        with torch.no_grad():
            one_hot = F.one_hot(indices, self.num_blocks).float()
            f = one_hot.sum(dim=(0, 1, 2)) / (
                indices.shape[0] * indices.shape[1] * self.top_k + 1e-8
            )
        P = probs.mean(dim=(0, 1))
        aux_loss = self.num_blocks * (f * P).sum()

        # ── 5. DURUM GÜNCELLEMESİ ──
        if self.training:
            self.step_count += 1
            with torch.no_grad():
                # Sığırcık Bağlılık: EMA ile usage takibi (sürü hafızası)
                self.block_usage = 0.99 * self.block_usage + 0.01 * f
                
                # Nash Regret: Uniform'dan sapma birikir
                uniform = torch.ones_like(f) / self.num_blocks
                self.cumulative_regret = 0.99 * self.cumulative_regret + (uniform - f)
                
                # Parisi Tavlama: Temperature soğut
                self._anneal_temperature()

        return probs, indices, aux_loss, weights

    def get_stats(self) -> dict:
        """Router istatistikleri (sıcaklık, kullanım, pişmanlık, teori durumu)."""
        return {
            "temperature": self.current_temperature.item(),
            "usage": self.block_usage.detach().clone(),
            "regret": self.cumulative_regret.detach().clone(),
            "step_count": self.step_count.item(),
            "exploration_noise": self._compute_exploration_noise().detach().clone(),
            "theory_status": {
                "nash_regret_active": self.cumulative_regret.abs().max().item() > 0.01,
                "parisi_annealing_active": self.current_temperature.item() > self.min_temperature.item(),
                "starling_exploration_active": self._compute_exploration_noise().max().item() > 0.01,
            }
        }

    @torch.no_grad()
    def get_predictive_indices(
        self,
        x: torch.Tensor,
        pool_input: bool = True,
    ) -> tuple[list[int], torch.Tensor]:
        """
        Teoreminin beyni: Modeli çalıştırmadan, sadece giriş embedding'ine
        bakarak hangi blokların gerekli olduğunu tahmin eder.
        
        Nash + Parisi + Sığırcık ETKİN:
        - Logitler regret ile düzeltilir (Nash)
        - Temperature annealing ile keşif/sömürü dengesi (Parisi)
        - Az kullanılan bloklar exploration bonus alır (Sığırcık)
        """
        self.eval()
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if pool_input and x.size(1) > 1:
            x = x.mean(dim=(0, 1), keepdim=True)

        T = self.current_temperature.clamp(min=0.1)
        logits = self.gate(x) / T
        
        # NASH: Regret düzeltmesi
        regret_bonus = self.cumulative_regret * self.regret_weight
        logits = logits + regret_bonus.unsqueeze(0).unsqueeze(0)
        
        # SIĞIRCIK: Exploration noise
        exploration = self._compute_exploration_noise()
        logits = logits + exploration.unsqueeze(0).unsqueeze(0)
        
        probs = F.softmax(logits, dim=-1)

        _, indices = torch.topk(probs, self.top_k, dim=-1)
        top_probs = torch.gather(probs, -1, indices)
        weights = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Squeeze ve CPU'ya al
        indices = indices.squeeze().cpu()
        weights = weights.squeeze().cpu()
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)
            weights = weights.unsqueeze(0)

        return indices.tolist(), weights
