"""
Sığırcık Dikkat Mekanizması (Starling Attention)
==================================================
Parisi'nin sığırcık sürüsü modelinden esinlenen sliding window dikkat.

Geleneksel self-attention O(N²) bellek kullanırken, bu mekanizma
her token'ın sadece k komşusuyla etkileşime girmesini sağlayarak
O(N·k) karmaşıklığa düşürür.

Temel Prensipler:
- Her kuş (token) yalnızca en yakın 7 komşusuna bakar (Parisi kuralı)
- Reynolds kuralları (ayrılma, hizalanma, uyum) dikkat ağırlıklarını yönlendirir
- Çoklu ölçekli pencereler farklı mesafelerdeki bağlamı yakalar
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from swarm_llm.config import SwarmConfig


def _build_sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: torch.device,
    causal: bool = True,
) -> torch.Tensor:
    """
    Vektörize edilmiş sliding window maskesi oluşturur.

    For döngüsü KULLANMAZ -- tamamen vektörize.
    Çıktı: (seq_len, seq_len) boyutunda bool tensor.
    True = maskelenmeli (dikkat edilmemeli).

    Args:
        seq_len: Dizi uzunluğu
        window_size: Pencere boyutu (örn. 7)
        device: Hedef cihaz
        causal: True ise sadece geçmişe bakılır (autoregressive)
    """
    # i-j mesafe matrisini hesapla: |i - j|
    positions = torch.arange(seq_len, device=device)
    # (seq_len, 1) - (1, seq_len) => (seq_len, seq_len)
    distance = (positions.unsqueeze(1) - positions.unsqueeze(0))

    half_w = window_size // 2

    if causal:
        # Sadece geçmişe bak: j <= i ve i - j < window_size
        mask = (distance < 0) | (distance >= window_size)
    else:
        # Çift yönlü: |i - j| <= half_w
        mask = distance.abs() > half_w

    return mask  # True = maskelenmeli


class StarlingAttention(nn.Module):
    """
    Sığırcık Dikkat Mekanizması (Starling Attention)

    Her token (kuş) yalnızca belirli sayıda komşuya dikkat eder.
    Unfold tabanlı verimli hesaplama ile O(N·w) bellek kullanır.

    Multi-head attention destekler ve Reynolds kurallarını
    (ayrılma, hizalanma, uyum) dikkat hesaplamasına entegre eder.
    """

    def __init__(self, config: SwarmConfig, causal: bool = True):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.neighbor_size = config.neighbor_size
        self.causal = causal
        self.scale = math.sqrt(self.head_dim)

        # QKV projeksiyon katmanları
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

        # Reynolds kuralı ağırlıkları (öğrenilebilir)
        self.separation_gate = nn.Parameter(
            torch.tensor(config.separation_weight)
        )
        self.alignment_gate = nn.Parameter(
            torch.tensor(config.alignment_weight)
        )
        self.cohesion_gate = nn.Parameter(
            torch.tensor(config.cohesion_weight)
        )

        self.attn_dropout = nn.Dropout(config.dropout)

        # Maske önbelleği (aynı seq_len için tekrar hesaplamamak adına)
        self._cached_mask: torch.Tensor | None = None
        self._cached_len: int = 0

    def _get_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Maske oluştur veya önbellekten al."""
        if self._cached_mask is not None and self._cached_len == seq_len:
            return self._cached_mask.to(device)

        mask = _build_sliding_window_mask(
            seq_len, self.neighbor_size, device, self.causal
        )
        self._cached_mask = mask
        self._cached_len = seq_len
        return mask

    def _apply_reynolds_rules(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        raw_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reynolds Kurallarını dikkat skorlarına uygular.

        1. Ayrılma (Separation): Çok benzer (yakın) token'ları iter,
           aşırı tekrarlamayı engeller.
        2. Hizalanma (Alignment): Komşuların yönelimine uyum sağlar,
           bağlamsal tutarlılık.
        3. Uyum (Cohesion): Grup merkezine doğru çekim,
           genel anlam bütünlüğü.
        """
        # --- Ayrılma (Separation) ---
        # Çok yüksek benzerlik skorlarını bastır (tekrar önleme)
        # Sigmoid ile [0,1] arasına sıkıştırılmış benzerlik
        similarity = torch.sigmoid(raw_scores)
        separation_penalty = -self.separation_gate * (similarity ** 2)

        # --- Hizalanma (Alignment) ---
        # Komşu key vektörlerinin ortalamasına yakınlık
        # Bu zaten standart attention'ın doğal davranışı, ağırlıkla güçlendiriyoruz
        alignment_bonus = self.alignment_gate * raw_scores

        # --- Uyum (Cohesion) ---
        # Skor dağılımını ortalamaya doğru çek (aşırı uçları yumuşat)
        score_mean = raw_scores.mean(dim=-1, keepdim=True)
        cohesion_pull = -self.cohesion_gate * (raw_scores - score_mean).abs()

        # Nihai skor = orijinal + Reynolds modifikasyonları
        modified_scores = raw_scores + separation_penalty + alignment_bonus + cohesion_pull
        return modified_scores

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        İleri geçiş.

        Args:
            x: (batch, seq_len, embed_dim) giriş tensörü
            past_kv: Önceki key-value çifti (otoregresif üretim için)

        Returns:
            out: (batch, seq_len, embed_dim) çıktı
            (k, v): Önbellek için key-value çifti
        """
        B, L, D = x.shape

        # QKV projeksiyon
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        # Şekil: (B, num_heads, L, head_dim)

        # KV önbelleği (otoregresif üretim)
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        kv_cache = (k, v)

        kv_len = k.shape[2]

        # --- Dikkat Skorları ---
        # (B, H, L, head_dim) @ (B, H, head_dim, kv_len) => (B, H, L, kv_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # --- Reynolds Kuralları ---
        scores = self._apply_reynolds_rules(q, k, scores)

        # --- Sliding Window Maskesi ---
        # Verimli, vektörize maske (for döngüsü YOK)
        mask = self._get_mask(kv_len, x.device)

        # Sorgu pozisyonlarını ayarla (past_kv varsa offset)
        if past_kv is not None and L < kv_len:
            # Üretim modunda: sadece son L token için maske satırları
            mask = mask[-L:]

        # Maskeyi uygula: maskelenen pozisyonlara -inf
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # --- Softmax & Dropout ---
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # --- Değer Toplama ---
        # (B, H, L, kv_len) @ (B, H, kv_len, head_dim) => (B, H, L, head_dim)
        out = torch.matmul(attn_weights, v)

        # Head'leri birleştir
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        return out, kv_cache


class MultiScaleStarlingAttention(nn.Module):
    """
    Çoklu Ölçekli Sığırcık Dikkat Mekanizması

    Gerçek sığırcık sürülerinde olduğu gibi, kuşlar hem
    yakın komşularını hem de sürünün genel hareketini algılar.

    Bu modül farklı pencere boyutlarında (7, 21, 63) dikkat hesaplar
    ve sonuçları birleştirir. Böylece:
    - 7: Kelime düzeyinde yerel bağlam
    - 21: Cümle düzeyinde orta bağlam
    - 63: Paragraf düzeyinde geniş bağlam
    """

    def __init__(self, config: SwarmConfig, causal: bool = True):
        super().__init__()
        self.windows = config.multi_scale_windows
        self.num_scales = len(self.windows)

        # Her ölçek için ayrı dikkat katmanı
        self.scale_attentions = nn.ModuleList()
        for w in self.windows:
            scale_config = SwarmConfig(
                vocab_size=config.vocab_size,
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
                neighbor_size=w,
                noise_strength=config.noise_strength,
                noise_learnable=config.noise_learnable,
                separation_weight=config.separation_weight,
                alignment_weight=config.alignment_weight,
                cohesion_weight=config.cohesion_weight,
            )
            self.scale_attentions.append(StarlingAttention(scale_config, causal))

        # Ölçekler arası karışım ağırlıkları (öğrenilebilir)
        self.scale_gates = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Çoklu ölçekli dikkat.

        Her pencere boyutunda dikkat hesaplanır, sonra
        öğrenilebilir ağırlıklarla birleştirilir.
        """
        gates = F.softmax(self.scale_gates, dim=0)

        if past_kv is None:
            past_kv = [None] * self.num_scales

        combined = torch.zeros_like(x)
        new_kv_caches = []

        for i, (attn, gate) in enumerate(zip(self.scale_attentions, gates)):
            out, kv = attn(x, past_kv[i])
            combined = combined + gate * out
            new_kv_caches.append(kv)

        return combined, new_kv_caches
