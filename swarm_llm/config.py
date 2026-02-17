"""
Swarm-LLM Model Konfigürasyonu
================================
Parisi'nin sığırcık sürüsü modelinden esinlenen tüm hiperparametreler.
"""

from dataclasses import dataclass


@dataclass
class SwarmConfig:
    """Swarm-LLM için tüm model hiperparametrelerini tutan konfigürasyon."""

    # --- Temel Model Parametreleri ---
    vocab_size: int = 32_000
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    max_seq_len: int = 4096
    dropout: float = 0.1

    # --- Sığırcık (Starling) Dikkat Parametreleri ---
    neighbor_size: int = 7        # Parisi'nin "7 Komşu" kuralı
    use_multi_scale: bool = True  # Çoklu ölçekli pencereler (7, 21, 63)

    # --- Parisi Gürültü (η) Parametreleri ---
    noise_strength: float = 0.02  # η: Gürültü şiddeti (yaratıcılık kontrolü)
    noise_learnable: bool = True  # Gürültünün öğrenilebilir olup olmadığı

    # --- Reynolds Kuralları Ağırlıkları ---
    separation_weight: float = 0.3   # Ayrılma: Çok yakın token'ları itme
    alignment_weight: float = 0.5    # Hizalanma: Komşu yönlerine uyum
    cohesion_weight: float = 0.2     # Uyum: Sürü merkezine çekim

    # --- FFN (Feed-Forward Network) ---
    ffn_multiplier: int = 4  # FFN gizli katman boyutu = embed_dim * ffn_multiplier

    # --- Nash MoE (Mixture of Experts) Parametreleri ---
    use_nash_moe: bool = False    # Nash MoE kullanılsın mı?
    num_experts: int = 8          # Expert sayısı
    top_k_experts: int = 2        # Her token için aktif expert sayısı
    nash_warmup_steps: int = 100  # Nash routing ısınma adımı

    # --- Eğitim ---
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100_000

    @property
    def head_dim(self) -> int:
        assert self.embed_dim % self.num_heads == 0, \
            f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
        return self.embed_dim // self.num_heads

    @property
    def multi_scale_windows(self) -> list[int]:
        """Çoklu ölçekli pencere boyutları: yerel, orta, geniş."""
        if self.use_multi_scale:
            return [
                self.neighbor_size,          # 7:  Yakın komşular
                self.neighbor_size * 3,      # 21: Orta mesafe
                self.neighbor_size * 9,      # 63: Geniş bağlam
            ]
        return [self.neighbor_size]
