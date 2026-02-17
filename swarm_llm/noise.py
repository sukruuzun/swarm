"""
Parisi Gürültü Modülü (η)
===========================
Parisi'nin istatistik mekaniği formülasyonundaki gürültü (η) parametresi,
sığırcık sürüsünde kuşların bireysel rastgele hareketlerini temsil eder.

Dil modeli bağlamında bu:
- Düşük η → Deterministik, tutarlı çıktı (greedy decoding benzeri)
- Yüksek η → Yaratıcı, çeşitli çıktı (yüksek temperature benzeri)
- Öğrenilebilir η → Model kendi "yaratıcılık" seviyesini öğrenir

Parisi Hamiltonian'ı:
    H = -Σ J_ij · s_i · s_j + η · Σ ξ_i · s_i

Burada:
    J_ij : Token'lar arası etkileşim gücü (dikkat ağırlıkları)
    s_i  : Token durumu (gizli temsil)
    η    : Gürültü şiddeti
    ξ_i  : Rasgele Gauss gürültüsü
"""

import torch
import torch.nn as nn


class ParisiNoise(nn.Module):
    """
    Parisi Gürültü Enjeksiyonu.

    Dikkat çıktısına kontrollü stokastik gürültü ekler.
    Gürültü şiddeti (η) isteğe bağlı olarak öğrenilebilir.

    Gürültü yalnızca eğitim sırasında aktiftir (eval modunda kapatılır).
    Ancak isteğe bağlı olarak üretim zamanında da açılabilir
    (yaratıcı metin üretimi için).
    """

    def __init__(self, embed_dim: int, noise_strength: float = 0.02, learnable: bool = True):
        super().__init__()
        self.embed_dim = embed_dim

        if learnable:
            # η öğrenilebilir parametre (her boyut için ayrı)
            self.eta = nn.Parameter(torch.full((embed_dim,), noise_strength))
        else:
            # Sabit η
            self.register_buffer('eta', torch.full((embed_dim,), noise_strength))

        # Gürültü ölçekleme: Katman normalizasyonu sonrası
        # gürültünün etkisini kontrol eden ekstra kapı
        self.noise_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

        self._force_noise = False  # Eval modunda bile gürültü ekleme bayrağı

    @property
    def force_noise(self) -> bool:
        """Eval modunda gürültü eklemeyi zorla (yaratıcı üretim)."""
        return self._force_noise

    @force_noise.setter
    def force_noise(self, value: bool):
        self._force_noise = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parisi gürültüsünü uygula.

        Args:
            x: (batch, seq_len, embed_dim) dikkat çıktısı

        Returns:
            Gürültü eklenmiş tensör (aynı şekil)
        """
        if not self.training and not self._force_noise:
            return x

        # ξ ~ N(0, 1) rasgele gürültü
        xi = torch.randn_like(x)

        # Gürültü kapısı: giriş bağımlı ölçekleme
        # Bazı pozisyonlarda gürültüye daha açık, bazılarında kapalı
        gate = self.noise_gate(x)  # (B, L, D)

        # η · gate · ξ
        noise = self.eta * gate * xi

        return x + noise


class AdaptiveParisiNoise(nn.Module):
    """
    Uyarlanabilir Parisi Gürültüsü.

    Standart ParisiNoise'dan farklı olarak, gürültü şiddetini
    girdinin "belirsizliğine" göre dinamik olarak ayarlar.

    Yüksek belirsizlik → Daha fazla gürültü (keşif)
    Düşük belirsizlik → Daha az gürültü (sömürü)

    Bu, sığırcık sürüsünde tehlike anında kuşların daha
    rasgele hareket etmesine benzer.
    """

    def __init__(self, embed_dim: int, noise_strength: float = 0.02):
        super().__init__()
        self.base_noise = ParisiNoise(embed_dim, noise_strength, learnable=True)

        # Belirsizlik tahmincisi
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Belirsizliğe dayalı uyarlanabilir gürültü.

        Args:
            x: (batch, seq_len, embed_dim) giriş

        Returns:
            Uyarlanabilir gürültü eklenmiş çıktı
        """
        if not self.training and not self.base_noise.force_noise:
            return x

        # Token bazında belirsizlik skoru [0, 1]
        uncertainty = self.uncertainty_estimator(x)  # (B, L, 1)

        # Temel gürültüyü hesapla
        noisy = self.base_noise(x)

        # Belirsizliğe göre gürültü miktarını ayarla
        # Yüksek belirsizlik → orijinal + daha çok gürültü
        # Düşük belirsizlik → orijinale daha yakın
        return x + uncertainty * (noisy - x)
