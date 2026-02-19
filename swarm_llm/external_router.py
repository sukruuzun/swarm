"""
External Parisi-Nash Router (Dışsal Yönlendirici)
==================================================
Eğitilmiş dev modelleri (Llama, Qwen vb.) "dilimleyip" yönetmek için:
router, hangi blokların (layer grupları) bu forward'da RAM'e yüklenip
çalıştırılacağına karar verir. Teorem: Nash dengesi + Parisi tavlama.

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
    Parisi-Nash tabanlı blok seçici. Mevcut v6 NashExpertRouter ile
    aynı mantık (MLP gate, temperature, load balancing, regret);
    çıktı olarak "hangi bloklar yüklenecek?" indeksleri verir.

    İki mod:
      - pool_input=False: Token bazlı routing (B, L, K) — eğitim / ince ayar
      - pool_input=True:  Batch bazlı tek karar (K,) — inference, gerçek loader
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

        self.gate = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, router_hidden, bias=False),
            nn.GELU(),
            nn.Linear(router_hidden, num_blocks, bias=False),
        )

        self.register_buffer("current_temperature", torch.tensor(2.0))
        self.register_buffer("cumulative_regret", torch.zeros(num_blocks))
        self.register_buffer("block_usage", torch.zeros(num_blocks))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

    def set_temperature(self, t: float):
        self.current_temperature.fill_(t)

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

        T = self.current_temperature.clamp(min=0.1)
        logits = self.gate(x) / T
        probs = F.softmax(logits, dim=-1)

        _, indices = torch.topk(probs, self.top_k, dim=-1)
        # Seçilen blokların ağırlıkları (normalize)
        top_probs = torch.gather(probs, -1, indices)
        weights = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Load balancing loss
        with torch.no_grad():
            one_hot = F.one_hot(indices, self.num_blocks).float()
            f = one_hot.sum(dim=(0, 1, 2)) / (
                indices.shape[0] * indices.shape[1] * self.top_k + 1e-8
            )
        P = probs.mean(dim=(0, 1))
        aux_loss = self.num_blocks * (f * P).sum()

        if self.training:
            self.step_count += 1
            with torch.no_grad():
                self.block_usage = 0.99 * self.block_usage + 0.01 * f
                uniform = torch.ones_like(f) / self.num_blocks
                self.cumulative_regret = 0.99 * self.cumulative_regret + (uniform - f)

        return probs, indices, aux_loss, weights

    def get_stats(self) -> dict:
        """Router istatistikleri (sıcaklık, kullanım, pişmanlık)."""
        return {
            "temperature": self.current_temperature.item(),
            "usage": self.block_usage.detach().clone(),
            "regret": self.cumulative_regret.detach().clone(),
            "step_count": self.step_count.item(),
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

        Kullanım:
            # Giriş cümlesini embed et
            x = embed(input_ids)  # (B, L, D)
            # Tahmin: hangi bloklar yüklenecek?
            block_indices, weights = router.get_predictive_indices(x)
            # Sadece bu blokları diskten RAM'e yükle

        Args:
            x: (B, L, embed_dim) veya (B, embed_dim) embedded input
            pool_input: True ise tüm sequence'i pool edip tek karar üretir

        Returns:
            block_indices: [i1, i2, ...] top_k blok indeksleri (list)
            weights: (top_k,) bu blokların ağırlıkları
        """
        self.eval()
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, D) → (B, 1, D)
        if pool_input and x.size(1) > 1:
            x = x.mean(dim=(0, 1), keepdim=True)  # (1, 1, D)

        T = self.current_temperature.clamp(min=0.1)
        logits = self.gate(x) / T
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
