"""
Sparse Block Loader (Seyrek Blok Yükleyici)
============================================
Eğitilmiş dev modelin bloklarını (layer gruplarını) Parisi-Nash router ile
seçip sadece top_k blok çalıştıran wrapper. "70B parametre diskte, 2B RAM'de."

Mimari:
  input_ids → embed → router(pool) → [blok_i1, blok_i2, ...] → ağırlıklı toplam → lm_head

Gerçek kullanım (Llama/HF):
  - blocks = [nn.Sequential(layer_i0..layer_i9) for i in 0..7]  # 8 expert x 10 layer
  - Router top_k=2 seçer → sadece 2 blok (20 layer) çalışır
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from swarm_llm.external_router import ExternalParisiNashRouter


class SparseBlockLoader(nn.Module):
    """
    N blok (nn.Module) alır; her forward'da router ile top_k blok seçer,
    sadece onları çalıştırıp ağırlıklı toplamlar. Geri yayılım sadece
    seçilen bloklara gider (diğerleri frozen veya yüklü değil).
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        router: ExternalParisiNashRouter,
        embed: nn.Module,
        lm_head: Optional[nn.Module] = None,
        pool_router_input: bool = True,
        lb_coeff: float = 0.0001,
    ):
        super().__init__()
        assert len(blocks) == router.num_blocks
        assert router.top_k >= 1

        self.blocks = blocks
        self.router = router
        self.embed = embed
        self.lm_head = lm_head
        self.pool_router_input = pool_router_input
        self.lb_coeff = lb_coeff
        self.num_blocks = len(blocks)
        self.top_k = router.top_k

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            input_ids: (B, L) token id
            targets: (B, L) ce_loss için (opsiyonel)

        Returns:
            logits, loss, ce_loss, aux_loss, selected_indices, moe_info
        """
        B, L = input_ids.shape
        x = self.embed(input_ids)  # (B, L, D)

        probs, indices, aux_loss, weights = self.router(
            x, pool_input=self.pool_router_input
        )
        # pool_router_input=True → indices (1,1,K), weights (1,1,K) → tek karar
        # pool_router_input=False → indices (B,L,K); burada ilk token kararını kullan
        if self.pool_router_input:
            indices = indices.squeeze(0).squeeze(0)  # (K,)
            weights = weights.squeeze(0).squeeze(0)  # (K,)
        else:
            indices = indices[:, 0, :]   # (B, K) — ilk token kararı
            weights = weights[:, 0, :]  # (B, K)

        out = torch.zeros_like(x)
        for k in range(self.top_k):
            if self.pool_router_input:
                idx = indices[k].item()
                out = out + weights[k] * self.blocks[idx](x)
            else:
                for b in range(B):
                    idx = indices[b, k].item()
                    out[b : b + 1] = out[b : b + 1] + weights[b, k] * self.blocks[idx](x[b : b + 1])

        logits = self.lm_head(out) if self.lm_head is not None else None
        ce_loss = None
        loss = None
        if targets is not None and logits is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            loss = ce_loss + self.lb_coeff * aux_loss

        info = self.router.get_stats()
        info["aux_loss"] = aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
        info["selected_indices"] = indices.detach().cpu().tolist()

        return {
            "logits": logits,
            "hidden_states": out,
            "loss": loss,
            "ce_loss": ce_loss,
            "aux_loss": aux_loss,
            "selected_indices": indices,
            "router_weights": weights,
            "moe_info": info,
        }

    def get_blocks_to_load(self, x: torch.Tensor) -> Tuple[List[int], torch.Tensor]:
        """
        Inference / loader için: bu girişe göre hangi blokların
        yükleneceğini döner. Diskten sadece bu blokları yükle.

        Returns:
            block_indices: [i1, i2, ...] top_k blok indeksleri
            weights: (K,) bu blokların ağırlıkları
        """
        self.router.eval()
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x_pool = x.mean(dim=(0, 1), keepdim=True)
            _, indices, _, weights = self.router(x_pool, pool_input=True)
            indices = indices.squeeze().cpu()
            weights = weights.squeeze().cpu()
            if indices.dim() == 0:
                indices = indices.unsqueeze(0)
                weights = weights.unsqueeze(0)
            return indices.tolist(), weights

    def set_annealing_step(self, step: int, total_steps: int):
        """Parisi tavlama: cosine schedule ile router sıcaklığını güncelle."""
        import math
        t_start, t_end = 2.0, 0.3
        progress = min(step / max(total_steps, 1), 1.0)
        t = t_end + 0.5 * (t_start - t_end) * (1 + math.cos(math.pi * progress))
        self.router.set_temperature(t)
