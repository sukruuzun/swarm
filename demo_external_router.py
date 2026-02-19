#!/usr/bin/env python3
"""
External Parisi-Nash Router + Sparse Block Loader Demo
======================================================
"MoE-fication" yolunda: dev model bloklarını router ile seçip sadece top_k
blok çalıştırma simülasyonu. Gerçek kullanımda blocks = Llama layer grupları.

Kullanım:
    python demo_external_router.py
"""

import torch
import torch.nn as nn

from swarm_llm.external_router import ExternalParisiNashRouter
from swarm_llm.sparse_loader import SparseBlockLoader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, L, D = 2, 16, 64
    vocab_size = 1000
    num_blocks = 8
    top_k = 2

    # Simülasyon: 8 "blok" (gerçekte her biri 10 layer olabilir)
    blocks = nn.ModuleList([
        nn.Sequential(
            nn.Linear(D, D * 2),
            nn.GELU(),
            nn.Linear(D * 2, D),
        )
        for _ in range(num_blocks)
    ])

    embed = nn.Embedding(vocab_size, D)
    router = ExternalParisiNashRouter(
        embed_dim=D,
        num_blocks=num_blocks,
        top_k=top_k,
    )
    lm_head = nn.Linear(D, vocab_size, bias=False)
    lm_head.weight = embed.weight  # weight tying

    loader = SparseBlockLoader(
        blocks=blocks,
        router=router,
        embed=embed,
        lm_head=lm_head,
        pool_router_input=True,
        lb_coeff=0.0001,
    ).to(device)

    # Tek forward: hangi bloklar seçildi?
    loader.eval()
    with torch.no_grad():
        x = torch.randint(0, vocab_size, (B, L), device=device)
        out = loader(x, targets=None)
    idx = out["moe_info"]["selected_indices"]
    print("╔" + "═" * 58 + "╗")
    print("║  EXTERNAL PARİSİ-NASH ROUTER + SPARSE BLOCK LOADER DEMO      ║")
    print("╚" + "═" * 58 + "╝")
    print()
    print(f"  Toplam blok: {num_blocks}  |  Seçilen (top_k): {top_k}")
    print(f"  Bu forward'da yüklenen bloklar: {idx}")
    print(f"  Router kullanım (son): {out['moe_info']['usage'].tolist()}")
    print()

    # Birkaç eğitim adımı (router + seçilen bloklar güncellenir)
    loader.train()
    opt = torch.optim.AdamW(loader.parameters(), lr=1e-3, weight_decay=0.01)
    for step in range(50):
        loader.set_annealing_step(step, 50)
        x = torch.randint(0, vocab_size, (B, L), device=device)
        y = torch.randint(0, vocab_size, (B, L), device=device)
        out = loader(x, targets=y)
        loss = out["loss"]
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(loader.parameters(), 1.0)
        opt.step()
        if (step + 1) % 10 == 0:
            idx = out["moe_info"]["selected_indices"]
            print(f"  Adım {step+1:>3d}  loss={loss.item():.4f}  seçilen bloklar={idx}")

    print()
    print("  get_blocks_to_load(x) — inference için hangi bloklar yüklenecek:")
    loader.eval()
    x = torch.randint(0, vocab_size, (1, 32), device=device)
    block_indices, weights = loader.get_blocks_to_load(loader.embed(x))
    print(f"    Blok indeksleri: {block_indices}")
    print(f"    Ağırlıklar:     {weights.tolist()}")
    print()
    print("  Gerçek Llama/Qwen entegrasyonu: blocks = [layer_grup_0, ..., layer_grup_N]")
    print("  Router bu indekslere göre sadece o blokları diskten RAM'e yükler.")
    print()


if __name__ == "__main__":
    main()
