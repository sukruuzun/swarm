#!/usr/bin/env python3
"""
Birleşik Parisi-Nash v6-scale: A100/H100 Büyük Model
====================================================
v5 mimarisini A100 (40GB) / H100 (80GB) için ölçeklendirir.

Hedef: ~350M parametre, WikiText-103, 15K step.
  - embed_dim 768, 12 layer, 12 head, 4 expert
  - WikiText-103 (~103M token) — overfitting azaltır
  - batch_size: VRAM'e göre 8/16/24 (T4/A100/H100)
  - 15K step, warmup 1000, min_lr 5e-5

Kullanım (Colab / A100 / H100):
    PYTHONUNBUFFERED=1 python3 -u experiment_unified_v6_scale.py

T4'te VRAM yetmez; batch_size=4 veya gradient checkpointing gerekir.
"""

import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from swarm_llm.config import SwarmConfig
from swarm_llm.unified import UnifiedParisiNashLLM


class TokenDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.data = tokens
        self.block_size = block_size

    def __len__(self):
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx):
        s = idx * self.block_size
        return self.data[s:s + self.block_size], self.data[s + 1:s + self.block_size + 1]


def load_data(block_size=256, dataset_name="wikitext-103"):
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    if dataset_name == "wikitext-103":
        print("  WikiText-103 yükleniyor...")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    else:
        print("  WikiText-2 yükleniyor...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    train_text = "\n".join([t for t in ds["train"]["text"] if t.strip()])
    val_text = "\n".join([t for t in ds["validation"]["text"] if t.strip()])
    train_tokens = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_tokens = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    print(f"  Eğitim: {len(train_tokens):,} token, Doğrulama: {len(val_tokens):,} token")
    return TokenDataset(train_tokens, block_size), TokenDataset(val_tokens, block_size), tokenizer


def get_batch_size_for_vram(device):
    """VRAM'e göre batch_size öner (A100/H100/T4)."""
    if not torch.cuda.is_available():
        return 8
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_gb >= 70:
        return 24  # H100 80GB / A100 80GB
    if total_gb >= 35:
        return 16  # A100 40GB
    if total_gb >= 14:
        return 8   # T4 16GB (350M model zor; 8 deneyebilir)
    return 4


@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=50, temperature=0.8, device='cpu'):
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    model.set_annealing_step(model.config.max_steps, model.config.max_steps)
    for _ in range(max_tokens):
        ctx = ids[:, -model.config.max_seq_len:]
        out = model(ctx)
        logits = out['logits'][:, -1, :] / temperature
        topk_v, _ = torch.topk(logits, 40)
        logits[logits < topk_v[:, -1:]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt], dim=1)
    model.train()
    return tokenizer.decode(ids[0].tolist())


@torch.no_grad()
def eval_ppl(model, loader, device, max_batches=80):
    model.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        out = model(x, targets=y)
        ce = out.get('ce_loss', out['loss'])
        if ce is not None:
            total += ce.item()
        n += 1
    model.train()
    return math.exp(total / max(n, 1))


def main():
    # ── v6-scale: A100/H100 için büyük model ──
    steps = 15_000
    block_size = 256
    lr = 5e-4
    min_lr = 5e-5
    warmup = 1000
    log_interval = 100
    eval_interval = 500
    dataset_name = "wikitext-103"

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )

    batch_size = get_batch_size_for_vram(device)
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  CUDA VRAM: {vram_gb:.1f} GB → batch_size = {batch_size}")
    else:
        batch_size = min(batch_size, 8)

    print()
    print("╔" + "═" * 68 + "╗")
    print("║  BİRLEŞİK PARİSİ-NASH v6-scale: A100/H100 BÜYÜK MODEL             ║")
    print("║  v_i = p_k · φ_k(Σ_j∈N(i) α_ij^k · V_k_j) + η·ε                  ║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("  v6-scale (v5 mimarisi, büyütülmüş):")
    print("    ✓ embed 768, 12 layer, 12 head, 4 expert (~350M param)")
    print("    ✓ WikiText-103 (~103M token)")
    print(f"    ✓ {steps} step, warmup {warmup}, min_lr {min_lr}")
    print("    ✓ lb_coeff 0.0001, decoupled clip (router 0.5, expert 1.0)")
    print(f"    ✓ batch_size {batch_size} (VRAM'e göre)")
    print()
    print(f"  Cihaz: {device} | Adım: {steps} | Dataset: {dataset_name}")
    print()

    train_ds, val_ds, tokenizer = load_data(block_size, dataset_name=dataset_name)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    # ── Model: v6-scale (768, 12, 12, 4 expert) ──
    config = SwarmConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        max_seq_len=block_size,
        dropout=0.1,
        neighbor_size=31,
        use_multi_scale=False,
        noise_strength=0.01,
        ffn_multiplier=2,
        use_nash_moe=True,
        num_experts=4,
        top_k_experts=1,
    )
    config.max_steps = steps
    model = UnifiedParisiNashLLM(config)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"  v6-scale Parametreleri: {total_p:,} ({total_p/1e6:.1f}M)")
    for k, v in model.count_parameters().items():
        if isinstance(v, int) and v > 0:
            pct = v / total_p * 100
            print(f"    {k:>25s}: {v:>12,} ({pct:>5.1f}%)")
    print()

    print("  Temperature Annealing Schedule (15K step):")
    for s in [0, 3000, 6000, 9000, 12000, 15000]:
        p = min(s / steps, 1.0)
        t = model.t_end + 0.5 * (model.t_start - model.t_end) * (1 + math.cos(math.pi * p))
        print(f"    Adım {s:>5d}: T = {t:.3f}")
    print()

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))

    router_params = model.get_router_params()
    expert_params = model.get_expert_params()

    def get_lr(step):
        if step < warmup:
            return lr * step / max(warmup, 1)
        p = (step - warmup) / max(steps - warmup, 1)
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))

    model.train()
    step, running_loss, running_ce, running_aux = 0, 0.0, 0.0, 0.0
    start = time.time()
    history = []
    best_ppl = float('inf')

    while step < steps:
        for x, y in train_loader:
            if step >= steps:
                break
            x, y = x.to(device), y.to(device)

            model.set_annealing_step(step, steps)

            cur_lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg['lr'] = cur_lr

            out = model(x, targets=y)
            loss = out['loss']

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(expert_params, 1.0)
            nn.utils.clip_grad_norm_(router_params, 0.5)
            optimizer.step()

            running_loss += loss.item()
            ce_val = out['ce_loss'].item() if out['ce_loss'] is not None else 0
            running_ce += ce_val
            aux_val = out['aux_loss'].item() if isinstance(out['aux_loss'], torch.Tensor) else out['aux_loss']
            running_aux += aux_val
            step += 1

            if step % log_interval == 0:
                avg = running_loss / log_interval
                avg_ce = running_ce / log_interval
                avg_aux = running_aux / log_interval
                elapsed = time.time() - start
                tps = step * batch_size * block_size / elapsed
                ppl = math.exp(min(avg_ce, 20))

                stats = model.get_nash_stats()
                T = stats['temperatures'][0]
                usage = stats['usages'][0] if stats['usages'][0] else [0] * 4
                u_str = " ".join([f"{u:.0%}" for u in usage])

                print(f"  [{step:>5d}/{steps}] "
                      f"PPL:{ppl:>8.1f} CE:{avg_ce:.4f} "
                      f"LB:{avg_aux:.2f} T:{T:.3f} "
                      f"LR:{cur_lr:.1e} T/s:{tps:>6,.0f} "
                      f"U:[{u_str}]", flush=True)

                history.append({
                    'step': step, 'loss': avg, 'ce_loss': avg_ce, 'ppl': ppl,
                    'aux_loss': avg_aux, 'temperature': T,
                    'usage': usage,
                })
                running_loss, running_ce, running_aux = 0.0, 0.0, 0.0

            if step % eval_interval == 0 or step == steps:
                if device.type == 'mps':
                    torch.mps.empty_cache()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                val_ppl = eval_ppl(model, val_loader, device)
                if val_ppl < best_ppl:
                    best_ppl = val_ppl

                print(f"\n  ── Adım {step} | Val PPL: {val_ppl:.1f} | En İyi: {best_ppl:.1f} ──")

                text = generate(model, tokenizer, "The history of science",
                               max_tokens=40, device=device)
                print(f"    \"{text[:150]}\"")

                stats = model.get_nash_stats()
                for i in range(min(4, len(stats['regrets']))):
                    r = stats['regrets'][i]
                    u = stats['usages'][i] if stats['usages'][i] else [0] * 4
                    r_str = [f"{v:+.2f}" for v in r]
                    u_str = [f"{v:.0%}" for v in u]
                    T = stats['temperatures'][i]
                    print(f"    K{i}: T={T:.2f} U=[{', '.join(u_str)}] R=[{', '.join(r_str)}]")
                print()

                if device.type == 'mps':
                    torch.mps.empty_cache()

    total_time = time.time() - start
    final_ppl = eval_ppl(model, val_loader, device, max_batches=200)

    print()
    print("╔" + "═" * 68 + "╗")
    print("║  SONUÇ: v6-scale (A100/H100)                                     ║")
    print("╚" + "═" * 68 + "╝")
    print()
    print(f"  v6-scale: {total_p/1e6:.1f}M param, WikiText-103, {steps} step")
    print(f"  En iyi Val PPL: {best_ppl:.1f} | Final: {final_ppl:.1f} | Süre: {total_time/60:.0f} dk")
    print()

    result = {
        'name': 'BİRLEŞİK v6-scale (A100/H100, WikiText-103)',
        'params': total_p,
        'final_ppl': final_ppl,
        'best_ppl': best_ppl,
        'time': total_time,
        'history': history,
        'config': {
            'embed_dim': 768, 'num_layers': 12, 'num_heads': 12,
            'num_experts': 4, 'dataset': dataset_name,
            'steps': steps, 'warmup': warmup, 'batch_size': batch_size,
            't_start': model.t_start, 't_end': model.t_end,
            'lb_coeff': model.lb_coeff, 'min_lr': min_lr,
        },
    }
    with open("results_unified_v6_scale.json", 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Sonuçlar: results_unified_v6_scale.json")
    print()


if __name__ == "__main__":
    main()
