#!/usr/bin/env python3
"""
Birleşik Parisi-Nash v3 Deneyi
================================
3 kritik iyileştirme ile birleşik modeli test eder.

Baseline (önceki deneylerden):
  ① Parisi (Ardışık Attn+FFN):   39.7M → PPL 302.6  (33 dk)
  ② Nash-Parisi (Ardışık):        57.4M → PPL 305.0  (50 dk)
  ③ Birleşik v1 (collapse):       57.1M → PPL 366.3  (50 dk)
  ④ Birleşik v2 (aşırı denge):    64.2M → PPL ~611   (120 dk)

v3 İyileştirmeleri:
  ✓ Temperature Annealing 2.0→0.3 (Parisi tavlama)
  ✓ LB katsayısı 0.001 (hafif denge)
  ✓ Top-K=1 (zorunlu uzmanlaşma)
  ✓ Sparse execution (sadece seçilen expert çalışır)

Kullanım:
    PYTHONUNBUFFERED=1 python3 -u experiment_unified_v3.py
"""

import gc
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


def load_data(block_size=256):
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    print("  WikiText-2 yükleniyor...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    train_text = "\n".join([t for t in ds["train"]["text"] if t.strip()])
    val_text = "\n".join([t for t in ds["validation"]["text"] if t.strip()])
    train_tokens = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_tokens = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    print(f"  Eğitim: {len(train_tokens):,} token, Doğrulama: {len(val_tokens):,} token")
    return TokenDataset(train_tokens, block_size), TokenDataset(val_tokens, block_size), tokenizer


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
    steps = 3000
    batch_size = 8
    block_size = 256
    lr = 5e-4
    log_interval = 50
    eval_interval = 500

    device = torch.device(
        'mps' if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available()
        else 'cpu'
    )

    print()
    print("╔" + "═" * 68 + "╗")
    print("║  BİRLEŞİK PARİSİ-NASH v3: TAVLAMA + TOP-1 + SPARSE               ║")
    print("║  v_i = φ_k(Σ_j∈N(i) α_ij^k · V_k_j) + η·ε  (tek expert)          ║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("  v3 İyileştirmeleri:")
    print("    ✓ Temperature Annealing 2.0 → 0.3 (Parisi tavlama)")
    print("    ✓ LB katsayısı 0.001 (hafif denge koruma)")
    print("    ✓ Top-K=1 (her token TEK expert'e bağlı)")
    print("    ✓ Sparse execution (sadece seçilen expert çalışır)")
    print()
    print(f"  Cihaz: {device} | Adım: {steps}")
    print()
    print("  Önceki sonuçlar:")
    print("    ① Parisi (Ardışık):     39.7M → PPL 302.6  (33 dk)")
    print("    ② Nash-Parisi (Ardışık): 57.4M → PPL 305.0  (50 dk)")
    print("    ③ Birleşik v1 (collapse): 57.1M → PPL 366.3")
    print("    ④ Birleşik v2 (aş.denge): 64.2M → PPL ~611")
    print()

    train_ds, val_ds, tokenizer = load_data(block_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    # ── Model: Birleşik v3 (top_k=1) ──
    config = SwarmConfig(
        vocab_size=tokenizer.vocab_size, embed_dim=384,
        num_heads=8, num_layers=8,
        max_seq_len=block_size, dropout=0.1,
        neighbor_size=31, use_multi_scale=False,
        noise_strength=0.01, ffn_multiplier=2,
        use_nash_moe=True, num_experts=4, top_k_experts=1,
    )
    model = UnifiedParisiNashLLM(config)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"  ⑤ BİRLEŞİK v3 Parametreleri: {total_p:,} ({total_p/1e6:.1f}M)")
    for k, v in model.count_parameters().items():
        if isinstance(v, int) and v > 0:
            pct = v / total_p * 100
            print(f"    {k:>25s}: {v:>12,} ({pct:>5.1f}%)")
    print()

    # Temperature Annealing bilgisi
    print("  Temperature Annealing Schedule:")
    for s in [0, 500, 1000, 1500, 2000, 2500, 3000]:
        p = min(s / steps, 1.0)
        t = model.t_end + 0.5 * (model.t_start - model.t_end) * (1 + math.cos(math.pi * p))
        print(f"    Adım {s:>5d}: T = {t:.3f}")
    print()

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))

    warmup = 300

    def get_lr(step):
        if step < warmup:
            return lr * step / max(warmup, 1)
        p = (step - warmup) / max(steps - warmup, 1)
        return 1e-5 + 0.5 * (lr - 1e-5) * (1 + math.cos(math.pi * p))

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

            # ── Temperature Annealing ──
            model.set_annealing_step(step, steps)

            cur_lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg['lr'] = cur_lr

            out = model(x, targets=y)
            loss = out['loss']

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

                # Expert kullanım (ilk katman)
                usage = stats['usages'][0] if stats['usages'][0] else [0]*4
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

                val_ppl = eval_ppl(model, val_loader, device)
                if val_ppl < best_ppl:
                    best_ppl = val_ppl

                print(f"\n  ── Adım {step} | Val PPL: {val_ppl:.1f} | En İyi: {best_ppl:.1f} ──")

                text = generate(model, tokenizer, "The history of science",
                               max_tokens=40, device=device)
                print(f"    \"{text[:150]}\"")

                # Expert detay
                stats = model.get_nash_stats()
                for i in range(min(4, len(stats['regrets']))):
                    r = stats['regrets'][i]
                    u = stats['usages'][i] if stats['usages'][i] else [0]*4
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
    print("║  SONUÇ: BİRLEŞİK v3 vs TÜM MODELLER                              ║")
    print("╚" + "═" * 68 + "╝")
    print()
    print(f"  {'Model':<45s} {'Param':>8s} {'PPL':>8s} {'Süre':>8s}")
    print(f"  {'─' * 73}")
    print(f"  {'① Parisi (Ardışık Attn+FFN)':<45s} {'39.7M':>8s} {'302.6':>8s} {'33dk':>8s}")
    print(f"  {'② Nash-Parisi (Ardışık Attn→MoE)':<45s} {'57.4M':>8s} {'305.0':>8s} {'50dk':>8s}")
    print(f"  {'③ Birleşik v1 (collapse)':<45s} {'57.1M':>8s} {'366.3':>8s} {'50dk':>8s}")
    print(f"  {'④ Birleşik v2 (aşırı denge)':<45s} {'64.2M':>8s} {'~611':>8s} {'120dk':>8s}")
    print(f"  {'⑤ BİRLEŞİK v3 (tavlama+top1)':<45s} "
          f"{total_p/1e6:>6.1f}M {best_ppl:>8.1f} {total_time/60:>5.0f}dk")

    # Değerlendirme
    print()
    if best_ppl < 302.6:
        print(f"  ★★★ BASELINE'I GEÇTİ! PPL {best_ppl:.1f} < 302.6 ★★★")
    elif best_ppl < 366.3:
        print(f"  ★★ v1'i GEÇTİ! PPL {best_ppl:.1f} < 366.3 (v1 sonucu)")
        print(f"     Baseline'a fark: {best_ppl - 302.6:+.1f}")
    elif best_ppl < 611:
        print(f"  ★ v2'yi GEÇTİ! PPL {best_ppl:.1f} < ~611 (v2 sonucu)")
    else:
        print(f"  v2 seviyesinde kaldı: PPL {best_ppl:.1f}")

    # Kaydet
    result = {
        'name': 'BİRLEŞİK v3 (tavlama+top1+sparse)',
        'params': total_p,
        'final_ppl': final_ppl,
        'best_ppl': best_ppl,
        'time': total_time,
        'history': history,
        'config': {
            't_start': model.t_start, 't_end': model.t_end,
            'lb_coeff': model.lb_coeff, 'top_k': 1,
            'num_experts': 4, 'embed_dim': 384, 'num_layers': 8,
        },
    }
    with open("results_unified_v3.json", 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Sonuçlar: results_unified_v3.json")
    print()


if __name__ == "__main__":
    main()
