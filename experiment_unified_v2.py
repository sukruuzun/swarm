#!/usr/bin/env python3
"""
Birleşik Parisi-Nash v2 Deneyi
================================
Sadece iyileştirilmiş birleşik modeli test eder.

Baseline sonuçlar (önceki deneyden):
  ① Parisi (Ardışık Attn+FFN):   39.7M params, PPL 302.6
  ② Nash-Parisi (Ardışık):        57.4M params, PPL 305.0

v2 İyileştirmeleri:
  1. Expert-spesifik Q,K,V (her expert farklı dikkat deseni)
  2. Load balancing loss (expert collapse önleme)
  3. Çift LayerNorm (pre+post)
  4. Düzeltilmiş regret matching

Kullanım:
    PYTHONUNBUFFERED=1 python3 -u experiment_unified_v2.py
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
        total += out['loss'].item()
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
    print("║  BİRLEŞİK PARİSİ-NASH v2: İYİLEŞTİRİLMİŞ TEK FORMÜL             ║")
    print("║  v_i = Σ_k w_k · φ_k(Σ_j∈N(i) α_ij^k · V_k_j) + η·ε             ║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("  v2 İyileştirmeleri:")
    print("    ✓ Expert-spesifik Q,K,V (farklı dikkat desenleri)")
    print("    ✓ Load balancing loss (collapse önleme)")
    print("    ✓ Çift LayerNorm (pre + post)")
    print("    ✓ Düzeltilmiş regret matching")
    print()
    print(f"  Cihaz: {device} | Adım: {steps}")
    print()
    print("  Baseline (önceki deneyden):")
    print("    ① Parisi (Ardışık):     39.7M params → PPL 302.6")
    print("    ② Nash-Parisi (Ardışık): 57.4M params → PPL 305.0")
    print("    ③ Birleşik v1:           57.1M params → PPL 366.3")
    print()

    train_ds, val_ds, tokenizer = load_data(block_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    # ── Model: Birleşik Parisi-Nash v2 ────────────────────────────────
    config = SwarmConfig(
        vocab_size=tokenizer.vocab_size, embed_dim=384,
        num_heads=8, num_layers=8,
        max_seq_len=block_size, dropout=0.1,
        neighbor_size=31, use_multi_scale=False,
        noise_strength=0.01, ffn_multiplier=2,
        use_nash_moe=True, num_experts=4, top_k_experts=2,
    )
    model = UnifiedParisiNashLLM(config)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"  ③ BİRLEŞİK v2 Parametreleri: {total_p:,} ({total_p/1e6:.1f}M)")
    for k, v in model.count_parameters().items():
        if isinstance(v, int) and v > 0:
            pct = v / total_p * 100
            print(f"    {k:>25s}: {v:>12,} ({pct:>5.1f}%)")
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
    step, running_loss, running_aux = 0, 0.0, 0.0
    start = time.time()
    history = []
    best_ppl = float('inf')

    while step < steps:
        for x, y in train_loader:
            if step >= steps:
                break
            x, y = x.to(device), y.to(device)

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
            aux_val = out['aux_loss'].item() if isinstance(out['aux_loss'], torch.Tensor) else out['aux_loss']
            running_aux += aux_val
            step += 1

            if step % log_interval == 0:
                avg = running_loss / log_interval
                avg_aux = running_aux / log_interval
                elapsed = time.time() - start
                tps = step * batch_size * block_size / elapsed
                ppl = math.exp(min(avg, 20))

                stats = model.get_nash_stats()
                avg_t = sum(stats['temperatures']) / len(stats['temperatures'])

                # Expert kullanım dağılımı (ilk katmandan)
                regrets = stats['regrets'][0]
                r_str = " ".join([f"{v:+.1f}" for v in regrets])

                print(f"  [{step:>5d}/{steps}] "
                      f"PPL:{ppl:>8.1f} Kayıp:{avg:.4f} "
                      f"LB:{avg_aux:.3f} T:{avg_t:.2f} "
                      f"LR:{cur_lr:.1e} T/s:{tps:>6,.0f} "
                      f"R:[{r_str}]", flush=True)

                history.append({
                    'step': step, 'loss': avg, 'ppl': ppl,
                    'aux_loss': avg_aux, 'temperature': avg_t,
                })
                running_loss, running_aux = 0.0, 0.0

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

                # Expert dağılım detayı
                stats = model.get_nash_stats()
                for i, r in enumerate(stats['regrets']):
                    r_str = [f"{v:+.3f}" for v in r]
                    t = stats['temperatures'][i]
                    print(f"    K{i}: T={t:.2f} [{', '.join(r_str)}]")
                print()

                if device.type == 'mps':
                    torch.mps.empty_cache()

    total_time = time.time() - start
    final_ppl = eval_ppl(model, val_loader, device, max_batches=200)

    print()
    print("╔" + "═" * 68 + "╗")
    print("║  SONUÇ: BİRLEŞİK v2 vs ÖNCEKİ SONUÇLAR                           ║")
    print("╚" + "═" * 68 + "╝")
    print()
    print(f"  {'Model':<45s} {'Param':>8s} {'PPL':>8s} {'Süre':>8s}")
    print(f"  {'─' * 73}")
    print(f"  {'① Parisi (Ardışık Attn+FFN)':<45s} {'39.7M':>8s} {'302.6':>8s} {'1995s':>8s}")
    print(f"  {'② Nash-Parisi (Ardışık Attn→MoE)':<45s} {'57.4M':>8s} {'305.0':>8s} {'3031s':>8s}")
    print(f"  {'③ Birleşik v1 (Tek Formül)':<45s} {'57.1M':>8s} {'366.3':>8s} {'2988s':>8s}")
    print(f"  {'④ BİRLEŞİK v2 (İyileştirilmiş)':<45s} "
          f"{total_p/1e6:>6.1f}M {best_ppl:>8.1f} {total_time:>6.0f}s")

    improvement = 366.3 - best_ppl
    vs_baseline = best_ppl - 302.6
    print()
    if improvement > 0:
        print(f"  v1'den iyileşme: {improvement:+.1f} PPL")
    if vs_baseline <= 0:
        print(f"  ★ Baseline'ı GEÇTİ: {vs_baseline:+.1f} PPL")
    else:
        print(f"  Baseline'a kalan fark: {vs_baseline:+.1f} PPL")

    # Kaydet
    result = {
        'name': 'BİRLEŞİK v2',
        'params': total_p,
        'final_ppl': final_ppl,
        'best_ppl': best_ppl,
        'time': total_time,
        'history': history,
    }
    with open("results_unified_v2.json", 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Sonuçlar: results_unified_v2.json")
    print()


if __name__ == "__main__":
    main()
