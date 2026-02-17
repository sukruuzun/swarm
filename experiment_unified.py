#!/usr/bin/env python3
"""
Birleşik Parisi-Nash Deneyi
============================
3 mimariyi karşılaştırır:
  1. Swarm-LLM (Sadece Parisi) -- ardışık attention + FFN
  2. Nash-Parisi (Ardışık)     -- ardışık attention + Nash MoE
  3. BİRLEŞİK Parisi-Nash     -- TEK FORMÜL (yeni mimari)

Formül:
  v_i(t+1) = Σ_k w_k^nash · [Σ_j∈N(i) α_ij^reynolds · φ_k(V_k_j)] + η·ε

Kullanım:
    PYTHONUNBUFFERED=1 python3 -u experiment_unified.py
    PYTHONUNBUFFERED=1 python3 -u experiment_unified.py --steps 3000
"""

import argparse
import gc
import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from swarm_llm.config import SwarmConfig
from swarm_llm.model import SwarmLLM
from swarm_llm.nash_parisi_model import NashParisiLLM
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
        total += model(x, targets=y)['loss'].item()
        n += 1
    model.train()
    return math.exp(total / max(n, 1))


def train_model(model, name, train_loader, val_loader, tokenizer, device, args):
    print(f"\n{'═' * 70}")
    print(f"  {name}")
    print(f"{'═' * 70}")

    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Parametreler: {total_p:,} ({total_p/1e6:.1f}M)")

    if hasattr(model, 'count_parameters'):
        for k, v in model.count_parameters().items():
            if isinstance(v, int) and v > 0:
                pct = v / total_p * 100
                print(f"    {k:>25s}: {v:>12,} ({pct:>5.1f}%)")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    warmup = min(300, args.steps // 5)

    def get_lr(step):
        if step < warmup:
            return args.lr * step / max(warmup, 1)
        p = (step - warmup) / max(args.steps - warmup, 1)
        return 1e-5 + 0.5 * (args.lr - 1e-5) * (1 + math.cos(math.pi * p))

    model.train()
    step, running_loss = 0, 0.0
    start = time.time()
    history = []
    best_ppl = float('inf')

    while step < args.steps:
        for x, y in train_loader:
            if step >= args.steps:
                break
            x, y = x.to(device), y.to(device)

            lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            out = model(x, targets=y)
            loss = out['loss']

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            step += 1

            if step % args.log_interval == 0:
                avg = running_loss / args.log_interval
                elapsed = time.time() - start
                tps = step * args.batch_size * args.block_size / elapsed
                ppl = math.exp(min(avg, 20))

                nash_info = ""
                if hasattr(model, 'get_nash_stats'):
                    stats = model.get_nash_stats()
                    avg_t = sum(stats['temperatures']) / len(stats['temperatures'])
                    nash_info = f" | T:{avg_t:.3f}"

                print(f"  [{step:>5d}/{args.steps}] "
                      f"Kayıp:{avg:>7.4f} PPL:{ppl:>8.1f} "
                      f"LR:{lr:.1e} T/s:{tps:>7,.0f}{nash_info}", flush=True)
                history.append({'step': step, 'loss': avg, 'ppl': ppl})
                running_loss = 0.0

            if step % args.eval_interval == 0 or step == args.steps:
                if device.type == 'mps':
                    torch.mps.empty_cache()

                val_ppl = eval_ppl(model, val_loader, device)
                if val_ppl < best_ppl:
                    best_ppl = val_ppl
                print(f"\n  ── Adım {step} | Val PPL: {val_ppl:.1f} | "
                      f"En İyi: {best_ppl:.1f} ──")
                text = generate(model, tokenizer, "The history of science",
                               max_tokens=40, device=device)
                print(f"    \"{text[:150]}\"")

                if hasattr(model, 'get_nash_stats'):
                    stats = model.get_nash_stats()
                    for i, r in enumerate(stats['regrets']):
                        r_str = [f"{v:+.1f}" for v in r]
                        print(f"    K{i}: [{', '.join(r_str)}]")
                print()

                if device.type == 'mps':
                    torch.mps.empty_cache()

    total_time = time.time() - start
    final_ppl = eval_ppl(model, val_loader, device, max_batches=200)

    print(f"  SONUÇ: {name}")
    print(f"    Son PPL: {final_ppl:.1f}, En İyi: {best_ppl:.1f}")
    print(f"    Süre: {total_time:.0f}s ({total_time/60:.1f} dk)")

    return {
        'name': name,
        'params': total_p,
        'final_ppl': final_ppl,
        'best_ppl': best_ppl,
        'time': total_time,
        'history': history,
    }


def main():
    parser = argparse.ArgumentParser(description="Birleşik Parisi-Nash Deneyi")
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--block-size', type=int, default=256)
    parser.add_argument('--embed-dim', type=int, default=384)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--eval-interval', type=int, default=500)
    args = parser.parse_args()

    device = torch.device(
        'mps' if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available()
        else 'cpu'
    )

    print()
    print("╔" + "═" * 68 + "╗")
    print("║  BİRLEŞİK PARİSİ-NASH: TEK FORMÜL DENEYİ                         ║")
    print("║  v_i = Σ_k w_k · [Σ_j∈N(i) α_ij · φ_k(V_k_j)] + η·ε             ║")
    print("╚" + "═" * 68 + "╝")
    print(f"\n  Cihaz: {device} | Adım: {args.steps} | Expert: {args.num_experts} (top-{args.top_k})")

    train_ds, val_ds, tokenizer = load_data(args.block_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    results = []

    # ── Model 1: Swarm-LLM (Sadece Parisi -- Baseline) ───────────────
    config1 = SwarmConfig(
        vocab_size=tokenizer.vocab_size, embed_dim=args.embed_dim,
        num_heads=args.num_heads, num_layers=args.num_layers,
        max_seq_len=args.block_size, dropout=0.1,
        neighbor_size=31, use_multi_scale=False,
        noise_strength=0.01, ffn_multiplier=4,
    )
    m1 = SwarmLLM(config1)
    r1 = train_model(m1, "① Parisi (Ardışık Attn+FFN)", train_loader, val_loader, tokenizer, device, args)
    results.append(r1)
    del m1; gc.collect()
    if device.type == 'mps': torch.mps.empty_cache()

    # ── Model 2: Nash-Parisi (Ardışık) ────────────────────────────────
    config2 = SwarmConfig(
        vocab_size=tokenizer.vocab_size, embed_dim=args.embed_dim,
        num_heads=args.num_heads, num_layers=args.num_layers,
        max_seq_len=args.block_size, dropout=0.1,
        neighbor_size=31, use_multi_scale=False,
        noise_strength=0.01, ffn_multiplier=2,
        use_nash_moe=True, num_experts=args.num_experts, top_k_experts=args.top_k,
    )
    m2 = NashParisiLLM(config2)
    r2 = train_model(m2, "② Nash-Parisi (Ardışık Attn→MoE)", train_loader, val_loader, tokenizer, device, args)
    results.append(r2)
    del m2; gc.collect()
    if device.type == 'mps': torch.mps.empty_cache()

    # ── Model 3: BİRLEŞİK Parisi-Nash (TEK FORMÜL) ──────────────────
    config3 = SwarmConfig(
        vocab_size=tokenizer.vocab_size, embed_dim=args.embed_dim,
        num_heads=args.num_heads, num_layers=args.num_layers,
        max_seq_len=args.block_size, dropout=0.1,
        neighbor_size=31, use_multi_scale=False,
        noise_strength=0.01, ffn_multiplier=2,
        use_nash_moe=True, num_experts=args.num_experts, top_k_experts=args.top_k,
    )
    m3 = UnifiedParisiNashLLM(config3)
    r3 = train_model(m3, "③ BİRLEŞİK Parisi-Nash (Tek Formül)", train_loader, val_loader, tokenizer, device, args)
    results.append(r3)

    # ── Karşılaştırma ────────────────────────────────────────────────
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  KARŞILAŞTIRMA: 3 MİMARİ                                          ║")
    print("╚" + "═" * 68 + "╝")
    print()
    print(f"  {'Model':<45s} {'Param':>8s} {'PPL':>8s} {'Süre':>8s}")
    print(f"  {'─' * 73}")
    for r in results:
        print(f"  {r['name']:<45s} {r['params']/1e6:>6.1f}M {r['best_ppl']:>8.1f} {r['time']:>6.0f}s")

    # PPL eğrisi
    print(f"\n  PPL Eğrisi:")
    print(f"  {'Adım':>6s}", end="")
    for r in results:
        short = r['name'][:15]
        print(f" | {short:>15s}", end="")
    print()
    print(f"  {'─' * 58}")
    max_len = max(len(r['history']) for r in results)
    for i in range(0, max_len, max(1, max_len // 15)):
        step = results[0]['history'][min(i, len(results[0]['history'])-1)]['step']
        print(f"  {step:>6d}", end="")
        for r in results:
            idx = min(i, len(r['history'])-1)
            print(f" | {r['history'][idx]['ppl']:>15.1f}", end="")
        print()

    best = min(results, key=lambda r: r['best_ppl'])
    print(f"\n  ★ EN İYİ: {best['name']} -- PPL {best['best_ppl']:.1f}")

    # JSON kaydet
    with open("results_unified.json", 'w') as f:
        json.dump([{k: v for k, v in r.items()} for r in results], f, indent=2, default=str)
    print(f"\n  Sonuçlar: results_unified.json")
    print()


if __name__ == "__main__":
    main()
