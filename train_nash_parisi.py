#!/usr/bin/env python3
"""
Nash-Parisi Stratejik Sığırcık Eğitim
========================================
Parisi (Sığırcık Dikkat) + Nash (Oyun Teorisi MoE) birleşik
modelini WikiText-2 üzerinde eğitir.

3 modeli karşılaştırır:
  1. Standart Swarm-LLM (sadece Parisi)
  2. Nash-Parisi LLM (Parisi + Nash MoE)

Kullanım:
    python train_nash_parisi.py
    python train_nash_parisi.py --steps 2000
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from swarm_llm.config import SwarmConfig
from swarm_llm.model import SwarmLLM
from swarm_llm.nash_parisi_model import NashParisiLLM


# ── Veri ─────────────────────────────────────────────────────────────────────

class TokenDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.data = tokens
        self.block_size = block_size
    def __len__(self):
        return (len(self.data) - 1) // self.block_size
    def __getitem__(self, idx):
        s = idx * self.block_size
        return self.data[s:s+self.block_size], self.data[s+1:s+self.block_size+1]


def load_data(block_size=128):
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

    train_ds = TokenDataset(train_tokens, block_size)
    val_ds = TokenDataset(val_tokens, block_size)
    return train_ds, val_ds, tokenizer


# ── Üretim ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=60, temperature=0.8, device='cpu'):
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
def eval_ppl(model, loader, device, max_batches=50):
    model.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches: break
        x, y = x.to(device), y.to(device)
        total += model(x, targets=y)['loss'].item()
        n += 1
    model.train()
    return math.exp(total / max(n, 1))


# ── Eğitim ───────────────────────────────────────────────────────────────────

def train_model(model, name, train_loader, val_loader, tokenizer, config, device, args):
    """Tek bir modeli eğit ve sonuçları döndür."""
    print(f"\n{'='*65}")
    print(f"  {name} Eğitimi")
    print(f"{'='*65}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametreler: {total_params:,} ({total_params/1e6:.1f}M)")

    if hasattr(model, 'count_parameters'):
        for k, v in model.count_parameters().items():
            if isinstance(v, int):
                print(f"    {k:>25s}: {v:>12,}")

    if hasattr(model, 'estimate_vram'):
        print(f"\n  VRAM Tahmini (seq={args.block_size}, batch={args.batch_size}):")
        for k, v in model.estimate_vram(args.block_size, args.batch_size).items():
            if v:
                print(f"    {k:>25s}: {v}")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    warmup = min(200, args.steps // 10)
    def get_lr(step):
        if step < warmup:
            return args.lr * step / warmup
        p = (step - warmup) / max(args.steps - warmup, 1)
        return 1e-5 + 0.5 * (args.lr - 1e-5) * (1 + math.cos(math.pi * p))

    # Eğitim öncesi
    print(f"\n  Eğitim öncesi üretim:")
    text = generate(model, tokenizer, "The history of", device=device, max_tokens=30)
    print(f"  > \"{text[:150]}\"")

    model.train()
    step, running_loss = 0, 0.0
    epoch = 0
    start = time.time()
    history = []

    while step < args.steps:
        epoch += 1
        for x, y in train_loader:
            if step >= args.steps: break
            x, y = x.to(device), y.to(device)

            lr = get_lr(step)
            for pg in optimizer.param_groups: pg['lr'] = lr

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

                # Nash istatistikleri
                nash_info = ""
                if hasattr(model, 'get_nash_stats') and step % (args.log_interval * 2) == 0:
                    stats = model.get_nash_stats()
                    avg_balance = sum(stats['expert_balance']) / len(stats['expert_balance'])
                    nash_info = f" | Denge: {avg_balance:.2f}"

                print(
                    f"  Adım {step:>5d}/{args.steps} | "
                    f"Kayıp: {avg:.4f} | "
                    f"PPL: {math.exp(avg):.1f} | "
                    f"LR: {lr:.2e} | "
                    f"T/s: {tps:,.0f}{nash_info}"
                )
                history.append({'step': step, 'loss': avg, 'ppl': math.exp(avg)})
                running_loss = 0.0

            if step % args.gen_interval == 0:
                print(f"\n  ─── Adım {step} Üretim ───")
                for p in ["The history of science", "In the early years", "The city of"]:
                    t = generate(model, tokenizer, p, max_tokens=40, device=device)
                    print(f"  > \"{t[:160]}\"")

                val_ppl = eval_ppl(model, val_loader, device)
                print(f"  Doğrulama PPL: {val_ppl:.1f}")

                # Nash expert dağılımı
                if hasattr(model, 'get_nash_stats'):
                    stats = model.get_nash_stats()
                    print(f"  Nash Expert Dengesi: {[f'{b:.2f}' for b in stats['expert_balance']]}")
                    print(f"  Nash Sıcaklıkları:   {[f'{t:.2f}' for t in stats['temperatures']]}")
                print()

    elapsed = time.time() - start
    final_ppl = eval_ppl(model, val_loader, device)

    print(f"\n  Eğitim Sonucu ({name}):")
    print(f"    Süre: {elapsed:.0f}s ({elapsed/60:.1f} dk)")
    print(f"    Son PPL: {final_ppl:.1f}")
    print(f"    Son üretim:")
    for p in ["The meaning of life", "Scientists discovered"]:
        t = generate(model, tokenizer, p, max_tokens=50, temperature=0.7, device=device)
        print(f"    > \"{t[:180]}\"")

    return {
        'name': name,
        'params': total_params,
        'final_ppl': final_ppl,
        'time': elapsed,
        'history': history,
    }


# ── Ana ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Nash-Parisi Eğitim")
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--embed-dim', type=int, default=192)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--num-heads', type=int, default=6)
    parser.add_argument('--num-experts', type=int, default=6)
    parser.add_argument('--top-k', type=int, default=2)
    parser.add_argument('--neighbor-size', type=int, default=31)
    parser.add_argument('--lr', type=float, default=6e-4)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--gen-interval', type=int, default=500)
    args = parser.parse_args()

    device = torch.device(
        'mps' if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available()
        else 'cpu'
    )

    print(f"\n{'='*65}")
    print(f"  Nash-Parisi Stratejik Sığırcık: Karşılaştırmalı Eğitim")
    print(f"  Cihaz: {device}")
    print(f"{'='*65}\n")

    # Veri
    train_ds, val_ds, tokenizer = load_data(args.block_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    results = []

    # ── Model 1: Swarm-LLM (Sadece Parisi) ───────────────────────────
    config_parisi = SwarmConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.block_size,
        dropout=0.1,
        neighbor_size=args.neighbor_size,
        use_multi_scale=False,
        noise_strength=0.01,
        learning_rate=args.lr,
        warmup_steps=200,
        max_steps=args.steps,
    )

    model_parisi = SwarmLLM(config_parisi)
    r1 = train_model(
        model_parisi, "Swarm-LLM (Sadece Parisi)",
        train_loader, val_loader, tokenizer, config_parisi, device, args,
    )
    results.append(r1)
    del model_parisi
    torch.cuda.empty_cache() if device.type == 'cuda' else None

    # ── Model 2: Nash-Parisi (Parisi + Nash MoE) ─────────────────────
    config_nash = SwarmConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.block_size,
        dropout=0.1,
        neighbor_size=args.neighbor_size,
        use_multi_scale=False,
        noise_strength=0.01,
        use_nash_moe=True,
        num_experts=args.num_experts,
        top_k_experts=args.top_k,
        learning_rate=args.lr,
        warmup_steps=200,
        max_steps=args.steps,
    )

    model_nash = NashParisiLLM(config_nash)
    r2 = train_model(
        model_nash, "Nash-Parisi (Parisi + Nash MoE)",
        train_loader, val_loader, tokenizer, config_nash, device, args,
    )
    results.append(r2)

    # ── Karşılaştırma ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  KARŞILAŞTIRMA SONUÇLARI")
    print(f"{'='*65}\n")

    print(f"  {'Model':<35s} {'Param':>10s} {'PPL':>8s} {'Süre':>8s}")
    print(f"  {'─'*65}")
    for r in results:
        print(
            f"  {r['name']:<35s} "
            f"{r['params']/1e6:>8.1f}M "
            f"{r['final_ppl']:>8.1f} "
            f"{r['time']:>6.0f}s"
        )

    # Kayıp eğrisi karşılaştırması
    print(f"\n  Kayıp Eğrisi Karşılaştırması:")
    print(f"  {'Adım':>6s}", end="")
    for r in results:
        short_name = r['name'][:20]
        print(f" | {short_name:>20s}", end="")
    print()
    print(f"  {'─'*55}")

    max_len = max(len(r['history']) for r in results)
    for i in range(0, max_len, 2):
        step = results[0]['history'][min(i, len(results[0]['history'])-1)]['step']
        print(f"  {step:>6d}", end="")
        for r in results:
            idx = min(i, len(r['history'])-1)
            ppl = r['history'][idx]['ppl']
            print(f" | {ppl:>20.1f}", end="")
        print()

    # Sonuç
    best = min(results, key=lambda r: r['final_ppl'])
    print(f"\n  En iyi model: {best['name']}")
    print(f"  En iyi PPL: {best['final_ppl']:.1f}")

    if len(results) >= 2:
        ppl_diff = results[0]['final_ppl'] - results[1]['final_ppl']
        if ppl_diff > 0:
            print(f"\n  Nash-Parisi, Parisi'den {ppl_diff:.1f} PPL daha iyi!")
            print(f"  Nash dengesi expert yük dengelemeyi iyileştirdi.")
        else:
            print(f"\n  Parisi, Nash-Parisi'den {-ppl_diff:.1f} PPL daha iyi.")
            print(f"  Nash MoE ek karmaşıklığı bu ölçekte henüz karşılamadı.")
            print(f"  Daha büyük modelde (50M+) Nash avantajı belirginleşir.")

    # Nash istatistikleri
    if hasattr(model_nash, 'get_nash_stats'):
        print(f"\n  Nash Expert İstatistikleri (son durum):")
        stats = model_nash.get_nash_stats()
        for i, (balance, temp, regret) in enumerate(zip(
            stats['expert_balance'], stats['temperatures'], stats['regrets']
        )):
            regret_str = [f"{r:.2f}" for r in regret]
            print(f"    Katman {i}: denge={balance:.2f}, sıcaklık={temp:.2f}")
            print(f"              pişmanlık={regret_str}")

    print()


if __name__ == "__main__":
    main()
