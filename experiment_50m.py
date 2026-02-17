#!/usr/bin/env python3
"""
50M Parametre Kapsamlı Deney: Parisi vs Nash-Parisi
=====================================================
Daha büyük modelle (50M) daha uzun eğitim (5000 adım) ile
Nash-Parisi mimarisinin avantajlarını test eder.

Deney Planı:
  Model 1: Swarm-LLM (Sadece Parisi)
    embed=384, layers=8, heads=8, FFN=4x → ~38M param
  Model 2: Nash-Parisi (Parisi + Nash MoE)
    embed=384, layers=8, heads=8, experts=8, top_k=2 → ~50M param
    (Expert paylaşımlı parametreler daha fazla -- ama hesaplama seyrek)

Metrikler:
  - Validation PPL (ana metrik)
  - Eğitim kaybı eğrisi
  - Token/saniye (verimlilik)
  - Nash expert denge skoru (zaman içinde)
  - Expert uzmanlaşma analizi
  - Bellek kullanımı

Kullanım:
    PYTHONUNBUFFERED=1 python3 -u experiment_50m.py
    PYTHONUNBUFFERED=1 python3 -u experiment_50m.py --steps 3000  # hızlı test
"""

import argparse
import gc
import json
import math
import os
import sys
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

    print(f"  Eğitim: {len(train_tokens):,} token")
    print(f"  Doğrulama: {len(val_tokens):,} token")
    print(f"  Blok boyutu: {block_size}")

    train_ds = TokenDataset(train_tokens, block_size)
    val_ds = TokenDataset(val_tokens, block_size)
    return train_ds, val_ds, tokenizer


# ── Üretim ve Değerlendirme ──────────────────────────────────────────────────

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


# ── Expert Analizi ───────────────────────────────────────────────────────────

@torch.no_grad()
def analyze_expert_specialization(model, loader, device, max_batches=20):
    """
    Expert'lerin hangi tür token'lara uzmanlaştığını analiz et.
    Her expert'in en çok aktive olduğu token dağılımını inceler.
    """
    if not hasattr(model, 'get_nash_stats'):
        return None

    model.eval()
    layer_expert_activations = {}

    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x = x.to(device)
        out = model(x)

        for layer_idx, moe_info in enumerate(out.get('moe_info', [])):
            if layer_idx not in layer_expert_activations:
                layer_expert_activations[layer_idx] = {}

            usage = moe_info.get('expert_usage', None)
            if usage is not None:
                for e_idx in range(len(usage)):
                    if e_idx not in layer_expert_activations[layer_idx]:
                        layer_expert_activations[layer_idx][e_idx] = []
                    layer_expert_activations[layer_idx][e_idx].append(usage[e_idx].item())

    model.train()

    analysis = {}
    for layer_idx, experts in layer_expert_activations.items():
        analysis[layer_idx] = {}
        for e_idx, usages in experts.items():
            analysis[layer_idx][e_idx] = {
                'mean_usage': sum(usages) / len(usages),
                'std_usage': (sum((u - sum(usages)/len(usages))**2 for u in usages) / len(usages)) ** 0.5,
            }
    return analysis


# ── Eğitim ───────────────────────────────────────────────────────────────────

def train_model(model, name, train_loader, val_loader, tokenizer, config, device, args):
    """Tek bir modeli eğit -- kapsamlı loglama ile."""
    separator = "=" * 70
    print(f"\n{separator}")
    print(f"  MODEL: {name}")
    print(f"{separator}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Toplam Parametre: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Eğitilebilir:     {trainable_params:,} ({trainable_params/1e6:.1f}M)")

    if hasattr(model, 'count_parameters'):
        print(f"\n  Parametre Dağılımı:")
        for k, v in model.count_parameters().items():
            if isinstance(v, int):
                pct = v / total_params * 100
                print(f"    {k:>25s}: {v:>12,} ({pct:>5.1f}%)")

    if hasattr(model, 'estimate_vram'):
        print(f"\n  VRAM Tahmini (seq={args.block_size}, batch={args.batch_size}):")
        for k, v in model.estimate_vram(args.block_size, args.batch_size).items():
            if v:
                print(f"    {k:>30s}: {v}")

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    warmup = min(500, args.steps // 5)

    def get_lr(step):
        if step < warmup:
            return args.lr * step / max(warmup, 1)
        p = (step - warmup) / max(args.steps - warmup, 1)
        return 1e-5 + 0.5 * (args.lr - 1e-5) * (1 + math.cos(math.pi * p))

    # Eğitim öncesi baseline
    val_ppl_before = eval_ppl(model, val_loader, device)
    print(f"\n  Eğitim öncesi Doğrulama PPL: {val_ppl_before:.1f}")
    print(f"  Eğitim öncesi üretim:")
    for p in ["The history of", "Scientists have"]:
        text = generate(model, tokenizer, p, max_tokens=30, device=device)
        print(f"    > \"{text[:140]}\"")

    model.train()
    step = 0
    running_loss = 0.0
    epoch = 0
    start_time = time.time()

    history = []
    nash_history = []
    best_val_ppl = float('inf')
    best_step = 0
    tokens_processed = 0

    print(f"\n  Eğitim başlıyor: {args.steps} adım, lr={args.lr}")
    print(f"  Warmup: {warmup} adım, Cosine decay sonrası")
    print(f"  {'─' * 66}")

    while step < args.steps:
        epoch += 1
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
            tokens_processed += x.numel()
            step += 1

            # Loglama
            if step % args.log_interval == 0:
                avg = running_loss / args.log_interval
                elapsed = time.time() - start_time
                tps = tokens_processed / elapsed

                nash_info = ""
                if hasattr(model, 'get_nash_stats'):
                    stats = model.get_nash_stats()
                    avg_balance = sum(stats['expert_balance']) / len(stats['expert_balance'])
                    avg_temp = sum(stats['temperatures']) / len(stats['temperatures'])
                    nash_info = f" | Denge:{avg_balance:.3f} | T:{avg_temp:.2f}"
                    nash_history.append({
                        'step': step,
                        'balance': avg_balance,
                        'temperature': avg_temp,
                        'regrets': [r for r in stats['regrets']],
                    })

                ppl = math.exp(min(avg, 20))
                print(
                    f"  [{step:>5d}/{args.steps}] "
                    f"Kayıp:{avg:>7.4f} PPL:{ppl:>8.1f} "
                    f"LR:{lr:.1e} T/s:{tps:>7,.0f}{nash_info}",
                    flush=True,
                )
                history.append({
                    'step': step,
                    'loss': avg,
                    'ppl': ppl,
                    'lr': lr,
                    'tps': tps,
                })
                running_loss = 0.0

            # Doğrulama + üretim
            if step % args.eval_interval == 0 or step == args.steps:
                val_ppl = eval_ppl(model, val_loader, device)
                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    best_step = step
                    improved = " ★ YENİ EN İYİ"
                else:
                    improved = ""

                elapsed_min = (time.time() - start_time) / 60
                print(f"\n  ╔══ Adım {step} Değerlendirme ══╗")
                print(f"  ║ Doğrulama PPL: {val_ppl:>8.1f}{improved}")
                print(f"  ║ En İyi PPL:    {best_val_ppl:>8.1f} (adım {best_step})")
                print(f"  ║ Geçen Süre:    {elapsed_min:>8.1f} dk")

                # Expert analizi (Nash modeli için)
                if hasattr(model, 'get_nash_stats'):
                    stats = model.get_nash_stats()
                    print(f"  ║ Expert Denge:  {[f'{b:.3f}' for b in stats['expert_balance']]}")
                    print(f"  ║ Sıcaklıklar:  {[f'{t:.3f}' for t in stats['temperatures']]}")
                    for i, regret in enumerate(stats['regrets']):
                        r_str = [f"{r:+.2f}" for r in regret]
                        print(f"  ║ K{i} Regret:    [{', '.join(r_str)}]")

                print(f"  ║")
                print(f"  ║ Üretimler:")
                prompts = [
                    "The history of science",
                    "In the early years of",
                    "The city of New York",
                    "According to the research",
                ]
                for p in prompts:
                    text = generate(model, tokenizer, p, max_tokens=40, device=device)
                    # İlk 150 karakter
                    clean = text[:150].replace('\n', ' ')
                    print(f"  ║   \"{clean}\"")

                print(f"  ╚{'═' * 40}╝\n")

    # Final
    total_time = time.time() - start_time
    final_ppl = eval_ppl(model, val_loader, device, max_batches=200)
    final_tps = tokens_processed / total_time

    print(f"\n  {'─' * 66}")
    print(f"  {name} -- SONUÇ")
    print(f"  {'─' * 66}")
    print(f"  Son Doğrulama PPL: {final_ppl:.1f}")
    print(f"  En İyi PPL:        {best_val_ppl:.1f} (adım {best_step})")
    print(f"  Toplam Süre:       {total_time:.0f}s ({total_time/60:.1f} dk)")
    print(f"  Ortalama T/s:      {final_tps:,.0f}")
    print(f"  Toplam Token:      {tokens_processed:,}")

    # Expert uzmanlaşma analizi
    expert_analysis = None
    if hasattr(model, 'get_nash_stats'):
        expert_analysis = analyze_expert_specialization(model, val_loader, device)
        if expert_analysis:
            print(f"\n  Expert Uzmanlaşma Analizi:")
            for layer_idx in sorted(expert_analysis.keys()):
                experts = expert_analysis[layer_idx]
                usages = [experts[e]['mean_usage'] for e in sorted(experts.keys())]
                max_u, min_u = max(usages), min(usages)
                ratio = max_u / max(min_u, 1e-8)
                bar_len = 20
                print(f"    Katman {layer_idx}: ", end="")
                for e_idx in sorted(experts.keys()):
                    u = experts[e_idx]['mean_usage']
                    filled = int(u * bar_len / max(max_u, 1e-8))
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(f"E{e_idx}[{bar}]{u:.3f} ", end="")
                print(f" (max/min oranı: {ratio:.1f}x)")

    # Son üretim örnekleri
    print(f"\n  Son Üretim Örnekleri:")
    for p in ["The meaning of life is", "Scientists discovered that", "In the beginning"]:
        text = generate(model, tokenizer, p, max_tokens=60, temperature=0.7, device=device)
        clean = text[:200].replace('\n', ' ')
        print(f"    \"{clean}\"")

    return {
        'name': name,
        'params': total_params,
        'final_ppl': final_ppl,
        'best_ppl': best_val_ppl,
        'best_step': best_step,
        'time': total_time,
        'tps': final_tps,
        'history': history,
        'nash_history': nash_history,
        'expert_analysis': expert_analysis,
    }


# ── Ana ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="50M Nash-Parisi Kapsamlı Deney")
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--block-size', type=int, default=256)
    parser.add_argument('--embed-dim', type=int, default=384)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=2)
    parser.add_argument('--ffn-mult-nash', type=int, default=2)
    parser.add_argument('--neighbor-size', type=int, default=31)
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
    print("║  50M PARAMETRE KAPSAMLI DENEY: PARİSİ vs NASH-PARİSİ              ║")
    print("║  Sığırcık Dikkat (Parisi) + Oyun Teorisi MoE (Nash)               ║")
    print("╚" + "═" * 68 + "╝")
    print()
    print(f"  Cihaz:       {device}")
    print(f"  Adım:        {args.steps}")
    print(f"  Batch:       {args.batch_size}")
    print(f"  Blok:        {args.block_size}")
    print(f"  Embed:       {args.embed_dim}")
    print(f"  Katman:      {args.num_layers}")
    print(f"  Head:        {args.num_heads}")
    print(f"  Expert:      {args.num_experts} (top-{args.top_k})")
    print(f"  Komşu:       {args.neighbor_size}")
    print(f"  LR:          {args.lr}")
    print()

    # Veri yükleme
    print("━" * 70)
    print("  VERİ HAZIRLAMA")
    print("━" * 70)
    train_ds, val_ds, tokenizer = load_data(args.block_size)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True,
    )

    results = []

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL 1: Swarm-LLM (Sadece Parisi)
    # ══════════════════════════════════════════════════════════════════════════
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
        ffn_multiplier=4,
        learning_rate=args.lr,
        warmup_steps=500,
        max_steps=args.steps,
    )

    model_parisi = SwarmLLM(config_parisi)
    r1 = train_model(
        model_parisi, "Swarm-LLM (Sadece Parisi)",
        train_loader, val_loader, tokenizer, config_parisi, device, args,
    )
    results.append(r1)

    # Belleği temizle
    del model_parisi
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL 2: Nash-Parisi (Parisi + Nash MoE)
    # ══════════════════════════════════════════════════════════════════════════
    # Nash MoE: Her expert küçük (2x), ama top-k=2 aktif → etkili FFN ≈ 4x
    # Bu sayede toplam hesaplama Parisi ile benzer olur
    # Parametre fazlası: sadece MoE yükü (router + ek expert ağırlıkları)
    nash_ffn = max(2, args.ffn_mult_nash)
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
        ffn_multiplier=nash_ffn,
        use_nash_moe=True,
        num_experts=args.num_experts,
        top_k_experts=args.top_k,
        nash_warmup_steps=200,
        learning_rate=args.lr,
        warmup_steps=500,
        max_steps=args.steps,
    )

    model_nash = NashParisiLLM(config_nash)
    r2 = train_model(
        model_nash, "Nash-Parisi (Parisi + Nash MoE)",
        train_loader, val_loader, tokenizer, config_nash, device, args,
    )
    results.append(r2)

    # ══════════════════════════════════════════════════════════════════════════
    # KARŞILAŞTIRMA
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  KARŞILAŞTIRMA SONUÇLARI                                          ║")
    print("╚" + "═" * 68 + "╝")

    print(f"\n  {'Model':<35s} {'Param':>8s} {'Son PPL':>9s} {'En İyi':>8s} {'Süre':>8s} {'T/s':>10s}")
    print(f"  {'─' * 82}")
    for r in results:
        print(
            f"  {r['name']:<35s} "
            f"{r['params']/1e6:>6.1f}M "
            f"{r['final_ppl']:>9.1f} "
            f"{r['best_ppl']:>8.1f} "
            f"{r['time']:>6.0f}s "
            f"{r['tps']:>9,.0f}"
        )

    # PPL eğrisi karşılaştırması
    print(f"\n  PPL Eğrisi Karşılaştırması:")
    print(f"  {'Adım':>6s}", end="")
    for r in results:
        short = r['name'][:25]
        print(f" | {short:>25s}", end="")
    print()
    print(f"  {'─' * 62}")

    max_len = max(len(r['history']) for r in results)
    for i in range(0, max_len, max(1, max_len // 20)):
        step_val = results[0]['history'][min(i, len(results[0]['history']) - 1)]['step']
        print(f"  {step_val:>6d}", end="")
        for r in results:
            idx = min(i, len(r['history']) - 1)
            ppl = r['history'][idx]['ppl']
            print(f" | {ppl:>25.1f}", end="")
        print()

    # Hız karşılaştırması
    if len(results) >= 2:
        speed_ratio = results[0]['tps'] / max(results[1]['tps'], 1)
        print(f"\n  Hız Farkı: Parisi {speed_ratio:.2f}x {'hızlı' if speed_ratio > 1 else 'yavaş'}")

    # PPL kazanımı
    if len(results) >= 2:
        ppl_diff = results[0]['final_ppl'] - results[1]['final_ppl']
        ppl_pct = ppl_diff / results[0]['final_ppl'] * 100
        best_ppl_diff = results[0]['best_ppl'] - results[1]['best_ppl']

        print(f"\n  PPL Analizi:")
        print(f"    Son PPL farkı:    {ppl_diff:+.1f} ({'Nash daha iyi' if ppl_diff > 0 else 'Parisi daha iyi'})")
        print(f"    En iyi PPL farkı: {best_ppl_diff:+.1f}")
        print(f"    Göreceli fark:    {ppl_pct:+.1f}%")

        # Parametre verimliliği
        param_ratio = results[1]['params'] / results[0]['params']
        ppl_per_param_0 = results[0]['final_ppl'] / (results[0]['params'] / 1e6)
        ppl_per_param_1 = results[1]['final_ppl'] / (results[1]['params'] / 1e6)
        print(f"\n  Parametre Verimliliği (PPL/M param):")
        print(f"    Parisi:      {ppl_per_param_0:.1f}")
        print(f"    Nash-Parisi: {ppl_per_param_1:.1f}")
        print(f"    Nash param oranı: {param_ratio:.2f}x Parisi")

    # Nash eğrisi
    if results[1]['nash_history']:
        print(f"\n  Nash Denge Evrimi (zaman içinde):")
        nh = results[1]['nash_history']
        for i in range(0, len(nh), max(1, len(nh) // 10)):
            entry = nh[i]
            print(f"    Adım {entry['step']:>5d}: denge={entry['balance']:.4f}, sıcaklık={entry['temperature']:.3f}")
        if len(nh) > 1:
            initial_balance = nh[0]['balance']
            final_balance = nh[-1]['balance']
            print(f"    Denge gelişimi: {initial_balance:.4f} → {final_balance:.4f} "
                  f"({'iyileşme' if final_balance > initial_balance else 'kötüleşme'})")

    # Expert uzmanlaşma
    if results[1]['expert_analysis']:
        print(f"\n  Expert Uzmanlaşma Özeti:")
        analysis = results[1]['expert_analysis']
        for layer_idx in sorted(analysis.keys()):
            experts = analysis[layer_idx]
            usages = sorted([experts[e]['mean_usage'] for e in experts], reverse=True)
            top3 = usages[:3]
            bot3 = usages[-3:]
            specialization = max(usages) / max(min(usages), 1e-8) - 1.0
            print(f"    Katman {layer_idx}: "
                  f"en aktif={top3[0]:.3f}, en az={bot3[-1]:.3f}, "
                  f"uzmanlaşma oranı={specialization:.1f}x")

    # Sonuç
    best = min(results, key=lambda r: r['final_ppl'])
    print(f"\n  ★ EN İYİ MODEL: {best['name']}")
    print(f"    PPL: {best['final_ppl']:.1f}")
    print(f"    Parametreler: {best['params']/1e6:.1f}M")
    print(f"    Süre: {best['time']:.0f}s")

    if len(results) >= 2:
        if results[1]['final_ppl'] < results[0]['final_ppl']:
            print(f"\n  ✓ Nash-Parisi 50M ölçekte Parisi'yi geçti!")
            print(f"    Nash MoE expert routing başarılı: daha iyi öğrenme.")
        else:
            gap = results[1]['final_ppl'] - results[0]['final_ppl']
            print(f"\n  → Parisi hala {gap:.1f} PPL önde.")
            if gap < 20:
                print(f"    Nash-Parisi yakın -- daha uzun eğitim veya daha büyük model ile geçebilir.")
            else:
                print(f"    Nash MoE overhead bu ölçekte henüz karşılanmadı.")

    # JSON kaydet
    save_path = "results_50m.json"
    save_data = []
    for r in results:
        save_r = {k: v for k, v in r.items() if k != 'expert_analysis'}
        save_r['expert_analysis'] = str(r.get('expert_analysis', ''))[:500]
        save_data.append(save_r)

    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Sonuçlar kaydedildi: {save_path}")
    print()


if __name__ == "__main__":
    main()
