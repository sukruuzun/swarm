#!/usr/bin/env python3
"""
Birleşik Parisi-Nash v5 Deneyi: Keskin Uzmanlaşma + Uzun Koşu
================================================================
v4'ün başlattığı uzmanlaşmayı, daha güçlü router ve uzun eğitimle
baseline seviyesine taşımayı hedefler.

v5 Değişiklikleri (v4 üzerine):
  ✓ 7500 step (v4: 3000) -- model doymamıştı
  ✓ MLP Router: LN → 384→64→GELU→4 (v4: lineer 384→4)
  ✓ lb_coeff = 0.0001 (v4: 0.001) -- agresif uzmanlaşma
  ✓ min LR = 5e-5 (v4: 1e-5) -- kuyrukta öğrenme devam
  ✓ Decoupled clipping: router 0.5, expert 1.0
  ✓ Temperature schedule 7500'e uzatılmış

Kullanım:
    PYTHONUNBUFFERED=1 python3 -u experiment_unified_v5.py
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
    steps = 7500
    batch_size = 8
    block_size = 256
    lr = 5e-4
    min_lr = 5e-5
    log_interval = 50
    eval_interval = 500

    device = torch.device(
        'mps' if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available()
        else 'cpu'
    )

    print()
    print("╔" + "═" * 68 + "╗")
    print("║  BİRLEŞİK PARİSİ-NASH v5: KESKİN UZMANLAŞMA + UZUN KOŞU           ║")
    print("║  v_i = p_k · φ_k(Σ_j∈N(i) α_ij^k · V_k_j) + η·ε                  ║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("  v5 İyileştirmeleri:")
    print("    ✓ MLP Router: LN → 384→64→GELU→4 (akıllı yönlendirme)")
    print("    ✓ 7500 step (v4: 3000) -- doymamış modeli uzun çalıştır")
    print(f"    ✓ min LR = {min_lr} (v4: 1e-5) -- kuyrukta öğrenme")
    print("    ✓ lb_coeff = 0.0001 (v4: 0.001) -- agresif uzmanlaşma")
    print("    ✓ Decoupled clipping: router 0.5, expert 1.0")
    print("    ✓ Temperature schedule 7500 step'e uzatılmış")
    print()
    print(f"  Cihaz: {device} | Adım: {steps}")
    print()
    print("  Önceki sonuçlar:")
    print("    ① Parisi (Ardışık):     39.7M → PPL 302.6  (33 dk)")
    print("    ② Nash-Parisi (Ardışık): 57.4M → PPL 305.0  (50 dk)")
    print("    ③ Birleşik v1 (collapse): 57.1M → PPL 366.3")
    print("    ④ Birleşik v2 (aş.denge): 64.2M → PPL ~611")
    print("    ⑤ Birleşik v3 (gradyan körlüğü): 64.2M → PPL 443.1")
    print("    ⑥ Birleşik v4 (diff. routing): 64.2M → PPL 386.9")
    print()

    train_ds, val_ds, tokenizer = load_data(block_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    # ── Model: Birleşik v5 (MLP router, low LB) ──
    config = SwarmConfig(
        vocab_size=tokenizer.vocab_size, embed_dim=384,
        num_heads=8, num_layers=8,
        max_seq_len=block_size, dropout=0.1,
        neighbor_size=31, use_multi_scale=False,
        noise_strength=0.01, ffn_multiplier=2,
        use_nash_moe=True, num_experts=4, top_k_experts=1,
    )
    config.max_steps = steps
    model = UnifiedParisiNashLLM(config)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"  ⑦ BİRLEŞİK v5 Parametreleri: {total_p:,} ({total_p/1e6:.1f}M)")
    for k, v in model.count_parameters().items():
        if isinstance(v, int) and v > 0:
            pct = v / total_p * 100
            print(f"    {k:>25s}: {v:>12,} ({pct:>5.1f}%)")
    print()

    # Temperature Annealing bilgisi (7500 step'e uzatılmış)
    print("  Temperature Annealing Schedule (7500 step):")
    for s in [0, 1000, 2000, 3000, 4000, 5000, 6000, 7500]:
        p = min(s / steps, 1.0)
        t = model.t_end + 0.5 * (model.t_start - model.t_end) * (1 + math.cos(math.pi * p))
        print(f"    Adım {s:>5d}: T = {t:.3f}")
    print()

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))

    # Decoupled clipping icin parametre gruplari
    router_params = model.get_router_params()
    expert_params = model.get_expert_params()

    warmup = 500

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

            # ── Temperature Annealing ──
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
    print("║  SONUÇ: BİRLEŞİK v5 vs TÜM MODELLER                              ║")
    print("╚" + "═" * 68 + "╝")
    print()
    print(f"  {'Model':<45s} {'Param':>8s} {'PPL':>8s} {'Süre':>8s}")
    print(f"  {'─' * 73}")
    print(f"  {'① Parisi (Ardışık Attn+FFN)':<45s} {'39.7M':>8s} {'302.6':>8s} {'33dk':>8s}")
    print(f"  {'② Nash-Parisi (Ardışık Attn→MoE)':<45s} {'57.4M':>8s} {'305.0':>8s} {'50dk':>8s}")
    print(f"  {'③ Birleşik v1 (collapse)':<45s} {'57.1M':>8s} {'366.3':>8s} {'50dk':>8s}")
    print(f"  {'④ Birleşik v2 (aşırı denge)':<45s} {'64.2M':>8s} {'~611':>8s} {'120dk':>8s}")
    print(f"  {'⑤ Birleşik v3 (gradyan körlüğü)':<45s} {'64.2M':>8s} {'443.1':>8s} {'9dk':>8s}")
    print(f"  {'⑥ Birleşik v4 (diff. routing)':<45s} {'64.2M':>8s} {'386.9':>8s} {'7dk':>8s}")
    print(f"  {'⑦ BİRLEŞİK v5 (keskin uzmanlaşma)':<45s} "
          f"{total_p/1e6:>6.1f}M {best_ppl:>8.1f} {total_time/60:>5.0f}dk")

    # Değerlendirme
    print()
    if best_ppl < 302.6:
        print(f"  ★★★ BASELINE'I GEÇTİ! PPL {best_ppl:.1f} < 302.6 (Parisi) ★★★")
        print(f"  ★★★ BİRLEŞİK TEORİ ÇALIŞIYOR! ★★★")
    elif best_ppl < 305.0:
        print(f"  ★★★ Nash-Parisi'yi GEÇTİ! PPL {best_ppl:.1f} < 305.0 ★★★")
        print(f"     Parisi baseline'a fark: {best_ppl - 302.6:+.1f}")
    elif best_ppl < 366.3:
        print(f"  ★★ v1'i GEÇTİ! PPL {best_ppl:.1f} < 366.3")
        print(f"     Baseline'a fark: {best_ppl - 302.6:+.1f}")
    elif best_ppl < 386.9:
        print(f"  ★ v4'ü GEÇTİ! PPL {best_ppl:.1f} < 386.9")
    else:
        print(f"  v4 seviyesinde: PPL {best_ppl:.1f}")

    # Kaydet
    result = {
        'name': 'BİRLEŞİK v5 (keskin uzmanlaşma + uzun koşu)',
        'params': total_p,
        'final_ppl': final_ppl,
        'best_ppl': best_ppl,
        'time': total_time,
        'history': history,
        'config': {
            't_start': model.t_start, 't_end': model.t_end,
            'lb_coeff': model.lb_coeff, 'top_k': 1,
            'num_experts': 4, 'embed_dim': 384, 'num_layers': 8,
            'router': 'MLP (384→64→4)', 'min_lr': min_lr,
            'steps': steps, 'warmup': warmup,
            'clip_expert': 1.0, 'clip_router': 0.5,
        },
    }
    with open("results_unified_v5.json", 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Sonuçlar: results_unified_v5.json")
    print()


if __name__ == "__main__":
    main()
