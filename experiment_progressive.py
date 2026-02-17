#!/usr/bin/env python3
"""
Strateji 1: Aşamalı Eğitim (Parisi → Nash-Parisi)
====================================================
Önce Sığırcık Dikkat (Parisi) ile modeli eğit,
sonra AYNI modele Nash MoE ekleyerek eğitime devam et.

Akış:
  Faz 1: SwarmLLM (Parisi) → 2500 adım eğit
         Model attention + embedding + FFN öğrenir
  
  Faz 2: Ağırlık Transferi
         Attention, embedding, noise, normları kopyala
         Nash MoE rastgele başlat
  
  Faz 3: NashParisiLLM → 2500 adım daha eğit
         Transferred weights: düşük LR (zaten öğrenmiş)
         Nash MoE weights: yüksek LR (yeni öğrenecek)

Karşılaştırma:
  A) Strateji 1 (aşamalı): Parisi 2500 → Nash-Parisi 2500
  B) Baseline (sıfırdan): Nash-Parisi 5000 adım

Kullanım:
    PYTHONUNBUFFERED=1 python3 -u experiment_progressive.py
"""

import argparse
import gc
import json
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

    train_ds = TokenDataset(train_tokens, block_size)
    val_ds = TokenDataset(val_tokens, block_size)
    return train_ds, val_ds, tokenizer


# ── Yardımcılar ──────────────────────────────────────────────────────────────

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


# ── Ağırlık Transferi ────────────────────────────────────────────────────────

def transfer_weights(
    src_model: SwarmLLM,
    dst_model: NashParisiLLM,
) -> dict:
    """
    SwarmLLM (Parisi) → NashParisiLLM ağırlık transferi.

    Transfer edilen bileşenler:
      - tok_emb (embedding)
      - layers[i].attn_norm (LayerNorm)
      - layers[i].ffn_norm (LayerNorm)
      - layers[i].attention (StarlingAttention -- birebir aynı)
      - layers[i].noise (AdaptiveParisiNoise -- birebir aynı)
      - final_norm (LayerNorm)
      - lm_head (weight tying ile tok_emb'den gelir)

    Transfer EDİLMEYEN (yeni, rastgele):
      - layers[i].nash_moe (expert'ler, router, shared expert)

    Returns:
        dict: Transfer istatistikleri
    """
    src_state = src_model.state_dict()
    dst_state = dst_model.state_dict()

    transferred = []
    skipped_shape = []
    skipped_missing = []
    new_params = []

    for dst_key in dst_state:
        # Nash MoE parametreleri -- yeni, transfer etme
        if 'nash_moe' in dst_key:
            new_params.append(dst_key)
            continue

        # rope parametreleri -- her modelde kendi rope'u var
        if 'rope' in dst_key:
            skipped_missing.append(dst_key)
            continue

        # Aynı isimli parametre src'de var mı?
        if dst_key in src_state:
            if src_state[dst_key].shape == dst_state[dst_key].shape:
                dst_state[dst_key] = src_state[dst_key].clone()
                transferred.append(dst_key)
            else:
                skipped_shape.append(
                    f"{dst_key}: src={src_state[dst_key].shape} dst={dst_state[dst_key].shape}"
                )
        else:
            skipped_missing.append(dst_key)

    # Yeni state'i yükle
    dst_model.load_state_dict(dst_state)

    stats = {
        'transferred': len(transferred),
        'new_params': len(new_params),
        'skipped_shape': len(skipped_shape),
        'skipped_missing': len(skipped_missing),
        'transferred_keys': transferred,
        'new_keys': new_params,
    }

    return stats


# ── Eğitim Fonksiyonları ─────────────────────────────────────────────────────

def make_lr_scheduler(base_lr, total_steps, warmup_steps):
    """Cosine decay with warmup."""
    def get_lr(step):
        if step < warmup_steps:
            return base_lr * step / max(warmup_steps, 1)
        p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 1e-5 + 0.5 * (base_lr - 1e-5) * (1 + math.cos(math.pi * p))
    return get_lr


def train_phase(
    model, name, train_loader, val_loader, tokenizer,
    device, num_steps, base_lr,
    log_interval=50, eval_interval=500,
    param_groups=None,
    start_step=0,
):
    """
    Bir eğitim fazını çalıştır.

    Args:
        param_groups: Opsiyonel -- differential LR için parametre grupları
                      [{'params': ..., 'lr': ...}, ...]
    """
    separator = "─" * 66

    if param_groups is not None:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01, betas=(0.9, 0.95))
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=0.01, betas=(0.9, 0.95),
        )

    warmup = min(300, num_steps // 5)
    lr_fn = make_lr_scheduler(base_lr, num_steps, warmup)

    model.train()
    step = 0
    running_loss = 0.0
    epoch = 0
    start_time = time.time()
    history = []
    tokens_processed = 0
    best_val_ppl = float('inf')

    print(f"  {separator}")
    print(f"  Eğitim: {num_steps} adım, LR={base_lr}")
    if param_groups:
        for i, pg in enumerate(param_groups):
            count = sum(p.numel() for p in pg['params'])
            print(f"    Grup {i}: {pg.get('group_name', '?'):>25s} | "
                  f"LR={pg['lr']:.1e} | {count:,} param")
    print(f"  {separator}")

    while step < num_steps:
        epoch += 1
        for x, y in train_loader:
            if step >= num_steps:
                break
            x, y = x.to(device), y.to(device)

            # LR güncelle
            scale = lr_fn(step)
            for pg in optimizer.param_groups:
                # Her grubun kendi base lr'si var, scale ile çarp
                pg_base = pg.get('base_lr', base_lr)
                pg['lr'] = scale * pg_base / base_lr

            out = model(x, targets=y)
            loss = out['loss']

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            tokens_processed += x.numel()
            step += 1
            global_step = start_step + step

            if step % log_interval == 0:
                avg = running_loss / log_interval
                elapsed = time.time() - start_time
                tps = tokens_processed / elapsed
                ppl = math.exp(min(avg, 20))

                nash_info = ""
                if hasattr(model, 'get_nash_stats'):
                    stats = model.get_nash_stats()
                    avg_temp = sum(stats['temperatures']) / len(stats['temperatures'])
                    nash_info = f" | T:{avg_temp:.3f}"

                print(
                    f"  [{global_step:>5d}] "
                    f"Kayıp:{avg:>7.4f} PPL:{ppl:>8.1f} "
                    f"LR:{scale:.1e} T/s:{tps:>7,.0f}{nash_info}",
                    flush=True,
                )
                history.append({
                    'step': global_step,
                    'loss': avg,
                    'ppl': ppl,
                    'tps': tps,
                })
                running_loss = 0.0

            if step % eval_interval == 0 or step == num_steps:
                # MPS bellek temizliği (eval öncesi)
                if device.type == 'mps':
                    torch.mps.empty_cache()

                val_ppl = eval_ppl(model, val_loader, device)
                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    improved = " ★"
                else:
                    improved = ""

                elapsed_min = (time.time() - start_time) / 60
                print(f"\n  ══ Adım {global_step} | Val PPL: {val_ppl:.1f}{improved} "
                      f"| En İyi: {best_val_ppl:.1f} | {elapsed_min:.1f} dk ══")

                for p in ["The history of science", "The city of New York"]:
                    text = generate(model, tokenizer, p, max_tokens=40, device=device)
                    print(f"    \"{text[:150]}\"")

                if hasattr(model, 'get_nash_stats'):
                    stats = model.get_nash_stats()
                    for i, regret in enumerate(stats['regrets']):
                        r_str = [f"{r:+.1f}" for r in regret]
                        print(f"    K{i}: [{', '.join(r_str)}]")

                # MPS bellek temizliği (eval sonrası)
                if device.type == 'mps':
                    torch.mps.empty_cache()

                print()

    total_time = time.time() - start_time
    return {
        'history': history,
        'best_ppl': best_val_ppl,
        'time': total_time,
        'tokens': tokens_processed,
    }


# ── Ana ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Strateji 1: Aşamalı Eğitim")
    parser.add_argument('--total-steps', type=int, default=5000)
    parser.add_argument('--phase1-ratio', type=float, default=0.5,
                        help="Faz 1'e ayrılan adım oranı (0.5 = yarısı Parisi)")
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
    parser.add_argument('--nash-lr-mult', type=float, default=3.0,
                        help="Nash MoE için LR çarpanı (yeni katmanlar daha hızlı öğrenir)")
    parser.add_argument('--transferred-lr-mult', type=float, default=0.3,
                        help="Transfer edilen katmanlar için LR çarpanı (zaten öğrenmiş)")
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--eval-interval', type=int, default=500)
    args = parser.parse_args()

    phase1_steps = int(args.total_steps * args.phase1_ratio)
    phase2_steps = args.total_steps - phase1_steps

    device = torch.device(
        'mps' if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available()
        else 'cpu'
    )

    print()
    print("╔" + "═" * 68 + "╗")
    print("║  STRATEJİ 1: AŞAMALI EĞİTİM (Parisi → Nash-Parisi)               ║")
    print("║  Curriculum Learning: Önce dikkat öğren, sonra expert routing ekle  ║")
    print("╚" + "═" * 68 + "╝")
    print()
    print(f"  Cihaz:          {device}")
    print(f"  Toplam Adım:    {args.total_steps}")
    print(f"  Faz 1 (Parisi): {phase1_steps} adım ({args.phase1_ratio*100:.0f}%)")
    print(f"  Faz 2 (Nash):   {phase2_steps} adım ({(1-args.phase1_ratio)*100:.0f}%)")
    print(f"  Nash LR çarpan: {args.nash_lr_mult}x (yeni MoE hızlı öğrenir)")
    print(f"  Transfer LR:    {args.transferred_lr_mult}x (eski ağırlıklar yavaş)")
    print(f"  Expert:         {args.num_experts} (top-{args.top_k})")
    print()

    # Veri
    print("━" * 70)
    print("  VERİ HAZIRLAMA")
    print("━" * 70)
    train_ds, val_ds, tokenizer = load_data(args.block_size)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # FAZ 1: Parisi (Sığırcık Dikkat) Eğitimi
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  FAZ 1: Parisi Eğitimi (Sığırcık Dikkat + FFN)                    ║")
    print("║  Model dikkat kalıplarını ve dil yapısını öğreniyor                ║")
    print("╚" + "═" * 68 + "╝")

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
    )

    model_parisi = SwarmLLM(config_parisi)
    total_p = sum(p.numel() for p in model_parisi.parameters())
    print(f"\n  Parisi Model: {total_p:,} parametre ({total_p/1e6:.1f}M)")

    model_parisi.to(device)

    # Faz 1 öncesi baseline
    val_ppl_init = eval_ppl(model_parisi, val_loader, device)
    print(f"  Başlangıç PPL: {val_ppl_init:.1f}")

    phase1_result = train_phase(
        model_parisi, "Faz 1: Parisi",
        train_loader, val_loader, tokenizer, device,
        num_steps=phase1_steps, base_lr=args.lr,
        log_interval=args.log_interval, eval_interval=args.eval_interval,
        start_step=0,
    )

    phase1_ppl = eval_ppl(model_parisi, val_loader, device)
    print(f"  Faz 1 Son PPL: {phase1_ppl:.1f}")
    print(f"  Faz 1 Süre: {phase1_result['time']:.0f}s ({phase1_result['time']/60:.1f} dk)")

    # ══════════════════════════════════════════════════════════════════════════
    # AĞIRLIK TRANSFERİ: Parisi → Nash-Parisi
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  AĞIRLIK TRANSFERİ: Parisi → Nash-Parisi                          ║")
    print("║  Attention + Embedding + Noise aktarılıyor                         ║")
    print("║  Nash MoE katmanları rastgele başlatılıyor                         ║")
    print("╚" + "═" * 68 + "╝")

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
        ffn_multiplier=args.ffn_mult_nash,
        use_nash_moe=True,
        num_experts=args.num_experts,
        top_k_experts=args.top_k,
        nash_warmup_steps=100,
        learning_rate=args.lr,
    )

    model_nash = NashParisiLLM(config_nash)
    total_nash = sum(p.numel() for p in model_nash.parameters())
    print(f"\n  Nash-Parisi Model: {total_nash:,} parametre ({total_nash/1e6:.1f}M)")

    # Transfer
    transfer_stats = transfer_weights(model_parisi, model_nash)

    print(f"\n  Transfer Sonucu:")
    print(f"    Aktarılan parametre sayısı: {transfer_stats['transferred']}")
    print(f"    Yeni (Nash MoE) parametre:  {transfer_stats['new_params']}")
    print(f"    Atlanılan (boyut uyumsuz):  {transfer_stats['skipped_shape']}")
    print(f"    Atlanılan (yok):            {transfer_stats['skipped_missing']}")

    # Transfer sonrası PPL (Nash MoE rastgele olduğu için kötüleşecek)
    model_nash.to(device)
    val_ppl_post_transfer = eval_ppl(model_nash, val_loader, device)
    print(f"\n  Transfer sonrası PPL: {val_ppl_post_transfer:.1f}")
    print(f"  (Faz 1 sonu PPL: {phase1_ppl:.1f} -- Nash MoE rastgele olduğu için artış normal)")

    # Parisi modelini bellekten sil ve MPS cache'i temizle
    del model_parisi
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # FAZ 2: Nash-Parisi Eğitimine Devam (Differential Learning Rate)
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  FAZ 2: Nash-Parisi Eğitimi (Transfer + Nash MoE)                 ║")
    print("║  Transferred weights: düşük LR (korunuyor)                        ║")
    print("║  Nash MoE weights: yüksek LR (hızlı öğrenme)                     ║")
    print("╚" + "═" * 68 + "╝")

    # Differential Learning Rate grupları
    transferred_params = []
    nash_moe_params = []

    for name, param in model_nash.named_parameters():
        if 'nash_moe' in name:
            nash_moe_params.append(param)
        else:
            transferred_params.append(param)

    transferred_lr = args.lr * args.transferred_lr_mult
    nash_lr = args.lr * args.nash_lr_mult

    param_groups = [
        {
            'params': transferred_params,
            'lr': transferred_lr,
            'base_lr': transferred_lr,
            'group_name': 'Transferred (Parisi)',
        },
        {
            'params': nash_moe_params,
            'lr': nash_lr,
            'base_lr': nash_lr,
            'group_name': 'Nash MoE (Yeni)',
        },
    ]

    t_count = sum(p.numel() for p in transferred_params)
    n_count = sum(p.numel() for p in nash_moe_params)
    print(f"\n  Transferred parametreler: {t_count:,} → LR={transferred_lr:.1e}")
    print(f"  Nash MoE parametreler:    {n_count:,} → LR={nash_lr:.1e}")
    print(f"  LR oranı: Nash/Transfer = {nash_lr/transferred_lr:.1f}x")

    phase2_result = train_phase(
        model_nash, "Faz 2: Nash-Parisi",
        train_loader, val_loader, tokenizer, device,
        num_steps=phase2_steps, base_lr=args.lr,
        log_interval=args.log_interval, eval_interval=args.eval_interval,
        param_groups=param_groups,
        start_step=phase1_steps,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SONUÇLAR
    # ══════════════════════════════════════════════════════════════════════════
    final_ppl = eval_ppl(model_nash, val_loader, device, max_batches=200)
    total_time = phase1_result['time'] + phase2_result['time']
    all_history = phase1_result['history'] + phase2_result['history']

    print()
    print("╔" + "═" * 68 + "╗")
    print("║  SONUÇLAR                                                          ║")
    print("╚" + "═" * 68 + "╝")

    print(f"""
  Strateji 1: Aşamalı Eğitim (Parisi → Nash-Parisi)
  ───────────────────────────────────────────────────
  Faz 1 (Parisi, {phase1_steps} adım):
    Süre:       {phase1_result['time']:.0f}s ({phase1_result['time']/60:.1f} dk)
    Son PPL:    {phase1_ppl:.1f}

  Ağırlık Transferi:
    Aktarılan:  {transfer_stats['transferred']} parametre
    Yeni:       {transfer_stats['new_params']} parametre (Nash MoE)
    PPL değişimi: {phase1_ppl:.1f} → {val_ppl_post_transfer:.1f} (transfer sonrası)

  Faz 2 (Nash-Parisi, {phase2_steps} adım):
    Süre:       {phase2_result['time']:.0f}s ({phase2_result['time']/60:.1f} dk)
    Son PPL:    {final_ppl:.1f}
    En İyi PPL: {phase2_result['best_ppl']:.1f}

  TOPLAM:
    Süre:       {total_time:.0f}s ({total_time/60:.1f} dk)
    Son PPL:    {final_ppl:.1f}
    """)

    # Önceki deney sonuçlarıyla karşılaştırma
    print("  Karşılaştırma (önceki 50M deney sonuçları):")
    print(f"  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │ {'Yöntem':<40s} {'PPL':>8s} {'Süre':>8s} │")
    print(f"  ├─────────────────────────────────────────────────────────────┤")
    print(f"  │ {'Parisi (bağımsız, 5000 adım)':<40s} {'235.1':>8s} {'59 dk':>8s} │")
    print(f"  │ {'Nash-Parisi (bağımsız, 5000 adım)':<40s} {'235.6':>8s} {'84 dk':>8s} │")
    print(f"  │ {'STRATEJİ 1 (aşamalı, 5000 adım)':<40s} {final_ppl:>8.1f} {total_time/60:>6.1f} dk │")
    print(f"  └─────────────────────────────────────────────────────────────┘")

    if final_ppl < 235:
        print(f"\n  ✓ Strateji 1 başarılı: Aşamalı eğitim bağımsız eğitimi geçti!")
    elif final_ppl < 240:
        print(f"\n  ≈ Strateji 1 rekabetçi: Bağımsız eğitimle benzer sonuç.")
    else:
        print(f"\n  → Strateji 1 henüz yeterli değil: Daha fazla Faz 2 adımı gerekebilir.")

    # Son üretim örnekleri
    print(f"\n  Son Üretim Örnekleri:")
    for p in ["The meaning of life is", "Scientists discovered that", "In the beginning"]:
        text = generate(model_nash, tokenizer, p, max_tokens=50, temperature=0.7, device=device)
        clean = text[:200].replace('\n', ' ')
        print(f"    \"{clean}\"")

    # Expert analizi
    if hasattr(model_nash, 'get_nash_stats'):
        stats = model_nash.get_nash_stats()
        print(f"\n  Nash Expert İstatistikleri:")
        for i, (temp, regret) in enumerate(zip(stats['temperatures'], stats['regrets'])):
            r_str = [f"{r:+.1f}" for r in regret]
            print(f"    Katman {i}: sıcaklık={temp:.3f}, regret=[{', '.join(r_str)}]")

    # PPL eğrisini kaydet
    print(f"\n  PPL Eğrisi:")
    print(f"  {'Adım':>6s} {'PPL':>10s} {'Faz':>10s}")
    print(f"  {'─' * 30}")
    for h in all_history[::max(1, len(all_history) // 20)]:
        faz = "Parisi" if h['step'] <= phase1_steps else "Nash-P"
        print(f"  {h['step']:>6d} {h['ppl']:>10.1f} {faz:>10s}")

    # JSON kaydet
    with open("results_progressive.json", 'w') as f:
        json.dump({
            'strategy': 'progressive',
            'phase1_steps': phase1_steps,
            'phase2_steps': phase2_steps,
            'phase1_ppl': phase1_ppl,
            'post_transfer_ppl': val_ppl_post_transfer,
            'final_ppl': final_ppl,
            'best_ppl': phase2_result['best_ppl'],
            'total_time': total_time,
            'history': all_history,
            'transfer_stats': {
                'transferred': transfer_stats['transferred'],
                'new': transfer_stats['new_params'],
            },
        }, f, indent=2, default=str)
    print(f"\n  Sonuçlar kaydedildi: results_progressive.json")
    print()


if __name__ == "__main__":
    main()
