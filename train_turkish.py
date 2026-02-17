#!/usr/bin/env python3
"""
Swarm-LLM Türkçe Eğitim
==========================
BellaTurca (Akademik Türkçe Derlem) veri seti üzerinde
Sığırcık Dikkat mekanizmalı dil modeli eğitimi.

Kullanım:
    python train_turkish.py
    python train_turkish.py --steps 3000
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


# ── Veri Seti ────────────────────────────────────────────────────────────────

class TokenizedTextDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, block_size: int):
        self.data = token_ids
        self.block_size = block_size

    def __len__(self):
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        x = self.data[start : start + self.block_size]
        y = self.data[start + 1 : start + self.block_size + 1]
        return x, y


def load_turkish_data(max_samples=5000):
    """
    BellaTurca Akademik Derlem'den Türkçe metin yükle ve tokenize et.

    GPT-2 tokenizer byte-level BPE kullandığı için Türkçe karakterleri
    de destekler (ş, ç, ğ, ı, ö, ü). Verimlilik açısından ideal değildir
    ama demo için yeterlidir.
    """
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    print("  BellaTurca Akademik Derlem yükleniyor...")
    ds = load_dataset(
        "turkish-nlp-suite/BellaTurca", "AkademikDerlem",
        split="train", streaming=True,
    )

    print("  GPT-2 tokenizer yükleniyor (Türkçe desteği ile)...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    print(f"  İlk {max_samples:,} örnek okunuyor...")
    all_tokens = []
    total_chars = 0
    count = 0

    for sample in ds:
        text = sample.get('text', '')
        if len(text) < 50:
            continue

        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        total_chars += len(text)
        count += 1

        if count % 500 == 0:
            print(f"    {count:,} örnek okundu... ({len(all_tokens):,} token)")

        if count >= max_samples:
            break

    all_tokens = torch.tensor(all_tokens, dtype=torch.long)

    # %90 eğitim, %10 doğrulama
    split = int(len(all_tokens) * 0.9)
    train_tokens = all_tokens[:split]
    val_tokens = all_tokens[split:]

    print(f"\n  Veri Seti Özeti:")
    print(f"    Toplam metin: {count:,} adet")
    print(f"    Toplam karakter: {total_chars:,}")
    print(f"    Toplam token: {len(all_tokens):,}")
    print(f"    Eğitim: {len(train_tokens):,} token")
    print(f"    Doğrulama: {len(val_tokens):,} token")
    print(f"    Ortalama token/metin: {len(all_tokens)//max(count,1):,}")
    print(f"    Vocab boyutu: {tokenizer.vocab_size:,}")

    # Örnek tokenizasyon göster
    sample_text = "Türkiye Cumhuriyeti bir hukuk devletidir."
    sample_tokens = tokenizer.encode(sample_text)
    sample_decoded = [tokenizer.decode([t]) for t in sample_tokens]
    print(f"\n  Örnek tokenizasyon:")
    print(f"    Metin: \"{sample_text}\"")
    print(f"    Token'lar: {sample_decoded}")
    print(f"    Token sayısı: {len(sample_tokens)}")

    return train_tokens, val_tokens, tokenizer


# ── Metin Üretimi ────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_sample(model, tokenizer, prompt, max_tokens=80, temperature=0.8, device='cpu'):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated = input_ids

    for _ in range(max_tokens):
        context = generated[:, -model.config.max_seq_len:]
        outputs = model(context)
        logits = outputs['logits'][:, -1, :] / temperature

        top_k = 40
        top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < top_vals[:, -1:]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)

    model.train()
    return tokenizer.decode(generated[0].tolist())


@torch.no_grad()
def compute_perplexity(model, dataloader, device, max_batches=50):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        outputs = model(x, targets=y)
        total_loss += outputs['loss'].item()
        total_batches += 1

    model.train()
    return math.exp(total_loss / max(total_batches, 1))


# ── Ana Eğitim ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Swarm-LLM Türkçe Eğitim")
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--neighbor-size', type=int, default=31)
    parser.add_argument('--lr', type=float, default=6e-4)
    parser.add_argument('--max-samples', type=int, default=3000,
                        help="Veri setinden okunacak maksimum örnek sayısı")
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--gen-interval', type=int, default=400)
    args = parser.parse_args()

    device = torch.device(
        'mps' if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available()
        else 'cpu'
    )

    print(f"\n{'='*65}")
    print(f"  Swarm-LLM: Türkçe Eğitim (Sığırcık Dikkat)")
    print(f"  Cihaz: {device}")
    print(f"{'='*65}\n")

    # ── 1. Türkçe Veri Yükleme ────────────────────────────────────────────
    print("─── Türkçe Veri Seti Hazırlanıyor ───\n")
    train_tokens, val_tokens, tokenizer = load_turkish_data(max_samples=args.max_samples)

    train_dataset = TokenizedTextDataset(train_tokens, args.block_size)
    val_dataset = TokenizedTextDataset(val_tokens, args.block_size)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, drop_last=True,
    )

    print(f"\n  Eğitim örnekleri: {len(train_dataset):,}")
    print(f"  Doğrulama örnekleri: {len(val_dataset):,}")

    # ── 2. Model Oluşturma ─────────────────────────────────────────────────
    print(f"\n─── Model Oluşturuluyor ───\n")

    config = SwarmConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.block_size,
        dropout=0.1,
        neighbor_size=args.neighbor_size,
        use_multi_scale=False,
        noise_strength=0.01,
        noise_learnable=True,
        learning_rate=args.lr,
        warmup_steps=min(200, args.steps // 10),
        max_steps=args.steps,
    )

    model = SwarmLLM(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params/1e6:.1f}M parametre")
    print(f"  Katmanlar: {args.num_layers}, Head: {args.num_heads}")
    print(f"  Embed: {args.embed_dim}, Pencere: {args.neighbor_size}")

    vram = model.estimate_vram(args.block_size, args.batch_size)
    print(f"  VRAM (Swarm): {vram['total_swarm']}")
    print(f"  VRAM (Standart): {vram['total_standard']}")
    print(f"  Kazanç: {vram['attention_savings']}")

    # ── 3. Optimizer ───────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=0.01, betas=(0.9, 0.95),
    )

    warmup_steps = config.warmup_steps
    def get_lr(step):
        if step < warmup_steps:
            return args.lr * step / warmup_steps
        progress = (step - warmup_steps) / max(args.steps - warmup_steps, 1)
        return 1e-5 + 0.5 * (args.lr - 1e-5) * (1 + math.cos(math.pi * progress))

    # ── 4. Eğitim ─────────────────────────────────────────────────────────
    turkish_prompts = [
        "Türkiye",
        "Bilim insanları",
        "Bu araştırma",
        "İstanbul",
        "Eğitim sisteminde",
    ]

    print(f"\n{'='*65}")
    print(f"  Eğitim Başlıyor! ({args.steps} adım)")
    print(f"{'='*65}\n")

    # Eğitim öncesi üretim
    print("  ─── Eğitim ÖNCE (rastgele ağırlıklar) ───")
    for p in turkish_prompts[:2]:
        text = generate_sample(model, tokenizer, p, max_tokens=30, device=device)
        print(f"  > \"{text[:150]}\"")
    print()

    model.train()
    step = 0
    epoch = 0
    best_val_ppl = float('inf')
    running_loss = 0.0
    start_time = time.time()

    while step < args.steps:
        epoch += 1
        for x, y in train_loader:
            if step >= args.steps:
                break

            x, y = x.to(device), y.to(device)

            lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            outputs = model(x, targets=y)
            loss = outputs['loss']

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            step += 1

            # Loglama
            if step % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                elapsed = time.time() - start_time
                tokens_sec = step * args.batch_size * args.block_size / elapsed

                print(
                    f"  Adım {step:>5d}/{args.steps} | "
                    f"Kayıp: {avg_loss:.4f} | "
                    f"PPL: {math.exp(avg_loss):.1f} | "
                    f"LR: {lr:.2e} | "
                    f"Token/s: {tokens_sec:,.0f}"
                )
                running_loss = 0.0

            # Üretim örnekleri
            if step % args.gen_interval == 0:
                print(f"\n  ─── Adım {step}: Türkçe Üretim Örnekleri ───")
                for p in turkish_prompts:
                    text = generate_sample(model, tokenizer, p, max_tokens=50, device=device)
                    short = text[:180].replace('\n', ' ')
                    print(f"  > \"{short}\"")

                val_ppl = compute_perplexity(model, val_loader, device)
                print(f"\n  Doğrulama PPL: {val_ppl:.1f}")

                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    os.makedirs("checkpoints", exist_ok=True)
                    torch.save({
                        'step': step, 'model_state_dict': model.state_dict(),
                        'config': config, 'val_ppl': val_ppl,
                    }, "checkpoints/swarm_turkish_best.pt")
                    print(f"  En iyi model kaydedildi! (PPL: {val_ppl:.1f})")
                print()

    # ── 5. Son Değerlendirme ───────────────────────────────────────────────
    elapsed = time.time() - start_time
    final_ppl = compute_perplexity(model, val_loader, device)

    print(f"\n{'='*65}")
    print(f"  Eğitim Tamamlandı!")
    print(f"{'='*65}\n")
    print(f"  Süre: {elapsed:.0f}s ({elapsed/60:.1f} dakika)")
    print(f"  En iyi PPL: {best_val_ppl:.1f}")
    print(f"  Son PPL: {final_ppl:.1f}")

    # Son üretim örnekleri
    print(f"\n  ─── Son Türkçe Üretim Örnekleri ───\n")

    final_prompts = [
        "Türkiye Cumhuriyeti",
        "Bilim ve teknoloji",
        "İstanbul'un tarihi",
        "Eğitim alanında yapılan",
        "Araştırma sonuçları gösteriyor ki",
        "Bu çalışmanın amacı",
        "Osmanlı İmparatorluğu",
        "Dünya genelinde",
    ]

    for p in final_prompts:
        text = generate_sample(model, tokenizer, p, max_tokens=60, temperature=0.7, device=device)
        short = text[:220].replace('\n', ' ')
        print(f"  > \"{short}\"\n")

    # Model kaydet
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'step': step, 'model_state_dict': model.state_dict(),
        'config': config, 'val_ppl': final_ppl, 'time': elapsed,
    }, "checkpoints/swarm_turkish_final.pt")
    print(f"  Model kaydedildi: checkpoints/swarm_turkish_final.pt")

    print(f"\n  Bellek Karşılaştırması:")
    print(f"    Swarm-LLM: {vram['total_swarm']}")
    print(f"    Standart:  {vram['total_standard']}")
    print(f"    Kazanç:    {vram['attention_savings']}")
    print()


if __name__ == "__main__":
    main()
