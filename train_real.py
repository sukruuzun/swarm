#!/usr/bin/env python3
"""
Swarm-LLM Gerçek Veri ile Eğitim
====================================
WikiText-2 veri seti üzerinde küçük bir Swarm-LLM eğitimi.

Bu script:
1. WikiText-2 veri setini indirir (~2MB, küçük ve hızlı)
2. GPT-2 tokenizer ile tokenize eder
3. ~15M parametrelik bir Swarm-LLM eğitir
4. Her 200 adımda örnek metin üretir
5. Eğitim sonunda model ağırlıklarını kaydeder

Kullanım:
    python train_real.py
    python train_real.py --steps 3000 --embed-dim 384
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
    """Token dizisini sabit uzunluklu bloklara böler."""

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


def load_and_tokenize_wikitext():
    """WikiText-2 veri setini indir ve tokenize et."""
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    print("  WikiText-2 indiriliyor...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    print("  GPT-2 tokenizer yükleniyor...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def tokenize_split(split_name):
        texts = dataset[split_name]["text"]
        full_text = "\n".join([t for t in texts if t.strip()])
        tokens = tokenizer.encode(full_text)
        return torch.tensor(tokens, dtype=torch.long)

    print("  Tokenize ediliyor...")
    train_tokens = tokenize_split("train")
    val_tokens = tokenize_split("validation")
    test_tokens = tokenize_split("test")

    print(f"  Eğitim: {len(train_tokens):,} token")
    print(f"  Doğrulama: {len(val_tokens):,} token")
    print(f"  Test: {len(test_tokens):,} token")
    print(f"  Vocab boyutu: {tokenizer.vocab_size:,}")

    return train_tokens, val_tokens, test_tokens, tokenizer


# ── Metin Üretimi ────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_sample(model, tokenizer, prompt, max_tokens=80, temperature=0.8, device='cpu'):
    """Model ile metin üret."""
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated = input_ids

    for _ in range(max_tokens):
        # Son max_seq_len token'ı al
        context = generated[:, -model.config.max_seq_len:]
        outputs = model(context)
        logits = outputs['logits'][:, -1, :] / temperature

        # Top-K filtering
        top_k = 40
        top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < top_vals[:, -1:]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)

    model.train()
    return tokenizer.decode(generated[0].tolist())


# ── Perplexity Hesaplama ─────────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(model, dataloader, device, max_batches=50):
    """Doğrulama perplexity'si hesapla."""
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
    avg_loss = total_loss / max(total_batches, 1)
    return math.exp(avg_loss)


# ── Ana Eğitim ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Swarm-LLM Gerçek Eğitim")
    parser.add_argument('--steps', type=int, default=2000, help="Eğitim adım sayısı")
    parser.add_argument('--batch-size', type=int, default=8, help="Batch boyutu")
    parser.add_argument('--block-size', type=int, default=128, help="Blok (context) boyutu")
    parser.add_argument('--embed-dim', type=int, default=256, help="Gömme boyutu")
    parser.add_argument('--num-layers', type=int, default=6, help="Katman sayısı")
    parser.add_argument('--num-heads', type=int, default=8, help="Head sayısı")
    parser.add_argument('--neighbor-size', type=int, default=31, help="Pencere boyutu")
    parser.add_argument('--lr', type=float, default=6e-4, help="Öğrenme hızı")
    parser.add_argument('--log-interval', type=int, default=50, help="Loglama aralığı")
    parser.add_argument('--gen-interval', type=int, default=250, help="Üretim aralığı")
    parser.add_argument('--save-path', type=str, default="checkpoints", help="Kayıt dizini")
    args = parser.parse_args()

    device = torch.device(
        'mps' if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available()
        else 'cpu'
    )

    print(f"\n{'='*65}")
    print(f"  Swarm-LLM: Gerçek Veri ile Eğitim")
    print(f"  Cihaz: {device}")
    print(f"{'='*65}\n")

    # ── 1. Veri Yükleme ────────────────────────────────────────────────────
    print("─── Veri Seti Hazırlanıyor ───\n")
    train_tokens, val_tokens, test_tokens, tokenizer = load_and_tokenize_wikitext()

    train_dataset = TokenizedTextDataset(train_tokens, args.block_size)
    val_dataset = TokenizedTextDataset(val_tokens, args.block_size)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, pin_memory=(device.type != 'cpu'), num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, pin_memory=(device.type != 'cpu'), num_workers=0, drop_last=True,
    )

    print(f"  Eğitim örnekleri: {len(train_dataset):,}")
    print(f"  Doğrulama örnekleri: {len(val_dataset):,}")
    print(f"  Blok boyutu: {args.block_size}")
    print(f"  Batch boyutu: {args.batch_size}")

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
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Model boyutu: {total_params:,} parametre ({total_params/1e6:.1f}M)")
    print(f"  Eğitilebilir: {trainable_params:,}")
    print(f"  Embed boyutu: {args.embed_dim}")
    print(f"  Katmanlar: {args.num_layers}")
    print(f"  Head sayısı: {args.num_heads}")
    print(f"  Pencere (w): {args.neighbor_size}")

    vram = model.estimate_vram(args.block_size, args.batch_size)
    print(f"  Tahmini VRAM (Swarm): {vram['total_swarm']}")
    print(f"  Tahmini VRAM (Standart): {vram['total_standard']}")
    print(f"  Dikkat kazancı: {vram['attention_savings']}")

    # ── 3. Optimizer ───────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=0.01, betas=(0.9, 0.95),
    )

    # Cosine warmup scheduler
    warmup_steps = config.warmup_steps
    def get_lr(step):
        if step < warmup_steps:
            return args.lr * step / warmup_steps
        progress = (step - warmup_steps) / max(args.steps - warmup_steps, 1)
        return 1e-5 + 0.5 * (args.lr - 1e-5) * (1 + math.cos(math.pi * progress))

    # ── 4. Eğitim Döngüsü ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Eğitim Başlıyor! ({args.steps} adım)")
    print(f"{'='*65}\n")

    model.train()
    step = 0
    epoch = 0
    best_val_ppl = float('inf')
    running_loss = 0.0
    start_time = time.time()

    # Başlangıç üretimi
    print("  ─── Eğitim ÖNCE (rastgele ağırlıklar) ───")
    sample = generate_sample(model, tokenizer, "The meaning of life is", device=device)
    print(f"  \"{sample[:200]}\"")
    print()

    while step < args.steps:
        epoch += 1
        for x, y in train_loader:
            if step >= args.steps:
                break

            x, y = x.to(device), y.to(device)

            # Öğrenme hızını güncelle
            lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # İleri + geri geçiş
            outputs = model(x, targets=y)
            loss = outputs['loss']

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            step += 1

            # ── Loglama ────────────────────────────────────────────────
            if step % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                elapsed = time.time() - start_time
                tokens_sec = step * args.batch_size * args.block_size / elapsed

                print(
                    f"  Adım {step:>5d}/{args.steps} | "
                    f"Kayıp: {avg_loss:.4f} | "
                    f"PPL: {math.exp(avg_loss):.1f} | "
                    f"LR: {lr:.2e} | "
                    f"Token/s: {tokens_sec:,.0f} | "
                    f"Süre: {elapsed:.0f}s"
                )
                running_loss = 0.0

            # ── Örnek Üretim ───────────────────────────────────────────
            if step % args.gen_interval == 0:
                print(f"\n  ─── Adım {step} Üretim Örnekleri ───")

                prompts = [
                    "The history of science",
                    "In the early years",
                    "The city of",
                ]
                for p in prompts:
                    text = generate_sample(model, tokenizer, p, max_tokens=50, device=device)
                    short = text[:180].replace('\n', ' ')
                    print(f"  > \"{short}\"")

                # Doğrulama perplexity
                val_ppl = compute_perplexity(model, val_loader, device)
                print(f"\n  Doğrulama PPL: {val_ppl:.1f}")

                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    print(f"  Yeni en iyi! (önceki: {best_val_ppl:.1f})")

                    # Model kaydet
                    os.makedirs(args.save_path, exist_ok=True)
                    save_file = os.path.join(args.save_path, "swarm_best.pt")
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'config': config,
                        'val_ppl': val_ppl,
                    }, save_file)
                    print(f"  Model kaydedildi: {save_file}")

                print()

    # ── 5. Son Değerlendirme ───────────────────────────────────────────────
    elapsed = time.time() - start_time

    print(f"\n{'='*65}")
    print(f"  Eğitim Tamamlandı!")
    print(f"{'='*65}\n")

    print(f"  Toplam süre: {elapsed:.0f} saniye ({elapsed/60:.1f} dakika)")
    print(f"  Toplam adım: {step}")
    print(f"  En iyi doğrulama PPL: {best_val_ppl:.1f}")

    # Son perplexity
    final_val_ppl = compute_perplexity(model, val_loader, device)
    print(f"  Son doğrulama PPL: {final_val_ppl:.1f}")

    # VRAM karşılaştırması
    print(f"\n  Bellek Karşılaştırması (seq={args.block_size}, batch={args.batch_size}):")
    print(f"    Swarm-LLM:  {vram['total_swarm']}")
    print(f"    Standart:   {vram['total_standard']}")
    print(f"    Kazanç:     {vram['attention_savings']}")

    # Son üretim örnekleri
    print(f"\n  ─── Son Üretim Örnekleri (Eğitim SONRASI) ───\n")
    final_prompts = [
        "The history of",
        "In 1920",
        "Scientists discovered that",
        "The population of the city",
        "During the war",
    ]
    for p in final_prompts:
        text = generate_sample(model, tokenizer, p, max_tokens=60, temperature=0.7, device=device)
        short = text[:200].replace('\n', ' ')
        print(f"  > \"{short}\"\n")

    # Son model kaydet
    os.makedirs(args.save_path, exist_ok=True)
    save_file = os.path.join(args.save_path, "swarm_final.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'config': config,
        'val_ppl': final_val_ppl,
        'total_time': elapsed,
    }, save_file)
    print(f"  Son model kaydedildi: {save_file}")
    print()


if __name__ == "__main__":
    main()
