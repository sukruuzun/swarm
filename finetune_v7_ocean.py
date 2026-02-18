#!/usr/bin/env python3
"""
v6 Model Fine-tuning: v7 Okyanus Veri Seti (DeepCoder Optimized)
==================================================================
Drive'dan v6 modelini yükleyip v7 Okyanus (kod + TR + mantık) ile fine-tune eder.

v7 Okyanus Dataset (DeepCoder için optimize):
  - Code (The Stack Smol): %60 - Python, JavaScript, C++ kodları
  - Turkish Wiki: %20 - Türkçe Wikipedia metinleri
  - TinyStories: %20 - Basit hikayeler (mantık/akıl yürütme)

Kritik İyileştirmeler:
  1. Tokenizer: Indentation-aware (4 boşluk, tab, newline özel tokenlar)
  2. Embedding: Yeni tokenlar için otomatik yeniden boyutlandırma
  3. Dataset Oranları: %60 Kod (DeepCoder hedefi)
"""

import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from swarm_llm.config import SwarmConfig
from swarm_llm.unified import UnifiedParisiNashLLM


class TokenDataset(IterableDataset):
    """Streaming token dataset."""
    def __init__(self, tokenizer, block_size=256, max_tokens=500_000_000):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_tokens = max_tokens
        self.buffer = []
        self.token_count = 0

    def __iter__(self):
        for tokens in self._tokenize_stream():
            self.buffer.extend(tokens)
            while len(self.buffer) >= self.block_size + 1:
                x = torch.tensor(self.buffer[:self.block_size], dtype=torch.long)
                y = torch.tensor(self.buffer[1:self.block_size+1], dtype=torch.long)
                self.buffer = self.buffer[self.block_size:]
                yield x, y
                self.token_count += self.block_size
                if self.token_count >= self.max_tokens:
                    return

    def _tokenize_stream(self):
        """Tokenize streaming data with optimized ratios for DeepCoder."""
        from datasets import load_dataset, interleave_datasets
        
        # ── DÜZELTME 3: the-stack-smol kullan (daha stabil) ──
        print("  The Stack Smol yükleniyor...")
        code_ds = load_dataset("bigcode/the-stack-smol", streaming=True, split="train")
        
        print("  Turkish Wiki yükleniyor...")
        tr_wiki = load_dataset("wikipedia", "20220301.tr", streaming=True, split="train")
        
        print("  TinyStories yükleniyor...")
        tiny_stories = load_dataset("roneneldan/TinyStories", streaming=True, split="train")
        
        # ── DÜZELTME 2: %60 Kod, %20 TR, %20 Mantık (DeepCoder hedefi) ──
        mixed = interleave_datasets(
            [code_ds, tr_wiki, tiny_stories], 
            probabilities=[0.60, 0.20, 0.20]  # Kod odaklı
        )
        
        for example in mixed:
            text = example.get("text", example.get("content", ""))
            if text.strip():
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > 0:
                    yield tokens


def collate_fn(batch):
    """Batch padding for variable-length sequences."""
    x_list, y_list = zip(*batch)
    max_len = max(len(x) for x in x_list)
    
    x_padded = []
    y_padded = []
    for x, y in batch:
        pad_len = max_len - len(x)
        x_padded.append(F.pad(x, (0, pad_len), value=0))
        y_padded.append(F.pad(y, (0, pad_len), value=-1))
    
    return torch.stack(x_padded), torch.stack(y_padded)


def resize_embeddings(model, new_vocab_size):
    """Embedding katmanını yeni vocab_size'a göre yeniden boyutlandır."""
    old_vocab_size = model.tok_emb.weight.shape[0]
    
    if new_vocab_size == old_vocab_size:
        return model
    
    print(f"  Embedding yeniden boyutlandırılıyor: {old_vocab_size} → {new_vocab_size}")
    
    embed_dim = model.tok_emb.weight.shape[1]
    
    # Yeni embedding oluştur
    new_emb = nn.Embedding(new_vocab_size, embed_dim)
    new_lm_head = nn.Linear(embed_dim, new_vocab_size, bias=False)
    
    # Eski ağırlıkları kopyala
    with torch.no_grad():
        new_emb.weight[:old_vocab_size] = model.tok_emb.weight
        new_lm_head.weight[:old_vocab_size] = model.lm_head.weight
        
        # Yeni tokenlar için rastgele başlat (küçük std)
        if new_vocab_size > old_vocab_size:
            nn.init.normal_(
                new_emb.weight[old_vocab_size:], 
                mean=0.0, 
                std=0.02
            )
            nn.init.normal_(
                new_lm_head.weight[old_vocab_size:], 
                mean=0.0, 
                std=0.02
            )
    
    # Değiştir
    model.tok_emb = new_emb
    model.lm_head = new_lm_head
    model.lm_head.weight = model.tok_emb.weight  # Weight tying
    
    # Config'i güncelle
    model.config.vocab_size = new_vocab_size
    
    return model


def load_checkpoint(checkpoint_path, device):
    """Load v6 checkpoint from Drive."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint bulunamadı: {checkpoint_path}")
    
    print(f"  Checkpoint yükleniyor: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    return ckpt


def create_model_from_config(config_dict, vocab_size):
    """Reconstruct model from config dict."""
    config = SwarmConfig(
        vocab_size=vocab_size,
        embed_dim=config_dict["embed_dim"],
        num_heads=config_dict["num_heads"],
        num_layers=config_dict["num_layers"],
        max_seq_len=config_dict["max_seq_len"],
        dropout=config_dict["dropout"],
        neighbor_size=config_dict["neighbor_size"],
        use_multi_scale=config_dict["use_multi_scale"],
        noise_strength=config_dict["noise_strength"],
        ffn_multiplier=config_dict["ffn_multiplier"],
        use_nash_moe=config_dict["use_nash_moe"],
        num_experts=config_dict["num_experts"],
        top_k_experts=config_dict["top_k_experts"],
    )
    return UnifiedParisiNashLLM(config)


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
def eval_ppl(model, loader, device, max_batches=50):
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


def get_batch_size_for_vram(device):
    """VRAM'e göre batch_size öner."""
    if not torch.cuda.is_available():
        return 4
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_gb >= 70:
        return 16
    if total_gb >= 35:
        return 8
    if total_gb >= 14:
        return 4
    return 2


def main():
    # ── Fine-tuning parametreleri ──
    steps = 5_000
    block_size = 256
    lr = 1e-4
    min_lr = 1e-5
    warmup = 500
    log_interval = 50
    eval_interval = 500
    checkpoint_interval = 1000
    
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
        batch_size = min(batch_size, 4)
    
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  v6 → v7 Fine-tuning: Okyanus Dataset (DeepCoder Optimized)     ║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("  v7 Okyanus Dataset (DeepCoder için optimize):")
    print("    ✓ Code (The Stack Smol): %60 - Python, JS, C++")
    print("    ✓ Turkish Wiki: %20 - Türkçe Wikipedia")
    print("    ✓ TinyStories: %20 - Basit hikayeler")
    print(f"    ✓ Streaming dataset (max 500M token)")
    print()
    print("  Kritik İyileştirmeler:")
    print("    ✓ Indentation-aware tokenizer (4 boşluk, tab, newline)")
    print("    ✓ Embedding otomatik yeniden boyutlandırma")
    print("    ✓ Kod odaklı dataset karışımı (%60)")
    print()
    print(f"  Fine-tuning: {steps} step, LR {lr}, warmup {warmup}")
    print(f"  Cihaz: {device} | batch_size: {batch_size}")
    print()
    
    # ── DÜZELTME 1: Tokenizer + Special Tokens ──
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Indentation-aware special tokenlar ekle
    special_tokens = {
        "additional_special_tokens": [
            "    ",  # 4 boşluk (Python standard)
            "  ",    # 2 boşluk (alternatif)
            "\t",    # Tab
            "\n",    # Newline (kod yapısı için kritik)
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    
    old_vocab_size = tokenizer.vocab_size - len(special_tokens["additional_special_tokens"])
    new_vocab_size = tokenizer.vocab_size
    
    print(f"  Tokenizer vocab_size: {old_vocab_size} → {new_vocab_size}")
    print(f"  Eklenen special tokenlar: {len(special_tokens['additional_special_tokens'])}")
    print()
    
    # ── v6 Checkpoint yükle ──
    checkpoint_path = "/content/drive/MyDrive/YES_Tools_Models/v6_final.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "v6_final.pt"
    
    ckpt = load_checkpoint(checkpoint_path, device)
    config_dict = ckpt["config"]
    best_ppl_v6 = ckpt.get("best_ppl", float('inf'))
    print(f"  v6 Best PPL: {best_ppl_v6:.1f}")
    print()
    
    # ── Model oluştur ve yükle ──
    # Önce eski vocab_size ile oluştur
    model = create_model_from_config(config_dict, config_dict["vocab_size"])
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    
    # ── Embedding'i yeniden boyutlandır (yeni tokenlar için) ──
    model = resize_embeddings(model, new_vocab_size)
    model.to(device)
    
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Model Parametreleri: {total_p:,} ({total_p/1e6:.1f}M)")
    print()
    
    # ── Temperature Annealing Schedule ──
    print("  Temperature Annealing Schedule (5K step):")
    for s in [0, 1000, 2500, 4000, 5000]:
        p = min(s / steps, 1.0)
        t = model.t_end + 0.5 * (model.t_start - model.t_end) * (1 + math.cos(math.pi * p))
        print(f"    Adım {s:>5d}: T = {t:.3f}")
    print()
    
    # ── Dataset ──
    print("  v7 Okyanus dataset yükleniyor (streaming)...")
    train_ds = TokenDataset(tokenizer, block_size, max_tokens=500_000_000)
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        collate_fn=collate_fn,
        num_workers=0,
    )
    print("  Dataset hazır!")
    print()
    
    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    
    # ── Decoupled Gradient Clipping ──
    router_params = model.get_router_params()
    expert_params = model.get_expert_params()
    
    def get_lr(step):
        if step < warmup:
            return lr * step / max(warmup, 1)
        p = (step - warmup) / max(steps - warmup, 1)
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))
    
    # ── Training Loop ──
    model.train()
    step, running_loss, running_ce, running_aux = 0, 0.0, 0.0, 0.0
    start = time.time()
    history = []
    best_ppl = float('inf')
    
    print("  Fine-tuning başlıyor...")
    print()
    
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
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            val_ppl = eval_ppl(model, train_loader, device, max_batches=20)
            if val_ppl < best_ppl:
                best_ppl = val_ppl
            
            print(f"\n  ── Adım {step} | Val PPL: {val_ppl:.1f} | En İyi: {best_ppl:.1f} ──")
            
            # Kod üretimi testi
            text = generate(model, tokenizer, "def hello():", max_tokens=30, device=device)
            print(f"    \"{text[:120]}\"")
            
            stats = model.get_nash_stats()
            for i in range(min(2, len(stats['regrets']))):
                r = stats['regrets'][i]
                u = stats['usages'][i] if stats['usages'][i] else [0] * 4
                r_str = [f"{v:+.2f}" for v in r]
                u_str = [f"{v:.0%}" for v in u]
                T = stats['temperatures'][i]
                print(f"    K{i}: T={T:.2f} U=[{', '.join(u_str)}] R=[{', '.join(r_str)}]")
            print()
        
        # ── Checkpoint kaydet ──
        if step % checkpoint_interval == 0 and step > 0:
            drive_dir = "/content/drive/MyDrive/YES_Tools_Models"
            if os.path.exists("/content/drive"):
                try:
                    os.makedirs(drive_dir, exist_ok=True)
                    save_path = os.path.join(drive_dir, f"v7_checkpoint_step_{step}.pt")
                    config_dict_save = {
                        "vocab_size": model.config.vocab_size,  # Güncellenmiş vocab_size
                        "embed_dim": config_dict["embed_dim"],
                        "num_heads": config_dict["num_heads"],
                        "num_layers": config_dict["num_layers"],
                        "max_seq_len": config_dict["max_seq_len"],
                        "dropout": config_dict["dropout"],
                        "neighbor_size": config_dict["neighbor_size"],
                        "use_multi_scale": config_dict["use_multi_scale"],
                        "noise_strength": config_dict["noise_strength"],
                        "ffn_multiplier": config_dict["ffn_multiplier"],
                        "use_nash_moe": config_dict["use_nash_moe"],
                        "num_experts": config_dict["num_experts"],
                        "top_k_experts": config_dict["top_k_experts"],
                    }
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "best_ppl": float(best_ppl),
                        "step": step,
                        "config": config_dict_save,
                    }, save_path)
                    print(f"  Checkpoint kaydedildi: {save_path}")
                except Exception as e:
                    print(f"  Checkpoint kayıt hatası: {e}")
    
    total_time = time.time() - start
    
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  SONUÇ: v7 Fine-tuning (DeepCoder Optimized)                      ║")
    print("╚" + "═" * 68 + "╝")
    print()
    print(f"  v6 Best PPL: {best_ppl_v6:.1f}")
    print(f"  v7 Best PPL: {best_ppl:.1f}")
    print(f"  Süre: {total_time/60:.0f} dk")
    print()
    
    # ── Final checkpoint kaydet ──
    drive_dir = "/content/drive/MyDrive/YES_Tools_Models"
    if os.path.exists("/content/drive"):
        try:
            os.makedirs(drive_dir, exist_ok=True)
            save_path = os.path.join(drive_dir, "v7_finetuned.pt")
            config_dict_save = {
                "vocab_size": model.config.vocab_size,
                "embed_dim": config_dict["embed_dim"],
                "num_heads": config_dict["num_heads"],
                "num_layers": config_dict["num_layers"],
                "max_seq_len": config_dict["max_seq_len"],
                "dropout": config_dict["dropout"],
                "neighbor_size": config_dict["neighbor_size"],
                "use_multi_scale": config_dict["use_multi_scale"],
                "noise_strength": config_dict["noise_strength"],
                "ffn_multiplier": config_dict["ffn_multiplier"],
                "use_nash_moe": config_dict["use_nash_moe"],
                "num_experts": config_dict["num_experts"],
                "top_k_experts": config_dict["top_k_experts"],
            }
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_ppl": float(best_ppl),
                "step": step,
                "config": config_dict_save,
            }, save_path)
            print(f"  Final checkpoint kaydedildi: {save_path}")
        except Exception as e:
            print(f"  Final checkpoint kayıt hatası: {e}")
    else:
        print("  Drive mount yok. Yerel: torch.save(..., 'v7_finetuned.pt')")
    print()


if __name__ == "__main__":
    main()
