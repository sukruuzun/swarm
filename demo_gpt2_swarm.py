#!/usr/bin/env python3
"""
GPT-2 + Swarm Dikkat Demosu (v2)
==================================
Gerçek bir dil modeli (GPT-2-124M) üzerinde Sığırcık Dikkat
mekanizmasının etkisini gösteren kapsamlı demo.

Önemli Bulgu:
  GPT-2 tam dikkatle (full attention) eğitildiği için, post-hoc
  olarak sliding window uygulamak kaliteyi düşürür. Ancak kısa
  bir fine-tuning ile model pencereli dikkate adapte olur.
  Mistral-7B gibi modeller sıfırdan sliding window ile eğitilir.

Kullanım:
    python demo_gpt2_swarm.py
    python demo_gpt2_swarm.py --prompt "The future of AI"
    python demo_gpt2_swarm.py --window-size 31
"""

import argparse
import copy
import math
import time
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Sığırcık Sliding Window Dikkat ──────────────────────────────────────────

def build_sliding_window_mask(seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
    """Causal sliding window maskesi (vektörize)."""
    positions = torch.arange(seq_len, device=device)
    distance = positions.unsqueeze(1) - positions.unsqueeze(0)
    mask = (distance < 0) | (distance >= window_size)
    return mask


class SwarmAttentionWrapper(nn.Module):
    """
    GPT-2'nin orijinal dikkat katmanını saran Sığırcık Dikkat.

    Orijinal Q/K/V projeksiyonlarını korur, sadece dikkat hesaplamasını
    sliding window ile sınırlar.
    """

    def __init__(self, original_attn, window_size: int = 7):
        super().__init__()
        self.original_attn = original_attn
        self.window_size = window_size

        self.c_attn = original_attn.c_attn
        self.c_proj = original_attn.c_proj
        self.attn_dropout = original_attn.attn_dropout
        self.resid_dropout = original_attn.resid_dropout
        self.n_head = original_attn.num_heads
        self.head_dim = original_attn.head_dim

        self._cached_mask = None
        self._cached_len = 0

    def _get_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self._cached_mask is not None and self._cached_len == seq_len:
            return self._cached_mask.to(device)
        mask = build_sliding_window_mask(seq_len, self.window_size, device)
        self._cached_mask = mask
        self._cached_len = seq_len
        return mask

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, L, D = hidden_states.shape

        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(D, dim=2)

        q = q.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        sw_mask = self._get_mask(L, hidden_states.device)
        scores = scores.masked_fill(sw_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.c_proj(out)
        out = self.resid_dropout(out)

        return (out, attn_weights)


# ── GPT-2 Patching ──────────────────────────────────────────────────────────

def patch_gpt2_with_swarm(model, window_size: int = 7):
    """GPT-2 dikkat katmanlarını Sığırcık Dikkat ile değiştirir."""
    patched = 0
    for block in model.transformer.h:
        block.attn = SwarmAttentionWrapper(block.attn, window_size)
        patched += 1
    return model, patched


# ── Yardımcılar ─────────────────────────────────────────────────────────────

@contextmanager
def timer(label: str):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"  {label}: {elapsed*1000:.1f} ms")


def count_active_attention(seq_len: int, window_size: int) -> dict:
    standard = seq_len * seq_len
    swarm = sum(min(i + 1, window_size) for i in range(seq_len))
    return {
        'standard': standard,
        'swarm': swarm,
        'ratio': standard / max(swarm, 1),
        'sparsity': 1 - swarm / standard,
    }


def print_header(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}\n")


def generate_text(model, tokenizer, prompt, max_tokens=60, temperature=0.7, device='cpu'):
    """Metin üretimi wrapper'ı."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True), output


def compute_perplexity(model, tokenizer, text, device='cpu'):
    """Perplexity hesapla."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(inputs['input_ids'], labels=inputs['input_ids'])
    return torch.exp(outputs.loss).item()


# ── Fine-Tuning ─────────────────────────────────────────────────────────────

def finetune_swarm_model(model, tokenizer, device, steps=150, lr=5e-5):
    """
    Sığırcık GPT-2'yi kısa fine-tuning ile adapte eder.

    Küçük bir metin örnekleminde sliding window'a uyum sağlaması için
    model ağırlıklarını günceller.
    """
    train_texts = [
        "The development of artificial intelligence has been one of the most significant technological advances of the 21st century.",
        "Machine learning algorithms can process vast amounts of data to identify patterns that would be impossible for humans to detect.",
        "Natural language processing enables computers to understand and generate human language with remarkable accuracy.",
        "Deep learning models use neural networks with many layers to learn complex representations of data.",
        "Reinforcement learning allows agents to learn optimal behaviors through trial and error in an environment.",
        "Computer vision systems can now identify objects in images with accuracy that rivals or exceeds human performance.",
        "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms.",
        "Transfer learning allows models trained on large datasets to be adapted for specific tasks with minimal additional training.",
        "Generative models can create realistic text, images, and audio that are increasingly difficult to distinguish from human-created content.",
        "The ethical implications of artificial intelligence include concerns about bias, privacy, and the impact on employment.",
        "Robotics combined with AI enables autonomous systems that can navigate complex environments and perform intricate tasks.",
        "Federated learning allows machine learning models to be trained across multiple devices without sharing raw data.",
        "Attention mechanisms allow neural networks to focus on the most relevant parts of the input when making predictions.",
        "Large language models demonstrate emergent capabilities that were not explicitly programmed into their training objectives.",
        "The field of AI safety focuses on ensuring that artificial intelligence systems behave in ways that are beneficial to humanity.",
        "Quantum computing may eventually accelerate certain machine learning algorithms by orders of magnitude.",
    ]

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print(f"  Fine-tuning: {steps} adım, lr={lr}")
    print(f"  Eğitim verisi: {len(train_texts)} metin parçası")

    losses = []
    for step in range(steps):
        text = train_texts[step % len(train_texts)]
        inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True).to(device)
        ids = inputs['input_ids']

        outputs = model(ids, labels=ids)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

        if (step + 1) % 50 == 0:
            avg_loss = sum(losses[-50:]) / len(losses[-50:])
            print(f"    Adım {step+1:>4d}/{steps} | Kayıp: {avg_loss:.4f}")

    model.eval()
    return losses


# ── Ana Demo ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GPT-2 + Swarm Attention Demo v2")
    parser.add_argument('--prompt', type=str, default="Artificial intelligence is",
                        help="Metin üretim promptu")
    parser.add_argument('--max-tokens', type=int, default=60,
                        help="Üretilecek maksimum token sayısı")
    parser.add_argument('--window-size', type=int, default=31,
                        help="Sığırcık pencere boyutu")
    parser.add_argument('--finetune-steps', type=int, default=150,
                        help="Fine-tuning adım sayısı")
    parser.add_argument('--temperature', type=float, default=0.7,
                        help="Üretim sıcaklığı")
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # ── 1. GPT-2 Yükleme ──────────────────────────────────────────────────
    print_header("GPT-2 + Sığırcık Dikkat Demo (v2)")

    print("  GPT-2 modeli yükleniyor...")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model_original = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model_original.eval()

    param_count = sum(p.numel() for p in model_original.parameters())
    print(f"  Model: GPT-2 ({param_count/1e6:.0f}M parametre)")
    print(f"  Katmanlar: {model_original.config.n_layer}, Head: {model_original.config.n_head}")
    print(f"  Embed: {model_original.config.n_embd}, Cihaz: {device}")

    # ── 2. Orijinal GPT-2 ─────────────────────────────────────────────────
    print_header("Adım 1: Orijinal GPT-2 (Tam Dikkat O(N²))")

    text_orig, _ = generate_text(
        model_original, tokenizer, args.prompt,
        max_tokens=args.max_tokens, temperature=args.temperature, device=device
    )
    print(f"  Prompt: \"{args.prompt}\"\n")
    print(f"  Üretilen metin:")
    print(f"  {'─'*55}")
    for line in text_orig.split('\n'):
        print(f"  {line}")
    print(f"  {'─'*55}")

    # ── 3. Sığırcık GPT-2 (Patch Only -- Fine-tune yok) ───────────────────
    print_header(f"Adım 2: Sadece Patch (w={args.window_size}, fine-tune YOK)")

    model_patched = copy.deepcopy(model_original)
    model_patched, n_patched = patch_gpt2_with_swarm(model_patched, args.window_size)
    model_patched.eval()
    print(f"  {n_patched} katman değiştirildi, pencere={args.window_size}\n")

    text_patched, _ = generate_text(
        model_patched, tokenizer, args.prompt,
        max_tokens=args.max_tokens, temperature=args.temperature, device=device
    )
    print(f"  Üretilen metin:")
    print(f"  {'─'*55}")
    for line in text_patched.split('\n'):
        print(f"  {line}")
    print(f"  {'─'*55}")

    print(f"\n  ⚠ GPT-2 tam dikkatle eğitildiği için, post-hoc sliding")
    print(f"    window uygulamak kaliteyi düşürür. Çözüm: fine-tuning!")

    # ── 4. Fine-Tuning ────────────────────────────────────────────────────
    print_header(f"Adım 3: Fine-Tuning ile Adaptasyon (w={args.window_size})")

    model_finetuned = copy.deepcopy(model_original)
    model_finetuned, _ = patch_gpt2_with_swarm(model_finetuned, args.window_size)

    losses = finetune_swarm_model(
        model_finetuned, tokenizer, device,
        steps=args.finetune_steps,
    )
    model_finetuned.eval()

    print(f"\n  Fine-tuning sonrası üretim:")
    text_ft, _ = generate_text(
        model_finetuned, tokenizer, args.prompt,
        max_tokens=args.max_tokens, temperature=args.temperature, device=device
    )
    print(f"  {'─'*55}")
    for line in text_ft.split('\n'):
        print(f"  {line}")
    print(f"  {'─'*55}")

    # ── 5. Perplexity Karşılaştırması ──────────────────────────────────────
    print_header("Adım 4: Perplexity (Şaşkınlık) Karşılaştırması")

    test_texts = [
        "The rapid advancement of technology has transformed the way we communicate and work in modern society.",
        "Scientists have discovered that regular exercise can significantly improve cognitive function and memory retention.",
        "The global economy faces unprecedented challenges as climate change continues to impact agricultural production worldwide.",
    ]

    print(f"  3 test cümlesi üzerinde ortalama perplexity:\n")

    for label, model_inst in [
        ("Orijinal GPT-2 (tam dikkat)", model_original),
        (f"Sadece Patch (w={args.window_size})", model_patched),
        (f"Fine-tuned  (w={args.window_size})", model_finetuned),
    ]:
        ppls = [compute_perplexity(model_inst, tokenizer, t, device) for t in test_texts]
        avg_ppl = sum(ppls) / len(ppls)
        print(f"  {label:<35s}: perplexity = {avg_ppl:>10.2f}")

    # ── 6. Farklı Pencere Boyutları (fine-tuned) ──────────────────────────
    print_header("Adım 5: Pencere Boyutu Etkisi (fine-tuned modeller)")

    windows = [7, 15, 31, 63]
    print(f"  Her pencere boyutu için {args.finetune_steps} adım fine-tuning yapılıyor...\n")

    for w in windows:
        model_w = copy.deepcopy(model_original)
        model_w, _ = patch_gpt2_with_swarm(model_w, w)
        finetune_swarm_model(model_w, tokenizer, device, steps=args.finetune_steps)
        model_w.eval()

        text_w, out_w = generate_text(
            model_w, tokenizer, args.prompt,
            max_tokens=args.max_tokens, temperature=args.temperature, device=device
        )

        ppls = [compute_perplexity(model_w, tokenizer, t, device) for t in test_texts]
        avg_ppl = sum(ppls) / len(ppls)
        stats = count_active_attention(out_w.shape[1], w)

        short = text_w[:140].replace('\n', ' ')
        if len(text_w) > 140:
            short += "..."

        print(f"  w={w:<4d} | ppl={avg_ppl:>8.1f} | seyreklik: {stats['sparsity']:.0%} | bellek kazancı: {stats['ratio']:.0f}x")
        print(f"         \"{short}\"\n")

        del model_w

    # ── 7. Bellek Analizi ──────────────────────────────────────────────────
    print_header("Adım 6: Bellek & Hesaplama Analizi")

    n_heads = model_original.config.n_head
    n_layers = model_original.config.n_layer
    w = args.window_size

    print(f"  GPT-2: {n_layers} katman, {n_heads} head, pencere={w}\n")

    def fmt(b):
        if b > 1e9: return f"{b/1e9:.2f} GB"
        if b > 1e6: return f"{b/1e6:.2f} MB"
        return f"{b/1e3:.1f} KB"

    print(f"  {'Dizi Uz.':<10} {'Standart':<14} {'Swarm':<14} {'Kazanç':<10} {'Seyreklik'}")
    print(f"  {'─'*58}")

    for sl in [128, 512, 1024, 2048, 4096, 8192]:
        stats = count_active_attention(sl, w)
        std_total = stats['standard'] * 4 * n_heads * n_layers
        swm_total = stats['swarm'] * 4 * n_heads * n_layers
        print(f"  {sl:<10} {fmt(std_total):<14} {fmt(swm_total):<14} "
              f"{stats['ratio']:<10.0f}x {stats['sparsity']:.1%}")

    # ── Özet ───────────────────────────────────────────────────────────────
    print_header("Sonuç ve Bulgular")

    print("  1. POST-HOC PATCHİNG vs SIFIRDAN EĞİTİM")
    print("     ────────────────────────────────────────")
    print("     GPT-2 tam dikkatle eğitildiğinden, sliding window'u")
    print("     doğrudan uygulamak tekrarlama sorununa yol açar.")
    print("     Ancak kısa fine-tuning ile model adapte olur.")
    print()
    print("  2. GERÇEK DÜNYA ÖRNEĞİ: MİSTRAL-7B")
    print("     ────────────────────────────────────────")
    print("     Mistral-7B sıfırdan w=4096 sliding window ile")
    print("     eğitilmiştir ve GPT-4 seviyesine yakın performans")
    print("     gösterirken çok daha az bellek kullanır.")
    print()
    print("  3. BELLEK KAZANCI")
    print("     ────────────────────────────────────────")
    print(f"     4096 token dizisinde w={w} ile dikkat matrisi")
    print(f"     {count_active_attention(4096, w)['ratio']:.0f}x küçülür.")
    print()
    print("  4. ÖNERİLEN PENCERE BOYUTLARI")
    print("     ────────────────────────────────────────")
    print("     Küçük model (<500M):  w = 7-31")
    print("     Orta model  (1-7B):   w = 64-512")
    print("     Büyük model (7B+):    w = 512-4096")
    print()
    print("  5. SONRAKI ADIMLAR")
    print("     ────────────────────────────────────────")
    print("     - Swarm-LLM'i sıfırdan eğitmek (main.py --train-demo)")
    print("     - Mistral-7B indirip sliding window'u görmek")
    print("     - Çoklu ölçekli pencere (7+21+63) denemek")
    print()


if __name__ == "__main__":
    main()
