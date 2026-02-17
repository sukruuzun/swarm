#!/usr/bin/env python3
"""
Swarm-LLM Demo & Test Scripti
================================
Sığırcık dikkat mekanizmasını gösteren kapsamlı demo.

Kullanım:
    python main.py                    # Tam demo
    python main.py --test-only        # Sadece testler
    python main.py --train-demo       # Mini eğitim demosu
    python main.py --benchmark        # Performans kıyaslaması
"""

import argparse
import time

import torch
import torch.nn.functional as F

from swarm_llm.config import SwarmConfig
from swarm_llm.model import SwarmLLM
from swarm_llm.attention import StarlingAttention, _build_sliding_window_mask
from swarm_llm.noise import ParisiNoise, AdaptiveParisiNoise
from swarm_llm.train import TextDataset, SwarmTrainer


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_sliding_window_mask():
    """Sliding window maskesinin doğruluğunu test eder."""
    print_header("1. Sliding Window Maske Testi")

    seq_len = 12
    window_size = 7

    # Causal maske (otoregresif -- sadece geçmişe bak)
    causal_mask = _build_sliding_window_mask(seq_len, window_size, torch.device('cpu'), causal=True)
    print("Causal Sliding Window Maskesi (True = maskelendi):")
    print(f"  Dizi uzunluğu: {seq_len}, Pencere: {window_size}\n")

    for i in range(seq_len):
        row = ""
        for j in range(seq_len):
            if causal_mask[i, j]:
                row += " ."  # Maskelendi (dikkat edilmiyor)
            else:
                row += " #"  # Aktif (dikkat ediliyor)
        visible = (~causal_mask[i]).sum().item()
        print(f"  Token {i:2d}: {row}   ({visible} komşu)")

    # Çift yönlü maske
    bidir_mask = _build_sliding_window_mask(seq_len, window_size, torch.device('cpu'), causal=False)
    print(f"\nÇift-Yönlü Sliding Window Maskesi:")
    for i in range(seq_len):
        row = ""
        for j in range(seq_len):
            if bidir_mask[i, j]:
                row += " ."
            else:
                row += " #"
        visible = (~bidir_mask[i]).sum().item()
        print(f"  Token {i:2d}: {row}   ({visible} komşu)")

    print("\n  # = Dikkat edilen pozisyon")
    print("  . = Maskelenen pozisyon (dikkat edilmiyor)")
    print("  [OK] Maske testi basarili!")


def test_starling_attention():
    """StarlingAttention'ın ileri geçişini test eder."""
    print_header("2. Sığırcık Dikkat (StarlingAttention) Testi")

    config = SwarmConfig(
        vocab_size=100,
        embed_dim=64,
        num_heads=4,
        neighbor_size=7,
        use_multi_scale=False,
    )
    attn = StarlingAttention(config, causal=True)

    # Test verisi
    batch_size, seq_len = 2, 20
    x = torch.randn(batch_size, seq_len, 64)

    # İleri geçiş
    out, kv = attn(x)

    print(f"  Giriş şekli:  {list(x.shape)}")
    print(f"  Çıktı şekli:  {list(out.shape)}")
    print(f"  KV önbellek K: {list(kv[0].shape)}")
    print(f"  KV önbellek V: {list(kv[1].shape)}")
    assert out.shape == x.shape, "Çıktı şekli uyuşmuyor!"
    print("  [OK] StarlingAttention testi basarili!")

    # Reynolds kuralları ağırlıkları
    print(f"\n  Reynolds Kuralları (öğrenilebilir):")
    print(f"    Ayrılma (separation): {attn.separation_gate.item():.3f}")
    print(f"    Hizalanma (alignment): {attn.alignment_gate.item():.3f}")
    print(f"    Uyum (cohesion):       {attn.cohesion_gate.item():.3f}")


def test_parisi_noise():
    """Parisi gürültü modüllerini test eder."""
    print_header("3. Parisi Gürültü (η) Testi")

    embed_dim = 64
    noise = ParisiNoise(embed_dim, noise_strength=0.05, learnable=True)
    adaptive_noise = AdaptiveParisiNoise(embed_dim, noise_strength=0.05)

    x = torch.randn(2, 10, embed_dim)

    # Eğitim modu
    noise.train()
    adaptive_noise.train()

    noisy_out = noise(x)
    adaptive_out = adaptive_noise(x)

    noise_diff = (noisy_out - x).abs().mean().item()
    adaptive_diff = (adaptive_out - x).abs().mean().item()

    print(f"  Standart gürültü farkı:    {noise_diff:.6f}")
    print(f"  Uyarlanabilir gürültü farkı: {adaptive_diff:.6f}")

    # Eval modu (gürültü kapalı olmalı)
    noise.eval()
    adaptive_noise.eval()

    eval_out = noise(x)
    eval_diff = (eval_out - x).abs().mean().item()
    print(f"  Eval modu farkı (0 olmalı): {eval_diff:.6f}")
    assert eval_diff == 0.0, "Eval modunda gürültü kapalı olmalı!"

    # Force noise (yaratıcılık modu)
    noise.force_noise = True
    force_out = noise(x)
    force_diff = (force_out - x).abs().mean().item()
    print(f"  Force-noise farkı:         {force_diff:.6f}")
    assert force_diff > 0, "Force-noise modunda gürültü olmalı!"

    print(f"\n  η parametreleri (ilk 8): {noise.eta.data[:8].tolist()}")
    print("  [OK] Parisi gürültü testi basarili!")


def test_full_model():
    """Tam SwarmLLM modelini test eder."""
    print_header("4. Tam Model (SwarmLLM) Testi")

    config = SwarmConfig(
        vocab_size=1000,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        max_seq_len=512,
        neighbor_size=7,
        use_multi_scale=True,
        noise_strength=0.02,
    )

    model = SwarmLLM(config)

    # Parametre sayıları
    params = model.count_parameters()
    print("  Model Parametreleri:")
    for key, val in params.items():
        print(f"    {key:>15s}: {val:>12,}")

    # VRAM tahmini
    print(f"\n  VRAM Tahmini (seq_len=2048, batch=4):")
    vram = model.estimate_vram(seq_len=2048, batch_size=4)
    for key, val in vram.items():
        print(f"    {key:>25s}: {val}")

    # İleri geçiş
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    targets = torch.randint(0, 1000, (batch_size, seq_len))

    outputs = model(input_ids, targets=targets)

    print(f"\n  İleri Geçiş:")
    print(f"    Giriş:   {list(input_ids.shape)}")
    print(f"    Logits:  {list(outputs['logits'].shape)}")
    print(f"    Kayıp:   {outputs['loss'].item():.4f}")
    print(f"    KV katman: {len(outputs['past_kvs'])}")

    assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)
    print("  [OK] Tam model testi basarili!")


def test_generation():
    """Metin üretimini test eder."""
    print_header("5. Metin Üretim Testi")

    config = SwarmConfig(
        vocab_size=100,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        max_seq_len=256,
        neighbor_size=7,
        use_multi_scale=False,
    )
    model = SwarmLLM(config)
    model.eval()

    # Prompt (başlangıç token'ları)
    prompt = torch.tensor([[1, 5, 10, 15, 20]])  # Batch=1, 5 token

    print(f"  Prompt token'ları: {prompt.tolist()[0]}")

    # Üretim
    with torch.no_grad():
        generated = model.generate(
            prompt,
            max_new_tokens=20,
            temperature=0.8,
            top_k=10,
        )

    print(f"  Üretilen token'lar: {generated.tolist()[0]}")
    print(f"  Yeni token sayısı: {generated.shape[1] - prompt.shape[1]}")
    assert generated.shape[1] == prompt.shape[1] + 20
    print("  [OK] Üretim testi basarili!")


def benchmark_attention():
    """Standart Attention vs Starling Attention performans kıyaslaması."""
    print_header("6. Performans Kıyaslaması")

    embed_dim = 256
    num_heads = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"  Cihaz: {device}")
    print(f"  Embed boyutu: {embed_dim}, Head sayısı: {num_heads}")
    print()

    seq_lengths = [256, 512, 1024, 2048]
    if device.type == 'cpu':
        seq_lengths = [256, 512, 1024]

    # --- Standart Self-Attention (O(N²)) ---
    class StandardAttention(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.qkv = torch.nn.Linear(d, 3 * d)
            self.proj = torch.nn.Linear(d, d)
            self.scale = (d // num_heads) ** 0.5
            self.num_heads = num_heads
            self.head_dim = d // num_heads

        def forward(self, x):
            B, L, D = x.shape
            qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(B, L, D)
            return self.proj(out)

    standard = StandardAttention(embed_dim).to(device)

    config = SwarmConfig(
        vocab_size=100,
        embed_dim=embed_dim,
        num_heads=num_heads,
        neighbor_size=7,
        use_multi_scale=False,
    )
    swarm = StarlingAttention(config, causal=True).to(device)

    print(f"  {'Dizi Uz.':<12} {'Standart (ms)':<18} {'Swarm (ms)':<18} {'Hızlanma':<12} {'Bellek Kazancı'}")
    print(f"  {'─'*72}")

    for seq_len in seq_lengths:
        x = torch.randn(1, seq_len, embed_dim, device=device)

        # Isınma
        with torch.no_grad():
            _ = standard(x)
            _ = swarm(x)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Standart zamanlama
        n_runs = 10
        start = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = standard(x)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
        std_time = (time.time() - start) / n_runs * 1000

        # Swarm zamanlama
        start = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = swarm(x)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
        swarm_time = (time.time() - start) / n_runs * 1000

        speedup = std_time / max(swarm_time, 0.001)
        mem_save = (seq_len * seq_len) / (seq_len * config.neighbor_size)

        print(
            f"  {seq_len:<12} {std_time:<18.2f} {swarm_time:<18.2f} "
            f"{speedup:<12.2f}x {mem_save:<.0f}x"
        )

    print(f"\n  Not: Bellek kazancı teorik (dikkat matrisi boyutu oranı).")
    print(f"  Gerçek hızlanma, maskeleme ek yükünden dolayı teorik kazançtan")
    print(f"  düşük olabilir. Sparse kernel kullanılarak iyileştirilebilir.")


def train_demo():
    """Mini eğitim demosu (rastgele veri ile)."""
    print_header("7. Mini Eğitim Demosu")

    config = SwarmConfig(
        vocab_size=256,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        max_seq_len=128,
        neighbor_size=7,
        use_multi_scale=False,
        noise_strength=0.01,
        learning_rate=1e-3,
        warmup_steps=50,
        max_steps=200,
    )

    model = SwarmLLM(config)

    # Rastgele "metin" verisi (demo amaçlı)
    total_tokens = 10_000
    token_ids = torch.randint(0, config.vocab_size, (total_tokens,))

    block_size = 64
    train_data = TextDataset(token_ids[:8000], block_size)
    val_data = TextDataset(token_ids[8000:], block_size)

    print(f"  Konfigürasyon:")
    print(f"    Vocab boyutu:   {config.vocab_size}")
    print(f"    Gömme boyutu:   {config.embed_dim}")
    print(f"    Katman sayısı:  {config.num_layers}")
    print(f"    Komşu boyutu:   {config.neighbor_size}")
    print(f"    Blok boyutu:    {block_size}")
    print(f"    Eğitim örnekleri: {len(train_data)}")
    print(f"    Doğrulama örnekleri: {len(val_data)}")
    print()

    trainer = SwarmTrainer(
        model=model,
        config=config,
        train_dataset=train_data,
        val_dataset=val_data,
        gradient_accumulation_steps=2,
        eval_interval=100,
        log_interval=20,
    )

    history = trainer.train()

    if history:
        print(f"\n  İlk kayıp:  {history[0]['loss']:.4f}")
        print(f"  Son kayıp:   {history[-1]['loss']:.4f}")
        improvement = (1 - history[-1]['loss'] / history[0]['loss']) * 100
        print(f"  İyileşme:    {improvement:.1f}%")


def print_architecture_summary():
    """Model mimarisi özetini yazdırır."""
    print_header("Swarm-LLM: Sığırcık Sürüsü Dil Modeli")

    config = SwarmConfig()
    model = SwarmLLM(config)

    print("  Mimari Genel Bakış:")
    print("  ┌─────────────────────────────────────┐")
    print("  │     Token Gömme (Embedding)          │")
    print("  │     + Döner Pozisyonel Kodlama (RoPE)│")
    print("  ├─────────────────────────────────────┤")
    for i in range(min(config.num_layers, 4)):
        attn_type = "Çoklu-Ölçekli" if config.use_multi_scale and i % 2 == 0 else "Standart"
        print(f"  │  Sığırcık Blok {i}                    │")
        print(f"  │    ├ LayerNorm                      │")
        print(f"  │    ├ {attn_type} Sığırcık Dikkat   │")
        print(f"  │    ├ Parisi Gürültü (η)             │")
        print(f"  │    ├ Residual Bağlantı              │")
        print(f"  │    ├ LayerNorm                      │")
        print(f"  │    ├ SwiGLU FFN                     │")
        print(f"  │    └ Residual Bağlantı              │")
        print(f"  ├─────────────────────────────────────┤")
    if config.num_layers > 4:
        print(f"  │  ... ({config.num_layers - 4} katman daha) ...          │")
        print(f"  ├─────────────────────────────────────┤")
    print("  │     Son LayerNorm                    │")
    print("  │     LM Head (weight tying)           │")
    print("  └─────────────────────────────────────┘")

    params = model.count_parameters()
    print(f"\n  Parametre Özeti:")
    for k, v in params.items():
        print(f"    {k:>15s}: {v:>12,}")

    print(f"\n  VRAM Karşılaştırması (seq=4096, batch=8):")
    vram = model.estimate_vram(seq_len=4096, batch_size=8)
    for k, v in vram.items():
        print(f"    {k:>25s}: {v}")

    print(f"\n  Sığırcık İlkeleri:")
    print(f"    Parisi '7 Komşu' kuralı  → neighbor_size = {config.neighbor_size}")
    print(f"    Reynolds ayrılma          → separation_weight = {config.separation_weight}")
    print(f"    Reynolds hizalanma        → alignment_weight = {config.alignment_weight}")
    print(f"    Reynolds uyum             → cohesion_weight = {config.cohesion_weight}")
    print(f"    Parisi gürültüsü (η)     → noise_strength = {config.noise_strength}")
    if config.use_multi_scale:
        print(f"    Çoklu ölçekli pencereler  → {config.multi_scale_windows}")


def main():
    parser = argparse.ArgumentParser(description="Swarm-LLM Demo")
    parser.add_argument('--test-only', action='store_true', help="Sadece testleri çalıştır")
    parser.add_argument('--train-demo', action='store_true', help="Mini eğitim demosu")
    parser.add_argument('--benchmark', action='store_true', help="Performans kıyaslaması")
    args = parser.parse_args()

    if args.test_only:
        test_sliding_window_mask()
        test_starling_attention()
        test_parisi_noise()
        test_full_model()
        test_generation()
        print_header("Tüm Testler Basarili!")
        return

    if args.benchmark:
        benchmark_attention()
        return

    if args.train_demo:
        train_demo()
        return

    # Tam demo
    print_architecture_summary()
    test_sliding_window_mask()
    test_starling_attention()
    test_parisi_noise()
    test_full_model()
    test_generation()
    benchmark_attention()
    train_demo()

    print_header("Demo Tamamlandi!")


if __name__ == "__main__":
    main()
