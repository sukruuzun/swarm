"""
HuggingFace Block Loader Test Suite
====================================
hf_loader.py'nin temel fonksiyonlarını test eder.

Kullanım:
    python test_hf_loader.py
    
    # Sadece belirli testleri çalıştır:
    python test_hf_loader.py --test basic
    python test_hf_loader.py --test save_load
    python test_hf_loader.py --test generate
"""

import torch
import torch.nn as nn
import argparse
import sys
import os

# swarm_llm'i import et
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swarm_llm.hf_loader import (
    HuggingFaceBlockLoader,
    QwenBlockWrapper,
    TupleCleaner,
    _StateDictModule,
)
from swarm_llm.external_router import ExternalParisiNashRouter


# ── Dummy Model (Test için) ────────────────────────────────────────────
class DummyLayer(nn.Module):
    """Basit transformer katman simülasyonu."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
    
    def forward(self, x, **kwargs):
        return (self.linear(self.norm(x)),)  # Tuple döndür (HF standardı)


class DummyModel(nn.Module):
    """HuggingFace model simülasyonu (Llama/Qwen yapısı)."""
    def __init__(self, vocab_size=100, embed_dim=64, num_layers=8):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.model.layers = nn.ModuleList([DummyLayer(embed_dim) for _ in range(num_layers)])
        self.model.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
    
    def parameters(self):
        params = []
        params.extend(self.model.embed_tokens.parameters())
        for layer in self.model.layers:
            params.extend(layer.parameters())
        params.extend(self.model.norm.parameters())
        params.extend(self.lm_head.parameters())
        return iter(params)


class DummyTokenizer:
    """Basit tokenizer simülasyonu."""
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.pad_token = None
        self.eos_token = "<eos>"
        self._vocab = {f"token_{i}": i for i in range(vocab_size)}
    
    def get_vocab(self):
        return self._vocab
    
    def encode(self, text, return_tensors=None, add_special_tokens=True):
        # Basit karakter-bazlı tokenizasyon
        tokens = [hash(word) % self.vocab_size for word in text.split()]
        if not tokens:
            tokens = [0]
        if return_tensors == "pt":
            return torch.tensor([tokens], dtype=torch.long)
        return tokens
    
    def decode(self, token_ids, skip_special_tokens=True):
        # Basit decode
        words = [f"w{tid}" for tid in token_ids]
        return " ".join(words)


# ── Test Fonksiyonları ──────────────────────────────────────────────────

def test_basic_initialization():
    """Test 1: Temel başlatma ve bloklara bölme."""
    print("\n" + "="*60)
    print("TEST 1: Temel Başlatma ve Bloklara Bölme")
    print("="*60)
    
    model = DummyModel(vocab_size=100, embed_dim=64, num_layers=8)
    tokenizer = DummyTokenizer(vocab_size=100)
    
    loader = HuggingFaceBlockLoader(
        model=model,
        tokenizer=tokenizer,
        num_blocks=4,
        top_k=2,
        device="cpu",
    )
    
    assert len(loader.blocks) == 4, f"Blok sayısı 4 olmalı, {len(loader.blocks)} bulundu"
    assert loader.top_k == 2, f"top_k 2 olmalı, {loader.top_k} bulundu"
    assert loader.layers_per_block == 2, f"layers_per_block 2 olmalı, {loader.layers_per_block} bulundu"
    assert loader.embed_dim == 64, f"embed_dim 64 olmalı, {loader.embed_dim} bulundu"
    
    print(f"  ✅ Blok sayısı: {len(loader.blocks)}")
    print(f"  ✅ top_k: {loader.top_k}")
    print(f"  ✅ layers_per_block: {loader.layers_per_block}")
    print(f"  ✅ embed_dim: {loader.embed_dim}")
    print(f"  ✅ is_qwen: {loader._is_qwen}")
    print("  ✅ TEST 1 BAŞARILI")
    return True


def test_no_sharding():
    """Test 2: No-sharding modu."""
    print("\n" + "="*60)
    print("TEST 2: No-Sharding Modu")
    print("="*60)
    
    model = DummyModel(vocab_size=100, embed_dim=64, num_layers=8)
    tokenizer = DummyTokenizer(vocab_size=100)
    
    loader = HuggingFaceBlockLoader(
        model=model,
        tokenizer=tokenizer,
        no_sharding=True,
        device="cpu",
    )
    
    assert len(loader.blocks) == 1, f"No-sharding'de 1 blok olmalı, {len(loader.blocks)} bulundu"
    assert loader.num_blocks == 1
    assert loader.top_k == 1
    
    print(f"  ✅ Blok sayısı: {len(loader.blocks)} (tek blok)")
    print(f"  ✅ Toplam katman: {len(loader.layers)}")
    print("  ✅ TEST 2 BAŞARILI")
    return True


def test_forward_pass():
    """Test 3: Forward pass testi."""
    print("\n" + "="*60)
    print("TEST 3: Forward Pass")
    print("="*60)
    
    model = DummyModel(vocab_size=100, embed_dim=64, num_layers=8)
    tokenizer = DummyTokenizer(vocab_size=100)
    
    loader = HuggingFaceBlockLoader(
        model=model,
        tokenizer=tokenizer,
        num_blocks=4,
        top_k=2,
        device="cpu",
    )
    
    input_ids = torch.tensor([[1, 5, 10, 20]], dtype=torch.long)
    outputs = loader.forward(input_ids)
    
    assert "logits" in outputs, "Çıktıda 'logits' olmalı"
    assert "hidden_states" in outputs, "Çıktıda 'hidden_states' olmalı"
    assert "selected_indices" in outputs, "Çıktıda 'selected_indices' olmalı"
    assert outputs["logits"] is not None, "Logits None olmamalı"
    assert outputs["logits"].shape == (1, 4, 100), f"Logits shape (1,4,100) olmalı, {outputs['logits'].shape} bulundu"
    
    print(f"  ✅ Logits shape: {outputs['logits'].shape}")
    print(f"  ✅ Hidden states shape: {outputs['hidden_states'].shape}")
    print(f"  ✅ Seçilen bloklar: {outputs['selected_indices']}")
    print(f"  ✅ Router weights: {outputs['router_weights']}")
    print("  ✅ TEST 3 BAŞARILI")
    return True


def test_generate():
    """Test 4: Metin üretimi testi."""
    print("\n" + "="*60)
    print("TEST 4: Metin Üretimi")
    print("="*60)
    
    model = DummyModel(vocab_size=100, embed_dim=64, num_layers=8)
    tokenizer = DummyTokenizer(vocab_size=100)
    
    loader = HuggingFaceBlockLoader(
        model=model,
        tokenizer=tokenizer,
        num_blocks=4,
        top_k=2,
        device="cpu",
        sticky_duration=10,
    )
    
    result = loader.generate(
        prompt="test input text",
        max_new_tokens=5,
        temperature=1.0,
        top_k=10,
    )
    
    assert isinstance(result, str), f"Sonuç string olmalı, {type(result)} bulundu"
    assert len(result) > 0, "Sonuç boş olmamalı"
    
    print(f"  ✅ Üretilen metin: '{result}'")
    print("  ✅ TEST 4 BAŞARILI")
    return True


def test_save_and_load():
    """Test 5: Diske kaydetme ve diskten yükleme."""
    print("\n" + "="*60)
    print("TEST 5: Diske Kaydetme ve Diskten Yükleme")
    print("="*60)
    
    import tempfile
    import shutil
    
    model = DummyModel(vocab_size=100, embed_dim=64, num_layers=8)
    tokenizer = DummyTokenizer(vocab_size=100)
    
    loader = HuggingFaceBlockLoader(
        model=model,
        tokenizer=tokenizer,
        num_blocks=4,
        top_k=2,
        device="cpu",
    )
    
    # Diske kaydet
    save_dir = tempfile.mkdtemp(prefix="swarm_test_")
    try:
        loader.save_blocks_to_disk(save_dir)
        
        # Dosyaların oluştuğunu kontrol et
        expected_files = ["block_0.pt", "block_1.pt", "block_2.pt", "block_3.pt", "router.pt"]
        for f in expected_files:
            path = os.path.join(save_dir, f)
            assert os.path.exists(path), f"{f} bulunamadı!"
            print(f"  ✅ {f} kaydedildi ({os.path.getsize(path) / 1024:.1f} KB)")
        
        # Diskten yükle (eager mode)
        loader_loaded = HuggingFaceBlockLoader.from_disk_blocks(
            tokenizer=tokenizer,
            save_dir=save_dir,
            device="cpu",
            lazy_load=False,
        )
        
        assert loader_loaded.num_blocks == 4, "Yüklenen blok sayısı 4 olmalı"
        assert loader_loaded.top_k == 2, "Yüklenen top_k 2 olmalı"
        assert loader_loaded.embed_dim == 64, "Yüklenen embed_dim 64 olmalı"
        
        print(f"  ✅ Diskten yükleme başarılı (eager mode)")
        print(f"     Blok sayısı: {loader_loaded.num_blocks}")
        print(f"     top_k: {loader_loaded.top_k}")
        
        # Diskten yükle (lazy mode)
        loader_lazy = HuggingFaceBlockLoader.from_disk_blocks(
            tokenizer=tokenizer,
            save_dir=save_dir,
            device="cpu",
            lazy_load=True,
        )
        
        assert loader_lazy._lazy_load == True
        assert len(loader_lazy._block_paths) == 4
        
        print(f"  ✅ Diskten yükleme başarılı (lazy mode)")
        print("  ✅ TEST 5 BAŞARILI")
        
    finally:
        shutil.rmtree(save_dir, ignore_errors=True)
    
    return True


def test_predict_blocks():
    """Test 6: Blok tahmin testi."""
    print("\n" + "="*60)
    print("TEST 6: Blok Tahmin (predict_blocks)")
    print("="*60)
    
    model = DummyModel(vocab_size=100, embed_dim=64, num_layers=8)
    tokenizer = DummyTokenizer(vocab_size=100)
    
    loader = HuggingFaceBlockLoader(
        model=model,
        tokenizer=tokenizer,
        num_blocks=4,
        top_k=2,
        device="cpu",
    )
    
    block_indices, weights = loader.predict_blocks("test prompt", prefetch=False)
    
    assert len(block_indices) == 2, f"2 blok tahmin edilmeli, {len(block_indices)} bulundu"
    assert all(0 <= idx < 4 for idx in block_indices), "Blok indeksleri 0-3 arası olmalı"
    
    print(f"  ✅ Tahmin edilen bloklar: {block_indices}")
    print(f"  ✅ Ağırlıklar: {weights.tolist()}")
    print("  ✅ TEST 6 BAŞARILI")
    return True


def test_vram_savings():
    """Test 7: VRAM tasarrufu hesaplama."""
    print("\n" + "="*60)
    print("TEST 7: VRAM Tasarrufu Hesaplama")
    print("="*60)
    
    model = DummyModel(vocab_size=100, embed_dim=64, num_layers=8)
    tokenizer = DummyTokenizer(vocab_size=100)
    
    loader = HuggingFaceBlockLoader(
        model=model,
        tokenizer=tokenizer,
        num_blocks=4,
        top_k=2,
        device="cpu",
    )
    
    savings = loader.estimate_vram_savings()
    
    assert "total_params" in savings
    assert "savings_ratio" in savings
    assert savings["savings_ratio"] > 1.0, "Tasarruf oranı 1'den büyük olmalı"
    
    print(f"  ✅ Toplam parametreler: {savings['total_params']:,}")
    print(f"  ✅ Tüm model VRAM: {savings['total_vram_gb']:.4f} GB")
    print(f"  ✅ Seyrek VRAM: {savings['sparse_vram_gb']:.4f} GB")
    print(f"  ✅ Tasarruf: {savings['savings_ratio']:.1f}x")
    print("  ✅ TEST 7 BAŞARILI")
    return True


def test_sticky_routing():
    """Test 8: Sticky routing testi."""
    print("\n" + "="*60)
    print("TEST 8: Sticky Routing")
    print("="*60)
    
    model = DummyModel(vocab_size=100, embed_dim=64, num_layers=8)
    tokenizer = DummyTokenizer(vocab_size=100)
    
    loader = HuggingFaceBlockLoader(
        model=model,
        tokenizer=tokenizer,
        num_blocks=4,
        top_k=2,
        device="cpu",
        sticky_duration=5,
    )
    
    # İlk forward: sticky blocks ayarlanmalı
    input_ids = torch.tensor([[1, 5, 10]], dtype=torch.long)
    outputs1 = loader.forward(input_ids, current_token_idx=0)
    
    # Sticky blocks'u ayarla
    loader._sticky_blocks = set(outputs1["selected_indices"])
    loader._sticky_until_token = 5
    
    # İkinci forward: aynı bloklar kullanılmalı (sticky)
    outputs2 = loader.forward(input_ids, current_token_idx=2)
    
    assert set(outputs2["selected_indices"]) == set(outputs1["selected_indices"]), \
        f"Sticky routing aktifken aynı bloklar kullanılmalı: {outputs1['selected_indices']} vs {outputs2['selected_indices']}"
    
    print(f"  ✅ İlk seçim: {outputs1['selected_indices']}")
    print(f"  ✅ Sticky seçim: {outputs2['selected_indices']} (aynı)")
    print("  ✅ TEST 8 BAŞARILI")
    return True


# ── Ana Test Çalıştırıcı ────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HuggingFace Block Loader Test Suite")
    parser.add_argument("--test", type=str, default="all", 
                       choices=["all", "basic", "no_sharding", "forward", "generate", 
                                "save_load", "predict", "vram", "sticky"],
                       help="Hangi testi çalıştıracağını seç")
    args = parser.parse_args()
    
    tests = {
        "basic": test_basic_initialization,
        "no_sharding": test_no_sharding,
        "forward": test_forward_pass,
        "generate": test_generate,
        "save_load": test_save_and_load,
        "predict": test_predict_blocks,
        "vram": test_vram_savings,
        "sticky": test_sticky_routing,
    }
    
    print("🧪 Swarm-LLM HuggingFace Block Loader Test Suite")
    print("=" * 60)
    
    passed = 0
    failed = 0
    errors = []
    
    if args.test == "all":
        test_list = tests.items()
    else:
        test_list = [(args.test, tests[args.test])]
    
    for name, test_fn in test_list:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
                errors.append(name)
        except Exception as e:
            failed += 1
            errors.append(f"{name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"📊 SONUÇ: {passed} başarılı, {failed} başarısız")
    if errors:
        print(f"❌ Başarısız testler: {errors}")
    else:
        print("✅ Tüm testler başarılı!")
    print("=" * 60)
    
    sys.exit(0 if failed == 0 else 1)
