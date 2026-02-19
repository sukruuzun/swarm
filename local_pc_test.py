"""
🖥️ LOKAL PC TESTİ: Drive'dan indirdiğin .pt dosyalarını kendi PC'nde çalıştır
===============================================================================
Kullanım:
    1. Drive'dan swarm_model_blocks/ klasörünü indir
    2. Bu script'in yanına koy:
       
       swarm/
       ├── local_pc_test.py          ← Bu dosya
       ├── swarm_llm/
       │   ├── hf_loader.py
       │   └── external_router.py
       └── model_blocks_qwen25_7b/   ← Drive'dan indirdiğin
           ├── block_0.pt
           ├── block_1.pt
           ├── ...
           ├── block_6.pt
           ├── router.pt
           └── rotary_emb.pt

    3. Çalıştır:
       python local_pc_test.py

    Gereksinimler:
       pip install torch transformers
"""

import os
import sys

# MPS (Apple Silicon) bellek limitini kaldır — torch'dan ONCE set edilmeli!
# Unified memory olduğu için güvenli — sadece 1 blok (~1.8 GB) bellekte
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch

# ── AYARLAR ──
SAVE_DIR = "model_blocks_qwen25_7b"  # .pt dosyalarının olduğu klasör
MODEL_NAME = "Qwen/Qwen2.5-7B"       # Tokenizer için (sadece config indirir, model DEĞİL)
PROMPT = "The history of artificial intelligence is"

print("="*60)
print("🖥️  LOKAL PC TESTİ — Sequential Lazy Loading")
print("="*60)

# Cihaz kontrolü
if torch.cuda.is_available():
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
    print(f"Apple Silicon (MPS) aktif")
else:
    device = "cpu"
    print(f"CPU modu (yavaş ama çalışır)")

# Dosya kontrolü
print(f"\n📂 Blok dizini: {SAVE_DIR}/")
if not os.path.exists(SAVE_DIR):
    print(f"❌ '{SAVE_DIR}' klasörü bulunamadı!")
    print(f"   Drive'dan indirip bu script'in yanına koy.")
    sys.exit(1)

pt_files = sorted([f for f in os.listdir(SAVE_DIR) if f.endswith('.pt')])
print(f"   Dosyalar: {len(pt_files)} adet")
for f in pt_files:
    size_mb = os.path.getsize(os.path.join(SAVE_DIR, f)) / (1024**2)
    print(f"   📄 {f}: {size_mb:.0f} MB")

# Tokenizer yükle (sadece config, model DEĞİL — çok küçük)
print(f"\n📥 Tokenizer yükleniyor: {MODEL_NAME}...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print(f"   Vocab: {tokenizer.vocab_size}")

# Sequential Lazy Loading
print(f"\n🔄 Sequential Lazy Loading başlatılıyor...")
print(f"   Her blok: diskten yükle → çalıştır → bellekten sil")
print(f"   VRAM: Sadece 1 blok (~1.8 GB) + router (~2 GB) = ~4 GB")

from swarm_llm.hf_loader import HuggingFaceBlockLoader

loader = HuggingFaceBlockLoader.from_disk_blocks(
    tokenizer=tokenizer,
    save_dir=SAVE_DIR,
    device=device,
    lazy_load=True,
    sequential_all=True,  # TÜM bloklar sırayla → tam kalite
)

# Metin üretimi
print(f"\n🧪 Metin üretimi başlıyor...")
print(f"   Prompt: '{PROMPT}'")

generated = loader.generate(
    prompt=PROMPT,
    max_new_tokens=10,  # Her token ~1 dk disk I/O, 10 token = ~10 dk
    temperature=0.8,
    top_k=40,
)

print(f"\n{'='*60}")
print(f"📝 ÜRETİLEN METİN:")
print(f"{'='*60}")
print(f"'{generated}'")
print(f"{'='*60}")

# Bellek durumu
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"\n💾 GPU VRAM: {allocated:.1f} GB kullanıldı (tam model ~14 GB olurdu)")

loader.stop_prefetching()

print(f"\n✅ TEST TAMAMLANDI!")
print(f"   Qwen 7B modeli {device.upper()}'de sequential lazy loading ile çalıştı.")
print(f"   Tam kalite, düşük bellek kullanımı.")
