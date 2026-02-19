"""
🎯 TEK AMAÇ: Qwen 7B'yi bloklara böl, Drive'a kaydet, lazy loading test et.
================================================================
Colab notebook'ta çalıştır (script değil, hücrelere yapıştır):

HÜCRE 1: Kurulum
HÜCRE 2: Model yükle + bloklara böl + kaydet
HÜCRE 3: Drive'a kopyala
HÜCRE 4: Sequential Lazy Loading testi
"""

# ═══════════════════════════════════════════════════════════════
# HÜCRE 1: KURULUM (Bu hücreyi ilk çalıştır)
# ═══════════════════════════════════════════════════════════════
print("📦 Kurulum başlıyor...")

import subprocess, sys, os

# Gerekli paketler
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                       "torch", "transformers", "accelerate", "huggingface_hub"])

# Repo'yu klonla (yoksa)
if not os.path.exists("/content/swarm"):
    subprocess.check_call(["git", "clone", "https://github.com/sukruuzun/swarm.git", "/content/swarm"])
else:
    subprocess.check_call(["git", "-C", "/content/swarm", "pull"])

sys.path.insert(0, "/content/swarm")

print("✅ Kurulum tamamlandı!")

# ═══════════════════════════════════════════════════════════════
# HÜCRE 2: MODEL YÜKLE + BLOKLARA BÖL + KAYDET
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("🚀 MODEL YÜKLEME + SHARDING")
print("="*60)

import torch

# GPU kontrolü
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# HF Token
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from google.colab import userdata
        hf_token = userdata.get('HF_TOKEN')
    except:
        pass
if not hf_token:
    hf_token = input("HF Token girin: ").strip()
print(f"✅ Token {'bulundu' if hf_token else 'BULUNAMADI!'}")

# Model yükle
MODEL_NAME = "Qwen/Qwen2.5-7B"
SAVE_DIR = "model_blocks_qwen25_7b"
NUM_BLOCKS = 7

print(f"\n🔄 {MODEL_NAME} yükleniyor...")
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    token=hf_token, 
    trust_remote_code=True,
    torch_dtype=torch.float16, 
    device_map="auto"
)
print(f"✅ Model yüklendi: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B parametre")

# Bloklara böl ve kaydet
print(f"\n📦 {NUM_BLOCKS} bloğa bölünüyor...")
from swarm_llm.hf_loader import HuggingFaceBlockLoader

loader = HuggingFaceBlockLoader(
    model=model,
    tokenizer=tokenizer,
    num_blocks=NUM_BLOCKS,
    top_k=2,
    no_sharding=False,
)

print(f"\n💾 Diske kaydediliyor: {SAVE_DIR}/")
loader.save_blocks_to_disk(SAVE_DIR)

# Dosya listesi
print(f"\n📂 Kaydedilen dosyalar:")
total_mb = 0
for f in sorted(os.listdir(SAVE_DIR)):
    if f.endswith('.pt'):
        size_mb = os.path.getsize(os.path.join(SAVE_DIR, f)) / (1024**2)
        total_mb += size_mb
        print(f"   {f}: {size_mb:.0f} MB")
print(f"   TOPLAM: {total_mb:.0f} MB ({total_mb/1024:.1f} GB)")

print("\n✅ BLOKLAMA TAMAMLANDI!")

# ═══════════════════════════════════════════════════════════════
# HÜCRE 3: GOOGLE DRIVE'A KOPYALA
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("📁 GOOGLE DRIVE'A KOPYALAMA")
print("="*60)

from google.colab import drive
drive.mount('/content/drive')

import shutil

DRIVE_TARGET = "/content/drive/MyDrive/swarm_model_blocks"
os.makedirs(DRIVE_TARGET, exist_ok=True)

files_copied = 0
for fname in sorted(os.listdir(SAVE_DIR)):
    if fname.endswith('.pt'):
        src = os.path.join(SAVE_DIR, fname)
        dst = os.path.join(DRIVE_TARGET, fname)
        size_mb = os.path.getsize(src) / (1024**2)
        print(f"   📄 {fname} ({size_mb:.0f} MB) → Drive...", end=" ", flush=True)
        shutil.copy2(src, dst)
        print("✅")
        files_copied += 1

print(f"\n✅ {files_copied} dosya Google Drive'a kopyalandı!")
print(f"📍 Konum: My Drive/swarm_model_blocks/")
print(f"\n💡 PC'ye İndirme:")
print(f"   1. drive.google.com → 'swarm_model_blocks' klasörü")
print(f"   2. Tüm .pt dosyalarını indir")
print(f"   3. Toplam ~14.5 GB")

# ═══════════════════════════════════════════════════════════════
# HÜCRE 4: SEQUENTIAL LAZY LOADING TESTİ (Opsiyonel)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("🧪 SEQUENTIAL LAZY LOADING TESTİ")
print("="*60)
print("   Tüm bloklar sırayla çalışır — tam kalite + düşük VRAM")

# Model'i bellekten kaldır
del model
del loader
torch.cuda.empty_cache()
print("🗑️  Orijinal model bellekten silindi")

# Diskten lazy load
from swarm_llm.hf_loader import HuggingFaceBlockLoader as Loader

lazy_loader = Loader.from_disk_blocks(
    tokenizer=tokenizer,
    save_dir=SAVE_DIR,
    device="auto",
    lazy_load=True,
    sequential_all=True,  # Tüm bloklar sırayla → tam kalite
)

print(f"\n🔄 Metin üretimi (sequential lazy)...")
prompt = "The history of artificial intelligence is"
generated = lazy_loader.generate(
    prompt=prompt,
    max_new_tokens=80,
    temperature=0.8,
    top_k=40,
)

print(f"\n📝 Prompt: '{prompt}'")
print(f"📝 Üretilen: '{generated}'")

# VRAM kontrolü
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"\n💾 VRAM kullanımı: {allocated:.1f} GB (tam model: ~14 GB)")

lazy_loader.stop_prefetching()
print("\n✅ TEST TAMAMLANDI!")
print("\n🎉 Artık .pt dosyalarını Drive'dan indir ve kendi Mac'inde çalıştır!")
