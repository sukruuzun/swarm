# Qwen2.5-7B ile Hızlı Başlangıç

## 1. Model Yükleme (Accelerate ile)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from swarm_llm.hf_loader import HuggingFaceBlockLoader
import torch

# HuggingFace token (Colab Secrets'tan veya environment variable'dan)
import os
hf_token = os.environ.get('HF_TOKEN')  # Token'ını buraya ayarla veya Colab Secrets kullan

# Qwen2.5-7B yükle
model_name = "Qwen/Qwen2.5-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_token,
    torch_dtype=torch.float16,  # VRAM tasarrufu
    device_map="auto",          # Accelerate otomatik dağıtım
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

## 2. Bloklara Bölme

Qwen2.5-7B: **28 layer** → 7 blok x 4 layer (veya 8 blok x 3.5 layer → 8 blok)

```python
# Qwen2.5-7B için önerilen yapılandırma
loader = HuggingFaceBlockLoader(
    model=model,
    tokenizer=tokenizer,
    num_blocks=7,      # 28 layer → 7 blok x 4 layer
    top_k=2,           # Her forward'da sadece 2 blok
    device="auto",
)

print(f"✅ Qwen2.5-7B bloklara bölündü")
print(f"   Toplam layer: {len(loader.layers)}")
print(f"   Blok sayısı: {loader.num_blocks}")
print(f"   Her blok: {loader.layers_per_block} layer")
```

## 3. Önceden Tahmin

```python
prompt = "The history of artificial intelligence"
block_indices, weights = loader.predict_blocks(prompt, prefetch=True)

print(f"🔮 Tahmin: Bloklar {block_indices}")
print(f"   Ağırlıklar: {[f'{w:.2%}' for w in weights.tolist()]}")
```

## 4. Forward Pass

```python
input_ids = tokenizer.encode(prompt, return_tensors="pt")
# Accelerate ile dağıtılmış modellerde device otomatik handle edilir
outputs = loader.forward(input_ids)

print(f"✅ Çalıştırılan bloklar: {outputs['selected_indices']}")
```

## 5. Metin Üretimi (Asenkron Prefetching ile)

```python
# Prefetching'i başlat
loader.start_prefetching()

generated = loader.generate(
    prompt="The future of AI is",
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
    prefetch_next=True,  # Bir sonraki adımın bloklarını önceden yükle
)

print(generated)
```

## 6. Lazy Loading (Diskten Yükleme)

```python
# Modeli diske kaydet
save_dir = "model_blocks_qwen25_7b"
loader.save_blocks_to_disk(save_dir)

# Modeli RAM'den kaldır
del model
torch.cuda.empty_cache()

# Diskten lazy yükle
loader_lazy = HuggingFaceBlockLoader.from_disk_blocks(
    tokenizer=tokenizer,
    save_dir=save_dir,
    lazy_load=True,
)

# Prefetching'i başlat
loader_lazy.start_prefetching()

# Forward: Sadece seçilen bloklar diskten yüklenir
outputs = loader_lazy.forward(input_ids)
```

## Qwen2.5-7B Özellikleri

- **Parametre:** ~7B
- **Layer sayısı:** 28
- **Önerilen blok yapısı:** 7 blok x 4 layer
- **VRAM (tüm model):** ~14GB (float16)
- **VRAM (lazy, top_k=2):** ~4GB
- **Tasarruf:** ~3.5x

## Sorun Giderme

### Accelerate Uyarısı
```
WARNING:accelerate.big_modeling:You shouldn't move a model that is dispatched...
```
✅ **Normal:** `device_map="auto"` kullanıldığında bu uyarı görülebilir. Kod otomatik handle eder.

### Model Yapısı
Qwen modelleri genelde `model.model.layers` yapısında. Kod otomatik bulur:
- `model.model.layers` ✅
- `model.layers` ✅
- `model.transformer.h` ✅

### VRAM Yetersizse
```python
# Daha fazla blok, daha az top_k
loader = HuggingFaceBlockLoader(
    model=model,
    tokenizer=tokenizer,
    num_blocks=14,     # 28 layer → 14 blok x 2 layer
    top_k=1,           # Her forward'da sadece 1 blok
)
```
