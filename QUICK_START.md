# Hızlı Başlangıç: Kendi Notebook'unu Yaz

## Temel Kullanım Örnekleri

### 1. Model Yükleme ve Bloklara Bölme

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from swarm_llm.hf_loader import HuggingFaceBlockLoader

# Model yükle
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Bloklara böl
loader = HuggingFaceBlockLoader(
    model=model,
    tokenizer=tokenizer,
    num_blocks=4,      # Kaç bloğa böleceğiz?
    top_k=2,           # Her forward'da kaç blok çalışacak?
    device="auto",
)
```

### 2. Önceden Tahmin (Teoreminin Beyni)

```python
# Giriş cümlesine bakarak hangi bloklar gerekli?
prompt = "The history of science is"

block_indices, weights = loader.predict_blocks(prompt)

print(f"Yüklenecek bloklar: {block_indices}")
print(f"Ağırlıklar: {weights.tolist()}")
```

### 3. Forward Pass (Sadece Seçilen Bloklar)

```python
# Forward: Router hangi blokları seçti?
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(loader.device)
outputs = loader.forward(input_ids)

print(f"Çalıştırılan bloklar: {outputs['selected_indices']}")
print(f"Router ağırlıkları: {outputs['router_weights']}")
```

### 4. Metin Üretimi

```python
# Dinamik blok seçimi ile üretim
generated = loader.generate(
    prompt="The history of science is",
    max_new_tokens=50,
    temperature=0.8,
    top_k=40,
)
print(generated)
```

### 5. VRAM Tasarrufu Analizi

```python
savings = loader.estimate_vram_savings()

print(f"Toplam VRAM: {savings['total_vram_gb']:.2f} GB")
print(f"Seyrek VRAM: {savings['sparse_vram_gb']:.3f} GB")
print(f"Tasarruf: {savings['savings_ratio']:.1f}x")
```

## Router'ı Doğrudan Kullanma

```python
from swarm_llm.external_router import ExternalParisiNashRouter
import torch

# Router oluştur
router = ExternalParisiNashRouter(
    embed_dim=768,
    num_blocks=8,
    top_k=2,
)

# Embedding'e bakarak tahmin
x = torch.randn(1, 10, 768)  # (batch, seq_len, embed_dim)
block_indices, weights = router.get_predictive_indices(x)

print(f"Seçilen bloklar: {block_indices}")
print(f"Ağırlıklar: {weights.tolist()}")
```

## Farklı Modeller İçin

### Llama-2
```python
model_name = "meta-llama/Llama-2-7b-hf"
# 32 layer → 8 blok x 4 layer
loader = HuggingFaceBlockLoader(model, tokenizer, num_blocks=8, top_k=2)
```

### Qwen
```python
model_name = "Qwen/Qwen-7B-Chat"
# Benzer yapı
loader = HuggingFaceBlockLoader(model, tokenizer, num_blocks=8, top_k=2)
```

### GPT-2 (Test için)
```python
model_name = "gpt2"
# 12 layer → 4 blok x 3 layer
loader = HuggingFaceBlockLoader(model, tokenizer, num_blocks=4, top_k=2)
```

## İpuçları

1. **Colab'da:** Küçük modellerle başla (GPT-2), sonra Llama'ya geç
2. **VRAM:** `top_k` değerini düşürerek daha az RAM kullan
3. **Blok sayısı:** `num_blocks` = toplam layer sayısı / layers_per_block
4. **Router fine-tuning:** Router'ı ince ayar yaparak blok seçimini iyileştirebilirsin
