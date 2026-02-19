# Swarm-LLM: Parisi-Nash Teoremi ile Dinamik Model Yönetimi

Parisi'nin sığırcık sürüsü modeli ve Nash oyun teorisini birleştirerek, eğitilmiş dev modelleri (Llama, Qwen) **sıfır eğitim maliyetiyle** dinamik olarak yöneten bir sistem.

## 🎯 Temel Özellikler

### 1. **Önceden Tahmin Mekanizması** (`get_predictive_indices`)
Modeli çalıştırmadan, sadece giriş cümlesine bakarak hangi blokların gerekli olduğunu tahmin eder.

```python
from swarm_llm.hf_loader import HuggingFaceBlockLoader

loader = HuggingFaceBlockLoader(model, tokenizer, num_blocks=8, top_k=2)

# Tahmin: hangi bloklar gerekli?
block_indices, weights = loader.predict_blocks("The history of science")
print(f"Yüklenecek bloklar: {block_indices}")  # [2, 5]
```

### 2. **Sıfır Eğitim Maliyeti**
Mevcut eğitilmiş bir modelin (Llama gibi) katmanlarını bloklara yerleştirdiğimizde, modeli yeniden eğitmeden teoreminle "yönetmeye" başlıyoruz.

### 3. **Dinamik RAM Yönetimi**
Eğer 10 bloktan sadece 2'sini yüklersen, 16 GB VRAM isteyen bir modeli ~3.2 GB VRAM ile çalıştırabilirsin.

**Örnek:** Llama-2-7B (32 layer) → 8 blok x 4 layer → Her forward'da sadece 2 blok → **~4x VRAM tasarrufu**

## 🚀 Hızlı Başlangıç

### Colab'da Çalıştır

```bash
# GitHub'dan klonla
git clone https://github.com/YOUR_USERNAME/swarm.git
cd swarm

# Colab notebook'u aç
# colab_demo_hf.ipynb
```

### Yerel Kurulum

```bash
pip install -r requirements.txt
```

### Kullanım Örneği

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from swarm_llm.hf_loader import HuggingFaceBlockLoader

# Model yükle
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Bloklara böl ve router ekle
loader = HuggingFaceBlockLoader(
    model=model,
    tokenizer=tokenizer,
    num_blocks=4,      # 12 layer → 4 blok x 3 layer
    top_k=2,           # Her forward'da sadece 2 blok
)

# Önceden tahmin
prompt = "The history of science is"
block_indices, weights = loader.predict_blocks(prompt)
print(f"Yüklenecek bloklar: {block_indices}")

# Forward: sadece seçilen bloklar çalışır
outputs = loader.generate(prompt, max_new_tokens=50)
print(outputs)
```

## 📊 VRAM Tasarrufu

```python
savings = loader.estimate_vram_savings()
print(f"Tasarruf oranı: {savings['savings_ratio']:.1f}x")
# Örnek: 70B model → 17.5B aktif (top_k=2, num_blocks=8)
```

## 🏗️ Mimari

### External Router (Dışsal Yönlendirici)
- **Parisi-Nash Router**: MLP gate + temperature annealing + load balancing
- **Önceden tahmin**: `get_predictive_indices()` - modeli çalıştırmadan blok seçimi
- **Dinamik seçim**: Her forward'da router hangi blokların çalışacağına karar verir

### Sparse Block Loader
- **Blok yönetimi**: Modeli N bloğa böler (örn. 8 blok)
- **Seyrek aktivasyon**: Her forward'da sadece top_k blok çalışır (örn. 2 blok)
- **Ağırlıklı birleştirme**: Router'ın verdiği ağırlıklarla blok çıktılarını birleştirir

## 📁 Proje Yapısı

```
swarm/
├── swarm_llm/
│   ├── external_router.py      # Parisi-Nash router (blok seçici)
│   ├── sparse_loader.py        # Sparse block loader (genel)
│   ├── hf_loader.py            # HuggingFace entegrasyonu
│   ├── unified.py              # Sıfırdan eğitim için unified model
│   └── ...
├── colab_demo_hf.ipynb         # Colab demo (HF entegrasyonu)
├── demo_external_router.py     # Yerel demo
└── requirements.txt
```

## 🔬 Teoreminin Açısından Önemi

### Tahmin Mekanizması
`get_predictive_indices` kısmı teoreminin beyni. Modelin tamamını çalıştırmadan, sadece giriş cümlesine bakarak "Benim 10 bloktan sadece 2. bloğa ihtiyacım var" diyor.

### Sıfır Eğitim Maliyeti
Mevcut eğitilmiş bir modelin (Llama gibi) katmanlarını bu bloklara yerleştirdiğimizde, modeli yeniden eğitmeden teoreminle "yönetmeye" başlıyoruz.

### Dinamik RAM Yönetimi
Eğer 10 bloktan sadece 2'sini yüklersen, 16 GB VRAM isteyen bir modeli ~3.2 GB VRAM ile çalıştırabilirsin.

## 🎓 Kullanım Senaryoları

1. **Evdeki laptopta Llama 70B çalıştırmak**
   - 70B → 8 blok → top_k=2 → ~17.5B aktif
   - 140GB VRAM → ~35GB VRAM

2. **Colab'da büyük modeller**
   - T4 (16GB) → Llama-2-7B → 4 blok → top_k=1 → ~1.75B aktif

3. **Edge cihazlar**
   - Küçük VRAM → Sadece gerekli blokları yükle

## 📝 Notlar

- **Desteklenen modeller**: Llama, Qwen, Mistral, GPT-2 (ve benzeri transformer mimarileri)
- **Router fine-tuning**: Router'ı ince ayar yaparak blok seçimini iyileştirebilirsin
- **Python 3.9+**: Tüm kod Python 3.9 uyumlu

## 🤝 Katkıda Bulunma

Pull request'ler memnuniyetle karşılanır! Büyük değişiklikler için önce bir issue açarak neyi değiştirmek istediğini tartışalım.

## 📄 Lisans

MIT License

## 🙏 Teşekkürler

- Parisi'nin sığırcık sürüsü modeli
- Nash oyun teorisi
- HuggingFace transformers
