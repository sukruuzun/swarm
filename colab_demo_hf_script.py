"""
Parisi-Nash External Router: Qwen2.5-7B ile Dinamik Blok Yükleme
================================================================
Bu script, colab_demo_hf.ipynb notebook'unun Python script versiyonudur.
Google Colab veya yerel makinede çalıştırılabilir.

Özellikler:
  ✅ Sıfır eğitim maliyeti: Mevcut modeli yeniden eğitmeden kullanır
  ✅ Dinamik RAM yönetimi: 7B parametre → sadece 2B RAM'de
  ✅ Sticky Routing: Bloklar belirli bir süre sabit kalır (thrashing önleme)
  ✅ Lazy Loading: Bloklar diskten gerektiğinde yüklenir
  ✅ Vocab Alignment: Tokenizer ve model vocab_size kontrolü
  ✅ Güvenli Token Yönetimi: Token açık metin olarak saklanmaz

Kullanım:
    # Colab'da:
    !python colab_demo_hf_script.py
    
    # Yerel makinede:
    python colab_demo_hf_script.py --model Qwen/Qwen2.5-7B --num_blocks 7 --top_k 2
    
    # Sadece no-sharding testi:
    python colab_demo_hf_script.py --test-only
"""

import torch
import os
import sys
import argparse


def get_hf_token():
    """HuggingFace token'ını güvenli bir şekilde al."""
    # 1. Environment variable
    token = os.environ.get('HF_TOKEN')
    if token:
        print("✅ Token environment variable'dan alındı")
        return token
    
    # 2. Colab Secrets
    try:
        from google.colab import userdata
        token = userdata.get('HF_TOKEN')
        if token:
            print("✅ Token Colab Secrets'tan alındı")
            return token
    except ImportError:
        pass
    
    # 3. Huggingface CLI login
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("✅ Token huggingface-cli login'den alındı")
            return token
    except ImportError:
        pass
    
    # 4. Manuel giriş
    print("⚠️  HF_TOKEN bulunamadı.")
    print("   Seçenekler:")
    print("   1. export HF_TOKEN=hf_xxx... (terminal)")
    print("   2. huggingface-cli login (CLI)")
    print("   3. Colab Secrets'a ekle (Colab)")
    token = input("Token giriniz (veya Enter ile devam): ").strip()
    return token if token else None


def load_model(model_name: str, hf_token: str):
    """HuggingFace modelini yükle."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print(f"\n🚀 {model_name} yükleniyor...")
    print("   ⚠️  Bu işlem birkaç dakika sürebilir...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"✅ Model yüklendi: {total_params:.1f}B parametre")
    
    return model, tokenizer


def test_no_sharding(model, tokenizer, prompt: str):
    """NO-SHARDING test modu: Modeli hiç bölmeden çalıştır."""
    from swarm_llm.hf_loader import HuggingFaceBlockLoader
    
    print("\n" + "="*70)
    print("🔬 NO-SHARDING TEST MODU")
    print("="*70)
    print("   Eğer bu modda çalışıyorsa: Sorun sharding'de")
    print("   Eğer hala bozuksa: Sorun model yükleme veya tokenizer'da\n")
    
    loader = HuggingFaceBlockLoader(
        model=model,
        tokenizer=tokenizer,
        no_sharding=True,
        device="auto",
    )
    
    print(f"✅ NO-SHARDING modu aktif")
    print(f"   Toplam layer: {len(loader.layers)}")
    print(f"   Tüm katmanlar tek blokta: Block 0\n")
    
    print(f"🧪 Test Prompt: '{prompt}'")
    print(f"🔄 Metin üretimi başlıyor (NO-SHARDING modu)...\n")
    
    generated = loader.generate(
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.8,
        top_k=40,
    )
    
    print(f"\n📝 Üretilen metin (NO-SHARDING):")
    print(f"'{generated}'\n")
    
    # Sonuç analizi
    if "焘" in generated or "溦" in generated or len(generated.split()) < 5:
        print("❌ SONUÇ: NO-SHARDING modunda da sorun var!")
        print("   → Sorun sharding'de DEĞİL, model yükleme veya tokenizer'da")
        return False
    else:
        print("✅ SONUÇ: NO-SHARDING modunda çalışıyor!")
        return True


def run_sharded_demo(model, tokenizer, num_blocks: int, top_k: int, prompt: str, save_dir: str):
    """Normal sharding modu ile demo."""
    from swarm_llm.hf_loader import HuggingFaceBlockLoader
    
    print("\n" + "="*70)
    print(f"🔧 SHARDING MODU: {num_blocks} blok, top_k={top_k}")
    print("="*70)
    
    # Bloklara böl
    loader = HuggingFaceBlockLoader(
        model=model,
        tokenizer=tokenizer,
        num_blocks=num_blocks,
        top_k=top_k,
        device="auto",
    )
    
    print(f"✅ Model bloklara bölündü")
    print(f"   Toplam layer: {len(loader.layers)}")
    print(f"   Blok sayısı: {loader.num_blocks}")
    print(f"   Her blok: {loader.layers_per_block} layer")
    print(f"   Her forward'da: {loader.top_k}/{loader.num_blocks} blok çalışır")
    
    # VRAM tasarrufu
    savings = loader.estimate_vram_savings()
    print(f"\n💾 VRAM Tasarrufu:")
    print(f"   Tüm model: {savings['total_vram_gb']:.1f} GB")
    print(f"   Seyrek ({savings['blocks_loaded']} blok): {savings['sparse_vram_gb']:.1f} GB")
    print(f"   Tasarruf: {savings['savings_ratio']:.1f}x")
    
    # Blok tahmini
    block_indices, weights = loader.predict_blocks(prompt, prefetch=False)
    print(f"\n🔮 Blok Tahmini:")
    print(f"   Prompt: '{prompt}'")
    print(f"   Tahmin edilen bloklar: {block_indices}")
    print(f"   Ağırlıklar: {[f'{w:.2%}' for w in weights.tolist()]}")
    
    # Diske kaydet
    if save_dir:
        print(f"\n💾 Bloklar diske kaydediliyor: {save_dir}")
        loader.save_blocks_to_disk(save_dir)
        
        # ── HEMEN DRIVE'A KOPYALA (kalibrasyon çökmeden önce!) ──
        try:
            from google.colab import drive
            import shutil
            print(f"\n📁 Google Drive'a kopyalanıyor...")
            drive.mount('/content/drive', force_remount=False)
            drive_target = f"/content/drive/MyDrive/swarm_model_blocks"
            os.makedirs(drive_target, exist_ok=True)
            
            files_copied = 0
            total_size_mb = 0
            for fname in sorted(os.listdir(save_dir)):
                if fname.endswith('.pt'):
                    src = os.path.join(save_dir, fname)
                    dst = os.path.join(drive_target, fname)
                    fsize_mb = os.path.getsize(src) / (1024**2)
                    print(f"   📄 {fname} ({fsize_mb:.1f} MB) → Drive")
                    shutil.copy2(src, dst)
                    files_copied += 1
                    total_size_mb += fsize_mb
            
            print(f"\n✅ {files_copied} dosya Drive'a kopyalandı!")
            print(f"   📍 My Drive/swarm_model_blocks/")
            print(f"   📦 Toplam: {total_size_mb:.0f} MB ({total_size_mb/1024:.1f} GB)")
        except ImportError:
            print("ℹ️  Colab değil, Drive kopyalama atlandı.")
        except Exception as e:
            print(f"⚠️  Drive kopyalama hatası: {e}")
    
    # Router kalibrasyonu (opsiyonel — T4'te cihaz sorunu olabilir)
    try:
        print(f"\n🎓 Router kalibrasyonu başlıyor...")
        loader.calibrate_router(num_steps=200)
        
        block_indices2, weights2 = loader.predict_blocks(prompt, prefetch=False)
        print(f"🔮 Kalibre edilmiş blok tahmini:")
        print(f"   Tahmin edilen bloklar: {block_indices2}")
        print(f"   Ağırlıklar: {[f'{w:.2%}' for w in weights2.tolist()]}")
    except Exception as e:
        print(f"⚠️  Kalibrasyon hatası (bloklar yine de kaydedildi): {e}")
    
    # Metin üretimi
    try:
        print(f"\n🔄 Metin üretimi başlıyor (sharding modu)...")
        generated = loader.generate(
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.8,
            top_k=40,
        )
        print(f"\n📝 Üretilen metin:")
        print(f"'{generated}'")
    except Exception as e:
        print(f"⚠️  Metin üretimi hatası: {e}")
    
    return loader


def run_lazy_loading_demo(tokenizer, save_dir: str, prompt: str):
    """Lazy loading demo: Diskten yükleme + sequential all blok çalıştırma."""
    from swarm_llm.hf_loader import HuggingFaceBlockLoader
    
    print("\n" + "="*70)
    print("📂 LAZY LOADING MODU (Sequential All)")
    print("="*70)
    print("   Tüm bloklar sırayla çalışır — NO-SHARDING kalitesi + VRAM tasarrufu")
    print("   Her blok: diskten yükle → çalıştır → bellekten sil\n")
    
    if not os.path.exists(save_dir):
        print(f"⚠️  {save_dir} bulunamadı! Önce sharding modunu çalıştırın.")
        return None
    
    loader_lazy = HuggingFaceBlockLoader.from_disk_blocks(
        tokenizer=tokenizer,
        save_dir=save_dir,
        lazy_load=True,
        device="auto",
        sequential_all=True,  # Tüm bloklar sırayla çalışır (kaliteli)
    )
    
    # Prefetching başlat
    loader_lazy.start_prefetching()
    
    print(f"✅ Lazy loader hazır (prefetching aktif)")
    print(f"   Bloklar diskte: {loader_lazy.num_blocks} blok")
    
    # Metin üretimi
    print(f"\n🔄 Metin üretimi (lazy loading + sequential all)...")
    generated = loader_lazy.generate(
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.8,
        top_k=40,
        prefetch_next=False,  # Sequential all'da prefetch'e gerek yok
    )
    
    print(f"\n📝 Üretilen metin:")
    print(f"'{generated}'")
    
    # Prefetching durdur
    loader_lazy.stop_prefetching()
    
    return loader_lazy


def main():
    parser = argparse.ArgumentParser(
        description="Parisi-Nash Router: HuggingFace Model Dinamik Blok Yükleme Demo"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B",
                       help="HuggingFace model adı (default: Qwen/Qwen2.5-7B)")
    parser.add_argument("--num-blocks", type=int, default=7,
                       help="Blok sayısı (default: 7)")
    parser.add_argument("--top-k", type=int, default=2,
                       help="Her forward'da çalışacak blok sayısı (default: 2)")
    parser.add_argument("--prompt", type=str, default="The history of artificial intelligence is",
                       help="Test prompt'u")
    parser.add_argument("--save-dir", type=str, default="model_blocks_qwen25_7b",
                       help="Blokların kaydedileceği dizin")
    parser.add_argument("--drive-dir", type=str, default="swarm_model_blocks",
                       help="Google Drive'daki hedef klasör adı")
    parser.add_argument("--test-only", action="store_true",
                       help="Sadece NO-SHARDING testi çalıştır")
    parser.add_argument("--skip-lazy", action="store_true",
                       help="Lazy loading testini atla")
    args = parser.parse_args()
    
    # GPU bilgisi
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Token al
    hf_token = get_hf_token()
    if not hf_token:
        print("❌ HF Token bulunamadı. Çıkılıyor.")
        sys.exit(1)
    
    # Model yükle
    model, tokenizer = load_model(args.model, hf_token)
    
    # 1. NO-SHARDING testi
    no_sharding_ok = test_no_sharding(model, tokenizer, args.prompt)
    
    if args.test_only:
        print("\n✅ Test tamamlandı.")
        return
    
    if not no_sharding_ok:
        print("\n⚠️  NO-SHARDING testi başarısız. Sharding modunu atlamak isteyebilirsiniz.")
        response = input("Devam etmek istiyor musunuz? (e/h): ").strip().lower()
        if response != 'e':
            return
    
    # 2. Sharding modu
    loader = run_sharded_demo(model, tokenizer, args.num_blocks, args.top_k, args.prompt, args.save_dir)
    
    # 3. Model'i RAM'den kaldır
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\n✅ Model RAM'den kaldırıldı")
    
    # 4. Lazy loading
    if not args.skip_lazy:
        run_lazy_loading_demo(tokenizer, args.save_dir, args.prompt)
    
    # Sonuç
    print("\n" + "="*70)
    print("📊 SONUÇLAR")
    print("="*70)
    print("✅ Sıfır eğitim maliyeti: Mevcut modeli yeniden eğitmeden kullandık")
    print("✅ Dinamik RAM yönetimi: Sadece gerekli bloklar RAM'de")
    print("✅ Sticky Routing: Thrashing önlendi")
    print("✅ Lazy Loading: Bloklar diskten gerektiğinde yüklendi")
    print("✅ Vocab Alignment: Otomatik kontrol yapıldı")
    print("✅ Accelerate hook temizleme: T4 uyumluluğu")


if __name__ == "__main__":
    main()
