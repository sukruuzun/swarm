"""
HuggingFace Model Loader (HF Entegrasyonu)
==========================================
Llama, Qwen gibi eğitilmiş dev modelleri bloklara bölüp,
Parisi-Nash router ile dinamik yükleme yapan wrapper.

Kullanım:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from swarm_llm.hf_loader import HuggingFaceBlockLoader

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

    loader = HuggingFaceBlockLoader(
        model=model,
        tokenizer=tokenizer,
        num_blocks=8,  # 32 layer → 8 blok x 4 layer
        top_k=2,        # Her forward'da sadece 2 blok
    )

    # Tahmin: hangi bloklar gerekli?
    prompt = "The history of science"
    block_indices, weights = loader.predict_blocks(prompt)
    print(f"Yüklenecek bloklar: {block_indices}")

    # Forward: sadece seçilen bloklar çalışır
    outputs = loader.generate(prompt, max_new_tokens=50)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import threading
from queue import Queue

from swarm_llm.external_router import ExternalParisiNashRouter


class QwenBlockWrapper(nn.Module):
    """
    Qwen2 için özel block wrapper.
    Sequential içindeki katmanlara position_embeddings geçirmek için.
    """
    def __init__(self, layers: list):
        super().__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x, position_embeddings=None, position_ids=None, attention_mask=None):
        """
        Qwen2 katmanlarını sırayla çalıştır, position_embeddings'i geçir.
        """
        for layer in self.layers:
            if position_embeddings is not None:
                x = layer(x, position_embeddings=position_embeddings, 
                         position_ids=position_ids, 
                         attention_mask=attention_mask)
            else:
                x = layer(x)
        return x


class TupleCleaner(nn.Module):
    """
    HuggingFace layer çıktılarını temizleyen wrapper.
    GPT-2/Llama/Qwen layer'ları tuple döndürür (hidden_states, past_key_values, ...),
    bu wrapper sadece hidden_states'i alır ve bir sonraki layer'a geçirir.
    
    KRİTİK: Qwen2 gibi modern modeller position_embeddings bekler (RoPE için).
    Bu parametreler katmanlara geçirilmelidir.
    """
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer
    
    def forward(self, x, position_embeddings=None, attention_mask=None, position_ids=None, **kwargs):
        """
        HuggingFace layer çıktısını temizle.
        
        KRİTİK NOT: 
        - Bu wrapper past_key_values (KV Cache) verisini kaybediyor.
        - Qwen2 gibi modeller position_embeddings bekler (RoPE için).
        - Bu parametreler katmana geçirilir ama çıktıda sadece hidden_states döndürülür.
        
        Args:
            x: hidden_states (Tensor)
            position_embeddings: RoPE için position embeddings (Qwen2 için gerekli)
            attention_mask: Attention mask
            position_ids: Position IDs
            **kwargs: Diğer parametreler (use_cache, vb.)
        """
        # Qwen2 katmanları position_embeddings bekler
        # Eğer verilmişse geçir, yoksa sadece x'i geçir (eski modeller için)
        if position_embeddings is not None:
            out = self.layer(
                x,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,  # KV Cache kullanmıyoruz
                **kwargs
            )
        else:
            # Eski modeller için (GPT-2, Llama-1, vb.)
            out = self.layer(x, **kwargs)
        
        # Tuple ise sadece hidden_states'i al (past_key_values kaybolur)
        if isinstance(out, tuple):
            return out[0]
        return out


class HuggingFaceBlockLoader(nn.Module):
    """
    HuggingFace modelini bloklara bölen ve Parisi-Nash router ile
    dinamik yükleme yapan wrapper. Sıfır eğitim maliyeti: mevcut
    modelin ağırlıklarını değiştirmez, sadece hangi blokların
    çalıştırılacağına karar verir.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        num_blocks: int = 8,
        top_k: int = 2,
        layers_per_block: Optional[int] = None,
        embed_dim: Optional[int] = None,
        device: str = "auto",
        sticky_duration: int = 25,  # Sticky routing: kaç token boyunca bloklar sabit kalır
        no_sharding: bool = False,  # Test modu: Modeli hiç bölmeden tek blok olarak çalıştır
    ):
        """
        Args:
            model: HuggingFace AutoModelForCausalLM (Llama, Qwen, vb.)
            tokenizer: HuggingFace tokenizer
            num_blocks: Modeli kaç bloğa böleceğiz (örn. 8)
            top_k: Her forward'da kaç blok çalışacak (örn. 2)
            layers_per_block: Her blokta kaç layer (None ise otomatik hesapla)
            embed_dim: Embedding boyutu (None ise model'den al)
            device: 'auto', 'cuda', 'cpu'
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.num_blocks = num_blocks
        self.top_k = top_k

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Model'i cihaza taşı (accelerate ile dağıtılmış modeller için kontrol)
        # device_map="auto" kullanıldığında model zaten GPU/CPU'ya dağıtılmıştır
        # ve tekrar taşınmaya çalışılamaz
        if hasattr(self.model, "hf_device_map") or hasattr(self.model, "device_map"):
            # Model zaten accelerate ile dağıtılmış, taşıma
            # Ana cihazı ilk parametrenin device'ından al
            try:
                first_param = next(self.model.parameters())
                self.device = first_param.device
                print(f"ℹ️  Model zaten {self.device} üzerinde dağıtılmış durumda (accelerate).")
            except:
                # Parametre bulunamazsa varsayılan cihazı kullan
                pass
        else:
            # Model henüz dağıtılmamış (GPT-2 gibi küçük modeller), taşı
            self.model.to(self.device)

        # KRİTİK: Önce layers'ı çıkar (rotary_emb layers'a bağlı)
        # Sonra Qwen kontrolü ve rotary embeddings
        self._is_qwen = self._detect_qwen_model()
        
        # Layers'ı ÖNCE çıkar (_extract_rotary_embeddings self.layers'a erişiyor)
        self.layers = self._extract_layers()
        
        self._rotary_emb = None
        if self._is_qwen and self.model is not None:
            self._rotary_emb = self._extract_rotary_embeddings()

        # NO-SHARDING TEST MODU: Modeli hiç bölmeden tek blok olarak çalıştır
        # Bu mod, sorunun sharding'den mi yoksa model yükleme/tokenizer'dan mı kaynaklandığını test eder
        self.no_sharding = no_sharding
        
        if no_sharding:
            # Tüm katmanları tek bir blok olarak kullan
            print("⚠️  NO-SHARDING TEST MODU: Model hiç bölünmeden tek blok olarak çalışacak")
            total_layers = len(self.layers)
            self.num_blocks = 1
            self.top_k = 1
            self.layers_per_block = total_layers
            # Tüm katmanları tek bir blokta topla
            wrapped_layers = [TupleCleaner(layer) for layer in self.layers]
            # Qwen için özel wrapper kullan
            if self._is_qwen:
                self.blocks = nn.ModuleList([QwenBlockWrapper(wrapped_layers)])
            else:
                self.blocks = nn.ModuleList([nn.Sequential(*wrapped_layers)])
            print(f"   Tüm {total_layers} katman tek blokta: Block 0")
        else:
            # Normal mod: Modeli bloklara böl
            total_layers = len(self.layers)

            if layers_per_block is None:
                layers_per_block = max(1, total_layers // num_blocks)
            self.layers_per_block = layers_per_block

            # Blokları oluştur
            self.blocks = self._create_blocks()

        # Embedding boyutunu bul
        if embed_dim is None:
            embed_dim = self._get_embed_dim()
        self.embed_dim = embed_dim

        # Router oluştur (no_sharding modunda router gerekli değil ama yine de oluştur)
        if no_sharding:
            # No-sharding modunda router kullanılmayacak ama yine de oluştur (API uyumluluğu için)
            self.router = ExternalParisiNashRouter(
                embed_dim=embed_dim,
                num_blocks=1,
                top_k=1,
            ).to(self.device)
        else:
            self.router = ExternalParisiNashRouter(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                top_k=top_k,
            ).to(self.device)
        
        # KRİTİK: Router'ı modelin dtype'ına cast et
        # Model float16 iken router float32 → LayerNorm'da dtype mismatch
        if self.model is not None:
            try:
                model_dtype = next(self.model.parameters()).dtype
                if model_dtype != torch.float32:
                    self.router = self.router.to(dtype=model_dtype)
                    print(f"   Router dtype: {model_dtype} (model ile eşleştirildi)")
            except StopIteration:
                pass

        # Embedding layer'ı bul
        self.embed_layer = self._get_embed_layer()
        
        # KRİTİK: Tokenizer ve Model vocab_size kontrolü
        # Bu kontrol, karakter kayması (offset) hatalarını önler
        self._validate_vocab_alignment()
        
        # KRİTİK: Embedding layer'a padding ekle (vocab size mismatch için)
        self._pad_embedding_layer()
        
        # Lazy loading için
        self._lazy_load = False
        self._block_paths = []
        self._loaded_blocks = {}
        
        # Korumalı bloklar (thrashing'i önlemek için)
        # Block 0 ve Block 1 varsayılan olarak kilitli (en sık kullanılan bloklar)
        # Block 0: Temel dil yapısı (alfabe, bağlaçlar)
        # Block 1: İlk transformer katmanları (sık kullanılır)
        self._locked_blocks = {0, 1}  # Set: {0, 1, ...} manuel olarak kilitlenebilir
        
        # Blok kullanım takibi (cleanup için)
        self._block_usage_count = {}  # {block_idx: kullanım_sayısı}
        
        # Sticky Routing: Bir kez seçilen bloklar belirli bir süre sabit kalır
        # Bu sayede her token için router çalıştırmak yerine, bloklar sabitlenir
        # Örnek: "Tarih" konusu için Blok 2 ve 3 seçildiyse, sonraki 25 token boyunca
        # aynı bloklar kullanılır (SSD trafiği biter, metin akıcılaşır)
        self._sticky_blocks = None  # Şu an sabitlenmiş bloklar (set veya None)
        self._sticky_until_token = 0  # Hangi token'a kadar sabit kalacak
        self._sticky_duration = sticky_duration  # Kaç token boyunca sabit kalır

    def _extract_layers(self) -> nn.ModuleList:
        """
        Model'den transformer layer'larını çıkar.
        Desteklenen yapılar:
        - Llama: model.model.layers
        - Qwen: model.model.layers veya model.layers
        - GPT-2: model.transformer.h
        - Mistral: model.model.layers
        """
        # Qwen/Llama/Mistral: model.model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        # Bazı Qwen varyantları: model.layers
        elif hasattr(self.model, "layers"):
            return self.model.layers
        # GPT-2: model.transformer.h
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        else:
            # Debug: Model yapısını göster
            print(f"⚠️  Model yapısı: {type(self.model)}")
            if hasattr(self.model, "model"):
                print(f"   model.model: {type(self.model.model)}")
                if hasattr(self.model.model, "__dict__"):
                    print(f"   model.model attributes: {list(self.model.model.__dict__.keys())[:10]}")
            raise ValueError(
                "Model yapısı desteklenmiyor. Llama/Qwen/Mistral gibi modeller bekleniyor.\n"
                f"Model tipi: {type(self.model)}\n"
                "Lütfen model yapısını kontrol edin veya issue açın."
            )

    def _detect_qwen_model(self) -> bool:
        """Qwen modeli olup olmadığını tespit et."""
        if self.model is None:
            return False
        model_type = type(self.model).__name__.lower()
        return 'qwen' in model_type or 'qwen2' in model_type
    
    def _extract_rotary_embeddings(self):
        """
        Qwen modelinden rotary embeddings'i çıkar.
        Yeni transformers: model.model.rotary_emb (model seviyesinde)
        Eski transformers: layers[0].self_attn.rotary_emb (layer seviyesinde)
        """
        if self.model is None:
            return None
        
        rotary_emb = None
        
        # Yöntem 1: model.model.rotary_emb (yeni transformers >= 4.38)
        try:
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'rotary_emb'):
                rotary_emb = self.model.model.rotary_emb
                print(f"✅ Rotary emb bulundu: model.model.rotary_emb ({type(rotary_emb).__name__})")
                return rotary_emb
        except Exception:
            pass
        
        # Yöntem 2: layers[0].self_attn.rotary_emb (eski transformers)
        try:
            if self.layers is not None and len(self.layers) > 0:
                first_layer = self.layers[0]
                if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'rotary_emb'):
                    rotary_emb = first_layer.self_attn.rotary_emb
                    print(f"✅ Rotary emb bulundu: layers[0].self_attn.rotary_emb ({type(rotary_emb).__name__})")
                    return rotary_emb
        except Exception:
            pass
        
        print(f"⚠️  Rotary embeddings bulunamadı. Model yapısını kontrol edin.")
        if hasattr(self.model, 'model'):
            attrs = [a for a in dir(self.model.model) if 'rotary' in a.lower() or 'rope' in a.lower()]
            print(f"   model.model'deki rotary/rope attributes: {attrs}")
        if self.layers is not None and len(self.layers) > 0:
            layer0 = self.layers[0]
            if hasattr(layer0, 'self_attn'):
                attrs = [a for a in dir(layer0.self_attn) if 'rotary' in a.lower() or 'rope' in a.lower()]
                print(f"   layers[0].self_attn'deki rotary/rope attributes: {attrs}")
        return None
    
    def _get_embed_layer(self) -> nn.Module:
        """
        Embedding layer'ı bul.
        Desteklenen yapılar:
        - Llama/Qwen: model.model.embed_tokens
        - GPT-2: model.transformer.wte
        - Bazı modeller: model.embed_tokens
        """
        # Qwen/Llama: model.model.embed_tokens
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return self.model.model.embed_tokens
        # GPT-2: model.transformer.wte
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "wte"):
            return self.model.transformer.wte
        # Bazı modeller: model.embed_tokens
        elif hasattr(self.model, "embed_tokens"):
            return self.model.embed_tokens
        else:
            # Debug bilgisi
            print(f"⚠️  Embedding layer bulunamadı. Model yapısı:")
            if hasattr(self.model, "model"):
                print(f"   model.model attributes: {[k for k in dir(self.model.model) if 'embed' in k.lower()]}")
            raise ValueError(
                "Embedding layer bulunamadı.\n"
                f"Model tipi: {type(self.model)}\n"
                "Lütfen model yapısını kontrol edin."
            )

    def _validate_vocab_alignment(self):
        """
        Tokenizer ve Model vocab_size'larının eşleştiğini kontrol et.
        Bu kontrol, karakter kayması (offset) hatalarını önler.
        Örnek: Model 'A' demek isterken '焘' (Çince karakter) basması.
        
        KRİTİK: Model sayılarla konuşur. Her kelimenin bir ID'si vardır.
        - Tokenizer: "Tarih" kelimesine 150 diyor
        - Model: 150 sayısını işliyor ve "200" diyor
        - LM Head: Eğer offset varsa, 200 "bilgi" yerine "焘" olabilir
        """
        # Tokenizer vocab_size
        tokenizer_vocab_size = None
        if hasattr(self.tokenizer, 'vocab_size'):
            tokenizer_vocab_size = self.tokenizer.vocab_size
        elif hasattr(self.tokenizer, 'get_vocab'):
            tokenizer_vocab_size = len(self.tokenizer.get_vocab())
        
        # Embedding vocab_size
        embed_vocab_size = None
        if hasattr(self.embed_layer, 'weight'):
            embed_vocab_size = self.embed_layer.weight.shape[0]
        
        # LM Head vocab_size
        lm_head_vocab_size = None
        if self.model is not None:
            if hasattr(self.model, 'lm_head') and hasattr(self.model.lm_head, 'weight'):
                lm_head_vocab_size = self.model.lm_head.weight.shape[0]
        
        # Kontrol ve uyarı
        vocab_sizes = {
            'tokenizer': tokenizer_vocab_size,
            'embedding': embed_vocab_size,
            'lm_head': lm_head_vocab_size,
        }
        
        # Tüm vocab_size'ları yazdır (debug için)
        print(f"📊 Vocab Size Kontrolü (Sayısal Karşılık Kontrolü):")
        for name, size in vocab_sizes.items():
            if size is not None:
                print(f"   {name}: {size}")
            else:
                print(f"   {name}: Bulunamadı")
        
        # Eşleşme kontrolü
        sizes = [v for v in vocab_sizes.values() if v is not None]
        if len(sizes) > 1:
            if len(set(sizes)) > 1:
                print(f"\n⚠️  KRİTİK UYARI: Vocab size uyumsuzluğu tespit edildi!")
                print(f"   Bu durum karakter kayması (offset) hatalarına neden olur.")
                print(f"   Örnek: Model 'A' demek isterken '焘' (Çince karakter) basabilir.")
                print(f"\n   🔍 Analiz:")
                print(f"   - Tokenizer vocab_size: {tokenizer_vocab_size}")
                print(f"   - Embedding vocab_size: {embed_vocab_size}")
                print(f"   - LM Head vocab_size: {lm_head_vocab_size}")
                print(f"\n   💡 Çözüm:")
                if embed_vocab_size and lm_head_vocab_size:
                    if embed_vocab_size == lm_head_vocab_size:
                        print(f"   - Embedding ve LM Head eşleşiyor ({embed_vocab_size})")
                        print(f"   - Sorun: Tokenizer ile Model arasında offset var")
                        offset = embed_vocab_size - (tokenizer_vocab_size or 0)
                        print(f"   - Offset: {offset} token (Model daha büyük)")
                        print(f"   - Model checkpoint'teki vocab_size kullanılacak: {embed_vocab_size}")
                        
                        # KRİTİK: Tokenizer vocab_size'ı model vocab_size'ına eşitle
                        # Padding token'ları ekle veya tokenizer'ı güncelle
                        if tokenizer_vocab_size and offset > 0:
                            print(f"\n   🔧 Tokenizer Padding: {offset} dummy token ekleniyor...")
                            self._pad_tokenizer_vocab(tokenizer_vocab_size, embed_vocab_size)
                    else:
                        print(f"   - Embedding ({embed_vocab_size}) != LM Head ({lm_head_vocab_size})")
                        print(f"   - Bu durum ciddi bir sorun! Checkpoint'i kontrol edin.")
                print(f"\n   🧪 Test: no_sharding=True ile test edin")
                print(f"   - Çalışıyorsa: Sorun sharding'de")
                print(f"   - Hala bozuksa: Sorun vocab mapping'de")
            else:
                print(f"✅ Vocab size'lar eşleşiyor: {sizes[0]}")
        
        # Token ID mapping kontrolü (opsiyonel ama önerilir)
        # Test: Basit bir token'ın ID'sini kontrol et
        try:
            test_tokens = ["The", "history", "of"]
            print(f"\n🔍 Token ID Mapping Kontrolü:")
            for test_token in test_tokens:
                try:
                    token_id = self.tokenizer.encode(test_token, add_special_tokens=False)[0]
                    decoded = self.tokenizer.decode([token_id])
                    print(f"   '{test_token}' → ID: {token_id} → Decode: '{decoded}'")
                    
                    # ID'nin vocab_size içinde olduğundan emin ol
                    if embed_vocab_size and token_id >= embed_vocab_size:
                        print(f"   ⚠️  UYARI: Token ID {token_id} >= Embedding vocab_size {embed_vocab_size}")
                        print(f"      Bu durum IndexError'a neden olabilir!")
                except Exception as e:
                    print(f"   ⚠️  Token '{test_token}' kontrol edilemedi: {e}")
        except Exception as e:
            print(f"   Token ID mapping kontrolü atlandı: {e}")
    
    def _pad_tokenizer_vocab(self, current_size: int, target_size: int):
        """
        Tokenizer vocab_size'ı model vocab_size'ına eşitlemek için padding token'ları ekle.
        
        KRİTİK: Bu fonksiyon tokenizer'ın vocab_size'ını artırmaz ama
        embedding layer'ın beklediği token ID'lerinin geçerli olduğundan emin olur.
        
        Not: HuggingFace tokenizer'ların vocab_size'ını değiştirmek zor olduğu için,
        bu fonksiyon sadece uyarı verir ve embedding layer'ın padding'i handle etmesini bekler.
        """
        offset = target_size - current_size
        if offset > 0:
            print(f"   ⚠️  Tokenizer vocab_size ({current_size}) < Model vocab_size ({target_size})")
            print(f"   ⚠️  Offset: {offset} token")
            print(f"   💡 Not: Tokenizer'ın vocab_size'ını değiştirmek zor.")
            print(f"   💡 Çözüm: Embedding layer padding'i handle edecek.")
            print(f"   💡 Tokenizer token ID'leri 0-{current_size-1} arası, Model 0-{target_size-1} bekliyor.")
            print(f"   💡 Eğer tokenizer token ID >= {current_size} kullanırsa IndexError oluşabilir.")
    
    def _pad_embedding_layer(self):
        """
        Embedding layer'a padding ekle (vocab size mismatch için).
        
        KRİTİK: Eğer tokenizer vocab_size < embedding vocab_size ise,
        embedding layer'ın son token'ları kullanılmıyor olabilir.
        Bu durumda embedding layer'ı tokenizer vocab_size'a göre clamp edebiliriz
        veya padding token'ları ekleyebiliriz.
        
        Ancak şu an için sadece kontrol yapıyoruz, gerçek padding eklemiyoruz
        çünkü model'in ağırlıklarını değiştirmek istemiyoruz.
        """
        if self.model is None:
            return
        
        # Tokenizer vocab_size
        tokenizer_vocab_size = None
        if hasattr(self.tokenizer, 'vocab_size'):
            tokenizer_vocab_size = self.tokenizer.vocab_size
        elif hasattr(self.tokenizer, 'get_vocab'):
            tokenizer_vocab_size = len(self.tokenizer.get_vocab())
        
        # Embedding vocab_size
        embed_vocab_size = None
        if hasattr(self.embed_layer, 'weight'):
            embed_vocab_size = self.embed_layer.weight.shape[0]
        
        # Eğer tokenizer vocab_size < embedding vocab_size ise
        if tokenizer_vocab_size and embed_vocab_size and tokenizer_vocab_size < embed_vocab_size:
            offset = embed_vocab_size - tokenizer_vocab_size
            print(f"\n💡 Embedding Padding Bilgisi:")
            print(f"   - Tokenizer vocab_size: {tokenizer_vocab_size}")
            print(f"   - Embedding vocab_size: {embed_vocab_size}")
            print(f"   - Offset: {offset} token (embedding'in son {offset} token'ı kullanılmıyor)")
            print(f"   - Tokenizer token ID'leri 0-{tokenizer_vocab_size-1} arası")
            print(f"   - Model embedding 0-{embed_vocab_size-1} arası bekliyor")
            print(f"   - Bu durum normal, model'in son token'ları padding için olabilir")
    
    def _get_embed_dim(self) -> int:
        """Embedding boyutunu bul."""
        embed_layer = self._get_embed_layer()
        if hasattr(embed_layer, "embedding_dim"):
            return embed_layer.embedding_dim
        elif hasattr(embed_layer, "weight"):
            return embed_layer.weight.shape[1]
        else:
            return 4096  # Varsayılan (Llama-2-7b)

    def _create_blocks(self) -> nn.ModuleList:
        """
        Layer'ları bloklara böl.
        Her layer'ı TupleCleaner ile sararak tuple çıktılarını temizleriz.
        
        KRİTİK: Qwen2 gibi modeller position_embeddings bekler.
        Qwen için özel QwenBlockWrapper kullanılır.
        """
        blocks = nn.ModuleList()
        total_layers = len(self.layers)

        for i in range(self.num_blocks):
            start_idx = i * self.layers_per_block
            end_idx = min((i + 1) * self.layers_per_block, total_layers)
            if start_idx < total_layers:
                block_layers = self.layers[start_idx:end_idx]
                # Her layer'ı TupleCleaner ile sar (tuple çıktılarını temizle)
                wrapped_layers = [TupleCleaner(layer) for layer in block_layers]
                
                # Qwen için özel wrapper kullan (position_embeddings geçirmek için)
                if self._is_qwen:
                    blocks.append(QwenBlockWrapper(wrapped_layers))
                else:
                    blocks.append(nn.Sequential(*wrapped_layers))
            else:
                # Boş blok (padding)
                blocks.append(nn.Identity())

        return blocks

    @torch.no_grad()
    def predict_blocks(self, prompt: str, prefetch: bool = True) -> Tuple[List[int], torch.Tensor]:
        """
        Teoreminin beyni: Giriş cümlesine bakarak hangi blokların
        gerekli olduğunu tahmin eder. Modeli çalıştırmadan önce çağrılır.

        Args:
            prompt: Giriş metni
            prefetch: True ise tahmin edilen blokları arka planda yüklemeye başlar

        Returns:
            block_indices: [i1, i2, ...] yüklenecek blok indeksleri
            weights: (top_k,) blok ağırlıkları
        """
        self.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        x = self.embed_layer(input_ids)  # (1, L, D)

        block_indices, weights = self.router.get_predictive_indices(x, pool_input=True)
        
        # Asenkron prefetching: Tahmin edilen blokları arka planda yükle
        if prefetch and self._lazy_load:
            self.prefetch_blocks(block_indices)
        
        return block_indices, weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,  # Zorunlu olarak False (KV Cache henüz desteklenmiyor)
        current_token_idx: Optional[int] = None,  # Sticky routing için: şu anki token indeksi
    ) -> dict:
        """
        Forward pass: Sadece router'ın seçtiği bloklar çalışır.
        Sticky Routing: Eğer sticky blocks varsa, router'ı atla ve sabitlenmiş blokları kullan.

        Args:
            input_ids: (B, L) token ids
            attention_mask: (B, L) attention mask (opsiyonel)
            use_cache: KV cache kullan (opsiyonel, şimdilik False zorunlu)
            current_token_idx: Şu anki token indeksi (sticky routing için)

        Returns:
            logits, hidden_states, selected_indices, router_info
        """
        # KRİTİK: KV Cache henüz desteklenmiyor (TupleCleaner past_key_values'i kaybediyor)
        # Şimdilik use_cache=False zorunlu tutuyoruz
        use_cache = False
        
        self.eval()
        B, L = input_ids.shape

        # Embedding
        x = self.embed_layer(input_ids)  # (B, L, D)
        
        # KRİTİK: Qwen2 katmanları position_embeddings'i ZORUNLU olarak bekler
        # position_embeddings = (cos, sin) tuple'ı — RoPE için gerekli
        # Bu olmadan TypeError: cannot unpack non-iterable NoneType object hatası alınır
        position_embeddings = None
        position_ids = None
        
        if self._is_qwen:
            try:
                # Position IDs oluştur
                position_ids = torch.arange(L, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand(B, -1)
                
                # Rotary embedding kaynağını bul (birden fazla fallback)
                rotary_emb = self._rotary_emb
                
                # Fallback 1: model.model.rotary_emb (yeni transformers >= 4.38)
                if rotary_emb is None and self.model is not None:
                    try:
                        if hasattr(self.model, 'model') and hasattr(self.model.model, 'rotary_emb'):
                            rotary_emb = self.model.model.rotary_emb
                    except Exception:
                        pass
                
                # Fallback 2: layers[0].self_attn.rotary_emb (eski transformers)
                if rotary_emb is None and self.layers is not None and len(self.layers) > 0:
                    try:
                        first_layer = self.layers[0]
                        if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'rotary_emb'):
                            rotary_emb = first_layer.self_attn.rotary_emb
                    except Exception:
                        pass
                
                if rotary_emb is not None:
                    # Rotary embeddings'den (cos, sin) hesapla
                    # Farklı transformers versiyonları farklı imza kullanır
                    cos, sin = None, None
                    
                    # Yöntem 1: rotary_emb(x, position_ids) — yeni transformers
                    try:
                        result = rotary_emb(x, position_ids)
                        if isinstance(result, tuple) and len(result) == 2:
                            cos, sin = result
                    except Exception:
                        pass
                    
                    # Yöntem 2: rotary_emb(position_ids, seq_len=L) — eski transformers
                    if cos is None:
                        try:
                            cos, sin = rotary_emb(position_ids, seq_len=L)
                        except Exception:
                            pass
                    
                    # Yöntem 3: rotary_emb(position_ids) — bazı versiyonlar
                    if cos is None:
                        try:
                            cos, sin = rotary_emb(position_ids)
                        except Exception:
                            pass
                    
                    if cos is not None and sin is not None:
                        # Device kontrolü
                        if cos.device != input_ids.device:
                            cos = cos.to(input_ids.device)
                        if sin.device != input_ids.device:
                            sin = sin.to(input_ids.device)
                        position_embeddings = (cos, sin)
                    else:
                        if not getattr(self, '_rotary_warn_printed', False):
                            print("⚠️  Rotary embeddings hesaplanamadı — tüm yöntemler başarısız")
                            self._rotary_warn_printed = True
                else:
                    if not getattr(self, '_rotary_warn_printed', False):
                        print("⚠️  Rotary embedding bulunamadı (ne _rotary_emb ne de katmanlarda)")
                        self._rotary_warn_printed = True
            except Exception as e:
                print(f"⚠️  Position embeddings hesaplanamadı: {e}")
                import traceback
                traceback.print_exc()

        # NO-SHARDING MODU: Router'ı atla, tüm katmanları tek blokta çalıştır
        if self.no_sharding:
            selected_indices = [0]  # Tek blok: Block 0
            weights = torch.ones(1)
            probs = None
            aux_loss = None
        # SEQUENTIAL LAZY MODU: Tüm blokları sırayla çalıştır (VRAM tasarrufu + kalite)
        # Transformer katmanları sıralıdır — blok atlamak kalite kaybına neden olur
        # Bu mod: yükle → çalıştır → bellekten sil — tüm bloklar çalışır ama sadece 1-2 bellekte
        elif getattr(self, '_sequential_all', False):
            selected_indices = list(range(self.num_blocks))
            weights = torch.ones(self.num_blocks) / self.num_blocks
            probs = None
            aux_loss = None
        # STICKY ROUTING: Eğer sticky blocks varsa ve henüz süresi dolmamışsa, router'ı atla
        elif self._sticky_blocks is not None and current_token_idx is not None:
            if current_token_idx < self._sticky_until_token:
                # Sticky blocks'u kullan, router'ı çalıştırma
                selected_indices = list(self._sticky_blocks)
                # Sticky blocks için dummy weights (router çalışmadığı için)
                weights = torch.ones(len(selected_indices)) / len(selected_indices)
                probs = None
                aux_loss = None
            else:
                # Sticky süresi doldu, router'ı tekrar çalıştır ve yeni bloklar seç
                self._sticky_blocks = None
                self._sticky_until_token = 0
                # Router ile blok seçimi
                probs, indices, aux_loss, weights = self.router(x, pool_input=True)
                indices = indices.squeeze().cpu().tolist()
                if isinstance(indices, int):
                    indices = [indices]
                selected_indices = indices[: self.top_k]
        else:
            # Normal router ile blok seçimi
            probs, indices, aux_loss, weights = self.router(x, pool_input=True)
            indices = indices.squeeze().cpu().tolist()
            if isinstance(indices, int):
                indices = [indices]
            selected_indices = indices[: self.top_k]

        # TÜM blokları sırayla çalıştır:
        # - Seçilen bloklar: Full hesaplama (VRAM kullanır)
        # - Seçilmeyen bloklar: Identity/residual geçiş (sıfır VRAM, sıfır hesaplama)
        # Bu sayede transformer'ın residual stream'i kesintisiz devam eder.
        # Layer dropping mantığı: output = input (bloğun katkısı atlanır, ama akış kopmaz)
        x_out = x
        selected_set = set(selected_indices)
        
        for idx in range(self.num_blocks):
            if idx in selected_set:
                # ── SEÇİLEN BLOK: Full hesaplama ──
                if self._lazy_load:
                    with self._prefetch_lock:
                        if idx in self._loaded_blocks:
                            block = self._loaded_blocks[idx]
                        else:
                            block = self._load_block_from_disk(idx)
                else:
                    block = self.blocks[idx]
                
                if isinstance(block, QwenBlockWrapper) and position_embeddings is not None:
                    block_out = block(x_out, position_embeddings=position_embeddings, 
                                     position_ids=position_ids, 
                                     attention_mask=attention_mask)
                else:
                    block_out = block(x_out)
                
                # DEFANSİF KODLAMA: Tuple kontrolü
                if isinstance(block_out, tuple):
                    x_out = block_out[0]
                elif hasattr(block_out, 'last_hidden_state'):
                    x_out = block_out.last_hidden_state
                elif isinstance(block_out, torch.Tensor):
                    x_out = block_out
                else:
                    try:
                        x_out = block_out[0] if hasattr(block_out, '__getitem__') else block_out
                    except Exception:
                        x_out = block_out
                
                while isinstance(x_out, (tuple, list)) and len(x_out) > 0:
                    x_out = x_out[0]
                
                if not isinstance(x_out, torch.Tensor):
                    raise TypeError(f"Blok {idx} çıktısı Tensor değil: {type(x_out)}")
            else:
                # ── SEÇİLMEYEN BLOK: Identity (residual geçiş) ──
                # x_out = x_out  →  Bloğun katkısı atlanır ama akış kopmaz
                pass

        # Final norm'dan önce tuple kontrolü
        if isinstance(x_out, tuple):
            x_out = x_out[0]
        elif hasattr(x_out, 'last_hidden_state'):
            x_out = x_out.last_hidden_state
        
        # Final norm ve LM head (model'e göre değişir)
        # Lazy loading durumunda model yok, final norm ve lm_head kaydedilmiş olmalı
        if self.model is not None:
            if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
                x_out = self.model.model.norm(x_out)
                logits = self.model.lm_head(x_out)
            elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "ln_f"):
                x_out = self.model.transformer.ln_f(x_out)
                logits = self.model.lm_head(x_out)
            else:
                # Basit fallback
                logits = self.model.lm_head(x_out) if hasattr(self.model, "lm_head") else None
        else:
            # Lazy loading: kaydedilmiş final norm ve lm_head kullan
            if self._final_norm is not None:
                x_out = self._final_norm(x_out)
            if self._lm_head is not None:
                logits = self._lm_head(x_out)
            else:
                logits = None
                print("⚠️  Lazy loading: LM head bulunamadı")

        # Router weights'i hazırla
        weights_cpu = weights.squeeze().cpu()
        if weights_cpu.dim() == 0:
            weights_cpu = weights_cpu.unsqueeze(0)
        weights_list = weights_cpu[: len(selected_indices)].tolist()

        # Blok kullanım takibini güncelle
        for idx in selected_indices:
            self._block_usage_count[idx] = self._block_usage_count.get(idx, 0) + 1
        
        # Lazy loading: Kullanılmayan blokları RAM'den kaldır (daha az agresif)
        # KRİTİK: Sadece gerçekten uzun süre kullanılmayan blokları sil
        # Thrashing'i önlemek için cleanup'ı daha az agresif yapıyoruz
        if self._lazy_load and len(self._loaded_blocks) > (self.top_k * 2) + len(self._locked_blocks):
            # Kilitli blokları ve seçili blokları hariç tut
            unused_blocks = set(self._loaded_blocks.keys()) - set(selected_indices) - self._locked_blocks
            
            # En az kullanılan blokları bul (kullanım sayısına göre)
            unused_with_counts = [
                (idx, self._block_usage_count.get(idx, 0))
                for idx in unused_blocks
            ]
            unused_with_counts.sort(key=lambda x: x[1])  # En az kullanılan önce
            
            # Sadece en az kullanılan birkaç bloğu kaldır (thrashing'i önlemek için)
            blocks_to_remove = max(1, len(unused_blocks) - self.top_k)
            for unused_idx, _ in unused_with_counts[:blocks_to_remove]:
                self._unload_block_from_memory(unused_idx)
                # Kullanım sayacını sıfırla (kaldırıldığı için)
                self._block_usage_count.pop(unused_idx, None)
        
        return {
            "logits": logits,
            "hidden_states": x_out,
            "selected_indices": selected_indices,
            "router_weights": weights_list,
            "router_info": self.router.get_stats(),
            "blocks_in_memory": list(self._loaded_blocks.keys()) if self._lazy_load else None,
        }

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        prefetch_next: bool = True,
    ) -> str:
        """
        Metin üretimi: Her adımda router hangi blokları kullanacağına karar verir.
        Asenkron prefetching ile bir sonraki adımın blokları önceden yüklenir.

        Args:
            prompt: Başlangıç metni
            max_new_tokens: Üretilecek maksimum token sayısı
            temperature: Sampling sıcaklığı
            top_k: Top-K sampling
            prefetch_next: True ise bir sonraki adımın bloklarını önceden yükle

        Returns:
            Üretilmiş metin
        """
        self.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated = input_ids

        # İlk blokları prefetch et
        if prefetch_next and self._lazy_load:
            first_indices, _ = self.predict_blocks(prompt, prefetch=True)

        # STICKY ROUTING: İlk prompt için router çalışır ve blokları sabitler
        initial_prompt_len = generated.shape[1]
        
        for step in range(max_new_tokens):
            current_token_idx = initial_prompt_len + step
            
            # Qwen2 gibi modern modeller causal masking'i kendi içinde halleder
            # Dışarıdan attention_mask göndermek dtype uyumsuzluğuna neden olabilir
            # (SDPA bool/float bekler, long gönderirsek RuntimeError)
            
            # Her adımda TÜM bağlamı forward'a gönder (KV Cache olmadığı için)
            outputs = self.forward(generated, attention_mask=None, use_cache=False, current_token_idx=current_token_idx)
            
            # Eğer sticky blocks yoksa veya süresi dolduysa, yeni blokları sabitle
            # NO-SHARDING modunda sticky routing yok (zaten tek blok var)
            if not self.no_sharding and self._sticky_blocks is None:
                selected_indices = outputs["selected_indices"]
                self._sticky_blocks = set(selected_indices)
                self._sticky_until_token = current_token_idx + self._sticky_duration
                print(f"🔒 Sticky Routing: Bloklar {self._sticky_blocks} sabitlendi (token {current_token_idx}-{self._sticky_until_token})")
            
            # Son token'ın logits'ini al (tüm bağlam üzerinden)
            logits = outputs["logits"][:, -1, :] / temperature

            # KRİTİK: Tokenizer vocab_size'dan büyük token ID'leri clamp et
            # Model 152064 token bekliyor ama tokenizer sadece 151643 token biliyor
            # Eğer model tokenizer'ın vocab_size'ından büyük bir ID üretirse, clamp et
            tokenizer_vocab_size = None
            if hasattr(self.tokenizer, 'vocab_size'):
                tokenizer_vocab_size = self.tokenizer.vocab_size
            elif hasattr(self.tokenizer, 'get_vocab'):
                tokenizer_vocab_size = len(self.tokenizer.get_vocab())
            
            if tokenizer_vocab_size and logits.size(-1) > tokenizer_vocab_size:
                # Logits'in son kısmını -inf yap (tokenizer'ın bilmediği token'lar)
                logits[:, tokenizer_vocab_size:] = float("-inf")
                if step == 0:  # Sadece ilk adımda yazdır
                    print(f"   ⚠️  Token ID clamp: Logits {logits.size(-1)} → {tokenizer_vocab_size} (tokenizer vocab_size) [sonraki adımlarda sessiz]")

            # Top-K sampling
            topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < topk_vals[:, -1:]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # KRİTİK: Üretilen token ID'sini tokenizer vocab_size'a clamp et
            if tokenizer_vocab_size:
                next_token = torch.clamp(next_token, 0, tokenizer_vocab_size - 1)
            
            # Yeni token'ı bağlama ekle (bir sonraki adım için)
            generated = torch.cat([generated, next_token], dim=1)

            # Prefetching: Sticky routing aktifken prefetch'e gerek yok
            # (Bloklar zaten sabitlenmiş ve RAM'de)
            if prefetch_next and self._lazy_load and step < max_new_tokens - 1:
                # Sadece sticky süresi dolmak üzereyse bir sonraki adımın bloklarını tahmin et
                if current_token_idx >= self._sticky_until_token - 5:  # 5 token önceden tahmin et
                    next_prompt = self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
                    next_indices, _ = self.predict_blocks(next_prompt, prefetch=True)

        result = self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        
        # Sticky blocks'u temizle (bir sonraki generate için)
        self._sticky_blocks = None
        self._sticky_until_token = 0
        
        return result

    def estimate_vram_savings(self) -> dict:
        """
        VRAM tasarrufu tahmini: Tüm model vs. sadece top_k blok.
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        block_params = sum(p.numel() for p in self.blocks[0].parameters())
        router_params = sum(p.numel() for p in self.router.parameters())

        # Tüm model (float32)
        total_vram = total_params * 4 / (1024**3)  # GB

        # Sadece top_k blok + router
        sparse_vram = (block_params * self.top_k + router_params) * 4 / (1024**3)

        return {
            "total_params": total_params,
            "total_vram_gb": total_vram,
            "sparse_vram_gb": sparse_vram,
            "savings_ratio": total_vram / max(sparse_vram, 1e-9),
            "blocks_loaded": f"{self.top_k}/{self.num_blocks}",
        }

    def save_blocks_to_disk(self, save_dir: str):
        """
        Modeli bloklara ayırıp diske kaydet (Sharding).
        Her blok ayrı bir dosya olarak kaydedilir: block_0.pt, block_1.pt, ...
        
        Args:
            save_dir: Blokların kaydedileceği dizin
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"💾 Model blokları diske kaydediliyor: {save_dir}")
        
        for i, block in enumerate(self.blocks):
            block_path = os.path.join(save_dir, f"block_{i}.pt")
            
            # KRİTİK: Tam modülü kaydet (state_dict DEĞİL!)
            # state_dict sadece sayıları kaydeder, modül yapısını kaybeder
            # torch.save(module) ise gerçek Qwen2DecoderLayer'ları korur
            # Böylece diskten yüklenince forward pass gerçek hesaplama yapar
            block_cpu = block.cpu()
            torch.save(block_cpu, block_path)
            block.to(self.device)  # Geri GPU'ya taşı (sharding modu için)
            
            num_layers = len(block.layers) if isinstance(block, QwenBlockWrapper) else len(list(block.children()))
            block_size_mb = os.path.getsize(block_path) / (1024**2)
            print(f"   Blok {i}: {block_size_mb:.2f} MB ({num_layers} layer) → {block_path}")
        
        # Final norm ve LM head'i de kaydet
        final_norm_state = None
        lm_head_state = None
        norm_type = None  # RMSNorm vs LayerNorm
        
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            final_norm_state = self.model.model.norm.state_dict()
            lm_head_state = self.model.lm_head.state_dict()
            norm_type = type(self.model.model.norm).__name__
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "ln_f"):
            final_norm_state = self.model.transformer.ln_f.state_dict()
            lm_head_state = self.model.lm_head.state_dict()
            norm_type = type(self.model.transformer.ln_f).__name__
        
        # Rotary embeddings'i AYRI DOSYA olarak kaydet (lazy loading için KRİTİK)
        # NOT: Yeni transformers'ta state_dict() boş {} dönüyor (parametresiz)
        # Bu yüzden modülü doğrudan kaydediyoruz
        if self._rotary_emb is not None:
            rotary_path = os.path.join(save_dir, "rotary_emb.pt")
            torch.save(self._rotary_emb, rotary_path)
            print(f"   Rotary embeddings kaydedildi: {type(self._rotary_emb).__name__} → rotary_emb.pt")
        
        # Router, embedding ve metadata'yı kaydet
        router_path = os.path.join(save_dir, "router.pt")
        torch.save({
            'router_state_dict': self.router.state_dict(),
            'embed_state_dict': self.embed_layer.state_dict(),
            'final_norm_state_dict': final_norm_state,
            'lm_head_state_dict': lm_head_state,
            'config': {
                'num_blocks': self.num_blocks,
                'top_k': self.top_k,
                'embed_dim': self.embed_dim,
                'layers_per_block': self.layers_per_block,
                'has_final_norm': final_norm_state is not None,
                'is_qwen': self._is_qwen,
                'norm_type': norm_type,
                'model_class': type(self.model).__name__,
                'has_rotary_emb': self._rotary_emb is not None,
            }
        }, router_path)
        
        print(f"   Router + Embedding + Final Norm + LM Head: {os.path.getsize(router_path) / (1024**2):.2f} MB")
        print(f"   Model tipi: {type(self.model).__name__} (is_qwen={self._is_qwen})")
        print(f"✅ Toplam {self.num_blocks} blok kaydedildi")

    @classmethod
    def from_disk_blocks(
        cls,
        tokenizer,
        save_dir: str,
        device: str = "auto",
        lazy_load: bool = True,
        sticky_duration: int = 25,
        sequential_all: bool = True,  # True: tüm bloklar sırayla çalışır (kaliteli), False: router seçer (hızlı)
    ):
        """
        Diskten blokları yükleyerek loader oluştur (Lazy Loading).
        
        Args:
            tokenizer: HuggingFace tokenizer
            save_dir: Blokların kaydedildiği dizin
            device: 'auto', 'cuda', 'cpu'
            lazy_load: True ise bloklar RAM'e yüklenmez, sadece gerektiğinde yüklenir
        
        Returns:
            HuggingFaceBlockLoader instance (bloklar diskte, RAM'de değil)
        """
        import os
        
        # Router config'i yükle
        router_path = os.path.join(save_dir, "router.pt")
        router_data = torch.load(router_path, map_location='cpu')
        config = router_data['config']
        
        if device == "auto":
            device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_obj = torch.device(device)
        
        # Router ve embedding oluştur
        router = ExternalParisiNashRouter(
            embed_dim=config['embed_dim'],
            num_blocks=config['num_blocks'],
            top_k=config['top_k'],
        )
        router.load_state_dict(router_data['router_state_dict'])
        router.to(device_obj)
        
        # Embedding için dummy model gerekli (sadece embed_layer için)
        # Checkpoint'ten gerçek ağırlık boyutunu alalım
        # Tokenizer'dan vocab_size almak yerine, checkpoint'teki gerçek boyutu kullan
        embed_dim = config['embed_dim']
        embed_weight = router_data['embed_state_dict']['weight']
        embed_vocab_size = embed_weight.shape[0]  # Checkpoint'teki gerçek vocab_size (örn: 152064)
        
        embed_layer = nn.Embedding(embed_vocab_size, embed_dim)
        embed_layer.load_state_dict(router_data['embed_state_dict'])
        embed_layer.to(device_obj)
        
        # Final norm ve LM head'i yükle (lazy loading için)
        final_norm = None
        lm_head = None
        
        # Final norm yükle (eğer varsa)
        if config.get('has_final_norm', False):
            if router_data.get('final_norm_state_dict'):
                # Diskteki ağırlıklara bakalım: Bias var mı?
                # Qwen ve bazı yeni Llama modelleri bias'sız LayerNorm kullanır
                has_bias = 'bias' in router_data['final_norm_state_dict']
                
                # LayerNorm'u diskteki yapıya göre oluştur
                final_norm = nn.LayerNorm(embed_dim, bias=has_bias)
                final_norm.load_state_dict(router_data['final_norm_state_dict'])
                final_norm.to(device_obj)
        
        # LM head yükle (has_final_norm kontrolünden bağımsız)
        if router_data.get('lm_head_state_dict'):
            # LM head için de checkpoint'teki gerçek vocab_size'ı kullan
            lm_head_weight = router_data['lm_head_state_dict']['weight']
            lm_head_vocab_size = lm_head_weight.shape[0]  # Checkpoint'teki gerçek vocab_size
            
            # KRİTİK: Embedding ve LM Head vocab_size'larının eşleştiğinden emin ol
            if embed_vocab_size != lm_head_vocab_size:
                raise ValueError(
                    f"Vocab size mismatch: Embedding={embed_vocab_size}, LM Head={lm_head_vocab_size}. "
                    f"Bu durum çıktı karakter bozulmasına neden olur. "
                    f"Örnek: Model 'A' demek isterken '焘' (Çince karakter) basabilir. "
                    f"Checkpoint'i kontrol edin veya modeli yeniden kaydedin."
                )
            
            # Tokenizer vocab_size kontrolü (opsiyonel ama önerilir)
            tokenizer_vocab_size = None
            if hasattr(tokenizer, 'vocab_size'):
                tokenizer_vocab_size = tokenizer.vocab_size
            elif hasattr(tokenizer, 'get_vocab'):
                tokenizer_vocab_size = len(tokenizer.get_vocab())
            
            if tokenizer_vocab_size is not None:
                if embed_vocab_size != tokenizer_vocab_size:
                    print(f"\n⚠️  KRİTİK UYARI: Tokenizer vocab_size ({tokenizer_vocab_size}) != Model vocab_size ({embed_vocab_size})")
                    print(f"   Bu durum karakter kayması (offset) hatalarına neden olur.")
                    print(f"   Örnek: Model 'A' demek isterken '焘' (Çince karakter) basabilir.")
                    print(f"\n   🔍 Analiz:")
                    offset = embed_vocab_size - tokenizer_vocab_size
                    print(f"   - Offset: {offset} token (Model daha büyük)")
                    print(f"   - Tokenizer: Token ID'leri 0-{tokenizer_vocab_size-1} arası")
                    print(f"   - Model: Token ID'leri 0-{embed_vocab_size-1} arası bekliyor")
                    print(f"\n   💡 Çözüm:")
                    print(f"   - Model checkpoint'teki vocab_size kullanılacak: {embed_vocab_size}")
                    print(f"   - Tokenizer'ın token ID'leri model'in embedding matrisine uyacak şekilde ayarlanmalı")
                    print(f"   - Eğer tokenizer'ın ID'leri model'in beklediği aralığın dışındaysa, IndexError oluşabilir")
                    print(f"\n   🧪 Test: no_sharding=True ile test edin")
                    print(f"   - Çalışıyorsa: Sorun sharding'de")
                    print(f"   - Hala bozuksa: Sorun vocab mapping offset'inde")
                else:
                    print(f"✅ Tokenizer ve Model vocab_size eşleşiyor: {embed_vocab_size}")
            
            lm_head = nn.Linear(embed_dim, lm_head_vocab_size, bias=False)
            lm_head.load_state_dict(router_data['lm_head_state_dict'])
            lm_head.to(device_obj)
        
        # Lazy loader oluştur
        loader = cls.__new__(cls)
        
        # KRİTİK: PyTorch modül yapısını başlat (nn.Module.__init__ çağrısı)
        nn.Module.__init__(loader)
        
        loader.tokenizer = tokenizer
        loader.num_blocks = config['num_blocks']
        loader.top_k = config['top_k']
        loader.device = device_obj
        loader.embed_dim = embed_dim
        loader.layers_per_block = config['layers_per_block']
        loader.router = router
        loader.embed_layer = embed_layer
        loader._final_norm = final_norm
        loader._lm_head = lm_head
        
        # Model tipi bilgisi (Qwen desteği için)
        loader._is_qwen = config.get('is_qwen', False)
        loader.no_sharding = False
        loader.model = None  # Lazy loading'de model RAM'de değil
        loader.layers = None  # Lazy loading'de layers yok
        
        # Sequential lazy mod: Tüm bloklar sırayla çalışır
        # Router eğitimsizken bu mod ZORUNLU (aksi halde çöp çıktı)
        loader._sequential_all = sequential_all
        if sequential_all:
            print(f"✅ Sequential lazy mod aktif: Tüm {config['num_blocks']} blok sırayla çalışacak")
            print(f"   VRAM tasarrufu: Sadece 1-2 blok aynı anda bellekte")
        
        # Rotary embeddings'i diskten yükle (Qwen modelleri için KRİTİK)
        # Ayrı dosya olarak kaydedildi: rotary_emb.pt
        loader._rotary_emb = None
        rotary_path = os.path.join(save_dir, "rotary_emb.pt")
        if os.path.exists(rotary_path):
            try:
                loader._rotary_emb = torch.load(rotary_path, map_location=device_obj, weights_only=False)
                loader._rotary_emb.eval()
                print(f"✅ Rotary embeddings diskten yüklendi: {type(loader._rotary_emb).__name__}")
            except Exception as e:
                print(f"⚠️  Rotary embeddings yüklenemedi: {e}")
        elif loader._is_qwen:
            print(f"⚠️  rotary_emb.pt bulunamadı — modeli tekrar save_blocks_to_disk ile kaydedin")
        
        # Blokları lazy yükle (şimdilik boş, gerektiğinde diskten yüklenecek)
        loader.blocks = nn.ModuleList()
        loader._block_paths = []
        loader._loaded_blocks = {}  # Cache: {block_idx: nn.Module}
        loader._lazy_load = lazy_load
        
        # Korumalı bloklar (thrashing'i önlemek için)
        loader._locked_blocks = {0, 1}
        
        # Blok kullanım takibi (cleanup için)
        loader._block_usage_count = {}
        
        # Sticky Routing
        loader._sticky_blocks = None
        loader._sticky_until_token = 0
        loader._sticky_duration = sticky_duration
        
        # Asenkron prefetching için
        loader._prefetch_queue = Queue()
        loader._prefetch_thread = None
        loader._prefetch_running = False
        loader._prefetch_lock = threading.Lock()
        
        # Blok yollarını kaydet ve save_dir'ı sakla (blok yapısını yeniden oluşturmak için)
        loader._save_dir = save_dir
        
        for i in range(config['num_blocks']):
            block_path = os.path.join(save_dir, f"block_{i}.pt")
            loader._block_paths.append(block_path)
            if lazy_load:
                loader.blocks.append(nn.Identity())  # Placeholder
            else:
                # Eager: hemen yükle (gerçek modülü torch.load ile)
                block = torch.load(block_path, map_location=device_obj, weights_only=False)
                block.eval()
                loader.blocks.append(block)
        
        loader.model = None
        loader.layers = None
        
        return loader

    def calibrate_router(
        self,
        calibration_prompts: list = None,
        num_steps: int = 100,
        lr: float = 1e-3,
    ):
        """
        Router'ı Knowledge Distillation ile eğit.
        
        Strateji:
        1. Teacher: Tüm bloklar sırayla → referans logits
        2. Ablasyon: Her top_k blok kombinasyonu test edilir, KL divergence ölçülür
        3. En iyi kombinasyon bulunur
        4. Router bu kombinasyonları seçmeyi öğrenir
        """
        import itertools
        
        if calibration_prompts is None:
            calibration_prompts = [
                "The history of artificial intelligence is",
                "In mathematics, a prime number is",
                "The capital of France is Paris, which is known for",
                "def fibonacci(n):\n    if n <= 1:\n        return n",
                "Machine learning models can be trained using",
                "The theory of relativity states that",
                "To cook a perfect pasta, you need to",
                "The human brain contains approximately",
            ]
        
        print(f"\n{'='*60}")
        print(f"🎓 ROUTER KALİBRASYONU (Knowledge Distillation)")
        print(f"{'='*60}")
        print(f"   Blok: {self.num_blocks}, top_k: {self.top_k}")
        
        all_combos = list(itertools.combinations(range(self.num_blocks), self.top_k))
        print(f"   Test edilecek kombinasyon: {len(all_combos)}")
        
        # ── ADIM 1: Teacher logits (tüm bloklar sırayla) ──
        print(f"\n📖 Adım 1: Teacher (tüm {self.num_blocks} blok)...")
        teacher_data = []
        
        for prompt in calibration_prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                x = self.embed_layer(input_ids)
                x_running = x.clone()
                for bi in range(self.num_blocks):
                    block = self._load_block_from_disk(bi) if self._lazy_load else self.blocks[bi]
                    x_running = self._run_single_block(block, x_running)
                t_logits = self._get_logits(x_running)
                teacher_data.append((x, t_logits))
        print(f"   ✅ {len(calibration_prompts)} teacher logits hazır")
        
        # ── ADIM 2: Ablasyon (her kombinasyon) ──
        print(f"\n🔬 Adım 2: {len(all_combos)} kombinasyon test ediliyor...")
        combo_scores = {}
        
        for combo in all_combos:
            total_kl = 0.0
            combo_set = set(combo)
            for x_embed, t_logits in teacher_data:
                with torch.no_grad():
                    x_running = x_embed.clone()
                    # Forward ile aynı mantık: seçilen → full, diğer → identity
                    for bi in range(self.num_blocks):
                        if bi in combo_set:
                            block = self._load_block_from_disk(bi) if self._lazy_load else self.blocks[bi]
                            x_running = self._run_single_block(block, x_running)
                        # else: identity (pass)
                    s_logits = self._get_logits(x_running)
                    t_p = torch.nn.functional.softmax(t_logits.float(), dim=-1)
                    s_lp = torch.nn.functional.log_softmax(s_logits.float(), dim=-1)
                    kl = torch.nn.functional.kl_div(s_lp, t_p, reduction='batchmean')
                    total_kl += kl.item()
            combo_scores[combo] = total_kl / len(teacher_data)
        
        sorted_combos = sorted(combo_scores.items(), key=lambda x: x[1])
        print(f"\n📊 En İyi 5 Kombinasyon:")
        for rank, (combo, kl) in enumerate(sorted_combos[:5]):
            m = ["🥇", "🥈", "🥉", "  ", "  "][rank]
            print(f"   {m} Blok {list(combo)}: KL = {kl:.4f}")
        
        best_combo = sorted_combos[0][0]
        print(f"\n   🏆 En iyi: {list(best_combo)} (KL={sorted_combos[0][1]:.4f})")
        print(f"   ❌ En kötü: {list(sorted_combos[-1][0])} (KL={sorted_combos[-1][1]:.4f})")
        
        # ── ADIM 3: Router eğitimi ──
        print(f"\n🔧 Adım 3: Router eğitimi ({num_steps} adım)...")
        
        # KESKİN hedef dağılım: En iyi 3 kombodan oluştur
        # Softmax(−KL / temperature) ile keskin ağırlıklandır
        kl_values = torch.tensor([kl for _, kl in sorted_combos], device=self.device)
        sharp_weights = torch.nn.functional.softmax(-kl_values / 5.0, dim=0)  # temperature=5 → keskin
        
        target_dist = torch.zeros(self.num_blocks, device=self.device)
        for (combo, _), weight in zip(sorted_combos, sharp_weights):
            for bi in combo:
                target_dist[bi] += weight.item()
        target_dist = target_dist / target_dist.sum()
        print(f"   Hedef dağılım: {[f'{t:.3f}' for t in target_dist.tolist()]}")
        
        # Her prompt için en iyi komboyu bul → prompt-specific hedefler
        prompt_targets = []
        for p_idx, (x_embed, t_logits) in enumerate(teacher_data):
            best_kl = float('inf')
            best_combo_for_prompt = best_combo
            for combo, kl_val in combo_scores.items():
                # Bu prompt için KL'yi ayrı hesapla
                with torch.no_grad():
                    x_r = x_embed.clone()
                    cs = set(combo)
                    for bi in range(self.num_blocks):
                        if bi in cs:
                            block = self._load_block_from_disk(bi) if self._lazy_load else self.blocks[bi]
                            x_r = self._run_single_block(block, x_r)
                    s_log = self._get_logits(x_r)
                    t_p = torch.nn.functional.softmax(t_logits.float(), dim=-1)
                    s_lp = torch.nn.functional.log_softmax(s_log.float(), dim=-1)
                    kl = torch.nn.functional.kl_div(s_lp, t_p, reduction='batchmean').item()
                    if kl < best_kl:
                        best_kl = kl
                        best_combo_for_prompt = combo
            
            # Bu prompt için keskin hedef
            pt = torch.zeros(self.num_blocks, device=self.device)
            for bi in best_combo_for_prompt:
                pt[bi] = 0.5  # Her seçilen blok %50
            prompt_targets.append(pt)
        
        # Router'ı float32'ye al ve eğit
        original_dtype = next(self.router.parameters()).dtype
        if original_dtype != torch.float32:
            self.router = self.router.float()
        self.router.train()
        
        # Yüksek lr + daha fazla adım
        optimizer = torch.optim.Adam(self.router.parameters(), lr=lr * 3)
        
        for step in range(num_steps):
            total_loss = 0.0
            for p_idx, (x_embed, _) in enumerate(teacher_data):
                probs, _, aux_loss, _ = self.router(x_embed.float(), pool_input=True)
                probs_sq = probs.squeeze()
                
                # İki loss: global hedef + prompt-specific hedef
                loss_global = torch.nn.functional.mse_loss(probs_sq, target_dist)
                loss_prompt = torch.nn.functional.mse_loss(probs_sq, prompt_targets[p_idx])
                loss = 0.3 * loss_global + 0.7 * loss_prompt + 0.001 * aux_loss
                
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (step + 1) % 20 == 0:
                print(f"   Adım {step+1}/{num_steps}: Loss = {total_loss / len(teacher_data):.6f}")
        
        self.router.eval()
        if original_dtype != torch.float32:
            self.router = self.router.to(dtype=original_dtype)
        
        # ── TEORİ ENTEGRASYONU ──
        # Router temperature'ını düşür (Parisi tavlama son aşaması)
        self.router.set_temperature(0.5)
        
        # Nash regret'i seed et: En iyi kombo bloklarına avantaj
        # Negatif regret = "bu blok yeterince seçildi" → avantaj
        # Pozitif regret = "bu blok az seçildi" → dezavantaj (bu durumda istemiyoruz)
        regret_seed = torch.zeros(self.num_blocks, device=self.device)
        usage_seed = torch.zeros(self.num_blocks, device=self.device)
        for bi in best_combo:
            regret_seed[bi] = 0.3   # Pozitif regret → logit bonusu → daha çok seçilir
            usage_seed[bi] = 0.4    # Yüksek usage → sürü hafızası
        for bi in range(self.num_blocks):
            if bi not in best_combo:
                regret_seed[bi] = -0.1  # Negatif regret → logit penalty
                usage_seed[bi] = 0.1
        self.router.cumulative_regret.copy_(regret_seed)
        self.router.block_usage.copy_(usage_seed)
        
        # Teori durumu raporu
        stats = self.router.get_stats()
        print(f"\n📐 Teori Durumu:")
        ts = stats.get('theory_status', {})
        print(f"   Nash Regret: {'✅ Aktif' if ts.get('nash_regret_active') else '❌ Pasif'}")
        print(f"   Parisi Tavlama: {'✅ Aktif' if ts.get('parisi_annealing_active') else '⏸️  Soğuma tamamlandı'}")
        print(f"   Sığırcık Keşif: {'✅ Aktif' if ts.get('starling_exploration_active') else '❌ Pasif'}")
        print(f"   Regret: {[f'{r:.3f}' for r in stats['regret'].tolist()]}")
        print(f"   Usage: {[f'{u:.3f}' for u in stats['usage'].tolist()]}")
        
        # Sonuç
        print(f"\n✅ Kalibrasyon tamamlandı!")
        test_prompt = calibration_prompts[0]
        idxs, wts = self.predict_blocks(test_prompt, prefetch=False)
        print(f"   Router seçimi: {idxs} (en iyi: {list(best_combo)})")
        print(f"   Ağırlıklar: {[f'{w:.2%}' for w in wts.tolist()]}")
        match = set(idxs) == set(best_combo)
        print(f"   {'✅ Eşleşti!' if match else '⚠️  Yakın ama tam eşleşmedi'}")
        
        # Tüm promptlar için kontrol
        match_count = 0
        for p_idx, prompt in enumerate(calibration_prompts):
            idxs_p, _ = self.predict_blocks(prompt, prefetch=False)
            pt_best = [i for i, v in enumerate(prompt_targets[p_idx]) if v > 0]
            if set(idxs_p) == set(pt_best):
                match_count += 1
        print(f"   Doğruluk: {match_count}/{len(calibration_prompts)} prompt eşleşti")
        print(f"{'='*60}\n")
    
    def _run_single_block(self, block, x):
        """Tek bir bloğu position_embeddings ile çalıştır."""
        position_embeddings = None
        L = x.shape[1]
        position_ids = torch.arange(L, device=self.device).unsqueeze(0)
        if self._is_qwen and self._rotary_emb is not None:
            try:
                cos, sin = self._rotary_emb(x, position_ids)
                position_embeddings = (cos.to(self.device), sin.to(self.device))
            except Exception:
                pass
        if isinstance(block, QwenBlockWrapper) and position_embeddings is not None:
            out = block(x, position_embeddings=position_embeddings,
                        position_ids=position_ids, attention_mask=None)
        else:
            out = block(x)
        return out[0] if isinstance(out, tuple) else out
    
    def _get_logits(self, hidden_states):
        """Hidden states'ten logits hesapla (final_norm + lm_head)."""
        # Final norm
        if self.model is not None:
            if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
                x = self.model.model.norm(hidden_states)
            else:
                x = hidden_states
            # LM head
            if hasattr(self.model, "lm_head"):
                return self.model.lm_head(x)
        else:
            # Lazy loading
            if self._final_norm is not None:
                x = self._final_norm(hidden_states)
            else:
                x = hidden_states
            if self._lm_head is not None:
                return self._lm_head(x)
        return None

    def _load_block_from_disk(self, block_idx: int) -> nn.Module:
        """
        Diskten bir bloğu yükle (Lazy Loading) ve cache'e ekle.
        KRİTİK: torch.load ile GERÇEK modül yüklenir (QwenBlockWrapper + Qwen2DecoderLayer)
        
        Args:
            block_idx: Yüklenecek blok indeksi
        
        Returns:
            Yüklenmiş blok modülü (gerçek forward pass yapar)
        """
        if block_idx in self._loaded_blocks:
            return self._loaded_blocks[block_idx]
        
        if block_idx >= len(self._block_paths):
            raise ValueError(f"Blok {block_idx} bulunamadı")
        
        block_path = self._block_paths[block_idx]
        print(f"📂 Diskten yükleniyor: block_{block_idx}.pt")
        
        # KRİTİK: Gerçek modülü yükle (state_dict değil!)
        # torch.save(module) ile kaydedildi, torch.load ile gerçek Qwen2DecoderLayer geri gelir
        block = torch.load(block_path, map_location=self.device, weights_only=False)
        
        # KRİTİK: Dtype uyumu! block.cpu() float32'ye dönüştürür ama model float16.
        # Embed layer'in dtype'ına cast et
        try:
            embed_dtype = next(self.embed_layer.parameters()).dtype
            block = block.to(dtype=embed_dtype)
        except (StopIteration, AttributeError):
            pass
        
        block.eval()
        
        # Cache'e ekle
        self._loaded_blocks[block_idx] = block
        return block

    def _unload_block_from_memory(self, block_idx: int):
        """
        RAM'den bir bloğu kaldır (bellek tasarrufu).
        Korumalı bloklar (locked blocks) kaldırılmaz.
        
        Args:
            block_idx: Kaldırılacak blok indeksi
        """
        # Korumalı blokları kontrol et (Block 0 varsayılan olarak kilitli)
        if block_idx in self._locked_blocks:
            print(f"🔒 Blok {block_idx} korumalı, RAM'den kaldırılmıyor")
            return
        
        if block_idx in self._loaded_blocks:
            del self._loaded_blocks[block_idx]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"🗑️  Blok {block_idx} RAM'den kaldırıldı")
    
    def lock_block(self, block_idx: int):
        """
        Bir bloğu korumalı yap (RAM'den kaldırılmasını engelle).
        
        Args:
            block_idx: Kilitlenecek blok indeksi
        """
        self._locked_blocks.add(block_idx)
        print(f"🔒 Blok {block_idx} kilitlendi (RAM'den kaldırılmayacak)")
    
    def unlock_block(self, block_idx: int):
        """
        Bir bloğun kilidini kaldır (RAM'den kaldırılabilir hale getir).
        
        Args:
            block_idx: Kilidi kaldırılacak blok indeksi
        """
        if block_idx in self._locked_blocks:
            self._locked_blocks.remove(block_idx)
            print(f"🔓 Blok {block_idx} kilidi kaldırıldı")

    def _prefetch_worker(self):
        """
        Arka plan thread'i: Prefetch kuyruğundaki blokları yükler.
        """
        while self._prefetch_running:
            try:
                block_idx = self._prefetch_queue.get(timeout=1.0)
                if block_idx is None:  # Shutdown signal
                    break
                
                # Blok zaten yüklü mü kontrol et
                with self._prefetch_lock:
                    if block_idx not in self._loaded_blocks:
                        self._load_block_from_disk(block_idx)
                
                self._prefetch_queue.task_done()
            except Exception:
                continue

    def start_prefetching(self, num_workers: int = 2):
        """
        Asenkron prefetching'i başlat.
        
        Args:
            num_workers: Paralel prefetch thread sayısı (default: 2)
        """
        if not hasattr(self, '_prefetch_threads'):
            self._prefetch_threads = []
        
        # Çalışan thread var mı kontrol et
        alive = [t for t in self._prefetch_threads if t.is_alive()]
        if len(alive) >= num_workers:
            return
        
        self._prefetch_running = True
        for i in range(num_workers - len(alive)):
            t = threading.Thread(target=self._prefetch_worker, daemon=True)
            t.start()
            self._prefetch_threads.append(t)
        
        print(f"🚀 Asenkron prefetching başlatıldı ({num_workers} worker)")

    def stop_prefetching(self):
        """Asenkron prefetching'i durdur."""
        self._prefetch_running = False
        # Her worker için shutdown signal
        for _ in range(len(getattr(self, '_prefetch_threads', []))):
            self._prefetch_queue.put(None)
        for t in getattr(self, '_prefetch_threads', []):
            t.join(timeout=2.0)
        self._prefetch_threads = []
        print("⏹️  Asenkron prefetching durduruldu")

    def prefetch_blocks(self, block_indices: List[int]):
        """
        Blokları önceden yükle (asenkron).
        Router'ın tahmin ettiği blokları arka planda yüklemeye başlar.
        
        Args:
            block_indices: Yüklenecek blok indeksleri listesi
        """
        if not self._lazy_load:
            return
        
        # Prefetching başlatılmamışsa başlat
        if not getattr(self, '_prefetch_threads', None) or not any(t.is_alive() for t in self._prefetch_threads):
            self.start_prefetching()
        
        # Blokları kuyruğa ekle
        for idx in block_indices:
            if idx not in self._loaded_blocks:
                self._prefetch_queue.put(idx)
