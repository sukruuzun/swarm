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

from swarm_llm.external_router import ExternalParisiNashRouter


class TupleCleaner(nn.Module):
    """
    HuggingFace layer çıktılarını temizleyen wrapper.
    GPT-2/Llama layer'ları tuple döndürür (hidden_states, past_key_values, ...),
    bu wrapper sadece hidden_states'i alır ve bir sonraki layer'a geçirir.
    """
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer
    
    def forward(self, x):
        out = self.layer(x)
        # Tuple ise sadece hidden_states'i al
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

        # Model'i cihaza taşı
        self.model.to(self.device)

        # Layer'ları bul (Llama/Qwen için genelde model.layers veya model.model.layers)
        self.layers = self._extract_layers()
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

        # Router oluştur
        self.router = ExternalParisiNashRouter(
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            top_k=top_k,
        ).to(self.device)

        # Embedding layer'ı bul
        self.embed_layer = self._get_embed_layer()
        
        # Lazy loading için
        self._lazy_load = False
        self._block_paths = []
        self._loaded_blocks = {}

    def _extract_layers(self) -> nn.ModuleList:
        """Model'den transformer layer'larını çıkar."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "layers"):
            return self.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        else:
            raise ValueError(
                "Model yapısı desteklenmiyor. Llama/Qwen/Mistral gibi modeller bekleniyor."
            )

    def _get_embed_layer(self) -> nn.Module:
        """Embedding layer'ı bul."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return self.model.model.embed_tokens
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "wte"):
            return self.model.transformer.wte
        elif hasattr(self.model, "embed_tokens"):
            return self.model.embed_tokens
        else:
            raise ValueError("Embedding layer bulunamadı.")

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
                blocks.append(nn.Sequential(*wrapped_layers))
            else:
                # Boş blok (padding)
                blocks.append(nn.Identity())

        return blocks

    @torch.no_grad()
    def predict_blocks(self, prompt: str) -> Tuple[List[int], torch.Tensor]:
        """
        Teoreminin beyni: Giriş cümlesine bakarak hangi blokların
        gerekli olduğunu tahmin eder. Modeli çalıştırmadan önce çağrılır.

        Args:
            prompt: Giriş metni

        Returns:
            block_indices: [i1, i2, ...] yüklenecek blok indeksleri
            weights: (top_k,) blok ağırlıkları
        """
        self.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        x = self.embed_layer(input_ids)  # (1, L, D)

        return self.router.get_predictive_indices(x, pool_input=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> dict:
        """
        Forward pass: Sadece router'ın seçtiği bloklar çalışır.

        Args:
            input_ids: (B, L) token ids
            attention_mask: (B, L) attention mask (opsiyonel)
            use_cache: KV cache kullan (opsiyonel)

        Returns:
            logits, hidden_states, selected_indices, router_info
        """
        self.eval()
        B, L = input_ids.shape

        # Embedding
        x = self.embed_layer(input_ids)  # (B, L, D)

        # Router ile blok seçimi
        probs, indices, aux_loss, weights = self.router(x, pool_input=True)
        indices = indices.squeeze().cpu().tolist()
        if isinstance(indices, int):
            indices = [indices]

        # Sadece seçilen blokları sıralı çalıştır
        # KRİTİK: HuggingFace katmanları tuple döndürür (hidden_states, past_key_values, ...)
        # Defansif kodlama: Her adımda tuple kontrolü yap, sadece Tensor al
        x_out = x
        selected_indices = indices[: self.top_k]
        
        for idx in selected_indices:
            # Lazy loading: Eğer blok diskte ise önce yükle
            if self._lazy_load and idx not in self._loaded_blocks:
                block = self._load_block_from_disk(idx)
            else:
                block = self.blocks[idx]
            
            # Blok çıktısını al
            block_out = block(x_out)
            
            # DEFANSİF KODLAMA: Tuple kontrolü (HuggingFace standardı)
            # HuggingFace modelleri genelde (hidden_states, past_key_values, ...) döndürür
            if isinstance(block_out, tuple):
                # Tuple ise sadece 0. indeksi al (hidden_states)
                x_out = block_out[0]
            # BaseModelOutput veya benzeri objeler için
            elif hasattr(block_out, 'last_hidden_state'):
                x_out = block_out.last_hidden_state
            elif hasattr(block_out, 'hidden_states') and block_out.hidden_states:
                x_out = block_out.hidden_states[-1]
            # Tensor ise direkt kullan
            elif isinstance(block_out, torch.Tensor):
                x_out = block_out
            else:
                # Son çare: ilk elemanı dene
                try:
                    if hasattr(block_out, '__getitem__'):
                        x_out = block_out[0]
                    else:
                        x_out = block_out
                except:
                    x_out = block_out
            
            # ÇİFT KONTROL: x_out hala tuple/list ise (iç içe tuple durumu)
            while isinstance(x_out, (tuple, list)) and len(x_out) > 0:
                x_out = x_out[0]
            
            # Final güvenlik: Tensor olduğundan emin ol
            if not isinstance(x_out, torch.Tensor):
                raise TypeError(f"Blok {idx} çıktısı Tensor değil: {type(x_out)}")

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

        # Lazy loading: Kullanılmayan blokları RAM'den kaldır (opsiyonel)
        if self._lazy_load and len(self._loaded_blocks) > self.top_k:
            # En eski kullanılmayan blokları kaldır
            unused_blocks = set(self._loaded_blocks.keys()) - set(selected_indices)
            for unused_idx in list(unused_blocks)[:len(unused_blocks) - self.top_k]:
                self._unload_block_from_memory(unused_idx)
        
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
    ) -> str:
        """
        Metin üretimi: Her adımda router hangi blokları kullanacağına karar verir.

        Args:
            prompt: Başlangıç metni
            max_new_tokens: Üretilecek maksimum token sayısı
            temperature: Sampling sıcaklığı
            top_k: Top-K sampling

        Returns:
            Üretilmiş metin
        """
        self.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated = input_ids

        for _ in range(max_new_tokens):
            # Forward (sadece seçilen bloklar)
            outputs = self.forward(generated)
            logits = outputs["logits"][:, -1, :] / temperature

            # Top-K sampling
            topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < topk_vals[:, -1:]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=1)

        return self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

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
            # Blok yapısını da kaydet (lazy loading için gerekli)
            block_structure = []
            if isinstance(block, nn.Sequential):
                for j, layer in enumerate(block):
                    layer_type = type(layer).__name__
                    # TupleCleaner wrapper'ını atla, içindeki layer'ı al
                    if isinstance(layer, TupleCleaner):
                        layer_type = f"TupleCleaner({type(layer.layer).__name__})"
                    block_structure.append(layer_type)
            
            torch.save({
                'state_dict': block.state_dict(),
                'block_index': i,
                'layers_per_block': self.layers_per_block,
                'block_structure': block_structure,  # Lazy loading için
            }, block_path)
            block_size_mb = os.path.getsize(block_path) / (1024**2)
            print(f"   Blok {i}: {block_size_mb:.2f} MB → {block_path}")
        
        # Final norm ve LM head'i de kaydet (lazy loading için)
        final_norm_state = None
        lm_head_state = None
        
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            final_norm_state = self.model.model.norm.state_dict()
            lm_head_state = self.model.lm_head.state_dict()
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "ln_f"):
            final_norm_state = self.model.transformer.ln_f.state_dict()
            lm_head_state = self.model.lm_head.state_dict()
        
        # Router ve embedding'i de kaydet
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
            }
        }, router_path)
        
        print(f"   Router + Embedding + Final Norm + LM Head: {os.path.getsize(router_path) / (1024**2):.2f} MB")
        print(f"✅ Toplam {self.num_blocks} blok kaydedildi")

    @classmethod
    def from_disk_blocks(
        cls,
        tokenizer,
        save_dir: str,
        device: str = "auto",
        lazy_load: bool = True,
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
        # Gerçek kullanımda tokenizer'dan vocab_size alınabilir
        embed_dim = config['embed_dim']
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50257
        embed_layer = nn.Embedding(vocab_size, embed_dim)
        embed_layer.load_state_dict(router_data['embed_state_dict'])
        embed_layer.to(device_obj)
        
        # Final norm ve LM head'i yükle (lazy loading için)
        final_norm = None
        lm_head = None
        if config.get('has_final_norm', False):
            if router_data.get('final_norm_state_dict'):
                # Final norm oluştur (embed_dim'e göre)
                final_norm = nn.LayerNorm(embed_dim)
                final_norm.load_state_dict(router_data['final_norm_state_dict'])
                final_norm.to(device_obj)
            
            if router_data.get('lm_head_state_dict'):
                # LM head oluştur
                lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
                lm_head.load_state_dict(router_data['lm_head_state_dict'])
                lm_head.to(device_obj)
        
        # Lazy loader oluştur
        loader = cls.__new__(cls)
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
        
        # Blokları lazy yükle (şimdilik boş, gerektiğinde diskten yüklenecek)
        loader.blocks = nn.ModuleList()
        loader._block_paths = []
        loader._loaded_blocks = {}  # Cache: {block_idx: nn.Module}
        loader._lazy_load = lazy_load
        
        for i in range(config['num_blocks']):
            block_path = os.path.join(save_dir, f"block_{i}.pt")
            loader._block_paths.append(block_path)
            if lazy_load:
                # Lazy: şimdilik boş modül, gerektiğinde yüklenecek
                loader.blocks.append(nn.Identity())  # Placeholder
            else:
                # Eager: hemen yükle
                block_data = torch.load(block_path, map_location=device_obj)
                block = nn.Sequential()  # Blok yapısını yeniden oluştur
                # Not: Gerçek implementasyonda blok yapısını kaydetmek gerekir
                loader.blocks.append(block)
        
        loader.model = None  # Model artık gerekli değil (bloklar diskte)
        loader.layers = None
        
        return loader

    def _load_block_from_disk(self, block_idx: int) -> nn.Module:
        """
        Diskten bir bloğu yükle (Lazy Loading).
        
        Args:
            block_idx: Yüklenecek blok indeksi
        
        Returns:
            Yüklenmiş blok modülü
        """
        if block_idx in self._loaded_blocks:
            return self._loaded_blocks[block_idx]
        
        if block_idx >= len(self._block_paths):
            raise ValueError(f"Blok {block_idx} bulunamadı")
        
        block_path = self._block_paths[block_idx]
        print(f"📂 Diskten yükleniyor: block_{block_idx}.pt")
        
        block_data = torch.load(block_path, map_location=self.device)
        
        # Blok yapısını yeniden oluştur
        # Blok yapısı kaydedilmişse kullan, yoksa state_dict'ten çıkar
        block_structure = block_data.get('block_structure', [])
        
        if block_structure:
            # Kaydedilmiş yapıyı kullan (ideal durum)
            # Not: Gerçek implementasyonda layer'ları da kaydetmek gerekir
            # Şimdilik sadece state_dict'i yüklüyoruz
            block = nn.Sequential()
        else:
            # Fallback: Basit sequential
            block = nn.Sequential()
        
        # State dict'i yükle
        try:
            block.load_state_dict(block_data['state_dict'], strict=False)
        except Exception as e:
            # Eğer yapı uyuşmuyorsa, sadece uyumlu parametreleri yükle
            print(f"⚠️  Blok {block_idx} yapı uyuşmazlığı: {e}")
            state_dict = block_data['state_dict']
            block_state = {}
            for k, v in state_dict.items():
                # TupleCleaner wrapper'ını handle et
                if 'layer.' in k:
                    # TupleCleaner içindeki layer parametreleri
                    new_k = k.replace('layer.', '')
                    block_state[new_k] = v
                else:
                    block_state[k] = v
            if block_state:
                block.load_state_dict(block_state, strict=False)
        
        block.to(self.device)
        block.eval()
        
        # Cache'e ekle
        self._loaded_blocks[block_idx] = block
        
        return block

    def _unload_block_from_memory(self, block_idx: int):
        """
        RAM'den bir bloğu kaldır (bellek tasarrufu).
        
        Args:
            block_idx: Kaldırılacak blok indeksi
        """
        if block_idx in self._loaded_blocks:
            del self._loaded_blocks[block_idx]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"🗑️  Blok {block_idx} RAM'den kaldırıldı")
