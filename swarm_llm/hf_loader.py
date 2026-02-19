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
        """Layer'ları bloklara böl."""
        blocks = nn.ModuleList()
        total_layers = len(self.layers)

        for i in range(self.num_blocks):
            start_idx = i * self.layers_per_block
            end_idx = min((i + 1) * self.layers_per_block, total_layers)
            if start_idx < total_layers:
                block_layers = self.layers[start_idx:end_idx]
                blocks.append(nn.Sequential(*block_layers))
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
        # Not: Transformer blokları tuple döndürebilir, sadece hidden_states'i al
        x_out = x
        selected_indices = indices[: self.top_k]
        
        for idx in selected_indices:
            block_out = self.blocks[idx](x_out)
            # Tuple ise sadece ilk elemanı al (hidden_states)
            if isinstance(block_out, tuple):
                block_out = block_out[0]
            x_out = block_out

        # Final norm ve LM head (model'e göre değişir)
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            x_out = self.model.model.norm(x_out)
            logits = self.model.lm_head(x_out)
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "ln_f"):
            x_out = self.model.transformer.ln_f(x_out)
            logits = self.model.lm_head(x_out)
        else:
            # Basit fallback
            logits = self.model.lm_head(x_out) if hasattr(self.model, "lm_head") else None

        # Router weights'i hazırla
        weights_cpu = weights.squeeze().cpu()
        if weights_cpu.dim() == 0:
            weights_cpu = weights_cpu.unsqueeze(0)
        weights_list = weights_cpu[: len(selected_indices)].tolist()

        return {
            "logits": logits,
            "hidden_states": x_out,
            "selected_indices": selected_indices,
            "router_weights": weights_list,
            "router_info": self.router.get_stats(),
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
