"""
Swarm-LLM: Sığırcık Dil Modeli
=================================
Parisi'nin sığırcık sürüsü modelini temel alan tam dil modeli.

Mimari:
  Token Embedding + Pozisyonel Kodlama
      ↓
  [Sığırcık Blok] × N katman
      │  ├─ LayerNorm
      │  ├─ StarlingAttention (sliding window)
      │  ├─ Parisi Gürültü (η)
      │  ├─ Residual Bağlantı
      │  ├─ LayerNorm
      │  ├─ Feed-Forward Network (SwiGLU)
      │  └─ Residual Bağlantı
      ↓
  Son LayerNorm → Lineer Projeksiyon → Logits

GPU Bellek Tasarrufu:
  Standart Transformer: O(N²·d) bellek
  Swarm-LLM:            O(N·w·d) bellek  (w << N)

  Örnek: N=10000, w=7, d=512
    Standart: 10000² × 512 ≈ 51.2 GB
    Swarm:    10000 × 7 × 512 ≈ 35.8 MB  (~1430x tasarruf)
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from swarm_llm.config import SwarmConfig
from swarm_llm.attention import StarlingAttention, MultiScaleStarlingAttention
from swarm_llm.noise import ParisiNoise, AdaptiveParisiNoise


class RotaryPositionalEmbedding(nn.Module):
    """
    Döner Pozisyonel Kodlama (RoPE).

    Sabit sinüzoidal yerine döner matrisler kullanır.
    Avantajları:
    - Göreceli pozisyon bilgisi doğal olarak dikkat skorlarına enjekte edilir
    - Extrapolation yeteneği (eğitimden uzun dizileri destekler)
    - Modern LLM'lerde (LLaMA, Mistral) standart yaklaşım
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        # θ_i = base^(-2i/d)  frekanslari
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Pozisyon önbelleği
        self._build_cache(max_seq_len)

    def _build_cache(self, max_len: int):
        t = torch.arange(max_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (max_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_len, dim)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0):
        """RoPE uygula. x: (batch, heads, seq_len, head_dim)"""
        cos = self.cos_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)

        # Döndürme: [x1, x2] -> [x1·cos - x2·sin, x1·sin + x2·cos]
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat([
            x1 * cos[..., :x1.shape[-1]] - x2 * sin[..., :x2.shape[-1]],
            x1 * sin[..., :x1.shape[-1]] + x2 * cos[..., :x2.shape[-1]],
        ], dim=-1)
        return rotated


class SwiGLU(nn.Module):
    """
    SwiGLU Aktivasyonu ile Feed-Forward Network.

    FFN(x) = (Swish(xW₁) ⊙ xV) W₂

    Standard ReLU FFN'den daha iyi performans gösterir.
    LLaMA ve PaLM gibi modern modellerde kullanılır.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.v(x)))


class SwarmBlock(nn.Module):
    """
    Sığırcık Transformer Bloğu.

    Bir sürü "formasyonu" gibi çalışır:
    1. Her kuş (token) komşularını gözlemler (StarlingAttention)
    2. Bireysel rasgele hareket ekler (ParisiNoise)
    3. Bilgiyi işler ve günceller (FFN)
    4. Önceki durumunu korur (Residual bağlantı)
    """

    def __init__(self, config: SwarmConfig, layer_idx: int, causal: bool = True):
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-Norm (RMSNorm yerine LayerNorm -- basitlik için)
        self.attn_norm = nn.LayerNorm(config.embed_dim)
        self.ffn_norm = nn.LayerNorm(config.embed_dim)

        # Dikkat mekanizması seçimi
        if config.use_multi_scale and layer_idx % 2 == 0:
            # Çift katmanlar: çoklu ölçekli dikkat (geniş bağlam)
            self.attention = MultiScaleStarlingAttention(config, causal)
            self._is_multi_scale = True
        else:
            # Tek katmanlar: standart sığırcık dikkat (verimli)
            self.attention = StarlingAttention(config, causal)
            self._is_multi_scale = False

        # Parisi gürültüsü (uyarlanabilir)
        self.noise = AdaptiveParisiNoise(config.embed_dim, config.noise_strength)

        # Feed-Forward Network (SwiGLU)
        hidden_dim = config.embed_dim * config.ffn_multiplier
        self.ffn = SwiGLU(config.embed_dim, hidden_dim, config.dropout)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv=None,
    ) -> tuple[torch.Tensor, object]:
        """
        İleri geçiş.

        Args:
            x: (batch, seq_len, embed_dim)
            past_kv: KV önbelleği

        Returns:
            out: (batch, seq_len, embed_dim)
            new_kv: Güncellenmiş KV önbelleği
        """
        # --- Dikkat Alt-Katmanı ---
        residual = x
        x = self.attn_norm(x)
        attn_out, new_kv = self.attention(x, past_kv)

        # Parisi gürültüsü ekle
        attn_out = self.noise(attn_out)

        x = residual + self.dropout(attn_out)

        # --- FFN Alt-Katmanı ---
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.dropout(self.ffn(x))

        return x, new_kv


class SwarmLLM(nn.Module):
    """
    Sürü Zekalı Dil Modeli (Swarm-LLM).

    Parisi'nin sığırcık sürüsü algoritmasını dil modellemesine
    uygulayan tam bir transformer modeli.

    Temel Yenilikler:
    1. O(N·w) dikkat karmaşıklığı (w=7 komşu)
    2. Reynolds kuralları ile dikkat modülasyonu
    3. Parisi gürültüsü (η) ile kontrollü yaratıcılık
    4. Çoklu ölçekli pencereler (7/21/63) ile hiyerarşik bağlam

    Kullanım:
        config = SwarmConfig(vocab_size=32000, embed_dim=512)
        model = SwarmLLM(config)

        # Eğitim
        input_ids = torch.randint(0, 32000, (2, 128))
        logits = model(input_ids)
        # logits shape: (2, 128, 32000)

        # Metin üretimi
        generated = model.generate(prompt_ids, max_new_tokens=100)
    """

    def __init__(self, config: SwarmConfig):
        super().__init__()
        self.config = config

        # Token gömme (embedding)
        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_dim)

        # Pozisyonel kodlama (RoPE)
        self.rope = RotaryPositionalEmbedding(
            config.head_dim, config.max_seq_len
        )

        # Gömme dropout'u
        self.emb_dropout = nn.Dropout(config.dropout)

        # Sığırcık blokları (sürü katmanları)
        self.layers = nn.ModuleList([
            SwarmBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # Son normalizasyon
        self.final_norm = nn.LayerNorm(config.embed_dim)

        # Dil modeli başlığı (LM Head)
        # Ağırlık paylaşımı (weight tying) -- gömme ile aynı ağırlıklar
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # Weight tying

        # Ağırlık başlatma
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Xavier/He başlatma."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_kvs: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        İleri geçiş.

        Args:
            input_ids: (batch, seq_len) token ID'leri
            targets: (batch, seq_len) hedef token ID'leri (eğitim için)
            past_kvs: KV önbellekleri listesi (üretim için)

        Returns:
            dict:
                'logits': (batch, seq_len, vocab_size) ham çıktı skorları
                'loss': Çapraz entropi kaybı (targets verilmişse)
                'past_kvs': Güncellenmiş KV önbellekleri
        """
        B, L = input_ids.shape

        # Token gömme
        x = self.tok_emb(input_ids)  # (B, L, D)
        x = self.emb_dropout(x)

        # Sığırcık blokları
        if past_kvs is None:
            past_kvs = [None] * self.config.num_layers

        new_kvs = []
        for i, layer in enumerate(self.layers):
            x, kv = layer(x, past_kvs[i])
            new_kvs.append(kv)

        # Son normalizasyon
        x = self.final_norm(x)

        # Logits
        logits = self.lm_head(x)  # (B, L, vocab_size)

        # Kayıp hesapla (eğitim modunda)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        return {
            'logits': logits,
            'loss': loss,
            'past_kvs': new_kvs,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        noise_boost: float = 0.0,
    ) -> torch.Tensor:
        """
        Otoregresif metin üretimi.

        Sığırcık sürüsü benzetmesi:
        - temperature: Sürünün genel "enerji" seviyesi
        - top_k: Her kuşun gözlemleyebileceği maksimum komşu sayısı
        - top_p: Sürü yoğunluk eşiği (nucleus sampling)
        - noise_boost: Ekstra Parisi gürültüsü (yaratıcılık artırıcı)

        Args:
            input_ids: (batch, seq_len) başlangıç token'ları
            max_new_tokens: Üretilecek maksimum token sayısı
            temperature: Sıcaklık parametresi
            top_k: Top-K örnekleme
            top_p: Nucleus (Top-P) örnekleme
            noise_boost: Üretim sırasında ekstra gürültü

        Returns:
            (batch, seq_len + max_new_tokens) üretilmiş token dizisi
        """
        self.eval()

        # Gürültü artırma (yaratıcılık modu)
        if noise_boost > 0:
            for layer in self.layers:
                layer.noise.base_noise.force_noise = True

        generated = input_ids
        past_kvs = None

        for _ in range(max_new_tokens):
            # KV önbelleği varsa sadece son token'ı ver
            if past_kvs is not None:
                curr_input = generated[:, -1:]
            else:
                curr_input = generated

            # İleri geçiş
            outputs = self.forward(curr_input, past_kvs=past_kvs)
            logits = outputs['logits'][:, -1, :]  # Son token'ın logits'i
            past_kvs = outputs['past_kvs']

            # Sıcaklık ölçekleme
            if temperature != 1.0:
                logits = logits / temperature

            # Ekstra gürültü (Parisi η boost)
            if noise_boost > 0:
                logits = logits + noise_boost * torch.randn_like(logits)

            # Top-K filtreleme
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_vals[:, -1:]] = float('-inf')

            # Top-P (Nucleus) filtreleme
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Kümülatif olasılık eşiğini aşanları kaldır
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float('-inf')

                # Orijinal sıraya geri döndür
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            # Örnekleme
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

        # Gürültü modunu kapat
        if noise_boost > 0:
            for layer in self.layers:
                layer.noise.base_noise.force_noise = False

        return generated

    def count_parameters(self) -> Dict[str, int]:
        """Model parametre sayılarını döndürür."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embedding = sum(p.numel() for p in self.tok_emb.parameters())
        attention = sum(
            p.numel()
            for layer in self.layers
            for name, p in layer.named_parameters()
            if 'attention' in name
        )
        return {
            'total': total,
            'trainable': trainable,
            'embedding': embedding,
            'attention': attention,
            'other': total - embedding - attention,
        }

    def estimate_vram(self, seq_len: int, batch_size: int = 1) -> Dict[str, str]:
        """
        Tahmini VRAM kullanımını hesaplar.

        Standart Transformer vs Swarm-LLM karşılaştırması.
        """
        d = self.config.embed_dim
        h = self.config.num_heads
        n_layers = self.config.num_layers
        w = self.config.neighbor_size

        # Parametre belleği (4 byte per float32)
        param_mem = sum(p.numel() * 4 for p in self.parameters())

        # Dikkat matrisi belleği
        # Standart: B × H × L × L × 4 bytes × N_layers
        standard_attn = batch_size * h * seq_len * seq_len * 4 * n_layers
        # Swarm: B × H × L × w × 4 bytes × N_layers
        swarm_attn = batch_size * h * seq_len * w * 4 * n_layers

        def _fmt(b):
            if b > 1e9:
                return f"{b / 1e9:.2f} GB"
            if b > 1e6:
                return f"{b / 1e6:.2f} MB"
            return f"{b / 1e3:.2f} KB"

        savings = standard_attn / max(swarm_attn, 1)

        return {
            'parameters': _fmt(param_mem),
            'standard_attention': _fmt(standard_attn),
            'swarm_attention': _fmt(swarm_attn),
            'attention_savings': f"{savings:.0f}x",
            'total_standard': _fmt(param_mem + standard_attn),
            'total_swarm': _fmt(param_mem + swarm_attn),
        }
