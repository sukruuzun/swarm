"""
Nash-Parisi Dil Modeli (Stratejik Sığırcıklar)
=================================================
Parisi'nin sığırcık sürüsü fiziği ile Nash'in oyun teorisini
birleştiren orijinal bir dil modeli mimarisi.

Formül:
  v_i(t+1) = Parisi(Komşu Etkileşimi) + Nash(En İyi Tepki)

Mimari:
  Token Embedding + RoPE
      ↓
  [Nash-Parisi Blok] × N katman
      │  ├─ LayerNorm
      │  ├─ StarlingAttention (Parisi: sliding window, Reynolds kuralları)
      │  ├─ Parisi Gürültü (η)
      │  ├─ Residual
      │  ├─ LayerNorm
      │  ├─ NashMoE (Nash: stratejik expert seçimi)      ← YENİ
      │  └─ Residual
      ↓
  Son LayerNorm → LM Head → Logits

Tasarruf Kaynakları:
  Attention: Parisi sliding window → O(N·w) bellek (w << N)
  FFN: Nash sparse experts → K/E aktif oran (örn. 2/8 = %25)
  Toplam: ~100-400x bellek kazancı (uzun dizilerde)

Karşılaştırma:
  Standart Transformer: O(N² · d) attention + O(d · 4d) FFN
  Nash-Parisi:          O(N·w · d) attention + O(d · 4d · K/E) FFN
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from swarm_llm.config import SwarmConfig
from swarm_llm.attention import StarlingAttention
from swarm_llm.noise import AdaptiveParisiNoise
from swarm_llm.model import RotaryPositionalEmbedding, SwiGLU
from swarm_llm.nash_moe import NashMoE, NashMoEEfficient


class NashParisiBlock(nn.Module):
    """
    Nash-Parisi Transformer Bloğu.

    Parisi Katmanı (Uyum):
      Token'lar sadece komşularıyla etkileşir.
      Reynolds kuralları ile dikkat modülasyonu.

    Nash Katmanı (Optimizasyon):
      Expert'ler stratejik olarak aktive olur.
      Nash dengesi ile yük dengeleme.

    Birleşim Metaforu:
      Sığırcık sürüsünde her kuş (token):
      1. Komşularını gözler (Parisi → StarlingAttention)
      2. Bireysel gürültü ekler (Parisi → η noise)
      3. Hangi "uzmanlık"ını kullanacağına karar verir (Nash → Expert seçimi)
    """

    def __init__(self, config: SwarmConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-Norm
        self.attn_norm = nn.LayerNorm(config.embed_dim)
        self.ffn_norm = nn.LayerNorm(config.embed_dim)

        # Parisi: Sığırcık Dikkat
        self.attention = StarlingAttention(config, causal=True)

        # Parisi: Gürültü (η)
        self.noise = AdaptiveParisiNoise(config.embed_dim, config.noise_strength)

        # Nash: Mixture of Experts (vektörize verimli versiyon)
        hidden_dim = config.embed_dim * config.ffn_multiplier
        self.nash_moe = NashMoEEfficient(
            embed_dim=config.embed_dim,
            hidden_dim=hidden_dim,
            num_experts=config.num_experts,
            top_k=config.top_k_experts,
            dropout=config.dropout,
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv=None,
    ) -> tuple[torch.Tensor, object, dict]:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            past_kv: KV önbelleği

        Returns:
            out: (batch, seq_len, embed_dim)
            new_kv: Güncellenmiş KV önbelleği
            moe_info: Nash MoE istatistikleri
        """
        # ── Parisi: Dikkat (komşu etkileşimi) ────────────────────────
        residual = x
        x = self.attn_norm(x)
        attn_out, new_kv = self.attention(x, past_kv)
        attn_out = self.noise(attn_out)
        x = residual + self.dropout(attn_out)

        # ── Nash: Expert Seçimi (stratejik optimizasyon) ──────────────
        residual = x
        x = self.ffn_norm(x)
        moe_out, moe_info = self.nash_moe(x)
        x = residual + self.dropout(moe_out)

        return x, new_kv, moe_info


class NashParisiLLM(nn.Module):
    """
    Nash-Parisi Stratejik Sığırcık Dil Modeli.

    Parisi + Nash birleşimli tam dil modeli.

    Kullanım:
        config = SwarmConfig(
            vocab_size=50257, embed_dim=256,
            num_experts=8, top_k_experts=2,
            use_nash_moe=True,
        )
        model = NashParisiLLM(config)
        logits = model(input_ids)

    Bellek Analizi (seq=2048, batch=4):
        Standart Transformer:  ~10 GB
        Swarm-LLM (Parisi):    ~200 MB  (attention tasarrufu)
        Nash-Parisi (bu model): ~80 MB  (attention + FFN tasarrufu)
    """

    def __init__(self, config: SwarmConfig):
        super().__init__()
        self.config = config

        # Token gömme
        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Nash-Parisi blokları
        self.layers = nn.ModuleList([
            NashParisiBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # Son normalizasyon
        self.final_norm = nn.LayerNorm(config.embed_dim)

        # LM Head (weight tying)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        # Ağırlık başlatma
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
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
    ) -> dict:
        """
        İleri geçiş.

        Returns:
            dict:
                'logits': (batch, seq_len, vocab_size)
                'loss': Çapraz entropi kaybı
                'past_kvs': KV önbellekleri
                'moe_info': Nash MoE istatistikleri (her katman)
        """
        B, L = input_ids.shape

        x = self.tok_emb(input_ids)
        x = self.emb_dropout(x)

        if past_kvs is None:
            past_kvs = [None] * self.config.num_layers

        new_kvs = []
        all_moe_info = []

        for i, layer in enumerate(self.layers):
            x, kv, moe_info = layer(x, past_kvs[i])
            new_kvs.append(kv)
            all_moe_info.append(moe_info)

        x = self.final_norm(x)
        logits = self.lm_head(x)

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
            'moe_info': all_moe_info,
        }

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embedding = sum(p.numel() for p in self.tok_emb.parameters())
        attention = sum(
            p.numel() for layer in self.layers
            for name, p in layer.named_parameters() if 'attention' in name
        )
        expert = sum(
            p.numel() for layer in self.layers
            for name, p in layer.named_parameters() if 'expert' in name
        )
        router = sum(
            p.numel() for layer in self.layers
            for name, p in layer.named_parameters() if 'router' in name
        )
        return {
            'total': total,
            'trainable': trainable,
            'embedding': embedding,
            'attention (Parisi)': attention,
            'experts (Nash)': expert,
            'router (Nash)': router,
        }

    def get_nash_stats(self) -> dict:
        """Tüm katmanlardaki Nash istatistiklerini topla."""
        stats = {
            'expert_balance': [],
            'temperatures': [],
            'regrets': [],
        }
        for i, layer in enumerate(self.layers):
            router = layer.nash_moe.router
            info = router._compute_stats(
                torch.zeros(1), torch.zeros(1, 1, self.config.top_k_experts, dtype=torch.long)
            )
            stats['expert_balance'].append(info['balance_score'])
            stats['temperatures'].append(info['temperature'])
            stats['regrets'].append(info['regret'].tolist())
        return stats

    def estimate_vram(self, seq_len: int, batch_size: int = 1) -> Dict[str, str]:
        """Standart vs Swarm vs Nash-Parisi VRAM karşılaştırması."""
        d = self.config.embed_dim
        h = self.config.num_heads
        n_layers = self.config.num_layers
        w = self.config.neighbor_size
        E = self.config.num_experts
        K = self.config.top_k_experts

        param_mem = sum(p.numel() * 4 for p in self.parameters())

        # Dikkat matrisi belleği
        standard_attn = batch_size * h * seq_len * seq_len * 4 * n_layers
        swarm_attn = batch_size * h * seq_len * w * 4 * n_layers

        # FFN belleği
        ffn_dim = d * self.config.ffn_multiplier
        standard_ffn = batch_size * seq_len * ffn_dim * 4 * n_layers * 2  # w1+w2
        # Nash: sadece K/E oranında aktif
        nash_ffn = standard_ffn * K / E

        def _fmt(b):
            if b > 1e9: return f"{b/1e9:.2f} GB"
            if b > 1e6: return f"{b/1e6:.2f} MB"
            return f"{b/1e3:.1f} KB"

        total_standard = param_mem + standard_attn + standard_ffn
        total_swarm = param_mem + swarm_attn + standard_ffn
        total_nash_parisi = param_mem + swarm_attn + nash_ffn

        return {
            'parameters': _fmt(param_mem),
            '--- Standart Transformer ---': '',
            'std_attention': _fmt(standard_attn),
            'std_ffn': _fmt(standard_ffn),
            'std_total': _fmt(total_standard),
            '--- Swarm-LLM (Parisi) ---': '',
            'swarm_attention': _fmt(swarm_attn),
            'swarm_ffn': _fmt(standard_ffn),
            'swarm_total': _fmt(total_swarm),
            '--- Nash-Parisi ---': '',
            'nash_attention': _fmt(swarm_attn),
            'nash_ffn': _fmt(nash_ffn),
            'nash_total': _fmt(total_nash_parisi),
            '--- Kazançlar ---': '',
            'attn_savings': f"{standard_attn / max(swarm_attn, 1):.0f}x",
            'ffn_savings': f"{standard_ffn / max(nash_ffn, 1):.0f}x",
            'total_savings': f"{total_standard / max(total_nash_parisi, 1):.1f}x",
        }
