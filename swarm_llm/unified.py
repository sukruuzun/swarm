"""
Birleşik Parisi-Nash Dikkat Mekanizması v5
============================================
Keskin Uzmanlaşma ve Uzun Koşu.

v5 İyileştirmeleri (v4 üzerine):
  1. MLP Router: Linear → LayerNorm + Linear(384→64) + GELU + Linear(64→4)
     Daha derin muhakeme, token türüne göre akıllı yönlendirme
  2. lb_coeff = 0.0001: Mikro denge koruma, agresif uzmanlaşmaya izin
  3. 7500 step + min LR 5e-5: Uzun koşu, kuyrukta öğrenme devam eder
  4. Decoupled gradient clipping: Router 0.5, Expert 1.0
  5. Temperature schedule 7500 step'e uzatılmış

Formül:
  v_i(t+1) = p_k(x_i) · φ_k(Σ_{j∈N(i)} α_ij^k · V_k_j) + η·ε

  p_k = softmax(MLP_router(x_i) / T)[k]  ← MLP differentiable!
  k = argmax(MLP_router(x_i))             ← hard seçim (forward)

Evrim:
  v1: Expert collapse        → v2: Load balancing eklendi
  v2: Aşırı denge (T takıldı) → v3: Temperature annealing
  v3: Gradyan körlüğü (%25)  → v4: Differentiable routing
  v4: Sınırlı uzmanlaşma     → v5: MLP router + düşük LB + uzun koşu
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from swarm_llm.config import SwarmConfig


# ── Sliding Window Mask ──────────────────────────────────────────────────────

def _build_sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: torch.device,
    causal: bool = True,
) -> torch.Tensor:
    """Vektörize sliding window maskesi. True = maskelenmeli."""
    positions = torch.arange(seq_len, device=device)
    distance = positions.unsqueeze(1) - positions.unsqueeze(0)
    if causal:
        mask = (distance < 0) | (distance >= window_size)
    else:
        half_w = window_size // 2
        mask = distance.abs() > half_w
    return mask


# ── Expert Attention Head ────────────────────────────────────────────────────

class ExpertAttentionHead(nn.Module):
    """
    Tam donanımlı Expert-Attention birimi.
    Her expert kendi Q_k, K_k, V_k, φ_k'sine sahip.
    """

    def __init__(self, embed_dim: int, num_heads: int, head_dim: int,
                 expand_dim: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = math.sqrt(head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.w1 = nn.Linear(embed_dim, expand_dim, bias=False)
        self.v_gate = nn.Linear(embed_dim, expand_dim, bias=False)
        self.w2 = nn.Linear(expand_dim, embed_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                reynolds_fn, attn_dropout: nn.Dropout) -> torch.Tensor:
        """Expert-spesifik dikkat + SwiGLU. (B, L, D) → (B, L, D)"""
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        scores = reynolds_fn(scores)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)

        return self.drop(self.w2(F.silu(self.w1(attn_out)) * self.v_gate(attn_out)))


# ── Nash Router v5 (MLP) ─────────────────────────────────────────────────────

class NashExpertRouter(nn.Module):
    """
    Nash Dengesi tabanlı expert router v5 -- MLP Differentiable.

    v5 farkı (v4 üzerine):
      - Linear(384→4) yerine MLP: LayerNorm → Linear(384→64) → GELU → Linear(64→4)
      - Daha derin muhakeme: token türüne göre akıllı routing
      - +12K parametre ama routing kalitesini katlıyor
      - LayerNorm sayesinde katmanlar arası tutarlı routing
    """

    def __init__(self, embed_dim: int, num_experts: int, top_k: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        router_hidden = 64
        self.gate = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, router_hidden, bias=False),
            nn.GELU(),
            nn.Linear(router_hidden, num_experts, bias=False),
        )

        self.register_buffer('current_temperature', torch.tensor(2.0))
        self.register_buffer('cumulative_regret', torch.zeros(num_experts))
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))

    def set_temperature(self, t: float):
        self.current_temperature.fill_(t)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            raw_probs: (B, L, E) tam softmax olasılıkları (differentiable!)
            indices:   (B, L, top_k) seçilen expert indeksleri
            aux_loss:  Load balancing loss
            probs:     (B, L, E) -- gradyan hesabı için
        """
        T = self.current_temperature.clamp(min=0.1)
        logits = self.gate(x) / T
        probs = F.softmax(logits, dim=-1)  # (B, L, E) -- differentiable

        # Hard seçim (forward için)
        _, indices = torch.topk(probs, self.top_k, dim=-1)  # (B, L, top_k)

        # ── Load Balancing Loss ──
        with torch.no_grad():
            one_hot = F.one_hot(indices, self.num_experts).float()
            f = one_hot.sum(dim=(0, 1, 2)) / (indices.shape[0] * indices.shape[1] * self.top_k)

        P = probs.mean(dim=(0, 1))
        aux_loss = self.num_experts * (f * P).sum()

        # İzleme
        if self.training:
            self.step_count += 1
            with torch.no_grad():
                self.expert_usage = 0.99 * self.expert_usage + 0.01 * f
                uniform = torch.ones_like(f) / self.num_experts
                self.cumulative_regret = 0.99 * self.cumulative_regret + (uniform - f)

        return probs, indices, aux_loss, probs


# ── Birleşik Parisi-Nash Dikkat v4 ───────────────────────────────────────────

class UnifiedParisiNashAttention(nn.Module):
    """
    Birleşik Parisi-Nash v5: MLP Router + Agresif Uzmanlaşma.

    TEK FORMÜL:
    v_i = p_k(x_i) · φ_k(Σ_{j∈N(i)} α_ij^k · V_k_j) + η·ε

    v5 farkları:
      - MLP router ile daha akıllı yönlendirme
      - Düşük LB (0.0001) ile serbest uzmanlaşma
      - Decoupled gradient clipping desteği
    """

    def __init__(self, config: SwarmConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.neighbor_size = config.neighbor_size

        expand_dim = config.embed_dim * config.ffn_multiplier
        self.expert_heads = nn.ModuleList([
            ExpertAttentionHead(
                config.embed_dim, config.num_heads, config.head_dim,
                expand_dim, config.dropout,
            )
            for _ in range(config.num_experts)
        ])

        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.router = NashExpertRouter(config.embed_dim, config.num_experts, config.top_k_experts)

        self.separation_gate = nn.Parameter(torch.tensor(config.separation_weight))
        self.alignment_gate = nn.Parameter(torch.tensor(config.alignment_weight))
        self.cohesion_gate = nn.Parameter(torch.tensor(config.cohesion_weight))

        self.noise_eta = nn.Parameter(torch.full((config.embed_dim,), config.noise_strength))
        self.noise_gate = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.Sigmoid(),
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self._cached_mask = None
        self._cached_len = 0

    def _get_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self._cached_mask is not None and self._cached_len == seq_len:
            return self._cached_mask.to(device)
        mask = _build_sliding_window_mask(seq_len, self.neighbor_size, device, causal=True)
        self._cached_mask = mask
        self._cached_len = seq_len
        return mask

    def _reynolds_modulate(self, scores: torch.Tensor) -> torch.Tensor:
        similarity = torch.sigmoid(scores)
        separation = -self.separation_gate * (similarity ** 2)
        alignment = self.alignment_gate * scores
        score_mean = scores.mean(dim=-1, keepdim=True)
        cohesion = -self.cohesion_gate * (scores - score_mean).abs()
        return scores + separation + alignment + cohesion

    def _apply_parisi_noise(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        xi = torch.randn_like(x)
        gate = self.noise_gate(x)
        return x + self.noise_eta * gate * xi

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Differentiable Routing ileri geçiş.

        1. Router → probs (differentiable) + indices (hard seçim)
        2. Tüm expert'leri hesapla (hepsinin çıktısı gerekli)
        3. output = Σ_k p_k × expert_k_output  (differentiable birleştirme)
           - Forward'da: sadece seçilen expert'in p_k'si anlamlı (diğerleri ~0)
           - Backward'da: gradyan p_k üzerinden router'a akar

        Returns:
            output, aux_loss, info
        """
        B, L, D = x.shape
        mask = self._get_mask(L, x.device)

        # ── Router: differentiable olasılıklar + hard seçim ──
        probs, indices, aux_loss, _ = self.router(x)
        # probs: (B, L, E) -- differentiable softmax
        # indices: (B, L, top_k) -- hard argmax

        # ── Tüm expert çıktılarını hesapla ──
        expert_outputs = []
        for expert in self.expert_heads:
            out = expert(x, mask, self._reynolds_modulate, self.attn_dropout)
            expert_outputs.append(out)

        # Stack: (B, L, E, D)
        stacked = torch.stack(expert_outputs, dim=2)

        # ── Differentiable routing: p_k × expert_k ──
        # probs: (B, L, E) → (B, L, E, 1) ile çarp
        # Bu, tüm expert çıktılarını olasılıklarıyla ağırlıklar
        # Düşük T'de probs keskin → neredeyse tek expert aktif
        # AMA gradyan hep akar (softmax differentiable)
        weighted = probs.unsqueeze(-1) * stacked  # (B, L, E, D)
        output = weighted.sum(dim=2)  # (B, L, D)

        # ── Çıktı projeksiyonu + Parisi gürültü ──
        output = self.out_proj(output)
        output = self._apply_parisi_noise(output)
        output = self.resid_dropout(output)

        # İstatistikler
        with torch.no_grad():
            usage = torch.zeros(self.num_experts, device=x.device)
            flat_idx = indices.squeeze(-1)  # (B, L)
            for i in range(self.num_experts):
                usage[i] = (flat_idx == i).float().mean()

        info = {
            'temperature': self.router.current_temperature.item(),
            'regret': self.router.cumulative_regret.clone(),
            'usage': usage,
            'aux_loss': aux_loss.item(),
        }
        return output, aux_loss, info


# ── Birleşik Blok v5 (Dual Norm) ────────────────────────────────────────────

class UnifiedBlock(nn.Module):
    """İki Normlu Birleşik Parisi-Nash Bloğu."""

    def __init__(self, config: SwarmConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.pre_norm = nn.LayerNorm(config.embed_dim)
        self.post_norm = nn.LayerNorm(config.embed_dim)
        self.unified_attn = UnifiedParisiNashAttention(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        residual = x
        x = self.pre_norm(x)
        out, aux_loss, info = self.unified_attn(x)
        out = self.post_norm(out)
        return residual + out, aux_loss, info


# ── Birleşik Parisi-Nash Dil Modeli v5 ───────────────────────────────────────

class UnifiedParisiNashLLM(nn.Module):
    """
    Birleşik Parisi-Nash v5: Keskin Uzmanlaşma.

    Evrim:
      v1: collapse → v2: +LB → v3: +annealing → v4: +differentiable
      v5: +MLP router + düşük LB + uzun koşu → BASELINE HEDEF
    """

    def __init__(self, config: SwarmConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.emb_dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            UnifiedBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.embed_dim)

        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.t_start = 2.0
        self.t_end = 0.3
        self.lb_coeff = 0.0001

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_annealing_step(self, step: int, total_steps: int):
        """Parisi Tavlama: Cosine schedule ile temperature ayarla."""
        progress = min(step / max(total_steps, 1), 1.0)
        temperature = self.t_end + 0.5 * (self.t_start - self.t_end) * (1 + math.cos(math.pi * progress))
        for layer in self.layers:
            layer.unified_attn.router.set_temperature(temperature)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> dict:
        B, L = input_ids.shape
        x = self.tok_emb(input_ids)
        x = self.emb_dropout(x)

        total_aux_loss = 0.0
        all_info = []

        for layer in self.layers:
            x, aux_loss, info = layer(x)
            total_aux_loss = total_aux_loss + aux_loss
            all_info.append(info)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        ce_loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )
            loss = ce_loss + self.lb_coeff * total_aux_loss

        return {
            'logits': logits,
            'loss': loss,
            'ce_loss': ce_loss,
            'aux_loss': total_aux_loss,
            'moe_info': all_info,
        }

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embedding = sum(p.numel() for p in self.tok_emb.parameters())

        expert_qkv = sum(
            p.numel() for layer in self.layers
            for name, p in layer.named_parameters()
            if any(s in name for s in ['q_proj', 'k_proj', 'v_proj'])
            and 'expert' in name
        )
        expert_ffn = sum(
            p.numel() for layer in self.layers
            for name, p in layer.named_parameters()
            if any(s in name for s in ['w1', 'w2', 'v_gate'])
            and 'expert' in name
        )
        router_params = sum(
            p.numel() for layer in self.layers
            for name, p in layer.named_parameters()
            if 'router' in name
        )

        return {
            'total': total,
            'trainable': trainable,
            'embedding': embedding,
            'expert_QKV (attn)': expert_qkv,
            'expert_FFN/φ (SwiGLU)': expert_ffn,
            'router (Nash)': router_params,
        }

    def get_router_params(self) -> list:
        """Decoupled gradient clipping için router parametrelerini döner."""
        return [p for n, p in self.named_parameters() if 'router' in n]

    def get_expert_params(self) -> list:
        """Decoupled gradient clipping için expert parametrelerini döner."""
        return [p for n, p in self.named_parameters() if 'router' not in n]

    def get_nash_stats(self) -> dict:
        stats = {
            'temperatures': [],
            'regrets': [],
            'usages': [],
        }
        for layer in self.layers:
            router = layer.unified_attn.router
            stats['temperatures'].append(router.current_temperature.item())
            stats['regrets'].append(router.cumulative_regret.tolist())
            stats['usages'].append(router.expert_usage.tolist())
        return stats
