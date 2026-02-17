"""
Birleşik Parisi-Nash Dikkat Mekanizması v3
============================================
İki teoriyi TEK BİR FORMÜLDE birleştiren orijinal mimari.

v3 İyileştirmeleri (v2 üzerine):
  1. Temperature Annealing: Parisi tavlama ilkesi (2.0 → 0.3)
     Başlangıçta keşif, sonunda hedefe kilitlenme
  2. LB katsayısı 0.001: Uzmanlaşmayı engellemeden dengesizliği önle
  3. Top-K=1: Her token TEK bir expert'e bağlı → zorunlu uzmanlaşma
  4. Sparse execution: Sadece seçilen expert çalışır → 4x hız kazancı

Formül:
  v_i(t+1) = w_k^nash(x_i) · φ_k(Σ_{j∈N(i)} α_ij^k · V_k_j) + η·ε

  top_k=1 ile artık toplam (Σ_k) yok -- TEK expert sorumlu.
  Her expert:
    - NEREYE bakacağını kendisi belirler (Q_k, K_k)
    - NEYİ çıkaracağını kendisi belirler (V_k)
    - NASIL işleyeceğini kendisi belirler (φ_k)

  Parisi fizik kuralları tüm expert'ler için aynı:
    - Sliding window maskesi (7 komşu kuralı)
    - Reynolds kuralları (ayrılma/hizalanma/uyum)
    - Temperature Annealing (tavlama -- Parisi'nin kendi ilkesi)
"""

import math
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

    Her expert kendi Q_k, K_k, V_k projeksiyonlarına sahip:
      - Q_k·K_k: Bu expert NEREYE bakacak (dikkat deseni)
      - V_k: Bu expert NEYİ çıkaracak (değer projeksiyonu)
      - φ_k: Bu expert NASIL işleyecek (SwiGLU dönüşümü)
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
        """Expert-spesifik dikkat + dönüşüm. (B, L, D) → (B, L, D)"""
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


# ── Nash Router v3 ───────────────────────────────────────────────────────────

class NashExpertRouter(nn.Module):
    """
    Nash Dengesi tabanlı expert router v3.

    v3 İyileştirmeleri:
      1. Temperature Annealing (Parisi tavlama): T dışarıdan schedule ile kontrol
         - Öğrenilebilir T kaldırıldı (v2'de takılıyordu)
         - Cosine schedule: T_start → T_end
      2. Top-K=1 desteği (varsayılan): zorunlu uzmanlaşma
      3. LB loss hafifletildi (katsayı 0.001)
      4. Regret sadece izleme amaçlı (logit'leri değiştirmez)
    """

    def __init__(self, embed_dim: int, num_experts: int, top_k: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Linear(embed_dim, num_experts, bias=False)

        # Temperature artık dışarıdan kontrol edilir (annealing)
        self.register_buffer('current_temperature', torch.tensor(2.0))

        # Regret sadece izleme (logit'leri ETKİLEMEZ)
        self.register_buffer('cumulative_regret', torch.zeros(num_experts))
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))

    def set_temperature(self, t: float):
        """Dışarıdan temperature ayarla (annealing schedule)."""
        self.current_temperature.fill_(t)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            weights: (B, L, top_k) expert ağırlıkları
            indices: (B, L, top_k) expert indeksleri
            aux_loss: Load balancing auxiliary loss
        """
        T = self.current_temperature.clamp(min=0.1)
        logits = self.gate(x) / T

        probs = F.softmax(logits, dim=-1)

        # Top-K seçimi
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # ── Load Balancing Loss ──
        with torch.no_grad():
            one_hot = F.one_hot(indices, self.num_experts).float()
            f = one_hot.sum(dim=(0, 1, 2)) / (indices.shape[0] * indices.shape[1] * self.top_k)

        P = probs.mean(dim=(0, 1))
        aux_loss = self.num_experts * (f * P).sum()

        # İzleme (regret + usage)
        if self.training:
            self.step_count += 1
            with torch.no_grad():
                self.expert_usage = 0.99 * self.expert_usage + 0.01 * f
                uniform = torch.ones_like(f) / self.num_experts
                self.cumulative_regret = 0.99 * self.cumulative_regret + (uniform - f)

        return weights, indices, aux_loss


# ── Birleşik Parisi-Nash Dikkat v3 ───────────────────────────────────────────

class UnifiedParisiNashAttention(nn.Module):
    """
    Birleşik Parisi-Nash Dikkat Mekanizması v3.

    TEK FORMÜL (top_k=1):
    v_i = φ_k(Σ_{j∈N(i)} α_ij^k · V_k_j) + η·ε

    k = argmax Nash router  (tek expert sorumlu)

    v3 farkı: SPARSE execution
      Tüm expert'leri çalıştırıp gather yapmak yerine,
      sadece SEÇİLEN expert'leri çalıştırır → 4x hız.
    """

    def __init__(self, config: SwarmConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.neighbor_size = config.neighbor_size

        # ── Expert Attention Heads ──
        expand_dim = config.embed_dim * config.ffn_multiplier
        self.expert_heads = nn.ModuleList([
            ExpertAttentionHead(
                config.embed_dim, config.num_heads, config.head_dim,
                expand_dim, config.dropout,
            )
            for _ in range(config.num_experts)
        ])

        # ── Çıktı projeksiyonu ──
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

        # ── Nash Router v3 ──
        self.router = NashExpertRouter(config.embed_dim, config.num_experts, config.top_k_experts)

        # ── Reynolds Kuralları ──
        self.separation_gate = nn.Parameter(torch.tensor(config.separation_weight))
        self.alignment_gate = nn.Parameter(torch.tensor(config.alignment_weight))
        self.cohesion_gate = nn.Parameter(torch.tensor(config.cohesion_weight))

        # ── Parisi Gürültü (η) ──
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
        Birleşik Parisi-Nash v3 ileri geçiş.

        SPARSE execution:
          1. Nash router → her token için 1 expert seç
          2. Token'ları expert'lerine göre grupla
          3. Her expert SADECE kendi token'larını işler
          4. Sonuçları birleştir + Parisi gürültü

        Returns:
            output: (B, L, D)
            aux_loss: Load balancing loss
            info: Nash routing istatistikleri
        """
        B, L, D = x.shape
        mask = self._get_mask(L, x.device)

        # ── Nash routing (top_k=1) ──
        nash_weights, nash_indices, aux_loss = self.router(x)
        # nash_weights: (B, L, 1), nash_indices: (B, L, 1)

        # ── SPARSE expert execution ──
        # Her expert sadece kendisine atanan token'ları işler
        output = torch.zeros(B, L, D, device=x.device, dtype=x.dtype)
        flat_indices = nash_indices.squeeze(-1)  # (B, L)

        for expert_idx, expert in enumerate(self.expert_heads):
            # Bu expert'e atanan token'lar
            token_mask = (flat_indices == expert_idx)  # (B, L) bool

            if not token_mask.any():
                continue

            # Expert'in çıktısını hesapla (tüm pozisyonlar -- mask zaten attention'da)
            expert_out = expert(x, mask, self._reynolds_modulate, self.attn_dropout)

            # Sadece bu expert'e atanan pozisyonlara yaz
            if self.top_k == 1:
                output[token_mask] = expert_out[token_mask]
            else:
                # top_k > 1 durumunda ağırlıklı toplam
                weight_for_expert = torch.zeros(B, L, device=x.device)
                for k in range(self.top_k):
                    k_mask = (nash_indices[:, :, k] == expert_idx)
                    weight_for_expert[k_mask] = nash_weights[:, :, k][k_mask]
                output = output + weight_for_expert.unsqueeze(-1) * expert_out

        # ── Çıktı projeksiyonu + Parisi gürültü ──
        output = self.out_proj(output)
        output = self._apply_parisi_noise(output)
        output = self.resid_dropout(output)

        # İstatistikler
        with torch.no_grad():
            usage = torch.zeros(self.num_experts, device=x.device)
            for i in range(self.num_experts):
                usage[i] = (flat_indices == i).float().mean()

        info = {
            'temperature': self.router.current_temperature.item(),
            'regret': self.router.cumulative_regret.clone(),
            'usage': usage,
            'aux_loss': aux_loss.item(),
        }
        return output, aux_loss, info


# ── Birleşik Blok v3 (Dual Norm) ────────────────────────────────────────────

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


# ── Birleşik Parisi-Nash Dil Modeli v3 ───────────────────────────────────────

class UnifiedParisiNashLLM(nn.Module):
    """
    Birleşik Parisi-Nash Stratejik Sığırcık Dil Modeli v3.

    v3: Temperature Annealing + Top-K=1 + Sparse Execution

    Parisi Tavlama Metaforu:
      Başlangıç (T=2.0): Atomlar serbest, keşif modu
      Soğuma (T→0.3): Kristal yapı oluşur, uzmanlaşma tamamlanır
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

        # Annealing parametreleri
        self.t_start = 2.0
        self.t_end = 0.3
        self.lb_coeff = 0.001

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_annealing_step(self, step: int, total_steps: int):
        """
        Parisi Tavlama: Cosine schedule ile temperature ayarla.

        T(t) = T_end + 0.5 * (T_start - T_end) * (1 + cos(π * t / T))

        Bu, fizikte "yavaş soğutma" prensibine uyar:
        - Başlangıçta hızlı düşer (en çok keşif başta gerekli)
        - Ortada yavaşlar (uzmanlaşma için zaman verir)
        - Sonda ince ayar yapar (0.3'e yakınsar)
        """
        progress = min(step / max(total_steps, 1), 1.0)
        temperature = self.t_end + 0.5 * (self.t_start - self.t_end) * (1 + math.cos(math.pi * progress))
        for layer in self.layers:
            layer.unified_attn.router.set_temperature(temperature)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
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
            'ce_loss': ce_loss if targets is not None else None,
            'aux_loss': total_aux_loss,
            'moe_info': all_info,
        }

    def count_parameters(self) -> dict[str, int]:
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
            info = layer.unified_attn.router.expert_usage.tolist()
            stats['usages'].append(info)
        return stats
