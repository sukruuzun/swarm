"""
Swarm-LLM: Sığırcık Algoritması ile Dikkat Mekanizması
========================================================
Parisi'nin "7 Komşu" kuralını uygulayan, GPU-verimli bir dil modeli.

Geleneksel O(N²) dikkat yerine O(N·w) karmaşıklıkta
sliding window tabanlı sürü etkileşimi kullanır.
"""

from swarm_llm.config import SwarmConfig
from swarm_llm.attention import StarlingAttention
from swarm_llm.noise import ParisiNoise
from swarm_llm.model import SwarmLLM
from swarm_llm.nash_moe import NashMoE, NashRouter
from swarm_llm.nash_parisi_model import NashParisiLLM
from swarm_llm.unified import UnifiedParisiNashLLM, UnifiedParisiNashAttention
from swarm_llm.external_router import ExternalParisiNashRouter
from swarm_llm.sparse_loader import SparseBlockLoader
from swarm_llm.hf_loader import HuggingFaceBlockLoader

__version__ = "0.6.0"

__all__ = [
    "SwarmConfig",
    "StarlingAttention",
    "ParisiNoise",
    "SwarmLLM",
    "NashMoE",
    "NashRouter",
    "NashParisiLLM",
    "UnifiedParisiNashLLM",
    "UnifiedParisiNashAttention",
    "ExternalParisiNashRouter",
    "SparseBlockLoader",
    "HuggingFaceBlockLoader",
]
