"""Heimdall: Gated DeltaNet + seyrek global dikkat hibrit dil modeli (CUDA / ROCm / Vulkan)."""

from .config import ARCHS, PRESETS, HeimdallConfig, apply_arch
from .model import HeimdallLM
from .gjallar import (
    GjallarAttention,
    GjallarGate,
    GjallarLayer,
    GjallarStack,
    TwoLevelBlockIndex,
)

__all__ = [
    "HeimdallConfig",
    "HeimdallLM",
    "PRESETS",
    "ARCHS",
    "apply_arch",
    "GjallarAttention",
    "GjallarGate",
    "GjallarLayer",
    "GjallarStack",
    "TwoLevelBlockIndex",
]
__version__ = "1.0.0"
