from typing import Literal

from exo.utils.pydantic_ext import FrozenModel

QuantizedBitWidth = Literal[2, 4]


class KIVIKVCacheConfig(FrozenModel):
    """Configuration for KIVI asymmetric low-bit KV cache quantization."""

    bit_width: QuantizedBitWidth = 2
    group_size: int = 32
    recent_window_size: int = 32
    head_dim: int = 64
    num_heads: int = 8


class KIVICacheTelemetry(FrozenModel):
    """Telemetry capturing memory savings and quantization fidelity of KIVI KV cache."""

    total_tokens_cached: int
    raw_fp16_bytes: int
    compressed_bytes: int
    compression_ratio: float
    key_cosine_similarity: float
    value_cosine_similarity: float

    @property
    def space_savings_percentage(self) -> float:
        return max(0.0, (1.0 - self.compression_ratio) * 100.0)
