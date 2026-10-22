import math

import numpy as np
from numpy.typing import NDArray

from exo.shared.types.bitkv_cache import (
    KIVICacheTelemetry,
    KIVIKVCacheConfig,
)
from exo.worker.cache.kivi_quantizer import KIVIQuantizer


class AsymmetricBitKVPool:
    """Hybrid KV cache pool maintaining a high-precision recent window and 2-bit compressed history."""

    def __init__(self, config: KIVIKVCacheConfig | None = None) -> None:
        self._config = config or KIVIKVCacheConfig()

        # Compressed historical storage chunks: (packed, scale, zero_point, (seq_len, head_dim))
        self._quantized_keys: list[tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.float32], tuple[int, int]]] = []
        self._quantized_values: list[tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.float32], tuple[int, int]]] = []

        # High-precision recent window buffer
        self._recent_keys: list[NDArray[np.float32]] = []
        self._recent_values: list[NDArray[np.float32]] = []

        # Ground truth tracking for validation/fidelity measurement
        self._ground_truth_keys: list[NDArray[np.float32]] = []
        self._ground_truth_values: list[NDArray[np.float32]] = []

    @property
    def config(self) -> KIVIKVCacheConfig:
        return self._config

    @property
    def total_tokens(self) -> int:
        hist_tokens = sum(shape[0] for _, _, _, shape in self._quantized_keys)
        recent_tokens = len(self._recent_keys)
        return hist_tokens + recent_tokens

    def append_kv(self, key_vector: NDArray[np.floating], value_vector: NDArray[np.floating]) -> None:
        """Appends a new token's Key and Value vectors of shape (head_dim,)."""
        k = np.asarray(key_vector, dtype=np.float32).reshape(1, -1)
        v = np.asarray(value_vector, dtype=np.float32).reshape(1, -1)

        self._recent_keys.append(k)
        self._recent_values.append(v)
        self._ground_truth_keys.append(k)
        self._ground_truth_values.append(v)

        # When recent buffer has accumulated enough tokens beyond recent_window_size,
        # evict and compress a complete chunk of group_size historical tokens.
        threshold = self._config.recent_window_size + self._config.group_size
        if len(self._recent_keys) >= threshold:
            chunk_size = self._config.group_size
            evict_k = np.concatenate(self._recent_keys[:chunk_size], axis=0)
            evict_v = np.concatenate(self._recent_values[:chunk_size], axis=0)

            # Quantize Key (Per-channel) and Value (Per-token)
            packed_k, scale_k, zp_k = KIVIQuantizer.quantize_key_per_channel(
                evict_k, bit_width=self._config.bit_width
            )
            packed_v, scale_v, zp_v = KIVIQuantizer.quantize_value_per_token(
                evict_v, bit_width=self._config.bit_width
            )

            self._quantized_keys.append((packed_k, scale_k, zp_k, evict_k.shape))
            self._quantized_values.append((packed_v, scale_v, zp_v, evict_v.shape))

            # Remove evicted chunk from recent buffer
            self._recent_keys = self._recent_keys[chunk_size:]
            self._recent_values = self._recent_values[chunk_size:]

    def reconstruct_full_key_value(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Reconstructs the continuous sequence Key and Value matrices."""
        k_blocks: list[NDArray[np.float32]] = []
        v_blocks: list[NDArray[np.float32]] = []

        # Dequantize compressed historical chunks
        for packed_k, scale_k, zp_k, shape in self._quantized_keys:
            dequant_k = KIVIQuantizer.dequantize_key_per_channel(
                packed_k, scale_k, zp_k, shape, bit_width=self._config.bit_width
            )
            k_blocks.append(dequant_k)

        for packed_v, scale_v, zp_v, shape in self._quantized_values:
            dequant_v = KIVIQuantizer.dequantize_value_per_token(
                packed_v, scale_v, zp_v, shape, bit_width=self._config.bit_width
            )
            v_blocks.append(dequant_v)

        # Concatenate unquantized recent window
        if self._recent_keys:
            k_blocks.append(np.concatenate(self._recent_keys, axis=0))
            v_blocks.append(np.concatenate(self._recent_values, axis=0))

        full_k = np.concatenate(k_blocks, axis=0) if k_blocks else np.empty((0, self._config.head_dim), dtype=np.float32)
        full_v = np.concatenate(v_blocks, axis=0) if v_blocks else np.empty((0, self._config.head_dim), dtype=np.float32)
        return full_k, full_v

    def compute_attention(self, query: NDArray[np.floating]) -> NDArray[np.float32]:
        """Computes scaled dot-product self-attention over the hybrid KV cache."""
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        full_k, full_v = self.reconstruct_full_key_value()
        if len(full_k) == 0:
            return np.zeros((1, self._config.head_dim), dtype=np.float32)

        # Scaled dot-product: Q @ K^T / sqrt(d)
        d_k = float(q.shape[-1])
        scores = np.matmul(q, full_k.T) / math.sqrt(d_k)

        # Softmax
        scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

        # Output = Attention @ V
        output = np.matmul(attn_weights, full_v)
        return output

    def get_telemetry(self) -> KIVICacheTelemetry:
        """Computes memory telemetry and quantization fidelity."""
        n_tokens = self.total_tokens
        head_dim = self._config.head_dim

        # Raw FP16 baseline size: 2 bytes per float16 for both Key and Value
        raw_fp16_bytes = n_tokens * head_dim * 2 * 2

        # Compressed bytes calculation:
        # Quantized history: packed bits + float32 scale & zero-point parameters
        compressed_bytes = 0
        for packed_k, scale_k, zp_k, _ in self._quantized_keys:
            # Packed 2-bit data bytes + FP16 scale and zero-point parameters
            compressed_bytes += packed_k.nbytes + (scale_k.size * 2) + (zp_k.size * 2)
        for packed_v, scale_v, zp_v, _ in self._quantized_values:
            compressed_bytes += packed_v.nbytes + (scale_v.size * 2) + (zp_v.size * 2)

        # Recent window in FP16 (2 bytes per float16)
        recent_count = len(self._recent_keys)
        compressed_bytes += recent_count * head_dim * 2 * 2

        compression_ratio = (compressed_bytes / raw_fp16_bytes) if raw_fp16_bytes > 0 else 1.0

        # Cosine similarity against unquantized ground truth
        full_k, full_v = self.reconstruct_full_key_value()
        gt_k = np.concatenate(self._ground_truth_keys, axis=0) if self._ground_truth_keys else full_k
        gt_v = np.concatenate(self._ground_truth_values, axis=0) if self._ground_truth_values else full_v

        k_sim = KIVIQuantizer.calculate_cosine_similarity(full_k, gt_k)
        v_sim = KIVIQuantizer.calculate_cosine_similarity(full_v, gt_v)

        return KIVICacheTelemetry(
            total_tokens_cached=n_tokens,
            raw_fp16_bytes=raw_fp16_bytes,
            compressed_bytes=compressed_bytes,
            compression_ratio=compression_ratio,
            key_cosine_similarity=k_sim,
            value_cosine_similarity=v_sim,
        )
