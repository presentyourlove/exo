import numpy as np

from exo.shared.types.bitkv_cache import (
    KIVIKVCacheConfig,
)
from exo.worker.cache import (
    AsymmetricBitKVPool,
    KIVIQuantizer,
)


def test_kivi_pack_and_unpack_2bit() -> None:
    np.random.seed(42)
    # Random 2-bit values in range [0, 3]
    orig = np.random.randint(0, 4, size=(8, 32), dtype=np.uint8)
    packed = KIVIQuantizer.pack_2bit(orig)
    assert packed.shape == (8, 8)  # 32 / 4 = 8

    unpacked = KIVIQuantizer.unpack_2bit(packed, original_length=32)
    np.testing.assert_array_equal(unpacked, orig)


def test_kivi_quantize_key_per_channel_fidelity() -> None:
    np.random.seed(123)
    seq_len, head_dim = 64, 32
    # Simulate Key matrix with channel outliers
    key = np.random.randn(seq_len, head_dim).astype(np.float32)
    key[:, 5] *= 8.0  # Channel outlier on index 5

    packed, scale, zp = KIVIQuantizer.quantize_key_per_channel(key, bit_width=2)
    dequant_key = KIVIQuantizer.dequantize_key_per_channel(
        packed, scale, zp, original_shape=(seq_len, head_dim), bit_width=2
    )

    similarity = KIVIQuantizer.calculate_cosine_similarity(key, dequant_key)
    # 2-bit uniform quantization of Gaussian variables gives ~0.92-0.94 fidelity
    assert similarity > 0.90


def test_kivi_quantize_value_per_token_fidelity() -> None:
    np.random.seed(456)
    seq_len, head_dim = 64, 32
    value = np.random.randn(seq_len, head_dim).astype(np.float32)

    packed, scale, zp = KIVIQuantizer.quantize_value_per_token(value, bit_width=2)
    dequant_value = KIVIQuantizer.dequantize_value_per_token(
        packed, scale, zp, original_shape=(seq_len, head_dim), bit_width=2
    )

    similarity = KIVIQuantizer.calculate_cosine_similarity(value, dequant_value)
    assert similarity > 0.90


def test_kivi_4bit_quantization() -> None:
    np.random.seed(789)
    seq_len, head_dim = 32, 16
    arr = np.random.randn(seq_len, head_dim).astype(np.float32)

    packed_k, scale_k, zp_k = KIVIQuantizer.quantize_key_per_channel(arr, bit_width=4)
    dequant_k = KIVIQuantizer.dequantize_key_per_channel(
        packed_k, scale_k, zp_k, original_shape=(seq_len, head_dim), bit_width=4
    )
    sim_k = KIVIQuantizer.calculate_cosine_similarity(arr, dequant_k)
    assert sim_k > 0.995


def test_asymmetric_bitkv_pool_sliding_window_eviction() -> None:
    np.random.seed(42)
    config = KIVIKVCacheConfig(bit_width=2, recent_window_size=8, group_size=8, head_dim=16)
    pool = AsymmetricBitKVPool(config=config)

    # 1. Append 8 tokens - all remain in recent window
    for _ in range(8):
        k = np.random.randn(16).astype(np.float32)
        v = np.random.randn(16).astype(np.float32)
        pool.append_kv(k, v)

    assert pool.total_tokens == 8
    assert len(pool._quantized_keys) == 0
    assert len(pool._recent_keys) == 8

    # 2. Append 16 more tokens (total 24 tokens)
    # Threshold = recent_window_size (8) + group_size (8) = 16.
    # When tokens reach 16, a chunk of 8 is evicted.
    # At 24 tokens, 2 chunks of 8 have been evicted and 8 remain in recent window.
    for _ in range(16):
        k = np.random.randn(16).astype(np.float32)
        v = np.random.randn(16).astype(np.float32)
        pool.append_kv(k, v)

    assert pool.total_tokens == 24
    assert len(pool._recent_keys) == 8  # Recent window maintained at 8
    assert len(pool._quantized_keys) == 2  # 2 chunks of 8 evicted

    full_k, full_v = pool.reconstruct_full_key_value()
    assert full_k.shape == (24, 16)
    assert full_v.shape == (24, 16)


def test_asymmetric_bitkv_pool_attention_and_telemetry() -> None:
    np.random.seed(101)
    config = KIVIKVCacheConfig(bit_width=2, recent_window_size=16, group_size=16, head_dim=32)
    pool = AsymmetricBitKVPool(config=config)

    # Feed 64 tokens into the cache
    for _ in range(64):
        k = np.random.randn(32).astype(np.float32)
        v = np.random.randn(32).astype(np.float32)
        pool.append_kv(k, v)

    # Compute attention with query
    query = np.random.randn(32).astype(np.float32)
    attn_out = pool.compute_attention(query)
    assert attn_out.shape == (1, 32)

    # Telemetry check
    telemetry = pool.get_telemetry()
    assert telemetry.total_tokens_cached == 64
    assert telemetry.raw_fp16_bytes > telemetry.compressed_bytes
    assert telemetry.space_savings_percentage > 50.0  # Significant memory reduction
    assert telemetry.key_cosine_similarity > 0.95
    assert telemetry.value_cosine_similarity > 0.95
