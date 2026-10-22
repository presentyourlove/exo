import numpy as np
from numpy.typing import NDArray

from exo.shared.types.bitkv_cache import QuantizedBitWidth


class KIVIQuantizer:
    """Asymmetric low-bit quantizer for Key (Per-channel) and Value (Per-token) tensors."""

    @staticmethod
    def pack_2bit(quantized: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Packs four 2-bit values along the last dimension into one uint8 byte."""
        orig_shape = quantized.shape
        last_dim = orig_shape[-1]
        pad_len = (4 - (last_dim % 4)) % 4
        arr = quantized & 0x03
        if pad_len > 0:
            pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, pad_len)]
            arr = np.pad(arr, pad_width, mode="constant", constant_values=0)

        r0 = arr[..., 0::4]
        r1 = arr[..., 1::4] << 2
        r2 = arr[..., 2::4] << 4
        r3 = arr[..., 3::4] << 6
        packed = r0 | r1 | r2 | r3
        return packed

    @staticmethod
    def unpack_2bit(packed: NDArray[np.uint8], original_length: int) -> NDArray[np.uint8]:
        """Unpacks packed uint8 bytes back into individual 2-bit values."""
        r0 = packed & 0x03
        r1 = (packed >> 2) & 0x03
        r2 = (packed >> 4) & 0x03
        r3 = (packed >> 6) & 0x03

        unpacked_shape = list(packed.shape)
        unpacked_shape[-1] = packed.shape[-1] * 4
        unpacked = np.empty(unpacked_shape, dtype=np.uint8)
        unpacked[..., 0::4] = r0
        unpacked[..., 1::4] = r1
        unpacked[..., 2::4] = r2
        unpacked[..., 3::4] = r3

        return unpacked[..., :original_length]

    @classmethod
    def quantize_key_per_channel(
        cls,
        key_tensor: NDArray[np.floating],
        bit_width: QuantizedBitWidth = 2,
    ) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.float32]]:
        """Quantizes Key tensor using Per-Channel grouping across sequence dimension.

        Args:
            key_tensor: Shape (seq_len, head_dim)

        Returns:
            Tuple of (packed_quantized_key, scale_per_channel, zero_point_per_channel)
        """
        arr = np.asarray(key_tensor, dtype=np.float32)
        _ = arr.shape
        max_int = (1 << bit_width) - 1

        # Per-channel: min and max along axis 0 (seq_len), shape (1, head_dim)
        k_min = np.min(arr, axis=0, keepdims=True)
        k_max = np.max(arr, axis=0, keepdims=True)

        scale = np.where(k_max > k_min, (k_max - k_min) / float(max_int), 1.0).astype(np.float32)
        zero_point = k_min.astype(np.float32)

        quantized = np.clip(np.round((arr - zero_point) / scale), 0, max_int).astype(np.uint8)

        if bit_width == 2:
            packed = cls.pack_2bit(quantized)
        else:
            # 4-bit packing
            k0 = quantized[..., 0::2]
            k1 = quantized[..., 1::2] << 4
            packed = k0 | k1

        return packed, scale, zero_point

    @classmethod
    def dequantize_key_per_channel(
        cls,
        packed_key: NDArray[np.uint8],
        scale: NDArray[np.float32],
        zero_point: NDArray[np.float32],
        original_shape: tuple[int, int],
        bit_width: QuantizedBitWidth = 2,
    ) -> NDArray[np.float32]:
        """Reconstructs float Key tensor from packed representation."""
        _seq_len, head_dim = original_shape
        if bit_width == 2:
            unpacked = cls.unpack_2bit(packed_key, head_dim).astype(np.float32)
        else:
            k0 = packed_key & 0x0F
            k1 = (packed_key >> 4) & 0x0F
            unpacked = np.empty(original_shape, dtype=np.float32)
            unpacked[..., 0::2] = k0
            unpacked[..., 1::2] = k1

        return (unpacked * scale) + zero_point

    @classmethod
    def quantize_value_per_token(
        cls,
        value_tensor: NDArray[np.floating],
        bit_width: QuantizedBitWidth = 2,
    ) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.float32]]:
        """Quantizes Value tensor using Per-Token grouping across head_dim dimension.

        Args:
            value_tensor: Shape (seq_len, head_dim)

        Returns:
            Tuple of (packed_quantized_value, scale_per_token, zero_point_per_token)
        """
        arr = np.asarray(value_tensor, dtype=np.float32)
        _ = arr.shape
        max_int = (1 << bit_width) - 1

        # Per-token: min and max along axis 1 (head_dim), shape (seq_len, 1)
        v_min = np.min(arr, axis=1, keepdims=True)
        v_max = np.max(arr, axis=1, keepdims=True)

        scale = np.where(v_max > v_min, (v_max - v_min) / float(max_int), 1.0).astype(np.float32)
        zero_point = v_min.astype(np.float32)

        quantized = np.clip(np.round((arr - zero_point) / scale), 0, max_int).astype(np.uint8)

        if bit_width == 2:
            packed = cls.pack_2bit(quantized)
        else:
            v0 = quantized[..., 0::2]
            v1 = quantized[..., 1::2] << 4
            packed = v0 | v1

        return packed, scale, zero_point

    @classmethod
    def dequantize_value_per_token(
        cls,
        packed_value: NDArray[np.uint8],
        scale: NDArray[np.float32],
        zero_point: NDArray[np.float32],
        original_shape: tuple[int, int],
        bit_width: QuantizedBitWidth = 2,
    ) -> NDArray[np.float32]:
        """Reconstructs float Value tensor from packed representation."""
        _seq_len, head_dim = original_shape
        if bit_width == 2:
            unpacked = cls.unpack_2bit(packed_value, head_dim).astype(np.float32)
        else:
            v0 = packed_value & 0x0F
            v1 = (packed_value >> 4) & 0x0F
            unpacked = np.empty(original_shape, dtype=np.float32)
            unpacked[..., 0::2] = v0
            unpacked[..., 1::2] = v1

        return (unpacked * scale) + zero_point

    @staticmethod
    def calculate_cosine_similarity(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
        """Computes average cosine similarity between two matrices."""
        a_flat = np.asarray(a, dtype=np.float32).ravel()
        b_flat = np.asarray(b, dtype=np.float32).ravel()
        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)
        if norm_a == 0.0 or norm_b == 0.0:
            return 1.0
        return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))
