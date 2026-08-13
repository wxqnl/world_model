"""Storage codecs for the unified WM3D cache."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import torch


class CacheCodecError(RuntimeError):
    pass


class JpegPackWriter:
    """Append independently decodable JPEG records with durable offsets."""

    def __init__(self, path: Path, *, quality: int = 92) -> None:
        self.path = Path(path)
        self.quality = int(quality)
        if not 80 <= self.quality <= 100:
            raise ValueError("JPEG quality must be in [80,100]")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self._fd = os.open(self.path, flags, 0o640)
        self._offset = 0
        self._closed = False

    def append(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append one uint8 frame ``[V,3,H,W]`` and return byte ranges."""

        if self._closed:
            raise CacheCodecError("JPEG pack writer is closed")
        if images.ndim != 4 or images.shape[1] != 3 or images.dtype != torch.uint8:
            raise ValueError("JPEG pack frame must be uint8 [V,3,H,W]")
        from torchvision.io import encode_jpeg

        offsets: list[int] = []
        lengths: list[int] = []
        for image in images.cpu():
            encoded = encode_jpeg(image.contiguous(), quality=self.quality)
            payload = encoded.numpy().tobytes()
            if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
                raise CacheCodecError("encoder produced a malformed JPEG record")
            written = 0
            while written < len(payload):
                count = os.write(self._fd, payload[written:])
                if count <= 0:
                    raise OSError("short write while publishing JPEG pack")
                written += count
            offsets.append(self._offset)
            lengths.append(len(payload))
            self._offset += len(payload)
        return (
            torch.tensor(offsets, dtype=torch.int64),
            torch.tensor(lengths, dtype=torch.int32),
        )

    def close(self) -> None:
        if self._closed:
            raise CacheCodecError("JPEG pack writer was closed twice")
        if self._offset <= 0:
            raise CacheCodecError("cannot publish an empty JPEG pack")
        os.fsync(self._fd)
        os.close(self._fd)
        self._fd = -1
        self._closed = True

    def __enter__(self) -> "JpegPackWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
        self._closed = True


def quantize_per_vector(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not value.is_floating_point() or value.ndim < 1:
        raise ValueError("quantized input must be a floating tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("quantized input contains NaN or Inf")
    maximum = value.float().abs().amax(dim=-1, keepdim=True)
    scale = (maximum / 127.0).clamp_min(torch.finfo(torch.float16).tiny)
    quantized = torch.round(value.float() / scale).clamp(-127, 127).to(torch.int8)
    return quantized.contiguous(), scale.to(torch.float16).contiguous()


def dequantize_per_vector(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if quantized.dtype != torch.int8:
        raise CacheCodecError(f"quantized values must be int8, got {quantized.dtype}")
    if scale.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise CacheCodecError(f"scale must be floating, got {scale.dtype}")
    if quantized.shape[:-1] + (1,) != scale.shape:
        raise CacheCodecError(
            f"quantized/scale shapes disagree: {quantized.shape} vs {scale.shape}"
        )
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise CacheCodecError("scale is non-finite or non-positive")
    return (quantized.float() * scale.float()).to(dtype)


class JpegPackReader:
    """pread-based random access to independently encoded JPEG records."""

    def __init__(self, path: Path):
        self.path = Path(path)
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self._fd = os.open(self.path, flags)
        self._size = int(os.fstat(self._fd).st_size)

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __del__(self) -> None:
        self.close()

    def decode(
        self,
        offsets: Iterable[int],
        lengths: Iterable[int],
    ) -> torch.Tensor:
        # torchvision is imported lazily so index/preflight tools do not load
        # CUDA/vision libraries.
        from torchvision.io import ImageReadMode, decode_jpeg

        images: list[torch.Tensor] = []
        for offset_value, length_value in zip(offsets, lengths, strict=True):
            offset = int(offset_value)
            length = int(length_value)
            if offset < 0 or length <= 0 or offset + length > self._size:
                raise CacheCodecError(
                    f"JPEG range [{offset},{offset + length}) exceeds {self._size}"
                )
            payload = os.pread(self._fd, length, offset)
            if len(payload) != length:
                raise CacheCodecError("short pread from JPEG pack")
            encoded = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
            image = decode_jpeg(encoded, mode=ImageReadMode.RGB)
            if image.ndim != 3 or image.shape[0] != 3:
                raise CacheCodecError("decoded JPEG is not RGB CHW")
            images.append(image)
        if not images:
            raise CacheCodecError("JPEG decode request is empty")
        shape = images[0].shape
        if any(image.shape != shape for image in images):
            raise CacheCodecError("JPEG records have inconsistent image shapes")
        return torch.stack(images)
