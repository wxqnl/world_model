"""Storage codecs for the production native WM3D-V7 5B cache.

The external VGGT representation remains D=2048.  To keep a 5,000--8,000 hour
corpus inside the planned storage envelope, feature vectors are stored as
symmetric int8 values with one FP16 scale per vector.  RGB supervision is
stored as independently decodable JPEG records in an append-only pack, so a
random training window never has to decode an entire source video.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import torch
from torchvision.io import ImageReadMode, decode_jpeg, encode_jpeg


class CodecIntegrityError(RuntimeError):
    """Raised when a cache payload cannot be decoded exactly as contracted."""


def quantize_per_vector(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetrically quantize the final dimension of a finite float tensor."""

    if not value.is_floating_point() or value.ndim < 1:
        raise ValueError("quantized feature input must be a floating tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("quantized feature input contains NaN or Inf")
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
    """Restore a tensor written by :func:`quantize_per_vector`."""

    if quantized.dtype != torch.int8:
        raise CodecIntegrityError(f"quantized values must be int8, got {quantized.dtype}")
    if scale.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise CodecIntegrityError(f"quantization scale must be floating, got {scale.dtype}")
    if quantized.shape[:-1] + (1,) != scale.shape:
        raise CodecIntegrityError(
            f"quantized/scale shapes disagree: {quantized.shape} vs {scale.shape}"
        )
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise CodecIntegrityError("quantization scale is non-finite or non-positive")
    return (quantized.float() * scale.float()).to(dtype)


class JpegPackWriter:
    """Append independent JPEG records and return their byte ranges."""

    def __init__(self, path: Path, *, quality: int = 92) -> None:
        self.path = Path(path)
        self.quality = int(quality)
        if not 80 <= self.quality <= 100:
            raise ValueError("production JPEG quality must be in [80,100]")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self._fd = os.open(self.path, flags, 0o640)
        self._offset = 0
        self._offset_rows: list[list[int]] = []
        self._length_rows: list[list[int]] = []
        self._views: int | None = None
        self._closed = False

    def append(self, images: torch.Tensor) -> None:
        """Append one frame of uint8 RGB images shaped ``[V,3,H,W]``."""

        if self._closed:
            raise RuntimeError("JPEG pack writer is closed")
        if images.ndim != 4 or images.shape[1] != 3 or images.dtype != torch.uint8:
            raise ValueError("JPEG pack frame must be uint8 [V,3,H,W]")
        views = int(images.shape[0])
        if self._views is None:
            self._views = views
        elif views != self._views:
            raise ValueError("JPEG pack view count changed within a shard")
        offsets: list[int] = []
        lengths: list[int] = []
        for image in images.cpu():
            encoded = encode_jpeg(image.contiguous(), quality=self.quality)
            payload = encoded.numpy().tobytes()
            if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
                raise CodecIntegrityError("torchvision produced a malformed JPEG record")
            written = 0
            while written < len(payload):
                count = os.write(self._fd, payload[written:])
                if count <= 0:
                    raise OSError("short write while publishing JPEG pack")
                written += count
            offsets.append(self._offset)
            lengths.append(len(payload))
            self._offset += len(payload)
        self._offset_rows.append(offsets)
        self._length_rows.append(lengths)

    def close(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._closed:
            raise RuntimeError("JPEG pack writer was closed twice")
        os.fsync(self._fd)
        os.close(self._fd)
        self._fd = -1
        self._closed = True
        directory_fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if not self._offset_rows:
            raise CodecIntegrityError("cannot publish an empty JPEG pack")
        return (
            torch.tensor(self._offset_rows, dtype=torch.int64),
            torch.tensor(self._length_rows, dtype=torch.int32),
        )

    def __enter__(self) -> "JpegPackWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
        self._closed = True


class JpegPackReader:
    """pread-based random reader safe to instantiate independently per worker."""

    def __init__(self, path: Path) -> None:
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
        images: list[torch.Tensor] = []
        for offset_value, length_value in zip(offsets, lengths, strict=True):
            offset = int(offset_value)
            length = int(length_value)
            if offset < 0 or length <= 0 or offset + length > self._size:
                raise CodecIntegrityError(
                    f"JPEG range [{offset},{offset + length}) exceeds pack size "
                    f"{self._size}: {self.path}"
                )
            payload = os.pread(self._fd, length, offset)
            if len(payload) != length:
                raise CodecIntegrityError("short pread from JPEG pack")
            encoded = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
            image = decode_jpeg(encoded, mode=ImageReadMode.RGB)
            if image.ndim != 3 or image.shape[0] != 3:
                raise CodecIntegrityError("decoded JPEG is not RGB CHW")
            images.append(image)
        if not images:
            raise CodecIntegrityError("JPEG decode request was empty")
        shape = images[0].shape
        if any(image.shape != shape for image in images):
            raise CodecIntegrityError("JPEG records in one frame have different shapes")
        return torch.stack(images)
