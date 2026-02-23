# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Fast WAV decoder that bypasses FFmpeg for simple PCM WAV files.

WAV files with uncompressed PCM audio can be decoded by simply parsing
the RIFF header and copying the raw sample data, which is much faster
than going through FFmpeg's full decoding pipeline.
"""

import io
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from torchcodec import _core as core, AudioSamples

# WAV format codes
WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE


@dataclass
class WavMetadata:
    """Metadata extracted from a WAV file header."""

    sample_rate: int
    num_channels: int
    bits_per_sample: int
    num_samples: int
    duration_seconds: float
    audio_format: int  # 1 = PCM, 3 = IEEE float (resolved from EXTENSIBLE if needed)
    _data_offset: int = 0  # Internal: offset to PCM data


def _parse_wav_chunks(
    read_at: Callable[[int, int], bytes],
    total_size: int | None = None,
) -> WavMetadata:
    # Parse WAV header using a function that reads bytes from different sources (bytes or file-like).
    header = read_at(0, 12)
    if len(header) < 12:
        raise ValueError("File too small to be a valid WAV")

    if header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise ValueError("Not a valid WAV file")

    pos = 12
    fmt_found = False
    audio_format = 0
    num_channels = 0
    sample_rate = 0
    bits_per_sample = 0

    while True:
        chunk_header = read_at(pos, 8)
        if len(chunk_header) < 8:
            break

        chunk_id = chunk_header[0:4]
        # chunk size is a 4-byte little-endian integer starting at offset 4, after the chunk ID
        chunk_size = struct.unpack("<I", chunk_header[4:8])[0]

        if chunk_id == b"fmt ":
            if chunk_size < 16:
                raise ValueError("Invalid fmt chunk size")

            fmt_data = read_at(pos + 8, min(chunk_size, 40))
            if len(fmt_data) < 16:
                raise ValueError("Invalid fmt chunk")

            audio_format, num_channels, sample_rate, _, _, bits_per_sample = (
                struct.unpack("<HHIIHH", fmt_data[0:16])
            )

            if audio_format == WAVE_FORMAT_EXTENSIBLE:
                if chunk_size < 40 or len(fmt_data) < 26:
                    raise ValueError("Invalid extensible fmt chunk size")
                audio_format = struct.unpack("<H", fmt_data[24:26])[0]

            if audio_format not in (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT):
                raise ValueError(f"Unsupported audio format: {audio_format}")

            fmt_found = True

        elif chunk_id == b"data":
            if not fmt_found:
                raise ValueError("data chunk found before fmt chunk")

            bytes_per_sample = bits_per_sample // 8
            num_samples = chunk_size // (num_channels * bytes_per_sample)

            return WavMetadata(
                sample_rate=sample_rate,
                num_channels=num_channels,
                bits_per_sample=bits_per_sample,
                num_samples=num_samples,
                duration_seconds=num_samples / sample_rate,
                audio_format=audio_format,
                _data_offset=pos + 8,
            )

        pos += 8 + chunk_size + (chunk_size & 1)
        if total_size is not None and pos >= total_size:
            break

    raise ValueError("No data chunk found")


def _samples_from_bytes(
    audio_bytes: bytes,
    metadata: WavMetadata,
) -> torch.Tensor:
    """
    Convert raw PCM bytes to a float32 tensor.

    Args:
        audio_bytes: Raw PCM audio data bytes
        metadata: WAV metadata for format information
    """
    bytes_per_sample = metadata.bits_per_sample // 8
    num_samples = len(audio_bytes) // bytes_per_sample // metadata.num_channels

    if num_samples == 0:
        return torch.empty((metadata.num_channels, 0), dtype=torch.float32)

    if metadata.audio_format == WAVE_FORMAT_IEEE_FLOAT:
        if metadata.bits_per_sample == 32:
            # f32: Direct frombuffer + deinterleave
            return (
                torch.frombuffer(audio_bytes, dtype=torch.float32)
                .view(num_samples, metadata.num_channels)
                .t()
            )
        elif metadata.bits_per_sample == 64:
            # f64: frombuffer + convert + deinterleave
            return (
                torch.frombuffer(audio_bytes, dtype=torch.float64)
                .to(torch.float32)
                .view(num_samples, metadata.num_channels)
                .t()
            )
        else:
            raise ValueError(f"Unsupported float bits: {metadata.bits_per_sample}")

    elif metadata.audio_format == WAVE_FORMAT_PCM:
        if metadata.bits_per_sample == 16:
            # s16: frombuffer + convert + deinterleave + normalize
            data = (
                torch.frombuffer(audio_bytes, dtype=torch.int16)
                .to(torch.float32)
                .view(num_samples, metadata.num_channels)
                .t()
            )
            data.div_(32768.0)
            return data

        elif metadata.bits_per_sample == 32:
            # s32: frombuffer + convert + deinterleave + normalize
            data = (
                torch.frombuffer(audio_bytes, dtype=torch.int32)
                .to(torch.float32)
                .view(num_samples, metadata.num_channels)
                .t()
            )
            data.div_(2147483648.0)
            return data

        elif metadata.bits_per_sample == 8:
            # u8: frombuffer + convert + deinterleave + normalize
            data = (
                torch.frombuffer(audio_bytes, dtype=torch.uint8)
                .to(torch.float32)
                .view(num_samples, metadata.num_channels)
                .t()
            )
            data.sub_(128.0).div_(128.0)
            return data

        elif metadata.bits_per_sample == 24:
            # s24: Use 24-bit helper + convert + deinterleave
            int32_data = _convert_24bit_pcm(memoryview(audio_bytes))
            data = (
                int32_data.to(torch.float32)
                .view(num_samples, metadata.num_channels)
                .t()
            )
            data.div_(8388608.0)
            return data

        else:
            raise ValueError(f"Unsupported PCM bits: {metadata.bits_per_sample}")
    else:
        raise ValueError(f"Unsupported audio format: {metadata.audio_format}")

    return torch.empty((metadata.num_channels, 0), dtype=torch.float32)


def _supports_buffer_protocol(obj) -> bool:
    """Check if object supports the buffer protocol for direct memory access."""
    try:
        memoryview(obj)
        return True
    except TypeError:
        return False


def _convert_24bit_pcm(data: memoryview | torch.Tensor) -> torch.Tensor:
    """Convert 24-bit PCM bytes to sign-extended int32 tensor.

    Args:
        data: Either a memoryview of raw bytes or a torch.Tensor of uint8 values

    Returns:
        torch.Tensor of int32 values with proper sign extension
    """
    # Handle both input types
    if isinstance(data, torch.Tensor):
        raw = data
    else:
        raw = torch.frombuffer(data, dtype=torch.uint8)

    n = len(raw) // 3

    if n == 0:
        return torch.empty(0, dtype=torch.int32)

    # Use vectorized operations but keep arithmetic right shift for performance
    padded = torch.zeros(n * 4, dtype=torch.uint8)
    padded[1::4] = raw[0::3]  # b0 - first byte of each 24-bit sample
    padded[2::4] = raw[1::3]  # b1 - second byte of each 24-bit sample
    padded[3::4] = raw[2::3]  # b2 - third byte of each 24-bit sample

    # Arithmetic right shift is much faster than torch.where for sign extension
    return padded.view(torch.int32) >> 8


class WavDecoder:
    """Fast WAV decoder that bypasses FFmpeg for simple PCM WAV files.

    This decoder handles different source types natively (like FFmpeg does)
    rather than converting everything to file-like objects:
    - bytes: direct memory access via slicing
    - file path: native file I/O
    - file-like: read/seek callbacks
    """

    def __init__(
        self,
        source: str | Path | io.RawIOBase | io.BufferedReader | bytes | Tensor,
        sample_rate: int | None = None,
        num_channels: int | None = None,
        stream_index: int | None = None,
    ):
        """
        Create a WavDecoder for the given source.

        Raises:
            ValueError: If source is not a valid WAV or doesn't match requirements.
            TypeError: If source type is not supported.
            OSError: If file cannot be opened.
        """
        if stream_index is not None and stream_index != 0:
            raise ValueError("WAV files only have stream index 0")

        # This is only set for file-like objects with getbuffer
        self._use_buffer_access = False

        if isinstance(source, bytes):
            self._source: bytes | io.BufferedReader | io.RawIOBase = source
            wav_metadata = _parse_wav_chunks(
                lambda offset, size: source[offset : offset + size], len(source)
            )

        elif isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() != ".wav":
                raise ValueError(f"Not a .wav file: {path}")
            file_handle = open(path, "rb")
            try:
                wav_metadata = _parse_wav_chunks(
                    lambda offset, size: (
                        file_handle.seek(offset, 0),
                        file_handle.read(size),
                    )[1]
                )
            except ValueError:
                file_handle.close()
                raise
            self._source = file_handle

        elif isinstance(source, (io.RawIOBase, io.BufferedReader)) or (
            hasattr(source, "read") and hasattr(source, "seek")
        ):
            # Try buffer access first for BytesIO objects, fall back to I/O
            if hasattr(source, "getbuffer"):
                try:
                    buffer = source.getbuffer()
                    self._use_buffer_access = True
                except Exception:
                    self._use_buffer_access = False
            else:
                self._use_buffer_access = False

            # Create read function based on available access method
            if self._use_buffer_access:

                def read_func(offset, size):
                    return bytes(buffer[offset : offset + size])

            else:

                def read_func(offset, size):
                    source.seek(offset, 0)
                    return source.read(size)

            wav_metadata = _parse_wav_chunks(read_func)
            self._source = source

        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        if sample_rate is not None and sample_rate != wav_metadata.sample_rate:
            raise ValueError(
                f"Sample rate mismatch: source={wav_metadata.sample_rate}, requested={sample_rate}"
            )
        if num_channels is not None and num_channels != wav_metadata.num_channels:
            raise ValueError(
                f"Channel count mismatch: source={wav_metadata.num_channels}, requested={num_channels}"
            )

        self._wav_metadata = wav_metadata
        self.stream_index = 0
        self.metadata = core._metadata.AudioStreamMetadata(
            duration_seconds_from_header=wav_metadata.duration_seconds,
            duration_seconds=wav_metadata.duration_seconds,
            begin_stream_seconds_from_header=0.0,
            begin_stream_seconds=0.0,
            bit_rate=wav_metadata.sample_rate
            * wav_metadata.num_channels
            * wav_metadata.bits_per_sample,
            codec="pcm",
            stream_index=0,
            sample_rate=wav_metadata.sample_rate,
            num_channels=wav_metadata.num_channels,
            sample_format=f"s{wav_metadata.bits_per_sample}",
        )

    @classmethod
    def try_create(
        cls,
        source: str | Path | io.RawIOBase | io.BufferedReader | bytes | Tensor,
        sample_rate: int | None = None,
        num_channels: int | None = None,
        stream_index: int | None = None,
    ) -> "WavDecoder | None":
        """
        Try to create a WavDecoder for the given source.

        Returns a WavDecoder if successful, None otherwise.
        For file-like objects, the seek position is restored on failure.
        """
        original_pos = None
        if isinstance(source, (io.RawIOBase, io.BufferedReader)):
            try:
                original_pos = source.tell()
            except (OSError, io.UnsupportedOperation):
                pass

        try:
            return cls(source, sample_rate, num_channels, stream_index)
        except (ValueError, TypeError, OSError):
            if original_pos is not None and isinstance(
                source, (io.RawIOBase, io.BufferedReader)
            ):
                try:
                    source.seek(original_pos, 0)
                except (OSError, io.UnsupportedOperation):
                    pass
            return None

    def get_all_samples(self) -> AudioSamples:
        return self.get_samples_played_in_range()

    def get_samples_played_in_range(
        self, start_seconds: float = 0.0, stop_seconds: float | None = None
    ) -> AudioSamples:
        if stop_seconds is not None and not start_seconds <= stop_seconds:
            raise ValueError(
                f"Invalid start seconds: {start_seconds}. "
                f"It must be less than or equal to stop seconds ({stop_seconds})."
            )

        sample_rate = self._wav_metadata.sample_rate
        bytes_per_sample = self._wav_metadata.bits_per_sample // 8
        bytes_per_frame = bytes_per_sample * self._wav_metadata.num_channels

        # Calculate sample range
        start_sample = round(start_seconds * sample_rate)
        end_sample = (
            round(stop_seconds * sample_rate)
            if stop_seconds is not None
            else self._wav_metadata.num_samples
        )

        # Clamp to valid range
        start_sample = max(0, min(start_sample, self._wav_metadata.num_samples))
        end_sample = max(start_sample, min(end_sample, self._wav_metadata.num_samples))
        num_samples = end_sample - start_sample

        if num_samples == 0:
            # Return empty tensor with correct shape
            data = torch.empty(
                (self._wav_metadata.num_channels, 0), dtype=torch.float32
            )
            return AudioSamples(
                data=data,
                pts_seconds=start_seconds,
                duration_seconds=0.0,
                sample_rate=sample_rate,
            )

        # Calculate byte offset and read only the needed bytes
        byte_offset = self._wav_metadata._data_offset + start_sample * bytes_per_frame
        num_bytes = num_samples * bytes_per_frame

        # Fast path for buffer-like sources: use torch.frombuffer with offset for common formats
        if _supports_buffer_protocol(self._source):
            if (
                self._wav_metadata.audio_format == WAVE_FORMAT_IEEE_FLOAT
                and self._wav_metadata.bits_per_sample == 32
            ):
                # f32: Direct frombuffer with offset - no expensive slice
                float_count = num_bytes // 4
                data = torch.frombuffer(
                    self._source,
                    dtype=torch.float32,
                    offset=byte_offset,
                    count=float_count,
                ).view(self._wav_metadata.num_channels, -1)

            elif (
                self._wav_metadata.audio_format == WAVE_FORMAT_IEEE_FLOAT
                and self._wav_metadata.bits_per_sample == 64
            ):
                # f64: Direct frombuffer, then convert to f32
                double_count = num_bytes // 8
                data = (
                    torch.frombuffer(
                        self._source,
                        dtype=torch.float64,
                        offset=byte_offset,
                        count=double_count,
                    )
                    .view(self._wav_metadata.num_channels, -1)
                    .to(torch.float32)
                )

            elif (
                self._wav_metadata.audio_format == WAVE_FORMAT_PCM
                and self._wav_metadata.bits_per_sample == 16
            ):
                # s16: Direct frombuffer, then normalize
                int16_count = num_bytes // 2
                data = (
                    torch.frombuffer(
                        self._source,
                        dtype=torch.int16,
                        offset=byte_offset,
                        count=int16_count,
                    )
                    .view(self._wav_metadata.num_channels, -1)
                    .to(torch.float32)
                )
                data.div_(32768.0)  # Normalize s16 to [-1, 1]

            elif (
                self._wav_metadata.audio_format == WAVE_FORMAT_PCM
                and self._wav_metadata.bits_per_sample == 32
            ):
                # s32: Direct frombuffer, then normalize
                int32_count = num_bytes // 4
                data = (
                    torch.frombuffer(
                        self._source,
                        dtype=torch.int32,
                        offset=byte_offset,
                        count=int32_count,
                    )
                    .view(self._wav_metadata.num_channels, -1)
                    .to(torch.float32)
                )
                data.div_(2147483648.0)  # Normalize s32 to [-1, 1]

            elif (
                self._wav_metadata.audio_format == WAVE_FORMAT_PCM
                and self._wav_metadata.bits_per_sample == 8
            ):
                # u8: Direct frombuffer, then normalize
                uint8_count = num_bytes
                data = (
                    torch.frombuffer(
                        self._source,
                        dtype=torch.uint8,
                        offset=byte_offset,
                        count=uint8_count,
                    )
                    .view(self._wav_metadata.num_channels, -1)
                    .to(torch.float32)
                )
                data.sub_(128.0).div_(128.0)  # Normalize u8 to [-1, 1]

            elif (
                self._wav_metadata.audio_format == WAVE_FORMAT_PCM
                and self._wav_metadata.bits_per_sample == 24
            ):
                # s24: Zero-copy path matching other formats (s16/s32/f32/f64)
                uint8_count = num_bytes
                raw_uint8 = torch.frombuffer(
                    self._source,
                    dtype=torch.uint8,
                    offset=byte_offset,  # Zero-copy: just offset pointer
                    count=uint8_count,  # Direct element count
                )
                int32_samples = _convert_24bit_pcm(raw_uint8)
                data = int32_samples.view(self._wav_metadata.num_channels, -1).to(
                    torch.float32
                )
                data.div_(8388608.0)  # Normalize s24 to [-1, 1]

            else:
                # Fallback for truly unsupported formats
                audio_bytes = self._source[byte_offset : byte_offset + num_bytes]
                data = _samples_from_bytes(audio_bytes, self._wav_metadata)
        else:
            # File-like source: use zero-copy buffer access when possible
            if self._use_buffer_access:
                # Zero-copy path using getbuffer() - eliminates ALL seek/read overhead
                try:
                    buffer = self._source.getbuffer()
                    audio_buffer = buffer[byte_offset : byte_offset + num_bytes]

                    # Convert directly from memoryview (zero copy)
                    data = _samples_from_bytes(audio_buffer, self._wav_metadata)
                except Exception:
                    # Fallback to regular I/O if getbuffer fails
                    self._use_buffer_access = False
                    self._source.seek(byte_offset, 0)
                    audio_bytes = self._source.read(num_bytes)
                    data = _samples_from_bytes(audio_bytes, self._wav_metadata)
            else:
                # Regular I/O path for file-like objects without buffer access
                self._source.seek(byte_offset, 0)
                audio_bytes = self._source.read(num_bytes)

                if len(audio_bytes) < num_bytes:
                    # Adjust num_samples for partial read
                    actual_bytes = len(audio_bytes)
                    num_samples = (
                        actual_bytes
                        // (self._wav_metadata.bits_per_sample // 8)
                        // self._wav_metadata.num_channels
                    )

                data = _samples_from_bytes(audio_bytes, self._wav_metadata)
        return AudioSamples(
            data=data,
            pts_seconds=start_seconds,
            duration_seconds=num_samples / sample_rate,
            sample_rate=sample_rate,
        )
