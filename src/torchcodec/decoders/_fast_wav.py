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
import os
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
            # s24: Use 24-bit conversion + convert + deinterleave
            np_data = np.frombuffer(audio_bytes, dtype=np.uint8)
            int32_data = _convert_24bit(np_data)
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


def _supports_buffer_protocol(obj) -> bool:
    """Check if object supports the buffer protocol for direct memory access."""
    try:
        memoryview(obj)
        return True
    except TypeError:
        return False


def _convert_24bit(data: np.ndarray) -> torch.Tensor:
    """Convert 24-bit PCM numpy array to sign-extended int32 tensor.

    Optimized numpy version for file I/O path.

    Args:
        data: numpy array of uint8 values (raw 24-bit bytes)

    Returns:
        torch.Tensor of int32 values with proper sign extension
    """
    n = len(data) // 3

    if n == 0:
        return torch.empty(0, dtype=torch.int32)

    padded = np.zeros(n * 4, dtype=np.uint8)
    padded[1::4] = data[0::3]  # b0 - first byte of each 24-bit sample
    padded[2::4] = data[1::3]  # b1 - second byte of each 24-bit sample
    padded[3::4] = data[2::3]  # b2 - third byte of each 24-bit sample

    # View as int32 and apply arithmetic right shift for sign extension
    int32_data = padded.view(np.int32) >> 8
    return torch.from_numpy(int32_data)


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
        self._file_path = None

        if isinstance(source, bytes):
            self._source: bytes | io.BufferedReader | io.RawIOBase = source
            wav_metadata = _parse_wav_chunks(
                lambda offset, size: source[offset : offset + size], len(source)
            )

        elif isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() != ".wav":
                raise ValueError(f"Not a .wav file: {path}")

            # Parse header and use numpy file I/O for optimal performance
            file_size = os.path.getsize(path)
            if file_size <= 0:
                raise ValueError("Empty file")

            # Read header portion for format detection
            with open(path, "rb") as f:
                header_size = min(1024, file_size)  # Read first 1KB for header
                header_bytes = f.read(header_size)

            wav_metadata = _parse_wav_chunks(
                lambda offset, size: header_bytes[offset : offset + size],
                len(header_bytes),
            )

            self._file_path = os.path.abspath(str(path))
            self._source = None  # Will use direct file reading

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

        elif isinstance(source, Tensor):
            # Handle tensor inputs directly
            if source.dtype != torch.uint8:
                raise ValueError("Tensor source must be uint8 dtype")
            # We will expect the tensor to be contiguous for slicing and frombuffer
            if not source.is_contiguous():
                source = source.contiguous()

            self._source = source
            wav_metadata = _parse_wav_chunks(
                lambda offset, size: source[offset : offset + size].numpy().tobytes(),
                len(source),
            )

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

    @dataclass
    class SampleRange:
        """Helper class to bundle sample range calculations."""

        start_sample: int
        end_sample: int
        num_samples: int
        byte_offset: int
        num_bytes: int

    def _calculate_sample_range(
        self, start_seconds: float, stop_seconds: float | None
    ) -> SampleRange:
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

        byte_offset = self._wav_metadata._data_offset + start_sample * bytes_per_frame
        num_bytes = num_samples * bytes_per_frame

        return self.SampleRange(
            start_sample=start_sample,
            end_sample=end_sample,
            num_samples=num_samples,
            byte_offset=byte_offset,
            num_bytes=num_bytes,
        )

    def _convert_raw_audio_data(self, raw_tensor: torch.Tensor) -> torch.Tensor:
        """Apply format-specific conversions to raw audio tensor."""
        fmt = self._wav_metadata

        if fmt.audio_format == WAVE_FORMAT_IEEE_FLOAT:
            if fmt.bits_per_sample == 32:
                # F32: no conversion needed
                return raw_tensor.view(fmt.num_channels, -1)
            elif fmt.bits_per_sample == 64:
                # F64: convert to float32
                return raw_tensor.to(torch.float32).view(fmt.num_channels, -1)

        elif fmt.audio_format == WAVE_FORMAT_PCM:
            if fmt.bits_per_sample == 32:
                # S32: convert + normalize
                data = raw_tensor.to(torch.float32).view(fmt.num_channels, -1)
                data.div_(2147483648.0)
                return data
            elif fmt.bits_per_sample == 16:
                # S16: convert + normalize
                data = raw_tensor.to(torch.float32).view(fmt.num_channels, -1)
                data.div_(32768.0)
                return data
            elif fmt.bits_per_sample == 8:
                # U8: convert + normalize
                data = raw_tensor.to(torch.float32).view(fmt.num_channels, -1)
                data.sub_(128.0).div_(128.0)
                return data
            elif fmt.bits_per_sample == 24:
                # S24: 24-bit conversion + normalize (raw_tensor is uint8)
                np_data = raw_tensor.numpy()
                int32_data = _convert_24bit(np_data)
                data = int32_data.to(torch.float32).view(fmt.num_channels, -1)
                data.div_(8388608.0)
                return data

        raise ValueError(
            f"Unsupported format: {fmt.audio_format}, {fmt.bits_per_sample}"
        )

    def _get_samples_data(self, sample_range: SampleRange) -> torch.Tensor:
        if self._file_path is not None or _supports_buffer_protocol(self._source):
            return self._handle_files_and_bytes(sample_range)

        # File-like objects (uses _samples_from_bytes)
        else:
            return self._handle_file_objects(sample_range)

    def _handle_file_objects(self, sample_range: SampleRange) -> torch.Tensor:
        """Handle file objects (BytesIO, file handles) using _samples_from_bytes."""
        if self._use_buffer_access:
            # Zero-copy path using getbuffer()
            try:
                buffer = self._source.getbuffer()
                audio_buffer = buffer[
                    sample_range.byte_offset : sample_range.byte_offset
                    + sample_range.num_bytes
                ]
                # Convert directly from memoryview (zero copy)
                return _samples_from_bytes(audio_buffer, self._wav_metadata)
            except Exception:
                # Fallback to regular I/O if getbuffer fails
                self._use_buffer_access = False

        # Regular I/O path for file-like objects without buffer access
        self._source.seek(sample_range.byte_offset, 0)
        audio_bytes = self._source.read(sample_range.num_bytes)

        return _samples_from_bytes(audio_bytes, self._wav_metadata)

    def _handle_files_and_bytes(self, sample_range: SampleRange) -> torch.Tensor:
        """Handle file paths and raw bytes using unified numpy approach."""
        fmt = self._wav_metadata

        # Determine numpy dtype and element size
        if fmt.audio_format == WAVE_FORMAT_IEEE_FLOAT:
            if fmt.bits_per_sample == 32:
                dtype, element_size = np.float32, 4
            elif fmt.bits_per_sample == 64:
                dtype, element_size = np.float64, 8
        elif fmt.audio_format == WAVE_FORMAT_PCM:
            if fmt.bits_per_sample == 32:
                dtype, element_size = np.int32, 4
            elif fmt.bits_per_sample == 16:
                dtype, element_size = np.int16, 2
            elif fmt.bits_per_sample == 8:
                dtype, element_size = np.uint8, 1
            elif fmt.bits_per_sample == 24:
                dtype, element_size = np.uint8, 1  # S24 loads as uint8

        count = sample_range.num_bytes // element_size

        # Load via numpy (file vs buffer)
        if self._file_path is not None:
            np_data = np.fromfile(
                self._file_path,
                dtype=dtype,
                count=count,
                offset=sample_range.byte_offset,
            )
        else:  # buffer protocol
            np_data = np.frombuffer(
                self._source, dtype=dtype, count=count, offset=sample_range.byte_offset
            )

        raw_tensor = torch.from_numpy(np_data)
        return self._convert_raw_audio_data(raw_tensor)

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
        sample_range = self._calculate_sample_range(start_seconds, stop_seconds)

        if sample_range.num_samples == 0:
            data = torch.empty(
                (self._wav_metadata.num_channels, 0), dtype=torch.float32
            )
            return AudioSamples(
                data=data,
                pts_seconds=start_seconds,
                duration_seconds=0.0,
                sample_rate=self._wav_metadata.sample_rate,
            )

        data = self._get_samples_data(sample_range)
        return AudioSamples(
            data=data,
            pts_seconds=sample_range.start_sample / self._wav_metadata.sample_rate,
            duration_seconds=sample_range.num_samples / self._wav_metadata.sample_rate,
            sample_rate=self._wav_metadata.sample_rate,
        )
