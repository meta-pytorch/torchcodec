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
    read_at: callable,
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


def _read_wav_header(source: io.IOBase) -> WavMetadata:
    """Parse WAV header from a seekable source."""

    def read_at(offset: int, size: int) -> bytes:
        source.seek(offset, 0)
        return source.read(size)

    return _parse_wav_chunks(read_at)


def _decode_samples_from_bytes(
    audio_bytes: bytes,
    metadata: WavMetadata,
    num_samples: int,
) -> torch.Tensor:
    """
    Decode raw PCM bytes to a float32 tensor.

    Args:
        audio_bytes: Raw PCM audio data bytes
        metadata: WAV metadata for format information
        num_samples: Number of samples (per channel) in audio_bytes
    """
    # Convert to tensor based on format
    if metadata.audio_format == WAVE_FORMAT_IEEE_FLOAT:
        if metadata.bits_per_sample == 32:
            # Interpret raw bytes as float32 (already normalized by convention)
            samples = torch.frombuffer(audio_bytes, dtype=torch.float32).clone()
        elif metadata.bits_per_sample == 64:
            # Interpret raw bytes as float64, then convert to float32
            samples = torch.frombuffer(audio_bytes, dtype=torch.float64).to(
                torch.float32
            )
        else:
            raise ValueError(f"Unsupported float bits: {metadata.bits_per_sample}")
    elif metadata.audio_format == WAVE_FORMAT_PCM:
        if metadata.bits_per_sample == 16:
            # Interpret raw bytes as int16
            int_samples = torch.frombuffer(audio_bytes, dtype=torch.int16)
            # Convert to float32, then normalize from [-32768, 32767] to [-1, 1]
            samples = int_samples.to(torch.float32).div_(
                torch.iinfo(torch.int16).max + 1
            )
        elif metadata.bits_per_sample == 32:
            # Interpret raw bytes as int32
            int_samples = torch.frombuffer(audio_bytes, dtype=torch.int32)
            # Convert to float32, then normalize from [-2^31, 2^31-1] to [-1, 1]
            samples = int_samples.to(torch.float32).div_(
                torch.iinfo(torch.int32).max + 1
            )
        elif metadata.bits_per_sample == 24:
            # There is no 24-bit dtype, so we use helper function
            samples = _decode_24bit_pcm(memoryview(audio_bytes))
        elif metadata.bits_per_sample == 8:
            # Interpret raw bytes as uint8
            uint_samples = torch.frombuffer(audio_bytes, dtype=torch.uint8)
            # Convert to float32, then normalize from [0, 255] to [-1, 1]
            samples = uint_samples.to(torch.float32).sub_(128.0).div_(128.0)
        else:
            raise ValueError(f"Unsupported PCM bits: {metadata.bits_per_sample}")
    else:
        raise ValueError(f"Unsupported audio format: {metadata.audio_format}")

    # Reshape to (num_channels, num_samples) - contiguous for efficiency
    if metadata.num_channels == 1:
        samples = samples.unsqueeze(0)
    else:
        samples = samples.view(num_samples, metadata.num_channels).t().contiguous()

    return samples


def decode_wav(
    source: io.IOBase,
    target_sample_rate: int | None = None,
    target_num_channels: int | None = None,
) -> tuple[torch.Tensor, WavMetadata]:
    """
    Decode a WAV file from a seekable source to a float32 tensor.
    """
    metadata = _read_wav_header(source)

    if target_sample_rate is not None and target_sample_rate != metadata.sample_rate:
        raise ValueError(
            f"Resampling not supported. Source: {metadata.sample_rate}Hz, target: {target_sample_rate}Hz"
        )

    if target_num_channels is not None and target_num_channels != metadata.num_channels:
        raise ValueError(
            f"Channel conversion not supported. Source: {metadata.num_channels}, target: {target_num_channels}"
        )

    # Read all audio data
    bytes_per_sample = metadata.bits_per_sample // 8
    data_size = metadata.num_samples * metadata.num_channels * bytes_per_sample
    source.seek(metadata._data_offset, 0)
    audio_bytes = source.read(data_size)

    samples = _decode_samples_from_bytes(audio_bytes, metadata, metadata.num_samples)
    return samples, metadata


def _decode_24bit_pcm(data: memoryview) -> torch.Tensor:
    """Decode 24-bit PCM samples to float32 tensor."""
    # Interpret raw bytes as uint8, reshape to (num_samples, 3)
    raw = torch.frombuffer(data, dtype=torch.uint8).reshape(-1, 3)

    # Combine 3 bytes into int32: byte0 | (byte1 << 8) | (byte2 << 16)
    samples_i32 = (
        raw[:, 0].to(torch.int32)
        | (raw[:, 1].to(torch.int32) << 8)
        | (raw[:, 2].to(torch.int32) << 16)
    )

    # Sign extend from 24 to 32 bits
    samples_i32 = torch.where(
        (samples_i32 & 0x800000) != 0,
        samples_i32 - 0x1000000,
        samples_i32,
    )

    # Convert to float32, then normalize from [-2^23, 2^23-1] to [-1, 1]
    return samples_i32.to(torch.float32) / 8388608.0


def is_wav_source(source: io.IOBase) -> bool:
    """Check if the given seekable source appears to be a WAV file."""
    source.seek(0, 0)
    header = source.read(12)
    source.seek(0, 0)
    return len(header) >= 12 and header[0:4] == b"RIFF" and header[8:12] == b"WAVE"


def is_wav_bytes(data: bytes) -> bool:
    """Check if the given bytes appear to be a WAV file."""
    return len(data) >= 12 and data[0:4] == b"RIFF" and data[8:12] == b"WAVE"


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
        source: bytes | io.IOBase,
        wav_metadata: WavMetadata,
    ):
        self._source = source
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
    def validate_and_init(
        cls,
        source: str | Path | io.RawIOBase | io.BufferedReader | bytes | Tensor,
        sample_rate: int | None = None,
        num_channels: int | None = None,
        stream_index: int | None = None,
    ) -> "WavDecoder | None":
        """
        Try to create a WavDecoder for the given source.

        Returns a WavDecoder if the fast path can be used, None otherwise.
        For file-like objects, the seek position is reset if we return None.

        Handles each source type natively (like FFmpeg's SingleStreamDecoder):
        - bytes: direct slicing (no BytesIO wrapper)
        - file path: native file I/O
        - file-like: read/seek callbacks
        """
        # WAV files only have one audio stream at index 0
        if stream_index is not None and stream_index != 0:
            return None

        # Skip text-mode files - let the main decoder path handle the error message
        if isinstance(source, io.TextIOBase):
            return None

        # Handle bytes directly (like FFmpeg's AVIOFromTensorContext)
        if isinstance(source, bytes):
            if not is_wav_bytes(source):
                return None
            try:
                wav_metadata = _parse_wav_chunks(
                    lambda offset, size: source[offset : offset + size], len(source)
                )
            except ValueError:
                return None
            if sample_rate is not None and sample_rate != wav_metadata.sample_rate:
                return None
            if num_channels is not None and num_channels != wav_metadata.num_channels:
                return None
            return cls(source, wav_metadata)

        # Handle file path (like FFmpeg's avformat_open_input with filename)
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() != ".wav":
                return None
            try:
                file_handle = open(path, "rb")
            except OSError:
                return None

            wav_metadata = None
            try:
                if is_wav_source(file_handle):
                    wav_metadata = _read_wav_header(file_handle)
            except ValueError:
                pass

            if (
                wav_metadata is None
                or (sample_rate is not None and sample_rate != wav_metadata.sample_rate)
                or (
                    num_channels is not None
                    and num_channels != wav_metadata.num_channels
                )
            ):
                file_handle.close()
                return None

            return cls(file_handle, wav_metadata)

        # Handle file-like objects (like FFmpeg's AVIOFileLikeContext)
        if isinstance(source, (io.RawIOBase, io.BufferedReader)) or (
            hasattr(source, "read") and hasattr(source, "seek")
        ):
            wav_metadata = None
            try:
                if is_wav_source(source):
                    wav_metadata = _read_wav_header(source)
            except ValueError:
                pass

            if (
                wav_metadata is None
                or (sample_rate is not None and sample_rate != wav_metadata.sample_rate)
                or (
                    num_channels is not None
                    and num_channels != wav_metadata.num_channels
                )
            ):
                source.seek(0, 0)
                return None

            return cls(source, wav_metadata)

        return None

    def get_all_samples(self) -> AudioSamples:
        """Returns all the audio samples from the source."""
        return self.get_samples_played_in_range()

    def get_samples_played_in_range(
        self, start_seconds: float = 0.0, stop_seconds: float | None = None
    ) -> AudioSamples:
        """Returns audio samples in the given range.

        This method reads only the bytes needed for the requested range,
        enabling efficient streaming from large or remote files.
        """
        if stop_seconds is not None and not start_seconds <= stop_seconds:
            raise ValueError(
                f"Invalid start seconds: {start_seconds}. "
                f"It must be less than or equal to stop seconds ({stop_seconds})."
            )

        metadata = self._wav_metadata
        sample_rate = metadata.sample_rate
        bytes_per_sample = metadata.bits_per_sample // 8
        bytes_per_frame = bytes_per_sample * metadata.num_channels

        # Calculate sample range
        start_sample = round(start_seconds * sample_rate)
        end_sample = (
            round(stop_seconds * sample_rate)
            if stop_seconds is not None
            else metadata.num_samples
        )

        # Clamp to valid range
        start_sample = max(0, min(start_sample, metadata.num_samples))
        end_sample = max(start_sample, min(end_sample, metadata.num_samples))
        num_samples = end_sample - start_sample

        if num_samples == 0:
            # Return empty tensor with correct shape
            data = torch.empty((metadata.num_channels, 0), dtype=torch.float32)
            return AudioSamples(
                data=data,
                pts_seconds=start_seconds,
                duration_seconds=0.0,
                sample_rate=sample_rate,
            )

        # Calculate byte offset and read only the needed bytes
        byte_offset = metadata._data_offset + start_sample * bytes_per_frame
        num_bytes = num_samples * bytes_per_frame

        # Handle bytes via slicing, file-like via seek/read
        if isinstance(self._source, bytes):
            audio_bytes = self._source[byte_offset : byte_offset + num_bytes]
        else:
            self._source.seek(byte_offset, 0)
            audio_bytes = self._source.read(num_bytes)

        data = _decode_samples_from_bytes(audio_bytes, metadata, num_samples)
        return AudioSamples(
            data=data,
            pts_seconds=start_seconds,
            duration_seconds=num_samples / sample_rate,
            sample_rate=sample_rate,
        )
