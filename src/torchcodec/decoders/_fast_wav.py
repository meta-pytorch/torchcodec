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


def _read_wav_header(data: bytes) -> WavMetadata:
    """Parse WAV header and return metadata including data offset."""
    if len(data) < 44:
        raise ValueError("File too small to be a valid WAV")

    if data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Not a valid WAV file")

    # Iterate through chunks to find fmt and data
    pos = 12
    fmt_found = False
    audio_format = 0
    num_channels = 0
    sample_rate = 0
    bits_per_sample = 0

    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        chunk_size = struct.unpack("<I", data[pos + 4 : pos + 8])[0]

        if chunk_id == b"fmt ":
            if chunk_size < 16:
                raise ValueError("Invalid fmt chunk size")

            audio_format, num_channels, sample_rate, _, _, bits_per_sample = (
                struct.unpack("<HHIIHH", data[pos + 8 : pos + 24])
            )

            # Handle WAVE_FORMAT_EXTENSIBLE - extract actual format from SubFormat GUID
            if audio_format == WAVE_FORMAT_EXTENSIBLE:
                # Extensible format requires at least 40 bytes (16 base + 2 cbSize + 22 extension)
                if chunk_size < 40:
                    raise ValueError("Invalid extensible fmt chunk size")
                # SubFormat GUID starts at offset 24 from chunk start (pos + 8 + 24 = pos + 32)
                # First two bytes of GUID encode the actual format (1=PCM, 3=float)
                audio_format = struct.unpack("<H", data[pos + 32 : pos + 34])[0]

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

        # Move to next chunk (chunks are word-aligned)
        pos += 8 + chunk_size + (chunk_size & 1)

    raise ValueError("No data chunk found")


def decode_wav_from_metadata(
    data: bytes,
    metadata: WavMetadata,
) -> torch.Tensor:
    """
    Decode WAV audio data using pre-parsed metadata.

    This is the optimized path that avoids re-parsing the header.
    """
    # Use memoryview for zero-copy access to the audio data
    bytes_per_sample = metadata.bits_per_sample // 8
    data_size = metadata.num_samples * metadata.num_channels * bytes_per_sample
    audio_view = memoryview(data)[
        metadata._data_offset : metadata._data_offset + data_size
    ]

    # Convert to tensor based on format
    if metadata.audio_format == WAVE_FORMAT_IEEE_FLOAT:
        if metadata.bits_per_sample == 32:
            # Interpret raw bytes as float32 (already normalized by convention)
            samples = torch.frombuffer(audio_view, dtype=torch.float32).clone()
        elif metadata.bits_per_sample == 64:
            # Interpret raw bytes as float64, then convert to float32
            samples = torch.frombuffer(audio_view, dtype=torch.float64).to(
                torch.float32
            )
        else:
            raise ValueError(f"Unsupported float bits: {metadata.bits_per_sample}")
    elif metadata.audio_format == WAVE_FORMAT_PCM:
        if metadata.bits_per_sample == 16:
            # Interpret raw bytes as int16
            int_samples = torch.frombuffer(audio_view, dtype=torch.int16)
            # Convert to float32, then normalize from [-32768, 32767] to [-1, 1]
            samples = int_samples.to(torch.float32).div_(
                torch.iinfo(torch.int16).max + 1
            )
        elif metadata.bits_per_sample == 32:
            # Interpret raw bytes as int32
            int_samples = torch.frombuffer(audio_view, dtype=torch.int32)
            # Convert to float32, then normalize from [-2^31, 2^31-1] to [-1, 1]
            samples = int_samples.to(torch.float32).div_(
                torch.iinfo(torch.int32).max + 1
            )
        elif metadata.bits_per_sample == 24:
            # There is no 24-bit dtype, so we use helper function
            samples = _decode_24bit_pcm(audio_view)
        elif metadata.bits_per_sample == 8:
            # Interpret raw bytes as uint8
            uint_samples = torch.frombuffer(audio_view, dtype=torch.uint8)
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
        samples = (
            samples.view(metadata.num_samples, metadata.num_channels).t().contiguous()
        )

    return samples


def decode_wav(
    data: bytes,
    target_sample_rate: int | None = None,
    target_num_channels: int | None = None,
) -> tuple[torch.Tensor, WavMetadata]:
    """
    Decode a WAV file from bytes to a float32 tensor.
    """
    metadata = _read_wav_header(data)

    if target_sample_rate is not None and target_sample_rate != metadata.sample_rate:
        raise ValueError(
            f"Resampling not supported. Source: {metadata.sample_rate}Hz, target: {target_sample_rate}Hz"
        )

    if target_num_channels is not None and target_num_channels != metadata.num_channels:
        raise ValueError(
            f"Channel conversion not supported. Source: {metadata.num_channels}, target: {target_num_channels}"
        )

    samples = decode_wav_from_metadata(data, metadata)
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


def is_wav_bytes(data: bytes) -> bool:
    """Check if the given bytes appear to be a WAV file."""
    return len(data) >= 12 and data[0:4] == b"RIFF" and data[8:12] == b"WAVE"


class WavDecoder:
    """Fast WAV decoder that bypasses FFmpeg for simple PCM WAV files."""

    def __init__(self, source_bytes: bytes, wav_metadata: WavMetadata):
        self._source_bytes = source_bytes
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
        """
        # WAV files only have one audio stream at index 0
        if stream_index is not None and stream_index != 0:
            return None

        source_bytes: bytes | None = None

        if isinstance(source, bytes):
            source_bytes = source
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() == ".wav":
                try:
                    with open(path, "rb") as f:
                        source_bytes = f.read()
                except OSError:
                    return None
        elif isinstance(source, (io.RawIOBase, io.BufferedReader)) or (
            hasattr(source, "read") and hasattr(source, "seek")
        ):
            source_bytes = source.read()
            # Will reset seek position below if we can't use fast path

        if source_bytes is None:
            return None

        if not is_wav_bytes(source_bytes):
            # Reset file-like object seek position for FFmpeg fallback
            if hasattr(source, "seek"):
                source.seek(0)
            return None

        try:
            wav_metadata = _read_wav_header(source_bytes)
        except ValueError:
            if hasattr(source, "seek"):
                source.seek(0)
            return None

        if sample_rate is not None and sample_rate != wav_metadata.sample_rate:
            if hasattr(source, "seek"):
                source.seek(0)
            return None

        if num_channels is not None and num_channels != wav_metadata.num_channels:
            if hasattr(source, "seek"):
                source.seek(0)
            return None

        return cls(source_bytes, wav_metadata)

    def get_all_samples(self) -> AudioSamples:
        """Returns all the audio samples from the source."""
        return self.get_samples_played_in_range()

    def get_samples_played_in_range(
        self, start_seconds: float = 0.0, stop_seconds: float | None = None
    ) -> AudioSamples:
        """Returns audio samples in the given range."""
        if stop_seconds is not None and not start_seconds <= stop_seconds:
            raise ValueError(
                f"Invalid start seconds: {start_seconds}. "
                f"It must be less than or equal to stop seconds ({stop_seconds})."
            )

        samples = decode_wav_from_metadata(self._source_bytes, self._wav_metadata)
        sample_rate = self._wav_metadata.sample_rate

        start_sample = round(start_seconds * sample_rate)
        end_sample = (
            round(stop_seconds * sample_rate) if stop_seconds else samples.shape[1]
        )

        data = samples[:, start_sample:end_sample]
        return AudioSamples(
            data=data,
            pts_seconds=start_seconds,
            duration_seconds=data.shape[1] / sample_rate,
            sample_rate=sample_rate,
        )
