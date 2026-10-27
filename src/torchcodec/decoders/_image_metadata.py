# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import struct
from dataclasses import dataclass
from pathlib import Path

import torch
from torchcodec._core import ops
from torchcodec.decoders._image_decoders import (
    _detect_image_format_from_header,
    _source_to_tensor,
)


@dataclass(frozen=True)
class ImageMetadata:
    """Metadata of an encoded image, as returned by :func:`get_image_metadata`.

    Every attribute describes the tensor that :func:`decode_image` (or the
    matching format-specific decoder) returns for this image.
    """

    format: str
    """The image format: one of ``"jpeg"``, ``"png"``, ``"webp"``, ``"gif"``,
    ``"avif"`` or ``"heic"``."""
    width: int
    """Width of the decoded image, in pixels. This is the **post-rotation**
    width: for 90 or -90 degree :attr:`rotation`, width and height are swapped
    compared to the dimensions stored in the file."""
    height: int
    """Height of the decoded image, in pixels. Like :attr:`width`, this is the
    **post-rotation** height."""
    num_channels: int
    """Number of channels of the decoded image when decoding with
    ``mode="UNCHANGED"``."""
    bit_depth: int
    """Number of bits per channel in the encoded image. When greater than 8,
    decoding with ``output_dtype="auto"`` returns a ``torch.uint16`` tensor."""
    num_frames: int
    """Number of frames (or images) in the file: 1 for still images, and the
    size of the leading ``N`` dimension of the decoded tensor for animated or
    multi-image files."""
    rotation: float | None
    """Rotation angle in degrees (counter-clockwise: one of 0, 90, -90 or 180)
    from the orientation metadata (EXIF orientation, or the AVIF/HEIC rotation
    and mirroring properties), or None if the image has no orientation
    metadata. The decoders always apply it, so the returned images are in the
    correct orientation. This uses the same convention as
    :attr:`VideoStreamMetadata.rotation
    <torchcodec.decoders.VideoStreamMetadata.rotation>`.

    The orientation metadata can also specify a mirroring, which the decoders
    apply too but isn't reported here, since it doesn't change the image
    dimensions. In that case, this is the rotation applied after mirroring the
    image left to right."""


class _ByteSource:
    # Random access over the encoded bytes. Paths are read lazily, so probing a
    # large file usually only reads its first few KB.

    def __init__(self, source):
        self._file = None
        if isinstance(source, (str, Path)):
            self._file = open(source, "rb")
            self._buf = None
            self.size = Path(source).stat().st_size
        elif isinstance(source, (bytes, bytearray)):
            self._buf = memoryview(source)
            self.size = len(source)
        elif isinstance(source, torch.Tensor):
            if (
                source.dtype != torch.uint8
                or source.ndim != 1
                or source.device.type != "cpu"
            ):
                raise ValueError(
                    "Input tensor must be a 1-dimensional uint8 CPU tensor, got "
                    f"dtype={source.dtype}, ndim={source.ndim}, device={source.device}."
                )
            self._buf = memoryview(source.contiguous().numpy())
            self.size = len(self._buf)
        else:
            raise TypeError(
                f"Unknown source type: {type(source)}. "
                "Supported types are str, Path, bytes and torch.Tensor."
            )

    def read(self, offset: int, size: int) -> bytes:
        # Returns fewer than `size` bytes when reading past the end.
        if self._buf is not None:
            return bytes(self._buf[offset : offset + size])
        assert self._file is not None
        self._file.seek(offset)
        return self._file.read(size)

    def read_exact(self, offset: int, size: int) -> bytes:
        data = self.read(offset, size)
        if len(data) != size:
            raise RuntimeError("Image is truncated or corrupted.")
        return data

    def close(self):
        if self._file is not None:
            self._file.close()


def _swaps_width_and_height(orientation: int | None) -> bool:
    # EXIF orientations 5 to 8 involve a 90 or 270 degree rotation.
    return orientation is not None and 5 <= orientation <= 8


_EXIF_INCORRECT_TAG = 0xFFFF


def _exif_orientation(tiff: bytes) -> int | None:
    # Port of fetch_exif_orientation() from Exif.h, quirks included, so that we
    # report exactly the orientation the decoders apply. `tiff` starts at the
    # TIFF header (i.e. after any "Exif\0\0" prefix). Returns the EXIF
    # orientation in [1, 8], or None if there's none (or it's invalid, in which
    # case the decoders don't apply anything either).
    n = len(tiff)
    byte_order = "<" if tiff[:2] == b"II" else ">"

    def u16(offset):
        if offset + 1 >= n:
            return _EXIF_INCORRECT_TAG
        return struct.unpack_from(byte_order + "H", tiff, offset)[0]

    if u16(2) != 0x2A:
        return None
    offset = (
        _EXIF_INCORRECT_TAG
        if 7 >= n
        else struct.unpack_from(byte_order + "I", tiff, 4)[0]
    )
    num_entries = u16(offset)
    offset += 2
    for _ in range(num_entries):
        tag = u16(offset)
        if tag == _EXIF_INCORRECT_TAG:
            break
        if tag == 0x0112:
            orientation = u16(offset + 8)
            return orientation if 1 <= orientation <= 8 else None
        offset += 12
    return None


# ----------------------------------------------------------------------------
# JPEG
# ----------------------------------------------------------------------------

# SOFn markers that libjpeg-turbo can decode. The other SOFn markers
# (hierarchical, and the reserved JPG marker) make jpeg_read_header() fail.
_JPEG_SUPPORTED_SOF = frozenset((0xC0, 0xC1, 0xC2, 0xC3, 0xC9, 0xCA, 0xCB))
_JPEG_UNSUPPORTED_SOF = frozenset((0xC5, 0xC6, 0xC7, 0xC8, 0xCD, 0xCE, 0xCF))


def _probe_jpeg(src: _ByteSource) -> tuple:
    # Walks the markers up to the first SOS, like jpeg_read_header(). The
    # decoder takes the EXIF orientation from the first APP1 segment, whatever
    # it contains.
    pos = 2  # skip SOI, already checked by the format sniffer
    sof = None
    orientation = None
    seen_app1 = False
    while True:
        chunk = src.read(pos, 4)
        if len(chunk) < 2:
            raise RuntimeError("JPEG is truncated: no SOS marker found.")
        if chunk[0] != 0xFF:
            # libjpeg skips garbage bytes between markers.
            window = src.read(pos, 4096)
            idx = window.find(b"\xff")
            pos += idx if idx > 0 else len(window)
            continue
        marker = chunk[1]
        if marker == 0xFF:  # fill byte
            pos += 1
            continue
        if marker == 0xDA:  # SOS
            break
        if marker == 0xD9:  # EOI
            raise RuntimeError("JPEG has no image data.")
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:  # standalone markers
            pos += 2
            continue
        if len(chunk) < 4:
            raise RuntimeError("JPEG is truncated.")
        segment_length = (chunk[2] << 8) | chunk[3]
        if segment_length < 2:
            raise RuntimeError("Invalid JPEG marker segment length.")
        if marker in _JPEG_SUPPORTED_SOF:
            if sof is not None:
                raise RuntimeError("JPEG has more than one SOF marker.")
            precision, height, width, num_components = struct.unpack(
                ">BHHB", src.read_exact(pos + 4, 6)
            )
            if segment_length != 8 + 3 * num_components:
                raise RuntimeError("Invalid JPEG SOF marker length.")
            if height == 0 or width == 0 or num_components == 0:
                raise RuntimeError("Empty JPEG image (DNL not supported).")
            sof = (precision, height, width, num_components)
        elif marker in _JPEG_UNSUPPORTED_SOF:
            raise RuntimeError(f"Unsupported JPEG process: SOF type 0x{marker:02x}.")
        elif marker == 0xE1 and not seen_app1:
            seen_app1 = True
            payload = src.read_exact(pos + 4, segment_length - 2)
            if len(payload) > 6:
                orientation = _exif_orientation(payload[6:])
        pos += 2 + segment_length

    if sof is None:
        raise RuntimeError("JPEG has no SOF marker before SOS.")
    precision, height, width, num_components = sof
    if _swaps_width_and_height(orientation):
        width, height = height, width
    return width, height, num_components, precision, 1, orientation


# ----------------------------------------------------------------------------
# PNG
# ----------------------------------------------------------------------------

_PNG_VALID_BIT_DEPTHS = {
    0: (1, 2, 4, 8, 16),  # gray
    2: (8, 16),  # RGB
    3: (1, 2, 4, 8),  # palette
    4: (8, 16),  # gray + alpha
    6: (8, 16),  # RGBA
}


def _probe_png(src: _ByteSource) -> tuple:
    # Reads IHDR and walks the chunks up to the first IDAT, which is what
    # png_read_info() sees. In particular, an eXIf chunk after IDAT is ignored
    # by the decoder.
    ihdr = src.read_exact(8, 8 + 13)
    length, chunk_type = struct.unpack_from(">I4s", ihdr)
    if chunk_type != b"IHDR" or length != 13:
        raise RuntimeError("PNG is missing its IHDR chunk.")
    width, height, bit_depth, color_type = struct.unpack_from(">IIBB", ihdr, 8)
    if not (0 < width < 2**31 and 0 < height < 2**31):
        raise RuntimeError(f"Invalid PNG dimensions: {width}x{height}.")
    if bit_depth not in _PNG_VALID_BIT_DEPTHS.get(color_type, ()):
        raise RuntimeError(
            f"Invalid PNG color type ({color_type}) and bit depth ({bit_depth})."
        )

    pos = 8 + 12 + 13
    num_palette = None
    has_trns = False
    exif = None
    while True:
        length, chunk_type = struct.unpack(">I4s", src.read_exact(pos, 8))
        if chunk_type == b"IDAT":
            break
        if chunk_type == b"IEND":
            raise RuntimeError("PNG has no IDAT chunk.")
        if chunk_type == b"PLTE":
            num_palette = length // 3
        elif chunk_type == b"tRNS" and not has_trns:
            # libpng ignores (with a warning) tRNS chunks that are invalid for
            # the color type.
            if color_type == 3:
                has_trns = num_palette is not None and 0 < length <= num_palette
            elif color_type == 0:
                has_trns = length == 2
            elif color_type == 2:
                has_trns = length == 6
        elif chunk_type == b"eXIf" and exif is None and length >= 2:
            payload = src.read_exact(pos + 8, length)
            if payload[0] == payload[1] and payload[0] in b"MI":
                exif = payload
        pos += 12 + length
    if color_type == 3 and num_palette is None:
        raise RuntimeError("PNG palette image has no PLTE chunk before IDAT.")

    orientation = _exif_orientation(exif) if exif is not None else None
    if _swaps_width_and_height(orientation):
        width, height = height, width
    if color_type == 3:
        num_channels = 4 if has_trns else 3
        bit_depth = 8  # palette entries are 8-bit RGB(A)
    else:
        num_channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    return width, height, num_channels, bit_depth, 1, orientation


# ----------------------------------------------------------------------------
# WebP
# ----------------------------------------------------------------------------

_WEBP_ANIMATION_FLAG = 0x02
_WEBP_EXIF_FLAG = 0x08
_WEBP_ALPHA_FLAG = 0x10


def _vp8_dimensions(src: _ByteSource, payload_pos: int) -> tuple[int, int]:
    header = src.read_exact(payload_pos, 10)
    if header[3:6] != b"\x9d\x01\x2a":
        raise RuntimeError("Invalid VP8 bitstream signature.")
    width, height = struct.unpack_from("<HH", header, 6)
    width, height = width & 0x3FFF, height & 0x3FFF
    if width == 0 or height == 0:
        raise RuntimeError("Invalid VP8 dimensions.")
    return width, height


def _vp8l_info(src: _ByteSource, payload_pos: int) -> tuple[int, int, bool]:
    header = src.read_exact(payload_pos, 5)
    if header[0] != 0x2F:
        raise RuntimeError("Invalid VP8L bitstream signature.")
    (bits,) = struct.unpack_from("<I", header, 1)
    if bits >> 29 != 0:
        raise RuntimeError("Invalid VP8L version.")
    width = (bits & 0x3FFF) + 1
    height = ((bits >> 14) & 0x3FFF) + 1
    has_alpha = bool((bits >> 28) & 1)
    return width, height, has_alpha


def _probe_webp(src: _ByteSource) -> tuple:
    # Mirrors WebPGetFeatures() (dimensions and alpha), WebPAnimDecoder (frame
    # count and canvas size) and WebPDemux (the EXIF chunk, which libwebpdemux
    # only exposes when the VP8X EXIF flag is set).
    first_type, first_size = struct.unpack("<4sI", src.read_exact(12, 8))
    orientation = None
    if first_type == b"VP8 ":
        width, height = _vp8_dimensions(src, 20)
        has_alpha = False
        num_frames = 1
    elif first_type == b"VP8L":
        width, height, has_alpha = _vp8l_info(src, 20)
        num_frames = 1
    elif first_type == b"VP8X":
        vp8x = src.read_exact(20, 10)
        flags = vp8x[0]
        canvas_width = int.from_bytes(vp8x[4:7], "little") + 1
        canvas_height = int.from_bytes(vp8x[7:10], "little") + 1
        is_animated = bool(flags & _WEBP_ANIMATION_FLAG)

        num_anmf = 0
        seen_alph = False
        bitstream = None  # (width, height, vp8l_alpha_or_None) of first image
        exif = None
        pos = 20 + first_size + (first_size & 1)
        while True:
            header = src.read(pos, 8)
            if len(header) < 8:
                break
            chunk_type, chunk_size = struct.unpack("<4sI", header)
            if chunk_type == b"ANMF":
                num_anmf += 1
            elif not is_animated and bitstream is None:
                if chunk_type == b"ALPH":
                    seen_alph = True
                elif chunk_type == b"VP8 ":
                    bitstream = (*_vp8_dimensions(src, pos + 8), None)
                elif chunk_type == b"VP8L":
                    bitstream = _vp8l_info(src, pos + 8)
            if chunk_type == b"EXIF" and exif is None and flags & _WEBP_EXIF_FLAG:
                exif = src.read_exact(pos + 8, chunk_size)
            pos += 8 + chunk_size + (chunk_size & 1)

        if exif is not None:
            if exif[:6] == b"Exif\x00\x00":
                exif = exif[6:]
            if len(exif) > 0:
                orientation = _exif_orientation(exif)

        if is_animated:
            if num_anmf == 0:
                raise RuntimeError("Animated WebP has no frames.")
            width, height = canvas_width, canvas_height
            has_alpha = bool(flags & _WEBP_ALPHA_FLAG)
            num_frames = num_anmf
        else:
            if bitstream is None:
                raise RuntimeError("WebP has no VP8 or VP8L bitstream.")
            width, height, vp8l_alpha = bitstream
            if (width, height) != (canvas_width, canvas_height):
                raise RuntimeError("WebP canvas and bitstream dimensions differ.")
            if vp8l_alpha is None:
                has_alpha = bool(flags & _WEBP_ALPHA_FLAG) or seen_alph
            else:
                has_alpha = vp8l_alpha or seen_alph
            num_frames = 1
    else:
        raise RuntimeError(f"Unexpected first WebP chunk: {first_type!r}.")

    if _swaps_width_and_height(orientation):
        width, height = height, width
    return width, height, 4 if has_alpha else 3, 8, num_frames, orientation


# ----------------------------------------------------------------------------
# HEIC (ISOBMFF / HEIF)
# ----------------------------------------------------------------------------


def _iter_boxes(data: bytes, start: int, end: int):
    # Yields (box_type, payload_start, payload_end) for the boxes in
    # data[start:end].
    pos = start
    while pos + 8 <= end:
        size, box_type = struct.unpack_from(">I4s", data, pos)
        header_size = 8
        if size == 1:
            if pos + 16 > end:
                raise RuntimeError("ISOBMFF box is truncated.")
            (size,) = struct.unpack_from(">Q", data, pos + 8)
            header_size = 16
        elif size == 0:
            size = end - pos
        if box_type == b"uuid":
            header_size += 16
        if size < header_size or pos + size > end:
            raise RuntimeError("Invalid or truncated ISOBMFF box.")
        yield box_type, pos + header_size, pos + size
        pos += size


def _iter_top_level_boxes(src: _ByteSource):
    # Like _iter_boxes, over the whole file, reading only box headers and the
    # payloads we ask for. Yields (box_type, payload_start, payload_end).
    pos = 0
    while True:
        header = src.read(pos, 16)
        if len(header) < 8:
            return
        size, box_type = struct.unpack_from(">I4s", header)
        header_size = 8
        if size == 1:
            if len(header) < 16:
                raise RuntimeError("ISOBMFF box is truncated.")
            (size,) = struct.unpack_from(">Q", header, 8)
            header_size = 16
        elif size == 0:  # the box extends to the end of the file
            size = src.size - pos
        if size < header_size:
            raise RuntimeError("Invalid ISOBMFF box size.")
        yield box_type, pos + header_size, pos + size
        pos += size


class _HeifMeta:
    # The parts of a 'meta' box we need: items, their references, and their
    # associated properties (in association order).

    def __init__(self, data: bytes):
        self.primary_id = None
        self.item_types: dict[int, bytes] = {}
        self.hidden: set[int] = set()
        # refs[ref_type][from_id] = [to_ids...], in file order
        self.refs: dict[bytes, dict[int, list[int]]] = {}
        self._refs_to: dict[bytes, dict[int, list[int]]] = {}
        self._properties: list[tuple[bytes, bytes]] = []
        self._associations: dict[int, list[int]] = {}

        for box_type, start, end in _iter_boxes(data, 4, len(data)):
            if box_type == b"pitm":
                version = data[start]
                fmt = ">H" if version == 0 else ">I"
                (self.primary_id,) = struct.unpack_from(fmt, data, start + 4)
            elif box_type == b"iinf":
                version = data[start]
                entries_start = start + 4 + (2 if version == 0 else 4)
                for infe_type, s, e in _iter_boxes(data, entries_start, end):
                    if infe_type == b"infe":
                        self._parse_infe(data, s, e)
            elif box_type == b"iref":
                version = data[start]
                id_fmt, id_size = (">H", 2) if version == 0 else (">I", 4)
                for ref_type, s, e in _iter_boxes(data, start + 4, end):
                    (from_id,) = struct.unpack_from(id_fmt, data, s)
                    (count,) = struct.unpack_from(">H", data, s + id_size)
                    first = s + id_size + 2
                    if first + count * id_size > e:
                        raise RuntimeError("Invalid 'iref' box.")
                    to_ids = [
                        struct.unpack_from(id_fmt, data, first + i * id_size)[0]
                        for i in range(count)
                    ]
                    self.refs.setdefault(ref_type, {}).setdefault(from_id, []).extend(
                        to_ids
                    )
                    for to_id in to_ids:
                        self._refs_to.setdefault(ref_type, {}).setdefault(
                            to_id, []
                        ).append(from_id)
            elif box_type == b"iprp":
                for iprp_type, s, e in _iter_boxes(data, start, end):
                    if iprp_type == b"ipco":
                        self._properties = [
                            (t, data[ps:pe]) for t, ps, pe in _iter_boxes(data, s, e)
                        ]
                    elif iprp_type == b"ipma":
                        self._parse_ipma(data, s, e)

    def _parse_infe(self, data, start, end):
        version = data[start]
        if version < 2:
            return
        flags = int.from_bytes(data[start + 1 : start + 4], "big")
        if version == 2:
            (item_id,) = struct.unpack_from(">H", data, start + 4)
            item_type = data[start + 8 : start + 12]
        else:
            (item_id,) = struct.unpack_from(">I", data, start + 4)
            item_type = data[start + 10 : start + 14]
        self.item_types[item_id] = bytes(item_type)
        if flags & 1:
            self.hidden.add(item_id)

    def _parse_ipma(self, data, start, end):
        version = data[start]
        flags = int.from_bytes(data[start + 1 : start + 4], "big")
        (entry_count,) = struct.unpack_from(">I", data, start + 4)
        pos = start + 8
        for _ in range(entry_count):
            if version < 1:
                (item_id,) = struct.unpack_from(">H", data, pos)
                pos += 2
            else:
                (item_id,) = struct.unpack_from(">I", data, pos)
                pos += 4
            num_associations = data[pos]
            pos += 1
            indices = self._associations.setdefault(item_id, [])
            for _ in range(num_associations):
                if flags & 1:
                    (value,) = struct.unpack_from(">H", data, pos)
                    pos += 2
                    indices.append(value & 0x7FFF)
                else:
                    indices.append(data[pos] & 0x7F)
                    pos += 1

    def properties(self, item_id: int) -> list[tuple[bytes, bytes]]:
        # (property_type, property_payload) pairs, in association order.
        # Property indices are 1-based; 0 means "no property".
        return [
            self._properties[i - 1]
            for i in self._associations.get(item_id, ())
            if 0 < i <= len(self._properties)
        ]

    def find_property(self, item_id: int, property_type: bytes) -> bytes | None:
        for t, payload in self.properties(item_id):
            if t == property_type:
                return payload
        return None

    def refs_from(self, ref_type: bytes, from_id: int) -> list[int]:
        return self.refs.get(ref_type, {}).get(from_id, [])

    def refs_to(self, ref_type: bytes, to_id: int) -> list[int]:
        return self._refs_to.get(ref_type, {}).get(to_id, [])


_HEIC_ALPHA_URNS = (
    b"urn:mpeg:avc:2015:auxid:1",
    b"urn:mpeg:hevc:2015:auxid:1",
    b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha",
)


def _aux_type(auxc_payload: bytes | None) -> bytes | None:
    # auxC is a FullBox holding a null-terminated URN.
    if auxc_payload is None:
        return None
    return auxc_payload[4:].split(b"\x00", 1)[0]


def _ispe_size(meta: _HeifMeta, item_id: int) -> tuple[int, int]:
    ispe = meta.find_property(item_id, b"ispe")
    if ispe is None or len(ispe) < 12:
        raise RuntimeError(f"Image item {item_id} has no valid 'ispe' property.")
    return struct.unpack_from(">II", ispe, 4)


def _read_meta_box(src: _ByteSource) -> bytes:
    # Returns the payload of the top-level 'meta' box, skipping everything else
    # (e.g. 'mdat') without reading it.
    for box_type, start, end in _iter_top_level_boxes(src):
        if box_type == b"meta":
            return src.read_exact(start, end - start)
    raise RuntimeError("HEIC has no 'meta' box.")


def _av1c_depth(av1c: bytes) -> int:
    flags = av1c[2]
    if flags & 0x20:  # twelve_bit
        return 12
    if flags & 0x40:  # high_bitdepth
        return 10
    return 8


def _flip_h(grid):
    return tuple(row[::-1] for row in grid)


def _flip_v(grid):
    return grid[::-1]


def _transpose(grid):
    return tuple(zip(*grid))


def _rotate_ccw(grid):
    return _flip_v(_transpose(grid))


# libheif applies the irot and imir properties in association order, so we
# compose them by applying them to a small grid of distinct values, and look up
# which EXIF orientation produces the same result. The transforms below are the
# ones of exif_orientation_transform() in Exif.h.
_ORIENTATION_REFERENCE_GRID = ((0, 1, 2), (3, 4, 5))
_ORIENTATION_FROM_GRID = {
    transform(_ORIENTATION_REFERENCE_GRID): orientation
    for orientation, transform in {
        1: lambda g: g,
        2: _flip_h,
        3: lambda g: _flip_h(_flip_v(g)),
        4: _flip_v,
        5: _transpose,
        6: lambda g: _flip_h(_transpose(g)),
        7: lambda g: _flip_h(_flip_v(_transpose(g))),
        8: lambda g: _flip_v(_transpose(g)),
    }.items()
}


_HEIC_IMAGE_ITEM_TYPES = frozenset(
    (
        b"jpeg",
        b"hvc1",
        b"av01",
        b"vvc1",
        b"avc1",
        b"unci",
        b"j2k1",
        b"mski",
        b"grid",
        b"iovl",
        b"iden",
        b"tili",
    )
)
_HEIC_DERIVED_ITEM_TYPES = frozenset((b"grid", b"iovl", b"iden"))


def _probe_heic(src: _ByteSource) -> tuple:
    # Mirrors libheif: the top-level images are the non-hidden image items that
    # aren't thumbnails or auxiliary images, in item-ID order. The decoder uses
    # the first one's geometry (and requires every other one to match).
    meta = _HeifMeta(_read_meta_box(src))

    image_ids = sorted(
        i for i, t in meta.item_types.items() if t in _HEIC_IMAGE_ITEM_TYPES
    )
    if meta.primary_id not in image_ids or meta.primary_id in meta.hidden:
        raise RuntimeError(
            "'pitm' box references an unsupported or non-existing image."
        )
    top_level = [
        i
        for i in image_ids
        if i not in meta.hidden
        and not meta.refs_from(b"thmb", i)
        and not meta.refs_from(b"auxl", i)
    ]
    if not top_level:
        raise RuntimeError("HEIC file should contain at least one top-level image.")
    item_id = top_level[0]

    width, height = _ispe_size(meta, item_id)
    grid = _ORIENTATION_REFERENCE_GRID
    has_orientation = False
    for prop_type, payload in meta.properties(item_id):
        if prop_type == b"clap":
            w_num, w_den, h_num, h_den = struct.unpack_from(">iIiI", payload)
            if w_den == 0 or h_den == 0:
                raise RuntimeError("Invalid 'clap' property.")
            width = (w_num + w_den // 2) // w_den
            height = (h_num + h_den // 2) // h_den
        elif prop_type == b"irot":
            has_orientation = True
            angle = payload[0] & 0x03  # counter-clockwise quarter turns
            for _ in range(angle):
                grid = _rotate_ccw(grid)
            if angle in (1, 3):
                width, height = height, width
        elif prop_type == b"imir":
            has_orientation = True
            # Axis 0 exchanges the top and bottom, axis 1 the left and right.
            grid = _flip_v(grid) if payload[0] & 0x01 == 0 else _flip_h(grid)
    orientation = _ORIENTATION_FROM_GRID[grid] if has_orientation else None

    # Luma bit depth, from the first non-derived image down the 'dimg' chain.
    coded_id = item_id
    for _ in range(len(meta.item_types) + 1):
        if meta.item_types.get(coded_id) not in _HEIC_DERIVED_ITEM_TYPES:
            break
        children = meta.refs_from(b"dimg", coded_id)
        if not children:
            raise RuntimeError("Derived HEIC image has no input image.")
        coded_id = children[0]
    coded_type = meta.item_types.get(coded_id)
    if coded_type == b"hvc1":
        hvcc = meta.find_property(coded_id, b"hvcC")
        if hvcc is None or len(hvcc) < 18:
            raise RuntimeError("No hvcC property in hvc1 type image.")
        bit_depth = (hvcc[17] & 0x07) + 8
    elif coded_type == b"av01":
        av1c = meta.find_property(coded_id, b"av1C")
        if av1c is None:
            raise RuntimeError("No av1C property in av01 type image.")
        bit_depth = _av1c_depth(av1c)
    elif coded_type == b"jpeg":
        bit_depth = 8
    else:
        raise RuntimeError(f"Unsupported HEIC image item type: {coded_type!r}.")

    def has_alpha_aux(image_id):
        return any(
            _aux_type(meta.find_property(i, b"auxC")) in _HEIC_ALPHA_URNS
            for i in meta.refs_to(b"auxl", image_id)
        )

    has_alpha = has_alpha_aux(item_id) or (
        meta.item_types[item_id] == b"grid"
        and any(has_alpha_aux(t) for t in meta.refs_from(b"dimg", item_id))
    )
    return width, height, 4 if has_alpha else 3, bit_depth, len(top_level), orientation


_PY_PROBES = {
    "jpeg": _probe_jpeg,
    "png": _probe_png,
    "webp": _probe_webp,
    "heic": _probe_heic,
}

# GIF and AVIF are probed in C++, with giflib and libavif. Both take a path
# directly, so that they can read the file lazily.
_CPP_PROBES = {
    "gif": (ops.probe_gif, ops.probe_gif_from_file),
    "avif": (ops.probe_avif, ops.probe_avif_from_file),
}
_CPP_NO_ORIENTATION = -1

# Rotation of each EXIF orientation, when decomposed into an optional
# left-to-right mirroring followed by a rotation.
_ROTATION_FROM_EXIF_ORIENTATION = {
    1: 0.0,
    2: 0.0,
    3: 180.0,
    4: 180.0,
    5: 90.0,
    6: -90.0,
    7: -90.0,
    8: 90.0,
}


def get_image_metadata(source: str | Path | bytes | torch.Tensor) -> ImageMetadata:
    """Get the metadata of an encoded image, without decoding it.

    The format is detected from the encoded data, like in
    :func:`~torchcodec.decoders.decode_image`. Supported formats are JPEG, PNG,
    WebP, GIF, AVIF and HEIC. Only the image headers are parsed, so this is
    much faster than decoding. When ``source`` is a path, only the parts of the
    file that hold the headers are read (except for GIF, which has no frame
    count in its header).

    See :class:`ImageMetadata` for the returned attributes, which all describe
    the output of the decoders.

    Example:

        .. code-block:: python

            from torchcodec.decoders import get_image_metadata

            metadata = get_image_metadata("image.jpg")
            print(metadata.width, metadata.height)

    Args:
        source (str, ``pathlib.Path``, bytes, or ``torch.Tensor``):
            The encoded image data: a path (``str`` or ``pathlib.Path``), a
            ``bytes`` object, or a 1-D uint8 ``torch.Tensor`` of the raw encoded
            bytes.

    Returns:
        ImageMetadata: The image metadata.
    """
    src = _ByteSource(source)
    try:
        fmt = _detect_image_format_from_header(src.read(0, 64))
        if fmt in _PY_PROBES:
            probe_result = _PY_PROBES[fmt](src)
        else:
            probe, probe_from_file = _CPP_PROBES[fmt]
            if isinstance(source, (str, Path)):
                probe_result = probe_from_file(str(source))
            else:
                probe_result = probe(_source_to_tensor(source))
            if probe_result[5] == _CPP_NO_ORIENTATION:
                probe_result = (*probe_result[:5], None)
    except (struct.error, IndexError):
        raise RuntimeError("Image is truncated or corrupted.") from None
    finally:
        src.close()
    width, height, num_channels, bit_depth, num_frames, orientation = probe_result
    return ImageMetadata(
        format=fmt,
        width=width,
        height=height,
        num_channels=num_channels,
        bit_depth=bit_depth,
        num_frames=num_frames,
        rotation=(
            None
            if orientation is None
            else _ROTATION_FROM_EXIF_ORIENTATION[orientation]
        ),
    )
