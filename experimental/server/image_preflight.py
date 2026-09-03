# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bounds an OpenAI ``image_url`` ``data:`` payload before it is decoded.

A ``data:`` image is attacker-controlled: the server binds without auth, and
the bytes go straight to ``stbi_load_from_memory``, which allocates
``width * height * channels`` host bytes before any engine-side limit is
consulted. A few-KB PNG can declare a 60000x60000 canvas. This module reads the
dimensions out of the container header without decoding pixels, so an oversized
image costs a header parse instead of a multi-GB allocation on a device whose
memory the engine already occupies.

Formats are limited to the ones the C++ loader actually decodes *and* this
module can measure. An unmeasurable format is refused rather than passed
through, because passing it through is exactly the bypass the caps exist to
close.
"""
from __future__ import annotations

import struct
from typing import Tuple

#: Ceiling on one decoded ``data:`` image payload. Well above the ~20 MB an
#: OpenAI vision client will send, well below the whole-body cap in
#: ``api_server``, which sizes for video rather than stills.
MAX_IMAGE_SOURCE_BYTES = 32 * 1024 * 1024

#: Ceiling on decoded pixels for one image. stb_image allocates the RGB buffer
#: and ``imageUtils`` copies it into a Tensor, so the true peak is ~2x the RGB
#: bytes: 64M px is ~384 MiB peak, which an Orin can absorb alongside a loaded
#: engine. The video path's own budget (``MAX_DECODE_PIXELS``) is separate and
#: applies per request rather than per image.
MAX_IMAGE_PIXELS = 64 * 1024 * 1024

#: Per-side ceiling, checked before the pixel product so a degenerate strip is
#: rejected on its own terms rather than sneaking under the area budget.
MAX_IMAGE_DIMENSION = 16384

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_GIF_SIGNATURES = (b"GIF87a", b"GIF89a")

#: JPEG frame headers (SOF0-SOF15) carry the dimensions. 0xC4/0xC8/0xCC fall in
#: the same numeric range but are DHT/JPG/DAC, not frame headers.
_JPEG_SOF_MARKERS = frozenset(m for m in range(0xC0, 0xD0)
                              if m not in (0xC4, 0xC8, 0xCC))

#: Markers that carry no payload length, so the walk steps over them by two.
_JPEG_STANDALONE_MARKERS = frozenset({0x01, 0xD8} | set(range(0xD0, 0xD8)))

_SUPPORTED = "PNG, JPEG, GIF and BMP"


def _truncated(fmt: str) -> ValueError:
    return ValueError(f"truncated {fmt} image header")


def _png_dimensions(data: bytes) -> Tuple[int, int]:
    # 8-byte signature, 4-byte chunk length, "IHDR", then two big-endian u32.
    if len(data) < 24:
        raise _truncated("PNG")
    if struct.unpack(">I", data[8:12])[0] != 13:
        raise ValueError("malformed PNG IHDR length")
    if data[12:16] != b"IHDR":
        raise ValueError("PNG image does not start with an IHDR chunk")
    return struct.unpack(">II", data[16:24])


def _gif_dimensions(data: bytes) -> Tuple[int, int]:
    if len(data) < 10:
        raise _truncated("GIF")
    return struct.unpack("<HH", data[6:10])


def _bmp_dimensions(data: bytes) -> Tuple[int, int]:
    if len(data) < 26:
        raise _truncated("BMP")
    header_size = struct.unpack("<I", data[14:18])[0]
    if header_size == 12:
        # BITMAPCOREHEADER stores both dimensions as unsigned 16-bit values.
        width, height = struct.unpack("<HH", data[18:22])
    else:
        width, height = struct.unpack("<ii", data[18:26])
    # A negative height marks a top-down bitmap. Negative width is invalid and
    # is intentionally returned unchanged for the caller's empty-canvas check.
    return width, abs(height)


def _jpeg_dimensions(data: bytes) -> Tuple[int, int]:
    """Walk the marker segments to the frame header.

    The walk is bounded by the payload length each segment declares, so it is
    linear in the input and cannot be steered into a loop by a crafted file.
    """
    offset = 2
    size = len(data)
    while offset + 4 <= size:
        if data[offset] != 0xFF:
            raise ValueError("malformed JPEG segment header")
        marker = data[offset + 1]
        if marker == 0xFF:  # fill byte before the real marker
            offset += 1
            continue
        if marker in _JPEG_STANDALONE_MARKERS:
            offset += 2
            continue
        if marker == 0xDA:  # start of scan: entropy data, no frame header left
            break
        length = struct.unpack(">H", data[offset + 2:offset + 4])[0]
        if length < 2:
            raise ValueError("malformed JPEG segment length")
        if marker in _JPEG_SOF_MARKERS:
            if length < 8:
                raise ValueError("malformed JPEG frame header length")
            if offset + 9 > size:
                raise _truncated("JPEG")
            height, width = struct.unpack(">HH", data[offset + 5:offset + 9])
            return width, height
        offset += 2 + length
    raise ValueError("JPEG image has no frame header")


def probe_image_dimensions(data: bytes) -> Tuple[int, int]:
    """``(width, height)`` read from an encoded image header, no pixel decode.

    Raises ``ValueError`` for a format this module cannot measure.
    """
    if data.startswith(_PNG_SIGNATURE):
        return _png_dimensions(data)
    if data.startswith(b"\xff\xd8"):
        return _jpeg_dimensions(data)
    if data.startswith(_GIF_SIGNATURES):
        return _gif_dimensions(data)
    if data.startswith(b"BM"):
        return _bmp_dimensions(data)
    raise ValueError(f"unsupported image format; the runtime decodes "
                     f"{_SUPPORTED}")


def preflight_image_data_url(url: str) -> Tuple[bytes, Tuple[int, int]]:
    """Decode and bound one ``data:image/...;base64,`` URL.

    Returns the encoded image bytes and the ``(width, height)`` its header
    declares, so the caller can charge the request's visual-token budget for an
    image that has no file to probe.

    Every rejection is a ``ValueError`` carrying only measured integers and the
    limit that was exceeded. The message reaches the client verbatim in a 400
    body, so no part of the payload -- not the media type, not a prefix of the
    base64 -- may appear in it.
    """
    from .media_source import decode_base64_data_url

    data = decode_base64_data_url(url,
                                  "image",
                                  strict=True,
                                  max_bytes=MAX_IMAGE_SOURCE_BYTES)
    width, height = probe_image_dimensions(data)
    if width <= 0 or height <= 0:
        raise ValueError("image header declares an empty canvas")
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ValueError(f"image is {width}x{height}, over the supported "
                         f"maximum of {MAX_IMAGE_DIMENSION} pixels per side")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"image is {width}x{height}, over the supported "
                         f"maximum of {MAX_IMAGE_PIXELS} decoded pixels")
    return data, (width, height)
