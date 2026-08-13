# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
========================================
Migrating from TorchVision to TorchCodec
========================================

The image decoders and encoders of ``torchvision.io`` now live in torchcodec.
This is a short guide to porting your code over. Everything you could do with
``torchvision.io`` you can do with TorchCodec, usually with a very similar call.
And TorchCodec supports many more features. To learn more about the image
decoding and encoding features of TorchCodec, refer to the
:ref:`image decoding <sphx_glr_generated_examples_decoding_image_decoding.py>`
and
:ref:`image encoding <sphx_glr_generated_examples_encoding_image_encoding.py>`
tutorials.

TL;DR
-----

- ``decode_image(x)`` -> ``decode_image(x)``, but watch out for the
  :ref:`changed defaults <decoding_defaults>`
- ``decode_jpeg(x, device="cuda")`` -> ``decode_jpeg(x, device="cuda")``, same
  caveat
- ``read_file(path)`` -> not needed, pass ``path`` to the decoder
- ``encode_jpeg(img, quality)`` -> ``JpegEncoder(img).to_tensor(quality=...)``
- ``write_jpeg(img, path, quality)`` -> ``JpegEncoder(img).to_file(path, quality=...)``
- ``encode_png(img, level)`` -> ``PngEncoder(img).to_tensor(compression_level=...)``
- ``write_png(img, path, level)`` -> ``PngEncoder(img).to_file(path, compression_level=...)``
- ``write_file(path, encoded)`` -> not needed, use ``to_file``

The rest of this guide goes over these one by one.
"""

# %%
# A bit of boilerplate first: let's make up some encoded image bytes to play
# with, by encoding a random image.
import torch

from torchcodec.encoders import JpegEncoder, PngEncoder

raw_image_bytes = JpegEncoder(
    torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
).to_tensor()

# %%
# Decoding
# --------
#
# ``torchvision.io.decode_image`` becomes
# :func:`torchcodec.decoders.decode_image`. Both accept raw encoded bytes, a
# tensor of encoded bytes, or a path to a file:
#
# .. code-block:: python
#
#     # Before
#     from torchvision.io import decode_image
#     image = decode_image("image.jpg")
#
#     # After
#     from torchcodec.decoders import decode_image
#     image = decode_image("image.jpg")
#
# The format-specific decoders map over one-to-one as well:
# ``decode_jpeg``, ``decode_png``, ``decode_webp``, ``decode_gif``, and
# torchcodec adds ``decode_avif`` and ``decode_heic`` without needing the
# separate ``torchvision-extra-decoders`` package.

from torchcodec.decoders import decode_image

image = decode_image(raw_image_bytes)
print(f"{image.shape = }, {image.dtype = }")

# %%
# ``torchvision.io.read_file`` has no equivalent, and you don't need one: pass
# the path (a ``str`` or a ``pathlib.Path``) straight to the decoder.
#
# .. code-block:: python
#
#     # Before
#     from torchvision.io import decode_image, read_file
#     image = decode_image(read_file("image.jpg"))
#
#     # After
#     from torchcodec.decoders import decode_image
#     image = decode_image("image.jpg")

# %%
# .. _decoding_defaults:
#
# A few decoding defaults changed
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# - ``mode`` now defaults to ``"RGB"`` instead of ``"UNCHANGED"``. If you were
#   relying on the source's own channel layout, pass ``mode="UNCHANGED"``.
# - The output is always ``torch.uint8`` by default, even for 16-bit sources.
#   To get torchvision's behaviour, where the dtype follows the source, pass
#   ``output_dtype="auto"``.
# - The ``apply_exif_orientation`` parameter is gone: EXIF orientation is
#   always applied.

print(f"{decode_image(raw_image_bytes, mode='GRAY').shape = }")
print(f"{decode_image(raw_image_bytes, output_dtype=torch.uint16).dtype = }")

# %%
# Encoding
# --------
#
# The encoding functions became classes: instantiate an encoder with the image,
# then choose where the encoded bytes should go.
#
# .. code-block:: python
#
#     # Before
#     from torchvision.io import encode_jpeg, write_jpeg
#     encoded = encode_jpeg(image, quality=80)  # to a tensor
#     write_jpeg(image, "image.jpg", quality=80)  # to a file
#
#     # After
#     from torchcodec.encoders import JpegEncoder
#     encoded = JpegEncoder(image).to_tensor(quality=80)  # to a tensor
#     JpegEncoder(image).to_file("image.jpg", quality=80)  # to a file
#
# PNG works the same way with :class:`~torchcodec.encoders.PngEncoder` and
# ``compression_level``:

print(f"{JpegEncoder(image).to_tensor(quality=80).shape = }")
print(f"{PngEncoder(image).to_tensor(compression_level=6).shape = }")

# %%
# There is no batch equivalent to ``encode_jpeg(list_of_images)``: an encoder
# takes a single image, so encode a batch with a plain Python loop. You're not
# losing any speed:
#
# .. code-block:: python
#
#     encoded = [JpegEncoder(image).to_tensor() for image in images]

# %%
# Encoders also support a third destination that torchvision didn't have: a
# file-like object, i.e. anything with ``write`` and ``seek``.
import io

buffer = io.BytesIO()
JpegEncoder(image).to_file_like(buffer)
print(f"{len(buffer.getvalue()) = }")
