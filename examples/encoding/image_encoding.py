# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
===============
Encoding images
===============

In this example, we'll learn how to encode an image tensor to JPEG or PNG using
the :class:`~torchcodec.encoders.JpegEncoder` and
:class:`~torchcodec.encoders.PngEncoder` classes.
"""

# %%
# First, a bit of boilerplate: we'll download an image from the web and define a
# plotting utility. You can ignore that part and jump right below to
# :ref:`encoding_image`.

import requests
import torch

from torchcodec.decoders import decode_image

url = "https://raw.githubusercontent.com/meta-pytorch/torchcodec/refs/heads/main/docs/source/_static/thumbnails/cat_encoding.jpeg"
response = requests.get(url, headers={"User-Agent": ""})
if response.status_code != 200:
    raise RuntimeError(f"Failed to download image. {response.status_code = }.")

# The image to encode, a CHW uint8 tensor. It could come from anywhere (e.g. a
# model output); here we just decode one.
image = decode_image(response.content)


def plot(image: torch.Tensor):
    try:
        import matplotlib.pyplot as plt
        from torchvision.transforms.v2.functional import to_pil_image
    except ImportError:
        print("Cannot plot, please run `pip install torchvision matplotlib`")
        return

    pil_image = to_pil_image(image)
    fig = plt.figure(figsize=(pil_image.width / 100, pil_image.height / 100))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(pil_image)
    ax.axis("off")


# %%
# .. _encoding_image:
#
# Encoding an image
# -----------------
#
# Encoders expect a 3D uint8 tensor in CHW layout (1 or 3 channels), which is
# exactly what our image is:
print(f"{image.shape = }, {image.dtype = }")
plot(image)

# %%
# We instantiate a :class:`~torchcodec.encoders.JpegEncoder` with the image, and
# encode it. Three destinations are supported: a file with
# :meth:`~torchcodec.encoders.JpegEncoder.to_file`, a file-like object with
# :meth:`~torchcodec.encoders.JpegEncoder.to_file_like`, or a 1D uint8 tensor of
# raw bytes with :meth:`~torchcodec.encoders.JpegEncoder.to_tensor`.
import io

from torchcodec.encoders import JpegEncoder

encoder = JpegEncoder(image)

encoder.to_file("image.jpg")  # to a file
encoder.to_file_like(io.BytesIO())  # to a file-like object
encoded = encoder.to_tensor()  # to a tensor

print(f"{encoded.shape = }, {encoded.dtype = }")

# %%
# That's it! We can decode the encoded bytes back to make sure everything worked:
from torchcodec.decoders import decode_jpeg

decoded = decode_jpeg(encoded)
print(f"{decoded.shape = }")
plot(decoded)

# %%
# :class:`~torchcodec.encoders.PngEncoder` works exactly the same way, and PNG is
# lossless (unlike JPEG):
from torchcodec.encoders import PngEncoder

encoded = PngEncoder(image).to_tensor()
print(f"{encoded.shape = }")

# %%
# Both encoders support encoding options: ``JpegEncoder`` takes a ``quality``
# (1-100), and ``PngEncoder`` takes a ``compression_level`` (0-9). For example, a
# lower JPEG quality yields a smaller output:
small = JpegEncoder(image).to_tensor(quality=10)
large = JpegEncoder(image).to_tensor(quality=95)
print(f"{small.numel() = }, {large.numel() = }")

# %%
# Encoding JPEGs on GPU
# ---------------------
#
# ``JpegEncoder`` can encode directly on a CUDA device with nvJPEG: just pass it
# an image that already lives on the GPU, and the encoding happens there. Only
# 3-channel RGB images are supported on CUDA. With :meth:`to_tensor
# <torchcodec.encoders.JpegEncoder.to_tensor>`, the encoded bytes stay on the GPU
# (call ``.cpu()`` to bring them back to the host).
#
# .. code-block:: python
#
#     from torchcodec.encoders import JpegEncoder
#
#     encoded = JpegEncoder(image.cuda()).to_tensor()  # encoded bytes on the GPU
#     # you can still use to_file and to_file_like, but the encoded bytes will
#     # be copied back to the CPU first.
#
# PNG encoding is CPU-only.

# %%
# Check the docstrings of the encoding methods to learn about the different
# encoding options.

# sphinx_gallery_thumbnail_path = '_static/thumbnails/cat_encoding.jpeg'
