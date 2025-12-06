# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
===================================================================
Decoder Transforms: Needs a tagline
===================================================================

In this example, we will describe the ``transforms`` parameter of the
:class:`~torchcodec.decoders.VideoDecoder` class.
"""

# %%
# First, a bit of boilerplate and definitions.


import torch
import requests
import tempfile
from pathlib import Path
import shutil
import subprocess
from time import perf_counter_ns
from IPython.display import Video

def store_video_to(url: str, local_video_path: Path):
    response = requests.get(url, headers={"User-Agent": ""})
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download video. {response.status_code = }.")

    with open(local_video_path, 'wb') as f:
        for chunk in response.iter_content():
            f.write(chunk)

def plot(frames: torch.Tensor, title : str | None = None):
    try:
        from torchvision.utils import make_grid
        from torchvision.transforms.v2.functional import to_pil_image
        import matplotlib.pyplot as plt
    except ImportError:
        print("Cannot plot, please run `pip install torchvision matplotlib`")
        return

    plt.rcParams["savefig.bbox"] = "tight"
    dpi = 300
    fig, ax = plt.subplots(figsize=(800/dpi, 600/dpi), dpi=dpi)
    ax.imshow(to_pil_image(make_grid(frames)))
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if title is not None:
        ax.set_title(title, fontsize=6)
    plt.tight_layout()

# %%
# Our example video
# -----------------
#
# We'll download a video from the internet and store it locally. We're
# purposefully retrieving a high resolution video to demonstrate using
# transforms to modify dimensions.

# Video source: https://www.pexels.com/video/an-african-penguin-at-the-beach-9140346/
# Author: Taryn Elliott.
url = "https://videos.pexels.com/video-files/9140346/9140346-uhd_3840_2160_25fps.mp4"

temp_dir = tempfile.mkdtemp()
penguin_video_path = Path(temp_dir) / "penguin.mp4"
store_video_to(url, penguin_video_path)

from torchcodec.decoders import VideoDecoder
print(f"Penguin video metadata: {VideoDecoder(penguin_video_path).metadata}")

# %%
# Some stuff about the video itself, including its resolution of 3840x2160.

# %%
# Applying transforms during pre-processing
# -----------------------------------------
#
# There are lots of reasons to apply transforms to video frames during pre-proc
# (list them). A typical example might look like:

from torchvision.transforms import v2

full_decoder = VideoDecoder(penguin_video_path)
full_mid_frame = full_decoder[465] # mid-point of the video
resized_post_mid_frame = v2.Resize(size=(360, 640))(full_mid_frame)

plot(resized_post_mid_frame, title="Resized to 360x640 after decoding")

# %%
# But we can now do it:
resize_decoder = VideoDecoder(
    penguin_video_path,
    transforms=[v2.Resize(size=(360, 640))]
)
resized_during_mid_frame = resize_decoder[465]

plot(resized_during_mid_frame, title="Resized to 360x640 during decoding")

# %%
# TorchCodec's relationship with TorchVision transforms
# -----------------------------------------------------
# Talk about the relationship between TorchVision transforms and TorchCodec
# decoder transforms. Importantly, they're not identical:
abs_diff = (resized_post_mid_frame.float() - resized_during_mid_frame.float()).abs()
(abs_diff == 0).all()

# %%
# But they're close enough that models won't be able to tell a difference:
(abs_diff <= 1).float().mean() >= 0.998

# %%
# Transform pipelines
# -------------------
# But wait - there's more!

crop_resize_decoder = VideoDecoder(
    penguin_video_path,
    transforms = [
        v2.Resize(size=(360, 640)),
        v2.CenterCrop(size=(300, 200))
    ]
)
crop_resized_during_mid_frame = crop_resize_decoder[465]
plot(crop_resized_during_mid_frame, title="Resized to 360x640 during decoding then center cropped")

# %%
# We also support `RandomCrop`. Reach out if there are particular transforms you want!

# %%
# Performance
# -----------
#
# The main motivation for doing this is performance.

# %%
shutil.rmtree(temp_dir)
# %%
