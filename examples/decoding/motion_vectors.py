# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
================================
Extracting motion vectors (CPU)
================================

This example shows how to export compressed-domain motion vectors using
``VideoDecoder``. Motion vectors are returned in a padded tensor along with
per-frame metadata and counts of valid vectors.
"""

# %%
# Download a sample video (same source as the basic decoding example).
import tempfile

import requests
import torch

from torchcodec.decoders import VideoDecoder
from torchcodec.encoders import VideoEncoder

url = "https://videos.pexels.com/video-files/854132/854132-sd_640_360_25fps.mp4"
response = requests.get(url, headers={"User-Agent": ""})
if response.status_code != 200:
    raise RuntimeError(f"Failed to download video. {response.status_code = }.")

raw_video_bytes = response.content

# %%
# Create a decoder with motion vector export enabled (CPU only).
decoder = VideoDecoder(raw_video_bytes, device="cpu", export_mvs=True)
mvs = decoder.get_motion_vectors_at([0, 1, 2])

print(mvs)
print(f"{mvs.data.shape = }")
print(f"{mvs.counts = }")
print(f"{mvs.frame_types = }")

# %%
# Motion vector fields in each 10-element row.
MV_FIELDS = [
    "source",
    "w",
    "h",
    "src_x",
    "src_y",
    "dst_x",
    "dst_y",
    "motion_x",
    "motion_y",
    "motion_scale",
]

# %%
# Use counts to slice valid vectors per frame.
frame_index = 0
count = int(mvs.counts[frame_index])
valid = mvs.data[frame_index, :count]
print(f"{count = }")
print(f"{valid.shape = }")

if count > 0:
    first_mv = valid[0].tolist()
    print(dict(zip(MV_FIELDS, first_mv)))

# %%
# Frame types are ASCII codes (e.g., 'I', 'P', 'B').
frame_type_chars = [chr(int(x)) for x in mvs.frame_types]
print(f"{frame_type_chars = }")

# %%
# Optional: visualize motion vectors over a frame.
# Note: this uses integer rounding for coordinates. For sub-pixel precision,
# scale coordinates or use a rendering backend that supports fixed-point shifts.
try:
    import matplotlib.pyplot as plt
    from torchvision.transforms.v2.functional import to_pil_image
except ImportError:
    print("Cannot plot, please run `pip install torchvision matplotlib`")
else:
    plot_index = int(torch.argmax(mvs.counts).item())
    if int(mvs.counts[plot_index]) == 0:
        print("No motion vectors available to plot.")
    else:
        frame = decoder.get_frame_at(plot_index).data
        fig, ax = plt.subplots()
        ax.imshow(to_pil_image(frame))

        valid = mvs.data[plot_index, : int(mvs.counts[plot_index])]
        for mv in valid:
            dst_x, dst_y = int(mv[5]), int(mv[6])
            motion_scale = int(mv[9])
            if motion_scale == 0:
                continue
            src_x = int(dst_x + mv[7].item() / motion_scale)
            src_y = int(dst_y + mv[8].item() / motion_scale)
            ax.arrow(
                src_x,
                src_y,
                dst_x - src_x,
                dst_y - src_y,
                color="red",
                width=0.5,
                head_width=2.0,
                length_includes_head=True,
            )
            ax.scatter([dst_x], [dst_y], s=5, c="blue")

        ax.set(xticks=[], yticks=[], title=f"Motion vectors (frame {plot_index})")
        plt.tight_layout()

# %%
# Optional: encode a short video with MV overlays using VideoEncoder.
# This overlay is a simple visualization (integer coordinates, no arrowheads).
def _draw_line(image: torch.Tensor, x0: int, y0: int, x1: int, y1: int):
    h, w = image.shape[1], image.shape[2]
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w - 1, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    color = torch.tensor([255, 0, 0], dtype=image.dtype)
    while True:
        image[:, y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


num_overlay_frames = 10
overlay_frames = decoder.get_frames_in_range(0, num_overlay_frames).data.clone()
overlay_mvs = decoder.get_motion_vectors_at(list(range(num_overlay_frames)))

max_draw_per_frame = None
for i in range(num_overlay_frames):
    count = int(overlay_mvs.counts[i])
    if count == 0:
        continue
    if max_draw_per_frame is None or count <= max_draw_per_frame:
        sample_indices = torch.arange(count)
    else:
        sample_indices = torch.linspace(
            0, count - 1, steps=max_draw_per_frame
        ).round().to(torch.int64)
    valid = overlay_mvs.data[i, sample_indices]
    for mv in valid:
        dst_x, dst_y = int(mv[5]), int(mv[6])
        motion_scale = int(mv[9])
        if motion_scale == 0:
            continue
        src_x = float(dst_x) + float(mv[7].item()) / motion_scale
        src_y = float(dst_y) + float(mv[8].item()) / motion_scale
        src_x = int(round(src_x))
        src_y = int(round(src_y))
        _draw_line(overlay_frames[i], src_x, src_y, dst_x, dst_y)

encoder = VideoEncoder(frames=overlay_frames, frame_rate=decoder.metadata.average_fps)
overlay_path = tempfile.NamedTemporaryFile(
    suffix=".mp4", prefix="motion_vectors_overlay_", delete=False
).name
encoder.to_file(overlay_path)
print(f"Wrote {overlay_path}")
