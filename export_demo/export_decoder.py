# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export a video decoding program, then AOTInductor-compile it into a .pt2.

The resulting .pt2 is loadable and runnable from a plain C++ process with no
Python runtime, see main.cpp.

    python export_demo/export_decoder.py test/resources/nasa_13013.mp4 /tmp/decoder.pt2
"""

import json
import sys
from pathlib import Path

import torch
from torchcodec._core import (
    create_video_decoder_from_tensor,
    get_frames_at_indices,
    get_json_metadata,
)

# get_frames_at_indices returns frames whose height and width are only known at
# runtime, i.e. its output has a data-dependent shape.
torch._dynamo.config.capture_dynamic_output_shape_ops = True


class VideoDecoding(torch.nn.Module):
    """encoded video bytes + frame indices -> decoded frames."""

    def forward(
        self, video_data: torch.Tensor, frame_indices: torch.Tensor
    ) -> torch.Tensor:
        decoder = create_video_decoder_from_tensor(video_data, num_threads=1)
        frames, pts_seconds, duration_seconds = get_frames_at_indices(
            decoder, frame_indices=frame_indices
        )
        return frames


def main(video_path: str, output_path: str) -> None:
    video_data = torch.frombuffer(Path(video_path).read_bytes(), dtype=torch.uint8)
    frame_indices = torch.tensor([0, 10, 20], dtype=torch.int64)

    # Metadata is a JSON string, which can't be an output of an exported program:
    # the tracer sees the fake impl's return value ("") and would bake that
    # constant into the graph. It has to be queried out of band - which is
    # exactly what the C++ side does, by calling the op through the dispatcher.
    decoder = create_video_decoder_from_tensor(video_data)
    print("metadata:", json.dumps(json.loads(get_json_metadata(decoder)), indent=2))

    dynamic_shapes = {
        "video_data": {0: torch.export.Dim.DYNAMIC},
        "frame_indices": {0: torch.export.Dim.DYNAMIC},
    }
    exported = torch.export.export(
        VideoDecoding(),
        (video_data, frame_indices),
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    print(exported.graph)

    eager = VideoDecoding()(video_data, frame_indices)
    from_export = exported.module()(video_data, frame_indices)
    torch.testing.assert_close(eager, from_export, atol=0, rtol=0)
    print(f"exported program runs: {tuple(from_export.shape)} {from_export.dtype}")

    torch._inductor.aoti_compile_and_package(exported, package_path=output_path)
    runner = torch._inductor.aoti_load_package(output_path)
    from_aoti = runner(video_data, frame_indices)
    torch.testing.assert_close(eager, from_aoti, atol=0, rtol=0)
    print(f"AOTInductor package runs: {tuple(from_aoti.shape)} -> {output_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
