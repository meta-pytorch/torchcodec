import paddle
paddle.enable_compat(scope={"torchcodec"})

import pytest
from dataclasses import dataclass, fields
from io import BytesIO
from typing import Callable, Mapping, Optional, Union

import os
import httpx
import numpy as np


@dataclass
class VideoMetadata(Mapping):
    total_num_frames: int
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None
    video_backend: Optional[str] = None
    frames_indices: Optional[list[int]] = None

    def __iter__(self):
        return (f.name for f in fields(self))

    def __len__(self):
        return len(fields(self))

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    @property
    def timestamps(self) -> list[float]:
        "Timestamps of the sampled frames in seconds."
        if self.fps is None or self.frames_indices is None:
            raise ValueError("Cannot infer video `timestamps` when `fps` or `frames_indices` is None.")
        return [frame_idx / self.fps for frame_idx in self.frames_indices]

    def update(self, dictionary):
        for key, value in dictionary.items():
            if hasattr(self, key):
                setattr(self, key, value)


def default_sample_indices_fn(metadata: VideoMetadata, num_frames=None, fps=None, **kwargs):
    total_num_frames = metadata.total_num_frames
    video_fps = metadata.fps

    if num_frames is None and fps is not None:
        num_frames = int(total_num_frames / video_fps * fps)
        if num_frames > total_num_frames:
            raise ValueError(
                f"When loading the video with fps={fps}, we computed num_frames={num_frames} "
                f"which exceeds total_num_frames={total_num_frames}. Check fps or video metadata."
            )

    if num_frames is not None:
        indices = np.arange(0, total_num_frames, total_num_frames / num_frames, dtype=int)
    else:
        indices = np.arange(0, total_num_frames, dtype=int)
    return indices


def read_video_decord(
    video_path: Union["URL", "Path"],
    sample_indices_fn: Callable,
    **kwargs,
):
    from decord import VideoReader, cpu

    vr = VideoReader(uri=video_path, ctx=cpu(0))  # decord has problems with gpu
    video_fps = vr.get_avg_fps()
    total_num_frames = len(vr)
    duration = total_num_frames / video_fps if video_fps else 0
    metadata = VideoMetadata(
        total_num_frames=int(total_num_frames),
        fps=float(video_fps),
        duration=float(duration),
        video_backend="decord",
    )

    indices = sample_indices_fn(metadata=metadata, **kwargs)
    video = vr.get_batch(indices).asnumpy()

    metadata.update(
        {
            "frames_indices": indices,
            "height": video.shape[1],
            "width": video.shape[2],
        }
    )
    return video, metadata

def read_video_torchcodec(
    video_path: Union["URL", "Path"],
    sample_indices_fn: Callable,
    **kwargs,
):
    from torchcodec.decoders import VideoDecoder  # import torchcodec

    decoder = VideoDecoder(
        video_path,
        seek_mode="exact",
        num_ffmpeg_threads=0,
    )
    metadata = VideoMetadata(
        total_num_frames=decoder.metadata.num_frames,
        fps=decoder.metadata.average_fps,
        duration=decoder.metadata.duration_seconds,
        video_backend="torchcodec",
        height=decoder.metadata.height,
        width=decoder.metadata.width,
    )
    indices = sample_indices_fn(metadata=metadata, **kwargs)

    video = decoder.get_frames_at(indices=indices).data
    video = video.contiguous()
    metadata.frames_indices = indices
    return video, metadata


VIDEO_DECODERS = {
    "decord": read_video_decord,
    "torchcodec": read_video_torchcodec,
}


def load_video(
    video,
    num_frames: Optional[int] = None,
    fps: Optional[Union[int, float]] = None,
    backend: str = "decord",
    sample_indices_fn: Optional[Callable] = None,
    **kwargs,
) -> np.ndarray:

    if fps is not None and num_frames is not None and sample_indices_fn is None:
        raise ValueError(
            "`num_frames`, `fps`, and `sample_indices_fn` are mutually exclusive arguments, please use only one!"
        )

    # If user didn't pass a sampling function, create one on the fly with default logic
    if sample_indices_fn is None:

        def sample_indices_fn_func(metadata, **fn_kwargs):
            return default_sample_indices_fn(metadata, num_frames=num_frames, fps=fps, **fn_kwargs)

        sample_indices_fn = sample_indices_fn_func

    # Early exit if provided an array or `PIL` frames
    if not isinstance(video, str):
        metadata = [None] * len(video)
        return video, metadata

    if video.startswith("http://") or video.startswith("https://"):
        file_obj = BytesIO(httpx.get(video, follow_redirects=True).content)
    elif os.path.isfile(video):
        file_obj = video
    else:
        raise TypeError("Incorrect format used for video. Should be an url linking to an video or a local path.")

    video_decoder = VIDEO_DECODERS[backend]
    video, metadata = video_decoder(file_obj, sample_indices_fn, **kwargs)
    return video, metadata
def test_video_decode():
    url = "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_video/example_video.mp4"
    video, metadata = load_video(url, backend="torchcodec")
    assert video.to(paddle.int64).sum().item() == 247759890390
    assert metadata.total_num_frames == 263
    assert metadata.fps == pytest.approx(29.99418249715141)
    assert metadata.width == 1920
    assert metadata.height == 1080
    assert metadata.duration == pytest.approx(8.768367)
    for i, idx in enumerate(metadata.frames_indices):
        assert idx == i
