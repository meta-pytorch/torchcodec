"""
==================================================
Basic Example to use TorchCodec to decode a video.
==================================================
"""

# A simple example showing how to decode the first few frames of a video.
# using the :class:`~torchcodec.decoders.SimpleVideoDecoder` class.
# %%
import inspect
import os

from torchcodec.decoders import SimpleVideoDecoder

my_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
video_file_path = os.path.dirname(my_path) + "/../test/resources/nasa_13013.mp4"
simple_decoder = SimpleVideoDecoder(video_file_path)
# Since `simple_decoder` is an iterable, you can get the total frame count for
# the best video stream by calling len().
num_frames = len(simple_decoder)
print(f"{video_file_path=} has {num_frames} frames")

# You can get the decoded frame by using the subscript operator.
first_frame = simple_decoder[0]
print(f"decoded frame has type {type(first_frame)}")
# The shape of the decoded frame is (H, W, C) where H and W are the height
# and width of the video frame, respectively. C is 3 because we have 3 channels
# red, green, and blue.
print(f"{first_frame.shape=}")
# The dtype of the decoded frame is uint8.
print(f"{first_frame.dtype=}")

last_frame = simple_decoder[num_frames - 1]
print(f"{last_frame.shape=}")

# TODO: add documentation for slices and metadata.

# %%
