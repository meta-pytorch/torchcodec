# torch.export → C++ video decoding, without a Python runtime

Working prototype for: **raw video bytes in → decoded frames out**, exported with
`torch.export`, compiled with AOTInductor, and run from a plain C++ program.
Metadata is read from the same C++ program by calling the op through the
dispatcher.

CPU only. Everything below was actually run; the outputs are copy-pasted.

## The pieces

* `export_decoder.py` — traces `bytes + indices -> frames`, checks the exported
  program against eager, and writes an AOTInductor `.pt2` package.
* `main.cpp` — a C++ program with no Python: `dlopen`s the torchcodec libraries,
  queries metadata through the dispatcher, and runs the `.pt2`.

## Run it

```bash
# 1. Build torchcodec (any normal build works, e.g. pip install -e .)

# 2. Export + AOTInductor-compile
python export_demo/export_decoder.py test/resources/nasa_13013.mp4 /tmp/decoder.pt2

# 3. Build the C++ program. Nothing here mentions Python.
TORCH=$(python -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).parent)')
g++ -std=c++17 -O2 export_demo/main.cpp -o /tmp/decode_video \
    -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" \
    -L"$TORCH/lib" -Wl,-rpath,"$TORCH/lib" -ltorch -ltorch_cpu -lc10 -ldl

# 4. Run it
/tmp/decode_video src/torchcodec test/resources/nasa_13013.mp4 /tmp/decoder.pt2
```

Output:

```
metadata: {
"averageFpsFromHeader": 29.97002997002997,
"beginStreamSecondsFromContent": 0,
"bestAudioStreamIndex": 4,
"bestVideoStreamIndex": 3,
"bitRate": 412365,
"codec": "h264",
"durationSecondsFromHeader": 13.013,
"endStreamSecondsFromContent": 13.013,
"height": 270,
"numFramesFromHeader": 390,
"width": 480
}
frames: [4, 3, 270, 480] unsigned char, sum=16251293
```

`16251293` is the same sum Python gets for frames `[0, 10, 20, 100]`, both from
the core ops and from `VideoDecoder.get_frames_at`. Note that the program was
exported with 3 indices and run with 4: both the encoded-bytes length and the
number of indices are dynamic.

`LD_DEBUG=libs` on that run loads 98 shared libraries and **zero** matches for
`libpython`.

## The exported graph

```
graph():
    %video_data : placeholder
    %frame_indices : placeholder
    %create_video_decoder_from_tensor : torchcodec_ns.create_video_decoder_from_tensor.default(%video_data, num_threads=1)
    %to : aten.to.dtype(%frame_indices, torch.int64)
    %get_frames_at_indices : torchcodec_ns.get_frames_at_indices.default(%create_video_decoder_from_tensor, frame_indices=%to)
    %getitem_3 : getitem(%get_frames_at_indices, 0)
    ... sym_size / _assert_scalar for the data-dependent H, W, C ...
    return (getitem_3,)
```

Two custom-op nodes, connected by the decoder handle. No mutation, no effect
tokens, no constants baked into the graph. `strict=True` works too, as long as
`dynamic_shapes` is passed (otherwise the number of indices is specialised).

## What had to change, and why

### 1. A single factory op: `create_video_decoder_from_tensor`

`create_from_tensor` + `add_video_stream` works in eager Python but not in a
traced graph: `add_video_stream` is a state mutation returning nothing, so
nothing orders it before the frame reads and nothing keeps it alive. Fusing the
two into one factory op makes the graph a pure dataflow DAG — the handle *is*
the dependency edge.

### 2. The frame-reading ops take a plain `Tensor` handle, not `Tensor(a!)`

`(a!)` is what today's ops use to stop `torch.compile` from reordering them. But
it also tells functionalization it may **clone** the argument, and cloning a
handle tensor copies the first 64 bytes of the C++ decoder object instead of the
object. That is undefined behaviour: `run_decompositions()` on an `(a!)` graph
gives wrong state or segfaults. Neither op on this path needs `(a!)`: random
frame access is self-contained and the decoder is fully configured by the
factory op.

Trade-off: without `(a!)` these ops can be CSE'd or DCE'd. Both are harmless
here (same handle + same indices ⇒ same frames; an unused decode is dead work),
but it would not be safe for the cursor-based ops (`seek_to_pts`,
`get_next_frame`), which is why they are left alone. The general fix for those
is effect tokens (`torch.library._register_effectful_op`), not `(a!)`.

### 3. Fake impls

* `create_from_{file,tensor}` / `_create_from_file_like`: `seek_mode` had no
  default, and arguments left at their schema default aren't forwarded to the
  fake, so tracing raised `TypeError` immediately. One-word fix each.
* `get_frames_at_indices`: was claiming `float32` data and 0-dim `float32`
  pts/duration; the real op returns `uint8` data and 1-D `float64`. The batch
  size is now taken from `frame_indices` instead of being data-dependent.

### 4. No pybind11 in the custom-ops library

`custom_ops.cpp` included `<pybind11/pybind11.h>` without using it, which left
93 undefined `Py*` symbols in `libtorchcodec_custom_opsN.so`. Some of those are
data symbols, so `dlopen` from a Python-free process would fail. Removing the
include (and the `pybind11::module` link for that target) brings it to zero.
`libtorchcodec_pybind_ops` is untouched — that one is genuinely a Python
extension.

## Known gaps

* **Frame dtype.** The fake hardcodes `uint8`. The real dtype depends on the
  `output_dtype` the stream was configured with, which the tracer can't see
  through the opaque handle. Proper fix: thread `output_dtype` into the
  frame-reading op's schema, or split the op per dtype.
* **Metadata can't be a graph output.** It's a JSON string; the tracer sees the
  fake's `""` and constant-folds it. Calling the op through the dispatcher from
  C++ works (as here) but builds a second, throwaway decoder. If metadata is
  needed inside the graph, it should be exposed as a numeric tensor
  (`[num_frames, height, width, ...]`).
* **`H`/`W`/`C` are data-dependent** (`u0, u1, u2`) so the graph carries
  `_assert_scalar` guards. AOTInductor handles this for a plain multi-output op,
  but *not* for one that is mutating or effectful — see the report's §4.6 for
  the two upstream inductor bugs.
* `custom_frame_mappings` isn't plumbed through the new factory op.
* No test coverage yet; `export_decoder.py` is the only thing exercising this.
