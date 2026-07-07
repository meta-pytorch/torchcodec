# Low-level decoding building blocks (Demuxer / Decoder / ColorConverter)

## Context

**Why:** `VideoDecoder` is a single coarse box: demux, decode, and color-conversion run
back-to-back on one thread, so frame N's decode can't start until frame N-1's
color-conversion finished. We want to let a user (or an agent doing hill-climbing over
pipeline parameters — thread counts, which stage on which thread, queue depths, batch
sizes, number of parallel videos) build their *own* overlapped pipeline out of
lower-level pieces. Because our PyTorch custom ops release the GIL, exposing the three
stages as separate GIL-free ops lets plain Python threads achieve real parallelism —
**we write no threading in C++**; we hand out building blocks and the user wires them.

Optimal overlap differs by device (motivating the breakdown):
- **CPU:** color-conversion (swscale) is expensive → overlap `[demux+decode]` ∥ `[color-convert]`.
- **GPU:** color-conversion is a cheap CUDA kernel → overlap `[demux]` ∥ `[decode+color-convert]`.

**Second bird:** users have asked for YUV output instead of RGB. Decode already produces
YUV; RGB only appears after color-conversion. Exposing color-conversion as its own step
lets us offer a `ColorConverter(output="yuv420p")` mode.

**Non-goals:** `VideoDecoder` and `SingleStreamDecoder`'s public op surface stay
behavior-identical. No duplication of demux/decode/convert logic — we *refactor* the
monolith into shared building blocks. First cut is **CPU, single video**; GPU and
multi-video are later phases (design accommodates them but they are out of scope here).

## Decisions (confirmed with user)

1. **Intermediate data = both forms.** Default: zero-copy **opaque handles** (raw
   `AVPacket`/`AVFrame` pointer laundered through a `[1]` int64 tensor, exactly like the
   existing `wrap_decoder_pointer_to_tensor`). Opt-in: **materialized** serializable
   forms (`Packet.to_tensor()` → bytes+meta; `Frame.to_tensor()` → native YUV
   planes+meta) for crossing a process boundary. Handles are thread-only (raw pointers
   are meaningless in another process).
2. **CPU single-video first.**
3. **YUV feature = ColorConverter output mode** (`output="yuv420p"`), producing a
   normalized planar YUV via swscale/filtergraph. This is distinct from the raw-frame
   `Frame.to_tensor()` materialization (which is native NV12/whatever, for portability).

## What exists today (grounding)

- **Demux** is inline `av_read_frame(...)` in `SingleStreamDecoder::decode_av_frame`
  (`SingleStreamDecoder.cpp:1532`); it is NOT abstracted. Seek/keyframe/index policy also
  lives in `SingleStreamDecoder`: `maybe_seek_to_before_desired_pts` (:1436),
  `can_we_avoid_seeking` (:1333), `scan_file_and_update_metadata_and_index` (:280),
  `get_pts`/`seconds_to_index_lower_bound`/`_upper_bound` (:1770/:1704/:1738), keyframe
  lookup helpers, and the `all_frames`/`key_frames` tables in the private `StreamInfo`.
- **Decode** already has a clean seam: `DeviceInterface::send_packet` / `send_eof_packet`
  / `receive_frame` / `flush` (`DeviceInterface.h:114-157`).
- **Color-convert** already has a clean seam:
  `DeviceInterface::convert_av_frame_to_frame_output` (`DeviceInterface.h:101`), wrapped
  by `SingleStreamDecoder::convert_av_frame_to_frame_output` (:1600, stamps pts/duration).
- Intermediate types: packets via `AutoAVPacket`/`ReferenceAVPacket`
  (`FFMPEGCommon.h:157-181`); frames via `UniqueAVFrame` (`FFMPEGCommon.h:84`).
- Boundary convention (`custom_ops.cpp`): C++ object → `[1]` int64 CPU tensor via
  `from_blob` + capturing deleter (`wrap_decoder_pointer_to_tensor` :109, unwrap :156).
  Ops in `STABLE_TORCH_LIBRARY(torchcodec_ns)` take only Tensor/int/float/bool/str;
  frames cross as `std::tuple<Tensor,Tensor,Tensor>` (`OpsFrameOutput` :202) with pts/
  duration as 0-D float64 tensors; every tensor-returning op has an `@register_fake` in
  `ops.py` (:250+). Decoder-mutating ops are annotated `Tensor(a!)`.

## Target Python API (new namespace: `torchcodec.pipeline`)

Building blocks (compose them on your own threads):

```python
from torchcodec.pipeline import Demuxer, Decoder, ColorConverter

demuxer   = Demuxer("v.mp4", stream_index=0, seek_mode="exact")
decoder   = Decoder(demuxer, num_ffmpeg_threads=4)   # bound to demuxer (needs codec params, stateful)
converter = ColorConverter(decoder, output="rgb")    # or output="yuv420p"  <-- YUV feature

# --- fully manual loop (thread it however you like) ---
for packet in demuxer:                 # demux  -> Packet (opaque handle)
    for frame in decoder.decode(packet):   # decode -> Frame (opaque YUV handle)
        out = converter.convert(frame)     # color-convert -> tensor (RGB or YUV)

# --- opt-in materialization to cross a process boundary ---
blob = packet.to_tensor()              # (uint8 bytes, int64 meta[pts,dts,flags,...])
pkt2 = Packet.from_tensor(*blob)       # reconstruct in another process
yuv  = frame.to_tensor()               # native YUV planes + JSON meta
```

Composed convenience (sequential, no threading) so `get_frames_at` / `get_frames_played_at`
keep working — the seek/index intelligence lives in the Demuxer:

```python
pipe = Pipeline(demuxer, decoder, converter)   # thin wire-up of the 3 blocks
fb = pipe.get_frames_at([10, 20, 30])          # reuses argsort/dedup/seek logic
fb = pipe.get_frames_played_at([1.0, 2.0])
```

Binding rules (match the C++ constraints):
- `Decoder` is **hard-bound** to one `Demuxer` (needs codec params; stateful; not
  thread-safe within a stream). One demux thread per video.
- `ColorConverter` on **CPU** may be standalone/shared; the `Decoder` arg supplies the
  transform/dtype config. (On GPU later it must share the decoder's DeviceInterface.)

## Design contract & completeness checklist

**We are the building-block layer, not a scheduler (hard non-goal).** We deliver GIL-free,
individually-callable stages over movable handles; we do NOT ship a thread pool, async
executor, pipeline runner, per-stage concurrency/ordering controls, or IPC machinery. The
user (or an agent hill-climbing over its own threading) writes the scheduling with plain
`threading` + `queue.Queue`. Every stage must therefore be *scheduler-friendly*:
- a stage is a plain callable `handle -> handle` (thin methods, no hidden global state);
- opaque handles are freely movable across threads within a process;
- packets (only) are serializable across a process boundary via `to_tensor()`.

To make sure the block set is complete for a caller building their own overlapped pipeline,
the design must also cover (fold into Phase 1 unless noted):
- **Handles carry `pts`.** The opaque `Packet`/`Frame` handles must expose `pts`/`duration`
  (as `FrameBatch` already does) so a caller running stages out-of-order for throughput can
  reorder results by presentation time. Document this reordering responsibility as the
  caller's, not ours.
- **Packet reuse.** Allow feeding the same packet(s) to two decoders/converters (e.g. two
  transforms) without re-demuxing — clone/refcount semantics on the packet handle.
- **Optional (later phase):** an explicit CPU→GPU transfer building block (pinned memory +
  dedicated stream) so H2D copy overlaps compute in a CPU-decode→GPU-train pattern; and a
  raw FFmpeg `filter_desc` escape hatch alongside the curated `transform_specs`.

## User code examples

### Use-case 1 — single video, overlapped demux+decode ∥ color-convert (CPU strategy)

The whole point: while thread B color-converts frame N-1, thread A already decodes frame N.
Because the ops release the GIL, these two Python threads run on two cores.

```python
import threading, queue
from torchcodec.pipeline import Demuxer, Decoder, ColorConverter

demuxer   = Demuxer("video.mp4", stream_index=0)
decoder   = Decoder(demuxer, num_ffmpeg_threads=4)   # bound to this demuxer
converter = ColorConverter(decoder, output="rgb")

frame_q = queue.Queue(maxsize=8)                 # bounded -> backpressure
_SENTINEL = object()
frames_out = []

def demux_and_decode():                          # thread A: demux + decode
    for packet in demuxer:                        # -> Packet (opaque handle)
        for frame in decoder.decode(packet):      # -> Frame (opaque YUV handle)
            frame_q.put(frame)
    for frame in decoder.flush():                 # drain the codec's buffered frames
        frame_q.put(frame)
    frame_q.put(_SENTINEL)

def color_convert():                             # thread B: YUV -> RGB tensor
    while (frame := frame_q.get()) is not _SENTINEL:
        out = converter.convert(frame)            # Frame dataclass (data, pts, duration)
        frames_out.append(out)

a = threading.Thread(target=demux_and_decode)
b = threading.Thread(target=color_convert)
a.start(); b.start(); a.join(); b.join()
```

An agent hill-climbs by varying: `num_ffmpeg_threads`, `frame_q` depth, and *which* stages share a
thread. E.g. the **GPU** strategy is one line different — put decode+convert together and
peel off only demux (color-convert is a cheap kernel there):

```python
def demux_only():                                # thread A: demux
    for packet in demuxer:
        packet_q.put(packet)
    packet_q.put(_SENTINEL)

def decode_and_convert():                        # thread B: decode + color-convert
    while (packet := packet_q.get()) is not _SENTINEL:
        for frame in decoder.decode(packet):
            frames_out.append(converter.convert(frame))
```

Index/time access stays a one-liner via the sequential convenience wrapper (seek/index
logic lives in the Demuxer):

```python
from torchcodec.pipeline import Pipeline
pipe = Pipeline(demuxer, decoder, converter)
fb = pipe.get_frames_at([10, 20, 30])            # FrameBatch, same output as VideoDecoder
fb = pipe.get_frames_played_at([1.0, 2.5])
```

YUV output (the second bird):

```python
yuv_conv = ColorConverter(decoder, output="yuv420p")
frame = next(iter(decoder.decode(next(iter(demuxer)))))
yuv = yuv_conv.convert(frame)                    # planar YUV instead of RGB
```

### Use-case 2 — multiple videos: N demuxers, N decoders, a shared pool of converters

One demux+decode thread **per video** (an `AVFormatContext` / decoder is not thread-safe,
and each `Decoder` is bound to its own `Demuxer`). Color-converters are **not** bound to a
video, so a small shared pool drains a common queue; each frame is tagged with its source
`video_id` so results are recoverable per video.

```python
import threading, queue
from torchcodec.pipeline import Demuxer, Decoder, ColorConverter

videos = ["a.mp4", "b.mp4", "c.mp4", "d.mp4"]
frame_q = queue.Queue(maxsize=64)                # (video_id, Frame) from all decoders
_SENTINEL = object()
results = {i: [] for i in range(len(videos))}    # recover frames per source video
results_lock = threading.Lock()

def demux_and_decode(video_id, path):            # one thread per video
    demuxer = Demuxer(path)
    decoder = Decoder(demuxer, num_ffmpeg_threads=1)  # 1 each: many run concurrently
    for packet in demuxer:
        for frame in decoder.decode(packet):
            frame_q.put((video_id, frame))
    for frame in decoder.flush():
        frame_q.put((video_id, frame))

def convert_worker():                            # pool: not bound to any video
    converter = ColorConverter(output="rgb")      # standalone; rebuilds config per input format
    while True:
        item = frame_q.get()
        if item is _SENTINEL:
            frame_q.put(_SENTINEL)                # re-post so peers also stop
            return
        video_id, frame = item
        rgb = converter.convert(frame)
        with results_lock:
            results[video_id].append(rgb)

producers = [threading.Thread(target=demux_and_decode, args=(i, p))
             for i, p in enumerate(videos)]
converters = [threading.Thread(target=convert_worker) for _ in range(4)]  # tunable pool size

for t in producers + converters: t.start()
for t in producers: t.join()
frame_q.put(_SENTINEL)                            # signal converter pool to drain
for t in converters: t.join()
# results[video_id] -> list of RGB frames for that video
```

Hill-climbing surface here: number of concurrent videos, `num_ffmpeg_threads` per decoder,
converter-pool size, and queue depth — an agent searches these for peak throughput.
(A standalone CPU `ColorConverter` recreates its swscale/filtergraph config when the input
frame format changes, so one pool can serve videos with differing formats. On GPU a
converter must instead be created per-decoder — deferred to the GPU phase.)

## C++ refactor (phased, no logic duplication)

### Phase 0 — extract `StreamDemuxer` (behavior-preserving, no new ops)
New `Demuxer.{h,cpp}` class `StreamDemuxer` (mechanism, not policy). Move — verbatim, not
rewritten — out of `SingleStreamDecoder`:
- ownership of `format_context_` + `avio_context_holder_`, the file/tensor/file-like
  constructors + `initialize_decoder`;
- `select_stream(optional<int>)` = the `av_find_best_stream` + `discard=AVDISCARD_ALL`
  half of `add_stream` (:471,:541-545); exposes selected `AVStream*`/`AVCodecParameters*`;
- `scan_and_build_index()` / `read_custom_frame_mappings(...)` (:280,:371) and the
  `all_frames`/`key_frames` tables (move `FrameInfo`);
- metadata + pts↔index conversions (`get_pts`, `seconds_to_index_lower_bound`/
  `_upper_bound`, keyframe lookup, key-frame identifier);
- `bool next_packet(ReferenceAVPacket&)` = the `av_read_frame` + stream-filter loop
  (:1531-1552);
- `seek_to_keyframe_before_pts(int64_t pts)` = keyframe-corrected `avformat_seek_file`
  (:1453-1473) **without** the flush (flush is the decoder's job);
- seek-avoidance as a **pure query** taking decode feedback as args:
  `bool can_avoid_seek(cursor, last_decoded_pts, reorder_buffer_size, in_flight_frames)`
  — index half stays in demuxer; codec half (`has_b_frames`, `thread_count`) passed in.
Also move `get_pts_or_dts` (anon ns :29-33) into `FFMPEGCommon.h`.
`SingleStreamDecoder` now delegates to `StreamDemuxer`. **All existing tests stay green** —
this is the de-risking step.

### Phase 1 — extract `PacketDecoder` + `ColorConverter`, then opaque handles + CPU thread API
- New `StreamDecoder.{h,cpp}` class `PacketDecoder`: absorbs codec-context construction
  (:504-532), holds the `DeviceInterface` + `SharedAVCodecContext`; API = the existing
  seam (`send_packet`/`send_eof`/`receive_frame`/`flush`) plus `reorder_buffer_size()` /
  `in_flight_frames()` accessors to feed `can_avoid_seek`.
- New `ColorConverter.{h,cpp}`: wraps `convert_av_frame_to_frame_output` + pts/duration
  stamping (:1600); gains a YUV output path (`output="yuv420p"`).
- `SingleStreamDecoder::decode_av_frame` collapses to orchestration over the three
  sub-objects (public methods keep their bodies → VideoDecoder unchanged).
- New `DecodeHandles.{h,cpp}`: opaque `Packet`/`Frame` handles via the existing launder
  pattern — heap `UniqueAVPacket`/`UniqueAVFrame` → `from_blob` `[1]` int64 tensor with a
  deleter running `av_packet_free`/`av_frame_free`. Each emitted packet is its own
  `av_packet_ref` (the `AutoAVPacket` reuse optimization can't cross a thread boundary).
- Materialization: `packet_to_tensors`/`packet_from_tensors`; `frame_to_yuv` (packed YUV
  tensor + JSON meta — prefer a single packed tensor over `Tensor[]`; verify stable-ABI
  `Tensor[]` support before relying on it).

### New custom ops (all in `STABLE_TORCH_LIBRARY(torchcodec_ns)`, `@register_fake` where they return tensors)
- Demuxer: `create_demuxer_from_file/_from_tensor/_from_file_like`, `demuxer_select_stream`,
  `demuxer_scan`, `demuxer_get_container_json_metadata`/`_stream_json_metadata`/
  `_key_frame_indices`, `demuxer_next_packet -> (Tensor, bool)`,
  `demuxer_seek_to_keyframe_before_pts`, `demuxer_seek_for_index`,
  `demuxer_index_for_pts`/`demuxer_pts_for_index`.
  Packet/packet-group handles expose `pts`/`dts`/`duration` accessors so a caller can
  reorder out-of-order results.
- Packet: `packet_to_tensors -> (Tensor,Tensor)`, `packet_from_tensors -> Tensor`.
- Decoder: `create_decoder_for_stream(demuxer, *, num_threads, transform_specs,
  output_dtype)` — the op keeps the existing `num_threads` name (as `_add_video_stream`
  does); the Python `Decoder` wrapper exposes it as `num_ffmpeg_threads` (default `1`) to
  match `VideoDecoder`. `decoder_send_packet -> int`, `decoder_send_eof -> int`,
  `decoder_receive_frame -> (Tensor, int)` (handle, status), `decoder_flush`.
- Color: `create_color_converter(decoder, *, transform_specs, output_dtype, output)`,
  `convert_frame_to_rgb -> (Tensor,Tensor,Tensor)`, `frame_to_yuv -> (Tensor, str)`.

### Hard problems (flagged; only #2–#4 in scope for CPU-first)
1. **GPU frame-surface lifetime (deferred to GPU phase).** NVDEC keeps one surface mapped,
   unmapping on the next `receive_frame` (`BetaCudaDeviceInterface.cpp:679`) → an opaque
   GPU frame handle dies on the next decode. Fix later via forced D2D YUV copy, or keep
   decode+convert coupled and split only demux, or a surface pool.
2. **Split seek-avoidance feedback.** `can_avoid_seek` needs the async "last decoded pts"
   from the decode thread. Kept as a pure demuxer query; the Python scheduler feeds back
   the latest decoded pts. Random access drives explicit `demuxer_seek_for_index` +
   `decoder_flush`; the optimization is for sequential runs.
3. **Flush-on-seek ordering across threads.** Demux-thread seek invalidates decode-thread
   state. `seek_to_keyframe_before_pts` deliberately does NOT flush; use a seek-epoch tag
   on packets so the decode thread flushes when the epoch changes.
4. **`AVFormatContext` not thread-safe** → exactly one demux thread per `Demuxer`;
   parallelism is demux(1)→decode(1)→convert(N) within a video, or one `Demuxer` per video.

## Files

- Change: `SingleStreamDecoder.{h,cpp}` (delegate to new classes), `custom_ops.cpp` (new
  ops + wrap/unwrap for Packet/Frame), `ops.py` (aliases + `@register_fake`),
  `FFMPEGCommon.h` (move `get_pts_or_dts`), `_core/CMakeLists`+BUCK sources (new files).
- Add: `_core/Demuxer.{h,cpp}`, `_core/StreamDecoder.{h,cpp}`, `_core/ColorConverter.{h,cpp}`,
  `_core/DecodeHandles.{h,cpp}`, `pipeline/{__init__,_demuxer,_decoder,_color_converter,
  _pipeline,_frame}.py`.
- Untouched: `decoders/_video_decoder.py`, all `VideoDecoder` ops.

## Verification

- **Phase 0/1 safety net:** full `pytest` (incl. `-m ""`) and the C++ gtest suite stay
  green after each extraction — proves behavior preservation for `VideoDecoder`. Pay
  special attention to seek-heuristic edge cases (`INT64_MIN` cursor init, mkv/webm index
  quirks noted at `SingleStreamDecoder.h:365-376`).
- **New API correctness:** new tests assert the building-block pipeline returns
  frame-identical output to `VideoDecoder.get_frames_at`/`get_frames_played_at` on the
  same assets (`assert_frames_equal`); round-trip `Packet.to_tensor()/from_tensor()` and
  `frame_to_yuv`; `ColorConverter(output="yuv420p")` shape/format checks.
- **Overlap win (the point):** a benchmark script decoding a video (a) monolithically vs
  (b) with demux+decode on one thread and color-convert on another (Python
  `threading` + `queue.Queue`), showing wall-clock speedup on CPU — confirms the GIL is
  released and the breakdown pays off.
- Run `pre-commit run --all-files` and `mypy` before finishing.

## Follow-ups (additive, non-breaking — NOT in scope now)

These extend the design without changing anything above: new demuxer methods and a new
one-shot decode op, layered on top of the per-packet building blocks once they exist. Do
NOT design for them now.

- **Packet-group / clip granularity.** Today's units are one packet in / one-or-more frames
  out (fine-grained, ideal for demux∥decode overlap within a single video). As a follow-up,
  add a multi-packet "packet group" handle plus a one-shot
  `decoder_decode_packets(decoder, packet_group) -> frame-group-handle` that drains a whole
  group in a single GIL-free call — the natural "decode this clip" unit for a
  many-videos/one-clip-each scheduling pattern. Purely additive: the per-packet
  `send_packet`/`receive_frame` path is unchanged.
- **Windowed / chunked demux.** Demuxer helpers that assemble those groups:
  `demuxer_next_packets(demuxer, num_packets)` and
  `demuxer_packets_for_time_range(demuxer, start, stop)` (seek + collect the covering
  packets into one group handle). Decomposes what `get_frames_played_in_range` does
  internally into a reusable "collect packets" block.
- **Optional building blocks** (also later): an explicit CPU→GPU transfer block (pinned
  memory + dedicated stream) so H2D copy overlaps compute; a raw FFmpeg `filter_desc`
  escape hatch alongside the curated `transform_specs`; packet clone/reuse so the same
  packets can feed two decoders/converters without re-demuxing.
