// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "PacketDecoder.h"

namespace facebook::torchcodec {

// TODO_API_BREAKDOWN P1: we should make sure the block APIs can dispatch to
// third-party extensions - all of them.

SharedAVCodecContext create_and_open_codec_context(
    AVStream* stream,
    const AVCodec* av_codec,
    DeviceInterface* device_interface,
    std::optional<int> thread_count) {
  AVCodecContext* raw_codec_context = avcodec_alloc_context3(av_codec);
  STD_TORCH_CHECK(
      raw_codec_context != nullptr, "Failed to allocate codec context");
  SharedAVCodecContext codec_context =
      make_shared_av_codec_context(raw_codec_context);

  int ret =
      avcodec_parameters_to_context(codec_context.get(), stream->codecpar);
  STD_TORCH_CHECK(ret == AVSUCCESS, "avcodec_parameters_to_context failed");

  codec_context->thread_count = thread_count.value_or(0);
  codec_context->pkt_timebase = stream->time_base;

  // We must register the hardware device context with the codec context before
  // calling avcodec_open2(). Otherwise, decoding will happen on the CPU and not
  // the hardware device.
  device_interface->register_hardware_device_with_codec(codec_context.get());
  ret = avcodec_open2(codec_context.get(), av_codec, nullptr);
  STD_TORCH_CHECK(
      ret >= AVSUCCESS, get_ffmpeg_error_string_from_error_code(ret));

  codec_context->time_base = stream->time_base;
  return codec_context;
}

namespace {
const AVCodec* find_decoder(
    AVStream* stream,
    DeviceInterface* device_interface) {
  const AVCodec* av_codec = avcodec_find_decoder(stream->codecpar->codec_id);
  STD_TORCH_CHECK(av_codec != nullptr, "Codec not found");
  if (stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
    av_codec = device_interface->find_codec(stream->codecpar->codec_id)
                   .value_or(av_codec);
  }
  return av_codec;
}
} // namespace

PacketDecoder::PacketDecoder(
    const Demuxer& demuxer,
    const StableDevice& device,
    std::optional<int> ffmpeg_thread_count) {
  device_interface_ = create_device_interface(device);
  STD_TORCH_CHECK(
      device_interface_ != nullptr,
      "Failed to create device interface. This should never happen, please report.");

  AVStream* stream = demuxer.active_stream();
  time_base_ = stream->time_base;
  const AVCodec* av_codec = find_decoder(stream, device_interface_.get());
  codec_context_ = create_and_open_codec_context(
      stream, av_codec, device_interface_.get(), ffmpeg_thread_count);
  device_interface_->initialize(codec_context_);

  VideoStreamOptions options;
  // This block hands frames to the caller untouched, so it must not discard
  // precision a later ColorConverter or materialize() might want.
  options.decode_precision = DecodePrecision::NATIVE;
  options.device = device;

  device_interface_->initialize_video_decoding(
      stream, demuxer.format_context(), options);

  // From the codec context: a hardware decoder may use a wider container.
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(codec_context_->pix_fmt);
  if (desc != nullptr) {
    bit_depth_ = static_cast<int64_t>(desc->comp[0].depth);
  }
}

int PacketDecoder::send_packet(AVPacket* packet) {
  // The decode seam expects a ReferenceAVPacket. Copy a reference of the
  // caller- owned packet into a temporary one (cheap, refcount bump); the
  // temporary is unref'd on scope exit while the caller retains ownership of
  // `packet`.
  AutoAVPacket auto_packet;
  ReferenceAVPacket ref(auto_packet);
  int status = av_packet_ref(ref.get(), packet);
  STD_TORCH_CHECK(status >= AVSUCCESS, "av_packet_ref failed");
  return device_interface_->send_packet(ref);
}

int PacketDecoder::send_eof() {
  return device_interface_->send_eof_packet();
}

int PacketDecoder::receive_frame(UniqueAVFrame& av_frame) {
  int status = device_interface_->receive_frame(av_frame);
  if (status == AVSUCCESS) {
    device_interface_->make_frame_standalone(av_frame);
  }
  return status;
}

FramePlanes frame_to_planes(
    const AVFrame& av_frame,
    const StableDevice& device,
    int64_t stream_bit_depth,
    const torch::stable::Tensor& tensor_handle) {
  auto pix_fmt = static_cast<AVPixelFormat>(av_frame.format);
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(pix_fmt);
  STD_TORCH_CHECK(desc != nullptr, "Unknown pixel format on decoded frame");
  const char* pix_fmt_name = av_get_pix_fmt_name(pix_fmt);
  std::string fmt_name = pix_fmt_name ? pix_fmt_name : "unknown";

  STD_TORCH_CHECK(
      !(desc->flags &
        (AV_PIX_FMT_FLAG_BITSTREAM | AV_PIX_FMT_FLAG_PAL |
         AV_PIX_FMT_FLAG_FLOAT)),
      "Cannot expose ",
      fmt_name,
      " as a view: sub-byte-packed, palettised, and float pixel formats need a copy.");
  STD_TORCH_CHECK(
      desc->nb_components >= 1 && desc->nb_components <= 4,
      fmt_name,
      " has an unsupported number of components.");

  const char* colorspace_name = av_color_space_name(av_frame.colorspace);
  const char* color_range_name = av_color_range_name(av_frame.color_range);

  FramePlanes result;
  result.pix_fmt = fmt_name;
  result.colorspace = colorspace_name ? colorspace_name : "unknown";
  result.color_range = color_range_name ? color_range_name : "unknown";

  // Where container and source depth differ, the samples sit in the high bits.
  int64_t container_depth = static_cast<int64_t>(desc->comp[0].depth);
  result.bit_depth =
      (stream_bit_depth > 0 && stream_bit_depth <= container_depth)
      ? stream_bit_depth
      : container_depth;
  result.sample_shift = container_depth - result.bit_depth;

  for (int c = 0; c < desc->nb_components; ++c) {
    const AVComponentDescriptor& comp = desc->comp[c];
    int64_t bytes_per_sample = (comp.depth > 8) ? 2 : 1;
    int64_t linesize = av_frame.linesize[comp.plane];
    STD_TORCH_CHECK(
        comp.depth <= 16 && linesize > 0 && comp.step % bytes_per_sample == 0 &&
            linesize % bytes_per_sample == 0,
        "Cannot expose component ",
        c,
        " of ",
        fmt_name,
        " as a view: its samples aren't a whole number of bytes, or the frame "
        "is stored bottom-up (negative line size).");

    // Each plan is output as a 2D tensor. Y is full size while U/V (the chroma)
    // are potentially subsampled by 2.
    // Only the chroma components are subsampled; luma and alpha are full size.
    bool is_chroma = !(desc->flags & AV_PIX_FMT_FLAG_RGB) && (c == 1 || c == 2);
    int64_t height = is_chroma
        ? AV_CEIL_RSHIFT(av_frame.height, desc->log2_chroma_h)
        : av_frame.height;
    int64_t width = is_chroma
        ? AV_CEIL_RSHIFT(av_frame.width, desc->log2_chroma_w)
        : av_frame.width;

    int64_t sizes[] = {height, width};
    int64_t strides[] = {
        linesize / bytes_per_sample, comp.step / bytes_per_sample};

    // The planes are views on the AVFrame's data. The AVFrame and its data are
    // owned by the tensor_handle. We want the planes to outlive the Python
    // tensor handle (see test_materialize_planes_outlive_frame). So for each
    // plane, we create a (shallow) copy of the tensor handle, and capture it in
    // the plane's deleter. As long as a handle [copy] lives, the AVFrame and
    // its data are alive.
    torch::stable::Tensor handle_copy = tensor_handle;
    result.planes.push_back(torch::stable::from_blob(
        av_frame.data[comp.plane] + comp.offset,
        {sizes, 2},
        {strides, 2},
        device,
        (comp.depth > 8) ? kStableUInt16 : kStableUInt8,
        [handle_copy](void*) {}));
  }

  return result;
}

} // namespace facebook::torchcodec
