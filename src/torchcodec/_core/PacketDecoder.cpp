// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "PacketDecoder.h"

namespace facebook::torchcodec {

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
    std::string_view device_variant,
    std::optional<int> ffmpeg_thread_count) {
  device_interface_ = create_device_interface(device, device_variant);
  STD_TORCH_CHECK(
      device_interface_ != nullptr,
      "Failed to create device interface. This should never happen, please report.");

  AVStream* stream = demuxer.active_stream();
  time_base_ = stream->time_base;
  const AVCodec* av_codec = find_decoder(stream, device_interface_.get());
  codec_context_ = create_and_open_codec_context(
      stream, av_codec, device_interface_.get(), ffmpeg_thread_count);
  device_interface_->initialize(codec_context_);
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
  return device_interface_->receive_frame(av_frame);
}

} // namespace facebook::torchcodec
