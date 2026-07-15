#pragma once
#include <map>
#include <optional>
#include <string>
#include <vector>
#include "AVIOContextHolder.h"
#include "DeviceInterface.h"
#include "FFMPEGCommon.h"
#include "StableABICompat.h"
#include "StreamOptions.h"

extern "C" {
#include <libavutil/dict.h>
}

namespace facebook::torchcodec {

/* clang-format off */
//
// Note: [Encoding loop, sample rate conversion and FIFO]
//
// The input samples are in a given format, sample rate, and number of channels.
// We may want to change these properties before encoding. The conversion is
// done in maybeConvertAudioAVFrame() and we rely on libswresample. When sample
// rate conversion is needed, this means two things:
// - swr will be storing samples in its internal buffers, which we'll need to
//   flush at the very end of the encoding process.
// - the converted AVFrame we get back from maybeConvertAudioAVFrame() typically
//   won't have the same number of samples as the original AVFrame. And that's
//   a problem, because some encoders expect AVFrames with a specific and
//   constant number of samples. If we were to send it as-is, we'd get an error
//   in avcodec_send_frame(). In order to feed the encoder with AVFrames
//   with the expected number of samples, we go through an intermediate FIFO
//   from which we can pull the exact number of samples that we need. Note that
//   this involves at least 2 additional copies.
//
// Unlike the old single-stream AudioEncoder where the FIFO was only used when
// sample rate conversion was needed with a fixed-frame-size encoder, the
// MultiStreamEncoder always creates a FIFO. This is because addSamples() can
// be called multiple times with arbitrary chunk sizes, so we always need to
// buffer and re-chunk samples into frame_size-sized batches.
//
/* clang-format on */

class FORCE_PUBLIC_VISIBILITY MultiStreamEncoder {
 public:
  ~MultiStreamEncoder();

  MultiStreamEncoder(const MultiStreamEncoder&) = delete;
  MultiStreamEncoder& operator=(const MultiStreamEncoder&) = delete;
  MultiStreamEncoder(MultiStreamEncoder&&) = delete;
  MultiStreamEncoder& operator=(MultiStreamEncoder&&) = delete;

  MultiStreamEncoder();

  int add_video_stream(
      int height,
      int width,
      double frame_rate,
      std::string device = "cpu",
      std::optional<std::string> codec = std::nullopt,
      std::optional<std::string> pixel_format = std::nullopt,
      std::optional<double> crf = std::nullopt,
      std::optional<std::string> preset = std::nullopt,
      std::optional<std::map<std::string, std::string>> extra_options =
          std::nullopt);
  int add_audio_stream(
      int sample_rate,
      int num_channels,
      std::optional<int> bit_rate = std::nullopt,
      std::optional<int> out_num_channels = std::nullopt,
      std::optional<int> out_sample_rate = std::nullopt);
  void open(std::string_view file_name);
  void open(
      std::string_view format_name,
      std::unique_ptr<AVIOContextHolder> avio_context_holder);
  void add_frames(const torch::stable::Tensor& frames, int stream_index);
  void add_samples(const torch::stable::Tensor& samples, int stream_index);
  void close();

 private:
  struct VideoStream {
    int in_height = 0;
    int in_width = 0;
    double in_frame_rate = 0;
    VideoStreamOptions options;
    UniqueAVCodecContext av_codec_context;
    AVStream* av_stream = nullptr;
    int num_encoded_frames = 0;
    std::unique_ptr<DeviceInterface> device_interface;
  };

  struct AudioStream {
    int in_sample_rate = -1;
    int in_num_channels = -1;
    int out_num_channels = -1;
    int out_sample_rate = -1;
    int frame_size = -1;
    int64_t last_encoded_av_frame_pts = 0;
    AudioStreamOptions options;
    UniqueAVCodecContext av_codec_context;
    AVStream* av_stream = nullptr;
    UniqueSwrContext swr_context;
    UniqueAVAudioFifo av_audio_fifo;
  };

  void initialize_video_stream(VideoStream& video_stream);
  void open_streams_and_write_header();
  void encode_video_frame(
      AutoAVPacket& auto_av_packet,
      const UniqueAVFrame& av_frame,
      VideoStream& video_stream);
  void initialize_audio_stream(AudioStream& audio_stream);
  void encode_audio_samples(
      const torch::stable::Tensor& samples,
      AudioStream& audio_stream);
  UniqueAVFrame maybe_convert_audio_av_frame(
      const UniqueAVFrame& av_frame,
      AudioStream& audio_stream);
  void encode_audio_frame_through_fifo(
      AutoAVPacket& auto_av_packet,
      const UniqueAVFrame& av_frame,
      AudioStream& audio_stream,
      bool flush_fifo = false);
  void encode_audio_frame(
      AutoAVPacket& auto_av_packet,
      const UniqueAVFrame& av_frame,
      AudioStream& audio_stream);
  void maybe_flush_swr_and_fifo(
      AutoAVPacket& auto_av_packet,
      AudioStream& audio_stream);
  void flush_buffers();

  UniqueEncodingAVFormatContext av_format_context_;
  std::vector<VideoStream> video_streams_;
  std::vector<AudioStream> audio_streams_;
  bool header_written_ = false;
  UniqueAVDictionary av_format_options_;

  std::unique_ptr<AVIOContextHolder> avio_context_holder_;
  bool closed_ = false;
};

} // namespace facebook::torchcodec
