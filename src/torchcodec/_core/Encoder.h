#pragma once
#include <map>
#include <optional>
#include <string>
#include <vector>
#include "AVIOContextHolder.h"
#include "DeviceInterface.h"
#include "FFMPEGCommon.h"
#include "StreamOptions.h"
#include "TCError.h"

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

  int addVideoStream(
      int height,
      int width,
      double frameRate,
      std::string device = "cpu",
      std::optional<std::string> codec = std::nullopt,
      std::optional<std::string> pixelFormat = std::nullopt,
      std::optional<double> crf = std::nullopt,
      std::optional<std::string> preset = std::nullopt,
      std::optional<std::map<std::string, std::string>> extraOptions =
          std::nullopt);
  int addAudioStream(
      int sampleRate,
      int numChannels,
      std::optional<int> bitRate = std::nullopt,
      std::optional<int> outNumChannels = std::nullopt,
      std::optional<int> outSampleRate = std::nullopt);
  void open(std::string_view fileName);
  void open(
      std::string_view formatName,
      std::unique_ptr<AVIOContextHolder> avioContextHolder);
  void addFrames(const tc::Tensor& frames, int streamIndex);
  void addSamples(const tc::Tensor& samples, int streamIndex);
  void close();

 private:
  struct VideoStream {
    int inHeight = 0;
    int inWidth = 0;
    double inFrameRate = 0;
    VideoStreamOptions options;
    UniqueAVCodecContext avCodecContext;
    AVStream* avStream = nullptr;
    int numEncodedFrames = 0;
    std::unique_ptr<DeviceInterface> deviceInterface;
  };

  struct AudioStream {
    int inSampleRate = -1;
    int inNumChannels = -1;
    int outNumChannels = -1;
    int outSampleRate = -1;
    int frameSize = -1;
    int64_t lastEncodedAVFramePts = 0;
    AudioStreamOptions options;
    UniqueAVCodecContext avCodecContext;
    AVStream* avStream = nullptr;
    UniqueSwrContext swrContext;
    UniqueAVAudioFifo avAudioFifo;
  };

  void initializeVideoStream(VideoStream& videoStream);
  void openStreamsAndWriteHeader();
  void encodeVideoFrame(
      AutoAVPacket& autoAVPacket,
      const UniqueAVFrame& avFrame,
      VideoStream& videoStream);
  void initializeAudioStream(AudioStream& audioStream);
  void encodeAudioSamples(const tc::Tensor& samples, AudioStream& audioStream);
  UniqueAVFrame maybeConvertAudioAVFrame(
      const UniqueAVFrame& avFrame,
      AudioStream& audioStream);
  void encodeAudioFrameThroughFifo(
      AutoAVPacket& autoAVPacket,
      const UniqueAVFrame& avFrame,
      AudioStream& audioStream,
      bool flushFifo = false);
  void encodeAudioFrame(
      AutoAVPacket& autoAVPacket,
      const UniqueAVFrame& avFrame,
      AudioStream& audioStream);
  void maybeFlushSwrAndFifo(
      AutoAVPacket& autoAVPacket,
      AudioStream& audioStream);
  void flushBuffers();

  UniqueEncodingAVFormatContext avFormatContext_;
  std::vector<VideoStream> videoStreams_;
  std::vector<AudioStream> audioStreams_;
  bool headerWritten_ = false;
  UniqueAVDictionary avFormatOptions_;

  std::unique_ptr<AVIOContextHolder> avioContextHolder_;
  bool closed_ = false;
};

} // namespace facebook::torchcodec
