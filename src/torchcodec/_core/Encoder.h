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
  void addFrames(const torch::stable::Tensor& frames, int streamIndex);
  void addSamples(const torch::stable::Tensor& samples, int streamIndex);
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
  void encodeAudioSamples(
      const torch::stable::Tensor& samples,
      AudioStream& audioStream);
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
