#pragma once
#include <torch/types.h>
#include "src/torchcodec/_core/FFMPEGCommon.h"

namespace facebook::torchcodec {
class AudioEncoder {
 public:
  ~AudioEncoder();

  // TODO-ENCODING: document in public docs that bit_rate value is only
  // best-effort, matching to the closest supported bit_rate. I.e. passing 1 is
  // like passing 0, which results in choosing the minimum supported bit rate.
  // Passing 44_100 could result in output being 44000 if only 44000 is
  // supported.
  AudioEncoder(
      const torch::Tensor wf,
      int sampleRate,
      std::string_view fileName,
      std::optional<int64_t> bit_rate = std::nullopt);
  void encode();

 private:
  void encodeInnerLoop(
      AutoAVPacket& autoAVPacket,
      const UniqueAVFrame& avFrame);
  void flushBuffers();

  UniqueEncodingAVFormatContext avFormatContext_;
  UniqueAVCodecContext avCodecContext_;
  int streamIndex_;

  const torch::Tensor wf_;
  // The *output* sample rate. We can't really decide for the user what it
  // should be. Particularly, the sample rate of the input waveform should match
  // this, and that's up to the user. If sample rates don't match, encoding will
  // still work but audio will be distorted.
  // We technically could let the user also specify the input sample rate, and
  // resample the waveform internally to match them, but that's not in scope for
  // an initial version (if at all).
  int sampleRate_;
};
} // namespace facebook::torchcodec
