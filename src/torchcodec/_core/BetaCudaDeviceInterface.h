// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// BETA CUDA device interface that provides direct control over NVDEC
// while keeping FFmpeg for demuxing. A lot of the logic, particularly the use
// of a cache for the decoders, is inspired by DALI's implementation which is
// APACHE 2.0:
// https://github.com/NVIDIA/DALI/blob/c7539676a24a8e9e99a6e8665e277363c5445259/dali/operators/video/frames_decoder_gpu.cc#L1
//
// NVDEC / NVCUVID docs:
// https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-video-decoder-api-prog-guide/index.html#using-nvidia-video-decoder-nvdecode-api

#pragma once

#include "src/torchcodec/_core/CUDACommon.h"
#include "src/torchcodec/_core/Cache.h"
#include "src/torchcodec/_core/DeviceInterface.h"
#include "src/torchcodec/_core/FFMPEGCommon.h"
#include "src/torchcodec/_core/NVDECCache.h"

#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>

#include "src/torchcodec/_core/nvcuvid_include/cuviddec.h"
#include "src/torchcodec/_core/nvcuvid_include/nvcuvid.h"

namespace facebook::torchcodec {

class BetaCudaDeviceInterface : public DeviceInterface {
 public:
  explicit BetaCudaDeviceInterface(const torch::Device& device);
  virtual ~BetaCudaDeviceInterface();

  void initialize(
      const AVStream* avStream,
      const UniqueDecodingAVFormatContext& avFormatCtx,
      const SharedAVCodecContext& codecContext) override;

  void convertAVFrameToFrameOutput(
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor =
          std::nullopt) override;

  int sendPacket(ReferenceAVPacket& packet) override;
  int sendEOFPacket() override;
  int receiveFrame(UniqueAVFrame& avFrame) override;
  void flush() override;

  // NVDEC callback functions (must be public for C callbacks)
  int streamPropertyChange(CUVIDEOFORMAT* videoFormat);
  int frameReadyForDecoding(CUVIDPICPARAMS* picParams);
  int frameReadyInDisplayOrder(CUVIDPARSERDISPINFO* dispInfo);

  std::string getDetails() override;

 private:
  int sendCuvidPacket(CUVIDSOURCEDATAPACKET& cuvidPacket);

  void initializeBSF(
      const AVCodecParameters* codecPar,
      const UniqueDecodingAVFormatContext& avFormatCtx);
  // Apply bitstream filter, returns filtered packet or original if no filter
  // needed.
  ReferenceAVPacket& applyBSF(
      ReferenceAVPacket& packet,
      ReferenceAVPacket& filteredPacket);

  CUdeviceptr previouslyMappedFrame_ = 0;
  void unmapPreviousFrame();

  UniqueAVFrame convertCudaFrameToAVFrame(
      CUdeviceptr framePtr,
      unsigned int pitch,
      const CUVIDPARSERDISPINFO& dispInfo);

  UniqueAVFrame transferCpuFrameToGpuNV12(UniqueAVFrame& cpuFrame);

  CUvideoparser videoParser_ = nullptr;
  UniqueCUvideodecoder decoder_;
  CUVIDEOFORMAT videoFormat_ = {};

  std::queue<CUVIDPARSERDISPINFO> readyFrames_;

  bool eofSent_ = false;

  AVRational timeBase_ = {0, 1};
  AVRational frameRateAvgFromFFmpeg_ = {0, 1};

  UniqueAVBSFContext bitstreamFilter_;

  // NPP context for color conversion
  UniqueNppContext nppCtx_;

  std::unique_ptr<DeviceInterface> cpuFallback_;
  bool nvcuvidAvailable_ = false;
  UniqueSwsContext swsContext_;
  SwsFrameContext prevSwsFrameContext_;
};

} // namespace facebook::torchcodec

/* clang-format off */
// Note: [General design, sendPacket, receiveFrame, frame ordering and NVCUVID callbacks]
//
// At a high level, this decoding interface mimics the FFmpeg send/receive
// architecture:
// - sendPacket(AVPacket) sends an AVPacket from the FFmpeg demuxer to the
//   NVCUVID parser.
// - receiveFrame(AVFrame) is a non-blocking call:
//   - if a frame is ready **in display order**, it must return it. By display
//   order, we mean that receiveFrame() must return frames with increasing pts
//   values when called successively.
//   - if no frame is ready, it must return AVERROR(EAGAIN) to indicate the
//   caller should send more packets.
//
// The rest of this note assumes you have a reasonable level of familiarity with
// the sendPacket/receiveFrame calling pattern. If you don't, look up the core
// decoding loop in SingleVideoDecoder.
//
// The frame re-ordering problem:
// Depending on the codec and on the encoding parameters, a packet from a video
// stream may contain exactly one frame, more than one frame, or a fraction of a
// frame. And, there may be non-linear frame dependencies because of B-frames,
// which need both past *and* future frames to be decoded. Consider the
// following stream, with frames presented in display order: I0 B1 P2 B3 P4 ...
// - I0 is an I-frame (also key frame, can be decoded independently)
// - B1 is a B-frame (bi-directional) which needs both I0 and P2 to be decoded
// - P2 is a P-frame (predicted frame) which only needs I0 to be decodec.
//
// Because B1 needs both I0 and P2 to be properly decoded, the decode order
// (packet order), defined by the encoder, must be: I0 P2 B1 P4 B3 ... which is
// different from the display order.
//
// SendPacket(AVPacket)'s job is just to pass down the packet to the NVCUVID
// parser by calling cuvidParseVideoData(packet). When
// cuvidParseVideoData(packet) is called, it may trigger callbacks,
// particularly:
// - streamPropertyChange(videoFormat): triggered once at the start of the
//   stream, and possibly later if the stream properties change (e.g.
//   resolution).
// - frameReadyForDecoding(picParams)): triggered **in decode order** when the
//   parser has accumulated enough data to decode a frame. We send that frame to
//   the NVDEC hardware for **async** decoding.
// - frameReadyInDisplayOrder(dispInfo)): triggered **in display order** when a
//   frame is ready to be "displayed" (returned). At that point, the parser also
//   gives us the pts of that frame. We store (a reference to) that frame in a
//   FIFO queue: readyFrames_.
//
// When receiveFrame(AVFrame) is called, if readyFrames_ is not empty, we pop
// the front of the queue, which is the next frame in display order, and map it
// to an AVFrame by calling cuvidMapVideoFrame(). If readyFrames_ is empty we
// return EAGAIN to indicate the caller should send more packets.
//
// There is potentially a small inefficiency due to the callback design: in
// order for us to know that a frame is ready in display order, we need the
// frameReadyInDisplayOrder callback to be triggered. This can only happen
// within cuvidParseVideoData(packet) in sendPacket(). This means there may be
// the following sequence of calls:
//
// sendPacket(relevantAVPacket)
//   cuvidParseVideoData(relevantAVPacket)
//     frameReadyForDecoding()
//       cuvidDecodePicture()            Send frame to NVDEC for async decoding
//
// receiveFrame() -> EAGAIN              Frame is potentially already decoded
//                                       and could be returned, but we don't
//                                       know because frameReadyInDisplayOrder
//                                       hasn't been triggered yet. We'll only
//                                       know after sending another,
//                                       potentially irrelevant packet.
//
// sendPacket(irrelevantAVPacket)
//   cuvidParseVideoData(irrelevantAVPacket)
//     frameReadyInDisplayOrder()       Only now do we know that our target
//                                      frame is ready.
//
// receiveFrame()                       return target frame
//
// How much this matters in practice is unclear, but probably negligible in
// general. Particularly when frames are decoded consecutively anyway, the
// "irrelevantPacket" is actually relevant for a future target frame.
//
// Note that the alternative is to *not* rely on the frameReadyInDisplayOrder
// callback. It's technically possible, but it would mean we now have to solve
// two hard, *codec-dependent* problems that the callback was solving for us:
// - we have to guess the frame's pts ourselves
// - we have to re-order the frames ourselves to preserve display order.
//
/* clang-format on */
