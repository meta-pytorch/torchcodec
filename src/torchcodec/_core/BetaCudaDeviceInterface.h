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

  void initializeInterface(AVStream* stream) override;

  void convertAVFrameToFrameOutput(
      const VideoStreamOptions& videoStreamOptions,
      const AVRational& timeBase,
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor =
          std::nullopt) override;

  bool canDecodePacketDirectly() const override {
    return true;
  }

  int sendPacket(ReferenceAVPacket& packet) override;
  int receiveFrame(UniqueAVFrame& avFrame) override;
  void flush() override;

  // NVDEC callback functions (must be public for C callbacks)
  int streamPropertyChange(CUVIDEOFORMAT* videoFormat);
  int frameReadyForDecoding(CUVIDPICPARAMS* picParams);
  int frameReadyInDisplayOrder(CUVIDPARSERDISPINFO* dispInfo);

 private:
  // Apply bitstream filter, modifies packet in-place
  void applyBSF(ReferenceAVPacket& packet);

  class FrameBuffer {
   public:
    enum class SlotState { BEING_DECODED, READY_FOR_OUTPUT };

    struct Slot {
      CUVIDPARSERDISPINFO dispInfo;
      SlotState state;
      int slotId;

      explicit Slot(int id, SlotState s) : state(s), slotId(id) {
        std::memset(&dispInfo, 0, sizeof(dispInfo));
        TORCH_CHECK(
            state == SlotState::BEING_DECODED,
            "Programmer: are you sure you want to create a slot that is not BEING_DECODED?");
      }
    };

    FrameBuffer() = default;
    ~FrameBuffer() = default;

    void markAsBeingDecoded(int slotId);
    void markSlotReadyAndSetInfo(int slotId, CUVIDPARSERDISPINFO* dispInfo);
    void free(int slotId);
    Slot* findReadySlotWithLowestPts();

    void clear() {
      map_.clear();
    }

   private:
    // Map of slotId to Slot
    std::unordered_map<int, Slot> map_;
  };

  UniqueAVFrame convertCudaFrameToAVFrame(
      CUdeviceptr framePtr,
      unsigned int pitch,
      const CUVIDPARSERDISPINFO& dispInfo);

  CUvideoparser videoParser_ = nullptr;
  UniqueCUvideodecoder decoder_;
  CUVIDEOFORMAT videoFormat_ = {};

  FrameBuffer frameBuffer_;

  bool eofSent_ = false;

  // Flush flag to prevent decode operations during flush (like DALI's
  // isFlushing_)
  bool isFlushing_ = false;

  AVRational timeBase_ = {0, 0};

  UniqueAVBSFContext bitstreamFilter_;

  // Default CUDA interface for color conversion.
  // TODONVDEC P2: we shouldn't need to keep a separate instance of the default.
  // See other TODO there about how interfaces should be completely independent.
  std::unique_ptr<DeviceInterface> defaultCudaInterface_;
};

} // namespace facebook::torchcodec

// Note: [sendPacket, receiveFrame, frame ordering and NVCUVID callbacks]
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
// Because B1 needs both I0 and P2 to be properly decoded, the decode order must
// be: I0 P2 B1 P4 B3 ... which is different from the display order.
//
// We don't have to worry about the decode order: it's up to the parser to
// figure that out. But we have to make sure that receiveFrame() returns frames
// in display order.
//
// SendPacket(AVPacket)'s job is just to send the packet to the NVCUVID parser
// by calling cuvidParseVideoData(packet). When cuvidParseVideoData(packet) is
// called, it may trigger callbacks, particularly:
// - frameReadyForDecoding(picParams)): triggered **in decode order** when the
//   parser has accumulated enough data to decode a frame. We send that frame to
//   the NVDEC hardware for **async** decoding. While that frame is being
//   decoded, we store a light reference (a Slot) to that frame in the
//   frameBuffer_, and mark that slot as BEING_DECODED. The value that uniquely
//   identifies that frame in the frameBuffer_ is its "slotId", which is given
//   to us by NVCUVID in the callback parameter: picParams->CurrPicIdx.
// - frameReadyInDisplayOrder(dispInfo)): triggered **in display order** when a
//   frame is ready to be "displayed" (returned). When it is triggered, we look
//   up the corresponding frame/slot in the frameBuffer_, using
//   dispInfo->picture_index to match it against a given BEING_DECODED slotId.
//   We mark that frame/slot as READY_FOR_OUTPUT.
//   Crucially, this callback also tells us the pts of that frame. We store
//   the pts and other relevant info the slot.
//
// Said differently, from the perspective of the frameBuffer_, at any point in
// time a slot/frame in the frameBuffer_ can be in 3 states:
// - empty: no slot for that slotId exists in the frameBuffer_
// - BEING_DECODED: frameReadyForDecoding was triggered for that frame, and the
//   frame was sent to NVDEC for async decoding. We don't know its pts because
//   the parser didn't trigger frameReadyInDisplayOrder() for that frame yet.
// - READY_FOR_OUTPUT: frameReadyInDisplayOrder was triggered for that frame, it
//   is decoded and ready to be mapped and returned. We know its pts. 
//
// Because frameReadyInDisplayOrder is triggered in display order, we know that
// if a slot is READY_FOR_OUTPUT, then all frames with a lower pts are also
// READY_FOR_OUTPUT, or already returned. So when receiveFrame() is called, we
// just need to look for the READY_FOR_OUTPUT slot with the lowest pts, and
// return that frame. This guarantees that receiveFrame() returns frames in
// display order. If no slot is READY_FOR_OUTPUT, then we return EAGAIN to
// indicate the caller should send more packets.
//
// Simple, innit?