// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "SingleStreamDecoder.h"
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include "Metadata.h"
#include "torch/types.h"

namespace facebook::torchcodec {
namespace {

// Some videos aren't properly encoded and do not specify pts values for
// packets, and thus for frames. Unset values correspond to INT64_MIN. When that
// happens, we fallback to the dts value which hopefully exists and is correct.
// Accessing AVFrames and AVPackets's pts values should **always** go through
// the helpers below. Then, the "pts" fields in our structs like FrameInfo.pts
// should be interpreted as "pts if it exists, dts otherwise".
// int64_t getPtsOrDts(ReferenceAVPacket& packet) {
//   return packet->pts == INT64_MIN ? packet->dts : packet->pts;
// }

int64_t getPtsOrDts(const UniqueAVFrame& avFrame) {
  return avFrame->pts == INT64_MIN ? avFrame->pkt_dts : avFrame->pts;
}

} // namespace

// --------------------------------------------------------------------------
// CONSTRUCTORS, INITIALIZATION, DESTRUCTORS
// --------------------------------------------------------------------------

SingleStreamDecoder::SingleStreamDecoder(
    const std::string& videoFilePath,
    SeekMode seekMode)
    : seekMode_(seekMode) {
  setFFmpegLogLevel();

  AVFormatContext* rawContext = nullptr;
  int status =
      avformat_open_input(&rawContext, videoFilePath.c_str(), nullptr, nullptr);
  TORCH_CHECK(
      status == 0,
      "Could not open input file: " + videoFilePath + " " +
          getFFMPEGErrorStringFromErrorCode(status));
  TORCH_CHECK(rawContext != nullptr);
  formatContext_.reset(rawContext);

  initializeDecoder();
}

SingleStreamDecoder::SingleStreamDecoder(
    std::unique_ptr<AVIOContextHolder> context,
    SeekMode seekMode)
    : seekMode_(seekMode), avioContextHolder_(std::move(context)) {
  setFFmpegLogLevel();

  TORCH_CHECK(avioContextHolder_, "Context holder cannot be null");

  // Because FFmpeg requires a reference to a pointer in the call to open, we
  // can't use a unique pointer here. Note that means we must call free if open
  // fails.
  AVFormatContext* rawContext = avformat_alloc_context();
  TORCH_CHECK(rawContext != nullptr, "Unable to alloc avformat context");

  rawContext->pb = avioContextHolder_->getAVIOContext();
  int status = avformat_open_input(&rawContext, nullptr, nullptr, nullptr);
  if (status != 0) {
    avformat_free_context(rawContext);
    TORCH_CHECK(
        false,
        "Failed to open input buffer: " +
            getFFMPEGErrorStringFromErrorCode(status));
  }

  formatContext_.reset(rawContext);

  initializeDecoder();
}

void SingleStreamDecoder::initializeDecoder() {
  TORCH_CHECK(!initialized_, "Attempted double initialization.");

  // In principle, the AVFormatContext should be filled in by the call to
  // avformat_open_input() which reads the header. However, some formats do not
  // store enough info in the header, so we call avformat_find_stream_info()
  // which decodes a few frames to get missing info. For more, see:
  //   https://ffmpeg.org/doxygen/7.0/group__lavf__decoding.html
  int status = avformat_find_stream_info(formatContext_.get(), nullptr);
  TORCH_CHECK(
      status >= 0,
      "Failed to find stream info: ",
      getFFMPEGErrorStringFromErrorCode(status));

  if (formatContext_->duration > 0) {
    AVRational defaultTimeBase{1, AV_TIME_BASE};
    containerMetadata_.durationSecondsFromHeader =
        ptsToSeconds(formatContext_->duration, defaultTimeBase);
  }

  if (formatContext_->bit_rate > 0) {
    containerMetadata_.bitRate = formatContext_->bit_rate;
  }

  int bestVideoStream = getBestStreamIndex(AVMEDIA_TYPE_VIDEO);
  if (bestVideoStream >= 0) {
    containerMetadata_.bestVideoStreamIndex = bestVideoStream;
  }

  int bestAudioStream = getBestStreamIndex(AVMEDIA_TYPE_AUDIO);
  if (bestAudioStream >= 0) {
    containerMetadata_.bestAudioStreamIndex = bestAudioStream;
  }

  for (unsigned int i = 0; i < formatContext_->nb_streams; i++) {
    AVStream* avStream = formatContext_->streams[i];
    StreamMetadata streamMetadata;

    TORCH_CHECK(
        static_cast<int>(i) == avStream->index,
        "Our stream index, " + std::to_string(i) +
            ", does not match AVStream's index, " +
            std::to_string(avStream->index) + ".");
    streamMetadata.streamIndex = i;
    streamMetadata.codecName = avcodec_get_name(avStream->codecpar->codec_id);
    streamMetadata.mediaType = avStream->codecpar->codec_type;
    streamMetadata.bitRate = avStream->codecpar->bit_rate;

    int64_t frameCount = avStream->nb_frames;
    if (frameCount > 0) {
      streamMetadata.numFramesFromHeader = frameCount;
    }

    if (avStream->duration > 0 && avStream->time_base.den > 0) {
      streamMetadata.durationSecondsFromHeader =
          ptsToSeconds(avStream->duration, avStream->time_base);
    }
    if (avStream->start_time != AV_NOPTS_VALUE) {
      streamMetadata.beginStreamSecondsFromHeader =
          ptsToSeconds(avStream->start_time, avStream->time_base);
    }

    if (avStream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      double fps = av_q2d(avStream->r_frame_rate);
      if (fps > 0) {
        streamMetadata.averageFpsFromHeader = fps;
      }
      streamMetadata.width = avStream->codecpar->width;
      streamMetadata.height = avStream->codecpar->height;
      streamMetadata.sampleAspectRatio =
          avStream->codecpar->sample_aspect_ratio;
      containerMetadata_.numVideoStreams++;
    } else if (avStream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
      AVSampleFormat format =
          static_cast<AVSampleFormat>(avStream->codecpar->format);
      streamMetadata.sampleRate =
          static_cast<int64_t>(avStream->codecpar->sample_rate);
      streamMetadata.numChannels =
          static_cast<int64_t>(getNumChannels(avStream->codecpar));

      // If the AVSampleFormat is not recognized, we get back nullptr. We have
      // to make sure we don't initialize a std::string with nullptr. There's
      // nothing to do on the else branch because we're already using an
      // optional; it'll just remain empty.
      const char* rawSampleFormat = av_get_sample_fmt_name(format);
      if (rawSampleFormat != nullptr) {
        streamMetadata.sampleFormat = std::string(rawSampleFormat);
      }
      containerMetadata_.numAudioStreams++;
    }

    streamMetadata.durationSecondsFromContainer =
        containerMetadata_.durationSecondsFromHeader;

    containerMetadata_.allStreamMetadata.push_back(streamMetadata);
  }

  if (seekMode_ == SeekMode::exact) {
    scanFileAndUpdateMetadataAndIndex();
  }

  initialized_ = true;
}

int SingleStreamDecoder::getBestStreamIndex(AVMediaType mediaType) {
  AVCodecOnlyUseForCallingAVFindBestStream avCodec = nullptr;
  int streamIndex =
      av_find_best_stream(formatContext_.get(), mediaType, -1, -1, &avCodec, 0);
  return streamIndex;
}

// --------------------------------------------------------------------------
// VIDEO METADATA QUERY API
// --------------------------------------------------------------------------

void SingleStreamDecoder::sortAllFrames() {
  // Sort the allFrames and keyFrames vecs in each stream, and also sets
  // additional fields of the FrameInfo entries like nextPts and frameIndex
  // This is called at the end of a scan, or when setting a user-defined frame
  // mapping.
  for (auto& [streamIndex, streamInfo] : streamInfos_) {
    std::sort(
        streamInfo.keyFrames.begin(),
        streamInfo.keyFrames.end(),
        [](const FrameInfo& frameInfo1, const FrameInfo& frameInfo2) {
          return frameInfo1.pts < frameInfo2.pts;
        });
    std::sort(
        streamInfo.allFrames.begin(),
        streamInfo.allFrames.end(),
        [](const FrameInfo& frameInfo1, const FrameInfo& frameInfo2) {
          return frameInfo1.pts < frameInfo2.pts;
        });

    size_t keyFrameIndex = 0;
    for (size_t i = 0; i < streamInfo.allFrames.size(); ++i) {
      streamInfo.allFrames[i].frameIndex = i;
      if (streamInfo.allFrames[i].isKeyFrame) {
        TORCH_CHECK(
            keyFrameIndex < streamInfo.keyFrames.size(),
            "The allFrames vec claims it has MORE keyFrames than the keyFrames vec. There's a bug in torchcodec.");
        streamInfo.keyFrames[keyFrameIndex].frameIndex = i;
        ++keyFrameIndex;
      }
      if (i + 1 < streamInfo.allFrames.size()) {
        streamInfo.allFrames[i].nextPts = streamInfo.allFrames[i + 1].pts;
      }
    }
    TORCH_CHECK(
        keyFrameIndex == streamInfo.keyFrames.size(),
        "The allFrames vec claims it has LESS keyFrames than the keyFrames vec. There's a bug in torchcodec.");
  }
}

void SingleStreamDecoder::scanFileAndUpdateMetadataAndIndex() {
  if (scannedAllStreams_) {
    return;
  }

  // HARD-CODED VALUES FOR gaid_bad.mp4 - bypassing scan to test DISCARD theory
  // Stream 0 (video): 252 frames, keyframes at pts 1024 and 129024
  {
    int streamIndex = 0;
    auto& streamMetadata = containerMetadata_.allStreamMetadata[streamIndex];
    streamMetadata.beginStreamPtsFromContent = 1024;
    streamMetadata.endStreamPtsFromContent = 130048;
    streamMetadata.numFramesFromContent = 252;
    streamMetadata.beginStreamPtsSecondsFromContent = 0.08;
    streamMetadata.endStreamPtsSecondsFromContent = 10.16;

    // All 252 video frames: pts from 1024 to 129536, stepping by 512
    std::vector<int64_t> videoPts = {1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192, 8704, 9216, 9728, 10240, 10752, 11264, 11776, 12288, 12800, 13312, 13824, 14336, 14848, 15360, 15872, 16384, 16896, 17408, 17920, 18432, 18944, 19456, 19968, 20480, 20992, 21504, 22016, 22528, 23040, 23552, 24064, 24576, 25088, 25600, 26112, 26624, 27136, 27648, 28160, 28672, 29184, 29696, 30208, 30720, 31232, 31744, 32256, 32768, 33280, 33792, 34304, 34816, 35328, 35840, 36352, 36864, 37376, 37888, 38400, 38912, 39424, 39936, 40448, 40960, 41472, 41984, 42496, 43008, 43520, 44032, 44544, 45056, 45568, 46080, 46592, 47104, 47616, 48128, 48640, 49152, 49664, 50176, 50688, 51200, 51712, 52224, 52736, 53248, 53760, 54272, 54784, 55296, 55808, 56320, 56832, 57344, 57856, 58368, 58880, 59392, 59904, 60416, 60928, 61440, 61952, 62464, 62976, 63488, 64000, 64512, 65024, 65536, 66048, 66560, 67072, 67584, 68096, 68608, 69120, 69632, 70144, 70656, 71168, 71680, 72192, 72704, 73216, 73728, 74240, 74752, 75264, 75776, 76288, 76800, 77312, 77824, 78336, 78848, 79360, 79872, 80384, 80896, 81408, 81920, 82432, 82944, 83456, 83968, 84480, 84992, 85504, 86016, 86528, 87040, 87552, 88064, 88576, 89088, 89600, 90112, 90624, 91136, 91648, 92160, 92672, 93184, 93696, 94208, 94720, 95232, 95744, 96256, 96768, 97280, 97792, 98304, 98816, 99328, 99840, 100352, 100864, 101376, 101888, 102400, 102912, 103424, 103936, 104448, 104960, 105472, 105984, 106496, 107008, 107520, 108032, 108544, 109056, 109568, 110080, 110592, 111104, 111616, 112128, 112640, 113152, 113664, 114176, 114688, 115200, 115712, 116224, 116736, 117248, 117760, 118272, 118784, 119296, 119808, 120320, 120832, 121344, 121856, 122368, 122880, 123392, 123904, 124416, 124928, 125440, 125952, 126464, 126976, 127488, 128000, 128512, 129024, 129536};
    std::vector<int64_t> videoKeyPts = {1024, 129024};

    for (int64_t pts : videoPts) {
      FrameInfo frameInfo = {pts};
      if (pts == 1024 || pts == 129024) {
        frameInfo.isKeyFrame = true;
        streamInfos_[streamIndex].keyFrames.push_back(frameInfo);
      }
      streamInfos_[streamIndex].allFrames.push_back(frameInfo);
    }
  }

  // Stream 1 (audio): 469 frames, all keyframes
  {
    int streamIndex = 1;
    auto& streamMetadata = containerMetadata_.allStreamMetadata[streamIndex];
    streamMetadata.beginStreamPtsFromContent = 0;
    streamMetadata.endStreamPtsFromContent = 483072;
    streamMetadata.numFramesFromContent = 469;
    streamMetadata.beginStreamPtsSecondsFromContent = 0;
    streamMetadata.endStreamPtsSecondsFromContent = 10.064;

    std::vector<int64_t> audioPts = {0, 3840, 4864, 5888, 6912, 7936, 8960, 9984, 11008, 12032, 13056, 14080, 15104, 16128, 17152, 18176, 19200, 20224, 21248, 22272, 23296, 24320, 25344, 26368, 27392, 28416, 29440, 30464, 31488, 32512, 33536, 34560, 35584, 36608, 37632, 38656, 39680, 40704, 41728, 42752, 43776, 44800, 45824, 46848, 47872, 48896, 49920, 50944, 51968, 52992, 54016, 55040, 56064, 57088, 58112, 59136, 60160, 61184, 62208, 63232, 64256, 65280, 66304, 67328, 68352, 69376, 70400, 71424, 72448, 73472, 74496, 75520, 76544, 77568, 78592, 79616, 80640, 81664, 82688, 83712, 84736, 85760, 86784, 87808, 88832, 89856, 90880, 91904, 92928, 93952, 94976, 96000, 97024, 98048, 99072, 100096, 101120, 102144, 103168, 104192, 105216, 106240, 107264, 108288, 109312, 110336, 111360, 112384, 113408, 114432, 115456, 116480, 117504, 118528, 119552, 120576, 121600, 122624, 123648, 124672, 125696, 126720, 127744, 128768, 129792, 130816, 131840, 132864, 133888, 134912, 135936, 136960, 137984, 139008, 140032, 141056, 142080, 143104, 144128, 145152, 146176, 147200, 148224, 149248, 150272, 151296, 152320, 153344, 154368, 155392, 156416, 157440, 158464, 159488, 160512, 161536, 162560, 163584, 164608, 165632, 166656, 167680, 168704, 169728, 170752, 171776, 172800, 173824, 174848, 175872, 176896, 177920, 178944, 179968, 180992, 182016, 183040, 184064, 185088, 186112, 187136, 188160, 189184, 190208, 191232, 192256, 193280, 194304, 195328, 196352, 197376, 198400, 199424, 200448, 201472, 202496, 203520, 204544, 205568, 206592, 207616, 208640, 209664, 210688, 211712, 212736, 213760, 214784, 215808, 216832, 217856, 218880, 219904, 220928, 221952, 222976, 224000, 225024, 226048, 227072, 228096, 229120, 230144, 231168, 232192, 233216, 234240, 235264, 236288, 237312, 238336, 239360, 240384, 241408, 242432, 243456, 244480, 245504, 246528, 247552, 248576, 249600, 250624, 251648, 252672, 253696, 254720, 255744, 256768, 257792, 258816, 259840, 260864, 261888, 262912, 263936, 264960, 265984, 267008, 268032, 269056, 270080, 271104, 272128, 273152, 274176, 275200, 276224, 277248, 278272, 279296, 280320, 281344, 282368, 283392, 284416, 285440, 286464, 287488, 288512, 289536, 290560, 291584, 292608, 293632, 294656, 295680, 296704, 297728, 298752, 299776, 300800, 301824, 302848, 303872, 304896, 305920, 306944, 307968, 308992, 310016, 311040, 312064, 313088, 314112, 315136, 316160, 317184, 318208, 319232, 320256, 321280, 322304, 323328, 324352, 325376, 326400, 327424, 328448, 329472, 330496, 331520, 332544, 333568, 334592, 335616, 336640, 337664, 338688, 339712, 340736, 341760, 342784, 343808, 344832, 345856, 346880, 347904, 348928, 349952, 350976, 352000, 353024, 354048, 355072, 356096, 357120, 358144, 359168, 360192, 361216, 362240, 363264, 364288, 365312, 366336, 367360, 368384, 369408, 370432, 371456, 372480, 373504, 374528, 375552, 376576, 377600, 378624, 379648, 380672, 381696, 382720, 383744, 384768, 385792, 386816, 387840, 388864, 389888, 390912, 391936, 392960, 393984, 395008, 396032, 397056, 398080, 399104, 400128, 401152, 402176, 403200, 404224, 405248, 406272, 407296, 408320, 409344, 410368, 411392, 412416, 413440, 414464, 415488, 416512, 417536, 418560, 419584, 420608, 421632, 422656, 423680, 424704, 425728, 426752, 427776, 428800, 429824, 430848, 431872, 432896, 433920, 434944, 435968, 436992, 438016, 439040, 440064, 441088, 442112, 443136, 444160, 445184, 446208, 447232, 448256, 449280, 450304, 451328, 452352, 453376, 454400, 455424, 456448, 457472, 458496, 459520, 460544, 461568, 462592, 463616, 464640, 465664, 466688, 467712, 468736, 469760, 470784, 471808, 472832, 473856, 474880, 475904, 476928, 477952, 478976, 480000, 481024, 482048};

    for (int64_t pts : audioPts) {
      FrameInfo frameInfo = {pts};
      frameInfo.isKeyFrame = true;  // All audio frames are keyframes
      streamInfos_[streamIndex].keyFrames.push_back(frameInfo);
      streamInfos_[streamIndex].allFrames.push_back(frameInfo);
    }
  }

  /* COMMENTED OUT - Original scan logic
  AutoAVPacket autoAVPacket;
  while (true) {
    ReferenceAVPacket packet(autoAVPacket);

    // av_read_frame is a misleading name: it gets the next **packet**.
    int status = av_read_frame(formatContext_.get(), packet.get());

    if (status == AVERROR_EOF) {
      break;
    }

    TORCH_CHECK(
        status == AVSUCCESS,
        "Failed to read frame from input file: ",
        getFFMPEGErrorStringFromErrorCode(status));

    if (packet->flags & AV_PKT_FLAG_DISCARD) {
      continue;
    }

    // We got a valid packet. Let's figure out what stream it belongs to and
    // record its relevant metadata.
    int streamIndex = packet->stream_index;
    auto& streamMetadata = containerMetadata_.allStreamMetadata[streamIndex];
    streamMetadata.beginStreamPtsFromContent = std::min(
        streamMetadata.beginStreamPtsFromContent.value_or(INT64_MAX),
        getPtsOrDts(packet));
    streamMetadata.endStreamPtsFromContent = std::max(
        streamMetadata.endStreamPtsFromContent.value_or(INT64_MIN),
        getPtsOrDts(packet) + packet->duration);
    streamMetadata.numFramesFromContent =
        streamMetadata.numFramesFromContent.value_or(0) + 1;

    // Note that we set the other value in this struct, nextPts, only after
    // we have scanned all packets and sorted by pts.
    FrameInfo frameInfo = {getPtsOrDts(packet)};
    if (packet->flags & AV_PKT_FLAG_KEY) {
      frameInfo.isKeyFrame = true;
      streamInfos_[streamIndex].keyFrames.push_back(frameInfo);
    }
    streamInfos_[streamIndex].allFrames.push_back(frameInfo);
  }

  // Set all per-stream metadata that requires knowing the content of all
  // packets.
  for (size_t streamIndex = 0;
       streamIndex < containerMetadata_.allStreamMetadata.size();
       ++streamIndex) {
    auto& streamMetadata = containerMetadata_.allStreamMetadata[streamIndex];
    auto avStream = formatContext_->streams[streamIndex];

    streamMetadata.numFramesFromContent =
        streamInfos_[streamIndex].allFrames.size();

    // This ensures that we are robust in handling cases where
    // we are decoding in exact mode and numFrames is 0. The current metadata
    // validation logic assumes that these values should not be None
    if (streamMetadata.numFramesFromContent.value() == 0) {
      streamMetadata.beginStreamPtsFromContent = 0;
      streamMetadata.endStreamPtsFromContent = 0;
    }

    if (streamMetadata.beginStreamPtsFromContent.has_value()) {
      streamMetadata.beginStreamPtsSecondsFromContent = ptsToSeconds(
          *streamMetadata.beginStreamPtsFromContent, avStream->time_base);
    }
    if (streamMetadata.endStreamPtsFromContent.has_value()) {
      streamMetadata.endStreamPtsSecondsFromContent = ptsToSeconds(
          *streamMetadata.endStreamPtsFromContent, avStream->time_base);
    }
  }

  // Reset the seek-cursor back to the beginning.
  int status = avformat_seek_file(formatContext_.get(), 0, INT64_MIN, 0, 0, 0);
  TORCH_CHECK(
      status >= 0,
      "Could not seek file to pts=0: ",
      getFFMPEGErrorStringFromErrorCode(status));
  */

  // Sort all frames by their pts.
  sortAllFrames();
  scannedAllStreams_ = true;
}

void SingleStreamDecoder::readCustomFrameMappingsUpdateMetadataAndIndex(
    int streamIndex,
    FrameMappings customFrameMappings) {
  TORCH_CHECK(
      customFrameMappings.all_frames.dtype() == torch::kLong &&
          customFrameMappings.is_key_frame.dtype() == torch::kBool &&
          customFrameMappings.duration.dtype() == torch::kLong,
      "all_frames and duration tensors must be int64 dtype, and is_key_frame tensor must be a bool dtype.");
  const torch::Tensor& all_frames =
      customFrameMappings.all_frames.to(torch::kLong);
  const torch::Tensor& is_key_frame =
      customFrameMappings.is_key_frame.to(torch::kBool);
  const torch::Tensor& duration = customFrameMappings.duration.to(torch::kLong);
  TORCH_CHECK(
      all_frames.size(0) == is_key_frame.size(0) &&
          is_key_frame.size(0) == duration.size(0),
      "all_frames, is_key_frame, and duration from custom_frame_mappings were not same size.");

  // Allocate vectors using num frames to reduce reallocations
  int64_t numFrames = all_frames.size(0);
  streamInfos_[streamIndex].allFrames.reserve(numFrames);
  streamInfos_[streamIndex].keyFrames.reserve(numFrames);
  // Use accessor to efficiently access tensor elements
  auto pts_data = all_frames.accessor<int64_t, 1>();
  auto is_key_frame_data = is_key_frame.accessor<bool, 1>();
  auto duration_data = duration.accessor<int64_t, 1>();

  auto& streamMetadata = containerMetadata_.allStreamMetadata[streamIndex];

  streamMetadata.beginStreamPtsFromContent = pts_data[0];
  streamMetadata.endStreamPtsFromContent =
      pts_data[numFrames - 1] + duration_data[numFrames - 1];

  auto avStream = formatContext_->streams[streamIndex];
  streamMetadata.beginStreamPtsSecondsFromContent = ptsToSeconds(
      *streamMetadata.beginStreamPtsFromContent, avStream->time_base);

  streamMetadata.endStreamPtsSecondsFromContent = ptsToSeconds(
      *streamMetadata.endStreamPtsFromContent, avStream->time_base);

  streamMetadata.numFramesFromContent = numFrames;
  for (int64_t i = 0; i < numFrames; ++i) {
    FrameInfo frameInfo;
    frameInfo.pts = pts_data[i];
    frameInfo.isKeyFrame = is_key_frame_data[i];
    streamInfos_[streamIndex].allFrames.push_back(frameInfo);
    if (frameInfo.isKeyFrame) {
      streamInfos_[streamIndex].keyFrames.push_back(frameInfo);
    }
  }
  sortAllFrames();
}

ContainerMetadata SingleStreamDecoder::getContainerMetadata() const {
  return containerMetadata_;
}

SeekMode SingleStreamDecoder::getSeekMode() const {
  return seekMode_;
}

int SingleStreamDecoder::getActiveStreamIndex() const {
  return activeStreamIndex_;
}

torch::Tensor SingleStreamDecoder::getKeyFrameIndices() {
  validateActiveStream(AVMEDIA_TYPE_VIDEO);
  validateScannedAllStreams("getKeyFrameIndices");

  const std::vector<FrameInfo>& keyFrames =
      streamInfos_[activeStreamIndex_].keyFrames;
  torch::Tensor keyFrameIndices =
      torch::empty({static_cast<int64_t>(keyFrames.size())}, {torch::kInt64});
  for (size_t i = 0; i < keyFrames.size(); ++i) {
    keyFrameIndices[i] = keyFrames[i].frameIndex;
  }

  return keyFrameIndices;
}

// --------------------------------------------------------------------------
// ADDING STREAMS API
// --------------------------------------------------------------------------

void SingleStreamDecoder::addStream(
    int streamIndex,
    AVMediaType mediaType,
    const torch::Device& device,
    const std::string_view deviceVariant,
    std::optional<int> ffmpegThreadCount) {
  TORCH_CHECK(
      activeStreamIndex_ == NO_ACTIVE_STREAM,
      "Can only add one single stream.");
  TORCH_CHECK(
      mediaType == AVMEDIA_TYPE_VIDEO || mediaType == AVMEDIA_TYPE_AUDIO,
      "Can only add video or audio streams.");
  TORCH_CHECK(formatContext_.get() != nullptr);

  AVCodecOnlyUseForCallingAVFindBestStream avCodec = nullptr;

  activeStreamIndex_ = av_find_best_stream(
      formatContext_.get(), mediaType, streamIndex, -1, &avCodec, 0);

  if (activeStreamIndex_ < 0) {
    throw std::invalid_argument(
        "No valid stream found in input file. Is " +
        std::to_string(streamIndex) + " of the desired media type?");
  }

  TORCH_CHECK(avCodec != nullptr);

  StreamInfo& streamInfo = streamInfos_[activeStreamIndex_];
  streamInfo.streamIndex = activeStreamIndex_;
  streamInfo.timeBase = formatContext_->streams[activeStreamIndex_]->time_base;
  streamInfo.stream = formatContext_->streams[activeStreamIndex_];
  streamInfo.avMediaType = mediaType;

  // This should never happen, checking just to be safe.
  TORCH_CHECK(
      streamInfo.stream->codecpar->codec_type == mediaType,
      "FFmpeg found stream with index ",
      activeStreamIndex_,
      " which is of the wrong media type.");

  deviceInterface_ = createDeviceInterface(device, deviceVariant);
  TORCH_CHECK(
      deviceInterface_ != nullptr,
      "Failed to create device interface. This should never happen, please report.");

  // TODO_CODE_QUALITY it's pretty meh to have a video-specific logic within
  // addStream() which is supposed to be generic
  if (mediaType == AVMEDIA_TYPE_VIDEO) {
    avCodec = makeAVCodecOnlyUseForCallingAVFindBestStream(
        deviceInterface_->findCodec(streamInfo.stream->codecpar->codec_id)
            .value_or(avCodec));
  }

  AVCodecContext* codecContext = avcodec_alloc_context3(avCodec);
  TORCH_CHECK(codecContext != nullptr);
  streamInfo.codecContext = makeSharedAVCodecContext(codecContext);

  int retVal = avcodec_parameters_to_context(
      streamInfo.codecContext.get(), streamInfo.stream->codecpar);
  TORCH_CHECK_EQ(retVal, AVSUCCESS);

  streamInfo.codecContext->thread_count = ffmpegThreadCount.value_or(0);
  streamInfo.codecContext->pkt_timebase = streamInfo.stream->time_base;

  // Note that we must make sure to register the harware device context
  // with the codec context before calling avcodec_open2(). Otherwise, decoding
  // will happen on the CPU and not the hardware device.
  deviceInterface_->registerHardwareDeviceWithCodec(
      streamInfo.codecContext.get());
  retVal = avcodec_open2(streamInfo.codecContext.get(), avCodec, nullptr);
  TORCH_CHECK(retVal >= AVSUCCESS, getFFMPEGErrorStringFromErrorCode(retVal));

  streamInfo.codecContext->time_base = streamInfo.stream->time_base;

  // Initialize the device interface with the codec context
  deviceInterface_->initialize(
      streamInfo.stream, formatContext_, streamInfo.codecContext);

  containerMetadata_.allStreamMetadata[activeStreamIndex_].codecName =
      std::string(avcodec_get_name(streamInfo.codecContext->codec_id));

  // We will only need packets from the active stream, so we tell FFmpeg to
  // discard packets from the other streams. Note that av_read_frame() may still
  // return some of those un-desired packet under some conditions, so it's still
  // important to discard/demux correctly in the inner decoding loop.
  for (unsigned int i = 0; i < formatContext_->nb_streams; ++i) {
    if (i != static_cast<unsigned int>(activeStreamIndex_)) {
      formatContext_->streams[i]->discard = AVDISCARD_ALL;
    }
  }
}

void SingleStreamDecoder::addVideoStream(
    int streamIndex,
    std::vector<Transform*>& transforms,
    const VideoStreamOptions& videoStreamOptions,
    std::optional<FrameMappings> customFrameMappings) {
  TORCH_CHECK(
      transforms.empty() || videoStreamOptions.device == torch::kCPU,
      " Transforms are only supported for CPU devices.");

  addStream(
      streamIndex,
      AVMEDIA_TYPE_VIDEO,
      videoStreamOptions.device,
      videoStreamOptions.deviceVariant,
      videoStreamOptions.ffmpegThreadCount);

  auto& streamMetadata =
      containerMetadata_.allStreamMetadata[activeStreamIndex_];

  if (seekMode_ == SeekMode::approximate) {
    TORCH_CHECK(
        streamMetadata.averageFpsFromHeader.has_value(),
        "Seek mode is approximate, but stream ",
        std::to_string(activeStreamIndex_),
        " does not have an average fps in its metadata.");
  }

  auto& streamInfo = streamInfos_[activeStreamIndex_];
  streamInfo.videoStreamOptions = videoStreamOptions;

  if (seekMode_ == SeekMode::custom_frame_mappings) {
    TORCH_CHECK(
        customFrameMappings.has_value(),
        "Missing frame mappings when custom_frame_mappings seek mode is set.");
    readCustomFrameMappingsUpdateMetadataAndIndex(
        activeStreamIndex_, customFrameMappings.value());
  }

  metadataDims_ =
      FrameDims(streamMetadata.height.value(), streamMetadata.width.value());
  FrameDims currInputDims = metadataDims_;
  for (auto& transform : transforms) {
    TORCH_CHECK(transform != nullptr, "Transforms should never be nullptr!");
    if (transform->getOutputFrameDims().has_value()) {
      resizedOutputDims_ = transform->getOutputFrameDims().value();
    }
    transform->validate(currInputDims);
    currInputDims = resizedOutputDims_.value_or(metadataDims_);

    // Note that we are claiming ownership of the transform objects passed in to
    // us.
    transforms_.push_back(std::unique_ptr<Transform>(transform));
  }

  deviceInterface_->initializeVideo(
      videoStreamOptions, transforms_, resizedOutputDims_);
}

void SingleStreamDecoder::addAudioStream(
    int streamIndex,
    const AudioStreamOptions& audioStreamOptions) {
  TORCH_CHECK(
      seekMode_ == SeekMode::approximate,
      "seek_mode must be 'approximate' for audio streams.");
  if (audioStreamOptions.numChannels.has_value()) {
    TORCH_CHECK(
        *audioStreamOptions.numChannels > 0 &&
            *audioStreamOptions.numChannels <= AV_NUM_DATA_POINTERS,
        "num_channels must be > 0 and <= AV_NUM_DATA_POINTERS (usually 8). Got: ",
        *audioStreamOptions.numChannels);
  }

  addStream(streamIndex, AVMEDIA_TYPE_AUDIO);

  auto& streamInfo = streamInfos_[activeStreamIndex_];
  streamInfo.audioStreamOptions = audioStreamOptions;

  // FFmpeg docs say that the decoder will try to decode natively in this
  // format, if it can. Docs don't say what the decoder does when it doesn't
  // support that format, but it looks like it does nothing, so this probably
  // doesn't hurt.
  streamInfo.codecContext->request_sample_fmt = AV_SAMPLE_FMT_FLTP;

  // Initialize device interface for audio
  deviceInterface_->initializeAudio(audioStreamOptions);
}

// --------------------------------------------------------------------------
// HIGH-LEVEL DECODING ENTRY-POINTS
// --------------------------------------------------------------------------

FrameOutput SingleStreamDecoder::getNextFrame() {
  auto output = getNextFrameInternal();
  if (streamInfos_[activeStreamIndex_].avMediaType == AVMEDIA_TYPE_VIDEO) {
    output.data = maybePermuteHWC2CHW(output.data);
  }
  return output;
}

FrameOutput SingleStreamDecoder::getNextFrameInternal(
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  validateActiveStream();
  UniqueAVFrame avFrame = decodeAVFrame([this](const UniqueAVFrame& avFrame) {
    return getPtsOrDts(avFrame) >= cursor_;
  });
  return convertAVFrameToFrameOutput(avFrame, preAllocatedOutputTensor);
}

FrameOutput SingleStreamDecoder::getFrameAtIndex(int64_t frameIndex) {
  auto frameOutput = getFrameAtIndexInternal(frameIndex);
  frameOutput.data = maybePermuteHWC2CHW(frameOutput.data);
  return frameOutput;
}

FrameOutput SingleStreamDecoder::getFrameAtIndexInternal(
    int64_t frameIndex,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  validateActiveStream(AVMEDIA_TYPE_VIDEO);

  const auto& streamInfo = streamInfos_[activeStreamIndex_];
  const auto& streamMetadata =
      containerMetadata_.allStreamMetadata[activeStreamIndex_];

  std::optional<int64_t> numFrames = streamMetadata.getNumFrames(seekMode_);
  if (numFrames.has_value()) {
    // If the frameIndex is negative, we convert it to a positive index
    frameIndex = frameIndex >= 0 ? frameIndex : frameIndex + numFrames.value();
  }
  validateFrameIndex(streamMetadata, frameIndex);

  // Only set cursor if we're not decoding sequentially: when decoding
  // sequentially, we don't need to seek anywhere, so by *not* setting the
  // cursor we allow canWeAvoidSeeking() to return true early.
  if (frameIndex != lastDecodedFrameIndex_ + 1) {
    int64_t pts = getPts(frameIndex);
    setCursorPtsInSeconds(ptsToSeconds(pts, streamInfo.timeBase));
  }

  auto result = getNextFrameInternal(preAllocatedOutputTensor);
  lastDecodedFrameIndex_ = frameIndex;
  return result;
}

FrameBatchOutput SingleStreamDecoder::getFramesAtIndices(
    const torch::Tensor& frameIndices) {
  validateActiveStream(AVMEDIA_TYPE_VIDEO);

  auto frameIndicesAccessor = frameIndices.accessor<int64_t, 1>();

  bool indicesAreSorted = true;
  for (int64_t i = 1; i < frameIndices.numel(); ++i) {
    if (frameIndicesAccessor[i] < frameIndicesAccessor[i - 1]) {
      indicesAreSorted = false;
      break;
    }
  }

  std::vector<size_t> argsort;
  if (!indicesAreSorted) {
    // if frameIndices is [13, 10, 12, 11]
    // when sorted, it's  [10, 11, 12, 13] <-- this is the sorted order we want
    //                                         to use to decode the frames
    // and argsort is     [ 1,  3,  2,  0]
    argsort.resize(frameIndices.numel());
    for (size_t i = 0; i < argsort.size(); ++i) {
      argsort[i] = i;
    }
    std::sort(
        argsort.begin(),
        argsort.end(),
        [&frameIndicesAccessor](size_t a, size_t b) {
          return frameIndicesAccessor[a] < frameIndicesAccessor[b];
        });
  }

  const auto& streamInfo = streamInfos_[activeStreamIndex_];
  const auto& videoStreamOptions = streamInfo.videoStreamOptions;
  FrameBatchOutput frameBatchOutput(
      frameIndices.numel(),
      resizedOutputDims_.value_or(metadataDims_),
      videoStreamOptions.device);

  auto previousIndexInVideo = -1;
  for (int64_t f = 0; f < frameIndices.numel(); ++f) {
    auto indexInOutput = indicesAreSorted ? f : argsort[f];
    auto indexInVideo = frameIndicesAccessor[indexInOutput];

    if ((f > 0) && (indexInVideo == previousIndexInVideo)) {
      // Avoid decoding the same frame twice
      auto previousIndexInOutput = indicesAreSorted ? f - 1 : argsort[f - 1];
      frameBatchOutput.data[indexInOutput].copy_(
          frameBatchOutput.data[previousIndexInOutput]);
      frameBatchOutput.ptsSeconds[indexInOutput] =
          frameBatchOutput.ptsSeconds[previousIndexInOutput];
      frameBatchOutput.durationSeconds[indexInOutput] =
          frameBatchOutput.durationSeconds[previousIndexInOutput];
    } else {
      FrameOutput frameOutput = getFrameAtIndexInternal(
          indexInVideo, frameBatchOutput.data[indexInOutput]);
      frameBatchOutput.ptsSeconds[indexInOutput] = frameOutput.ptsSeconds;
      frameBatchOutput.durationSeconds[indexInOutput] =
          frameOutput.durationSeconds;
    }
    previousIndexInVideo = indexInVideo;
  }
  frameBatchOutput.data = maybePermuteHWC2CHW(frameBatchOutput.data);
  return frameBatchOutput;
}

FrameBatchOutput SingleStreamDecoder::getFramesInRange(
    int64_t start,
    int64_t stop,
    int64_t step) {
  validateActiveStream(AVMEDIA_TYPE_VIDEO);

  const auto& streamMetadata =
      containerMetadata_.allStreamMetadata[activeStreamIndex_];
  const auto& streamInfo = streamInfos_[activeStreamIndex_];
  TORCH_CHECK(
      start >= 0, "Range start, " + std::to_string(start) + " is less than 0.");
  TORCH_CHECK(
      step > 0, "Step must be greater than 0; is " + std::to_string(step));

  // Note that if we do not have the number of frames available in our
  // metadata, then we assume that the upper part of the range is valid.
  std::optional<int64_t> numFrames = streamMetadata.getNumFrames(seekMode_);
  if (numFrames.has_value()) {
    TORCH_CHECK(
        stop <= numFrames.value(),
        "Range stop, " + std::to_string(stop) +
            ", is more than the number of frames, " +
            std::to_string(numFrames.value()));
  }

  int64_t numOutputFrames = std::ceil((stop - start) / double(step));
  const auto& videoStreamOptions = streamInfo.videoStreamOptions;
  FrameBatchOutput frameBatchOutput(
      numOutputFrames,
      resizedOutputDims_.value_or(metadataDims_),
      videoStreamOptions.device);

  for (int64_t i = start, f = 0; i < stop; i += step, ++f) {
    FrameOutput frameOutput =
        getFrameAtIndexInternal(i, frameBatchOutput.data[f]);
    frameBatchOutput.ptsSeconds[f] = frameOutput.ptsSeconds;
    frameBatchOutput.durationSeconds[f] = frameOutput.durationSeconds;
  }
  frameBatchOutput.data = maybePermuteHWC2CHW(frameBatchOutput.data);
  return frameBatchOutput;
}

FrameOutput SingleStreamDecoder::getFramePlayedAt(double seconds) {
  validateActiveStream(AVMEDIA_TYPE_VIDEO);
  StreamInfo& streamInfo = streamInfos_[activeStreamIndex_];
  double lastDecodedStartTime =
      ptsToSeconds(lastDecodedAvFramePts_, streamInfo.timeBase);
  double lastDecodedEndTime = ptsToSeconds(
      lastDecodedAvFramePts_ + lastDecodedAvFrameDuration_,
      streamInfo.timeBase);
  if (seconds >= lastDecodedStartTime && seconds < lastDecodedEndTime) {
    // We are in the same frame as the one we just returned. However, since we
    // don't cache it locally, we have to rewind back.
    seconds = lastDecodedStartTime;
  }

  setCursorPtsInSeconds(seconds);
  UniqueAVFrame avFrame =
      decodeAVFrame([seconds, this](const UniqueAVFrame& avFrame) {
        StreamInfo& streamInfo = streamInfos_[activeStreamIndex_];
        double frameStartTime =
            ptsToSeconds(getPtsOrDts(avFrame), streamInfo.timeBase);
        double frameEndTime = ptsToSeconds(
            getPtsOrDts(avFrame) + getDuration(avFrame), streamInfo.timeBase);
        if (frameStartTime > seconds) {
          // FFMPEG seeked past the frame we are looking for even though we
          // set max_ts to be our needed timestamp in avformat_seek_file()
          // in maybeSeekToBeforeDesiredPts().
          // This could be a bug in FFMPEG:
          // https://trac.ffmpeg.org/ticket/11137 In this case we return the
          // very next frame instead of throwing an exception.
          // TODO: Maybe log to stderr for Debug builds?
          return true;
        }
        return seconds >= frameStartTime && seconds < frameEndTime;
      });

  // Convert the frame to tensor.
  FrameOutput frameOutput = convertAVFrameToFrameOutput(avFrame);
  frameOutput.data = maybePermuteHWC2CHW(frameOutput.data);
  return frameOutput;
}

FrameBatchOutput SingleStreamDecoder::getFramesPlayedAt(
    const torch::Tensor& timestamps) {
  validateActiveStream(AVMEDIA_TYPE_VIDEO);

  const auto& streamMetadata =
      containerMetadata_.allStreamMetadata[activeStreamIndex_];

  double minSeconds = streamMetadata.getBeginStreamSeconds(seekMode_);
  std::optional<double> maxSeconds =
      streamMetadata.getEndStreamSeconds(seekMode_);

  // The frame played at timestamp t and the one played at timestamp `t +
  // eps` are probably the same frame, with the same index. The easiest way to
  // avoid decoding that unique frame twice is to convert the input timestamps
  // to indices, and leverage the de-duplication logic of getFramesAtIndices.

  torch::Tensor frameIndices =
      torch::empty({timestamps.numel()}, torch::kInt64);
  auto frameIndicesAccessor = frameIndices.accessor<int64_t, 1>();
  auto timestampsAccessor = timestamps.accessor<double, 1>();

  for (int64_t i = 0; i < timestamps.numel(); ++i) {
    auto frameSeconds = timestampsAccessor[i];
    TORCH_CHECK(
        frameSeconds >= minSeconds,
        "frame pts is " + std::to_string(frameSeconds) +
            "; must be greater than or equal to " + std::to_string(minSeconds) +
            ".");

    // Note that if we can't determine the maximum number of seconds from the
    // metadata, then we assume the frame's pts is valid.
    if (maxSeconds.has_value()) {
      TORCH_CHECK(
          frameSeconds < maxSeconds.value(),
          "frame pts is " + std::to_string(frameSeconds) +
              "; must be less than " + std::to_string(maxSeconds.value()) +
              ".");
    }

    frameIndicesAccessor[i] = secondsToIndexLowerBound(frameSeconds);
  }

  return getFramesAtIndices(frameIndices);
}

FrameBatchOutput SingleStreamDecoder::getFramesPlayedInRange(
    double startSeconds,
    double stopSeconds) {
  validateActiveStream(AVMEDIA_TYPE_VIDEO);
  const auto& streamMetadata =
      containerMetadata_.allStreamMetadata[activeStreamIndex_];
  TORCH_CHECK(
      startSeconds <= stopSeconds,
      "Start seconds (" + std::to_string(startSeconds) +
          ") must be less than or equal to stop seconds (" +
          std::to_string(stopSeconds) + ".");

  const auto& streamInfo = streamInfos_[activeStreamIndex_];
  const auto& videoStreamOptions = streamInfo.videoStreamOptions;

  // Special case needed to implement a half-open range. At first glance, this
  // may seem unnecessary, as our search for stopFrame can return the end, and
  // we don't include stopFramIndex in our output. However, consider the
  // following scenario:
  //
  //   frame=0, pts=0.0
  //   frame=1, pts=0.3
  //
  //   interval A: [0.2, 0.2)
  //   interval B: [0.2, 0.15)
  //
  // Both intervals take place between the pts values for frame 0 and frame 1,
  // which by our abstract player, means that both intervals map to frame 0.
  // By the definition of a half open interval, interval A should return no
  // frames. Interval B should return frame 0. However, for both A and B, the
  // individual values of the intervals will map to the same frame indices
  // below. Hence, we need this special case below.
  if (startSeconds == stopSeconds) {
    FrameBatchOutput frameBatchOutput(
        0,
        resizedOutputDims_.value_or(metadataDims_),
        videoStreamOptions.device);
    frameBatchOutput.data = maybePermuteHWC2CHW(frameBatchOutput.data);
    return frameBatchOutput;
  }

  double minSeconds = streamMetadata.getBeginStreamSeconds(seekMode_);
  TORCH_CHECK(
      startSeconds >= minSeconds,
      "Start seconds is " + std::to_string(startSeconds) +
          "; must be greater than or equal to " + std::to_string(minSeconds) +
          ".");

  // Note that if we can't determine the maximum seconds from the metadata,
  // then we assume upper range is valid.
  std::optional<double> maxSeconds =
      streamMetadata.getEndStreamSeconds(seekMode_);
  if (maxSeconds.has_value()) {
    TORCH_CHECK(
        startSeconds < maxSeconds.value(),
        "Start seconds is " + std::to_string(startSeconds) +
            "; must be less than " + std::to_string(maxSeconds.value()) + ".");
    TORCH_CHECK(
        stopSeconds <= maxSeconds.value(),
        "Stop seconds (" + std::to_string(stopSeconds) +
            "; must be less than or equal to " +
            std::to_string(maxSeconds.value()) + ").");
  }

  // Note that we look at nextPts for a frame, and not its pts or duration.
  // Our abstract player displays frames starting at the pts for that frame
  // until the pts for the next frame. There are two consequences:
  //
  //   1. We ignore the duration for a frame. A frame is played until the
  //   next frame replaces it. This model is robust to durations being 0 or
  //   incorrect; our source of truth is the pts for frames. If duration is
  //   accurate, the nextPts for a frame would be equivalent to pts +
  //   duration.
  //   2. In order to establish if the start of an interval maps to a
  //   particular frame, we need to figure out if it is ordered after the
  //   frame's pts, but before the next frames's pts.

  int64_t startFrameIndex = secondsToIndexLowerBound(startSeconds);
  int64_t stopFrameIndex = secondsToIndexUpperBound(stopSeconds);
  int64_t numFrames = stopFrameIndex - startFrameIndex;

  FrameBatchOutput frameBatchOutput(
      numFrames,
      resizedOutputDims_.value_or(metadataDims_),
      videoStreamOptions.device);
  for (int64_t i = startFrameIndex, f = 0; i < stopFrameIndex; ++i, ++f) {
    FrameOutput frameOutput =
        getFrameAtIndexInternal(i, frameBatchOutput.data[f]);
    frameBatchOutput.ptsSeconds[f] = frameOutput.ptsSeconds;
    frameBatchOutput.durationSeconds[f] = frameOutput.durationSeconds;
  }
  frameBatchOutput.data = maybePermuteHWC2CHW(frameBatchOutput.data);

  return frameBatchOutput;
}

// Note [Audio Decoding Design]
// This note explains why audio decoding is implemented the way it is, and why
// it inherently differs from video decoding.
//
// Like for video, FFmpeg exposes the concept of a frame for audio streams. An
// audio frame is a contiguous sequence of samples, where a sample consists of
// `numChannels` values. An audio frame, or a sequence thereof, is always
// converted into a tensor of shape `(numChannels, numSamplesPerChannel)`.
//
// The notion of 'frame' in audio isn't what users want to interact with.
// Users want to interact with samples. The C++ and core APIs return frames,
// because we want those to be close to FFmpeg concepts, but the higher-level
// public APIs expose samples. As a result:
// - We don't expose index-based APIs for audio, because that would mean
//   exposing the concept of audio frame. For now, we think exposing
//   time-based APIs is more natural.
// - We never perform a scan for audio streams. We don't need to, since we
// won't
//   be converting timestamps to indices. That's why we enforce the seek_mode
//   to be "approximate" (which is slightly misleading, because technically
//   the output samples will be at their exact positions. But this
//   incongruence is only exposed at the C++/core private levels).
//
// Audio frames are of variable dimensions: in the same stream, a frame can
// contain 1024 samples and the next one may contain 512 [1]. This makes it
// impossible to stack audio frames in the same way we can stack video frames.
// This is one of the main reasons we cannot reuse the same pre-allocation
// logic we have for videos in getFramesPlayedInRange(): pre-allocating a
// batch requires constant (and known) frame dimensions. That's also why
// *concatenated* along the samples dimension, not stacked.
//
// [IMPORTANT!] There is one key invariant that we must respect when decoding
// audio frames:
//
// BEFORE DECODING FRAME i, WE MUST DECODE ALL FRAMES j < i.
//
// Always. Why? We don't know. What we know is that if we don't, we get
// clipped, incorrect audio as output [2]. All other (correct) libraries like
// TorchAudio or Decord do something similar, whether it was intended or not.
// This has a few implications:
// - The **only** place we're allowed to seek to in an audio stream is the
//   stream's beginning. This ensures that if we need a frame, we'll have
//   decoded all previous frames.
// - Because of that, we don't allow the public APIs to seek. Public APIs can
//   call next() and `getFramesPlayedInRangeAudio()`, but they cannot manually
//   seek.
// - We try not to seek, when we can avoid it. Typically if the next frame we
//   need is in the future, we don't seek back to the beginning, we just
//   decode all the frames in-between.
//
// [2] If you're brave and curious, you can read the long "Seek offset for
// audio" note in https://github.com/pytorch/torchcodec/pull/507/files, which
// sums up past (and failed) attemps at working around this issue.
AudioFramesOutput SingleStreamDecoder::getFramesPlayedInRangeAudio(
    double startSeconds,
    std::optional<double> stopSecondsOptional) {
  validateActiveStream(AVMEDIA_TYPE_AUDIO);

  if (stopSecondsOptional.has_value()) {
    TORCH_CHECK(
        startSeconds <= *stopSecondsOptional,
        "Start seconds (" + std::to_string(startSeconds) +
            ") must be less than or equal to stop seconds (" +
            std::to_string(*stopSecondsOptional) + ").");
  }

  StreamInfo& streamInfo = streamInfos_[activeStreamIndex_];

  if (stopSecondsOptional.has_value() && startSeconds == *stopSecondsOptional) {
    // For consistency with video
    int numChannels = getNumChannels(streamInfo.codecContext);
    return AudioFramesOutput{torch::empty({numChannels, 0}), 0.0};
  }

  auto startPts = secondsToClosestPts(startSeconds, streamInfo.timeBase);
  if (startPts < lastDecodedAvFramePts_ + lastDecodedAvFrameDuration_) {
    // If we need to seek backwards, then we have to seek back to the
    // beginning of the stream. See [Audio Decoding Design].
    setCursor(INT64_MIN);
  }

  // TODO-AUDIO Pre-allocate a long-enough tensor instead of creating a vec +
  // cat(). This would save a copy. We know the duration of the output and the
  // sample rate, so in theory we know the number of output samples.
  std::vector<torch::Tensor> frames;

  std::optional<double> firstFramePtsSeconds = std::nullopt;
  auto stopPts = stopSecondsOptional.has_value()
      ? secondsToClosestPts(*stopSecondsOptional, streamInfo.timeBase)
      : INT64_MAX;
  auto finished = false;
  while (!finished) {
    try {
      UniqueAVFrame avFrame =
          decodeAVFrame([startPts, stopPts](const UniqueAVFrame& avFrame) {
            return startPts < getPtsOrDts(avFrame) + getDuration(avFrame) &&
                stopPts > getPtsOrDts(avFrame);
          });
      auto frameOutput = convertAVFrameToFrameOutput(avFrame);
      if (!firstFramePtsSeconds.has_value()) {
        firstFramePtsSeconds = frameOutput.ptsSeconds;
      }
      frames.push_back(frameOutput.data);
    } catch (const EndOfFileException&) {
      finished = true;
    }

    // If stopSeconds is in [begin, end] of the last decoded frame, we should
    // stop decoding more frames. Note that if we were to use [begin, end),
    // which may seem more natural, then we would decode the frame starting at
    // stopSeconds, which isn't what we want!
    auto lastDecodedAvFrameEnd =
        lastDecodedAvFramePts_ + lastDecodedAvFrameDuration_;
    finished |= (lastDecodedAvFramePts_) <= stopPts &&
        (stopPts <= lastDecodedAvFrameEnd);
  }

  auto lastSamples = deviceInterface_->maybeFlushAudioBuffers();
  if (lastSamples.has_value()) {
    frames.push_back(*lastSamples);
  }

  TORCH_CHECK(
      frames.size() > 0 && firstFramePtsSeconds.has_value(),
      "No audio frames were decoded. ",
      "This is probably because start_seconds is too high(",
      startSeconds,
      "),",
      "or because stop_seconds(",
      stopSecondsOptional,
      ") is too low.");

  return AudioFramesOutput{torch::cat(frames, 1), *firstFramePtsSeconds};
}

// --------------------------------------------------------------------------
// SEEKING APIs
// --------------------------------------------------------------------------

void SingleStreamDecoder::setCursorPtsInSeconds(double seconds) {
  // We don't allow public audio decoding APIs to seek, see [Audio Decoding
  // Design]
  validateActiveStream(AVMEDIA_TYPE_VIDEO);
  setCursor(
      secondsToClosestPts(seconds, streamInfos_[activeStreamIndex_].timeBase));
}

void SingleStreamDecoder::setCursor(int64_t pts) {
  cursorWasJustSet_ = true;
  cursor_ = pts;
}

bool SingleStreamDecoder::canWeAvoidSeeking() const {
  // Returns true if we can avoid seeking in the AVFormatContext based on
  // heuristics that rely on the target cursor_ and the last decoded frame.
  // Seeking is expensive, so we try to avoid it when possible.
  const StreamInfo& streamInfo = streamInfos_.at(activeStreamIndex_);
  if (streamInfo.avMediaType == AVMEDIA_TYPE_AUDIO) {
    // For audio, we only need to seek if a backwards seek was requested
    // within getFramesPlayedInRangeAudio(), when setCursorPtsInSeconds() was
    // called. For more context, see [Audio Decoding Design]
    return !cursorWasJustSet_;
  } else if (!cursorWasJustSet_) {
    // For videos, when decoding consecutive frames, we don't need to seek.
    return true;
  }

  if (cursor_ < lastDecodedAvFramePts_) {
    // We can never skip a seek if we are seeking backwards.
    return false;
  }
  if (lastDecodedAvFramePts_ == cursor_) {
    // We are seeking to the exact same frame as we are currently at. Without
    // caching we have to rewind back and decode the frame again.
    // TODO: https://github.com/pytorch/torchcodec/issues/84 we could
    // implement caching.
    return false;
  }
  // We are seeking forwards. We can skip a seek if both the last decoded frame
  // and cursor_ share the same keyframe:
  // Videos have I frames and non-I frames (P and B frames). Non-I frames need
  // data from the previous I frame to be decoded.
  //
  // Imagine the cursor is at a random frame with PTS=lastDecodedAvFramePts (x
  // for brevity) and we wish to seek to a user-specified PTS=y.
  //
  // If y < x, we don't have a choice but to seek backwards to the highest I
  // frame before y.
  //
  // If y > x, we have two choices:
  //
  // 1. We could keep decoding forward until we hit y. Illustrated below:
  //
  // I    P     P    P    I    P    P    P    I    P    P    I    P
  //                           x         y
  //
  // 2. We could try to jump to an I frame between x and y (indicated by j
  // below). And then start decoding until we encounter y. Illustrated below:
  //
  // I    P     P    P    I    P    P    P    I    P    P    I    P
  //                           x              j         y
  // (2) is only more efficient than (1) if there is an I frame between x and y.
  int lastKeyFrame = getKeyFrameIdentifier(lastDecodedAvFramePts_);
  int targetKeyFrame = getKeyFrameIdentifier(cursor_);
  return lastKeyFrame >= 0 && targetKeyFrame >= 0 &&
      lastKeyFrame == targetKeyFrame;
}

// This method looks at currentPts and desiredPts and seeks in the
// AVFormatContext if it is needed. We can skip seeking in certain cases. See
// the comment of canWeAvoidSeeking() for details.
void SingleStreamDecoder::maybeSeekToBeforeDesiredPts() {
  validateActiveStream();
  StreamInfo& streamInfo = streamInfos_[activeStreamIndex_];

  decodeStats_.numSeeksAttempted++;
  if (canWeAvoidSeeking()) {
    decodeStats_.numSeeksSkipped++;
    return;
  }

  int64_t desiredPts = cursor_;

  // For some encodings like H265, FFMPEG sometimes seeks past the point we
  // set as the max_ts. So we use our own index to give it the exact pts of
  // the key frame that we want to seek to.
  // See https://github.com/pytorch/torchcodec/issues/179 for more details.
  // See https://trac.ffmpeg.org/ticket/11137 for the underlying ffmpeg bug.
  if (!streamInfo.keyFrames.empty()) {
    int desiredKeyFrameIndex = getKeyFrameIndexForPtsUsingScannedIndex(
        streamInfo.keyFrames, desiredPts);
    desiredKeyFrameIndex = std::max(desiredKeyFrameIndex, 0);
    desiredPts = streamInfo.keyFrames[desiredKeyFrameIndex].pts;
  }

  int status = avformat_seek_file(
      formatContext_.get(),
      streamInfo.streamIndex,
      INT64_MIN,
      desiredPts,
      desiredPts,
      0);
  TORCH_CHECK(
      status >= 0,
      "Could not seek file to pts=",
      std::to_string(desiredPts),
      ": ",
      getFFMPEGErrorStringFromErrorCode(status));

  decodeStats_.numFlushes++;
  deviceInterface_->flush();
}

// --------------------------------------------------------------------------
// LOW-LEVEL DECODING
// --------------------------------------------------------------------------

UniqueAVFrame SingleStreamDecoder::decodeAVFrame(
    std::function<bool(const UniqueAVFrame&)> filterFunction) {
  validateActiveStream();

  resetDecodeStats();

  maybeSeekToBeforeDesiredPts();
  cursorWasJustSet_ = false;

  UniqueAVFrame avFrame(av_frame_alloc());
  AutoAVPacket autoAVPacket;
  int status = AVSUCCESS;
  bool reachedEOF = false;

  // The default implementation uses avcodec_receive_frame and
  // avcodec_send_packet, while specialized interfaces can override for
  // hardware-specific optimizations.
  while (true) {
    status = deviceInterface_->receiveFrame(avFrame);

    if (status != AVSUCCESS && status != AVERROR(EAGAIN)) {
      // Non-retriable error
      break;
    }

    decodeStats_.numFramesReceivedByDecoder++;
    // Is this the kind of frame we're looking for?
    if (status == AVSUCCESS && filterFunction(avFrame)) {
      // Yes, this is the frame we'll return; break out of the decoding loop.
      break;
    } else if (status == AVSUCCESS) {
      // No, but we received a valid frame - just not the kind we're looking
      // for. The logic below will read packets and send them to the decoder.
      // But since we did just receive a frame, we should skip reading more
      // packets and sending them to the decoder and just try to receive more
      // frames from the decoder.
      continue;
    }

    if (reachedEOF) {
      // We don't have any more packets to receive. So keep on pulling frames
      // from decoder's internal buffers.
      continue;
    }

    // We still haven't found the frame we're looking for. So let's read more
    // packets and send them to the decoder.
    ReferenceAVPacket packet(autoAVPacket);
    do {
      status = av_read_frame(formatContext_.get(), packet.get());
      decodeStats_.numPacketsRead++;

      if (status == AVERROR_EOF) {
        // End of file reached. We must drain the decoder
        status = deviceInterface_->sendEOFPacket();
        TORCH_CHECK(
            status >= AVSUCCESS,
            "Could not flush decoder: ",
            getFFMPEGErrorStringFromErrorCode(status));

        reachedEOF = true;
        break;
      }

      TORCH_CHECK(
          status >= AVSUCCESS,
          "Could not read frame from input file: ",
          getFFMPEGErrorStringFromErrorCode(status));

    } while (packet->stream_index != activeStreamIndex_);

    if (reachedEOF) {
      // We don't have any more packets to send to the decoder. So keep on
      // pulling frames from its internal buffers.
      continue;
    }

    printf("packet pts = %ld, discard flag = %d\n", packet->pts, packet->flags & AV_PKT_FLAG_DISCARD);

    // We got a valid packet. Send it to the decoder, and we'll receive it in
    // the next iteration.
    status = deviceInterface_->sendPacket(packet);
    TORCH_CHECK(
        status >= AVSUCCESS,
        "Could not push packet to decoder: ",
        getFFMPEGErrorStringFromErrorCode(status));

    decodeStats_.numPacketsSentToDecoder++;
  }

  if (status < AVSUCCESS) {
    if (reachedEOF || status == AVERROR_EOF) {
      throw SingleStreamDecoder::EndOfFileException(
          "Requested next frame while there are no more frames left to "
          "decode.");
    }
    TORCH_CHECK(
        false,
        "Could not receive frame from decoder: ",
        getFFMPEGErrorStringFromErrorCode(status));
  }

  // Note that we don't flush the decoder when we reach EOF (even though
  // that's mentioned in
  // https://ffmpeg.org/doxygen/trunk/group__lavc__encdec.html). This is
  // because we may have packets internally in the decoder that we haven't
  // received as frames. Eventually we will either hit AVERROR_EOF from
  // av_receive_frame() or the user will have seeked to a different location
  // in the file and that will flush the decoder.
  lastDecodedAvFramePts_ = getPtsOrDts(avFrame);
  lastDecodedAvFrameDuration_ = getDuration(avFrame);

  return avFrame;
}

// --------------------------------------------------------------------------
// AVFRAME <-> FRAME OUTPUT CONVERSION
// --------------------------------------------------------------------------

FrameOutput SingleStreamDecoder::convertAVFrameToFrameOutput(
    UniqueAVFrame& avFrame,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  // Convert the frame to tensor.
  FrameOutput frameOutput;
  frameOutput.ptsSeconds = ptsToSeconds(
      getPtsOrDts(avFrame),
      formatContext_->streams[activeStreamIndex_]->time_base);
  frameOutput.durationSeconds = ptsToSeconds(
      getDuration(avFrame),
      formatContext_->streams[activeStreamIndex_]->time_base);
  deviceInterface_->convertAVFrameToFrameOutput(
      avFrame, frameOutput, std::move(preAllocatedOutputTensor));
  return frameOutput;
}

// --------------------------------------------------------------------------
// OUTPUT ALLOCATION AND SHAPE CONVERSION
// --------------------------------------------------------------------------

// Returns a [N]CHW *view* of a [N]HWC input tensor, if the options require
// so. The [N] leading batch-dimension is optional i.e. the input tensor can
// be 3D or 4D. Calling permute() is guaranteed to return a view as per the
// docs: https://pytorch.org/docs/stable/generated/torch.permute.html
torch::Tensor SingleStreamDecoder::maybePermuteHWC2CHW(
    torch::Tensor& hwcTensor) {
  if (streamInfos_[activeStreamIndex_].videoStreamOptions.dimensionOrder ==
      "NHWC") {
    return hwcTensor;
  }
  auto numDimensions = hwcTensor.dim();
  auto shape = hwcTensor.sizes();
  if (numDimensions == 3) {
    TORCH_CHECK(shape[2] == 3, "Not a HWC tensor: ", shape);
    return hwcTensor.permute({2, 0, 1});
  } else if (numDimensions == 4) {
    TORCH_CHECK(shape[3] == 3, "Not a NHWC tensor: ", shape);
    return hwcTensor.permute({0, 3, 1, 2});
  } else {
    TORCH_CHECK(
        false, "Expected tensor with 3 or 4 dimensions, got ", numDimensions);
  }
}

// --------------------------------------------------------------------------
// PTS <-> INDEX CONVERSIONS
// --------------------------------------------------------------------------

int SingleStreamDecoder::getKeyFrameIdentifier(int64_t pts) const {
  // This function "identifies" a key frame for a given pts value.
  // We use the term "identifier" rather than "index" because the nature of the
  // index that is returned depends on various factors:
  // - If seek_mode is exact, we return the index of the key frame in the
  //   scanned key-frame vector (streamInfo.keyFrames). So the returned value is
  //   in [0, num_key_frames).
  // - If seek_mode is approximate, we use av_index_search_timestamp() which
  //   may return a value in [0, num_key_frames) like for mkv, but also a value
  //   in [0, num_frames) like for mp4. It really depends on the container.
  //
  //  The range of the "identifier" doesn't matter that much, for now we only
  //  use it to uniquely identify a key frame in canWeAvoidSeeking().
  const StreamInfo& streamInfo = streamInfos_.at(activeStreamIndex_);
  if (streamInfo.keyFrames.empty()) {
    return av_index_search_timestamp(
        streamInfo.stream, pts, AVSEEK_FLAG_BACKWARD);
  } else {
    return getKeyFrameIndexForPtsUsingScannedIndex(streamInfo.keyFrames, pts);
  }
}

int SingleStreamDecoder::getKeyFrameIndexForPtsUsingScannedIndex(
    const std::vector<SingleStreamDecoder::FrameInfo>& keyFrames,
    int64_t pts) const {
  auto upperBound = std::upper_bound(
      keyFrames.begin(),
      keyFrames.end(),
      pts,
      [](int64_t pts, const SingleStreamDecoder::FrameInfo& frameInfo) {
        return pts < frameInfo.pts;
      });
  if (upperBound == keyFrames.begin()) {
    return -1;
  }
  return upperBound - 1 - keyFrames.begin();
}

int64_t SingleStreamDecoder::secondsToIndexLowerBound(double seconds) {
  auto& streamInfo = streamInfos_[activeStreamIndex_];
  switch (seekMode_) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact: {
      auto frame = std::lower_bound(
          streamInfo.allFrames.begin(),
          streamInfo.allFrames.end(),
          seconds,
          [&streamInfo](const FrameInfo& info, double start) {
            return ptsToSeconds(info.nextPts, streamInfo.timeBase) <= start;
          });

      return frame - streamInfo.allFrames.begin();
    }
    case SeekMode::approximate: {
      auto& streamMetadata =
          containerMetadata_.allStreamMetadata[activeStreamIndex_];
      TORCH_CHECK(
          streamMetadata.averageFpsFromHeader.has_value(),
          "Cannot use approximate mode since we couldn't find the average fps from the metadata.");
      return std::floor(seconds * streamMetadata.averageFpsFromHeader.value());
    }
    default:
      TORCH_CHECK(false, "Unknown SeekMode");
  }
}

int64_t SingleStreamDecoder::secondsToIndexUpperBound(double seconds) {
  auto& streamInfo = streamInfos_[activeStreamIndex_];
  switch (seekMode_) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact: {
      auto frame = std::upper_bound(
          streamInfo.allFrames.begin(),
          streamInfo.allFrames.end(),
          seconds,
          [&streamInfo](double stop, const FrameInfo& info) {
            return stop <= ptsToSeconds(info.pts, streamInfo.timeBase);
          });

      return frame - streamInfo.allFrames.begin();
    }
    case SeekMode::approximate: {
      auto& streamMetadata =
          containerMetadata_.allStreamMetadata[activeStreamIndex_];
      TORCH_CHECK(
          streamMetadata.averageFpsFromHeader.has_value(),
          "Cannot use approximate mode since we couldn't find the average fps from the metadata.");
      return std::ceil(seconds * streamMetadata.averageFpsFromHeader.value());
    }
    default:
      TORCH_CHECK(false, "Unknown SeekMode");
  }
}

int64_t SingleStreamDecoder::getPts(int64_t frameIndex) {
  auto& streamInfo = streamInfos_[activeStreamIndex_];
  switch (seekMode_) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact:
      return streamInfo.allFrames[frameIndex].pts;
    case SeekMode::approximate: {
      auto& streamMetadata =
          containerMetadata_.allStreamMetadata[activeStreamIndex_];
      TORCH_CHECK(
          streamMetadata.averageFpsFromHeader.has_value(),
          "Cannot use approximate mode since we couldn't find the average fps from the metadata.");
      return secondsToClosestPts(
          frameIndex / streamMetadata.averageFpsFromHeader.value(),
          streamInfo.timeBase);
    }
    default:
      TORCH_CHECK(false, "Unknown SeekMode");
  }
}

// --------------------------------------------------------------------------
// STREAM AND METADATA APIS
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// VALIDATION UTILS
// --------------------------------------------------------------------------

void SingleStreamDecoder::validateActiveStream(
    std::optional<AVMediaType> avMediaType) {
  auto errorMsg =
      "Provided stream index=" + std::to_string(activeStreamIndex_) +
      " was not previously added.";
  TORCH_CHECK(activeStreamIndex_ != NO_ACTIVE_STREAM, errorMsg);
  TORCH_CHECK(streamInfos_.count(activeStreamIndex_) > 0, errorMsg);

  int allStreamMetadataSize =
      static_cast<int>(containerMetadata_.allStreamMetadata.size());
  TORCH_CHECK(
      activeStreamIndex_ >= 0 && activeStreamIndex_ < allStreamMetadataSize,
      "Invalid stream index=" + std::to_string(activeStreamIndex_) +
          "; valid indices are in the range [0, " +
          std::to_string(allStreamMetadataSize) + ").");

  if (avMediaType.has_value()) {
    TORCH_CHECK(
        streamInfos_[activeStreamIndex_].avMediaType == avMediaType.value(),
        "The method you called isn't supported. ",
        "If you're seeing this error, you are probably trying to call an ",
        "unsupported method on an audio stream.");
  }
}

void SingleStreamDecoder::validateScannedAllStreams(const std::string& msg) {
  TORCH_CHECK(
      scannedAllStreams_,
      "Must scan all streams to update metadata before calling ",
      msg);
}

void SingleStreamDecoder::validateFrameIndex(
    const StreamMetadata& streamMetadata,
    int64_t frameIndex) {
  if (frameIndex < 0) {
    throw std::out_of_range(
        "Invalid frame index=" + std::to_string(frameIndex) +
        " for streamIndex=" + std::to_string(streamMetadata.streamIndex) +
        "; negative indices must have an absolute value less than the number of frames, "
        "and the number of frames must be known.");
  }

  // Note that if we do not have the number of frames available in our
  // metadata, then we assume that the frameIndex is valid.
  std::optional<int64_t> numFrames = streamMetadata.getNumFrames(seekMode_);
  if (numFrames.has_value()) {
    if (frameIndex >= numFrames.value()) {
      throw std::out_of_range(
          "Invalid frame index=" + std::to_string(frameIndex) +
          " for streamIndex=" + std::to_string(streamMetadata.streamIndex) +
          "; must be less than " + std::to_string(numFrames.value()));
    }
  }
}

// --------------------------------------------------------------------------
// MORALLY PRIVATE UTILS
// --------------------------------------------------------------------------

SingleStreamDecoder::DecodeStats SingleStreamDecoder::getDecodeStats() const {
  return decodeStats_;
}

std::ostream& operator<<(
    std::ostream& os,
    const SingleStreamDecoder::DecodeStats& stats) {
  os << "DecodeStats{"
     << "numFramesReceivedByDecoder=" << stats.numFramesReceivedByDecoder
     << ", numPacketsRead=" << stats.numPacketsRead
     << ", numPacketsSentToDecoder=" << stats.numPacketsSentToDecoder
     << ", numSeeksAttempted=" << stats.numSeeksAttempted
     << ", numSeeksSkipped=" << stats.numSeeksSkipped
     << ", numFlushes=" << stats.numFlushes << "}";

  return os;
}

void SingleStreamDecoder::resetDecodeStats() {
  decodeStats_ = DecodeStats{};
}

double SingleStreamDecoder::getPtsSecondsForFrame(int64_t frameIndex) {
  validateActiveStream(AVMEDIA_TYPE_VIDEO);
  validateScannedAllStreams("getPtsSecondsForFrame");

  const auto& streamInfo = streamInfos_[activeStreamIndex_];
  const auto& streamMetadata =
      containerMetadata_.allStreamMetadata[activeStreamIndex_];
  validateFrameIndex(streamMetadata, frameIndex);

  return ptsToSeconds(
      streamInfo.allFrames[frameIndex].pts, streamInfo.timeBase);
}

std::string SingleStreamDecoder::getDeviceInterfaceDetails() const {
  TORCH_CHECK(deviceInterface_ != nullptr, "Device interface doesn't exist.");
  return deviceInterface_->getDetails();
}

} // namespace facebook::torchcodec
