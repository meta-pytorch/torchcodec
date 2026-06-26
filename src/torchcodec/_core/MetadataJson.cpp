// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "MetadataJson.h"

#include <array>
#include <charconv>
#include <map>
#include <sstream>

#include "TCError.h"

namespace facebook::torchcodec {

namespace {

// Shortest round-trippable representation of a double, matching fmt::to_string
// (which the torch adapter previously used). std::to_string would truncate to 6
// decimals and change the metadata's precision.
std::string toString(double value) {
  std::array<char, 64> buffer;
  auto [ptr, ec] =
      std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
  if (ec != std::errc()) {
    return std::to_string(value);
  }
  return std::string(buffer.data(), ptr);
}

std::string quoteValue(const std::string& value) {
  return "\"" + value + "\"";
}

std::string mapToJson(const std::map<std::string, std::string>& metadataMap) {
  std::stringstream ss;
  ss << "{\n";
  auto it = metadataMap.begin();
  while (it != metadataMap.end()) {
    ss << "\"" << it->first << "\": " << it->second;
    ++it;
    if (it != metadataMap.end()) {
      ss << ",\n";
    } else {
      ss << "\n";
    }
  }
  ss << "}";

  return ss.str();
}

void writeFallbackBasedMetadata(
    std::map<std::string, std::string>& map,
    const StreamMetadata& streamMetadata,
    SeekMode seekMode) {
  auto durationSeconds = streamMetadata.getDurationSeconds(seekMode);
  if (durationSeconds.has_value()) {
    map["durationSeconds"] = toString(durationSeconds.value());
  }

  auto numFrames = streamMetadata.getNumFrames(seekMode);
  if (numFrames.has_value()) {
    map["numFrames"] = std::to_string(numFrames.value());
  }

  double beginStreamSeconds = streamMetadata.getBeginStreamSeconds(seekMode);
  map["beginStreamSeconds"] = toString(beginStreamSeconds);

  auto endStreamSeconds = streamMetadata.getEndStreamSeconds(seekMode);
  if (endStreamSeconds.has_value()) {
    map["endStreamSeconds"] = toString(endStreamSeconds.value());
  }

  auto averageFps = streamMetadata.getAverageFps(seekMode);
  if (averageFps.has_value()) {
    map["averageFps"] = toString(averageFps.value());
  }
}

} // namespace

std::string getVideoJsonMetadata(SingleStreamDecoder* videoDecoder) {
  ContainerMetadata videoMetadata = videoDecoder->getContainerMetadata();
  auto maybeBestVideoStreamIndex = videoMetadata.bestVideoStreamIndex;

  std::map<std::string, std::string> metadataMap;
  // serialize the metadata into a string std::stringstream ss;
  double durationSecondsFromHeader = 0;
  if (maybeBestVideoStreamIndex.has_value() &&
      videoMetadata.allStreamMetadata[*maybeBestVideoStreamIndex]
          .durationSecondsFromHeader.has_value()) {
    durationSecondsFromHeader =
        videoMetadata.allStreamMetadata[*maybeBestVideoStreamIndex]
            .durationSecondsFromHeader.value_or(0);
  } else {
    // Fallback to container-level duration if stream duration is not found.
    durationSecondsFromHeader =
        videoMetadata.durationSecondsFromHeader.value_or(0);
  }
  metadataMap["durationSecondsFromHeader"] =
      toString(durationSecondsFromHeader);

  if (videoMetadata.bitRate.has_value()) {
    metadataMap["bitRate"] = toString(videoMetadata.bitRate.value());
  }

  if (maybeBestVideoStreamIndex.has_value()) {
    auto streamMetadata =
        videoMetadata.allStreamMetadata[*maybeBestVideoStreamIndex];
    if (streamMetadata.numFramesFromContent.has_value()) {
      metadataMap["numFramesFromHeader"] =
          std::to_string(*streamMetadata.numFramesFromContent);
    } else if (streamMetadata.numFramesFromHeader.has_value()) {
      metadataMap["numFramesFromHeader"] =
          std::to_string(*streamMetadata.numFramesFromHeader);
    }
    if (streamMetadata.beginStreamPtsSecondsFromContent.has_value()) {
      metadataMap["beginStreamSecondsFromContent"] =
          toString(*streamMetadata.beginStreamPtsSecondsFromContent);
    }
    if (streamMetadata.endStreamPtsSecondsFromContent.has_value()) {
      metadataMap["endStreamSecondsFromContent"] =
          toString(*streamMetadata.endStreamPtsSecondsFromContent);
    }
    if (streamMetadata.codecName.has_value()) {
      metadataMap["codec"] = quoteValue(streamMetadata.codecName.value());
    }
    if (streamMetadata.postRotationWidth.has_value()) {
      metadataMap["width"] = std::to_string(*streamMetadata.postRotationWidth);
    }
    if (streamMetadata.postRotationHeight.has_value()) {
      metadataMap["height"] =
          std::to_string(*streamMetadata.postRotationHeight);
    }
    if (streamMetadata.averageFpsFromHeader.has_value()) {
      metadataMap["averageFpsFromHeader"] =
          toString(*streamMetadata.averageFpsFromHeader);
    }
  }
  if (videoMetadata.bestVideoStreamIndex.has_value()) {
    metadataMap["bestVideoStreamIndex"] =
        std::to_string(*videoMetadata.bestVideoStreamIndex);
  }
  if (videoMetadata.bestAudioStreamIndex.has_value()) {
    metadataMap["bestAudioStreamIndex"] =
        std::to_string(*videoMetadata.bestAudioStreamIndex);
  }

  return mapToJson(metadataMap);
}

std::string getContainerJsonMetadata(SingleStreamDecoder* videoDecoder) {
  auto containerMetadata = videoDecoder->getContainerMetadata();

  std::map<std::string, std::string> map;

  if (containerMetadata.durationSecondsFromHeader.has_value()) {
    map["durationSecondsFromHeader"] =
        toString(*containerMetadata.durationSecondsFromHeader);
  }

  if (containerMetadata.bitRate.has_value()) {
    map["bitRate"] = toString(*containerMetadata.bitRate);
  }

  if (containerMetadata.bestVideoStreamIndex.has_value()) {
    map["bestVideoStreamIndex"] =
        std::to_string(*containerMetadata.bestVideoStreamIndex);
  }
  if (containerMetadata.bestAudioStreamIndex.has_value()) {
    map["bestAudioStreamIndex"] =
        std::to_string(*containerMetadata.bestAudioStreamIndex);
  }

  map["numStreams"] =
      std::to_string(containerMetadata.allStreamMetadata.size());

  return mapToJson(map);
}

std::string getStreamJsonMetadata(
    SingleStreamDecoder* videoDecoder,
    int64_t stream_index) {
  auto allStreamMetadata =
      videoDecoder->getContainerMetadata().allStreamMetadata;
  TC_CHECK_INDEX(
      stream_index >= 0 &&
          stream_index < static_cast<int64_t>(allStreamMetadata.size()),
      "stream_index out of bounds: " + std::to_string(stream_index));

  auto streamMetadata = allStreamMetadata[stream_index];
  auto seekMode = videoDecoder->getSeekMode();
  int activeStreamIndex = videoDecoder->getActiveStreamIndex();

  std::map<std::string, std::string> map;

  if (streamMetadata.durationSecondsFromHeader.has_value()) {
    map["durationSecondsFromHeader"] =
        toString(*streamMetadata.durationSecondsFromHeader);
  }
  if (streamMetadata.bitRate.has_value()) {
    map["bitRate"] = toString(*streamMetadata.bitRate);
  }
  if (streamMetadata.numFramesFromContent.has_value()) {
    map["numFramesFromContent"] =
        std::to_string(*streamMetadata.numFramesFromContent);
  }
  if (streamMetadata.numFramesFromHeader.has_value()) {
    map["numFramesFromHeader"] =
        std::to_string(*streamMetadata.numFramesFromHeader);
  }
  if (streamMetadata.beginStreamSecondsFromHeader.has_value()) {
    map["beginStreamSecondsFromHeader"] =
        toString(*streamMetadata.beginStreamSecondsFromHeader);
  }
  if (streamMetadata.beginStreamPtsSecondsFromContent.has_value()) {
    map["beginStreamSecondsFromContent"] =
        toString(*streamMetadata.beginStreamPtsSecondsFromContent);
  }
  if (streamMetadata.endStreamPtsSecondsFromContent.has_value()) {
    map["endStreamSecondsFromContent"] =
        toString(*streamMetadata.endStreamPtsSecondsFromContent);
  }
  if (streamMetadata.codecName.has_value()) {
    map["codec"] = quoteValue(streamMetadata.codecName.value());
  }
  if (streamMetadata.postRotationWidth.has_value()) {
    map["width"] = std::to_string(*streamMetadata.postRotationWidth);
  }
  if (streamMetadata.postRotationHeight.has_value()) {
    map["height"] = std::to_string(*streamMetadata.postRotationHeight);
  }
  if (streamMetadata.sampleAspectRatio.has_value()) {
    map["sampleAspectRatioNum"] =
        std::to_string((*streamMetadata.sampleAspectRatio).num);
    map["sampleAspectRatioDen"] =
        std::to_string((*streamMetadata.sampleAspectRatio).den);
  }
  if (streamMetadata.rotation.has_value()) {
    map["rotation"] = std::to_string(*streamMetadata.rotation);
  }
  if (auto name = streamMetadata.getColorPrimariesName()) {
    map["colorPrimaries"] = quoteValue(*name);
  }
  if (auto name = streamMetadata.getColorSpaceName()) {
    map["colorSpace"] = quoteValue(*name);
  }
  if (auto name = streamMetadata.getColorTransferCharacteristicName()) {
    map["colorTransferCharacteristic"] = quoteValue(*name);
  }
  if (streamMetadata.pixelFormat.has_value()) {
    map["pixelFormat"] = quoteValue(streamMetadata.pixelFormat.value());
  }
  if (streamMetadata.averageFpsFromHeader.has_value()) {
    map["averageFpsFromHeader"] =
        toString(*streamMetadata.averageFpsFromHeader);
  }
  if (streamMetadata.sampleRate.has_value()) {
    map["sampleRate"] = std::to_string(*streamMetadata.sampleRate);
  }
  if (streamMetadata.numChannels.has_value()) {
    map["numChannels"] = std::to_string(*streamMetadata.numChannels);
  }
  if (streamMetadata.sampleFormat.has_value()) {
    map["sampleFormat"] = quoteValue(streamMetadata.sampleFormat.value());
  }
  if (streamMetadata.mediaType == AVMEDIA_TYPE_VIDEO) {
    map["mediaType"] = quoteValue("video");
  } else if (streamMetadata.mediaType == AVMEDIA_TYPE_AUDIO) {
    map["mediaType"] = quoteValue("audio");
  } else {
    map["mediaType"] = quoteValue("other");
  }

  // Check whether content-based metadata is available for this stream.
  // In exact mode: content-based metadata exists for all streams.
  // In approximate mode: content-based metadata does not exist for any stream.
  // In custom_frame_mappings: content-based metadata exists only for the active
  // stream.
  //
  // Our fallback logic assumes content-based metadata is available.
  // It is available for decoding on the active stream, but would break
  // when getting metadata from non-active streams.
  if ((seekMode != SeekMode::custom_frame_mappings) ||
      (seekMode == SeekMode::custom_frame_mappings &&
       stream_index == activeStreamIndex)) {
    writeFallbackBasedMetadata(map, streamMetadata, seekMode);
  } else if (seekMode == SeekMode::custom_frame_mappings) {
    // If this is not the active stream, then we don't have content-based
    // metadata for custom frame mappings. In that case, we want the same
    // behavior as we would get with approximate mode. Encoding this behavior in
    // the fallback logic itself is tricky and not worth it for this corner
    // case. So we hardcode in approximate mode.
    //
    // TODO: This hacky behavior is only necessary because the custom frame
    //       mapping is supplied in SingleStreamDecoder::addVideoStream() rather
    //       than in the constructor. And it's supplied to addVideoStream() and
    //       not the constructor because we need to know the stream index. If we
    //       can encode the relevant stream indices into custom frame mappings
    //       itself, then we can put it in the constructor.
    writeFallbackBasedMetadata(map, streamMetadata, SeekMode::approximate);
  }

  return mapToJson(map);
}

} // namespace facebook::torchcodec
