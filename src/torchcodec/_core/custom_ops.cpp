// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/pybind11.h>
#include <cstdint>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include "AVIOFileLikeContext.h"
#include "AVIOTensorContext.h"
#include "Encoder.h"
#include "SingleStreamDecoder.h"
#include "StableABICompat.h"
#include "ValidationUtils.h"

namespace facebook::torchcodec {

// ==============================
// Define the operators
// ==============================
// All instances of accepting the decoder as a tensor must be annotated with
// `Tensor(a!)`. The `(a!)` part normally indicates that the tensor is being
// mutated in place. We need it to make sure that torch.compile does not reorder
// calls to these functions. For more detail, see:
//   https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native#readme
STABLE_TORCH_LIBRARY(torchcodec_ns, m) {
  // Note: impl_abstract_pystub is not available in stable ABI.
  // The Python stubs are handled via Python-side registration instead.
  m.def("create_from_file(str filename, str? seek_mode=None) -> Tensor");
  m.def(
      "encode_audio_to_file(Tensor samples, int sample_rate, str filename, int? bit_rate=None, int? num_channels=None, int? desired_sample_rate=None) -> ()");
  m.def(
      "encode_audio_to_tensor(Tensor samples, int sample_rate, str format, int? bit_rate=None, int? num_channels=None, int? desired_sample_rate=None) -> Tensor");
  m.def(
      "_encode_audio_to_file_like(Tensor samples, int sample_rate, str format, int file_like_context, int? bit_rate=None, int? num_channels=None, int? desired_sample_rate=None) -> ()");
  m.def(
      "encode_video_to_file(Tensor frames, float frame_rate, str filename, str? codec=None, str? pixel_format=None, float? crf=None, str? preset=None, str[]? extra_options=None) -> ()");
  m.def(
      "encode_video_to_tensor(Tensor frames, float frame_rate, str format, str? codec=None, str? pixel_format=None, float? crf=None, str? preset=None, str[]? extra_options=None) -> Tensor");
  m.def(
      "_encode_video_to_file_like(Tensor frames, float frame_rate, str format, int file_like_context, str? codec=None, str? pixel_format=None, float? crf=None, str? preset=None, str[]? extra_options=None) -> ()");
  m.def(
      "create_from_tensor(Tensor video_tensor, str? seek_mode=None) -> Tensor");
  m.def(
      "_create_from_file_like(int file_like_context, str? seek_mode=None) -> Tensor");
  m.def(
      "_add_video_stream(Tensor(a!) decoder, *, int? num_threads=None, str? dimension_order=None, int? stream_index=None, str device=\"cpu\", str device_variant=\"ffmpeg\", str transform_specs=\"\", Tensor? custom_frame_mappings_all_frames=None, Tensor? custom_frame_mappings_is_key_frame=None, Tensor? custom_frame_mappings_duration=None, str? color_conversion_library=None) -> ()");
  m.def(
      "add_video_stream(Tensor(a!) decoder, *, int? num_threads=None, str? dimension_order=None, int? stream_index=None, str device=\"cpu\", str device_variant=\"ffmpeg\", str transform_specs=\"\", Tensor? custom_frame_mappings_all_frames=None, Tensor? custom_frame_mappings_is_key_frame=None, Tensor? custom_frame_mappings_duration=None) -> ()");
  m.def(
      "add_audio_stream(Tensor(a!) decoder, *, int? stream_index=None, int? sample_rate=None, int? num_channels=None) -> ()");
  m.def("seek_to_pts(Tensor(a!) decoder, float seconds) -> ()");
  m.def("get_next_frame(Tensor(a!) decoder) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frame_at_pts(Tensor(a!) decoder, float seconds) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frame_at_index(Tensor(a!) decoder, *, int frame_index) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frames_at_indices(Tensor(a!) decoder, *, Tensor frame_indices) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frames_in_range(Tensor(a!) decoder, *, int start, int stop, int? step=None) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frames_by_pts_in_range(Tensor(a!) decoder, *, float start_seconds, float stop_seconds) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frames_by_pts_in_range_audio(Tensor(a!) decoder, *, float start_seconds, float? stop_seconds) -> (Tensor, Tensor)");
  m.def(
      "get_frames_by_pts(Tensor(a!) decoder, *, Tensor timestamps) -> (Tensor, Tensor, Tensor)");
  m.def("_get_key_frame_indices(Tensor(a!) decoder) -> Tensor");
  m.def("get_json_metadata(Tensor(a!) decoder) -> str");
  m.def("get_container_json_metadata(Tensor(a!) decoder) -> str");
  m.def(
      "get_stream_json_metadata(Tensor(a!) decoder, int stream_index) -> str");
  m.def("_get_json_ffmpeg_library_versions() -> str");
  m.def("_get_backend_details(Tensor(a!) decoder) -> str");
  m.def(
      "_test_frame_pts_equality(Tensor(a!) decoder, *, int frame_index, float pts_seconds_to_test) -> bool");
  m.def("scan_all_streams_to_update_metadata(Tensor(a!) decoder) -> ()");
  m.def("_destroy_decoder(Tensor(a!) decoder) -> ()");
}

namespace {

// Global registry for managing decoder lifetime.
// Since stable ABI's from_blob doesn't support deleters, we store the unique_ptr
// in this map and use the raw pointer value in the tensor. Cleanup happens when
// the registry entry is explicitly removed or when the process exits.
// Note: This map is not thread-safe; concurrent decoder creation/destruction
// from multiple threads is not supported.
std::unordered_map<SingleStreamDecoder*, std::unique_ptr<SingleStreamDecoder>>
    g_decoder_registry;

StableTensor wrapDecoderPointerToTensor(
    std::unique_ptr<SingleStreamDecoder> uniqueDecoder) {
  SingleStreamDecoder* decoder = uniqueDecoder.get();

  // Store in registry for lifetime management
  g_decoder_registry[decoder] = std::move(uniqueDecoder);

  // Create a tensor containing the pointer value
  StableTensor tensor = stableEmpty({1}, kStableInt64, StableDevice(kStableCPU));
  *tensor.mutable_data_ptr<int64_t>() = reinterpret_cast<int64_t>(decoder);

  return tensor;
}

SingleStreamDecoder* unwrapTensorToGetDecoder(StableTensor& tensor) {
  STABLE_CHECK(
      stableIsContiguous(tensor),
      "fake decoder tensor must be contiguous! This is an internal error, please report on the torchcodec issue tracker.");
  int64_t ptrValue = *tensor.const_data_ptr<int64_t>();
  SingleStreamDecoder* decoder = reinterpret_cast<SingleStreamDecoder*>(ptrValue);
  return decoder;
}

// Destroys a decoder and removes it from the registry.
// This must be called to avoid memory leaks.
void destroyDecoder(StableTensor& tensor) {
  int64_t ptrValue = *tensor.const_data_ptr<int64_t>();
  SingleStreamDecoder* decoder = reinterpret_cast<SingleStreamDecoder*>(ptrValue);
  auto it = g_decoder_registry.find(decoder);
  if (it != g_decoder_registry.end()) {
    g_decoder_registry.erase(it);
  }
}

// The elements of this tuple are all tensors that represent a single frame:
//   1. The frame data, which is a multidimensional tensor.
//   2. A single float value for the pts in seconds.
//   3. A single float value for the duration in seconds.
// The reason we use Tensors for the second and third values is so we can run
// under torch.compile().
using OpsFrameOutput = std::tuple<StableTensor, StableTensor, StableTensor>;

OpsFrameOutput makeOpsFrameOutput(FrameOutput& frame) {
  return std::make_tuple(
      frame.data,
      stableScalarTensor(frame.ptsSeconds),
      stableScalarTensor(frame.durationSeconds));
}

// All elements of this tuple are tensors of the same leading dimension. The
// tuple represents the frames for N total frames, where N is the dimension of
// each stacked tensor. The elments are:
//   1. Stacked tensor of data for all N frames. Each frame is also a
//   multidimensional tensor.
//   2. Tensor of N pts values in seconds, where each pts is a single
//   float.
//   3. Tensor of N durationis in seconds, where each duration is a
//   single float.
using OpsFrameBatchOutput = std::tuple<StableTensor, StableTensor, StableTensor>;

OpsFrameBatchOutput makeOpsFrameBatchOutput(FrameBatchOutput& batch) {
  return std::make_tuple(batch.data, batch.ptsSeconds, batch.durationSeconds);
}

// The elements of this tuple are all tensors that represent the concatenation
// of multiple audio frames:
//   1. The frames data (concatenated)
//   2. A single float value for the pts of the first frame, in seconds.
using OpsAudioFramesOutput = std::tuple<StableTensor, StableTensor>;

OpsAudioFramesOutput makeOpsAudioFramesOutput(AudioFramesOutput& audioFrames) {
  return std::make_tuple(
      audioFrames.data,
      stableScalarTensor(audioFrames.ptsSeconds));
}

std::string quoteValue(const std::string& value) {
  return "\"" + value + "\"";
}

// Helper function to unflatten extra_options, alternating keys and values
std::map<std::string, std::string> unflattenExtraOptions(
    const std::vector<std::string>& opts) {
  std::map<std::string, std::string> optionsMap;
  for (size_t i = 0; i < opts.size(); i += 2) {
    optionsMap[opts[i]] = opts[i + 1];
  }
  return optionsMap;
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

SeekMode seekModeFromString(std::string seekMode) {
  if (seekMode == "exact") {
    return SeekMode::exact;
  } else if (seekMode == "approximate") {
    return SeekMode::approximate;
  } else if (seekMode == "custom_frame_mappings") {
    return SeekMode::custom_frame_mappings;
  } else {
    STABLE_CHECK(false, "Invalid seek mode: " + std::string(seekMode));
  }
}

void writeFallbackBasedMetadata(
    std::map<std::string, std::string>& map,
    const StreamMetadata& streamMetadata,
    SeekMode seekMode) {
  auto durationSeconds = streamMetadata.getDurationSeconds(seekMode);
  if (durationSeconds.has_value()) {
    map["durationSeconds"] = std::to_string(durationSeconds.value());
  }

  auto numFrames = streamMetadata.getNumFrames(seekMode);
  if (numFrames.has_value()) {
    map["numFrames"] = std::to_string(numFrames.value());
  }

  double beginStreamSeconds = streamMetadata.getBeginStreamSeconds(seekMode);
  map["beginStreamSeconds"] = std::to_string(beginStreamSeconds);

  auto endStreamSeconds = streamMetadata.getEndStreamSeconds(seekMode);
  if (endStreamSeconds.has_value()) {
    map["endStreamSeconds"] = std::to_string(endStreamSeconds.value());
  }

  auto averageFps = streamMetadata.getAverageFps(seekMode);
  if (averageFps.has_value()) {
    map["averageFps"] = std::to_string(averageFps.value());
  }
}

int checkedToPositiveInt(const std::string& str) {
  int ret = 0;
  try {
    ret = std::stoi(str);
  } catch (const std::invalid_argument&) {
    STABLE_CHECK(false, "String cannot be converted to an int:" + str);
  } catch (const std::out_of_range&) {
    STABLE_CHECK(false, "String would become integer out of range:" + str);
  }
  STABLE_CHECK(ret > 0, "String must be a positive integer:" + str);
  return ret;
}

int checkedToNonNegativeInt(const std::string& str) {
  int ret = 0;
  try {
    ret = std::stoi(str);
  } catch (const std::invalid_argument&) {
    STABLE_CHECK(false, "String cannot be converted to an int:" + str);
  } catch (const std::out_of_range&) {
    STABLE_CHECK(false, "String would become integer out of range:" + str);
  }
  STABLE_CHECK(ret >= 0, "String must be a non-negative integer:" + str);
  return ret;
}

// Resize transform specs take the form:
//
//   "resize, <height>, <width>"
//
// Where "resize" is the string literal and <height> and <width> are positive
// integers.
Transform* makeResizeTransform(
    const std::vector<std::string>& resizeTransformSpec) {
  STABLE_CHECK(
      resizeTransformSpec.size() == 3,
      "resizeTransformSpec must have 3 elements including its name");
  int height = checkedToPositiveInt(resizeTransformSpec[1]);
  int width = checkedToPositiveInt(resizeTransformSpec[2]);
  return new ResizeTransform(FrameDims(height, width));
}

// Crop transform specs take the form:
//
//   "crop, <height>, <width>, <x>, <y>"
//
// Where "crop" is the string literal and <height>, <width>, <x> and <y> are
// positive integers. <x> and <y> are the x and y coordinates of the top left
// corner of the crop. Note that we follow the PyTorch convention of (height,
// width) for specifying image dimensions; FFmpeg uses (width, height).
Transform* makeCropTransform(
    const std::vector<std::string>& cropTransformSpec) {
  STABLE_CHECK(
      cropTransformSpec.size() == 5,
      "cropTransformSpec must have 5 elements including its name");
  int height = checkedToPositiveInt(cropTransformSpec[1]);
  int width = checkedToPositiveInt(cropTransformSpec[2]);
  int x = checkedToNonNegativeInt(cropTransformSpec[3]);
  int y = checkedToNonNegativeInt(cropTransformSpec[4]);
  return new CropTransform(FrameDims(height, width), x, y);
}

// CenterCrop transform specs take the form:
//
//   "center_crop, <height>, <width>"
//
// Where "center_crop" is the string literal and <height>, <width> are
// positive integers. Note that we follow the PyTorch convention of (height,
// width) for specifying image dimensions; FFmpeg uses (width, height).
Transform* makeCenterCropTransform(
    const std::vector<std::string>& cropTransformSpec) {
  STABLE_CHECK(
      cropTransformSpec.size() == 3,
      "cropTransformSpec must have 3 elements including its name");
  int height = checkedToPositiveInt(cropTransformSpec[1]);
  int width = checkedToPositiveInt(cropTransformSpec[2]);
  return new CropTransform(FrameDims(height, width));
}

std::vector<std::string> split(const std::string& str, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(str);
  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

// The transformSpecsRaw string is always in the format:
//
//   "name1, param1, param2, ...; name2, param1, param2, ...; ..."
//
// Where "nameX" is the name of the transform, and "paramX" are the parameters.
std::vector<Transform*> makeTransforms(const std::string& transformSpecsRaw) {
  std::vector<Transform*> transforms;
  std::vector<std::string> transformSpecs = split(transformSpecsRaw, ';');
  for (const std::string& transformSpecRaw : transformSpecs) {
    std::vector<std::string> transformSpec = split(transformSpecRaw, ',');
    STABLE_CHECK(
        transformSpec.size() >= 1,
        "Invalid transform spec: " + transformSpecRaw);

    auto name = transformSpec[0];
    if (name == "resize") {
      transforms.push_back(makeResizeTransform(transformSpec));
    } else if (name == "crop") {
      transforms.push_back(makeCropTransform(transformSpec));
    } else if (name == "center_crop") {
      transforms.push_back(makeCenterCropTransform(transformSpec));
    } else {
      STABLE_CHECK(false, "Invalid transform name: " + name);
    }
  }
  return transforms;
}

} // namespace

// ==============================
// Implementations for the operators
// ==============================

// Create a SingleStreamDecoder from file and wrap the pointer in a tensor.
StableTensor create_from_file(
    std::string filename,
    std::optional<std::string> seek_mode = std::nullopt) {
  std::string filenameStr(filename);

  SeekMode realSeek = SeekMode::exact;
  if (seek_mode.has_value()) {
    realSeek = seekModeFromString(seek_mode.value());
  }

  std::unique_ptr<SingleStreamDecoder> uniqueDecoder =
      std::make_unique<SingleStreamDecoder>(filenameStr, realSeek);

  return wrapDecoderPointerToTensor(std::move(uniqueDecoder));
}

// Create a SingleStreamDecoder from the actual bytes of a video and wrap the
// pointer in a tensor. The SingleStreamDecoder will decode the provided bytes.
StableTensor create_from_tensor(
    StableTensor video_tensor,
    std::optional<std::string> seek_mode = std::nullopt) {
  STABLE_CHECK(stableIsContiguous(video_tensor), "video_tensor must be contiguous");
  STABLE_CHECK(
      video_tensor.scalar_type() == kStableUInt8,
      "video_tensor must be kUInt8");

  SeekMode realSeek = SeekMode::exact;
  if (seek_mode.has_value()) {
    realSeek = seekModeFromString(seek_mode.value());
  }

  auto avioContextHolder =
      std::make_unique<AVIOFromTensorContext>(video_tensor);

  std::unique_ptr<SingleStreamDecoder> uniqueDecoder =
      std::make_unique<SingleStreamDecoder>(
          std::move(avioContextHolder), realSeek);
  return wrapDecoderPointerToTensor(std::move(uniqueDecoder));
}

StableTensor _create_from_file_like(
    int64_t file_like_context,
    std::optional<std::string> seek_mode) {
  auto fileLikeContext =
      reinterpret_cast<AVIOFileLikeContext*>(file_like_context);
  STABLE_CHECK(
      fileLikeContext != nullptr, "file_like_context must be a valid pointer");
  std::unique_ptr<AVIOFileLikeContext> avioContextHolder(fileLikeContext);

  SeekMode realSeek = SeekMode::exact;
  if (seek_mode.has_value()) {
    realSeek = seekModeFromString(seek_mode.value());
  }

  std::unique_ptr<SingleStreamDecoder> uniqueDecoder =
      std::make_unique<SingleStreamDecoder>(
          std::move(avioContextHolder), realSeek);
  return wrapDecoderPointerToTensor(std::move(uniqueDecoder));
}

void _add_video_stream(
    StableTensor& decoder,
    std::optional<int64_t> num_threads = std::nullopt,
    std::optional<std::string> dimension_order = std::nullopt,
    std::optional<int64_t> stream_index = std::nullopt,
    std::string device = "cpu",
    std::string device_variant = "ffmpeg",
    std::string transform_specs = "",
    std::optional<StableTensor> custom_frame_mappings_all_frames = std::nullopt,
    std::optional<StableTensor> custom_frame_mappings_is_key_frame = std::nullopt,
    std::optional<StableTensor> custom_frame_mappings_duration = std::nullopt,
    std::optional<std::string> color_conversion_library = std::nullopt) {
  VideoStreamOptions videoStreamOptions;
  videoStreamOptions.ffmpegThreadCount = num_threads;

  if (dimension_order.has_value()) {
    std::string stdDimensionOrder{dimension_order.value()};
    STABLE_CHECK(stdDimensionOrder == "NHWC" || stdDimensionOrder == "NCHW", "dimension_order must be NHWC or NCHW");
    videoStreamOptions.dimensionOrder = stdDimensionOrder;
  }
  if (color_conversion_library.has_value()) {
    std::string stdColorConversionLibrary{color_conversion_library.value()};
    if (stdColorConversionLibrary == "filtergraph") {
      videoStreamOptions.colorConversionLibrary =
          ColorConversionLibrary::FILTERGRAPH;
    } else if (stdColorConversionLibrary == "swscale") {
      videoStreamOptions.colorConversionLibrary =
          ColorConversionLibrary::SWSCALE;
    } else {
      STABLE_CHECK(
          false,
          "Invalid color_conversion_library=" + stdColorConversionLibrary +
          ". color_conversion_library must be either filtergraph or swscale.");
    }
  }

  validateDeviceInterface(std::string(device), std::string(device_variant));

  videoStreamOptions.device = StableDevice(std::string(device));
  videoStreamOptions.deviceVariant = device_variant;

  std::vector<Transform*> transforms =
      makeTransforms(std::string(transform_specs));

  std::optional<SingleStreamDecoder::FrameMappings> converted_mappings =
      std::nullopt;
  if (custom_frame_mappings_all_frames.has_value()) {
    STABLE_CHECK(
        custom_frame_mappings_is_key_frame.has_value() &&
            custom_frame_mappings_duration.has_value(),
        "All three custom_frame_mappings tensors must be provided together");
    converted_mappings = SingleStreamDecoder::FrameMappings{
        std::move(custom_frame_mappings_all_frames.value()),
        std::move(custom_frame_mappings_is_key_frame.value()),
        std::move(custom_frame_mappings_duration.value())};
  }
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  videoDecoder->addVideoStream(
      stream_index.value_or(-1),
      transforms,
      videoStreamOptions,
      converted_mappings);
}

// Add a new video stream at `stream_index` using the provided options.
void add_video_stream(
    StableTensor& decoder,
    std::optional<int64_t> num_threads = std::nullopt,
    std::optional<std::string> dimension_order = std::nullopt,
    std::optional<int64_t> stream_index = std::nullopt,
    std::string device = "cpu",
    std::string device_variant = "ffmpeg",
    std::string transform_specs = "",
    std::optional<StableTensor> custom_frame_mappings_all_frames = std::nullopt,
    std::optional<StableTensor> custom_frame_mappings_is_key_frame = std::nullopt,
    std::optional<StableTensor> custom_frame_mappings_duration = std::nullopt) {
  _add_video_stream(
      decoder,
      num_threads,
      dimension_order,
      stream_index,
      device,
      device_variant,
      transform_specs,
      custom_frame_mappings_all_frames,
      custom_frame_mappings_is_key_frame,
      custom_frame_mappings_duration);
}

void add_audio_stream(
    StableTensor& decoder,
    std::optional<int64_t> stream_index = std::nullopt,
    std::optional<int64_t> sample_rate = std::nullopt,
    std::optional<int64_t> num_channels = std::nullopt) {
  AudioStreamOptions audioStreamOptions;
  audioStreamOptions.sampleRate = sample_rate;
  audioStreamOptions.numChannels = num_channels;

  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  videoDecoder->addAudioStream(stream_index.value_or(-1), audioStreamOptions);
}

// Seek to a particular presentation timestamp in the video in seconds.
void seek_to_pts(StableTensor& decoder, double seconds) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  videoDecoder->setCursorPtsInSeconds(seconds);
}

// Get the next frame from the video as a tuple that has the frame data, pts and
// duration as tensors.
OpsFrameOutput get_next_frame(StableTensor& decoder) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  FrameOutput result;
  try {
    result = videoDecoder->getNextFrame();
  } catch (const SingleStreamDecoder::EndOfFileException& e) {
    throw std::out_of_range(e.what());
  }
  return makeOpsFrameOutput(result);
}

// Return the frame that is visible at a given timestamp in seconds. Each frame
// in FFMPEG has a presentation timestamp and a duration. The frame visible at a
// given timestamp T has T >= PTS and T < PTS + Duration.
OpsFrameOutput get_frame_at_pts(StableTensor& decoder, double seconds) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  FrameOutput result;
  try {
    result = videoDecoder->getFramePlayedAt(seconds);
  } catch (const SingleStreamDecoder::EndOfFileException& e) {
    throw std::out_of_range(e.what());
  }
  return makeOpsFrameOutput(result);
}

// Return the frame that is visible at a given index in the video.
OpsFrameOutput get_frame_at_index(StableTensor& decoder, int64_t frame_index) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  auto result = videoDecoder->getFrameAtIndex(frame_index);
  return makeOpsFrameOutput(result);
}

// Return the frames at given indices for a given stream
OpsFrameBatchOutput get_frames_at_indices(
    StableTensor& decoder,
    const StableTensor& frame_indices) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  auto result = videoDecoder->getFramesAtIndices(frame_indices);
  return makeOpsFrameBatchOutput(result);
}

// Return the frames inside a range as a single stacked Tensor. The range is
// defined as [start, stop).
OpsFrameBatchOutput get_frames_in_range(
    StableTensor& decoder,
    int64_t start,
    int64_t stop,
    std::optional<int64_t> step = std::nullopt) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  auto result = videoDecoder->getFramesInRange(start, stop, step.value_or(1));
  return makeOpsFrameBatchOutput(result);
}

// Return the frames at given ptss for a given stream
OpsFrameBatchOutput get_frames_by_pts(
    StableTensor& decoder,
    const StableTensor& timestamps) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  auto result = videoDecoder->getFramesPlayedAt(timestamps);
  return makeOpsFrameBatchOutput(result);
}

// Return the frames inside the range as a single stacked Tensor. The range is
// defined as [start_seconds, stop_seconds). The frames are stacked in pts
// order.
OpsFrameBatchOutput get_frames_by_pts_in_range(
    StableTensor& decoder,
    double start_seconds,
    double stop_seconds) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  auto result =
      videoDecoder->getFramesPlayedInRange(start_seconds, stop_seconds);
  return makeOpsFrameBatchOutput(result);
}

OpsAudioFramesOutput get_frames_by_pts_in_range_audio(
    StableTensor& decoder,
    double start_seconds,
    std::optional<double> stop_seconds = std::nullopt) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  auto result =
      videoDecoder->getFramesPlayedInRangeAudio(start_seconds, stop_seconds);
  return makeOpsAudioFramesOutput(result);
}

void encode_audio_to_file(
    const StableTensor& samples,
    int64_t sample_rate,
    std::string file_name,
    std::optional<int64_t> bit_rate = std::nullopt,
    std::optional<int64_t> num_channels = std::nullopt,
    std::optional<int64_t> desired_sample_rate = std::nullopt) {
  AudioStreamOptions audioStreamOptions;
  audioStreamOptions.bitRate = validateOptionalInt64ToInt(bit_rate, "bit_rate");
  audioStreamOptions.numChannels =
      validateOptionalInt64ToInt(num_channels, "num_channels");
  audioStreamOptions.sampleRate =
      validateOptionalInt64ToInt(desired_sample_rate, "desired_sample_rate");
  AudioEncoder(
      samples,
      validateInt64ToInt(sample_rate, "sample_rate"),
      file_name,
      audioStreamOptions)
      .encode();
}

StableTensor encode_audio_to_tensor(
    const StableTensor& samples,
    int64_t sample_rate,
    std::string format,
    std::optional<int64_t> bit_rate = std::nullopt,
    std::optional<int64_t> num_channels = std::nullopt,
    std::optional<int64_t> desired_sample_rate = std::nullopt) {
  auto avioContextHolder = std::make_unique<AVIOToTensorContext>();
  AudioStreamOptions audioStreamOptions;
  audioStreamOptions.bitRate = validateOptionalInt64ToInt(bit_rate, "bit_rate");
  audioStreamOptions.numChannels =
      validateOptionalInt64ToInt(num_channels, "num_channels");
  audioStreamOptions.sampleRate =
      validateOptionalInt64ToInt(desired_sample_rate, "desired_sample_rate");
  return AudioEncoder(
             samples,
             validateInt64ToInt(sample_rate, "sample_rate"),
             format,
             std::move(avioContextHolder),
             audioStreamOptions)
      .encodeToTensor();
}

void _encode_audio_to_file_like(
    const StableTensor& samples,
    int64_t sample_rate,
    std::string format,
    int64_t file_like_context,
    std::optional<int64_t> bit_rate = std::nullopt,
    std::optional<int64_t> num_channels = std::nullopt,
    std::optional<int64_t> desired_sample_rate = std::nullopt) {
  auto fileLikeContext =
      reinterpret_cast<AVIOFileLikeContext*>(file_like_context);
  STABLE_CHECK(
      fileLikeContext != nullptr, "file_like_context must be a valid pointer");
  std::unique_ptr<AVIOFileLikeContext> avioContextHolder(fileLikeContext);

  AudioStreamOptions audioStreamOptions;
  audioStreamOptions.bitRate = validateOptionalInt64ToInt(bit_rate, "bit_rate");
  audioStreamOptions.numChannels =
      validateOptionalInt64ToInt(num_channels, "num_channels");
  audioStreamOptions.sampleRate =
      validateOptionalInt64ToInt(desired_sample_rate, "desired_sample_rate");

  AudioEncoder encoder(
      samples,
      validateInt64ToInt(sample_rate, "sample_rate"),
      format,
      std::move(avioContextHolder),
      audioStreamOptions);
  encoder.encode();
}

void encode_video_to_file(
    const StableTensor& frames,
    double frame_rate,
    std::string file_name,
    std::optional<std::string> codec = std::nullopt,
    std::optional<std::string> pixel_format = std::nullopt,
    std::optional<double> crf = std::nullopt,
    std::optional<std::string> preset = std::nullopt,
    std::optional<std::vector<std::string>> extra_options = std::nullopt) {
  VideoStreamOptions videoStreamOptions;
  videoStreamOptions.codec = std::move(codec);
  videoStreamOptions.pixelFormat = std::move(pixel_format);
  videoStreamOptions.crf = crf;
  videoStreamOptions.preset = preset;

  if (extra_options.has_value()) {
    videoStreamOptions.extraOptions =
        unflattenExtraOptions(extra_options.value());
  }

  VideoEncoder(frames, frame_rate, file_name, videoStreamOptions).encode();
}

StableTensor encode_video_to_tensor(
    const StableTensor& frames,
    double frame_rate,
    std::string format,
    std::optional<std::string> codec = std::nullopt,
    std::optional<std::string> pixel_format = std::nullopt,
    std::optional<double> crf = std::nullopt,
    std::optional<std::string> preset = std::nullopt,
    std::optional<std::vector<std::string>> extra_options = std::nullopt) {
  auto avioContextHolder = std::make_unique<AVIOToTensorContext>();
  VideoStreamOptions videoStreamOptions;
  videoStreamOptions.codec = std::move(codec);
  videoStreamOptions.pixelFormat = std::move(pixel_format);
  videoStreamOptions.crf = crf;
  videoStreamOptions.preset = preset;

  if (extra_options.has_value()) {
    videoStreamOptions.extraOptions =
        unflattenExtraOptions(extra_options.value());
  }

  return VideoEncoder(
             frames,
             frame_rate,
             format,
             std::move(avioContextHolder),
             videoStreamOptions)
      .encodeToTensor();
}

void _encode_video_to_file_like(
    const StableTensor& frames,
    double frame_rate,
    std::string format,
    int64_t file_like_context,
    std::optional<std::string> codec = std::nullopt,
    std::optional<std::string> pixel_format = std::nullopt,
    std::optional<double> crf = std::nullopt,
    std::optional<std::string> preset = std::nullopt,
    std::optional<std::vector<std::string>> extra_options = std::nullopt) {
  auto fileLikeContext =
      reinterpret_cast<AVIOFileLikeContext*>(file_like_context);
  STABLE_CHECK(
      fileLikeContext != nullptr, "file_like_context must be a valid pointer");
  std::unique_ptr<AVIOFileLikeContext> avioContextHolder(fileLikeContext);

  VideoStreamOptions videoStreamOptions;
  videoStreamOptions.codec = std::move(codec);
  videoStreamOptions.pixelFormat = std::move(pixel_format);
  videoStreamOptions.crf = crf;
  videoStreamOptions.preset = preset;

  if (extra_options.has_value()) {
    videoStreamOptions.extraOptions =
        unflattenExtraOptions(extra_options.value());
  }

  VideoEncoder encoder(
      frames,
      frame_rate,
      format,
      std::move(avioContextHolder),
      videoStreamOptions);
  encoder.encode();
}

// For testing only. We need to implement this operation as a core library
// function because what we're testing is round-tripping pts values as
// double-precision floating point numbers from C++ to Python and back to C++.
// We want to make sure that the value is preserved exactly, bit-for-bit, during
// this process.
//
// Returns true if for the given decoder, the pts
// value when converted to seconds as a double is exactly pts_seconds_to_test.
// Returns false otherwise.
bool _test_frame_pts_equality(
    StableTensor& decoder,
    int64_t frame_index,
    double pts_seconds_to_test) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  return pts_seconds_to_test ==
      videoDecoder->getPtsSecondsForFrame(frame_index);
}

StableTensor _get_key_frame_indices(StableTensor& decoder) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  return videoDecoder->getKeyFrameIndices();
}

// Get the metadata from the video as a string.
std::string get_json_metadata(StableTensor& decoder) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);

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
      std::to_string(durationSecondsFromHeader);

  if (videoMetadata.bitRate.has_value()) {
    metadataMap["bitRate"] = std::to_string(videoMetadata.bitRate.value());
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
          std::to_string(*streamMetadata.beginStreamPtsSecondsFromContent);
    }
    if (streamMetadata.endStreamPtsSecondsFromContent.has_value()) {
      metadataMap["endStreamSecondsFromContent"] =
          std::to_string(*streamMetadata.endStreamPtsSecondsFromContent);
    }
    if (streamMetadata.codecName.has_value()) {
      metadataMap["codec"] = quoteValue(streamMetadata.codecName.value());
    }
    if (streamMetadata.width.has_value()) {
      metadataMap["width"] = std::to_string(*streamMetadata.width);
    }
    if (streamMetadata.height.has_value()) {
      metadataMap["height"] = std::to_string(*streamMetadata.height);
    }
    if (streamMetadata.averageFpsFromHeader.has_value()) {
      metadataMap["averageFpsFromHeader"] =
          std::to_string(*streamMetadata.averageFpsFromHeader);
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

// Get the container metadata as a string.
std::string get_container_json_metadata(StableTensor& decoder) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);

  auto containerMetadata = videoDecoder->getContainerMetadata();

  std::map<std::string, std::string> map;

  if (containerMetadata.durationSecondsFromHeader.has_value()) {
    map["durationSecondsFromHeader"] =
        std::to_string(*containerMetadata.durationSecondsFromHeader);
  }

  if (containerMetadata.bitRate.has_value()) {
    map["bitRate"] = std::to_string(*containerMetadata.bitRate);
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

// Get the stream metadata as a string.
std::string get_stream_json_metadata(
    StableTensor& decoder,
    int64_t stream_index) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  auto allStreamMetadata =
      videoDecoder->getContainerMetadata().allStreamMetadata;
  if (stream_index < 0 ||
      stream_index >= static_cast<int64_t>(allStreamMetadata.size())) {
    throw std::out_of_range(
        "stream_index out of bounds: " + std::to_string(stream_index));
  }

  auto streamMetadata = allStreamMetadata[stream_index];
  auto seekMode = videoDecoder->getSeekMode();
  int activeStreamIndex = videoDecoder->getActiveStreamIndex();

  std::map<std::string, std::string> map;

  if (streamMetadata.durationSecondsFromHeader.has_value()) {
    map["durationSecondsFromHeader"] =
        std::to_string(*streamMetadata.durationSecondsFromHeader);
  }
  if (streamMetadata.bitRate.has_value()) {
    map["bitRate"] = std::to_string(*streamMetadata.bitRate);
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
        std::to_string(*streamMetadata.beginStreamSecondsFromHeader);
  }
  if (streamMetadata.beginStreamPtsSecondsFromContent.has_value()) {
    map["beginStreamSecondsFromContent"] =
        std::to_string(*streamMetadata.beginStreamPtsSecondsFromContent);
  }
  if (streamMetadata.endStreamPtsSecondsFromContent.has_value()) {
    map["endStreamSecondsFromContent"] =
        std::to_string(*streamMetadata.endStreamPtsSecondsFromContent);
  }
  if (streamMetadata.codecName.has_value()) {
    map["codec"] = quoteValue(streamMetadata.codecName.value());
  }
  if (streamMetadata.width.has_value()) {
    map["width"] = std::to_string(*streamMetadata.width);
  }
  if (streamMetadata.height.has_value()) {
    map["height"] = std::to_string(*streamMetadata.height);
  }
  if (streamMetadata.sampleAspectRatio.has_value()) {
    map["sampleAspectRatioNum"] =
        std::to_string((*streamMetadata.sampleAspectRatio).num);
    map["sampleAspectRatioDen"] =
        std::to_string((*streamMetadata.sampleAspectRatio).den);
  }
  if (streamMetadata.averageFpsFromHeader.has_value()) {
    map["averageFpsFromHeader"] =
        std::to_string(*streamMetadata.averageFpsFromHeader);
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

// Returns version information about the various FFMPEG libraries that are
// loaded in the program's address space.
// TODO: ideally we'd have a more robust way of getting the ffmpeg version,
// we're using av_version_info() which is not standardized and shouldn't be
// parsed by code (which we do!). See
// https://github.com/pytorch/torchcodec/issues/100
std::string _get_json_ffmpeg_library_versions() {
  std::stringstream ss;
  ss << "{\n";

  unsigned int version = avfilter_version();
  ss << "\"libavfilter\": [" << AV_VERSION_MAJOR(version) << ", "
     << AV_VERSION_MINOR(version) << ", " << AV_VERSION_MICRO(version)
     << "],\n";
  version = avutil_version();
  ss << "\"libavutil\": [" << AV_VERSION_MAJOR(version) << ", "
     << AV_VERSION_MINOR(version) << ", " << AV_VERSION_MICRO(version)
     << "],\n";
  version = avcodec_version();
  ss << "\"libavcodec\": [" << AV_VERSION_MAJOR(version) << ", "
     << AV_VERSION_MINOR(version) << ", " << AV_VERSION_MICRO(version)
     << "],\n";
  version = avformat_version();
  ss << "\"libavformat\": [" << AV_VERSION_MAJOR(version) << ", "
     << AV_VERSION_MINOR(version) << ", " << AV_VERSION_MICRO(version)
     << "],\n";
  ss << "\"ffmpeg_version\": \"" << av_version_info() << "\"\n";
  ss << "}\n";

  return ss.str();
}

std::string get_backend_details(StableTensor& decoder) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  return videoDecoder->getDeviceInterfaceDetails();
}

// Scans video packets to get more accurate metadata like frame count, exact
// keyframe positions, etc. Exact keyframe positions are useful for efficient
// accurate seeking. Note that this function reads the entire video but it does
// not decode frames. Reading a video file is much cheaper than decoding it.
void scan_all_streams_to_update_metadata(StableTensor& decoder) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  videoDecoder->scanFileAndUpdateMetadataAndIndex();
}

// Destroys the decoder and frees associated resources.
// This should be called when the decoder is no longer needed.
void _destroy_decoder(StableTensor& decoder) {
  destroyDecoder(decoder);
}

STABLE_TORCH_LIBRARY_IMPL(torchcodec_ns, BackendSelect, m) {
  m.impl("create_from_file", TORCH_BOX(&create_from_file));
  m.impl("create_from_tensor", TORCH_BOX(&create_from_tensor));
  m.impl("_create_from_file_like", TORCH_BOX(&_create_from_file_like));
  m.impl(
      "_get_json_ffmpeg_library_versions",
      TORCH_BOX(&_get_json_ffmpeg_library_versions));
  m.impl("encode_video_to_file", TORCH_BOX(&encode_video_to_file));
  m.impl("encode_video_to_tensor", TORCH_BOX(&encode_video_to_tensor));
  m.impl("_encode_video_to_file_like", TORCH_BOX(&_encode_video_to_file_like));
}

STABLE_TORCH_LIBRARY_IMPL(torchcodec_ns, CPU, m) {
  m.impl("encode_audio_to_file", TORCH_BOX(&encode_audio_to_file));
  m.impl("encode_audio_to_tensor", TORCH_BOX(&encode_audio_to_tensor));
  m.impl("_encode_audio_to_file_like", TORCH_BOX(&_encode_audio_to_file_like));
  m.impl("encode_video_to_file", TORCH_BOX(&encode_video_to_file));
  m.impl("encode_video_to_tensor", TORCH_BOX(&encode_video_to_tensor));
  m.impl("_encode_video_to_file_like", TORCH_BOX(&_encode_video_to_file_like));
  m.impl("seek_to_pts", TORCH_BOX(&seek_to_pts));
  m.impl("add_video_stream", TORCH_BOX(&add_video_stream));
  m.impl("_add_video_stream", TORCH_BOX(&_add_video_stream));
  m.impl("add_audio_stream", TORCH_BOX(&add_audio_stream));
  m.impl("get_next_frame", TORCH_BOX(&get_next_frame));
  m.impl("_get_key_frame_indices", TORCH_BOX(&_get_key_frame_indices));
  m.impl("get_json_metadata", TORCH_BOX(&get_json_metadata));
  m.impl("get_container_json_metadata", TORCH_BOX(&get_container_json_metadata));
  m.impl("get_stream_json_metadata", TORCH_BOX(&get_stream_json_metadata));
  m.impl("get_frame_at_pts", TORCH_BOX(&get_frame_at_pts));
  m.impl("get_frame_at_index", TORCH_BOX(&get_frame_at_index));
  m.impl("get_frames_at_indices", TORCH_BOX(&get_frames_at_indices));
  m.impl("get_frames_in_range", TORCH_BOX(&get_frames_in_range));
  m.impl("get_frames_by_pts_in_range", TORCH_BOX(&get_frames_by_pts_in_range));
  m.impl(
      "get_frames_by_pts_in_range_audio",
      TORCH_BOX(&get_frames_by_pts_in_range_audio));
  m.impl("get_frames_by_pts", TORCH_BOX(&get_frames_by_pts));
  m.impl("_test_frame_pts_equality", TORCH_BOX(&_test_frame_pts_equality));
  m.impl(
      "scan_all_streams_to_update_metadata",
      TORCH_BOX(&scan_all_streams_to_update_metadata));

  m.impl("_get_backend_details", TORCH_BOX(&get_backend_details));
  m.impl("_destroy_decoder", TORCH_BOX(&_destroy_decoder));
}

} // namespace facebook::torchcodec
