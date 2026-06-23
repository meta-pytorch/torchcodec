// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/AVIOTensorContext.h"
#include "src/torchcodec/_core/SingleStreamDecoder.h"

#include <c10/util/Flags.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <filesystem>
#include <fstream>
#include <iostream>

#ifdef FBCODE_BUILD
#include "tools/cxx/Resources.h"
#endif

using namespace ::testing;

C10_DEFINE_bool(
    dump_frames_for_debugging,
    false,
    "If true, we dump frames as bmp files for debugging.");

namespace facebook::torchcodec {

inline torch::stable::Tensor to_stable_tensor(const torch::Tensor& tensor) {
  torch::Tensor* p = new torch::Tensor(tensor);
  return torch::stable::Tensor(reinterpret_cast<AtenTensorHandle>(p));
}

inline torch::Tensor to_aten_tensor(const torch::stable::Tensor& t) {
  return *reinterpret_cast<torch::Tensor*>(t.get());
}

std::string get_resource_path(const std::string& filename) {
#ifdef FBCODE_BUILD
  std::string filepath = "pytorch/torchcodec/test/resources/" + filename;
  filepath = build::getResourcePath(filepath).string();
#else
  std::filesystem::path dir_path = std::filesystem::path(__FILE__);
  std::string filepath =
      dir_path.parent_path().string() + "/resources/" + filename;
#endif
  return filepath;
}

class SingleStreamDecoderTest : public testing::TestWithParam<bool> {
 protected:
  std::unique_ptr<SingleStreamDecoder> create_decoder_from_path(
      const std::string& filepath,
      bool use_memory_buffer) {
    if (use_memory_buffer) {
      std::ostringstream output_string_stream;
      std::ifstream input(filepath, std::ios::binary);
      output_string_stream << input.rdbuf();
      content_ = output_string_stream.str();

      // Note that we copy the data from the string into a new buffer. The
      // tensor has ownership of that buffer. This is not strictly necessary,
      // as the lifetime of the content_ string will outlast the decoder. But,
      // we do it to test the common usage where the decoder should own the
      // memory through the tensor.
      int64_t length = content_.length();
      char* data = new char[length];
      std::memcpy(data, content_.data(), length);
      auto deleter = [data](void*) { delete[] data; };
      torch::Tensor tensor = torch::from_blob(
          static_cast<void*>(data), {length}, deleter, {torch::kUInt8});

      auto context_holder =
          std::make_unique<AVIOFromTensorContext>(to_stable_tensor(tensor));
      return std::make_unique<SingleStreamDecoder>(
          std::move(context_holder), SeekMode::approximate);
    } else {
      return std::make_unique<SingleStreamDecoder>(
          filepath, SeekMode::approximate);
    }
  }

  std::string content_;
};

TEST_P(SingleStreamDecoderTest, ReturnsFpsAndDurationForVideoInMetadata) {
  std::string path = get_resource_path("nasa_13013.mp4");
  std::unique_ptr<SingleStreamDecoder> decoder =
      create_decoder_from_path(path, GetParam());
  ContainerMetadata metadata = decoder->get_container_metadata();
  EXPECT_EQ(metadata.num_audio_streams, 2);
  EXPECT_EQ(metadata.num_video_streams, 2);
#if LIBAVFORMAT_VERSION_MAJOR >= 60
  EXPECT_NEAR(metadata.bit_rate.value(), 412365, 1e-1);
#else
  EXPECT_NEAR(metadata.bit_rate.value(), 324915, 1e-1);
#endif
  EXPECT_EQ(metadata.all_stream_metadata.size(), 6);
  const auto& video_stream = metadata.all_stream_metadata[3];
  EXPECT_EQ(video_stream.media_type, AVMEDIA_TYPE_VIDEO);
  EXPECT_EQ(video_stream.codec_name, "h264");
  EXPECT_NEAR(*video_stream.average_fps_from_header, 29.97f, 1e-1);
  EXPECT_NEAR(*video_stream.bit_rate, 128783, 1e-1);
  EXPECT_NEAR(*video_stream.duration_seconds_from_header, 13.013, 1e-1);
  EXPECT_EQ(video_stream.num_frames_from_header, 390);
  EXPECT_FALSE(video_stream.begin_stream_pts_seconds_from_content.has_value());
  EXPECT_FALSE(video_stream.end_stream_pts_seconds_from_content.has_value());
  EXPECT_FALSE(video_stream.num_frames_from_content.has_value());
  decoder->scan_file_and_update_metadata_and_index();
  metadata = decoder->get_container_metadata();
  const auto& video_stream1 = metadata.all_stream_metadata[3];
  EXPECT_EQ(*video_stream1.begin_stream_pts_seconds_from_content, 0);
  EXPECT_EQ(*video_stream1.end_stream_pts_seconds_from_content, 13.013);
  EXPECT_EQ(*video_stream1.num_frames_from_content, 390);
}

TEST(SingleStreamDecoderTest, MissingVideoFileThrowsException) {
  EXPECT_THROW(
      SingleStreamDecoder("/this/file/does/not/exist"), std::runtime_error);
}

void dump_tensor_to_disk(
    const torch::Tensor& tensor,
    const std::string& filename) {
  std::vector<char> bytes = torch::pickle_save(tensor);
  std::ofstream fout(filename, std::ios::out | std::ios::binary);
  fout.write(bytes.data(), bytes.size());
  fout.close();
}

torch::Tensor read_tensor_from_disk(const std::string& filename) {
  std::string filepath = get_resource_path(filename);
  std::ifstream file(filepath, std::ios::binary);
  std::vector<char> data(
      (std::istreambuf_iterator<char>(file)),
      (std::istreambuf_iterator<char>()));
  VLOG(3) << "Read tensor from disk: " << filepath << ": " << data.size()
          << std::endl;
  return torch::pickle_load(data).toTensor().permute({2, 0, 1});
}

torch::Tensor float_and_normalize_frame(const torch::Tensor& frame) {
  torch::Tensor float_frame = frame.toType(torch::kFloat32);
  torch::Tensor normalized_frame = float_frame / 255.0;
  return normalized_frame;
}

double compute_average_cosine_similarity(
    const torch::Tensor& frame1,
    const torch::Tensor& frame2) {
  torch::Tensor frame1_norm = float_and_normalize_frame(frame1);
  torch::Tensor frame2_norm = float_and_normalize_frame(frame2);
  torch::Tensor cosine_similarities =
      torch::cosine_similarity(frame1_norm, frame2_norm);
  double average_cosine_similarity = cosine_similarities.mean().item<float>();
  return average_cosine_similarity;
}

TEST(SingleStreamDecoderTest, RespectsOutputTensorDimensionOrderFromOptions) {
  std::string path = get_resource_path("nasa_13013.mp4");
  std::unique_ptr<SingleStreamDecoder> decoder =
      std::make_unique<SingleStreamDecoder>(path);
  VideoStreamOptions video_stream_options;
  video_stream_options.dimension_order = "NHWC";
  std::vector<Transform*> transforms;
  decoder->add_video_stream(-1, transforms, video_stream_options);
  auto tensor = to_aten_tensor(decoder->get_next_frame().data);
  EXPECT_EQ(tensor.sizes(), std::vector<long>({270, 480, 3}));
}

TEST_P(SingleStreamDecoderTest, ReturnsFirstTwoFramesOfVideo) {
  std::string path = get_resource_path("nasa_13013.mp4");
  std::unique_ptr<SingleStreamDecoder> our_decoder =
      create_decoder_from_path(path, GetParam());
  std::vector<Transform*> transforms;
  our_decoder->add_video_stream(-1, transforms);
  auto output = our_decoder->get_next_frame();
  torch::Tensor tensor0_from_our_decoder = to_aten_tensor(output.data);
  EXPECT_EQ(tensor0_from_our_decoder.sizes(), std::vector<long>({3, 270, 480}));
  EXPECT_EQ(output.pts_seconds, 0.0);
  output = our_decoder->get_next_frame();
  torch::Tensor tensor1_from_our_decoder = to_aten_tensor(output.data);
  EXPECT_EQ(tensor1_from_our_decoder.sizes(), std::vector<long>({3, 270, 480}));
  EXPECT_EQ(output.pts_seconds, 1'001. / 30'000);

  torch::Tensor tensor0_from_ffmpeg =
      read_tensor_from_disk("nasa_13013.mp4.stream3.frame000000.pt");
  torch::Tensor tensor1_from_ffmpeg =
      read_tensor_from_disk("nasa_13013.mp4.stream3.frame000001.pt");

  EXPECT_EQ(tensor1_from_ffmpeg.sizes(), std::vector<long>({3, 270, 480}));
  EXPECT_TRUE(torch::equal(tensor0_from_our_decoder, tensor0_from_ffmpeg));
  EXPECT_TRUE(torch::equal(tensor1_from_our_decoder, tensor1_from_ffmpeg));
  EXPECT_TRUE(
      torch::allclose(tensor0_from_our_decoder, tensor0_from_ffmpeg, 0.1, 20));
  EXPECT_EQ(tensor1_from_ffmpeg.sizes(), std::vector<long>({3, 270, 480}));
  EXPECT_TRUE(
      torch::allclose(tensor1_from_our_decoder, tensor1_from_ffmpeg, 0.1, 20));

  if (FLAGS_dump_frames_for_debugging) {
    dump_tensor_to_disk(tensor0_from_our_decoder, "tensor0FromOurDecoder.pt");
    dump_tensor_to_disk(tensor1_from_our_decoder, "tensor1FromOurDecoder.pt");
  }
}

TEST_P(SingleStreamDecoderTest, DecodesFramesInABatchInNCHW) {
  std::string path = get_resource_path("nasa_13013.mp4");
  std::unique_ptr<SingleStreamDecoder> our_decoder =
      create_decoder_from_path(path, GetParam());
  our_decoder->scan_file_and_update_metadata_and_index();
  int best_video_stream_index =
      *our_decoder->get_container_metadata().best_video_stream_index;
  std::vector<Transform*> transforms;
  our_decoder->add_video_stream(best_video_stream_index, transforms);
  // Frame with index 180 corresponds to timestamp 6.006.
  auto frame_indices = to_stable_tensor(torch::tensor({0, 180}));
  auto output = our_decoder->get_frames_at_indices(frame_indices);
  auto tensor = to_aten_tensor(output.data);
  EXPECT_EQ(tensor.sizes(), std::vector<long>({2, 3, 270, 480}));

  torch::Tensor tensor0_from_ffmpeg =
      read_tensor_from_disk("nasa_13013.mp4.stream3.frame000000.pt");
  torch::Tensor tensor_time6_from_ffmpeg =
      read_tensor_from_disk("nasa_13013.mp4.time6.000000.pt");

  EXPECT_TRUE(torch::equal(tensor[0], tensor0_from_ffmpeg));
  EXPECT_TRUE(torch::equal(tensor[1], tensor_time6_from_ffmpeg));
}

TEST_P(SingleStreamDecoderTest, DecodesFramesInABatchInNHWC) {
  std::string path = get_resource_path("nasa_13013.mp4");
  std::unique_ptr<SingleStreamDecoder> our_decoder =
      create_decoder_from_path(path, GetParam());
  our_decoder->scan_file_and_update_metadata_and_index();
  int best_video_stream_index =
      *our_decoder->get_container_metadata().best_video_stream_index;
  VideoStreamOptions video_stream_options;
  video_stream_options.dimension_order = "NHWC";
  std::vector<Transform*> transforms;
  our_decoder->add_video_stream(
      best_video_stream_index, transforms, video_stream_options);
  // Frame with index 180 corresponds to timestamp 6.006.
  auto frame_indices = to_stable_tensor(torch::tensor({0, 180}));
  auto output = our_decoder->get_frames_at_indices(frame_indices);
  auto tensor = to_aten_tensor(output.data);
  EXPECT_EQ(tensor.sizes(), std::vector<long>({2, 270, 480, 3}));

  torch::Tensor tensor0_from_ffmpeg =
      read_tensor_from_disk("nasa_13013.mp4.stream3.frame000000.pt");
  torch::Tensor tensor_time6_from_ffmpeg =
      read_tensor_from_disk("nasa_13013.mp4.time6.000000.pt");

  tensor = tensor.permute({0, 3, 1, 2});
  EXPECT_TRUE(torch::equal(tensor[0], tensor0_from_ffmpeg));
  EXPECT_TRUE(torch::equal(tensor[1], tensor_time6_from_ffmpeg));
}

TEST_P(SingleStreamDecoderTest, SeeksCloseToEof) {
  std::string path = get_resource_path("nasa_13013.mp4");
  std::unique_ptr<SingleStreamDecoder> our_decoder =
      create_decoder_from_path(path, GetParam());
  std::vector<Transform*> transforms;
  our_decoder->add_video_stream(-1, transforms);
  our_decoder->set_cursor_pts_in_seconds(388388. / 30'000);
  auto output = our_decoder->get_next_frame();
  EXPECT_EQ(output.pts_seconds, 388'388. / 30'000);
  output = our_decoder->get_next_frame();
  EXPECT_EQ(output.pts_seconds, 389'389. / 30'000);
  EXPECT_THROW(our_decoder->get_next_frame(), std::exception);
}

TEST_P(SingleStreamDecoderTest, GetsFramePlayedAtTimestamp) {
  std::string path = get_resource_path("nasa_13013.mp4");
  std::unique_ptr<SingleStreamDecoder> our_decoder =
      create_decoder_from_path(path, GetParam());
  std::vector<Transform*> transforms;
  our_decoder->add_video_stream(-1, transforms);
  auto output = our_decoder->get_frame_played_at(6.006);
  EXPECT_EQ(output.pts_seconds, 6.006);
  // The frame's duration is 0.033367 according to ffprobe,
  // so the next frame is played at timestamp=6.039367.
  const double k_next_frame_pts = 6.039366666666667;
  // The frame that is played a microsecond before the next frame is still
  // the previous frame.
  output = our_decoder->get_frame_played_at(k_next_frame_pts - 1e-6);
  EXPECT_EQ(output.pts_seconds, 6.006);
  // The frame that is played at the exact pts of the frame is the next
  // frame.
  output = our_decoder->get_frame_played_at(k_next_frame_pts);
  EXPECT_EQ(output.pts_seconds, k_next_frame_pts);

  // This is the timestamp of the last frame in this video.
  constexpr double k_pts_of_last_frame_in_video_stream = 389'389. / 30'000;
  constexpr double k_duration_of_last_frame_in_video_stream = 1'001. / 30'000;
  constexpr double k_pts_plus_duration_of_last_frame =
      k_pts_of_last_frame_in_video_stream +
      k_duration_of_last_frame_in_video_stream;
  // Sanity check: make sure duration is strictly positive.
  EXPECT_GT(
      k_pts_plus_duration_of_last_frame, k_pts_of_last_frame_in_video_stream);
  output = our_decoder->get_frame_played_at(
      k_pts_plus_duration_of_last_frame - 1e-6);
  EXPECT_EQ(output.pts_seconds, k_pts_of_last_frame_in_video_stream);
}

TEST_P(SingleStreamDecoderTest, SeeksToFrameWithSpecificPts) {
  std::string path = get_resource_path("nasa_13013.mp4");
  std::unique_ptr<SingleStreamDecoder> our_decoder =
      create_decoder_from_path(path, GetParam());
  std::vector<Transform*> transforms;
  our_decoder->add_video_stream(-1, transforms);
  our_decoder->set_cursor_pts_in_seconds(6.0);
  auto output = our_decoder->get_next_frame();
  torch::Tensor tensor6_from_our_decoder = to_aten_tensor(output.data);
  EXPECT_EQ(output.pts_seconds, 180'180. / 30'000);
  torch::Tensor tensor6_from_ffmpeg =
      read_tensor_from_disk("nasa_13013.mp4.time6.000000.pt");
  EXPECT_TRUE(torch::equal(tensor6_from_our_decoder, tensor6_from_ffmpeg));
  EXPECT_EQ(our_decoder->get_decode_stats().num_seeks_attempted, 1);
  // lastDecodedAvFramePts_ is initialized to INT64_MIN, so the
  // first seek is always performed even though timestamp=6 and timestamp=0
  // share the same keyframe.
  EXPECT_EQ(our_decoder->get_decode_stats().num_seeks_skipped, 0);
  // There are about 180 packets/frames between timestamp=0 and timestamp=6 at
  // ~30 fps.
  EXPECT_GT(our_decoder->get_decode_stats().num_packets_read, 180);
  EXPECT_GT(our_decoder->get_decode_stats().num_packets_sent_to_decoder, 180);

  our_decoder->set_cursor_pts_in_seconds(6.1);
  output = our_decoder->get_next_frame();
  auto tensor61_from_our_decoder = to_aten_tensor(output.data);
  EXPECT_EQ(output.pts_seconds, 183'183. / 30'000);
  torch::Tensor tensor61_from_ffmpeg =
      read_tensor_from_disk("nasa_13013.mp4.time6.100000.pt");
  EXPECT_TRUE(torch::equal(tensor61_from_our_decoder, tensor61_from_ffmpeg));
  EXPECT_EQ(our_decoder->get_decode_stats().num_seeks_attempted, 1);
  // We skipped the seek since timestamp=6 and timestamp=6.1 share the same
  // keyframe.
  EXPECT_EQ(our_decoder->get_decode_stats().num_seeks_skipped, 1);
  // If we had seeked, we would have gone to frame=0 (because that was the key
  // frame before timestamp=6.1). Because we skipped that seek the number of
  // packets we send to the decoder is minimal. This is partly why torchvision
  // is slower than decord. In fact we are more efficient than decord because we
  // rely on FFMPEG's key frame index instead of reading the entire file
  // ourselves. ^_^
  EXPECT_LT(our_decoder->get_decode_stats().num_packets_read, 10);
  EXPECT_LT(our_decoder->get_decode_stats().num_packets_sent_to_decoder, 10);

  our_decoder->set_cursor_pts_in_seconds(10.0);
  output = our_decoder->get_next_frame();
  auto tensor10_from_our_decoder = to_aten_tensor(output.data);
  EXPECT_EQ(output.pts_seconds, 300'300. / 30'000);
  torch::Tensor tensor10_from_ffmpeg =
      read_tensor_from_disk("nasa_13013.mp4.time10.000000.pt");
  EXPECT_TRUE(torch::equal(tensor10_from_our_decoder, tensor10_from_ffmpeg));
  EXPECT_EQ(our_decoder->get_decode_stats().num_seeks_attempted, 1);
  // We cannot skip a seek here because timestamp=10 has a different keyframe
  // than timestamp=6.
  EXPECT_EQ(our_decoder->get_decode_stats().num_seeks_skipped, 0);
  // The keyframe is at timestamp=8. So we seek to there and decode until
  // timestamp=10. There are about 60 packets/frames between timestamp=8 and
  // timestamp=10 at ~30 fps.
  EXPECT_GT(our_decoder->get_decode_stats().num_packets_read, 60);
  EXPECT_GT(our_decoder->get_decode_stats().num_packets_sent_to_decoder, 60);

  our_decoder->set_cursor_pts_in_seconds(6.0);
  output = our_decoder->get_next_frame();
  tensor6_from_our_decoder = to_aten_tensor(output.data);
  EXPECT_EQ(output.pts_seconds, 180'180. / 30'000);
  EXPECT_TRUE(torch::equal(tensor6_from_our_decoder, tensor6_from_ffmpeg));
  EXPECT_EQ(our_decoder->get_decode_stats().num_seeks_attempted, 1);
  // We cannot skip a seek here because timestamp=6 has a different keyframe
  // than timestamp=10.
  EXPECT_EQ(our_decoder->get_decode_stats().num_seeks_skipped, 0);
  // There are about 180 packets/frames between timestamp=0 and timestamp=6 at
  // ~30 fps.
  EXPECT_GT(our_decoder->get_decode_stats().num_packets_read, 180);
  EXPECT_GT(our_decoder->get_decode_stats().num_packets_sent_to_decoder, 180);

  constexpr double k_pts_of_last_frame_in_video_stream =
      389'389. / 30'000; // ~12.9
  our_decoder->set_cursor_pts_in_seconds(k_pts_of_last_frame_in_video_stream);
  output = our_decoder->get_next_frame();
  auto tensor7_from_our_decoder = to_aten_tensor(output.data);
  EXPECT_EQ(output.pts_seconds, 389'389. / 30'000);
  torch::Tensor tensor7_from_ffmpeg =
      read_tensor_from_disk("nasa_13013.mp4.time12.979633.pt");
  EXPECT_TRUE(torch::equal(tensor7_from_our_decoder, tensor7_from_ffmpeg));
  EXPECT_EQ(our_decoder->get_decode_stats().num_seeks_attempted, 1);
  // We cannot skip a seek here because timestamp=6 has a different keyframe
  // than timestamp=12.9.
  EXPECT_EQ(our_decoder->get_decode_stats().num_seeks_skipped, 0);
  // There are about 150 packets/frames between timestamp=8 and timestamp=12.9
  // at ~30 fps.
  EXPECT_GE(our_decoder->get_decode_stats().num_packets_read, 150);
  EXPECT_GE(our_decoder->get_decode_stats().num_packets_sent_to_decoder, 150);

  if (FLAGS_dump_frames_for_debugging) {
    dump_tensor_to_disk(tensor7_from_ffmpeg, "tensor7FromFFMPEG.pt");
    dump_tensor_to_disk(tensor7_from_our_decoder, "tensor7FromOurDecoder.pt");
  }
}

TEST_P(SingleStreamDecoderTest, PreAllocatedTensorFilterGraph) {
  std::string path = get_resource_path("nasa_13013.mp4");
  auto pre_allocated_output_tensor =
      to_stable_tensor(torch::empty({270, 480, 3}, {torch::kUInt8}));

  std::unique_ptr<SingleStreamDecoder> our_decoder =
      SingleStreamDecoderTest::create_decoder_from_path(path, GetParam());
  our_decoder->scan_file_and_update_metadata_and_index();
  int best_video_stream_index =
      *our_decoder->get_container_metadata().best_video_stream_index;
  VideoStreamOptions video_stream_options;
  video_stream_options.color_conversion_library =
      ColorConversionLibrary::FILTERGRAPH;
  std::vector<Transform*> transforms;
  our_decoder->add_video_stream(
      best_video_stream_index, transforms, video_stream_options);
  auto output =
      our_decoder->get_frame_at_index_internal(0, pre_allocated_output_tensor);
  EXPECT_EQ(output.data.data_ptr(), pre_allocated_output_tensor.data_ptr());
}

TEST_P(SingleStreamDecoderTest, PreAllocatedTensorSwscale) {
  std::string path = get_resource_path("nasa_13013.mp4");
  auto pre_allocated_output_tensor =
      to_stable_tensor(torch::empty({270, 480, 3}, {torch::kUInt8}));

  std::unique_ptr<SingleStreamDecoder> our_decoder =
      SingleStreamDecoderTest::create_decoder_from_path(path, GetParam());
  our_decoder->scan_file_and_update_metadata_and_index();
  int best_video_stream_index =
      *our_decoder->get_container_metadata().best_video_stream_index;
  VideoStreamOptions video_stream_options;
  video_stream_options.color_conversion_library =
      ColorConversionLibrary::SWSCALE;
  std::vector<Transform*> transforms;
  our_decoder->add_video_stream(
      best_video_stream_index, transforms, video_stream_options);
  auto output =
      our_decoder->get_frame_at_index_internal(0, pre_allocated_output_tensor);
  EXPECT_EQ(output.data.data_ptr(), pre_allocated_output_tensor.data_ptr());
}

TEST_P(SingleStreamDecoderTest, GetAudioMetadata) {
  std::string path = get_resource_path("nasa_13013.mp4.audio.mp3");
  std::unique_ptr<SingleStreamDecoder> decoder =
      create_decoder_from_path(path, GetParam());
  ContainerMetadata metadata = decoder->get_container_metadata();
  EXPECT_EQ(metadata.num_audio_streams, 1);
  EXPECT_EQ(metadata.num_video_streams, 0);
  EXPECT_EQ(metadata.all_stream_metadata.size(), 1);

  const auto& audio_stream = metadata.all_stream_metadata[0];
  EXPECT_EQ(audio_stream.media_type, AVMEDIA_TYPE_AUDIO);
  EXPECT_NEAR(*audio_stream.duration_seconds_from_header, 13.25, 1e-1);
}

INSTANTIATE_TEST_SUITE_P(
    FromFileAndMemory,
    SingleStreamDecoderTest,
    testing::Bool());

} // namespace facebook::torchcodec
