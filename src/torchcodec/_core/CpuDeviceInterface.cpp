// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "CpuDeviceInterface.h"

namespace facebook::torchcodec {
namespace {

static bool g_cpu = registerDeviceInterface(
    DeviceInterfaceKey(torch::kCPU),
    [](const torch::Device& device) { return new CpuDeviceInterface(device); });

} // namespace

CpuDeviceInterface::CpuDeviceInterface(const torch::Device& device)
    : DeviceInterface(device) {
  TORCH_CHECK(g_cpu, "CpuDeviceInterface was not registered!");
  TORCH_CHECK(
      device_.type() == torch::kCPU, "Unsupported device: ", device_.str());
}

void CpuDeviceInterface::initialize(
    const AVStream* avStream,
    [[maybe_unused]] const UniqueDecodingAVFormatContext& avFormatCtx,
    const SharedAVCodecContext& codecContext) {
  TORCH_CHECK(avStream != nullptr, "avStream is null");
  codecContext_ = codecContext;
  timeBase_ = avStream->time_base;
}

void CpuDeviceInterface::initializeVideo(
    const VideoStreamOptions& videoStreamOptions,
    const std::vector<std::unique_ptr<Transform>>& transforms,
    const std::optional<FrameDims>& resizedOutputDims) {
  avMediaType_ = AVMEDIA_TYPE_VIDEO;
  videoStreamOptions_ = videoStreamOptions;
  resizedOutputDims_ = resizedOutputDims;

  // We can use swscale when we have a single resize transform.
  // With a single resize, we use swscale twice:
  // first for color conversion (YUV->RGB24), then for resize in RGB24 space.
  //
  // Note that this means swscale will not support the case of having several,
  // back-to-back resizes or other transforms.
  //
  // We calculate this value during initialization but we don't refer to it
  // until getColorConversionLibrary() is called. Calculating this value during
  // initialization saves us from having to save all of the transforms.
  areTransformsSwScaleCompatible_ = transforms.empty() ||
      (transforms.size() == 1 && transforms[0]->isResize());

  // Note that we do not expose this capability in the public API, only through
  // the core API.
  //
  // Same as above, we calculate this value during initialization and refer to
  // it in getColorConversionLibrary().
  userRequestedSwScale_ = videoStreamOptions_.colorConversionLibrary ==
      ColorConversionLibrary::SWSCALE;

  // We can only use swscale when we have a single resize transform. Note that
  // we actually decide on whether or not to actually use swscale at the last
  // possible moment, when we actually convert the frame. This is because we
  // need to know the actual frame dimensions.
  if (transforms.size() == 1 && transforms[0]->isResize()) {
    auto resize = dynamic_cast<ResizeTransform*>(transforms[0].get());
    TORCH_CHECK(resize != nullptr, "ResizeTransform expected but not found!");
    swsFlags_ = resize->getSwsFlags();
  }

  // If we have any transforms, replace filters_ with the filter strings from
  // the transforms. As noted above, we decide between swscale and filtergraph
  // when we actually decode a frame.
  std::stringstream filters;
  bool first = true;
  for (const auto& transform : transforms) {
    if (!first) {
      filters << ",";
    }
    filters << transform->getFilterGraphCpu();
    first = false;
  }
  if (!transforms.empty()) {
    // Note [Transform and Format Conversion Order]
    // We have to ensure that all user filters happen AFTER the explicit format
    // conversion. That is, we want the filters to be applied in RGB24, not the
    // pixel format of the input frame.
    //
    // The ouput frame will always be in RGB24, as we specify the sink node with
    // AV_PIX_FORMAT_RGB24. Filtergraph will automatically insert a filter
    // conversion to ensure the output frame matches the pixel format
    // specified in the sink. But by default, it will insert it after the user
    // filters. We need an explicit format conversion to get the behavior we
    // want.
    filters_ = "format=rgb24," + filters.str();
  }

  initialized_ = true;
}

void CpuDeviceInterface::initializeAudio(
    const AudioStreamOptions& audioStreamOptions) {
  avMediaType_ = AVMEDIA_TYPE_AUDIO;
  audioStreamOptions_ = audioStreamOptions;
  initialized_ = true;
}

ColorConversionLibrary CpuDeviceInterface::getColorConversionLibrary(
    const FrameDims& outputDims) const {
  // swscale requires widths to be multiples of 32:
  // https://stackoverflow.com/questions/74351955/turn-off-sw-scale-conversion-to-planar-yuv-32-byte-alignment-requirements
  bool isWidthSwScaleCompatible = (outputDims.width % 32) == 0;

  // We want to use swscale for color conversion if possible because it is
  // faster than filtergraph. The following are the conditions we need to meet
  // to use it.
  //
  // Note that we treat the transform limitation differently from the width
  // limitation. That is, we consider the transforms being compatible with
  // swscale as a hard requirement. If the transforms are not compatiable,
  // then we will end up not applying the transforms, and that is wrong.
  //
  // The width requirement, however, is a soft requirement. Even if we don't
  // meet it, we let the user override it. We have tests that depend on this
  // behavior. Since we don't expose the ability to choose swscale or
  // filtergraph in our public API, this is probably okay. It's also the only
  // way that we can be certain we are testing one versus the other.
  if (areTransformsSwScaleCompatible_ &&
      (userRequestedSwScale_ || isWidthSwScaleCompatible)) {
    return ColorConversionLibrary::SWSCALE;
  } else {
    return ColorConversionLibrary::FILTERGRAPH;
  }
}

void CpuDeviceInterface::convertAVFrameToFrameOutput(
    UniqueAVFrame& avFrame,
    FrameOutput& frameOutput,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  TORCH_CHECK(initialized_, "CpuDeviceInterface was not initialized.");

  if (avMediaType_ == AVMEDIA_TYPE_AUDIO) {
    convertAudioAVFrameToFrameOutput(avFrame, frameOutput);
  } else {
    convertVideoAVFrameToFrameOutput(
        avFrame, frameOutput, preAllocatedOutputTensor);
  }
}

// Note [preAllocatedOutputTensor with swscale and filtergraph]:
// Callers may pass a pre-allocated tensor, where the output.data tensor will
// be stored. This parameter is honored in any case, but it only leads to a
// speed-up when swscale is used. With swscale, we can tell ffmpeg to place the
// decoded frame directly into `preAllocatedtensor.data_ptr()`. We haven't yet
// found a way to do that with filtegraph.
// TODO: Figure out whether that's possible!
// Dimension order of the preAllocatedOutputTensor must be HWC, regardless of
// `dimension_order` parameter. It's up to callers to re-shape it if needed.
void CpuDeviceInterface::convertVideoAVFrameToFrameOutput(
    UniqueAVFrame& avFrame,
    FrameOutput& frameOutput,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  // Note that we ignore the dimensions from the metadata; we don't even bother
  // storing them. The resized dimensions take priority. If we don't have any,
  // then we use the dimensions from the actual decoded frame. We use the actual
  // decoded frame and not the metadata for two reasons:
  //
  //   1. Metadata may be wrong. If we access to more accurate information, we
  //      should use it.
  //   2. Video streams can have variable resolution. This fact is not captured
  //      in the stream  metadata.
  //
  // Both cases cause problems for our batch APIs, as we allocate
  // FrameBatchOutputs based on the the stream metadata. But single-frame APIs
  // can still work in such situations, so they should.
  auto outputDims =
      resizedOutputDims_.value_or(FrameDims(avFrame->height, avFrame->width));

  if (preAllocatedOutputTensor.has_value()) {
    auto shape = preAllocatedOutputTensor.value().sizes();
    TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == outputDims.height) &&
            (shape[1] == outputDims.width) && (shape[2] == 3),
        "Expected pre-allocated tensor of shape ",
        outputDims.height,
        "x",
        outputDims.width,
        "x3, got ",
        shape);
  }

  auto colorConversionLibrary = getColorConversionLibrary(outputDims);
  torch::Tensor outputTensor;

  if (colorConversionLibrary == ColorConversionLibrary::SWSCALE) {
    outputTensor = preAllocatedOutputTensor.value_or(
        allocateEmptyHWCTensor(outputDims, torch::kCPU));

    int resultHeight =
        convertAVFrameToTensorUsingSwScale(avFrame, outputTensor, outputDims);

    // If this check failed, it would mean that the frame wasn't reshaped to
    // the expected height.
    // TODO: Can we do the same check for width?
    TORCH_CHECK(
        resultHeight == outputDims.height,
        "resultHeight != outputDims.height: ",
        resultHeight,
        " != ",
        outputDims.height);

    frameOutput.data = outputTensor;
  } else if (colorConversionLibrary == ColorConversionLibrary::FILTERGRAPH) {
    outputTensor = convertAVFrameToTensorUsingFilterGraph(avFrame, outputDims);

    // Similarly to above, if this check fails it means the frame wasn't
    // reshaped to its expected dimensions by filtergraph.
    auto shape = outputTensor.sizes();
    TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == outputDims.height) &&
            (shape[1] == outputDims.width) && (shape[2] == 3),
        "Expected output tensor of shape ",
        outputDims.height,
        "x",
        outputDims.width,
        "x3, got ",
        shape);

    if (preAllocatedOutputTensor.has_value()) {
      // We have already validated that preAllocatedOutputTensor and
      // outputTensor have the same shape.
      preAllocatedOutputTensor.value().copy_(outputTensor);
      frameOutput.data = preAllocatedOutputTensor.value();
    } else {
      frameOutput.data = outputTensor;
    }
  } else {
    TORCH_CHECK(
        false,
        "Invalid color conversion library: ",
        static_cast<int>(colorConversionLibrary));
  }
}

int CpuDeviceInterface::convertAVFrameToTensorUsingSwScale(
    const UniqueAVFrame& avFrame,
    torch::Tensor& outputTensor,
    const FrameDims& outputDims) {
  enum AVPixelFormat frameFormat =
      static_cast<enum AVPixelFormat>(avFrame->format);

  bool needsResize =
      (avFrame->height != outputDims.height ||
       avFrame->width != outputDims.width);

  // We need to compare the current frame context with our previous frame
  // context. If they are different, then we need to re-create our colorspace
  // conversion objects. We create our colorspace conversion objects late so
  // that we don't have to depend on the unreliable metadata in the header.
  // And we sometimes re-create them because it's possible for frame
  // resolution to change mid-stream. Finally, we want to reuse the colorspace
  // conversion objects as much as possible for performance reasons.
  SwsFrameContext swsFrameContext(
      avFrame->width,
      avFrame->height,
      frameFormat,
      needsResize ? avFrame->width : outputDims.width,
      needsResize ? avFrame->height : outputDims.height);

  if (!swsContext_ || prevSwsFrameContext_ != swsFrameContext) {
    swsContext_ = createSwsContext(
        swsFrameContext,
        avFrame->colorspace,

        // See [Transform and Format Conversion Order] for more on the output
        // pixel format.
        /*outputFormat=*/AV_PIX_FMT_RGB24,

        // No flags for color conversion. When resizing is needed, we use a
        // separate swscale context with the appropriate resize flags.
        /*swsFlags=*/0);
    prevSwsFrameContext_ = swsFrameContext;
  }

  // When resizing is needed, we do sws_scale twice: first convert to RGB24 at
  // original resolution, then resize in RGB24 space. This ensures transforms
  // happen in the output color space (RGB24) rather than the input color space
  // (YUV).
  //
  // When no resize is needed, we do color conversion directly into the output
  // tensor.

  torch::Tensor colorConvertedTensor = needsResize
      ? allocateEmptyHWCTensor(
            FrameDims(avFrame->height, avFrame->width), torch::kCPU)
      : outputTensor;

  uint8_t* colorConvertedPointers[4] = {
      colorConvertedTensor.data_ptr<uint8_t>(), nullptr, nullptr, nullptr};
  int colorConvertedWidth = colorConvertedTensor.sizes()[1];
  int colorConvertedLinesizes[4] = {colorConvertedWidth * 3, 0, 0, 0};

  int colorConvertedHeight = sws_scale(
      swsContext_.get(),
      avFrame->data,
      avFrame->linesize,
      0,
      avFrame->height,
      colorConvertedPointers,
      colorConvertedLinesizes);

  TORCH_CHECK(
      colorConvertedHeight == avFrame->height,
      "Color conversion swscale pass failed: colorConvertedHeight != avFrame->height: ",
      colorConvertedHeight,
      " != ",
      avFrame->height);

  if (needsResize) {
    // Use cached swscale context for resizing, similar to the color conversion
    // context caching above.
    SwsFrameContext resizeSwsFrameContext(
        avFrame->width,
        avFrame->height,
        AV_PIX_FMT_RGB24,
        outputDims.width,
        outputDims.height);

    if (!resizeSwsContext_ ||
        prevResizeSwsFrameContext_ != resizeSwsFrameContext) {
      resizeSwsContext_ = createSwsContext(
          resizeSwsFrameContext,
          AVCOL_SPC_RGB,
          /*outputFormat=*/AV_PIX_FMT_RGB24,
          /*swsFlags=*/swsFlags_);
      prevResizeSwsFrameContext_ = resizeSwsFrameContext;
    }

    uint8_t* srcPointers[4] = {
        colorConvertedTensor.data_ptr<uint8_t>(), nullptr, nullptr, nullptr};
    int srcLinesizes[4] = {avFrame->width * 3, 0, 0, 0};

    uint8_t* dstPointers[4] = {
        outputTensor.data_ptr<uint8_t>(), nullptr, nullptr, nullptr};
    int expectedOutputWidth = outputTensor.sizes()[1];
    int dstLinesizes[4] = {expectedOutputWidth * 3, 0, 0, 0};

    colorConvertedHeight = sws_scale(
        resizeSwsContext_.get(),
        srcPointers,
        srcLinesizes,
        0,
        avFrame->height,
        dstPointers,
        dstLinesizes);
  }

  return colorConvertedHeight;
}

torch::Tensor CpuDeviceInterface::convertAVFrameToTensorUsingFilterGraph(
    const UniqueAVFrame& avFrame,
    const FrameDims& outputDims) {
  enum AVPixelFormat avFrameFormat =
      static_cast<enum AVPixelFormat>(avFrame->format);

  FiltersContext filtersContext(
      avFrame->width,
      avFrame->height,
      avFrameFormat,
      avFrame->sample_aspect_ratio,
      outputDims.width,
      outputDims.height,
      /*outputFormat=*/AV_PIX_FMT_RGB24,
      filters_,
      timeBase_);

  if (!filterGraph_ || prevFiltersContext_ != filtersContext) {
    filterGraph_ =
        std::make_unique<FilterGraph>(filtersContext, videoStreamOptions_);
    prevFiltersContext_ = std::move(filtersContext);
  }
  return rgbAVFrameToTensor(filterGraph_->convert(avFrame));
}

void CpuDeviceInterface::convertAudioAVFrameToFrameOutput(
    UniqueAVFrame& srcAVFrame,
    FrameOutput& frameOutput) {
  AVSampleFormat srcSampleFormat =
      static_cast<AVSampleFormat>(srcAVFrame->format);
  AVSampleFormat outSampleFormat = AV_SAMPLE_FMT_FLTP;

  int srcSampleRate = srcAVFrame->sample_rate;
  int outSampleRate = audioStreamOptions_.sampleRate.value_or(srcSampleRate);

  int srcNumChannels = getNumChannels(codecContext_);
  TORCH_CHECK(
      srcNumChannels == getNumChannels(srcAVFrame),
      "The frame has ",
      getNumChannels(srcAVFrame),
      " channels, expected ",
      srcNumChannels,
      ". If you are hitting this, it may be because you are using "
      "a buggy FFmpeg version. FFmpeg4 is known to fail here in some "
      "valid scenarios. Try to upgrade FFmpeg?");
  int outNumChannels = audioStreamOptions_.numChannels.value_or(srcNumChannels);

  bool mustConvert =
      (srcSampleFormat != outSampleFormat || srcSampleRate != outSampleRate ||
       srcNumChannels != outNumChannels);

  UniqueAVFrame convertedAVFrame;
  if (mustConvert) {
    if (!swrContext_) {
      swrContext_.reset(createSwrContext(
          srcSampleFormat,
          outSampleFormat,
          srcSampleRate,
          outSampleRate,
          srcAVFrame,
          outNumChannels));
    }

    convertedAVFrame = convertAudioAVFrameSamples(
        swrContext_,
        srcAVFrame,
        outSampleFormat,
        outSampleRate,
        outNumChannels);
  }
  const UniqueAVFrame& avFrame = mustConvert ? convertedAVFrame : srcAVFrame;

  AVSampleFormat format = static_cast<AVSampleFormat>(avFrame->format);
  TORCH_CHECK(
      format == outSampleFormat,
      "Something went wrong, the frame didn't get converted to the desired format. ",
      "Desired format = ",
      av_get_sample_fmt_name(outSampleFormat),
      "source format = ",
      av_get_sample_fmt_name(format));

  int numChannels = getNumChannels(avFrame);
  TORCH_CHECK(
      numChannels == outNumChannels,
      "Something went wrong, the frame didn't get converted to the desired ",
      "number of channels = ",
      outNumChannels,
      ". Got ",
      numChannels,
      " instead.");

  auto numSamples = avFrame->nb_samples;

  frameOutput.data = torch::empty({numChannels, numSamples}, torch::kFloat32);

  if (numSamples > 0) {
    uint8_t* outputChannelData =
        static_cast<uint8_t*>(frameOutput.data.data_ptr());
    auto numBytesPerChannel = numSamples * av_get_bytes_per_sample(format);
    for (auto channel = 0; channel < numChannels;
         ++channel, outputChannelData += numBytesPerChannel) {
      std::memcpy(
          outputChannelData,
          avFrame->extended_data[channel],
          numBytesPerChannel);
    }
  }
}

std::optional<torch::Tensor> CpuDeviceInterface::maybeFlushAudioBuffers() {
  // When sample rate conversion is involved, swresample buffers some of the
  // samples in-between calls to swr_convert (see the libswresample docs).
  // That's because the last few samples in a given frame require future
  // samples from the next frame to be properly converted. This function
  // flushes out the samples that are stored in swresample's buffers.
  if (!swrContext_) {
    return std::nullopt;
  }
  auto numRemainingSamples = // this is an upper bound
      swr_get_out_samples(swrContext_.get(), 0);

  if (numRemainingSamples == 0) {
    return std::nullopt;
  }

  int numChannels =
      audioStreamOptions_.numChannels.value_or(getNumChannels(codecContext_));
  torch::Tensor lastSamples =
      torch::empty({numChannels, numRemainingSamples}, torch::kFloat32);

  std::vector<uint8_t*> outputBuffers(numChannels);
  for (auto i = 0; i < numChannels; i++) {
    outputBuffers[i] = static_cast<uint8_t*>(lastSamples[i].data_ptr());
  }

  auto actualNumRemainingSamples = swr_convert(
      swrContext_.get(), outputBuffers.data(), numRemainingSamples, nullptr, 0);

  return lastSamples.narrow(
      /*dim=*/1, /*start=*/0, /*length=*/actualNumRemainingSamples);
}

std::string CpuDeviceInterface::getDetails() {
  return std::string("CPU Device Interface.");
}

} // namespace facebook::torchcodec
