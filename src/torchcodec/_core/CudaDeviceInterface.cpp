#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>
#include <mutex>

#include "src/torchcodec/_core/Cache.h"
#include "src/torchcodec/_core/CudaDeviceInterface.h"
#include "src/torchcodec/_core/FFMPEGCommon.h"

extern "C" {
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {
namespace {

static bool g_cuda = registerDeviceInterface(
    DeviceInterfaceKey(torch::kCUDA),
    [](const torch::Device& device) {
      return new CudaDeviceInterface(device);
    });

// We reuse cuda contexts across VideoDeoder instances. This is because
// creating a cuda context is expensive. The cache mechanism is as follows:
// 1. There is a cache of size MAX_CONTEXTS_PER_GPU_IN_CACHE cuda contexts for
//    each GPU.
// 2. When we destroy a SingleStreamDecoder instance we release the cuda context
// to
//    the cache if the cache is not full.
// 3. When we create a SingleStreamDecoder instance we try to get a cuda context
// from
//    the cache. If the cache is empty we create a new cuda context.

// Set to -1 to have an infinitely sized cache. Set it to 0 to disable caching.
// Set to a positive number to have a cache of that size.
const int MAX_CONTEXTS_PER_GPU_IN_CACHE = -1;
PerGpuCache<AVBufferRef, Deleterp<AVBufferRef, void, av_buffer_unref>>
    g_cached_hw_device_ctxs(MAX_CUDA_GPUS, MAX_CONTEXTS_PER_GPU_IN_CACHE);

int getFlagsAVHardwareDeviceContextCreate() {
// 58.26.100 introduced the concept of reusing the existing cuda context
// which is much faster and lower memory than creating a new cuda context.
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(58, 26, 100)
  return AV_CUDA_USE_CURRENT_CONTEXT;
#else
  return 0;
#endif
}

UniqueAVBufferRef getHardwareDeviceContext(const torch::Device& device) {
  enum AVHWDeviceType type = av_hwdevice_find_type_by_name("cuda");
  TORCH_CHECK(type != AV_HWDEVICE_TYPE_NONE, "Failed to find cuda device");
  int deviceIndex = getDeviceIndex(device);

  UniqueAVBufferRef hardwareDeviceCtx = g_cached_hw_device_ctxs.get(device);
  if (hardwareDeviceCtx) {
    return hardwareDeviceCtx;
  }

  // Create hardware device context
  c10::cuda::CUDAGuard deviceGuard(device);
  // We set the device because we may be called from a different thread than
  // the one that initialized the cuda context.
  TORCH_CHECK(
      cudaSetDevice(deviceIndex) == cudaSuccess, "Failed to set CUDA device");
  AVBufferRef* hardwareDeviceCtxRaw = nullptr;
  std::string deviceOrdinal = std::to_string(deviceIndex);

  int err = av_hwdevice_ctx_create(
      &hardwareDeviceCtxRaw,
      type,
      deviceOrdinal.c_str(),
      nullptr,
      getFlagsAVHardwareDeviceContextCreate());

  if (err < 0) {
    /* clang-format off */
    TORCH_CHECK(
        false,
        "Failed to create specified HW device. This typically happens when ",
        "your installed FFmpeg doesn't support CUDA (see ",
        "https://github.com/pytorch/torchcodec#installing-cuda-enabled-torchcodec",
        "). FFmpeg error: ", getFFMPEGErrorStringFromErrorCode(err));
    /* clang-format on */
  }

  return UniqueAVBufferRef(hardwareDeviceCtxRaw);
}

} // namespace

CudaDeviceInterface::CudaDeviceInterface(const torch::Device& device)
    : DeviceInterface(device) {
  TORCH_CHECK(g_cuda, "CudaDeviceInterface was not registered!");
  TORCH_CHECK(
      device_.type() == torch::kCUDA, "Unsupported device: ", device_.str());

  initializeCudaContextWithPytorch(device_);

  hardwareDeviceCtx_ = getHardwareDeviceContext(device_);
  nppCtx_ = getNppStreamContext(device_);
}

CudaDeviceInterface::~CudaDeviceInterface() {
  if (hardwareDeviceCtx_) {
    g_cached_hw_device_ctxs.addIfCacheHasCapacity(
        device_, std::move(hardwareDeviceCtx_));
  }
  returnNppStreamContextToCache(device_, std::move(nppCtx_));
}

void CudaDeviceInterface::initialize(
    const AVStream* avStream,
    const UniqueDecodingAVFormatContext& avFormatCtx,
    const SharedAVCodecContext& codecContext) {
  TORCH_CHECK(avStream != nullptr, "avStream is null");
  codecContext_ = codecContext;
  timeBase_ = avStream->time_base;

  // TODO: Ideally, we should keep all interface implementations independent.
  cpuInterface_ = createDeviceInterface(torch::kCPU);
  TORCH_CHECK(
      cpuInterface_ != nullptr, "Failed to create CPU device interface");
  cpuInterface_->initialize(avStream, avFormatCtx, codecContext);
  cpuInterface_->initializeVideo(
      VideoStreamOptions(),
      {},
      /*resizedOutputDims=*/std::nullopt);
}

void CudaDeviceInterface::initializeVideo(
    const VideoStreamOptions& videoStreamOptions,
    [[maybe_unused]] const std::vector<std::unique_ptr<Transform>>& transforms,
    [[maybe_unused]] const std::optional<FrameDims>& resizedOutputDims) {
  videoStreamOptions_ = videoStreamOptions;
}

void CudaDeviceInterface::registerHardwareDeviceWithCodec(
    AVCodecContext* codecContext) {
  TORCH_CHECK(
      hardwareDeviceCtx_, "Hardware device context has not been initialized");
  TORCH_CHECK(codecContext != nullptr, "codecContext is null");
  codecContext->hw_device_ctx = av_buffer_ref(hardwareDeviceCtx_.get());
}

UniqueAVFrame CudaDeviceInterface::maybeConvertAVFrameToNV12OrRGB24(
    UniqueAVFrame& avFrame) {
  // We need FFmpeg filters to handle those conversion cases which are not
  // directly implemented in CUDA or CPU device interface (in case of a
  // fallback).

  // Input frame is on CPU, we will just pass it to CPU device interface, so
  // skipping filters context as CPU device interface will handle everything for
  // us.
  if (avFrame->format != AV_PIX_FMT_CUDA) {
    return std::move(avFrame);
  }

  auto hwFramesCtx =
      reinterpret_cast<AVHWFramesContext*>(avFrame->hw_frames_ctx->data);
  TORCH_CHECK(
      hwFramesCtx != nullptr,
      "The AVFrame does not have a hw_frames_ctx. "
      "That's unexpected, please report this to the TorchCodec repo.");

  AVPixelFormat actualFormat = hwFramesCtx->sw_format;

  // If the frame is already in NV12 format, we don't need to do anything.
  if (actualFormat == AV_PIX_FMT_NV12) {
    return std::move(avFrame);
  }

  AVPixelFormat outputFormat;
  std::stringstream filters;

  unsigned version_int = avfilter_version();
  if (version_int < AV_VERSION_INT(8, 0, 103)) {
    // Color conversion support ('format=' option) was added to scale_cuda from
    // n5.0. With the earlier version of ffmpeg we have no choice but use CPU
    // filters. See:
    // https://github.com/FFmpeg/FFmpeg/commit/62dc5df941f5e196164c151691e4274195523e95
    outputFormat = AV_PIX_FMT_RGB24;

    auto actualFormatName = av_get_pix_fmt_name(actualFormat);
    TORCH_CHECK(
        actualFormatName != nullptr,
        "The actual format of a frame is unknown to FFmpeg. "
        "That's unexpected, please report this to the TorchCodec repo.");

    filters << "hwdownload,format=" << actualFormatName;
  } else {
    // Actual output color format will be set via filter options
    outputFormat = AV_PIX_FMT_CUDA;

    filters << "scale_cuda=format=nv12:interp_algo=bilinear";
  }

  enum AVPixelFormat frameFormat =
      static_cast<enum AVPixelFormat>(avFrame->format);

  auto newContext = std::make_unique<FiltersContext>(
      avFrame->width,
      avFrame->height,
      frameFormat,
      avFrame->sample_aspect_ratio,
      avFrame->width,
      avFrame->height,
      outputFormat,
      filters.str(),
      timeBase_,
      av_buffer_ref(avFrame->hw_frames_ctx));

  if (!nv12Conversion_ || *nv12ConversionContext_ != *newContext) {
    nv12Conversion_ =
        std::make_unique<FilterGraph>(*newContext, videoStreamOptions_);
    nv12ConversionContext_ = std::move(newContext);
  }
  auto filteredAVFrame = nv12Conversion_->convert(avFrame);

  // If this check fails it means the frame wasn't
  // reshaped to its expected dimensions by filtergraph.
  TORCH_CHECK(
      (filteredAVFrame->width == nv12ConversionContext_->outputWidth) &&
          (filteredAVFrame->height == nv12ConversionContext_->outputHeight),
      "Expected frame from filter graph of ",
      nv12ConversionContext_->outputWidth,
      "x",
      nv12ConversionContext_->outputHeight,
      ", got ",
      filteredAVFrame->width,
      "x",
      filteredAVFrame->height);

  return filteredAVFrame;
}

void CudaDeviceInterface::convertAVFrameToFrameOutput(
    UniqueAVFrame& avFrame,
    FrameOutput& frameOutput,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  validatePreAllocatedTensorShape(preAllocatedOutputTensor, avFrame);

  // All of our CUDA decoding assumes NV12 format. We handle non-NV12 formats by
  // converting them to NV12.
  avFrame = maybeConvertAVFrameToNV12OrRGB24(avFrame);

  if (avFrame->format != AV_PIX_FMT_CUDA) {
    // The frame's format is AV_PIX_FMT_CUDA if and only if its content is on
    // the GPU. In this branch, the frame is on the CPU. There are two possible
    // reasons:
    //
    //   1. During maybeConvertAVFrameToNV12OrRGB24(), we had a non-NV12 format
    //      frame and we're on FFmpeg 4.4 or earlier. In such cases, we had to
    //      use CPU filters and we just converted the frame to RGB24.
    //   2. This is what NVDEC gave us if it wasn't able to decode a frame, for
    //      whatever reason. Typically that happens if the video's encoder isn't
    //      supported by NVDEC.
    //
    // In both cases, we have a frame on the CPU. We send the frame back to the
    // CUDA device when we're done.

    enum AVPixelFormat frameFormat =
        static_cast<enum AVPixelFormat>(avFrame->format);

    FrameOutput cpuFrameOutput;
    if (frameFormat == AV_PIX_FMT_RGB24) {
      // Reason 1 above. The frame is already in RGB24, we just need to convert
      // it to a tensor.
      cpuFrameOutput.data = rgbAVFrameToTensor(avFrame);
    } else {
      // Reason 2 above. We need to do a full conversion which requires an
      // actual CPU device.
      cpuInterface_->convertAVFrameToFrameOutput(avFrame, cpuFrameOutput);
    }

    // Finally, we need to send the frame back to the GPU. Note that the
    // pre-allocated tensor is on the GPU, so we can't send that to the CPU
    // device interface. We copy it over here.
    if (preAllocatedOutputTensor.has_value()) {
      preAllocatedOutputTensor.value().copy_(cpuFrameOutput.data);
      frameOutput.data = preAllocatedOutputTensor.value();
    } else {
      frameOutput.data = cpuFrameOutput.data.to(device_);
    }

    usingCPUFallback_ = true;
    return;
  }

  usingCPUFallback_ = false;

  // Above we checked that the AVFrame was on GPU, but that's not enough, we
  // also need to check that the AVFrame is in AV_PIX_FMT_NV12 format (8 bits),
  // because this is what the NPP color conversion routines expect. This SHOULD
  // be enforced by our call to maybeConvertAVFrameToNV12OrRGB24() above.
  TORCH_CHECK(
      avFrame->hw_frames_ctx != nullptr,
      "The AVFrame does not have a hw_frames_ctx. This should never happen");
  AVHWFramesContext* hwFramesCtx =
      reinterpret_cast<AVHWFramesContext*>(avFrame->hw_frames_ctx->data);
  TORCH_CHECK(
      hwFramesCtx != nullptr,
      "The AVFrame does not have a valid hw_frames_ctx. This should never happen");

  AVPixelFormat actualFormat = hwFramesCtx->sw_format;
  TORCH_CHECK(
      actualFormat == AV_PIX_FMT_NV12,
      "The AVFrame is ",
      (av_get_pix_fmt_name(actualFormat) ? av_get_pix_fmt_name(actualFormat)
                                         : "unknown"),
      ", but we expected AV_PIX_FMT_NV12. "
      "That's unexpected, please report this to the TorchCodec repo.");

  // Figure out the NVDEC stream from the avFrame's hardware context.
  // In reality, we know that this stream is hardcoded to be the default stream
  // by FFmpeg:
  // https://github.com/FFmpeg/FFmpeg/blob/66e40840d15b514f275ce3ce2a4bf72ec68c7311/libavutil/hwcontext_cuda.c#L387-L388
  TORCH_CHECK(
      hwFramesCtx->device_ctx != nullptr,
      "The AVFrame's hw_frames_ctx does not have a device_ctx. ");
  auto cudaDeviceCtx =
      static_cast<AVCUDADeviceContext*>(hwFramesCtx->device_ctx->hwctx);
  TORCH_CHECK(cudaDeviceCtx != nullptr, "The hardware context is null");
  at::cuda::CUDAStream nvdecStream = // That's always the default stream. Sad.
      c10::cuda::getStreamFromExternal(cudaDeviceCtx->stream, device_.index());

  frameOutput.data = convertNV12FrameToRGB(
      avFrame, device_, nppCtx_, nvdecStream, preAllocatedOutputTensor);
}

// inspired by https://github.com/FFmpeg/FFmpeg/commit/ad67ea9
// we have to do this because of an FFmpeg bug where hardware decoding is not
// appropriately set, so we just go off and find the matching codec for the CUDA
// device
std::optional<const AVCodec*> CudaDeviceInterface::findCodec(
    const AVCodecID& codecId) {
  void* i = nullptr;
  const AVCodec* codec = nullptr;
  while ((codec = av_codec_iterate(&i)) != nullptr) {
    if (codec->id != codecId || !av_codec_is_decoder(codec)) {
      continue;
    }

    const AVCodecHWConfig* config = nullptr;
    for (int j = 0; (config = avcodec_get_hw_config(codec, j)) != nullptr;
         ++j) {
      if (config->device_type == AV_HWDEVICE_TYPE_CUDA) {
        return codec;
      }
    }
  }

  return std::nullopt;
}

std::string CudaDeviceInterface::getDetails() {
  // Note: for this interface specifically the fallback is only known after a
  // frame has been decoded, not before: that's when FFmpeg decides to fallback,
  // so we can't know earlier.
  return std::string("FFmpeg CUDA Device Interface. Using ") +
      (usingCPUFallback_ ? "CPU fallback." : "NVDEC.");
}

} // namespace facebook::torchcodec
