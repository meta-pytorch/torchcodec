#include <cuda_runtime.h>
#include <mutex>

#include "Cache.h"
#include "CudaDeviceInterface.h"
#include "FFMPEGCommon.h"
#include "StableABICompat.h"
#include "ValidationUtils.h"

extern "C" {
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {
namespace {

static bool g_cuda = register_device_interface(
    DeviceInterfaceKey(kStableCUDA, /*variant=*/"ffmpeg"),
    [](const StableDevice& device) { return new CudaDeviceInterface(device); });

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

int get_flags_av_hardware_device_context_create() {
// 58.26.100 introduced the concept of reusing the existing cuda context
// which is much faster and lower memory than creating a new cuda context.
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(58, 26, 100)
  return AV_CUDA_USE_CURRENT_CONTEXT;
#else
  return 0;
#endif
}

UniqueAVBufferRef get_hardware_device_context(const StableDevice& device) {
  enum AVHWDeviceType type = av_hwdevice_find_type_by_name("cuda");
  STD_TORCH_CHECK(type != AV_HWDEVICE_TYPE_NONE, "Failed to find cuda device");
  int device_index = get_device_index(device);

  UniqueAVBufferRef hardware_device_ctx = g_cached_hw_device_ctxs.get(device);
  if (hardware_device_ctx) {
    return hardware_device_ctx;
  }

  // Create hardware device context
  StableDeviceGuard device_guard(device.index());
  // We set the device because we may be called from a different thread than
  // the one that initialized the cuda context.
  STD_TORCH_CHECK(
      cudaSetDevice(device_index) == cudaSuccess, "Failed to set CUDA device");
  AVBufferRef* hardware_device_ctx_raw = nullptr;
  std::string device_ordinal = std::to_string(device_index);

  int err = av_hwdevice_ctx_create(
      &hardware_device_ctx_raw,
      type,
      device_ordinal.c_str(),
      nullptr,
      get_flags_av_hardware_device_context_create());

  if (err < 0) {
    /* clang-format off */
    STD_TORCH_CHECK(
        false,
        "Failed to create specified HW device. This typically happens when ",
        "your installed FFmpeg doesn't support CUDA (see ",
        "https://github.com/pytorch/torchcodec#installing-cuda-enabled-torchcodec",
        "). FFmpeg error: ", get_ffmpeg_error_string_from_error_code(err));
    /* clang-format on */
  }

  return UniqueAVBufferRef(hardware_device_ctx_raw);
}

} // namespace

CudaDeviceInterface::CudaDeviceInterface(const StableDevice& device)
    : DeviceInterface(device) {
  STD_TORCH_CHECK(g_cuda, "CudaDeviceInterface was not registered!");
  STD_TORCH_CHECK(
      device_.type() == kStableCUDA, "Unsupported device: must be CUDA");

  // Resolve unspecified device index (-1) to the actual current CUDA device.
  device_.set_index(get_device_index(device_));

  initialize_cuda_context_with_pytorch(device_);

  hardware_device_ctx_ = get_hardware_device_context(device_);
}

CudaDeviceInterface::~CudaDeviceInterface() {
  if (hardware_device_ctx_) {
    g_cached_hw_device_ctxs.add_if_cache_has_capacity(
        device_, std::move(hardware_device_ctx_));
  }
}

void CudaDeviceInterface::initialize(
    const SharedAVCodecContext& codec_context) {
  codec_context_ = codec_context;
}

void CudaDeviceInterface::initialize_video(
    const AVStream* av_stream,
    const UniqueDecodingAVFormatContext& av_format_ctx,
    const VideoStreamOptions& video_stream_options,
    [[maybe_unused]] const std::vector<std::unique_ptr<Transform>>& transforms,
    [[maybe_unused]] const std::optional<FrameDims>& resized_output_dims) {
  STD_TORCH_CHECK(av_stream != nullptr, "avStream is null");
  time_base_ = av_stream->time_base;
  video_stream_options_ = video_stream_options;

  // TODO: Ideally, we should keep all interface implementations independent.
  cpu_interface_ = create_device_interface(kStableCPU);
  STD_TORCH_CHECK(
      cpu_interface_ != nullptr, "Failed to create CPU device interface");
  cpu_interface_->initialize(codec_context_);
  cpu_interface_->initialize_video(
      av_stream,
      av_format_ctx,
      VideoStreamOptions(),
      {},
      /*resizedOutputDims=*/std::nullopt);
}

void CudaDeviceInterface::register_hardware_device_with_codec(
    AVCodecContext* codec_context) {
  STD_TORCH_CHECK(
      hardware_device_ctx_, "Hardware device context has not been initialized");
  STD_TORCH_CHECK(codec_context != nullptr, "codecContext is null");
  codec_context->hw_device_ctx = av_buffer_ref(hardware_device_ctx_.get());
}

UniqueAVFrame CudaDeviceInterface::maybe_convert_av_frame_to_nv12_or_rgb24(
    UniqueAVFrame& av_frame) {
  // We need FFmpeg filters to handle those conversion cases which are not
  // directly implemented in CUDA or CPU device interface (in case of a
  // fallback).

  // Input frame is on CPU, we will just pass it to CPU device interface, so
  // skipping filters context as CPU device interface will handle everything for
  // us.
  if (av_frame->format != AV_PIX_FMT_CUDA) {
    return std::move(av_frame);
  }

  auto hw_frames_ctx =
      reinterpret_cast<AVHWFramesContext*>(av_frame->hw_frames_ctx->data);
  STD_TORCH_CHECK(
      hw_frames_ctx != nullptr,
      "The AVFrame does not have a hw_frames_ctx. "
      "That's unexpected, please report this to the TorchCodec repo.");

  AVPixelFormat actual_format = hw_frames_ctx->sw_format;

  // If the frame is already in NV12 format, we don't need to do anything.
  if (actual_format == AV_PIX_FMT_NV12) {
    return std::move(av_frame);
  }

  AVPixelFormat output_format;
  std::stringstream filters;

  unsigned version_int = avfilter_version();
  if (version_int < AV_VERSION_INT(8, 0, 103)) {
    // Color conversion support ('format=' option) was added to scale_cuda from
    // n5.0. With the earlier version of ffmpeg we have no choice but use CPU
    // filters. See:
    // https://github.com/FFmpeg/FFmpeg/commit/62dc5df941f5e196164c151691e4274195523e95
    output_format = AV_PIX_FMT_RGB24;

    auto actual_format_name = av_get_pix_fmt_name(actual_format);
    STD_TORCH_CHECK(
        actual_format_name != nullptr,
        "The actual format of a frame is unknown to FFmpeg. "
        "That's unexpected, please report this to the TorchCodec repo.");

    filters << "hwdownload,format=" << actual_format_name;
  } else {
    // Actual output color format will be set via filter options
    output_format = AV_PIX_FMT_CUDA;

    filters << "scale_cuda=format=nv12:interp_algo=bilinear";
  }

  enum AVPixelFormat frame_format =
      static_cast<enum AVPixelFormat>(av_frame->format);

  auto new_config = std::make_unique<FiltersConfig>(
      av_frame->width,
      av_frame->height,
      frame_format,
      av_frame->sample_aspect_ratio,
      av_frame->width,
      av_frame->height,
      output_format,
      filters.str(),
      time_base_,
      av_buffer_ref(av_frame->hw_frames_ctx));

  if (!nv12_conversion_ || *nv12_conversion_config_ != *new_config) {
    nv12_conversion_ =
        std::make_unique<FilterGraph>(*new_config, video_stream_options_);
    nv12_conversion_config_ = std::move(new_config);
  }
  auto filtered_av_frame = nv12_conversion_->convert(av_frame);

  // If this check fails it means the frame wasn't
  // reshaped to its expected dimensions by filtergraph.
  STD_TORCH_CHECK(
      (filtered_av_frame->width == nv12_conversion_config_->output_width) &&
          (filtered_av_frame->height == nv12_conversion_config_->output_height),
      "Expected frame from filter graph of ",
      nv12_conversion_config_->output_width,
      "x",
      nv12_conversion_config_->output_height,
      ", got ",
      filtered_av_frame->width,
      "x",
      filtered_av_frame->height);

  return filtered_av_frame;
}

void CudaDeviceInterface::convert_av_frame_to_frame_output(
    UniqueAVFrame& av_frame,
    FrameOutput& frame_output,
    std::optional<torch::stable::Tensor> pre_allocated_output_tensor) {
  validate_pre_allocated_tensor_shape(
      pre_allocated_output_tensor,
      FrameDims(av_frame->height, av_frame->width));

  has_decoded_frame_ = true;

  // All of our CUDA decoding assumes NV12 format. We handle non-NV12 formats by
  // converting them to NV12.
  av_frame = maybe_convert_av_frame_to_nv12_or_rgb24(av_frame);

  if (av_frame->format != AV_PIX_FMT_CUDA) {
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

    enum AVPixelFormat frame_format =
        static_cast<enum AVPixelFormat>(av_frame->format);

    FrameOutput cpu_frame_output;
    if (frame_format == AV_PIX_FMT_RGB24) {
      // Reason 1 above. The frame is already in RGB24, we just need to convert
      // it to a tensor.
      cpu_frame_output.data = rgb_av_frame_to_tensor(av_frame);
    } else {
      // Reason 2 above. We need to do a full conversion which requires an
      // actual CPU device.
      cpu_interface_->convert_av_frame_to_frame_output(
          av_frame, cpu_frame_output);
    }

    // Finally, we need to send the frame back to the GPU. Note that the
    // pre-allocated tensor is on the GPU, so we can't send that to the CPU
    // device interface. We copy it over here.
    if (pre_allocated_output_tensor.has_value()) {
      torch::stable::copy_(
          pre_allocated_output_tensor.value(), cpu_frame_output.data);
      frame_output.data = pre_allocated_output_tensor.value();
    } else {
      frame_output.data = torch::stable::to(cpu_frame_output.data, device_);
    }

    using_cpu_fallback_ = true;
    return;
  }

  using_cpu_fallback_ = false;

  // Above we checked that the AVFrame was on GPU, but that's not enough, we
  // also need to check that the AVFrame is in AV_PIX_FMT_NV12 format (8 bits),
  // because this is what our color conversion kernel expects. This SHOULD
  // be enforced by our call to maybeConvertAVFrameToNV12OrRGB24() above.
  STD_TORCH_CHECK(
      av_frame->hw_frames_ctx != nullptr,
      "The AVFrame does not have a hw_frames_ctx. This should never happen");
  AVHWFramesContext* hw_frames_ctx =
      reinterpret_cast<AVHWFramesContext*>(av_frame->hw_frames_ctx->data);
  STD_TORCH_CHECK(
      hw_frames_ctx != nullptr,
      "The AVFrame does not have a valid hw_frames_ctx. This should never happen");

  AVPixelFormat actual_format = hw_frames_ctx->sw_format;
  STD_TORCH_CHECK(
      actual_format == AV_PIX_FMT_NV12,
      "The AVFrame is ",
      (av_get_pix_fmt_name(actual_format) ? av_get_pix_fmt_name(actual_format)
                                          : "unknown"),
      ", but we expected AV_PIX_FMT_NV12. "
      "That's unexpected, please report this to the TorchCodec repo.");

  // Figure out the NVDEC stream from the avFrame's hardware context.
  // In reality, we know that this stream is hardcoded to be the default stream
  // by FFmpeg:
  // https://github.com/FFmpeg/FFmpeg/blob/66e40840d15b514f275ce3ce2a4bf72ec68c7311/libavutil/hwcontext_cuda.c#L387-L388
  STD_TORCH_CHECK(
      hw_frames_ctx->device_ctx != nullptr,
      "The AVFrame's hw_frames_ctx does not have a device_ctx. ");
  auto cuda_device_ctx =
      static_cast<AVCUDADeviceContext*>(hw_frames_ctx->device_ctx->hwctx);
  STD_TORCH_CHECK(cuda_device_ctx != nullptr, "The hardware context is null");

  cudaStream_t nvdec_stream = // That's always the default stream. Sad.
      cuda_device_ctx->stream;

  frame_output.data = convert_yuv_frame_to_rgb(
      av_frame,
      device_,
      nvdec_stream,
      pre_allocated_output_tensor,
      FrameDims(av_frame->height, av_frame->width),
      /*isP016=*/false,
      /*bitDepth=*/8,
      cached_color_matrix_);
}

// inspired by https://github.com/FFmpeg/FFmpeg/commit/ad67ea9
// we have to do this because of an FFmpeg bug where hardware decoding is not
// appropriately set, so we just go off and find the matching codec for the CUDA
// device
std::optional<const AVCodec*> CudaDeviceInterface::find_codec(
    const AVCodecID& codec_id,
    bool is_decoder) {
  void* i = nullptr;
  const AVCodec* codec = nullptr;
  while ((codec = av_codec_iterate(&i)) != nullptr) {
    STD_TORCH_CHECK(
        codec != nullptr,
        "codec returned by av_codec_iterate should not be null");
    if (is_decoder) {
      if (codec->id != codec_id || !av_codec_is_decoder(codec)) {
        continue;
      }
    } else {
      if (codec->id != codec_id || !av_codec_is_encoder(codec)) {
        continue;
      }
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

std::string CudaDeviceInterface::get_details() {
  // Note: for this interface specifically the fallback is only known after a
  // frame has been decoded, not before: that's when FFmpeg decides to fallback,
  // so we can't know earlier.
  if (!has_decoded_frame_) {
    return std::string(
        "FFmpeg CUDA Device Interface. Fallback status unknown (no frames decoded).");
  }
  return std::string("FFmpeg CUDA Device Interface. Using ") +
      (using_cpu_fallback_ ? "CPU fallback." : "NVDEC.");
}

// --------------------------------------------------------------------------
// Below are methods exclusive to video encoding:
// --------------------------------------------------------------------------

AVPixelFormat CudaDeviceInterface::get_encoding_pixel_format(
    [[maybe_unused]] const AVCodec& av_codec,
    const std::optional<std::string>& user_pixel_format) const {
  STD_TORCH_CHECK(
      !user_pixel_format.has_value(),
      "Video encoding on GPU currently only supports the nv12 pixel format. "
      "Do not set pixel_format to use nv12 by default.");
  return CudaDeviceInterface::CUDA_ENCODING_PIXEL_FORMAT;
}

UniqueAVFrame CudaDeviceInterface::convert_tensor_to_av_frame_for_encoding(
    const torch::stable::Tensor& tensor,
    int frame_index,
    AVCodecContext* codec_context) {
  STD_TORCH_CHECK(
      tensor.dim() == 3 && tensor.sizes()[0] == 3,
      "Expected 3D RGB tensor (CHW format), got ",
      tensor.dim(),
      "D tensor");
  STD_TORCH_CHECK(
      tensor.device().type() == kStableCUDA,
      "Expected tensor on CUDA device, got: ",
      device_type_name(tensor.device().type()));

  UniqueAVFrame av_frame(av_frame_alloc());
  STD_TORCH_CHECK(av_frame != nullptr, "Failed to allocate AVFrame");
  int height = static_cast<int>(tensor.sizes()[1]);
  int width = static_cast<int>(tensor.sizes()[2]);

  // TODO-VideoEncoder: (P1) Unify AVFrame creation with CPU method
  av_frame->format = AV_PIX_FMT_CUDA;
  av_frame->height = height;
  av_frame->width = width;
  av_frame->pts = frame_index;

  // FFmpeg's av_hwframe_get_buffer is used to allocate memory on CUDA device.
  // TODO-VideoEncoder: (P2) Consider using pytorch to allocate CUDA memory for
  // efficiency
  int ret =
      av_hwframe_get_buffer(codec_context->hw_frames_ctx, av_frame.get(), 0);
  STD_TORCH_CHECK(
      ret >= 0,
      "Failed to allocate hardware frame: ",
      get_ffmpeg_error_string_from_error_code(ret));

  STD_TORCH_CHECK(
      av_frame != nullptr && av_frame->data[0] != nullptr,
      "avFrame must be pre-allocated with CUDA memory");

  // TODO VideoEncoder: Investigate ways to avoid this copy
  torch::stable::Tensor hwc_frame =
      torch::stable::contiguous(stable_permute(tensor, {1, 2, 0}));

  float rgb_to_yuv_matrix[3][4];
  compute_rgb_to_yuv_matrix(
      codec_context->colorspace, codec_context->color_range, rgb_to_yuv_matrix);

  cudaStream_t stream = get_current_cuda_stream(device_.index());
  launch_rgb_to_nv12_kernel(
      hwc_frame.const_data_ptr<uint8_t>(),
      av_frame->data[0],
      av_frame->data[1],
      width,
      height,
      validate_int64_to_int(
          hwc_frame.stride(0) * static_cast<int64_t>(hwc_frame.element_size()),
          "rgbPitch"),
      av_frame->linesize[0],
      av_frame->linesize[1],
      rgb_to_yuv_matrix,
      stream);

  av_frame->colorspace = codec_context->colorspace;
  av_frame->color_range = codec_context->color_range;
  return av_frame;
}

// Allocates and initializes AVHWFramesContext, and sets pixel format fields
// to enable encoding with CUDA device. The hw_frames_ctx field is needed by
// FFmpeg to allocate frames on GPU's memory.
void CudaDeviceInterface::setup_hardware_frame_context_for_encoding(
    AVCodecContext* codec_context) {
  STD_TORCH_CHECK(codec_context != nullptr, "codecContext is null");
  STD_TORCH_CHECK(
      hardware_device_ctx_, "Hardware device context has not been initialized");

  AVBufferRef* hw_frames_ctx_ref =
      av_hwframe_ctx_alloc(hardware_device_ctx_.get());
  STD_TORCH_CHECK(
      hw_frames_ctx_ref != nullptr,
      "Failed to allocate hardware frames context for codec");

  codec_context->sw_pix_fmt = CudaDeviceInterface::CUDA_ENCODING_PIXEL_FORMAT;
  // Always set pixel format to support CUDA encoding.
  codec_context->pix_fmt = AV_PIX_FMT_CUDA;

  AVHWFramesContext* hw_frames_ctx =
      reinterpret_cast<AVHWFramesContext*>(hw_frames_ctx_ref->data);
  hw_frames_ctx->format = codec_context->pix_fmt;
  hw_frames_ctx->sw_format = codec_context->sw_pix_fmt;
  hw_frames_ctx->width = codec_context->width;
  hw_frames_ctx->height = codec_context->height;

  int ret = av_hwframe_ctx_init(hw_frames_ctx_ref);
  if (ret < 0) {
    av_buffer_unref(&hw_frames_ctx_ref);
    STD_TORCH_CHECK(
        false,
        "Failed to initialize CUDA frames context for codec: ",
        get_ffmpeg_error_string_from_error_code(ret));
  }
  codec_context->hw_frames_ctx = hw_frames_ctx_ref;
}
} // namespace facebook::torchcodec
