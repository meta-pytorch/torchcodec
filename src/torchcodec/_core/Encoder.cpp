#include <sstream>

#include "Encoder.h"
#include "StableABICompat.h"

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {

namespace {

torch::stable::Tensor validate_samples(const torch::stable::Tensor& samples) {
  STD_TORCH_CHECK(
      samples.scalar_type() == kStableFloat32,
      "samples must have float32 dtype, got ",
      (samples.scalar_type()));
  STD_TORCH_CHECK(
      samples.device().type() == kStableCPU,
      "samples must be on CPU, got ",
      device_type_name(samples.device().type()));
  STD_TORCH_CHECK(
      samples.dim() == 2,
      "samples must have 2 dimensions, got ",
      samples.dim());

  // We enforce this, but if we get user reports we should investigate whether
  // that's actually needed.
  int num_channels = static_cast<int>(samples.sizes()[0]);
  STD_TORCH_CHECK(
      num_channels <= AV_NUM_DATA_POINTERS,
      "Trying to encode ",
      num_channels,
      " channels, but FFmpeg only supports ",
      AV_NUM_DATA_POINTERS,
      " channels per frame.");

  return torch::stable::contiguous(samples);
}

void validate_sample_rate(const AVCodec& av_codec, int sample_rate) {
  const int* supported_sample_rates = get_supported_sample_rates(av_codec);
  if (supported_sample_rates == nullptr) {
    return;
  }

  for (auto i = 0; supported_sample_rates[i] != 0; ++i) {
    if (sample_rate == supported_sample_rates[i]) {
      return;
    }
  }
  std::stringstream supported_rates;
  for (auto i = 0; supported_sample_rates[i] != 0; ++i) {
    if (i > 0) {
      supported_rates << ", ";
    }
    supported_rates << supported_sample_rates[i];
  }

  STD_TORCH_CHECK(
      false,
      "invalid sample rate=",
      sample_rate,
      ". Supported sample rate values are: ",
      supported_rates.str());
}

static const std::vector<AVSampleFormat> preferred_formats_order = {
    AV_SAMPLE_FMT_FLTP,
    AV_SAMPLE_FMT_FLT,
    AV_SAMPLE_FMT_DBLP,
    AV_SAMPLE_FMT_DBL,
    AV_SAMPLE_FMT_S64P,
    AV_SAMPLE_FMT_S64,
    AV_SAMPLE_FMT_S32P,
    AV_SAMPLE_FMT_S32,
    AV_SAMPLE_FMT_S16P,
    AV_SAMPLE_FMT_S16,
    AV_SAMPLE_FMT_U8P,
    AV_SAMPLE_FMT_U8};

AVSampleFormat find_best_output_sample_format(const AVCodec& av_codec) {
  const AVSampleFormat* supported_sample_formats =
      get_supported_output_sample_formats(av_codec);

  // Find a sample format that the encoder supports. We prefer using FLT[P],
  // since this is the format of the input samples. If FLTP isn't supported
  // then we'll need to convert the AVFrame's format. Our heuristic is to encode
  // into the format with the highest resolution.
  if (supported_sample_formats == nullptr) {
    // Can't really validate anything in this case, best we can do is hope that
    // FLTP is supported by the encoder. If not, FFmpeg will raise.
    return AV_SAMPLE_FMT_FLTP;
  }

  for (AVSampleFormat preferred_format : preferred_formats_order) {
    for (int i = 0; supported_sample_formats[i] != -1; ++i) {
      if (supported_sample_formats[i] == preferred_format) {
        return preferred_format;
      }
    }
  }
  // We should always find a match in preferred_formats_order, so we should
  // always return earlier. But in the event that a future FFmpeg version
  // defines an additional sample format that isn't in preferred_formats_order,
  // we fallback:
  return supported_sample_formats[0];
}

void close_avio_context(
    AVFormatContext* av_format_context,
    AVIOContextHolder* avio_context_holder) {
  if (!av_format_context || !av_format_context->pb) {
    return;
  }

  if (av_format_context->pb->error == 0) {
    avio_flush(av_format_context->pb);
  }

  if (!avio_context_holder) {
    if (av_format_context->pb->error == 0) {
      avio_close(av_format_context->pb);
    }
  }

  av_format_context->pb = nullptr;
}

torch::stable::Tensor validate_frames(
    const torch::stable::Tensor& frames,
    const AVCodecContext* av_codec_context = nullptr,
    DeviceInterface* device_interface = nullptr) {
  STD_TORCH_CHECK(
      frames.scalar_type() == kStableUInt8,
      "frames must have uint8 dtype, got ",
      frames.scalar_type());
  STD_TORCH_CHECK(
      frames.dim() == 4,
      "frames must have 4 dimensions (N, C, H, W), got ",
      frames.dim());
  STD_TORCH_CHECK(
      frames.sizes()[1] == 3,
      "frame must have 3 channels (R, G, B), got ",
      frames.sizes()[1]);
  if (device_interface != nullptr) {
    auto& expected_device = device_interface->device();
    auto frames_device = frames.device();
    STD_TORCH_CHECK(
        frames_device == expected_device,
        "All frames must be on the same device. Expected ",
        device_type_name(expected_device.type()),
        ":",
        expected_device.index(),
        ", got ",
        device_type_name(frames_device.type()),
        ":",
        frames_device.index());
  }
  if (av_codec_context) {
    STD_TORCH_CHECK(
        static_cast<int>(frames.sizes()[2]) == av_codec_context->height &&
            static_cast<int>(frames.sizes()[3]) == av_codec_context->width,
        "All frames must have the same dimensions. Expected height=",
        av_codec_context->height,
        " width=",
        av_codec_context->width,
        ", got height=",
        frames.sizes()[2],
        " width=",
        frames.sizes()[3]);
  }
  return torch::stable::contiguous(frames);
}

AVPixelFormat validate_pixel_format(
    const AVCodec& av_codec,
    const std::string& target_pixel_format) {
  AVPixelFormat pixel_format = av_get_pix_fmt(target_pixel_format.c_str());

  // Validate that the encoder supports this pixel format
  const AVPixelFormat* supported_formats =
      get_supported_pixel_formats(av_codec);
  if (supported_formats != nullptr) {
    for (int i = 0; supported_formats[i] != AV_PIX_FMT_NONE; ++i) {
      if (supported_formats[i] == pixel_format) {
        return pixel_format;
      }
    }
  }

  std::stringstream error_msg;
  // av_get_pix_fmt failed to find a pix_fmt
  if (pixel_format == AV_PIX_FMT_NONE) {
    error_msg << "Unknown pixel format: " << target_pixel_format;
  } else {
    error_msg << "Specified pixel format " << target_pixel_format
              << " is not supported by the " << av_codec.name << " encoder.";
  }
  // Build error message, similar to FFmpeg's error log
  error_msg << "\nSupported pixel formats for " << av_codec.name << ":";
  for (int i = 0; supported_formats[i] != AV_PIX_FMT_NONE; ++i) {
    error_msg << " " << av_get_pix_fmt_name(supported_formats[i]);
  }
  STD_TORCH_CHECK(false, error_msg.str());
}

void try_to_validate_codec_option(
    const AVCodec& av_codec,
    const char* option_name,
    const std::string& value) {
  if (!av_codec.priv_class) {
    return;
  }
  const AVOption* option = av_opt_find2(
      // Convert obj arg from const AVClass* const* to non-const void*
      // First cast to remove const, then cast to void*
      const_cast<void*>(static_cast<const void*>(&av_codec.priv_class)),
      option_name,
      nullptr,
      0,
      AV_OPT_SEARCH_FAKE_OBJ,
      nullptr);
  // If option is not found we cannot validate it, let FFmpeg handle it
  if (!option) {
    return;
  }
  // Validate if option is defined as a numeric type
  if (option->type == AV_OPT_TYPE_INT || option->type == AV_OPT_TYPE_INT64 ||
      option->type == AV_OPT_TYPE_FLOAT || option->type == AV_OPT_TYPE_DOUBLE) {
    try {
      double numeric_value = std::stod(value);
      STD_TORCH_CHECK(
          numeric_value >= option->min && numeric_value <= option->max,
          option_name,
          "=",
          numeric_value,
          " is out of valid range [",
          option->min,
          ", ",
          option->max,
          "] for this codec. For more details, run 'ffmpeg -h encoder=",
          av_codec.name,
          "'");
    } catch (const std::invalid_argument&) {
      STD_TORCH_CHECK(
          false,
          "Option ",
          option_name,
          " expects a numeric value but got '",
          value,
          "'");
    }
  }
}

void sort_codec_options(
    const AVFormatContext* av_format_context,
    const std::map<std::string, std::string>& extra_options,
    UniqueAVDictionary& codec_dict,
    UniqueAVDictionary& format_dict) {
  // Accepts a map of options as input, then sorts them into codec options and
  // format options. The sorted options are returned into two separate dicts.
  const AVClass* format_class = avformat_get_class();
  const AVClass* muxer_class = av_format_context->oformat
      ? av_format_context->oformat->priv_class
      : nullptr;
  for (const auto& [key, value] : extra_options) {
    // Check if option is generic format option
    const AVOption* fmt_opt = av_opt_find2(
        &format_class,
        key.c_str(),
        nullptr,
        0,
        AV_OPT_SEARCH_CHILDREN | AV_OPT_SEARCH_FAKE_OBJ,
        nullptr);
    // Check if option is muxer-specific option
    // (Returned from `ffmpeg -h muxer=mp4`)
    const AVOption* muxer_opt = nullptr;
    if (muxer_class) {
      muxer_opt = av_opt_find2(
          &muxer_class,
          key.c_str(),
          nullptr,
          0,
          AV_OPT_SEARCH_FAKE_OBJ,
          nullptr);
    }
    if (fmt_opt || muxer_opt) {
      // Pass container-format options to formatDict to be used in
      // avformat_write_header
      av_dict_set(format_dict.get_address(), key.c_str(), value.c_str(), 0);
    } else {
      // By default, pass as codec option to be used in avcodec_open2
      av_dict_set(codec_dict.get_address(), key.c_str(), value.c_str(), 0);
    }
  }
}

} // namespace

MultiStreamEncoder::~MultiStreamEncoder() {
  close();
}

MultiStreamEncoder::MultiStreamEncoder() {
  set_ffmpeg_log_level();
}

void MultiStreamEncoder::open(std::string_view file_name) {
  STD_TORCH_CHECK(!closed_, "Cannot open after close() was called.");
  STD_TORCH_CHECK(!header_written_, "open() was already called.");

  AVFormatContext* av_format_context = nullptr;
  int status = avformat_alloc_output_context2(
      &av_format_context, nullptr, nullptr, file_name.data());

  STD_TORCH_CHECK(
      av_format_context != nullptr,
      "Couldn't allocate AVFormatContext. ",
      "The destination file is ",
      file_name,
      ", check the desired extension? ",
      get_ffmpeg_error_string_from_error_code(status));
  av_format_context_.reset(av_format_context);

  status =
      avio_open(&av_format_context_->pb, file_name.data(), AVIO_FLAG_WRITE);
  STD_TORCH_CHECK(
      status >= 0,
      "avio_open failed. The destination file is ",
      file_name,
      ", make sure it's a valid path? ",
      get_ffmpeg_error_string_from_error_code(status));

  open_streams_and_write_header();
}

void MultiStreamEncoder::open(
    std::string_view format_name,
    std::unique_ptr<AVIOContextHolder> avio_context_holder) {
  STD_TORCH_CHECK(!closed_, "Cannot open after close() was called.");
  STD_TORCH_CHECK(!header_written_, "open() was already called.");

  avio_context_holder_ = std::move(avio_context_holder);

  // Map mkv -> matroska when used as format name
  format_name = (format_name == "mkv") ? "matroska" : format_name;
  AVFormatContext* av_format_context = nullptr;
  int status = avformat_alloc_output_context2(
      &av_format_context, nullptr, format_name.data(), nullptr);

  STD_TORCH_CHECK(
      av_format_context != nullptr,
      "Couldn't allocate AVFormatContext. ",
      "Check the desired format? Got format=",
      format_name,
      ". ",
      get_ffmpeg_error_string_from_error_code(status));
  av_format_context_.reset(av_format_context);

  av_format_context_->pb = avio_context_holder_->get_avio_context();

  open_streams_and_write_header();
}

int MultiStreamEncoder::add_video_stream(
    int height,
    int width,
    double frame_rate,
    std::string device,
    std::optional<std::string> codec,
    std::optional<std::string> pixel_format,
    std::optional<double> crf,
    std::optional<std::string> preset,
    std::optional<std::map<std::string, std::string>> extra_options) {
  STD_TORCH_CHECK(height > 0, "height must be > 0, got ", height);
  STD_TORCH_CHECK(width > 0, "width must be > 0, got ", width);
  STD_TORCH_CHECK(frame_rate > 0, "frame_rate must be > 0, got ", frame_rate);
  VideoStream video_stream;
  StableDevice stable_device(std::move(device));
  video_stream.device_interface = create_device_interface(
      stable_device,
      stable_device.type() == kStableCUDA ? "ffmpeg" : "default");
  video_stream.in_height = height;
  video_stream.in_width = width;
  video_stream.in_frame_rate = frame_rate;
  video_stream.options.codec = std::move(codec);
  video_stream.options.pixel_format = std::move(pixel_format);
  video_stream.options.crf = crf;
  video_stream.options.preset = std::move(preset);
  video_stream.options.extra_options = std::move(extra_options);
  video_streams_.push_back(std::move(video_stream));
  return static_cast<int>(video_streams_.size() - 1);
}

int MultiStreamEncoder::add_audio_stream(
    int sample_rate,
    int num_channels,
    std::optional<int> bit_rate,
    std::optional<int> out_num_channels,
    std::optional<int> out_sample_rate) {
  STD_TORCH_CHECK(
      sample_rate > 0, "sample_rate must be > 0, got ", sample_rate);
  STD_TORCH_CHECK(
      num_channels > 0, "num_channels must be > 0, got ", num_channels);
  STD_TORCH_CHECK(
      num_channels <= AV_NUM_DATA_POINTERS,
      "Trying to encode ",
      num_channels,
      " channels, but FFmpeg only supports ",
      AV_NUM_DATA_POINTERS,
      " channels per frame.");

  AudioStream audio_stream;
  audio_stream.in_sample_rate = sample_rate;
  audio_stream.in_num_channels = num_channels;
  audio_stream.options.bit_rate = bit_rate;
  audio_stream.options.num_channels = out_num_channels;
  audio_stream.options.sample_rate = out_sample_rate;
  audio_streams_.push_back(std::move(audio_stream));
  return static_cast<int>(audio_streams_.size() - 1);
}

void MultiStreamEncoder::initialize_video_stream(VideoStream& video_stream) {
  auto device_type = video_stream.device_interface->device().type();

  const AVCodec* av_codec = nullptr;
  // If codec arg is provided, find codec using logic similar to FFmpeg:
  // https://github.com/FFmpeg/FFmpeg/blob/master/fftools/ffmpeg_opt.c#L804-L835
  if (video_stream.options.codec.has_value()) {
    const std::string& codec = video_stream.options.codec.value();
    // Try to find codec by name ("libx264", "libsvtav1")
    av_codec = avcodec_find_encoder_by_name(codec.c_str());
    // Try to find by codec descriptor ("h264", "av1")
    if (!av_codec) {
      const AVCodecDescriptor* desc =
          avcodec_descriptor_get_by_name(codec.c_str());
      if (desc) {
        av_codec = avcodec_find_encoder(desc->id);
      }
    }
  } else {
    STD_TORCH_CHECK(
        av_format_context_->oformat != nullptr,
        "Output format is null, unable to find default codec.");
    // Try to substitute the default codec with its hardware equivalent
    // This will return std::nullopt when device is CPU.
    auto hw_codec = video_stream.device_interface->find_codec(
        av_format_context_->oformat->video_codec, /*isDecoder=*/false);
    if (hw_codec.has_value()) {
      av_codec = hw_codec.value();
    }
    if (!av_codec) {
      av_codec = avcodec_find_encoder(av_format_context_->oformat->video_codec);
    }
  }
  STD_TORCH_CHECK(
      av_codec != nullptr,
      "Video codec ",
      video_stream.options.codec.has_value()
          ? video_stream.options.codec.value() + " "
          : "",
      "not found. To see available codecs, run: ffmpeg -encoders");

  AVCodecContext* av_codec_context = avcodec_alloc_context3(av_codec);
  STD_TORCH_CHECK(
      av_codec_context != nullptr, "Couldn't allocate codec context.");
  video_stream.av_codec_context.reset(av_codec_context);

  int out_height = video_stream.in_height;
  int out_width = video_stream.in_width;
  AVPixelFormat out_pixel_format = AV_PIX_FMT_NONE;

  if (video_stream.options.pixel_format.has_value()) {
    if (device_type == kStableCUDA) {
      STD_TORCH_CHECK(
          false,
          "Video encoding on GPU currently only supports the nv12 pixel format. "
          "Do not set pixel_format to use nv12 by default.");
    }
    out_pixel_format = validate_pixel_format(
        *av_codec, video_stream.options.pixel_format.value());
  } else {
    if (device_type == kStableCUDA) {
      // Default to nv12 pixel format when encoding on GPU.
      out_pixel_format = DeviceInterface::CUDA_ENCODING_PIXEL_FORMAT;
    } else {
      const AVPixelFormat* formats = get_supported_pixel_formats(*av_codec);
      // Use first listed pixel format as default (often yuv420p).
      // This is similar to FFmpeg's logic:
      // https://www.ffmpeg.org/doxygen/4.0/decode_8c_source.html#l01087
      // If pixel formats are undefined for some reason, try yuv420p
      out_pixel_format = (formats && formats[0] != AV_PIX_FMT_NONE)
          ? formats[0]
          : AV_PIX_FMT_YUV420P;
    }
  }

  // Configure codec parameters
  video_stream.av_codec_context->codec_id = av_codec->id;
  video_stream.av_codec_context->width = out_width;
  video_stream.av_codec_context->height = out_height;
  video_stream.av_codec_context->pix_fmt = out_pixel_format;
  video_stream.av_codec_context->framerate =
      av_d2q(video_stream.in_frame_rate, INT_MAX);
  video_stream.av_codec_context->time_base =
      av_inv_q(video_stream.av_codec_context->framerate);

  // Set flag for containers that require extradata to be in the codec context
  if (av_format_context_->oformat->flags & AVFMT_GLOBALHEADER) {
    video_stream.av_codec_context->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }

  // Apply videoStreamOptions
  UniqueAVDictionary av_codec_options;
  if (video_stream.options.extra_options.has_value()) {
    for (const auto& [key, value] :
         video_stream.options.extra_options.value()) {
      try_to_validate_codec_option(*av_codec, key.c_str(), value);
    }
    sort_codec_options(
        av_format_context_.get(),
        video_stream.options.extra_options.value(),
        av_codec_options,
        av_format_options_);
  }

  if (video_stream.options.crf.has_value()) {
    std::string crf_value = std::to_string(video_stream.options.crf.value());
    try_to_validate_codec_option(*av_codec, "crf", crf_value);
    av_dict_set(av_codec_options.get_address(), "crf", crf_value.c_str(), 0);
  }

  if (video_stream.options.preset.has_value()) {
    av_dict_set(
        av_codec_options.get_address(),
        "preset",
        video_stream.options.preset.value().c_str(),
        0);
  }

  if (device_type == kStableCUDA) {
    video_stream.device_interface->register_hardware_device_with_codec(
        video_stream.av_codec_context.get());
    video_stream.device_interface->setup_hardware_frame_context_for_encoding(
        video_stream.av_codec_context.get());
  }

  int status = avcodec_open2(
      video_stream.av_codec_context.get(),
      av_codec,
      av_codec_options.get_address());

  STD_TORCH_CHECK(
      status == AVSUCCESS,
      "avcodec_open2 failed: ",
      get_ffmpeg_error_string_from_error_code(status));

  video_stream.av_stream =
      avformat_new_stream(av_format_context_.get(), nullptr);
  STD_TORCH_CHECK(
      video_stream.av_stream != nullptr, "Couldn't create new stream.");

  // Set the stream time base to encode correct frame timestamps
  video_stream.av_stream->time_base = video_stream.av_codec_context->time_base;
  // Set the stream frame rate to store correct frame durations for some
  // containers (webm, mkv)
  video_stream.av_stream->r_frame_rate =
      video_stream.av_codec_context->framerate;

  status = avcodec_parameters_from_context(
      video_stream.av_stream->codecpar, video_stream.av_codec_context.get());
  STD_TORCH_CHECK(
      status == AVSUCCESS,
      "avcodec_parameters_from_context failed: ",
      get_ffmpeg_error_string_from_error_code(status));
}

void MultiStreamEncoder::initialize_audio_stream(AudioStream& audio_stream) {
  // We use the AVFormatContext's default codec for that
  // specific format/container.
  const AVCodec* av_codec =
      avcodec_find_encoder(av_format_context_->oformat->audio_codec);
  STD_TORCH_CHECK(av_codec != nullptr, "Codec not found");

  AVCodecContext* av_codec_context = avcodec_alloc_context3(av_codec);
  STD_TORCH_CHECK(
      av_codec_context != nullptr, "Couldn't allocate codec context.");
  audio_stream.av_codec_context.reset(av_codec_context);

  auto desired_bit_rate = audio_stream.options.bit_rate;
  if (desired_bit_rate.has_value()) {
    STD_TORCH_CHECK(
        *desired_bit_rate >= 0,
        "bit_rate=",
        *desired_bit_rate,
        " must be >= 0.");
  }
  // bit_rate=None defaults to 0, which is what the FFmpeg CLI seems to use as
  // well when "-b:a" isn't specified.
  audio_stream.av_codec_context->bit_rate = desired_bit_rate.value_or(0);

  int out_num_channels =
      audio_stream.options.num_channels.value_or(audio_stream.in_num_channels);
  audio_stream.out_num_channels = out_num_channels;
  validate_num_channels(*av_codec, out_num_channels);
  set_default_channel_layout(audio_stream.av_codec_context, out_num_channels);

  int out_sample_rate =
      audio_stream.options.sample_rate.value_or(audio_stream.in_sample_rate);
  audio_stream.out_sample_rate = out_sample_rate;
  validate_sample_rate(*av_codec, out_sample_rate);
  audio_stream.av_codec_context->sample_rate = out_sample_rate;
  audio_stream.av_codec_context->time_base = AVRational{1, out_sample_rate};

  // Input samples are expected to be FLTP. Not all encoders support FLTP, so we
  // may need to convert the samples into a supported output sample format,
  // which is what the `.sample_fmt` defines.
  audio_stream.av_codec_context->sample_fmt =
      find_best_output_sample_format(*av_codec);

  int status =
      avcodec_open2(audio_stream.av_codec_context.get(), av_codec, nullptr);
  STD_TORCH_CHECK(
      status == AVSUCCESS,
      "avcodec_open2 failed: ",
      get_ffmpeg_error_string_from_error_code(status));

  // We're allocating the stream here. Streams are meant to be freed by
  // avformat_free_context(avFormatContext), which we call in the
  // avFormatContext_'s destructor.
  audio_stream.av_stream =
      avformat_new_stream(av_format_context_.get(), nullptr);
  STD_TORCH_CHECK(
      audio_stream.av_stream != nullptr, "Couldn't create new audio stream.");

  status = avcodec_parameters_from_context(
      audio_stream.av_stream->codecpar, audio_stream.av_codec_context.get());
  STD_TORCH_CHECK(
      status == AVSUCCESS,
      "avcodec_parameters_from_context failed: ",
      get_ffmpeg_error_string_from_error_code(status));

  // If a codec supports variable frame size, frame_size may not be defined, in
  // which case we default to 256 like torchaudio.
  audio_stream.frame_size = audio_stream.av_codec_context->frame_size > 0
      ? audio_stream.av_codec_context->frame_size
      : 256;

  // We always create a FIFO so that addSamples() can be called multiple times
  // with various chunk sizes that are then buffered and encoded in frame_size
  // sized batches.
  auto av_audio_fifo = av_audio_fifo_alloc(
      audio_stream.av_codec_context->sample_fmt,
      out_num_channels,
      audio_stream.frame_size * 2);
  STD_TORCH_CHECK(av_audio_fifo != nullptr, "Couldn't create AVAudioFifo.");
  audio_stream.av_audio_fifo.reset(av_audio_fifo);
}

void MultiStreamEncoder::open_streams_and_write_header() {
  STD_TORCH_CHECK(
      !video_streams_.empty() || !audio_streams_.empty(),
      "Call addVideoStream() or addAudioStream() before open().");

  for (auto& video_stream : video_streams_) {
    initialize_video_stream(video_stream);
  }
  for (auto& audio_stream : audio_streams_) {
    initialize_audio_stream(audio_stream);
  }

  int status = avformat_write_header(
      av_format_context_.get(), av_format_options_.get_address());
  STD_TORCH_CHECK(
      status == AVSUCCESS,
      "Error in avformat_write_header: ",
      get_ffmpeg_error_string_from_error_code(status));
  header_written_ = true;
}

void MultiStreamEncoder::add_frames(
    const torch::stable::Tensor& frames,
    int stream_index) {
  STD_TORCH_CHECK(!closed_, "Cannot add frames after close() was called.");
  STD_TORCH_CHECK(header_written_, "Call open() before addFrames().");
  STD_TORCH_CHECK(
      stream_index >= 0 &&
          stream_index < static_cast<int>(video_streams_.size()),
      "Invalid stream index ",
      stream_index,
      ". Number of video streams: ",
      video_streams_.size());
  auto& video_stream = video_streams_[stream_index];
  auto validated_frames = validate_frames(
      frames,
      video_stream.av_codec_context.get(),
      video_stream.device_interface.get());

  AutoAVPacket auto_av_packet;
  // TODO MultiStreamEncoder: Consider using accessor for potential performance
  // improvement
  int num_frames = static_cast<int>(validated_frames.sizes()[0]);
  for (int i = 0; i < num_frames; ++i) {
    torch::stable::Tensor curr_frame = select_row(validated_frames, i);
    int frame_index = video_stream.num_encoded_frames + i;
    UniqueAVFrame av_frame =
        video_stream.device_interface->convert_tensor_to_av_frame_for_encoding(
            curr_frame, frame_index, video_stream.av_codec_context.get());
    STD_TORCH_CHECK(
        av_frame != nullptr,
        "convertTensorToAVFrameForEncoding failed for frame ",
        frame_index,
        " on device: ",
        device_type_name(validated_frames.device().type()));
    encode_video_frame(auto_av_packet, av_frame, video_stream);
  }
  video_stream.num_encoded_frames += num_frames;
}

void MultiStreamEncoder::encode_video_frame(
    AutoAVPacket& auto_av_packet,
    const UniqueAVFrame& av_frame,
    VideoStream& video_stream) {
  auto status =
      avcodec_send_frame(video_stream.av_codec_context.get(), av_frame.get());
  STD_TORCH_CHECK(
      status == AVSUCCESS,
      "Error while sending frame: ",
      get_ffmpeg_error_string_from_error_code(status));

  while (status >= 0) {
    ReferenceAVPacket packet(auto_av_packet);
    status = avcodec_receive_packet(
        video_stream.av_codec_context.get(), packet.get());
    if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) {
      if (status == AVERROR_EOF) {
        // Flush remaining buffered packets
        status = av_interleaved_write_frame(av_format_context_.get(), nullptr);
        STD_TORCH_CHECK(
            status == AVSUCCESS,
            "Failed to flush packet: ",
            get_ffmpeg_error_string_from_error_code(status));
      }
      return;
    }
    STD_TORCH_CHECK(
        status >= 0,
        "Error receiving packet: ",
        get_ffmpeg_error_string_from_error_code(status));

    // The code below is borrowed from torchaudio:
    // https://github.com/pytorch/audio/blob/b6a3368a45aaafe05f1a6a9f10c68adc5e944d9e/src/libtorio/ffmpeg/stream_writer/encoder.cpp#L46
    // Setting packet->duration to 1 allows the last frame to be properly
    // encoded, and needs to be set before calling av_packet_rescale_ts.
    if (packet->duration == 0) {
      packet->duration = 1;
    }
    av_packet_rescale_ts(
        packet.get(),
        video_stream.av_codec_context->time_base,
        video_stream.av_stream->time_base);
    packet->stream_index = video_stream.av_stream->index;

    status = av_interleaved_write_frame(av_format_context_.get(), packet.get());
    STD_TORCH_CHECK(
        status == AVSUCCESS,
        "Error in av_interleaved_write_frame: ",
        get_ffmpeg_error_string_from_error_code(status));
  }
}

void MultiStreamEncoder::add_samples(
    const torch::stable::Tensor& samples,
    int stream_index) {
  STD_TORCH_CHECK(!closed_, "Cannot add samples after close() was called.");
  STD_TORCH_CHECK(header_written_, "Call open() before addSamples().");
  STD_TORCH_CHECK(
      stream_index >= 0 &&
          stream_index < static_cast<int>(audio_streams_.size()),
      "Invalid stream index ",
      stream_index,
      ". Number of audio streams: ",
      audio_streams_.size());
  auto& audio_stream = audio_streams_[stream_index];
  auto validated_samples = validate_samples(samples);
  STD_TORCH_CHECK(
      static_cast<int>(validated_samples.sizes()[0]) ==
          audio_stream.in_num_channels,
      "Expected ",
      audio_stream.in_num_channels,
      " channels, got ",
      validated_samples.sizes()[0]);
  encode_audio_samples(validated_samples, audio_stream);
}

void MultiStreamEncoder::encode_audio_samples(
    const torch::stable::Tensor& samples,
    AudioStream& audio_stream) {
  UniqueAVFrame av_frame = allocate_av_frame(
      audio_stream.frame_size,
      audio_stream.in_sample_rate,
      audio_stream.in_num_channels,
      AV_SAMPLE_FMT_FLTP);

  AutoAVPacket auto_av_packet;

  const uint8_t* psamples =
      static_cast<const uint8_t*>(samples.const_data_ptr());
  int num_samples = static_cast<int>(samples.sizes()[1]); // per channel
  int num_encoded_samples = 0; // per channel
  int num_bytes_per_sample = static_cast<int>(samples.element_size());
  int num_bytes_per_channel = num_samples * num_bytes_per_sample;

  while (num_encoded_samples < num_samples) {
    int num_samples_to_encode =
        std::min(audio_stream.frame_size, num_samples - num_encoded_samples);
    int num_bytes_to_encode = num_samples_to_encode * num_bytes_per_sample;

    for (int ch = 0; ch < audio_stream.in_num_channels; ch++) {
      std::memcpy(
          av_frame->data[ch],
          psamples + ch * num_bytes_per_channel,
          num_bytes_to_encode);
    }
    psamples += num_bytes_to_encode;

    // Above, we set the AVFrame's .nb_samples to AVCodecContext.frame_size so
    // that the frame buffers are allocated to a big enough size. Here, we reset
    // it to the exact number of samples that need to be encoded, otherwise the
    // encoded frame would contain more samples than necessary and our results
    // wouldn't match the ffmpeg CLI.
    av_frame->nb_samples = num_samples_to_encode;

    UniqueAVFrame converted_av_frame =
        maybe_convert_audio_av_frame(av_frame, audio_stream);
    encode_audio_frame_through_fifo(
        auto_av_packet, converted_av_frame, audio_stream);

    num_encoded_samples += num_samples_to_encode;
  }
  STD_TORCH_CHECK(
      num_encoded_samples == num_samples, "Hmmmmmm something went wrong.");
}

UniqueAVFrame MultiStreamEncoder::maybe_convert_audio_av_frame(
    const UniqueAVFrame& av_frame,
    AudioStream& audio_stream) {
  if (static_cast<AVSampleFormat>(av_frame->format) ==
          audio_stream.av_codec_context->sample_fmt &&
      get_num_channels(av_frame) == audio_stream.out_num_channels &&
      av_frame->sample_rate == audio_stream.out_sample_rate) {
    // Note: the clone references the same underlying data, it's a cheap copy.
    return UniqueAVFrame(av_frame_clone(av_frame.get()));
  }

  if (!audio_stream.swr_context) {
    audio_stream.swr_context.reset(create_swr_context(
        static_cast<AVSampleFormat>(av_frame->format),
        audio_stream.av_codec_context->sample_fmt,
        av_frame->sample_rate,
        audio_stream.out_sample_rate,
        av_frame,
        audio_stream.out_num_channels));
  }
  // convertAudioAVFrameSamples uses avFrame's extended_data field, so we ensure
  // it's the same as data. This should always be the case since we validated
  // earlier that we have less than AV_NUM_DATA_POINTERS channels.
  STD_TORCH_CHECK(
      av_frame->data == av_frame->extended_data,
      "Codec context data and extended_data pointers differ, this is unexpected.");
  UniqueAVFrame converted_av_frame = convert_audio_av_frame_samples(
      audio_stream.swr_context,
      av_frame,
      audio_stream.av_codec_context->sample_fmt,
      audio_stream.out_sample_rate,
      audio_stream.out_num_channels);

  if (av_frame->sample_rate == audio_stream.out_sample_rate) {
    STD_TORCH_CHECK(
        converted_av_frame->nb_samples == av_frame->nb_samples,
        "convertedAVFrame->nb_samples=",
        converted_av_frame->nb_samples,
        " differs from ",
        "avFrame->nb_samples=",
        av_frame->nb_samples,
        "This is unexpected, please report on the TorchCodec bug tracker.");
  }
  return converted_av_frame;
}

void MultiStreamEncoder::encode_audio_frame_through_fifo(
    AutoAVPacket& auto_av_packet,
    const UniqueAVFrame& av_frame,
    AudioStream& audio_stream,
    // flushFifo is only set to true in maybeFlushSwrAndFifo(), i.e. at the very
    // end of the encoding process when we're flushing buffers. We also want to
    // flush the FIFO so as to not leave any remaining samples in it.
    bool flush_fifo) {
  if (av_frame != nullptr) {
    int num_samples_written = av_audio_fifo_write(
        audio_stream.av_audio_fifo.get(),
        reinterpret_cast<void**>(av_frame->data),
        av_frame->nb_samples);
    STD_TORCH_CHECK(
        num_samples_written == av_frame->nb_samples,
        "Tried to write ",
        av_frame->nb_samples,
        " samples, but only wrote ",
        num_samples_written);
  }

  UniqueAVFrame newav_frame = allocate_av_frame(
      audio_stream.frame_size,
      audio_stream.av_codec_context->sample_rate,
      audio_stream.out_num_channels,
      audio_stream.av_codec_context->sample_fmt);

  // Explaining the while bound:
  // - if we're not flushing the FIFO, i.e. in most cases, we want to pull
  //   exactly `frame_size` samples from the FIFO, so we have to stop before it
  //   contains less than `frame_size` samples.
  // - if we're flushing the FIFO, we want to read from the FIFO until the very
  //   last sample it contains.
  //
  // In both cases, for as long as we can, we're trying to pull exactly
  // `frame_size` samples from the FIFO and send each `frame_size`-sized avFrame
  // to encodeAudioFrame(). Only the very last avFrame of the encoding process
  // is allowed to contain less than frame_size samples. That only happens when
  // flushFifo is true.
  while (av_audio_fifo_size(audio_stream.av_audio_fifo.get()) >=
         (flush_fifo ? 1 : audio_stream.frame_size)) {
    int samples_to_read = std::min(
        av_audio_fifo_size(audio_stream.av_audio_fifo.get()),
        newav_frame->nb_samples);
    int num_samples_read = av_audio_fifo_read(
        audio_stream.av_audio_fifo.get(),
        reinterpret_cast<void**>(newav_frame->data),
        samples_to_read);
    STD_TORCH_CHECK(
        num_samples_read == samples_to_read,
        "Tried to read ",
        samples_to_read,
        " samples, but only read ",
        num_samples_read);

    newav_frame->nb_samples = num_samples_read;
    encode_audio_frame(auto_av_packet, newav_frame, audio_stream);
  }
}

void MultiStreamEncoder::encode_audio_frame(
    AutoAVPacket& auto_av_packet,
    const UniqueAVFrame& av_frame,
    AudioStream& audio_stream) {
  if (av_frame != nullptr) {
    av_frame->pts = audio_stream.last_encoded_av_frame_pts;
    audio_stream.last_encoded_av_frame_pts += av_frame->nb_samples;
  }

  auto status =
      avcodec_send_frame(audio_stream.av_codec_context.get(), av_frame.get());
  STD_TORCH_CHECK(
      status == AVSUCCESS,
      "Error while sending frame: ",
      get_ffmpeg_error_string_from_error_code(status));

  while (status >= 0) {
    ReferenceAVPacket packet(auto_av_packet);
    status = avcodec_receive_packet(
        audio_stream.av_codec_context.get(), packet.get());
    if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) {
      if (status == AVERROR_EOF) {
        // Flush the packets that were potentially buffered by
        // av_interleaved_write_frame(). See corresponding block in
        // TorchAudio:
        // https://github.com/pytorch/audio/blob/d60ce09e2c532d5bf2e05619e700ab520543465e/src/libtorio/ffmpeg/stream_writer/encoder.cpp#L21
        status = av_interleaved_write_frame(av_format_context_.get(), nullptr);
        STD_TORCH_CHECK(
            status == AVSUCCESS,
            "Failed to flush packet: ",
            get_ffmpeg_error_string_from_error_code(status));
      }
      return;
    }
    STD_TORCH_CHECK(
        status >= 0,
        "Error receiving packet: ",
        get_ffmpeg_error_string_from_error_code(status));

    packet->stream_index = audio_stream.av_stream->index;
    av_packet_rescale_ts(
        packet.get(),
        audio_stream.av_codec_context->time_base,
        audio_stream.av_stream->time_base);

    status = av_interleaved_write_frame(av_format_context_.get(), packet.get());
    STD_TORCH_CHECK(
        status == AVSUCCESS,
        "Error in av_interleaved_write_frame: ",
        get_ffmpeg_error_string_from_error_code(status));
  }
}

void MultiStreamEncoder::maybe_flush_swr_and_fifo(
    AutoAVPacket& auto_av_packet,
    AudioStream& audio_stream) {
  // When sample conversion is involved, libswresample may have buffered some
  // samples that we need to flush into the FIFO before draining it.
  UniqueAVFrame swr_frame(nullptr);
  if (audio_stream.swr_context != nullptr) {
    int num_remaining_samples = // this is an upper bound
        swr_get_out_samples(audio_stream.swr_context.get(), 0);
    if (num_remaining_samples > 0) {
      swr_frame = allocate_av_frame(
          num_remaining_samples,
          audio_stream.out_sample_rate,
          audio_stream.out_num_channels,
          audio_stream.av_codec_context->sample_fmt);
      int actual_num_remaining_samples = swr_convert(
          audio_stream.swr_context.get(),
          swr_frame->data,
          swr_frame->nb_samples,
          nullptr,
          0);
      swr_frame->nb_samples = actual_num_remaining_samples;
    }
  }

  // Flush any remaining swr samples into the FIFO, then drain it.
  encode_audio_frame_through_fifo(
      auto_av_packet, swr_frame, audio_stream, /*flushFifo=*/true);
}

void MultiStreamEncoder::flush_buffers() {
  for (auto& audio_stream : audio_streams_) {
    if (audio_stream.av_stream != nullptr) {
      AutoAVPacket audio_av_packet;
      maybe_flush_swr_and_fifo(audio_av_packet, audio_stream);
      encode_audio_frame(audio_av_packet, UniqueAVFrame(nullptr), audio_stream);
    }
  }
  for (auto& video_stream : video_streams_) {
    if (video_stream.av_stream != nullptr) {
      AutoAVPacket video_av_packet;
      encode_video_frame(video_av_packet, UniqueAVFrame(nullptr), video_stream);
    }
  }
}

void MultiStreamEncoder::close() {
  if (closed_) {
    return;
  }
  // TODO MultiStreamEncoder: Revisit if "closed_" flag is useful
  closed_ = true;

  if (header_written_) {
    flush_buffers();

    int status = av_write_trailer(av_format_context_.get());
    // av_write_trailer returns mfra atom size (positive) for fragmented
    // containers. All FFmpeg errors are negative, so positive is not an error.
    if (status > 0) {
      status = AVSUCCESS;
    }
    STD_TORCH_CHECK(
        status == AVSUCCESS,
        "Error in av_write_trailer: ",
        get_ffmpeg_error_string_from_error_code(status));
  }

  close_avio_context(av_format_context_.get(), avio_context_holder_.get());
}

} // namespace facebook::torchcodec
