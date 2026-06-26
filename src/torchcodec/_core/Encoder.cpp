#include <algorithm>
#include <sstream>

#include "Encoder.h"
#include "TCError.h"

#include <cstring>

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {

namespace {

tc::Tensor validateSamples(const tc::Tensor& samples) {
  TC_CHECK(
      samples.scalar_type() == tc::kFloat32,
      "samples must have float32 dtype, got ",
      (samples.scalar_type()));
  TC_CHECK(
      samples.device().type() == tc::kCPU,
      "samples must be on CPU, got ",
      deviceTypeName(samples.device().type()));
  TC_CHECK(
      samples.dim() == 2,
      "samples must have 2 dimensions, got ",
      samples.dim());

  // We enforce this, but if we get user reports we should investigate whether
  // that's actually needed.
  int numChannels = static_cast<int>(samples.sizes()[0]);
  TC_CHECK(
      numChannels <= AV_NUM_DATA_POINTERS,
      "Trying to encode ",
      numChannels,
      " channels, but FFmpeg only supports ",
      AV_NUM_DATA_POINTERS,
      " channels per frame.");

  return tc::contiguous(samples);
}

void validateSampleRate(const AVCodec& avCodec, int sampleRate) {
  const int* supportedSampleRates = getSupportedSampleRates(avCodec);
  if (supportedSampleRates == nullptr) {
    return;
  }

  for (auto i = 0; supportedSampleRates[i] != 0; ++i) {
    if (sampleRate == supportedSampleRates[i]) {
      return;
    }
  }
  std::stringstream supportedRates;
  for (auto i = 0; supportedSampleRates[i] != 0; ++i) {
    if (i > 0) {
      supportedRates << ", ";
    }
    supportedRates << supportedSampleRates[i];
  }

  TC_CHECK(
      false,
      "invalid sample rate=",
      sampleRate,
      ". Supported sample rate values are: ",
      supportedRates.str());
}

static const std::vector<AVSampleFormat> preferredFormatsOrder = {
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

AVSampleFormat findBestOutputSampleFormat(const AVCodec& avCodec) {
  const AVSampleFormat* supportedSampleFormats =
      getSupportedOutputSampleFormats(avCodec);

  // Find a sample format that the encoder supports. We prefer using FLT[P],
  // since this is the format of the input samples. If FLTP isn't supported
  // then we'll need to convert the AVFrame's format. Our heuristic is to encode
  // into the format with the highest resolution.
  if (supportedSampleFormats == nullptr) {
    // Can't really validate anything in this case, best we can do is hope that
    // FLTP is supported by the encoder. If not, FFmpeg will raise.
    return AV_SAMPLE_FMT_FLTP;
  }

  for (AVSampleFormat preferredFormat : preferredFormatsOrder) {
    for (int i = 0; supportedSampleFormats[i] != -1; ++i) {
      if (supportedSampleFormats[i] == preferredFormat) {
        return preferredFormat;
      }
    }
  }
  // We should always find a match in preferredFormatsOrder, so we should always
  // return earlier. But in the event that a future FFmpeg version defines an
  // additional sample format that isn't in preferredFormatsOrder, we fallback:
  return supportedSampleFormats[0];
}

void closeAVIOContext(
    AVFormatContext* avFormatContext,
    AVIOContextHolder* avioContextHolder) {
  if (!avFormatContext || !avFormatContext->pb) {
    return;
  }

  if (avFormatContext->pb->error == 0) {
    avio_flush(avFormatContext->pb);
  }

  if (!avioContextHolder) {
    if (avFormatContext->pb->error == 0) {
      avio_close(avFormatContext->pb);
    }
  }

  avFormatContext->pb = nullptr;
}

tc::Tensor validateFrames(
    const tc::Tensor& frames,
    const AVCodecContext* avCodecContext = nullptr,
    DeviceInterface* deviceInterface = nullptr) {
  TC_CHECK(
      frames.scalar_type() == tc::kUInt8,
      "frames must have uint8 dtype, got ",
      frames.scalar_type());
  TC_CHECK(
      frames.dim() == 4,
      "frames must have 4 dimensions (N, C, H, W), got ",
      frames.dim());
  TC_CHECK(
      frames.sizes()[1] == 3,
      "frame must have 3 channels (R, G, B), got ",
      frames.sizes()[1]);
  if (deviceInterface != nullptr) {
    auto& expectedDevice = deviceInterface->device();
    auto framesDevice = frames.device();
    TC_CHECK(
        framesDevice == expectedDevice,
        "All frames must be on the same device. Expected ",
        deviceTypeName(expectedDevice.type()),
        ":",
        expectedDevice.index(),
        ", got ",
        deviceTypeName(framesDevice.type()),
        ":",
        framesDevice.index());
  }
  if (avCodecContext) {
    TC_CHECK(
        static_cast<int>(frames.sizes()[2]) == avCodecContext->height &&
            static_cast<int>(frames.sizes()[3]) == avCodecContext->width,
        "All frames must have the same dimensions. Expected height=",
        avCodecContext->height,
        " width=",
        avCodecContext->width,
        ", got height=",
        frames.sizes()[2],
        " width=",
        frames.sizes()[3]);
  }
  return tc::contiguous(frames);
}

AVPixelFormat validatePixelFormat(
    const AVCodec& avCodec,
    const std::string& targetPixelFormat) {
  AVPixelFormat pixelFormat = av_get_pix_fmt(targetPixelFormat.c_str());

  // Validate that the encoder supports this pixel format
  const AVPixelFormat* supportedFormats = getSupportedPixelFormats(avCodec);
  if (supportedFormats != nullptr) {
    for (int i = 0; supportedFormats[i] != AV_PIX_FMT_NONE; ++i) {
      if (supportedFormats[i] == pixelFormat) {
        return pixelFormat;
      }
    }
  }

  std::stringstream errorMsg;
  // av_get_pix_fmt failed to find a pix_fmt
  if (pixelFormat == AV_PIX_FMT_NONE) {
    errorMsg << "Unknown pixel format: " << targetPixelFormat;
  } else {
    errorMsg << "Specified pixel format " << targetPixelFormat
             << " is not supported by the " << avCodec.name << " encoder.";
  }
  // Build error message, similar to FFmpeg's error log
  errorMsg << "\nSupported pixel formats for " << avCodec.name << ":";
  for (int i = 0; supportedFormats[i] != AV_PIX_FMT_NONE; ++i) {
    errorMsg << " " << av_get_pix_fmt_name(supportedFormats[i]);
  }
  TC_CHECK(false, errorMsg.str());
}

void tryToValidateCodecOption(
    const AVCodec& avCodec,
    const char* optionName,
    const std::string& value) {
  if (!avCodec.priv_class) {
    return;
  }
  const AVOption* option = av_opt_find2(
      // Convert obj arg from const AVClass* const* to non-const void*
      // First cast to remove const, then cast to void*
      const_cast<void*>(static_cast<const void*>(&avCodec.priv_class)),
      optionName,
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
      double numericValue = std::stod(value);
      TC_CHECK(
          numericValue >= option->min && numericValue <= option->max,
          optionName,
          "=",
          numericValue,
          " is out of valid range [",
          option->min,
          ", ",
          option->max,
          "] for this codec. For more details, run 'ffmpeg -h encoder=",
          avCodec.name,
          "'");
    } catch (const std::invalid_argument&) {
      TC_CHECK(
          false,
          "Option ",
          optionName,
          " expects a numeric value but got '",
          value,
          "'");
    }
  }
}

void sortCodecOptions(
    const AVFormatContext* avFormatContext,
    const std::map<std::string, std::string>& extraOptions,
    UniqueAVDictionary& codecDict,
    UniqueAVDictionary& formatDict) {
  // Accepts a map of options as input, then sorts them into codec options and
  // format options. The sorted options are returned into two separate dicts.
  const AVClass* formatClass = avformat_get_class();
  const AVClass* muxerClass =
      avFormatContext->oformat ? avFormatContext->oformat->priv_class : nullptr;
  for (const auto& [key, value] : extraOptions) {
    // Check if option is generic format option
    const AVOption* fmtOpt = av_opt_find2(
        &formatClass,
        key.c_str(),
        nullptr,
        0,
        AV_OPT_SEARCH_CHILDREN | AV_OPT_SEARCH_FAKE_OBJ,
        nullptr);
    // Check if option is muxer-specific option
    // (Returned from `ffmpeg -h muxer=mp4`)
    const AVOption* muxerOpt = nullptr;
    if (muxerClass) {
      muxerOpt = av_opt_find2(
          &muxerClass,
          key.c_str(),
          nullptr,
          0,
          AV_OPT_SEARCH_FAKE_OBJ,
          nullptr);
    }
    if (fmtOpt || muxerOpt) {
      // Pass container-format options to formatDict to be used in
      // avformat_write_header
      av_dict_set(formatDict.getAddress(), key.c_str(), value.c_str(), 0);
    } else {
      // By default, pass as codec option to be used in avcodec_open2
      av_dict_set(codecDict.getAddress(), key.c_str(), value.c_str(), 0);
    }
  }
}

} // namespace

MultiStreamEncoder::~MultiStreamEncoder() {
  close();
}

MultiStreamEncoder::MultiStreamEncoder() {
  setFFmpegLogLevel();
}

void MultiStreamEncoder::open(std::string_view fileName) {
  TC_CHECK(!closed_, "Cannot open after close() was called.");
  TC_CHECK(!headerWritten_, "open() was already called.");

  AVFormatContext* avFormatContext = nullptr;
  int status = avformat_alloc_output_context2(
      &avFormatContext, nullptr, nullptr, fileName.data());

  TC_CHECK(
      avFormatContext != nullptr,
      "Couldn't allocate AVFormatContext. ",
      "The destination file is ",
      fileName,
      ", check the desired extension? ",
      getFFMPEGErrorStringFromErrorCode(status));
  avFormatContext_.reset(avFormatContext);

  status = avio_open(&avFormatContext_->pb, fileName.data(), AVIO_FLAG_WRITE);
  TC_CHECK(
      status >= 0,
      "avio_open failed. The destination file is ",
      fileName,
      ", make sure it's a valid path? ",
      getFFMPEGErrorStringFromErrorCode(status));

  openStreamsAndWriteHeader();
}

void MultiStreamEncoder::open(
    std::string_view formatName,
    std::unique_ptr<AVIOContextHolder> avioContextHolder) {
  TC_CHECK(!closed_, "Cannot open after close() was called.");
  TC_CHECK(!headerWritten_, "open() was already called.");

  avioContextHolder_ = std::move(avioContextHolder);

  // Map mkv -> matroska when used as format name
  formatName = (formatName == "mkv") ? "matroska" : formatName;
  AVFormatContext* avFormatContext = nullptr;
  int status = avformat_alloc_output_context2(
      &avFormatContext, nullptr, formatName.data(), nullptr);

  TC_CHECK(
      avFormatContext != nullptr,
      "Couldn't allocate AVFormatContext. ",
      "Check the desired format? Got format=",
      formatName,
      ". ",
      getFFMPEGErrorStringFromErrorCode(status));
  avFormatContext_.reset(avFormatContext);

  avFormatContext_->pb = avioContextHolder_->getAVIOContext();

  openStreamsAndWriteHeader();
}

int MultiStreamEncoder::addVideoStream(
    int height,
    int width,
    double frameRate,
    std::string device,
    std::optional<std::string> codec,
    std::optional<std::string> pixelFormat,
    std::optional<double> crf,
    std::optional<std::string> preset,
    std::optional<std::map<std::string, std::string>> extraOptions) {
  TC_CHECK(height > 0, "height must be > 0, got ", height);
  TC_CHECK(width > 0, "width must be > 0, got ", width);
  TC_CHECK(frameRate > 0, "frame_rate must be > 0, got ", frameRate);
  VideoStream videoStream;
  tc::Device stableDevice(std::move(device));
  videoStream.deviceInterface = createDeviceInterface(
      stableDevice, stableDevice.type() == tc::kCUDA ? "ffmpeg" : "default");
  videoStream.inHeight = height;
  videoStream.inWidth = width;
  videoStream.inFrameRate = frameRate;
  videoStream.options.codec = std::move(codec);
  videoStream.options.pixelFormat = std::move(pixelFormat);
  videoStream.options.crf = crf;
  videoStream.options.preset = std::move(preset);
  videoStream.options.extraOptions = std::move(extraOptions);
  videoStreams_.push_back(std::move(videoStream));
  return static_cast<int>(videoStreams_.size() - 1);
}

int MultiStreamEncoder::addAudioStream(
    int sampleRate,
    int numChannels,
    std::optional<int> bitRate,
    std::optional<int> outNumChannels,
    std::optional<int> outSampleRate) {
  TC_CHECK(sampleRate > 0, "sample_rate must be > 0, got ", sampleRate);
  TC_CHECK(numChannels > 0, "num_channels must be > 0, got ", numChannels);
  TC_CHECK(
      numChannels <= AV_NUM_DATA_POINTERS,
      "Trying to encode ",
      numChannels,
      " channels, but FFmpeg only supports ",
      AV_NUM_DATA_POINTERS,
      " channels per frame.");

  AudioStream audioStream;
  audioStream.inSampleRate = sampleRate;
  audioStream.inNumChannels = numChannels;
  audioStream.options.bitRate = bitRate;
  audioStream.options.numChannels = outNumChannels;
  audioStream.options.sampleRate = outSampleRate;
  audioStreams_.push_back(std::move(audioStream));
  return static_cast<int>(audioStreams_.size() - 1);
}

void MultiStreamEncoder::initializeVideoStream(VideoStream& videoStream) {
  auto deviceType = videoStream.deviceInterface->device().type();

  const AVCodec* avCodec = nullptr;
  // If codec arg is provided, find codec using logic similar to FFmpeg:
  // https://github.com/FFmpeg/FFmpeg/blob/master/fftools/ffmpeg_opt.c#L804-L835
  if (videoStream.options.codec.has_value()) {
    const std::string& codec = videoStream.options.codec.value();
    // Try to find codec by name ("libx264", "libsvtav1")
    avCodec = avcodec_find_encoder_by_name(codec.c_str());
    // Try to find by codec descriptor ("h264", "av1")
    if (!avCodec) {
      const AVCodecDescriptor* desc =
          avcodec_descriptor_get_by_name(codec.c_str());
      if (desc) {
        avCodec = avcodec_find_encoder(desc->id);
      }
    }
  } else {
    TC_CHECK(
        avFormatContext_->oformat != nullptr,
        "Output format is null, unable to find default codec.");
    // Try to substitute the default codec with its hardware equivalent
    // This will return std::nullopt when device is CPU.
    auto hwCodec = videoStream.deviceInterface->findCodec(
        avFormatContext_->oformat->video_codec, /*isDecoder=*/false);
    if (hwCodec.has_value()) {
      avCodec = hwCodec.value();
    }
    if (!avCodec) {
      avCodec = avcodec_find_encoder(avFormatContext_->oformat->video_codec);
    }
  }
  TC_CHECK(
      avCodec != nullptr,
      "Video codec ",
      videoStream.options.codec.has_value()
          ? videoStream.options.codec.value() + " "
          : "",
      "not found. To see available codecs, run: ffmpeg -encoders");

  AVCodecContext* avCodecContext = avcodec_alloc_context3(avCodec);
  TC_CHECK(avCodecContext != nullptr, "Couldn't allocate codec context.");
  videoStream.avCodecContext.reset(avCodecContext);

  int outHeight = videoStream.inHeight;
  int outWidth = videoStream.inWidth;
  AVPixelFormat outPixelFormat = AV_PIX_FMT_NONE;

  if (videoStream.options.pixelFormat.has_value()) {
    if (deviceType == tc::kCUDA) {
      TC_CHECK(
          false,
          "Video encoding on GPU currently only supports the nv12 pixel format. "
          "Do not set pixel_format to use nv12 by default.");
    }
    outPixelFormat =
        validatePixelFormat(*avCodec, videoStream.options.pixelFormat.value());
  } else {
    if (deviceType == tc::kCUDA) {
      // Default to nv12 pixel format when encoding on GPU.
      outPixelFormat = DeviceInterface::CUDA_ENCODING_PIXEL_FORMAT;
    } else {
      const AVPixelFormat* formats = getSupportedPixelFormats(*avCodec);
      // Use first listed pixel format as default (often yuv420p).
      // This is similar to FFmpeg's logic:
      // https://www.ffmpeg.org/doxygen/4.0/decode_8c_source.html#l01087
      // If pixel formats are undefined for some reason, try yuv420p
      outPixelFormat = (formats && formats[0] != AV_PIX_FMT_NONE)
          ? formats[0]
          : AV_PIX_FMT_YUV420P;
    }
  }

  // Configure codec parameters
  videoStream.avCodecContext->codec_id = avCodec->id;
  videoStream.avCodecContext->width = outWidth;
  videoStream.avCodecContext->height = outHeight;
  videoStream.avCodecContext->pix_fmt = outPixelFormat;
  videoStream.avCodecContext->framerate =
      av_d2q(videoStream.inFrameRate, INT_MAX);
  videoStream.avCodecContext->time_base =
      av_inv_q(videoStream.avCodecContext->framerate);

  // Set flag for containers that require extradata to be in the codec context
  if (avFormatContext_->oformat->flags & AVFMT_GLOBALHEADER) {
    videoStream.avCodecContext->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }

  // Apply videoStreamOptions
  UniqueAVDictionary avCodecOptions;
  if (videoStream.options.extraOptions.has_value()) {
    for (const auto& [key, value] : videoStream.options.extraOptions.value()) {
      tryToValidateCodecOption(*avCodec, key.c_str(), value);
    }
    sortCodecOptions(
        avFormatContext_.get(),
        videoStream.options.extraOptions.value(),
        avCodecOptions,
        avFormatOptions_);
  }

  if (videoStream.options.crf.has_value()) {
    std::string crfValue = std::to_string(videoStream.options.crf.value());
    tryToValidateCodecOption(*avCodec, "crf", crfValue);
    av_dict_set(avCodecOptions.getAddress(), "crf", crfValue.c_str(), 0);
  }

  if (videoStream.options.preset.has_value()) {
    av_dict_set(
        avCodecOptions.getAddress(),
        "preset",
        videoStream.options.preset.value().c_str(),
        0);
  }

  if (deviceType == tc::kCUDA) {
    videoStream.deviceInterface->registerHardwareDeviceWithCodec(
        videoStream.avCodecContext.get());
    videoStream.deviceInterface->setupHardwareFrameContextForEncoding(
        videoStream.avCodecContext.get());
  }

  int status = avcodec_open2(
      videoStream.avCodecContext.get(), avCodec, avCodecOptions.getAddress());

  TC_CHECK(
      status == AVSUCCESS,
      "avcodec_open2 failed: ",
      getFFMPEGErrorStringFromErrorCode(status));

  videoStream.avStream = avformat_new_stream(avFormatContext_.get(), nullptr);
  TC_CHECK(videoStream.avStream != nullptr, "Couldn't create new stream.");

  // Set the stream time base to encode correct frame timestamps
  videoStream.avStream->time_base = videoStream.avCodecContext->time_base;
  // Set the stream frame rate to store correct frame durations for some
  // containers (webm, mkv)
  videoStream.avStream->r_frame_rate = videoStream.avCodecContext->framerate;

  status = avcodec_parameters_from_context(
      videoStream.avStream->codecpar, videoStream.avCodecContext.get());
  TC_CHECK(
      status == AVSUCCESS,
      "avcodec_parameters_from_context failed: ",
      getFFMPEGErrorStringFromErrorCode(status));
}

void MultiStreamEncoder::initializeAudioStream(AudioStream& audioStream) {
  // We use the AVFormatContext's default codec for that
  // specific format/container.
  const AVCodec* avCodec =
      avcodec_find_encoder(avFormatContext_->oformat->audio_codec);
  TC_CHECK(avCodec != nullptr, "Codec not found");

  AVCodecContext* avCodecContext = avcodec_alloc_context3(avCodec);
  TC_CHECK(avCodecContext != nullptr, "Couldn't allocate codec context.");
  audioStream.avCodecContext.reset(avCodecContext);

  auto desiredBitRate = audioStream.options.bitRate;
  if (desiredBitRate.has_value()) {
    TC_CHECK(
        *desiredBitRate >= 0, "bit_rate=", *desiredBitRate, " must be >= 0.");
  }
  // bit_rate=None defaults to 0, which is what the FFmpeg CLI seems to use as
  // well when "-b:a" isn't specified.
  audioStream.avCodecContext->bit_rate = desiredBitRate.value_or(0);

  int outNumChannels =
      audioStream.options.numChannels.value_or(audioStream.inNumChannels);
  audioStream.outNumChannels = outNumChannels;
  validateNumChannels(*avCodec, outNumChannels);
  setDefaultChannelLayout(audioStream.avCodecContext, outNumChannels);

  int outSampleRate =
      audioStream.options.sampleRate.value_or(audioStream.inSampleRate);
  audioStream.outSampleRate = outSampleRate;
  validateSampleRate(*avCodec, outSampleRate);
  audioStream.avCodecContext->sample_rate = outSampleRate;
  audioStream.avCodecContext->time_base = AVRational{1, outSampleRate};

  // Input samples are expected to be FLTP. Not all encoders support FLTP, so we
  // may need to convert the samples into a supported output sample format,
  // which is what the `.sample_fmt` defines.
  audioStream.avCodecContext->sample_fmt = findBestOutputSampleFormat(*avCodec);

  int status =
      avcodec_open2(audioStream.avCodecContext.get(), avCodec, nullptr);
  TC_CHECK(
      status == AVSUCCESS,
      "avcodec_open2 failed: ",
      getFFMPEGErrorStringFromErrorCode(status));

  // We're allocating the stream here. Streams are meant to be freed by
  // avformat_free_context(avFormatContext), which we call in the
  // avFormatContext_'s destructor.
  audioStream.avStream = avformat_new_stream(avFormatContext_.get(), nullptr);
  TC_CHECK(
      audioStream.avStream != nullptr, "Couldn't create new audio stream.");

  status = avcodec_parameters_from_context(
      audioStream.avStream->codecpar, audioStream.avCodecContext.get());
  TC_CHECK(
      status == AVSUCCESS,
      "avcodec_parameters_from_context failed: ",
      getFFMPEGErrorStringFromErrorCode(status));

  // If a codec supports variable frame size, frame_size may not be defined, in
  // which case we default to 256 like torchaudio.
  audioStream.frameSize = audioStream.avCodecContext->frame_size > 0
      ? audioStream.avCodecContext->frame_size
      : 256;

  // We always create a FIFO so that addSamples() can be called multiple times
  // with various chunk sizes that are then buffered and encoded in frame_size
  // sized batches.
  auto avAudioFifo = av_audio_fifo_alloc(
      audioStream.avCodecContext->sample_fmt,
      outNumChannels,
      audioStream.frameSize * 2);
  TC_CHECK(avAudioFifo != nullptr, "Couldn't create AVAudioFifo.");
  audioStream.avAudioFifo.reset(avAudioFifo);
}

void MultiStreamEncoder::openStreamsAndWriteHeader() {
  TC_CHECK(
      !videoStreams_.empty() || !audioStreams_.empty(),
      "Call addVideoStream() or addAudioStream() before open().");

  for (auto& videoStream : videoStreams_) {
    initializeVideoStream(videoStream);
  }
  for (auto& audioStream : audioStreams_) {
    initializeAudioStream(audioStream);
  }

  int status = avformat_write_header(
      avFormatContext_.get(), avFormatOptions_.getAddress());
  TC_CHECK(
      status == AVSUCCESS,
      "Error in avformat_write_header: ",
      getFFMPEGErrorStringFromErrorCode(status));
  headerWritten_ = true;
}

void MultiStreamEncoder::addFrames(const tc::Tensor& frames, int streamIndex) {
  TC_CHECK(!closed_, "Cannot add frames after close() was called.");
  TC_CHECK(headerWritten_, "Call open() before addFrames().");
  TC_CHECK(
      streamIndex >= 0 && streamIndex < static_cast<int>(videoStreams_.size()),
      "Invalid stream index ",
      streamIndex,
      ". Number of video streams: ",
      videoStreams_.size());
  auto& videoStream = videoStreams_[streamIndex];
  auto validatedFrames = validateFrames(
      frames,
      videoStream.avCodecContext.get(),
      videoStream.deviceInterface.get());

  AutoAVPacket autoAVPacket;
  // TODO MultiStreamEncoder: Consider using accessor for potential performance
  // improvement
  int numFrames = static_cast<int>(validatedFrames.sizes()[0]);
  for (int i = 0; i < numFrames; ++i) {
    tc::Tensor currFrame = tc::selectRow(validatedFrames, i);
    int frameIndex = videoStream.numEncodedFrames + i;
    UniqueAVFrame avFrame =
        videoStream.deviceInterface->convertTensorToAVFrameForEncoding(
            currFrame, frameIndex, videoStream.avCodecContext.get());
    TC_CHECK(
        avFrame != nullptr,
        "convertTensorToAVFrameForEncoding failed for frame ",
        frameIndex,
        " on device: ",
        deviceTypeName(validatedFrames.device().type()));
    encodeVideoFrame(autoAVPacket, avFrame, videoStream);
  }
  videoStream.numEncodedFrames += numFrames;
}

void MultiStreamEncoder::encodeVideoFrame(
    AutoAVPacket& autoAVPacket,
    const UniqueAVFrame& avFrame,
    VideoStream& videoStream) {
  auto status =
      avcodec_send_frame(videoStream.avCodecContext.get(), avFrame.get());
  TC_CHECK(
      status == AVSUCCESS,
      "Error while sending frame: ",
      getFFMPEGErrorStringFromErrorCode(status));

  while (status >= 0) {
    ReferenceAVPacket packet(autoAVPacket);
    status =
        avcodec_receive_packet(videoStream.avCodecContext.get(), packet.get());
    if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) {
      if (status == AVERROR_EOF) {
        // Flush remaining buffered packets
        status = av_interleaved_write_frame(avFormatContext_.get(), nullptr);
        TC_CHECK(
            status == AVSUCCESS,
            "Failed to flush packet: ",
            getFFMPEGErrorStringFromErrorCode(status));
      }
      return;
    }
    TC_CHECK(
        status >= 0,
        "Error receiving packet: ",
        getFFMPEGErrorStringFromErrorCode(status));

    // The code below is borrowed from torchaudio:
    // https://github.com/pytorch/audio/blob/b6a3368a45aaafe05f1a6a9f10c68adc5e944d9e/src/libtorio/ffmpeg/stream_writer/encoder.cpp#L46
    // Setting packet->duration to 1 allows the last frame to be properly
    // encoded, and needs to be set before calling av_packet_rescale_ts.
    if (packet->duration == 0) {
      packet->duration = 1;
    }
    av_packet_rescale_ts(
        packet.get(),
        videoStream.avCodecContext->time_base,
        videoStream.avStream->time_base);
    packet->stream_index = videoStream.avStream->index;

    status = av_interleaved_write_frame(avFormatContext_.get(), packet.get());
    TC_CHECK(
        status == AVSUCCESS,
        "Error in av_interleaved_write_frame: ",
        getFFMPEGErrorStringFromErrorCode(status));
  }
}

void MultiStreamEncoder::addSamples(
    const tc::Tensor& samples,
    int streamIndex) {
  TC_CHECK(!closed_, "Cannot add samples after close() was called.");
  TC_CHECK(headerWritten_, "Call open() before addSamples().");
  TC_CHECK(
      streamIndex >= 0 && streamIndex < static_cast<int>(audioStreams_.size()),
      "Invalid stream index ",
      streamIndex,
      ". Number of audio streams: ",
      audioStreams_.size());
  auto& audioStream = audioStreams_[streamIndex];
  auto validatedSamples = validateSamples(samples);
  TC_CHECK(
      static_cast<int>(validatedSamples.sizes()[0]) ==
          audioStream.inNumChannels,
      "Expected ",
      audioStream.inNumChannels,
      " channels, got ",
      validatedSamples.sizes()[0]);
  encodeAudioSamples(validatedSamples, audioStream);
}

void MultiStreamEncoder::encodeAudioSamples(
    const tc::Tensor& samples,
    AudioStream& audioStream) {
  UniqueAVFrame avFrame = allocateAVFrame(
      audioStream.frameSize,
      audioStream.inSampleRate,
      audioStream.inNumChannels,
      AV_SAMPLE_FMT_FLTP);

  AutoAVPacket autoAVPacket;

  const uint8_t* psamples =
      static_cast<const uint8_t*>(samples.const_data_ptr());
  int numSamples = static_cast<int>(samples.sizes()[1]); // per channel
  int numEncodedSamples = 0; // per channel
  int numBytesPerSample = static_cast<int>(samples.element_size());
  int numBytesPerChannel = numSamples * numBytesPerSample;

  while (numEncodedSamples < numSamples) {
    int numSamplesToEncode =
        std::min(audioStream.frameSize, numSamples - numEncodedSamples);
    int numBytesToEncode = numSamplesToEncode * numBytesPerSample;

    for (int ch = 0; ch < audioStream.inNumChannels; ch++) {
      std::memcpy(
          avFrame->data[ch],
          psamples + ch * numBytesPerChannel,
          numBytesToEncode);
    }
    psamples += numBytesToEncode;

    // Above, we set the AVFrame's .nb_samples to AVCodecContext.frame_size so
    // that the frame buffers are allocated to a big enough size. Here, we reset
    // it to the exact number of samples that need to be encoded, otherwise the
    // encoded frame would contain more samples than necessary and our results
    // wouldn't match the ffmpeg CLI.
    avFrame->nb_samples = numSamplesToEncode;

    UniqueAVFrame convertedAVFrame =
        maybeConvertAudioAVFrame(avFrame, audioStream);
    encodeAudioFrameThroughFifo(autoAVPacket, convertedAVFrame, audioStream);

    numEncodedSamples += numSamplesToEncode;
  }
  TC_CHECK(numEncodedSamples == numSamples, "Hmmmmmm something went wrong.");
}

UniqueAVFrame MultiStreamEncoder::maybeConvertAudioAVFrame(
    const UniqueAVFrame& avFrame,
    AudioStream& audioStream) {
  if (static_cast<AVSampleFormat>(avFrame->format) ==
          audioStream.avCodecContext->sample_fmt &&
      getNumChannels(avFrame) == audioStream.outNumChannels &&
      avFrame->sample_rate == audioStream.outSampleRate) {
    // Note: the clone references the same underlying data, it's a cheap copy.
    return UniqueAVFrame(av_frame_clone(avFrame.get()));
  }

  if (!audioStream.swrContext) {
    audioStream.swrContext.reset(createSwrContext(
        static_cast<AVSampleFormat>(avFrame->format),
        audioStream.avCodecContext->sample_fmt,
        avFrame->sample_rate,
        audioStream.outSampleRate,
        avFrame,
        audioStream.outNumChannels));
  }
  // convertAudioAVFrameSamples uses avFrame's extended_data field, so we ensure
  // it's the same as data. This should always be the case since we validated
  // earlier that we have less than AV_NUM_DATA_POINTERS channels.
  TC_CHECK(
      avFrame->data == avFrame->extended_data,
      "Codec context data and extended_data pointers differ, this is unexpected.");
  UniqueAVFrame convertedAVFrame = convertAudioAVFrameSamples(
      audioStream.swrContext,
      avFrame,
      audioStream.avCodecContext->sample_fmt,
      audioStream.outSampleRate,
      audioStream.outNumChannels);

  if (avFrame->sample_rate == audioStream.outSampleRate) {
    TC_CHECK(
        convertedAVFrame->nb_samples == avFrame->nb_samples,
        "convertedAVFrame->nb_samples=",
        convertedAVFrame->nb_samples,
        " differs from ",
        "avFrame->nb_samples=",
        avFrame->nb_samples,
        "This is unexpected, please report on the TorchCodec bug tracker.");
  }
  return convertedAVFrame;
}

void MultiStreamEncoder::encodeAudioFrameThroughFifo(
    AutoAVPacket& autoAVPacket,
    const UniqueAVFrame& avFrame,
    AudioStream& audioStream,
    // flushFifo is only set to true in maybeFlushSwrAndFifo(), i.e. at the very
    // end of the encoding process when we're flushing buffers. We also want to
    // flush the FIFO so as to not leave any remaining samples in it.
    bool flushFifo) {
  if (avFrame != nullptr) {
    int numSamplesWritten = av_audio_fifo_write(
        audioStream.avAudioFifo.get(),
        reinterpret_cast<void**>(avFrame->data),
        avFrame->nb_samples);
    TC_CHECK(
        numSamplesWritten == avFrame->nb_samples,
        "Tried to write ",
        avFrame->nb_samples,
        " samples, but only wrote ",
        numSamplesWritten);
  }

  UniqueAVFrame newavFrame = allocateAVFrame(
      audioStream.frameSize,
      audioStream.avCodecContext->sample_rate,
      audioStream.outNumChannels,
      audioStream.avCodecContext->sample_fmt);

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
  while (av_audio_fifo_size(audioStream.avAudioFifo.get()) >=
         (flushFifo ? 1 : audioStream.frameSize)) {
    int samplesToRead = std::min(
        av_audio_fifo_size(audioStream.avAudioFifo.get()),
        newavFrame->nb_samples);
    int numSamplesRead = av_audio_fifo_read(
        audioStream.avAudioFifo.get(),
        reinterpret_cast<void**>(newavFrame->data),
        samplesToRead);
    TC_CHECK(
        numSamplesRead == samplesToRead,
        "Tried to read ",
        samplesToRead,
        " samples, but only read ",
        numSamplesRead);

    newavFrame->nb_samples = numSamplesRead;
    encodeAudioFrame(autoAVPacket, newavFrame, audioStream);
  }
}

void MultiStreamEncoder::encodeAudioFrame(
    AutoAVPacket& autoAVPacket,
    const UniqueAVFrame& avFrame,
    AudioStream& audioStream) {
  if (avFrame != nullptr) {
    avFrame->pts = audioStream.lastEncodedAVFramePts;
    audioStream.lastEncodedAVFramePts += avFrame->nb_samples;
  }

  auto status =
      avcodec_send_frame(audioStream.avCodecContext.get(), avFrame.get());
  TC_CHECK(
      status == AVSUCCESS,
      "Error while sending frame: ",
      getFFMPEGErrorStringFromErrorCode(status));

  while (status >= 0) {
    ReferenceAVPacket packet(autoAVPacket);
    status =
        avcodec_receive_packet(audioStream.avCodecContext.get(), packet.get());
    if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) {
      if (status == AVERROR_EOF) {
        // Flush the packets that were potentially buffered by
        // av_interleaved_write_frame(). See corresponding block in
        // TorchAudio:
        // https://github.com/pytorch/audio/blob/d60ce09e2c532d5bf2e05619e700ab520543465e/src/libtorio/ffmpeg/stream_writer/encoder.cpp#L21
        status = av_interleaved_write_frame(avFormatContext_.get(), nullptr);
        TC_CHECK(
            status == AVSUCCESS,
            "Failed to flush packet: ",
            getFFMPEGErrorStringFromErrorCode(status));
      }
      return;
    }
    TC_CHECK(
        status >= 0,
        "Error receiving packet: ",
        getFFMPEGErrorStringFromErrorCode(status));

    packet->stream_index = audioStream.avStream->index;
    av_packet_rescale_ts(
        packet.get(),
        audioStream.avCodecContext->time_base,
        audioStream.avStream->time_base);

    status = av_interleaved_write_frame(avFormatContext_.get(), packet.get());
    TC_CHECK(
        status == AVSUCCESS,
        "Error in av_interleaved_write_frame: ",
        getFFMPEGErrorStringFromErrorCode(status));
  }
}

void MultiStreamEncoder::maybeFlushSwrAndFifo(
    AutoAVPacket& autoAVPacket,
    AudioStream& audioStream) {
  // When sample conversion is involved, libswresample may have buffered some
  // samples that we need to flush into the FIFO before draining it.
  UniqueAVFrame swrFrame(nullptr);
  if (audioStream.swrContext != nullptr) {
    int numRemainingSamples = // this is an upper bound
        swr_get_out_samples(audioStream.swrContext.get(), 0);
    if (numRemainingSamples > 0) {
      swrFrame = allocateAVFrame(
          numRemainingSamples,
          audioStream.outSampleRate,
          audioStream.outNumChannels,
          audioStream.avCodecContext->sample_fmt);
      int actualNumRemainingSamples = swr_convert(
          audioStream.swrContext.get(),
          swrFrame->data,
          swrFrame->nb_samples,
          nullptr,
          0);
      swrFrame->nb_samples = actualNumRemainingSamples;
    }
  }

  // Flush any remaining swr samples into the FIFO, then drain it.
  encodeAudioFrameThroughFifo(
      autoAVPacket, swrFrame, audioStream, /*flushFifo=*/true);
}

void MultiStreamEncoder::flushBuffers() {
  for (auto& audioStream : audioStreams_) {
    if (audioStream.avStream != nullptr) {
      AutoAVPacket audioAVPacket;
      maybeFlushSwrAndFifo(audioAVPacket, audioStream);
      encodeAudioFrame(audioAVPacket, UniqueAVFrame(nullptr), audioStream);
    }
  }
  for (auto& videoStream : videoStreams_) {
    if (videoStream.avStream != nullptr) {
      AutoAVPacket videoAVPacket;
      encodeVideoFrame(videoAVPacket, UniqueAVFrame(nullptr), videoStream);
    }
  }
}

void MultiStreamEncoder::close() {
  if (closed_) {
    return;
  }
  // TODO MultiStreamEncoder: Revisit if "closed_" flag is useful
  closed_ = true;

  if (headerWritten_) {
    flushBuffers();

    int status = av_write_trailer(avFormatContext_.get());
    // av_write_trailer returns mfra atom size (positive) for fragmented
    // containers. All FFmpeg errors are negative, so positive is not an error.
    if (status > 0) {
      status = AVSUCCESS;
    }
    TC_CHECK(
        status == AVSUCCESS,
        "Error in av_write_trailer: ",
        getFFMPEGErrorStringFromErrorCode(status));
  }

  closeAVIOContext(avFormatContext_.get(), avioContextHolder_.get());
}

} // namespace facebook::torchcodec
