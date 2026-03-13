// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ToneMap.h"

#include <torch/headeronly/util/Exception.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/hdr_dynamic_metadata.h>
#include <libavutil/imgutils.h>
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
}

namespace facebook::torchcodec {

namespace {

// ---------------------------------------------------------------------------
// PQ (SMPTE ST 2084) constants and EOTF
// ---------------------------------------------------------------------------
constexpr double PQ_M1 = 0.1593017578125; // = 2610/16384
constexpr double PQ_M2 = 78.84375; // = 2523*128/4096
constexpr double PQ_C1 = 0.8359375; // = 3424/4096
constexpr double PQ_C2 = 18.8515625; // = 2413*32/4096
constexpr double PQ_C3 = 18.6875; // = 2392*32/4096

// PQ EOTF: signal E in [0,1] → linear luminance in nits [0, 10000]
inline double pqEOTF(double E) {
  double Em = std::pow(E, 1.0 / PQ_M2);
  double num = std::max(Em - PQ_C1, 0.0);
  double den = PQ_C2 - PQ_C3 * Em;
  if (den <= 0.0) {
    return 0.0;
  }
  return 10000.0 * std::pow(num / den, 1.0 / PQ_M1);
}

// ---------------------------------------------------------------------------
// HLG (ARIB STD-B67) inverse OETF + OOTF
// ---------------------------------------------------------------------------
constexpr double HLG_A = 0.17883277;
constexpr double HLG_B = 0.28466892;
constexpr double HLG_C = 0.55991073;

// HLG inverse OETF: signal E in [0,1] → scene-linear [0,1]
inline double hlgInverseOETF(double E) {
  if (E <= 0.0) {
    return 0.0;
  }
  if (E <= 0.5) {
    return E * E / 3.0;
  }
  return (std::exp((E - HLG_C) / HLG_A) + HLG_B) / 12.0;
}

// ---------------------------------------------------------------------------
// BT.2020 NCL YCbCr → R'G'B' (non-linear signal, [0, 1])
// ---------------------------------------------------------------------------
// BT.2020 coefficients: Kr=0.2627, Kb=0.0593
constexpr double BT2020_KR = 0.2627;
constexpr double BT2020_KB = 0.0593;
constexpr double BT2020_KG = 1.0 - BT2020_KR - BT2020_KB;

// ---------------------------------------------------------------------------
// BT.2020 → BT.709 gamut mapping matrix (3x3, on linear RGB)
//
// Derived from the chromaticity coordinates of BT.2020 and BT.709 primaries
// with the D65 white point. M = M_XYZ_to_709 * M_2020_to_XYZ.
// ---------------------------------------------------------------------------
// clang-format off
constexpr double GAMUT_MAP[3][3] = {
    { 1.6605, -0.5877, -0.0728},
    {-0.1246,  1.1330, -0.0084},
    {-0.0182, -0.1006,  1.1187}
};
// clang-format on

// ---------------------------------------------------------------------------
// Hable (Uncharted 2) tone mapping curve
// ---------------------------------------------------------------------------
constexpr double HABLE_A = 0.15;
constexpr double HABLE_B = 0.50;
constexpr double HABLE_C = 0.10;
constexpr double HABLE_D = 0.20;
constexpr double HABLE_E = 0.02;
constexpr double HABLE_F = 0.30;

inline double hable(double x) {
  return ((x * (HABLE_A * x + HABLE_C * HABLE_B) + HABLE_D * HABLE_E) /
          (x * (HABLE_A * x + HABLE_B) + HABLE_D * HABLE_F)) -
      HABLE_E / HABLE_F;
}

// ---------------------------------------------------------------------------
// BT.709 OETF: linear L [0,1] → non-linear V [0,1]
// ---------------------------------------------------------------------------
inline double bt709OETF(double L) {
  if (L < 0.018) {
    return 4.5 * L;
  }
  return 1.099 * std::pow(L, 0.45) - 0.099;
}

// ---------------------------------------------------------------------------
// Signal peak detection from frame side data
// ---------------------------------------------------------------------------
[[maybe_unused]] double getSignalPeakNits(const AVFrame* frame) {
  // Try to get MaxCLL from content light level metadata.
  const AVFrameSideData* cllSD =
      av_frame_get_side_data(frame, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
  if (cllSD) {
    auto* lightLevel =
        reinterpret_cast<const AVContentLightMetadata*>(cllSD->data);
    if (lightLevel->MaxCLL > 0) {
      return static_cast<double>(lightLevel->MaxCLL);
    }
  }

  // Try mastering display metadata.
  const AVFrameSideData* mdSD =
      av_frame_get_side_data(frame, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
  if (mdSD) {
    auto* mastering =
        reinterpret_cast<const AVMasteringDisplayMetadata*>(mdSD->data);
    if (mastering->has_luminance && av_q2d(mastering->max_luminance) > 0) {
      return av_q2d(mastering->max_luminance);
    }
  }

  // Default: PQ system peak is 10000 nits, HLG nominal peak is 1000 nits.
  if (frame->color_trc == AVCOL_TRC_SMPTE2084) {
    return 10000.0;
  }
  return 1000.0;
}

// ---------------------------------------------------------------------------
// Read a 10-bit sample from a YUV plane.
// Handles both yuv420p10le (planar, 2 bytes per sample little-endian) and
// p010le (semi-planar, 2 bytes per sample with upper 10 bits).
// ---------------------------------------------------------------------------
inline uint16_t read10BitSample(const uint8_t* data, bool isP010) {
  uint16_t val = data[0] | (static_cast<uint16_t>(data[1]) << 8);
  if (isP010) {
    // P010 stores 10-bit values in the upper 10 bits of a 16-bit word.
    val >>= 6;
  }
  return val;
}

// ---------------------------------------------------------------------------
// PQ linearization LUT for 10-bit input (1024 entries).
// Maps normalized signal [0/1023 .. 1023/1023] → linear nits [0, 10000].
// Built once, lazily.
// ---------------------------------------------------------------------------
class PQ_LUT {
 public:
  static const PQ_LUT& instance() {
    static PQ_LUT lut;
    return lut;
  }

  double operator[](int idx) const {
    return table_[idx];
  }

 private:
  PQ_LUT() {
    for (int i = 0; i < 1024; ++i) {
      double signal = static_cast<double>(i) / 1023.0;
      table_[i] = pqEOTF(signal);
    }
  }

  double table_[1024];
};

// ---------------------------------------------------------------------------
// HLG inverse OETF LUT for 10-bit input.
// Maps signal → scene-linear.
// ---------------------------------------------------------------------------
class HLG_LUT {
 public:
  static const HLG_LUT& instance() {
    static HLG_LUT lut;
    return lut;
  }

  double operator[](int idx) const {
    return table_[idx];
  }

 private:
  HLG_LUT() {
    for (int i = 0; i < 1024; ++i) {
      double signal = static_cast<double>(i) / 1023.0;
      table_[i] = hlgInverseOETF(signal);
    }
  }

  double table_[1024];
};

} // namespace

bool isHDRFrame(const AVFrame* frame) {
  return frame->color_trc == AVCOL_TRC_SMPTE2084 ||
      frame->color_trc == AVCOL_TRC_ARIB_STD_B67;
}

UniqueAVFrame toneMapHDRFrame(const UniqueAVFrame& src) {
  const AVFrame* frame = src.get();

  const bool isPQ = (frame->color_trc == AVCOL_TRC_SMPTE2084);
  const bool isHLG = (frame->color_trc == AVCOL_TRC_ARIB_STD_B67);
  STD_TORCH_CHECK(
      isPQ || isHLG,
      "toneMapHDRFrame: unsupported transfer characteristic: ",
      static_cast<int>(frame->color_trc));

  const AVPixelFormat pixFmt = static_cast<AVPixelFormat>(frame->format);
  const bool isP010 = (pixFmt == AV_PIX_FMT_P010LE);
  const bool isYUV420P10 = (pixFmt == AV_PIX_FMT_YUV420P10LE);
  STD_TORCH_CHECK(
      isP010 || isYUV420P10,
      "toneMapHDRFrame: unsupported pixel format: ",
      av_get_pix_fmt_name(pixFmt),
      ". Expected yuv420p10le or p010le.");

  const int width = frame->width;
  const int height = frame->height;
  const bool isLimitedRange = (frame->color_range != AVCOL_RANGE_JPEG);

  // Nominal peak luminance for normalization (matches npl=300 convention).
  constexpr double SDR_WHITE = 300.0;
  // Hardcoded peak in SDR-relative units (matches peak=4 convention).
  // 4.0 × 300 = 1200 nits effective content peak.
  constexpr double peakSDR = 4.0;
  const double hablePeakInv = 1.0 / hable(peakSDR);

  // For HLG, compute system gamma and OOTF parameters.
  // Assume a 1000 nit display by default.
  constexpr double HLG_DISPLAY_LW = 1000.0;
  const double hlgGamma =
      std::max(1.0, 1.2 + 0.42 * std::log10(HLG_DISPLAY_LW / 1000.0));

  // Allocate output frame in RGB24.
  UniqueAVFrame dst(av_frame_alloc());
  STD_TORCH_CHECK(dst != nullptr, "Failed to allocate output AVFrame");
  dst->format = AV_PIX_FMT_RGB24;
  dst->width = width;
  dst->height = height;
  int ret = av_frame_get_buffer(dst.get(), 0);
  STD_TORCH_CHECK(ret >= 0, "Failed to allocate output frame buffer");

  // Tag the output as BT.709 SDR.
  dst->colorspace = AVCOL_SPC_BT709;
  dst->color_primaries = AVCOL_PRI_BT709;
  dst->color_trc = AVCOL_TRC_BT709;
  dst->color_range = AVCOL_RANGE_JPEG; // full range RGB

  // Pointers to source planes.
  const uint8_t* srcY = frame->data[0];
  const int srcYStride = frame->linesize[0];

  // For planar (yuv420p10le): U is data[1], V is data[2]
  // For semi-planar (p010le): UV interleaved in data[1]
  const uint8_t* srcU = frame->data[1];
  const int srcUStride = frame->linesize[1];
  const uint8_t* srcV = isP010 ? nullptr : frame->data[2];
  const int srcVStride = isP010 ? 0 : frame->linesize[2];

  uint8_t* dstData = dst->data[0];
  const int dstStride = dst->linesize[0];

  // Limited range 10-bit: Y [64, 940], UV [64, 960]
  // Full range 10-bit:    Y [0, 1023], UV [0, 1023]
  const double yMin = isLimitedRange ? 64.0 : 0.0;
  const double yRange = isLimitedRange ? (940.0 - 64.0) : 1023.0;
  const double uvMin = isLimitedRange ? 64.0 : 0.0;
  const double uvRange = isLimitedRange ? (960.0 - 64.0) : 1023.0;

  // Pre-derive YCbCr → RGB coefficients from BT.2020 NCL
  // R' = Y' + (2 - 2*Kr) * Cr
  // B' = Y' + (2 - 2*Kb) * Cb
  // G' = (Y' - Kr*R' - Kb*B') / Kg
  const double crToR = 2.0 * (1.0 - BT2020_KR);
  const double cbToB = 2.0 * (1.0 - BT2020_KB);
  const double crToG = -2.0 * BT2020_KR * (1.0 - BT2020_KR) / BT2020_KG;
  const double cbToG = -2.0 * BT2020_KB * (1.0 - BT2020_KB) / BT2020_KG;

  const PQ_LUT& pqLut = PQ_LUT::instance();
  const HLG_LUT& hlgLut = HLG_LUT::instance();

  for (int y = 0; y < height; ++y) {
    const uint8_t* yRow = srcY + y * srcYStride;
    // Chroma is subsampled 2x vertically and horizontally (420).
    const int chromaY = y / 2;
    const uint8_t* uRow = srcU + chromaY * srcUStride;
    const uint8_t* vRow = isP010 ? nullptr : (srcV + chromaY * srcVStride);

    uint8_t* outRow = dstData + y * dstStride;

    for (int x = 0; x < width; ++x) {
      // Read 10-bit Y sample.
      uint16_t yVal = read10BitSample(yRow + x * 2, isP010);

      // Read 10-bit chroma samples (subsampled).
      int chromaX = x / 2;
      uint16_t uVal, vVal;
      if (isP010) {
        // P010: UV interleaved as U0 V0 U1 V1 ...
        uVal = read10BitSample(uRow + chromaX * 4, true);
        vVal = read10BitSample(uRow + chromaX * 4 + 2, true);
      } else {
        // Planar: separate U and V planes.
        uVal = read10BitSample(uRow + chromaX * 2, false);
        vVal = read10BitSample(vRow + chromaX * 2, false);
      }

      // Normalize to [0, 1] (Y') and [-0.5, 0.5] (Cb, Cr).
      double yNorm =
          std::clamp((static_cast<double>(yVal) - yMin) / yRange, 0.0, 1.0);
      double cb = (static_cast<double>(uVal) - uvMin) / uvRange - 0.5;
      double cr = (static_cast<double>(vVal) - uvMin) / uvRange - 0.5;

      // YCbCr → R'G'B' (non-linear signal, [0, 1])
      double rSignal = std::clamp(yNorm + crToR * cr, 0.0, 1.0);
      double gSignal = std::clamp(yNorm + crToG * cr + cbToG * cb, 0.0, 1.0);
      double bSignal = std::clamp(yNorm + cbToB * cb, 0.0, 1.0);

      // Linearize using EOTF.
      double rLin, gLin, bLin;
      if (isPQ) {
        // Use LUT: clamp the 10-bit signal to valid index.
        int rIdx =
            std::clamp(static_cast<int>(std::round(rSignal * 1023.0)), 0, 1023);
        int gIdx =
            std::clamp(static_cast<int>(std::round(gSignal * 1023.0)), 0, 1023);
        int bIdx =
            std::clamp(static_cast<int>(std::round(bSignal * 1023.0)), 0, 1023);
        // PQ EOTF returns nits; normalize to SDR-relative.
        rLin = pqLut[rIdx] / SDR_WHITE;
        gLin = pqLut[gIdx] / SDR_WHITE;
        bLin = pqLut[bIdx] / SDR_WHITE;
      } else {
        // HLG: inverse OETF gives scene-linear [0, 1]
        int rIdx =
            std::clamp(static_cast<int>(std::round(rSignal * 1023.0)), 0, 1023);
        int gIdx =
            std::clamp(static_cast<int>(std::round(gSignal * 1023.0)), 0, 1023);
        int bIdx =
            std::clamp(static_cast<int>(std::round(bSignal * 1023.0)), 0, 1023);
        rLin = hlgLut[rIdx];
        gLin = hlgLut[gIdx];
        bLin = hlgLut[bIdx];

        // Apply HLG OOTF: display_linear = Lw * scene_linear * Y^(gamma-1)
        double luma = BT2020_KR * rLin + BT2020_KG * gLin + BT2020_KB * bLin;
        double ootfScale =
            HLG_DISPLAY_LW * std::pow(std::max(luma, 0.0), hlgGamma - 1.0);
        rLin = rLin * ootfScale / SDR_WHITE;
        gLin = gLin * ootfScale / SDR_WHITE;
        bLin = bLin * ootfScale / SDR_WHITE;
      }

      // BT.2020 → BT.709 gamut mapping (3x3 matrix on linear RGB).
      double r709 = GAMUT_MAP[0][0] * rLin + GAMUT_MAP[0][1] * gLin +
          GAMUT_MAP[0][2] * bLin;
      double g709 = GAMUT_MAP[1][0] * rLin + GAMUT_MAP[1][1] * gLin +
          GAMUT_MAP[1][2] * bLin;
      double b709 = GAMUT_MAP[2][0] * rLin + GAMUT_MAP[2][1] * gLin +
          GAMUT_MAP[2][2] * bLin;

      // Clamp negatives (out-of-gamut colors).
      r709 = std::max(r709, 0.0);
      g709 = std::max(g709, 0.0);
      b709 = std::max(b709, 0.0);

      // Hable tone mapping.
      // We tonemap per-channel using max-RGB to preserve hue, similar to
      // FFmpeg's vf_tonemap.
      double sig = std::max({r709, g709, b709});
      if (sig > 0.0) {
        double mappedSig = hable(sig) * hablePeakInv;
        double scale = mappedSig / sig;
        r709 *= scale;
        g709 *= scale;
        b709 *= scale;
      }

      // BT.709 OETF (gamma encode) + quantize to uint8.
      int rOut = static_cast<int>(std::clamp(
          bt709OETF(std::clamp(r709, 0.0, 1.0)) * 255.0 + 0.5, 0.0, 255.0));
      int gOut = static_cast<int>(std::clamp(
          bt709OETF(std::clamp(g709, 0.0, 1.0)) * 255.0 + 0.5, 0.0, 255.0));
      int bOut = static_cast<int>(std::clamp(
          bt709OETF(std::clamp(b709, 0.0, 1.0)) * 255.0 + 0.5, 0.0, 255.0));

      outRow[x * 3 + 0] = static_cast<uint8_t>(rOut);
      outRow[x * 3 + 1] = static_cast<uint8_t>(gOut);
      outRow[x * 3 + 2] = static_cast<uint8_t>(bOut);
    }
  }

  return dst;
}

} // namespace facebook::torchcodec
