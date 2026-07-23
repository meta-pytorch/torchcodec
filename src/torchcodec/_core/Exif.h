// @nolint (improperly imported third-party code)
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this
license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without
modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright
notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote
products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is"
and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are
disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any
direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#pragma once
// Functions in this module are taken from OpenCV
// https://github.com/opencv/opencv/blob/097891e311fae1d8354eb092a0fd0171e630d78c/modules/imgcodecs/src/exif.cpp
//
// Ported from torchvision's csrc/io/image/cpu/exif.h into torchcodec.
// Changes made:
// - make input ptr const.
// - make the exif values an enum

#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>

#include "StableABICompat.h"

namespace facebook::torchcodec {
namespace exif_private {

constexpr uint16_t ENDIANNESS_INTEL = 0x49;
constexpr uint16_t ENDIANNESS_MOTO = 0x4d;
constexpr uint16_t REQ_EXIF_TAG_MARK = 0x2a;
constexpr uint16_t ORIENTATION_EXIF_TAG = 0x0112;
constexpr uint16_t INCORRECT_TAG = -1;

// EXIF orientation values as defined in JEITA CP-3451C section 4.6.4.A. Each
// name describes where the 0th row and 0th column of the stored pixels end up
// in the correctly-displayed image.
enum class ExifOrientation : int {
  Unspecified = -1,
  TopLeft = 1, // normal orientation
  TopRight = 2, // needs horizontal flip
  BottomRight = 3, // needs 180 rotation
  BottomLeft = 4, // needs vertical flip
  LeftTop = 5, // mirrored horizontal & rotate 270 CW
  RightTop = 6, // rotate 90 CW
  RightBottom = 7, // mirrored horizontal & rotate 90 CW
  LeftBottom = 8, // needs 270 CW rotation
};

class ExifDataReader {
 public:
  ExifDataReader(const unsigned char* p, size_t s) : _ptr(p), _size(s) {}

  size_t size() const {
    return _size;
  }

  const unsigned char& operator[](size_t index) const {
    STD_TORCH_CHECK(index < _size);
    return _ptr[index];
  }

 protected:
  const unsigned char* _ptr;
  size_t _size;
};

inline uint16_t get_endianness(const ExifDataReader& exif_data) {
  if ((exif_data.size() < 1) ||
      (exif_data.size() > 1 && exif_data[0] != exif_data[1])) {
    return 0;
  }
  if (exif_data[0] == 'I') {
    return ENDIANNESS_INTEL;
  }
  if (exif_data[0] == 'M') {
    return ENDIANNESS_MOTO;
  }
  return 0;
}

inline uint16_t get_uint16(
    const ExifDataReader& exif_data,
    uint16_t endianness,
    const size_t offset) {
  if (offset + 1 >= exif_data.size()) {
    return INCORRECT_TAG;
  }

  if (endianness == ENDIANNESS_INTEL) {
    return exif_data[offset] + (exif_data[offset + 1] << 8);
  }
  return (exif_data[offset] << 8) + exif_data[offset + 1];
}

inline uint32_t get_uint32(
    const ExifDataReader& exif_data,
    uint16_t endianness,
    const size_t offset) {
  if (offset + 3 >= exif_data.size()) {
    return INCORRECT_TAG;
  }

  if (endianness == ENDIANNESS_INTEL) {
    return exif_data[offset] + (exif_data[offset + 1] << 8) +
        (exif_data[offset + 2] << 16) + (exif_data[offset + 3] << 24);
  }
  return (exif_data[offset] << 24) + (exif_data[offset + 1] << 16) +
      (exif_data[offset + 2] << 8) + exif_data[offset + 3];
}

inline ExifOrientation fetch_exif_orientation(
    const unsigned char* exif_data_ptr,
    size_t size) {
  STD_TORCH_CHECK(exif_data_ptr != nullptr, "exif_data_ptr cannot be null");

  ExifOrientation exif_orientation = ExifOrientation::Unspecified;

  // Exif binary structure looks like this
  // First 6 bytes: [E, x, i, f, 0, 0]
  // Endianness, 2 bytes : [M, M] or [I, I]
  // Tag mark, 2 bytes: [0, 0x2a]
  // Offset, 4 bytes
  // Num entries, 2 bytes
  // Tag entries and data, tag has 2 bytes and its data has 10 bytes
  // For more details:
  // http://www.media.mit.edu/pia/Research/deepview/exif.html

  ExifDataReader exif_data(exif_data_ptr, size);
  auto endianness = get_endianness(exif_data);

  // Checking whether Tag Mark (0x002A) correspond to one contained in the
  // Jpeg file
  uint16_t tag_mark = get_uint16(exif_data, endianness, 2);
  if (tag_mark == REQ_EXIF_TAG_MARK) {
    auto offset = get_uint32(exif_data, endianness, 4);
    size_t num_entry = get_uint16(exif_data, endianness, offset);
    offset += 2; // go to start of tag fields
    constexpr size_t tiff_field_size = 12;
    for (size_t entry = 0; entry < num_entry; entry++) {
      // Here we just search for orientation tag and parse it
      auto tag_num = get_uint16(exif_data, endianness, offset);
      if (tag_num == INCORRECT_TAG) {
        break;
      }
      if (tag_num == ORIENTATION_EXIF_TAG) {
        exif_orientation = static_cast<ExifOrientation>(
            get_uint16(exif_data, endianness, offset + 8));
        break;
      }
      offset += tiff_field_size;
    }
  }
  return exif_orientation;
}

inline torch::stable::Tensor exif_orientation_transform(
    const torch::stable::Tensor& image,
    ExifOrientation orientation) {
  switch (orientation) {
    case ExifOrientation::TopRight:
      return stable_flip(image, {-1});
    case ExifOrientation::BottomRight:
      // 180 rotation, equivalent to flipping both horizontally and vertically.
      return stable_flip(image, {-2, -1});
    case ExifOrientation::BottomLeft:
      return stable_flip(image, {-2});
    case ExifOrientation::LeftTop:
      return torch::stable::transpose(image, -1, -2);
    case ExifOrientation::RightTop:
      return stable_flip(torch::stable::transpose(image, -1, -2), {-1});
    case ExifOrientation::RightBottom:
      return stable_flip(torch::stable::transpose(image, -1, -2), {-2, -1});
    case ExifOrientation::LeftBottom:
      return stable_flip(torch::stable::transpose(image, -1, -2), {-2});
    default:
      // TopLeft is the normal orientation; Unspecified and any unknown value
      // (including the invalid 0) are treated as no transform.
      return image;
  }
}

} // namespace exif_private
} // namespace facebook::torchcodec
