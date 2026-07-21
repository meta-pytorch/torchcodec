// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "DecodeGif.h"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>

#include <torch/headeronly/core/MemoryFormat.h>

#include <algorithm>
#include <cstdint>
#include <optional>

#include "ImageCommon.h"
#include "StableABICompat.h"
#include "giflib/gif_lib.h"

namespace facebook::torchcodec {

namespace {

struct SourceCtx {
  const uint8_t* ptr; // current read position in the input tensor
  size_t count; // number of bytes left to read
};

// Reader passed to DGifOpen(): giflib calls it to pull `bytes` encoded bytes
// into `output`. We serve them from the input tensor (see SourceCtx), advancing
// as we go, and return the number of bytes actually read.
int read_callback(GifFileType* gif_file, GifByteType* output, int bytes) {
  // UserData was set in DGifOpen().
  auto* source_ctx = static_cast<SourceCtx*>(gif_file->UserData);
  size_t num_bytes_to_read =
      std::min(static_cast<size_t>(bytes), source_ctx->count);
  std::copy(source_ctx->ptr, source_ctx->ptr + num_bytes_to_read, output);
  source_ctx->ptr += num_bytes_to_read;
  source_ctx->count -= num_bytes_to_read;
  return static_cast<int>(num_bytes_to_read);
}

} // namespace

torch::stable::Tensor decode_gif(const torch::stable::Tensor& input) {
  // LibGif docs: https://giflib.sourceforge.net/intro.html
  // Refer over there for more details on the libgif API, API ref, and a
  // detailed description of the GIF format.

  validate_encoded_data(input);

  int error = D_GIF_SUCCEEDED;

  // We're using DGifOpen. The other entrypoints of libgif are DGifOpenFileName
  // and DGifOpenFileHandle but we don't want to use those, since we need to
  // read the encoded bytes from a tensor of encoded bytes, not from a file (for
  // consistency with existing jpeg and png decoders). Using DGifOpen is the
  // only way to read from a custom source, via the read_callback reader above.

  // TODO: We are potentially doing an unnecessary copy of the encoded bytes:
  // - 1 copy from file to tensor (in the Python _read_file_to_tensor())
  // - 1 copy from tensor to GIFLIB buffers (in read_callback())
  // Since we're vendoring GIFLIB we can potentially modify the calls to
  // InternalRead() and just set the `buf` pointer to the tensor data directly.
  // That might even save allocation of those buffers.
  // If we do that, we'd have to make sure the buffers are never written to by
  // GIFLIB, otherwise we'd be overriding the tensor data.
  SourceCtx source_ctx{
      .ptr = input.const_data_ptr<uint8_t>(),
      .count = static_cast<size_t>(input.numel())};
  GifFileType* gif_file =
      DGifOpen(static_cast<void*>(&source_ctx), read_callback, &error);

  STD_TORCH_CHECK(
      (gif_file != nullptr) && (error == D_GIF_SUCCEEDED),
      "DGifOpen() failed - ",
      error);

  if (DGifSlurp(gif_file) == GIF_ERROR) {
    auto slurp_error = gif_file->Error;
    DGifCloseFile(gif_file, &error);
    STD_TORCH_CHECK(false, "DGifSlurp() failed - ", slurp_error);
  }
  auto num_images = gif_file->ImageCount;

  // This check should already be done within DGifSlurp(), just to be safe.
  STD_TORCH_CHECK(
      num_images > 0, "GIF file should contain at least one image!");

  GifColorType bg{0, 0, 0};
  if (gif_file->SColorMap) {
    bg = gif_file->SColorMap->Colors[gif_file->SBackGroundColor];
  }

  // The GIFLIB docs say that the canvas's height and width are potentially
  // ignored by modern viewers, so to be on the safe side we set the output
  // height to max(canvas_height, first_image_height). Same for width.
  // https://giflib.sourceforge.net/whatsinagif/bits_and_bytes.html
  auto output_h =
      std::max(gif_file->SHeight, gif_file->SavedImages[0].ImageDesc.Height);
  auto output_w =
      std::max(gif_file->SWidth, gif_file->SavedImages[0].ImageDesc.Width);

  auto output = torch::stable::empty(
      {static_cast<int64_t>(num_images),
       3,
       static_cast<int64_t>(output_h),
       static_cast<int64_t>(output_w)},
      kStableUInt8,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      torch::headeronly::MemoryFormat::ChannelsLast);
  auto output_a = mutable_accessor<uint8_t, 4>(output);
  for (int i = 0; i < num_images; ++i) {
    const SavedImage& img = gif_file->SavedImages[i];

    GraphicsControlBlock gcb;
    DGifSavedExtensionToGCB(gif_file, i, &gcb);

    const GifImageDesc& desc = img.ImageDesc;
    const ColorMapObject* cmap =
        desc.ColorMap ? desc.ColorMap : gif_file->SColorMap;
    STD_TORCH_CHECK(
        cmap != nullptr,
        "Global and local color maps are missing. This should never happen!");

    // When going from one image to another, there is a "disposal method" which
    // specifies how to handle the transition. E.g. DISPOSE_DO_NOT means that
    // the current image should essentially be drawn on top of the previous
    // canvas. The pixels of that previous canvas will appear on the new one if
    // either:
    // - a pixel is transparent in the current image
    // - the current image is smaller than the canvas, hence exposing its pixels
    // The "background" disposal method means that the current canvas should be
    // set to the background color.
    // We only support these 2 modes and default to DISPOSE_DO_NOT when the
    // disposal method is unspecified, or when it's set to DISPOSE_PREVIOUS
    // which according to GIFLIB is not widely supported.
    // (https://giflib.sourceforge.net/whatsinagif/animation_and_transparency.html).
    // This is consistent with default behaviour in the majority of web browsers
    // and image libraries like Pillow.
    if (i > 0 &&
        (gcb.DisposalMode == DISPOSAL_UNSPECIFIED ||
         gcb.DisposalMode == DISPOSE_DO_NOT ||
         gcb.DisposalMode == DISPOSE_PREVIOUS)) {
      copy_frame(output, i, output, i - 1);
    } else {
      // Background. If bg wasn't defined, it will be (0, 0, 0).
      for (int h = 0; h < gif_file->SHeight; ++h) {
        for (int w = 0; w < gif_file->SWidth; ++w) {
          output_a[i][0][h][w] = bg.Red;
          output_a[i][1][h][w] = bg.Green;
          output_a[i][2][h][w] = bg.Blue;
        }
      }
    }

    // The 'continue' blocks are to limit the frame to the output canvas. The
    // output tensor is allocated from the canvas (SHeight/SWidth) and the first
    // frame dimensions, but desc.{Top,Left,Height,Width} of any frame may place
    // pixels outside it (a later frame larger than the first, or any frame with
    // a non-zero offset).  We just drop the pixels that would land outside of
    // the allocated tensor.
    for (int h = 0; h < desc.Height; ++h) {
      const auto y = static_cast<int64_t>(desc.Top) + h;
      if (y < 0 || y >= output_h) {
        continue;
      }
      for (int w = 0; w < desc.Width; ++w) {
        const auto x = static_cast<int64_t>(desc.Left) + w;
        if (x < 0 || x >= output_w) {
          continue;
        }
        auto c = img.RasterBits[h * desc.Width + w];
        if (c == gcb.TransparentColor) {
          continue;
        }
        GifColorType rgb = cmap->Colors[c];
        output_a[i][0][y][x] = rgb.Red;
        output_a[i][1][y][x] = rgb.Green;
        output_a[i][2][y][x] = rgb.Blue;
      }
    }
  }

  // Remove the batch dim if there's only one image, so a still GIF decodes to a
  // (C, H, W) tensor like the other image decoders.
  if (num_images == 1) {
    output = select_row(output, 0);
  }

  DGifCloseFile(gif_file, &error);
  STD_TORCH_CHECK(error == D_GIF_SUCCEEDED, "DGifCloseFile() failed - ", error);

  return output;
}

} // namespace facebook::torchcodec
