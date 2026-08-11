// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Decode video frames from a plain C++ process: no Python interpreter, no
// libpython, no torchcodec Python package. Only libtorch + the torchcodec
// shared libraries + the .pt2 produced by export_decoder.py.
//
//   g++ -std=c++17 -O2 main.cpp -o decode_video \
//       -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" \
//       -L"$TORCH/lib" -Wl,-rpath,"$TORCH/lib" -ltorch -ltorch_cpu -lc10 -ldl
//
//   ./decode_video <torchcodec_lib_dir> <video_file> <decoder.pt2>

#include <dlfcn.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

namespace {

// The ops register themselves into the PyTorch dispatcher from the static
// initializers of libtorchcodec_custom_opsN, so all a C++ program has to do is
// load it. libtorchcodec_coreN is loaded first, and RTLD_GLOBAL so that the
// custom ops library resolves against it.
void load_torchcodec(const std::string& lib_dir) {
  for (const char* name :
       {"libtorchcodec_core7.so", "libtorchcodec_custom_ops7.so"}) {
    const std::string path = lib_dir + "/" + name;
    if (dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL) == nullptr) {
      throw std::runtime_error("dlopen(" + path + ") failed: " + dlerror());
    }
  }
}

// Boxed dispatcher call: every argument, including the ones that have a default
// in the schema, is pushed onto the stack in declaration order.
c10::IValue call_op(const char* name, std::vector<c10::IValue> stack) {
  const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(name, "");
  op.callBoxed(&stack);
  TORCH_CHECK(stack.size() == 1, "expected a single return value from ", name);
  return std::move(stack[0]);
}

at::Tensor read_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  TORCH_CHECK(file, "could not open ", path);
  const std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  at::Tensor data = at::empty({size}, at::kByte);
  TORCH_CHECK(
      file.read(reinterpret_cast<char*>(data.data_ptr<uint8_t>()), size),
      "could not read ",
      path);
  return data;
}

} // namespace

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0]
              << " <torchcodec_lib_dir> <video_file> <decoder.pt2>\n";
    return 1;
  }
  const std::string lib_dir = argv[1];
  const std::string video_file = argv[2];
  const std::string package_path = argv[3];

  load_torchcodec(lib_dir);

  const at::Tensor video_data = read_file(video_file);
  const at::Tensor frame_indices =
      at::tensor({0, 10, 20, 100}, at::TensorOptions().dtype(at::kLong));

  // Metadata. This can't come out of the exported program: it's a JSON string,
  // and the tracer would constant-fold the fake impl's empty return value into
  // the graph. So we call the op directly through the dispatcher instead. Note
  // that this is a *second*, throwaway decoder, used only for its metadata; the
  // one that decodes frames lives inside the exported program.
  const at::Tensor decoder =
      call_op(
          "torchcodec_ns::create_video_decoder_from_tensor",
          {video_data,
           /*seek_mode=*/c10::IValue(),
           /*num_threads=*/c10::IValue(),
           /*dimension_order=*/c10::IValue(),
           /*stream_index=*/c10::IValue(),
           /*device=*/std::string("cpu"),
           /*device_variant=*/std::string("default"),
           /*transform_specs=*/std::string(""),
           /*output_dtype=*/std::string("uint8")})
          .toTensor();
  const std::string metadata =
      call_op("torchcodec_ns::get_json_metadata", {decoder}).toStringRef();
  std::cout << "metadata: " << metadata << "\n";

  // Decoding, by running the exported + AOTInductor-compiled program.
  torch::inductor::AOTIModelPackageLoader loader(package_path);
  const std::vector<at::Tensor> outputs =
      loader.run({video_data, frame_indices});

  const at::Tensor& frames = outputs.at(0);
  std::cout << "frames: " << frames.sizes() << " " << frames.dtype()
            << ", sum=" << frames.sum().item<int64_t>() << "\n";

  return 0;
}
