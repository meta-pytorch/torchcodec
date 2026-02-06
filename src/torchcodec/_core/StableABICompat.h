// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

// ===========================================================================
// PyTorch Stable ABI Compatibility Header
// ===========================================================================
//
// This header provides compatibility macros for error handling that work with
// PyTorch's stable ABI API. It replaces TORCH_CHECK and related macros.

#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace facebook::torchcodec::detail {

// Helper for streaming std::optional to ostream
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::optional<T>& opt) {
  if (opt.has_value()) {
    os << *opt;
  } else {
    os << "nullopt";
  }
  return os;
}

// Helper to build error message from variadic args
template <typename... Args>
std::string buildErrorMessage(Args&&... args) {
  std::ostringstream oss;
  using namespace facebook::torchcodec::detail; // Make operator<< visible
  (oss << ... << std::forward<Args>(args));
  return oss.str();
}

// Helper for when no message is provided
inline std::string buildErrorMessage() {
  return "Check failed";
}

} // namespace facebook::torchcodec::detail

// ===========================================================================
// Error Handling Macros
// ===========================================================================
// Replacement for TORCH_CHECK() that works with stable ABI.
// Throws std::runtime_error on failure.

#define STABLE_CHECK(cond, ...)                                          \
  do {                                                                   \
    if (!(cond)) {                                                       \
      throw std::runtime_error(                                          \
          facebook::torchcodec::detail::buildErrorMessage(__VA_ARGS__)); \
    }                                                                    \
  } while (false)

// Equality check - replacement for TORCH_CHECK_EQ
#define STABLE_CHECK_EQ(a, b, ...) \
  STABLE_CHECK((a) == (b), "Expected " #a " == " #b ". ", ##__VA_ARGS__)

// Index error check - throws std::out_of_range which pybind11 maps to
// IndexError Use this for index validation errors that should raise IndexError
// in Python
#define STABLE_CHECK_INDEX(cond, ...)                                    \
  do {                                                                   \
    if (!(cond)) {                                                       \
      throw std::out_of_range(                                           \
          facebook::torchcodec::detail::buildErrorMessage(__VA_ARGS__)); \
    }                                                                    \
  } while (false)
