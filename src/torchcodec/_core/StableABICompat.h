// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/csrc/stable/library.h>

#include <stdexcept>
#include <string>

// Index error check - throws std::out_of_range which pybind11 maps to
// IndexError Use this for index validation errors that should raise IndexError
// in Python
#define STABLE_CHECK_INDEX(cond, msg)            \
  do {                                           \
    if (!(cond)) {                               \
      throw std::out_of_range(std::string(msg)); \
    }                                            \
  } while (false)
