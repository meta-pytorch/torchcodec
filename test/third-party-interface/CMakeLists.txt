# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# See test_third_party_interface.py for context

cmake_minimum_required(VERSION 3.18)
project(ThirdPartyInterfaceTest)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(TorchCodec REQUIRED)

function(make_torchcodec_targets torchcodec_variant)
    set(libname "torchcodec_third_party_interface_test${torchcodec_variant}")
    set(sources ThirdPartyInterfaceTest.cpp)

    # Building executable to check the linkage
    add_library(${libname} SHARED ${sources})
    set_target_properties(${libname} PROPERTIES PREFIX "")
    set_target_properties(${libname} PROPERTIES CXX_STANDARD 17)
    target_link_libraries(${libname}
      ${TORCH_LIBRARIES}
      torchcodec::core${torchcodec_variant}
      torchcodec::ffmpeg${torchcodec_variant}
    )
endfunction()

foreach(variant IN LISTS TORCHCODEC_VARIANTS)
    make_torchcodec_targets(${variant})
endforeach()
