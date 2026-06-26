cmake_minimum_required(VERSION 3.14)
include(CMakePrintHelpers)
project(TorchCodecTests)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

add_executable(
  VideoDecoderTest
  VideoDecoderTest.cpp
)

target_include_directories(VideoDecoderTest SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
target_include_directories(VideoDecoderTest SYSTEM PRIVATE ${libav_include_dirs})
target_include_directories(VideoDecoderTest PRIVATE ../)

target_link_libraries(
  VideoDecoderTest
  ${libtorchcodec_library_name}
  ${libtorchcodec_custom_ops_name}
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(VideoDecoderTest)


add_executable(
  MetadataTest
  MetadataTest.cpp
)

target_include_directories(MetadataTest SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
target_include_directories(MetadataTest SYSTEM PRIVATE ${libav_include_dirs})
target_include_directories(MetadataTest PRIVATE ../)

target_link_libraries(
  MetadataTest
  ${libtorchcodec_library_name}
  ${libtorchcodec_custom_ops_name}
  GTest::gtest_main
)

gtest_discover_tests(MetadataTest)


# tc::Tensor is the torch-free core tensor type. Its test only needs the core
# library (which contains TCTensor.cpp); it does not depend on torch itself.
add_executable(
  TCTensorTest
  TCTensorTest.cpp
)

target_include_directories(TCTensorTest PRIVATE ../)

target_link_libraries(
  TCTensorTest
  ${libtorchcodec_library_name}
  GTest::gtest_main
)

gtest_discover_tests(TCTensorTest)


# Verifies the torch-linked custom-ops library registers tc's allocator + CUDA
# backend, so it links custom_ops (whose static initializer does the wiring).
add_executable(
  TCTorchBackendTest
  TCTorchBackendTest.cpp
)

target_include_directories(TCTorchBackendTest PRIVATE ../)

# The test references no custom_ops symbol directly (it only checks that loading
# custom_ops ran its static initializer, which registers the tc hooks). Under
# the default -Wl,--as-needed, the linker would DROP the unused custom_ops
# dependency and the initializer would never run. Force it to be retained with
# --no-as-needed placed right before custom_ops on the link line. (Linux/GNU ld
# only; the real Python path dlopens custom_ops explicitly so doesn't need this.)
if(LINUX)
  set(_tctorchbackend_custom_ops
      "-Wl,--no-as-needed" ${libtorchcodec_custom_ops_name} "-Wl,--as-needed")
else()
  set(_tctorchbackend_custom_ops ${libtorchcodec_custom_ops_name})
endif()

target_link_libraries(
  TCTorchBackendTest
  ${libtorchcodec_library_name}
  ${_tctorchbackend_custom_ops}
  GTest::gtest_main
)

gtest_discover_tests(TCTorchBackendTest)
