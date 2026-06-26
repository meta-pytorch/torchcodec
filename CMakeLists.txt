cmake_minimum_required(VERSION 3.18)
project(TorchCodec)

# Define LINUX platform variable globally
if (UNIX AND NOT APPLE)
    set(LINUX TRUE)
else()
    set(LINUX FALSE)
endif()

# When OFF, build a torch-free core + pybind frontend (CPU only): no
# find_package(Torch), no torch linkage, and the torch adapter (custom_ops) and
# torch hooks (TCTorchBackend) are skipped. Used to prove torchcodec decoding
# does not require PyTorch.
option(ENABLE_TORCH "Build torch-dependent components" ON)

add_subdirectory(src/torchcodec/_core)


option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
