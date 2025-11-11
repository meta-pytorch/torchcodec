cmake_minimum_required(VERSION 3.18)
project(TorchCodec)

# Define LINUX platform variable globally
if (UNIX AND NOT APPLE)
    set(LINUX TRUE)
else()
    set(LINUX FALSE)
endif()

add_subdirectory(src/torchcodec/_core)


option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
