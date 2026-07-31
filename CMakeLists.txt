cmake_minimum_required(VERSION 3.18)
project(TorchCodec)

# Define LINUX platform variable globally
if (UNIX AND NOT APPLE)
    set(LINUX TRUE)
else()
    set(LINUX FALSE)
endif()

# Auto-detect PyTorch's cmake prefix path so that find_package(Torch) works
# without the user having to pass -DTorch_DIR manually.
if(NOT DEFINED Torch_DIR)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c
                "import torch; print(torch.utils.cmake_prefix_path)"
            OUTPUT_VARIABLE _torch_prefix
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _torch_result
        )
        if(_torch_result EQUAL 0)
            list(APPEND CMAKE_PREFIX_PATH "${_torch_prefix}")
        endif()
    endif()
endif()

add_subdirectory(src/torchcodec/_core)


option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
