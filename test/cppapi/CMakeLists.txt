cmake_minimum_required(VERSION 3.18)
project(CppApiTest)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(TorchCodec REQUIRED)

function(make_torchcodec_targets torchcodec_variant)
    set(binname "torchcodec_cppapitest${torchcodec_variant}")
    set(sources CppApiTest.cpp)

    # Building executable to check the linkage
    add_executable(${binname} ${sources})
    # Avoid adding the "lib" prefix which we already add explicitly.
    set_target_properties(${binname} PROPERTIES PREFIX "")
    set_target_properties(${binname} PROPERTIES CXX_STANDARD 17)
    target_link_libraries(${binname}
      ${TORCH_LIBRARIES}
      torchcodec::core${torchcodec_variant}
      torchcodec::ffmpeg${torchcodec_variant}
    )
endfunction()

foreach(variant IN LISTS TORCHCODEC_VARIANTS)
    make_torchcodec_targets(${variant})
endforeach()
