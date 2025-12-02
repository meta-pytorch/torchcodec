# This file fetches the non-GPL ffmpeg libraries from the torchcodec S3 bucket,
# and exposes them as CMake targets so we can dynamically link against them.
# These libraries were built on the CI via the build_ffmpeg.yaml workflow.

# Avoid warning: see https://cmake.org/cmake/help/latest/policy/CMP0135.html
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)

set(
    base_url
    https://pytorch.s3.amazonaws.com/torchcodec/ffmpeg/2025-03-14
)

if (LINUX)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(
            platform_url
            ${base_url}/linux_aarch64
        )

        set(
            f4_sha256
            a310a2ed9ffe555fd3278dae15065541098dd35e124564671dcda6a6620ac842
        )
        set(
            f5_sha256
            89ca7996bccbc2db49adaa401d20fdbabffe0e1b4e07a0f81d6b143e858b7c8d
        )
        set(
            f6_sha256
            ae44c67b4587d061b8e9cc8990ca891ee013fe52ad79e5016ba29871562621da
        )
        set(
            f7_sha256
            948e2cac66ca6f68ff526d5e84138e94bce0f1a7c83f502d15d85d0bd3ddc112
        )
        set(
            f8_sha256
            b9cfd99ae75a14e58300854967d4dc49de0b3daa551df51ea1f52a3f08d2c8af
        )
    elseif (LINUX)  # assume x86_64
        set(
            platform_url
            ${base_url}/linux_x86_64
        )

        set(
            f4_sha256
            1a083f1922443bedb5243d04896383b8c606778a7ddb9d886c8303e55339fe0c
        )
        set(
            f5_sha256
            65d6ad54082d94dcb3f801d73df2265e0e1bb303c7afbce7723e3b77ccd0e207
        )
        set(
            f6_sha256
            8bd5939c2f4a4b072e837e7870c13fe7d13824e5ff087ab534e4db4e90b7be9c
        )
        set(
            f7_sha256
            1cb946d8b7c6393c2c3ebe1f900b8de7a2885fe614c45d4ec32c9833084f2f26
        )
        set(
            f8_sha256
            c55b3c1a4b5e4d5fdd7c632bea3ab6f45b4e37cc8e0999dda3f84a8ed8defad8
        )
  endif()
elseif (APPLE)
    set(
        platform_url
        ${base_url}/macos_arm64
    )
    set(
        f4_sha256
        f0335434529d9e19359eae0fe912dd9e747667534a1c92e662f5219a55dfad8c
    )
    set(
        f5_sha256
        cfc3449c9af6863731a431ce89e32c08c5f8ece94b306fb6b695828502a76166
    )
    set(
        f6_sha256
        ec47b4783c342038e720e33b2fdfa55a9a490afb1cf37a26467733983688647e
    )
    set(
        f7_sha256
        48a4fc8ce098305cfd4a58f40889249c523ca3c285f66ba704b5bad0e3ada53a
    )
    set(
        f8_sha256
        beb936b76f25d2621228a12cdb67c9ae3d1eff7aa713ef8d1167ebf0c25bd5ec
    )
elseif (WIN32)
    set(
        platform_url
        ${base_url}/windows_x86_64
    )
    set(
        f4_sha256
        270a1aa8892225267e68a7eb87c417931da30dccbf08ee2bde8833e659cab5cb
    )
    set(
        f5_sha256
        b8b2a349a847e56a6da875b066dff1cae53cb8ee7cf5ba9321ec1243dea0cde0
    )
    set(
        f6_sha256
        5d9f8c76dc55f790fa31d825985e9270bf9e498b8bfec21a0ad3a1feb1fa053a
    )
    set(
        f7_sha256
        ae391ace382330e912793b70b68529ee7c91026d2869b4df7e7c3e7d3656bdd5
    )
    set(
        f8_sha256
        bac845ac79876b104959cb0e7b9dec772a261116344dd17d2f97e7ddfac4a73f
    )
else()
    message(
        FATAL_ERROR
        "Unsupported operating system: ${CMAKE_SYSTEM_NAME}"
    )
endif()

FetchContent_Declare(
    f4
    URL ${platform_url}/4.4.4.tar.gz
    URL_HASH
    SHA256=${f4_sha256}
)
FetchContent_Declare(
    f5
    URL ${platform_url}/5.1.4.tar.gz
    URL_HASH
    SHA256=${f5_sha256}
)
FetchContent_Declare(
    f6
    URL ${platform_url}/6.1.1.tar.gz
    URL_HASH
    SHA256=${f6_sha256}
)
FetchContent_Declare(
    f7
    URL ${platform_url}/7.0.1.tar.gz
    URL_HASH
    SHA256=${f7_sha256}
)
FetchContent_Declare(
    f8
    URL ${platform_url}/8.0.tar.gz
    URL_HASH
    SHA256=${f8_sha256}
)

FetchContent_MakeAvailable(f4 f5 f6 f7 f8)

# makes add_ffmpeg_target available
include("${CMAKE_CURRENT_SOURCE_DIR}/../share/cmake/TorchCodec/ffmpeg_versions.cmake")

# Note: the f?_SOURCE_DIR variables were set by FetchContent_MakeAvailable
add_ffmpeg_target(4 "${f4_SOURCE_DIR}")
add_ffmpeg_target(5 "${f5_SOURCE_DIR}")
add_ffmpeg_target(6 "${f6_SOURCE_DIR}")
add_ffmpeg_target(7 "${f7_SOURCE_DIR}")
add_ffmpeg_target(8 "${f8_SOURCE_DIR}")
