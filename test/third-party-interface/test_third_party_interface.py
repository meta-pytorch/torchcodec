# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
We allow third-parties to build their own C++ TorchCodec extensions via the DeviceInterface API.
This test ensures that such third-party extensions can be built correctly.
"""

import os
import subprocess

from pathlib import Path

import torch
import torchcodec


def test_third_party_interface_pkgconfig(tmp_path):
    # Test building of third-party-interface. Since
    # TORCHCODEC_FFMPEG{ver}_INSTALL_PREFIX is not provided, FFmpeg should be
    # found via pkg-config
    cmake_args = [
        "cmake",
        "-DCMAKE_BUILD_TYPE=Debug",
        "-DCMAKE_VERBOSE_MAKEFILE=ON",
        f"-DCMAKE_PREFIX_PATH={torchcodec.cmake_prefix_path};{torch.utils.cmake_prefix_path}",
        Path(__file__).parent,
    ]
    result = subprocess.run(cmake_args, cwd=tmp_path)
    assert result.returncode == 0

    result = subprocess.run(["cmake", "--build", "."], cwd=tmp_path)
    assert result.returncode == 0

    # loading built .so in the separate process to avoid flooding current process
    ver = f"{torchcodec.ffmpeg_major_version}"
    result = subprocess.run(
        [
            "python3",
            "-c",
            f"import torch; torch.ops.load_library('torchcodec_third_party_interface_test{ver}.so')",
        ],
        cwd=tmp_path,
    )
    assert result.returncode == 0


def test_third_party_interface_fails_when_no_ffmpeg(tmp_path):
    # Test that passing non-existing TORCHCODEC_FFMPEG{ver}_INSTALL_PREFIX
    # makes cmake configuration fail
    cmake_args = [
        "cmake",
        "-DCMAKE_BUILD_TYPE=Debug",
        "-DCMAKE_VERBOSE_MAKEFILE=ON",
        f"-DCMAKE_PREFIX_PATH={torchcodec.cmake_prefix_path};{torch.utils.cmake_prefix_path}",
        Path(__file__).parent,
    ]
    ver = f"{torchcodec.ffmpeg_major_version}"
    my_env = os.environ.copy()
    my_env[f"TORCHCODEC_FFMPEG{ver}_INSTALL_PREFIX"] = (
        Path(__file__).parent / "no-such-dir"
    )

    # cmake config should fail as we've set ffmpeg install prefix to the not existing
    # directory
    result = subprocess.run(cmake_args, cwd=tmp_path, env=my_env)
    assert result.returncode != 0


def test_third_party_interface_with_prefix(tmp_path):
    # Test that passing a valid TORCHCODEC_FFMPEG{ver}_INSTALL_PREFIX uses those
    # FFmpeg libraries.
    cmake_args = [
        "cmake",
        "-DCMAKE_BUILD_TYPE=Debug",
        "-DCMAKE_VERBOSE_MAKEFILE=ON",
        f"-DCMAKE_PREFIX_PATH={torchcodec.cmake_prefix_path};{torch.utils.cmake_prefix_path}",
        Path(__file__).parent,
    ]

    # In this test we are calculating the prefix of installed ffmpeg version from the location
    # of its libavcodec.pc file. Potentially, on the custom ffmpeg install with custom layout
    # our calculation can be wrong and test might fail.
    result = subprocess.run(
        ["pkg-config", "--variable=prefix", "libavcodec"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    ver = f"{torchcodec.ffmpeg_major_version}"
    my_env = os.environ.copy()
    my_env[f"TORCHCODEC_FFMPEG{ver}_INSTALL_PREFIX"] = Path(f"{result.stdout.strip()}")

    result = subprocess.run(cmake_args, cwd=tmp_path, env=my_env)
    assert result.returncode == 0

    result = subprocess.run(["cmake", "--build", "."], cwd=tmp_path)
    assert result.returncode == 0

    # loading built .so in the separate process to avoid flooding current process
    ver = f"{torchcodec.ffmpeg_major_version}"
    result = subprocess.run(
        [
            "python3",
            "-c",
            f"import torch; torch.ops.load_library('torchcodec_third_party_interface_test{ver}.so')",
        ],
        cwd=tmp_path,
    )
    assert result.returncode == 0
