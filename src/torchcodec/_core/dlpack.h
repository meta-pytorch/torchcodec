// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Vendored, trimmed copy of the DLPack spec header
// (https://github.com/dmlc/dlpack), based on DLPack v1.x.
//
// Why vendored: the torch-free torchcodec core must not depend on PyTorch
// headers. PyTorch ships <ATen/dlpack.h>, but it is explicitly guarded with
//   #if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)
// and the core is compiled with TORCH_TARGET_VERSION defined, so it cannot be
// included here. We therefore vendor the small, stable C structs we need.
//
// We export via the classic (unversioned) `DLManagedTensor` + a PyCapsule named
// "dltensor", which is the maximally-compatible exchange path understood by
// numpy (>=1.23), torch, cupy, jax, and tensorflow.

#ifndef TORCHCODEC_DLPACK_H_
#define TORCHCODEC_DLPACK_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//! \brief The DLPack version (major.minor).
typedef struct {
  uint32_t major;
  uint32_t minor;
} DLPackVersion;

//! \brief The device type in DLDevice.
#ifdef __cplusplus
typedef enum : int32_t {
#else
typedef enum {
#endif
  kDLCPU = 1,
  kDLCUDA = 2,
  kDLCUDAHost = 3,
  kDLOpenCL = 4,
  kDLVulkan = 7,
  kDLMetal = 8,
  kDLVPI = 9,
  kDLROCM = 10,
  kDLROCMHost = 11,
  kDLExtDev = 12,
  kDLCUDAManaged = 13,
  kDLOneAPI = 14,
  kDLWebGPU = 15,
  kDLHexagon = 16,
  kDLMAIA = 17,
} DLDeviceType;

//! \brief A Device for Tensor and operator.
typedef struct {
  DLDeviceType device_type;
  int32_t device_id;
} DLDevice;

//! \brief The type code options DLDataType.
typedef enum {
  kDLInt = 0U,
  kDLUInt = 1U,
  kDLFloat = 2U,
  kDLOpaqueHandle = 3U,
  kDLBfloat = 4U,
  kDLComplex = 5U,
  kDLBool = 6U,
} DLDataTypeCode;

//! \brief The data type the tensor can hold.
//!
//! Examples:
//!   - float32: code = kDLFloat, bits = 32, lanes = 1
//!   - uint8:   code = kDLUInt,  bits = 8,  lanes = 1
//!   - bool:    code = kDLBool,  bits = 8,  lanes = 1
typedef struct {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
} DLDataType;

//! \brief Plain C Tensor object, does not manage memory.
typedef struct {
  void* data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t* shape;
  int64_t* strides; // in number of elements, not bytes
  uint64_t byte_offset;
} DLTensor;

//! \brief C Tensor object, manages memory of DLTensor.
//!
//! This is the classic (pre-v1.0) exchange struct, paired with a PyCapsule
//! named "dltensor". It remains the most widely supported import path across
//! frameworks.
typedef struct DLManagedTensor {
  DLTensor dl_tensor;
  void* manager_ctx;
  void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;

// bit masks used in DLManagedTensorVersioned
#define DLPACK_FLAG_BITMASK_READ_ONLY (1UL << 0UL)
#define DLPACK_FLAG_BITMASK_IS_COPIED (1UL << 1UL)

//! \brief A versioned and managed C Tensor object (current standard, paired
//! with a PyCapsule named "dltensor_versioned"). Provided for completeness;
//! the core currently exports the unversioned form above.
typedef struct DLManagedTensorVersioned {
  DLPackVersion version;
  void* manager_ctx;
  void (*deleter)(struct DLManagedTensorVersioned* self);
  uint64_t flags;
  DLTensor dl_tensor;
} DLManagedTensorVersioned;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TORCHCODEC_DLPACK_H_
