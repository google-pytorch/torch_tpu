/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "torch_tpu/common/to_string.h"

#include <cstdint>
#include <string>
#include <string_view>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Types.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

std::string ToString(const c10d::AllgatherOptions& options) {
  return absl::StrCat("{asyncOp=", options.asyncOp, "}");
}

std::string ToString(const c10d::BroadcastOptions& options) {
  return absl::StrCat("{rootRank=", options.rootRank,
                      ", rootTensor=", options.rootTensor,
                      ", asyncOp=", options.asyncOp, "}");
}

std::string ToString(const c10d::ScatterOptions& options) {
  return absl::StrCat("{rootRank=", options.rootRank,
                      ", asyncOp=", options.asyncOp, "}");
}

std::string ToString(const c10d::GatherOptions& options) {
  return absl::StrCat("{rootRank=", options.rootRank,
                      ", asyncOp=", options.asyncOp, "}");
}

std::string ToString(const c10d::ReduceScatterOptions& options) {
  return absl::StrCat("{reduceOp=", ToString(options.reduceOp),
                      ", asyncOp=", options.asyncOp, "}");
}

std::string ToString(const c10d::AllToAllOptions& options) {
  return absl::StrCat("{asyncOp=", options.asyncOp, "}");
}

std::string ToString(const c10d::BarrierOptions& options) {
  return absl::StrCat("{device_ids=[", absl::StrJoin(options.device_ids, ", "),
                      "], asyncOp=", options.asyncOp, "}");
}

std::string ToString(c10d::ReduceOp::RedOpType reduce_op_type) {
  switch (reduce_op_type) {
    // go/keep-sorted start
    case c10d::ReduceOp::AVG:
      return "avg";
    case c10d::ReduceOp::BAND:
      return "band";
    case c10d::ReduceOp::BOR:
      return "bor";
    case c10d::ReduceOp::BXOR:
      return "bxor";
    case c10d::ReduceOp::MAX:
      return "max";
    case c10d::ReduceOp::MIN:
      return "min";
    case c10d::ReduceOp::PREMUL_SUM:
      return "premul_sum";
    case c10d::ReduceOp::PRODUCT:
      return "product";
    case c10d::ReduceOp::SUM:
      return "sum";
    case c10d::ReduceOp::UNUSED:
      return "unused";
      // default case not needed, since the switch is exhaustive.
      // go/keep-sorted end
  };
  return absl::StrFormat("enum%d", reduce_op_type);
}

std::string ToString(const at::Tensor& tensor, const std::string& name) {
  std::string result;
  if (!name.empty()) {
    absl::StrAppend(&result, name, ": ");
  }
  if (!tensor.defined()) {
    absl::StrAppend(&result, "undefined");
    return result;
  }
  absl::StrAppend(                                                //
      &result,                                                    //
      "shape=[", absl::StrJoin(tensor.sizes(), ", "), "], ",      //
      "strides=[", absl::StrJoin(tensor.strides(), ", "), "], ",  //
      "numel=", tensor.numel(), ", ",                             //
      "dtype=", tensor.scalar_type(), ", ",                       //
      "device=", ToString(tensor.device()), ", ",                 //
      "is_contiguous=", tensor.is_contiguous(), ", ",             //
      "is_cpu=", tensor.is_cpu(), ", ",                           //
      "is_cuda=", tensor.is_cuda(), ", ",                         //
      "is_meta=", tensor.is_meta(), ", ",                         //
      "storage_offset=", tensor.storage_offset(), ", ");

  if (tensor.defined() && tensor.storage().unsafeGetStorageImpl() != nullptr) {
    absl::StrAppend(&result, "storage_use_count=", tensor.storage().use_count(),
                    ", ");
    if (tensor.storage().data_ptr().get_context() != nullptr) {
      // FIXME: The following code is commented out because it causes ASAN
      // violations on ops_test.py.
      //
      // DeviceBufferRef* buffer_ref = static_cast<DeviceBufferRef*>(
      //     tensor.storage().data_ptr().get_context());
      // if (buffer_ref != nullptr && !buffer_ref->dimensions.empty()) {
      //   ss << "buffer_ref_dims= not null [" << buffer_ref->dimensions.size()
      //      << "],\n";
      // } else {
      //   ss << "buffer_ref_dims=null";
      // }
    } else {
      absl::StrAppend(&result, "context=null (but storage is defined)");
    }
  } else {
    absl::StrAppend(&result, "storage_is_null(undefined)");
  }
  return result;
}

std::string ToString(const at::Scalar& scalar, const std::string& name) {
  std::string result;
  if (!name.empty()) {
    absl::StrAppend(&result, name, ": ");
  }
  if (scalar.isFloatingPoint()) {
    absl::StrAppend(&result, scalar.toDouble());
  } else if (scalar.isIntegral(/*include_bool=*/false)) {
    absl::StrAppend(&result, scalar.toLong());
  } else if (scalar.isBoolean()) {
    absl::StrAppend(&result, scalar.toBool() ? "true" : "false");
  } else if (scalar.isComplex()) {
    absl::StrAppend(&result, ToString(scalar.toComplexDouble()));
  } else {
    absl::StrAppend(&result, "(type: ", static_cast<int32_t>(scalar.type()),
                    ", value: ?)");
  }
  return result;
}

std::string_view ToString(const at::ScalarType scalar_type) {
  // See https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch-dtype
  // for the list of supported PyTorch dtype names. When there are aliases (e.g.
  // complex64 -> cfloat), we prefer the name with the explicit width (e.g.
  // complex64).
  switch (scalar_type) {
    // Unsigned ints.
    case at::ScalarType::UInt1:
      return "uint1";
    case at::ScalarType::UInt2:
      return "uint2";
    case at::ScalarType::UInt3:
      return "uint3";
    case at::ScalarType::UInt4:
      return "uint4";
    case at::ScalarType::UInt5:
      return "uint5";
    case at::ScalarType::UInt6:
      return "uint6";
    case at::ScalarType::UInt7:
      return "uint7";
    case at::ScalarType::Byte:
      return "uint8";
    case at::ScalarType::UInt16:
      return "uint16";
    case at::ScalarType::UInt32:
      return "uint32";
    case at::ScalarType::UInt64:
      return "uint64";
    // Signed ints.
    case at::ScalarType::Int1:
      return "int1";
    case at::ScalarType::Int2:
      return "int2";
    case at::ScalarType::Int3:
      return "int3";
    case at::ScalarType::Int4:
      return "int4";
    case at::ScalarType::Int5:
      return "int5";
    case at::ScalarType::Int6:
      return "int6";
    case at::ScalarType::Int7:
      return "int7";
    case at::ScalarType::Char:
      return "int8";
    case at::ScalarType::Short:
      return "int16";
    case at::ScalarType::Int:
      return "int32";
    case at::ScalarType::Long:
      return "int64";
    // Quantized signed ints.
    case at::ScalarType::QInt8:
      return "qint8";
    case at::ScalarType::QInt32:
      return "qint32";
    // Quantized unsigned ints.
    case at::ScalarType::QUInt2x4:
      return "quint2x4";
    case at::ScalarType::QUInt4x2:
      return "quint4x2";
    case at::ScalarType::QUInt8:
      return "quint8";
    // Uninterpreted bits.
    case at::ScalarType::Bits1x8:
      return "bits1x8";
    case at::ScalarType::Bits2x4:
      return "bits2x4";
    case at::ScalarType::Bits4x2:
      return "bits4x2";
    case at::ScalarType::Bits8:
      return "bits8";
    case at::ScalarType::Bits16:
      return "bits16";
    // Floating points.
    case at::ScalarType::Half:
      return "float16";
    case at::ScalarType::BFloat16:
      return "bfloat16";
    case at::ScalarType::Float:
      return "float32";
    case at::ScalarType::Double:
      return "float64";
    // Complex types.
    case at::ScalarType::ComplexFloat:
      return "complex64";
    case at::ScalarType::ComplexDouble:
      return "complex128";
    case at::ScalarType::ComplexHalf:
      return "complex32";  // Boolean.
    case at::ScalarType::Bool:
      return "bool";
    // FP8 types: https://docs.pytorch.org/docs/stable/tensor_attributes.html
    case at::ScalarType::Float8_e4m3fn:
      return "float8_e4m3fn";
    case at::ScalarType::Float8_e5m2:
      return "float8_e5m2";
    case at::ScalarType::Float8_e4m3fnuz:
      return "float8_e4m3fnuz";
    case at::ScalarType::Float8_e5m2fnuz:
      return "float8_e5m2fnuz";
    case at::ScalarType::Float8_e8m0fnu:
      return "float8_e8m0fnu";
    case at::ScalarType::Float4_e2m1fn_x2:
      return "float4_e2m1fn_x2";
    // These cases are listed for completeness but should never be reached.
    case at::ScalarType::Undefined:
      return "<undefined>";
    case at::ScalarType::NumOptions:
      return "<num_options>";
      // Deliberately omitting the default case. We want the compiler to
      // force us to handle all cases explicitly.
  }
}

std::string ToString(const mlir::Type type) {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << type;
  return str;
}

}  // namespace torch_tpu
