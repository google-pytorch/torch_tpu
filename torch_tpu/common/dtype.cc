// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_tpu/common/dtype.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "ATen/AccumulateType.h"
#include "ATen/core/ATen_fwd.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "c10/core/ScalarType.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/macro_utils.h"
#include "torch_tpu/common/to_string.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

namespace internal {

xla::PrimitiveType ToXlaType(const mlir::ElementType element_type) {
  switch (element_type) {
    case mlir::ElementType::PRED:
      return xla::PrimitiveType::PRED;
    case mlir::ElementType::I2:
      return xla::PrimitiveType::S2;
    case mlir::ElementType::I4:
      return xla::PrimitiveType::S4;
    case mlir::ElementType::I8:
      return xla::PrimitiveType::S8;
    case mlir::ElementType::I16:
      return xla::PrimitiveType::S16;
    case mlir::ElementType::I32:
      return xla::PrimitiveType::S32;
    case mlir::ElementType::I64:
      return xla::PrimitiveType::S64;
    case mlir::ElementType::UI2:
      return xla::PrimitiveType::U2;
    case mlir::ElementType::UI4:
      return xla::PrimitiveType::U4;
    case mlir::ElementType::UI8:
      return xla::PrimitiveType::U8;
    case mlir::ElementType::UI16:
      return xla::PrimitiveType::U16;
    case mlir::ElementType::UI32:
      return xla::PrimitiveType::U32;
    case mlir::ElementType::UI64:
      return xla::PrimitiveType::U64;
    case mlir::ElementType::BF16:
      return xla::PrimitiveType::BF16;
    case mlir::ElementType::F16:
      return xla::PrimitiveType::F16;
    case mlir::ElementType::F32:
      return xla::PrimitiveType::F32;
    case mlir::ElementType::F64:
      return xla::PrimitiveType::F64;
    case mlir::ElementType::F4E2M1FN:
      return xla::PrimitiveType::F4E2M1FN;
    case mlir::ElementType::F8E4M3:
      return xla::PrimitiveType::F8E4M3;
    case mlir::ElementType::F8E4M3FN:
      return xla::PrimitiveType::F8E4M3FN;
    case mlir::ElementType::F8E4M3FNUZ:
      return xla::PrimitiveType::F8E4M3FNUZ;
    case mlir::ElementType::F8E5M2:
      return xla::PrimitiveType::F8E5M2;
    case mlir::ElementType::F8E5M2FNUZ:
      return xla::PrimitiveType::F8E5M2FNUZ;
    case mlir::ElementType::F8E8M0FNU:
      return xla::PrimitiveType::F8E8M0FNU;
    case mlir::ElementType::COMPLEXF32:
      return xla::PrimitiveType::C64;
    case mlir::ElementType::COMPLEXF64:
      return xla::PrimitiveType::C128;
    case mlir::ElementType::F8E3M4:
      return xla::PrimitiveType::F8E3M4;
    case mlir::ElementType::F8E4M3B11FNUZ:
      return xla::PrimitiveType::F8E4M3B11FNUZ;
    case mlir::ElementType::F6E2M3FN:
      return xla::PrimitiveType::F6E2M3FN;
    case mlir::ElementType::F6E3M2FN:
      return xla::PrimitiveType::F6E3M2FN;
      // Deliberately omitting the default case. Without the default case, if an
      // enum value is added in the future and the author forgets to update this
      // switch statement, the C++ compiler will generate a warning (treated as
      // an error), forcing the author to handle the new case.
  }

  // We get at::ScalarType from PyTorch and xla::PrimitiveType from XLA/PJRT,
  // but all mlir::ElementType values are created by us and should be valid.
  // Therefore we should never reach here.
  ABSL_LOG(FATAL)  // CRASH_OK
      << "unsupported mlir::ElementType: " << ToShortString(element_type);
}

absl::StatusOr<mlir::ElementType> ToElementType(
    const at::ScalarType scalar_type) {
  switch (scalar_type) {
    // Floating points.
    case at::kFloat8_e4m3fn:
      return mlir::ElementType::F8E4M3FN;
    case at::kFloat8_e4m3fnuz:
      return mlir::ElementType::F8E4M3FNUZ;
    case at::kFloat8_e5m2:
      return mlir::ElementType::F8E5M2;
    case at::kFloat8_e5m2fnuz:
      return mlir::ElementType::F8E5M2FNUZ;
    case at::kFloat8_e8m0fnu:
      return mlir::ElementType::F8E8M0FNU;
    case at::kHalf:
      return mlir::ElementType::F16;
    case at::kBFloat16:
      return mlir::ElementType::BF16;
    case at::kFloat:
      return mlir::ElementType::F32;
    case at::kDouble:
      return mlir::ElementType::F64;
    case at::kFloat4_e2m1fn_x2:
      return mlir::ElementType::F4E2M1FN;
    // Complex types.
    case at::kComplexFloat:
      return mlir::ElementType::COMPLEXF32;
    case at::kComplexDouble:
      return mlir::ElementType::COMPLEXF64;
    // Signed ints.
    case at::kChar:
      return mlir::ElementType::I8;
    case at::kShort:
      return mlir::ElementType::I16;
    case at::kInt:
      return mlir::ElementType::I32;
    case at::kLong:
      return mlir::ElementType::I64;
    // Unsigned ints.
    case at::kByte:
      return mlir::ElementType::UI8;
    case at::kUInt16:
      return mlir::ElementType::UI16;
    case at::kUInt32:
      return mlir::ElementType::UI32;
    case at::kUInt64:
      return mlir::ElementType::UI64;
      // Boolean.
    case at::kBool:
      return mlir::ElementType::PRED;
// BComplex32 is not in the released 2.13.0 (its headers report 2.13 but do
// not define it); it lands in 2.14.
#if TT_TORCH_VERSION_GE(2, 14)
    case at::ScalarType::BComplex32:
#endif
      // These cases should never happen.
    // go/keep-sorted start
    case at::kBits16:
    case at::kBits1x8:
    case at::kBits2x4:
    case at::kBits4x2:
    case at::kBits8:
    case at::kComplexHalf:
    case at::kInt1:
    case at::kInt2:
    case at::kInt3:
    case at::kInt4:
    case at::kInt5:
    case at::kInt6:
    case at::kInt7:
    case at::kQInt32:
    case at::kQInt8:
    case at::kQUInt2x4:
    case at::kQUInt4x2:
    case at::kQUInt8:
    case at::kUInt1:
    case at::kUInt2:
    case at::kUInt3:
    case at::kUInt4:
    case at::kUInt5:
    case at::kUInt6:
    case at::kUInt7:
    case c10::ScalarType::NumOptions:
    case c10::ScalarType::Undefined:
      // go/keep-sorted end
      break;
      // Deliberately omitting the default case. Without the default case, if an
      // enum value is added in the future and the author forgets to update this
      // switch statement, the C++ compiler will generate a warning (treated as
      // an error), forcing the author to handle the new case.
  }
  return TT_ERROR(error::kPythonNotImplementedError)
         << "TorchTPU does not yet support dtype " << ToString(scalar_type);
}

absl::StatusOr<mlir::ElementType> ToElementType(
    const xla::PrimitiveType xla_type) {
  switch (xla_type) {
    // Boolean
    case xla::PrimitiveType::PRED:
      return mlir::ElementType::PRED;
    // Signed ints {2,4,8,16,32,64}
    case xla::PrimitiveType::S2:
      return mlir::ElementType::I2;
    case xla::PrimitiveType::S4:
      return mlir::ElementType::I4;
    case xla::PrimitiveType::S8:
      return mlir::ElementType::I8;
    case xla::PrimitiveType::S16:
      return mlir::ElementType::I16;
    case xla::PrimitiveType::S32:
      return mlir::ElementType::I32;
    case xla::PrimitiveType::S64:
      return mlir::ElementType::I64;
    // Unsigned ints {2,3,8,16,32,64}
    case xla::PrimitiveType::U2:
      return mlir::ElementType::UI2;
    case xla::PrimitiveType::U4:
      return mlir::ElementType::UI4;
    case xla::PrimitiveType::U8:
      return mlir::ElementType::UI8;
    case xla::PrimitiveType::U16:
      return mlir::ElementType::UI16;
    case xla::PrimitiveType::U32:
      return mlir::ElementType::UI32;
    case xla::PrimitiveType::U64:
      return mlir::ElementType::UI64;
    // Floats {16,bf16,32,64}
    case xla::PrimitiveType::F16:
      return mlir::ElementType::F16;
    case xla::PrimitiveType::BF16:
      return mlir::ElementType::BF16;
    case xla::PrimitiveType::F32:
      return mlir::ElementType::F32;
    case xla::PrimitiveType::F64:
      return mlir::ElementType::F64;
    // FP8 types
    case xla::PrimitiveType::F8E5M2:
      return mlir::ElementType::F8E5M2;
    case xla::PrimitiveType::F8E4M3:
      return mlir::ElementType::F8E4M3;
    case xla::PrimitiveType::F8E4M3FN:
      return mlir::ElementType::F8E4M3FN;
    case xla::PrimitiveType::F8E4M3B11FNUZ:
      return mlir::ElementType::F8E4M3B11FNUZ;
    case xla::PrimitiveType::F8E3M4:
      return mlir::ElementType::F8E3M4;
    case xla::PrimitiveType::F8E5M2FNUZ:
      return mlir::ElementType::F8E5M2FNUZ;
    case xla::PrimitiveType::F8E4M3FNUZ:
      return mlir::ElementType::F8E4M3FNUZ;
    // MX types
    case xla::PrimitiveType::F8E8M0FNU:
      return mlir::ElementType::F8E8M0FNU;
    case xla::PrimitiveType::F4E2M1FN:
      return mlir::ElementType::F4E2M1FN;
    case xla::PrimitiveType::F6E2M3FN:
      return mlir::ElementType::F6E2M3FN;
    case xla::PrimitiveType::F6E3M2FN:
      return mlir::ElementType::F6E3M2FN;
    // Complex types
    case xla::PrimitiveType::C64:
      return mlir::ElementType::COMPLEXF32;
    case xla::PrimitiveType::C128:
      return mlir::ElementType::COMPLEXF64;
      // go/keep-sorted start
    case xla::PrimitiveType::BUFFER:
    case xla::PrimitiveType::OPAQUE_TYPE:
    case xla::PrimitiveType::PRIMITIVE_TYPE_INVALID:
    case xla::PrimitiveType::PrimitiveType_INT_MAX_SENTINEL_DO_NOT_USE_:
    case xla::PrimitiveType::PrimitiveType_INT_MIN_SENTINEL_DO_NOT_USE_:
    case xla::PrimitiveType::S1:
    case xla::PrimitiveType::TOKEN:
    case xla::PrimitiveType::TUPLE:
    case xla::PrimitiveType::U1:
    // go/keep-sorted end
    default:
      return TT_ERROR(error::kInvalidArgument)
             << "unsupported XLA PrimitiveType: " << xla_type;
      // Deliberately omitting the default case. Without the default case, if an
      // enum value is added in the future and the author forgets to update this
      // switch statement, the C++ compiler will generate a warning (treated as
      // an error), forcing the author to handle the new case.
  }
  return TT_ERROR(error::kInvalidArgument)
         << "invalid XLA PrimitiveType: " << xla_type;
}

static std::optional<mlir::ElementType> ToIntegerElementType(
    mlir::IntegerType int_type) {
  const bool is_signed = int_type.isSignless();
  switch (int_type.getWidth()) {
    case 1:
      return mlir::ElementType::PRED;
    case 2:
      return is_signed ? mlir::ElementType::I2 : mlir::ElementType::UI2;
    case 4:
      return is_signed ? mlir::ElementType::I4 : mlir::ElementType::UI4;
    case 8:
      return is_signed ? mlir::ElementType::I8 : mlir::ElementType::UI8;
    case 16:
      return is_signed ? mlir::ElementType::I16 : mlir::ElementType::UI16;
    case 32:
      return is_signed ? mlir::ElementType::I32 : mlir::ElementType::UI32;
    case 64:
      return is_signed ? mlir::ElementType::I64 : mlir::ElementType::UI64;
  }
  return std::nullopt;
}

absl::StatusOr<mlir::ElementType> ToElementType(mlir::Type type) {
  auto element_type =
      llvm::TypeSwitch<mlir::Type, std::optional<mlir::ElementType>>(type)
          .Case<mlir::IntegerType>(ToIntegerElementType)
          .Case<mlir::BFloat16Type>(
              [](auto) { return mlir::ElementType::BF16; })
          .Case<mlir::Float16Type>([](auto) { return mlir::ElementType::F16; })
          .Case<mlir::Float32Type>([](auto) { return mlir::ElementType::F32; })
          .Case<mlir::Float64Type>([](auto) { return mlir::ElementType::F64; })
          .Case<mlir::Float4E2M1FNType>(
              [](auto) { return mlir::ElementType::F4E2M1FN; })
          .Case<mlir::Float8E4M3FNType>(
              [](auto) { return mlir::ElementType::F8E4M3FN; })
          .Case<mlir::Float8E4M3FNUZType>(
              [](auto) { return mlir::ElementType::F8E4M3FNUZ; })
          .Case<mlir::Float8E5M2Type>(
              [](auto) { return mlir::ElementType::F8E5M2; })
          .Case<mlir::Float8E5M2FNUZType>(
              [](auto) { return mlir::ElementType::F8E5M2FNUZ; })
          .Case<mlir::Float8E8M0FNUType>(
              [](auto) { return mlir::ElementType::F8E8M0FNU; })
          .Case<mlir::Float6E2M3FNType>(
              [](auto) { return mlir::ElementType::F6E2M3FN; })
          .Case<mlir::Float6E3M2FNType>(
              [](auto) { return mlir::ElementType::F6E3M2FN; })
          .Case<mlir::ComplexType>([](mlir::ComplexType complex_type)
                                       -> std::optional<mlir::ElementType> {
            mlir::Type element_type = complex_type.getElementType();
            if (element_type.isF32()) {
              return mlir::ElementType::COMPLEXF32;
            }
            if (element_type.isF64()) {
              return mlir::ElementType::COMPLEXF64;
            }
            return std::nullopt;
          })
          .Default([&](mlir::Type) { return std::nullopt; });

  if (!element_type) {
    return TT_ERROR(error::kInvalidArgument)
           << "unsupported MLIR type: " << ToString(type);
  }
  return *element_type;
}

at::ScalarType ToScalarType(const mlir::ElementType element_type) {
  // See c10/core/ScalarType.h for valid ScalarType values.
  switch (element_type) {
    case mlir::ElementType::PRED:
      return at::ScalarType::Bool;
    case mlir::ElementType::I2:
      return at::ScalarType::Int2;
    case mlir::ElementType::I4:
      return at::ScalarType::Int4;
    case mlir::ElementType::I8:
      return at::ScalarType::Char;
    case mlir::ElementType::I16:
      return at::ScalarType::Short;
    case mlir::ElementType::I32:
      return at::ScalarType::Int;
    case mlir::ElementType::I64:
      return at::ScalarType::Long;
    case mlir::ElementType::UI2:
      return at::ScalarType::UInt2;
    case mlir::ElementType::UI4:
      return at::ScalarType::UInt4;
    case mlir::ElementType::UI8:
      return at::ScalarType::Byte;
    case mlir::ElementType::UI16:
      return at::ScalarType::UInt16;
    case mlir::ElementType::UI32:
      return at::ScalarType::UInt32;
    case mlir::ElementType::UI64:
      return at::ScalarType::UInt64;
    case mlir::ElementType::BF16:
      return at::ScalarType::BFloat16;
    case mlir::ElementType::F16:
      return at::ScalarType::Half;
    case mlir::ElementType::F32:
      return at::ScalarType::Float;
    case mlir::ElementType::F64:
      return at::ScalarType::Double;
    case mlir::ElementType::F8E4M3:
      return at::ScalarType::Float8_e4m3fn;
    case mlir::ElementType::F8E4M3FN:
      return at::ScalarType::Float8_e4m3fn;
    case mlir::ElementType::F8E4M3FNUZ:
      return at::ScalarType::Float8_e4m3fnuz;
    case mlir::ElementType::F8E5M2:
      return at::ScalarType::Float8_e5m2;
    case mlir::ElementType::F8E5M2FNUZ:
      return at::ScalarType::Float8_e5m2fnuz;
    case mlir::ElementType::F8E8M0FNU:
      return at::ScalarType::Float8_e8m0fnu;
    case mlir::ElementType::F4E2M1FN:
      return at::ScalarType::Float4_e2m1fn_x2;
    case mlir::ElementType::COMPLEXF32:
      return at::ScalarType::ComplexFloat;
    case mlir::ElementType::COMPLEXF64:
      return at::ScalarType::ComplexDouble;
    // These types have no corresponding ScalarType.
    case mlir::ElementType::F6E2M3FN:
    case mlir::ElementType::F6E3M2FN:
    case mlir::ElementType::F8E3M4:
    case mlir::ElementType::F8E4M3B11FNUZ:
      break;
  }

  ABSL_LOG(FATAL)  // CRASH_OK
      << "unsupported mlir::ElementType: " << ToShortString(element_type);
}

static at::ScalarType ToScalarType(const mlir::IntegerType int_type) {
  // For historical reasons, MLIR uses signless integer type for signed integers
  const bool is_signed = int_type.isSignless();
  switch (int_type.getWidth()) {
    case 1:
      return at::kBool;
    case 2:
      return is_signed ? at::kInt2 : at::kUInt2;
    case 4:
      return is_signed ? at::kInt4 : at::kUInt4;
    case 8:
      return is_signed ? at::kChar : at::kByte;
    case 16:
      return is_signed ? at::kShort : at::kUInt16;
    case 32:
      return is_signed ? at::kInt : at::kUInt32;
    case 64:
      return is_signed ? at::kLong : at::kUInt64;
  }
  ABSL_LOG(FATAL)  // CRASH_OK
      << "Cannot convert MLIR integer type " << mlir::debugString(int_type)
      << " to a PyTorch type";
}

static at::ScalarType ToScalarType(const mlir::ComplexType complex_type) {
  mlir::Type element_type = complex_type.getElementType();
  if (element_type.isF32()) {
    return at::kComplexFloat;
  }
  if (element_type.isF64()) {
    return at::kComplexDouble;
  }
  ABSL_LOG(FATAL)  // CRASH_OK
      << "Unsupported complex element type for MLIR type "
      << mlir::debugString(complex_type);
}

at::ScalarType ToScalarType(mlir::Type type) {
  using ::torch_tpu::internal::ToScalarType;
  auto scalar_type =
      llvm::TypeSwitch<mlir::Type, at::ScalarType>(type)
          .Case<mlir::IntegerType>(
              [](mlir::IntegerType int_type) { return ToScalarType(int_type); })
          .Case<mlir::BFloat16Type>([](auto) { return at::kBFloat16; })
          .Case<mlir::Float16Type>([](auto) { return at::kHalf; })
          .Case<mlir::Float32Type>([](auto) { return at::kFloat; })
          .Case<mlir::Float64Type>([](auto) { return at::kDouble; })
          .Case<mlir::Float4E2M1FNType>(
              [](auto) { return at::kFloat4_e2m1fn_x2; })
          .Case<mlir::Float8E4M3FNType>([](auto) { return at::kFloat8_e4m3fn; })
          .Case<mlir::Float8E4M3FNUZType>(
              [](auto) { return at::kFloat8_e4m3fnuz; })
          .Case<mlir::Float8E5M2Type>([](auto) { return at::kFloat8_e5m2; })
          .Case<mlir::Float8E5M2FNUZType>(
              [](auto) { return at::kFloat8_e5m2fnuz; })
          .Case<mlir::Float8E8M0FNUType>(
              [](auto) { return at::kFloat8_e8m0fnu; })
          .Case<mlir::ComplexType>([](mlir::ComplexType complex_type) {
            return ToScalarType(complex_type);
          })
          .Default([&](mlir::Type) {
            ABSL_LOG(FATAL)  // CRASH_OK
                << "cannot convert MLIR type " << ::torch_tpu::ToString(type)
                << " to a PyTorch type";
            // Unreachable. Just to make the compiler happy.
            return at::ScalarType::Undefined;
          });
  return scalar_type;
}

}  // namespace internal

std::string_view ToString(const mlir::ElementType element_type) {
  return ToString(internal::ToScalarType(element_type));
}

std::string_view ToShortString(const mlir::ElementType element_type) {
  // WARNING: These names are used for computing cache keys, so they must be
  // stable. Changing them may cause key collisions in the persistent cache.
  switch (element_type) {
    case mlir::ElementType::PRED:
      return "pred";
    case mlir::ElementType::I2:
      return "i2";
    case mlir::ElementType::I4:
      return "i4";
    case mlir::ElementType::I8:
      return "i8";
    case mlir::ElementType::I16:
      return "i16";
    case mlir::ElementType::I32:
      return "i32";
    case mlir::ElementType::I64:
      return "i64";
    case mlir::ElementType::UI2:
      return "ui2";
    case mlir::ElementType::UI4:
      return "ui4";
    case mlir::ElementType::UI8:
      return "ui8";
    case mlir::ElementType::UI16:
      return "ui16";
    case mlir::ElementType::UI32:
      return "ui32";
    case mlir::ElementType::UI64:
      return "ui64";
    case mlir::ElementType::BF16:
      return "bf16";
    case mlir::ElementType::F16:
      return "f16";
    case mlir::ElementType::F32:
      return "f32";
    case mlir::ElementType::F64:
      return "f64";
    case mlir::ElementType::F4E2M1FN:
      return "f4E2M1FN";
    case mlir::ElementType::F6E2M3FN:
      return "f6E2M3FN";
    case mlir::ElementType::F6E3M2FN:
      return "f6E3M2FN";
    case mlir::ElementType::F8E3M4:
      return "f8E3M4";
    case mlir::ElementType::F8E4M3:
      return "f8E4M3";
    case mlir::ElementType::F8E4M3FN:
      return "f8E4M3FN";
    case mlir::ElementType::F8E4M3FNUZ:
      return "f8E4M3FNUZ";
    case mlir::ElementType::F8E4M3B11FNUZ:
      return "f8E4M3B11FNUZ";
    case mlir::ElementType::F8E5M2:
      return "f8E5M2";
    case mlir::ElementType::F8E5M2FNUZ:
      return "f8E5M2FNUZ";
    case mlir::ElementType::F8E8M0FNU:
      return "f8E8M0FNU";
    case mlir::ElementType::COMPLEXF32:
      return "complexf32";
    case mlir::ElementType::COMPLEXF64:
      return "complexf64";
  }
}

namespace {

bool IsSupportedBufferType(mlir::ElementType type) {
  static const absl::NoDestructor<absl::flat_hash_set<mlir::ElementType>>
      kSupportedTypes(
          {// Boolean
           mlir::ElementType::PRED,
           // Signed integers
           mlir::ElementType::I2, mlir::ElementType::I4, mlir::ElementType::I8,
           mlir::ElementType::I16, mlir::ElementType::I32,
           mlir::ElementType::I64,
           // Unsigned integers
           mlir::ElementType::UI2, mlir::ElementType::UI4,
           mlir::ElementType::UI8, mlir::ElementType::UI16,
           mlir::ElementType::UI32, mlir::ElementType::UI64,
           // Floats
           mlir::ElementType::F16, mlir::ElementType::BF16,
           mlir::ElementType::F32, mlir::ElementType::F64,
           // FP8 types
           mlir::ElementType::F8E5M2, mlir::ElementType::F8E4M3,
           mlir::ElementType::F8E4M3FN, mlir::ElementType::F8E4M3B11FNUZ,
           mlir::ElementType::F8E3M4, mlir::ElementType::F8E5M2FNUZ,
           mlir::ElementType::F8E4M3FNUZ, mlir::ElementType::F8E8M0FNU,
           // FP6 types
           mlir::ElementType::F6E2M3FN, mlir::ElementType::F6E3M2FN,
           // FP4 types
           mlir::ElementType::F4E2M1FN,
           // Complex types
           mlir::ElementType::COMPLEXF32, mlir::ElementType::COMPLEXF64});
  return kSupportedTypes->contains(type);
}

[[nodiscard]] Dimensions GetNegativeValuesIn(const at::IntArrayRef ints) {
  Dimensions result;
  for (const int64_t v : ints) {
    if (v < 0) {
      result.push_back(v);
    }
  }
  return result;
}

template <typename Container>
[[nodiscard]] std::string JoinBySerialComma(const Container& container) {
  std::string result;
  const int num_items = container.size();
  if (num_items == 1) {
    result = absl::StrCat(container[0]);
  } else if (num_items == 2) {
    result = absl::StrCat(container[0], " and ", container[1]);
  } else if (num_items >= 3) {
    result = absl::StrJoin(
        absl::MakeConstSpan(container).subspan(0, num_items - 1), ", ");
    absl::StrAppend(&result, ", and ", container[num_items - 1]);
  }
  return result;
}

}  // namespace

absl::StatusOr<int64_t> ValidateTensorByteSize(at::IntArrayRef size,
                                               mlir::ElementType element_type) {
  TT_RET_CHECK(IsSupportedBufferType(element_type),
               error::kPythonNotImplementedError)
      << "element type " << ToShortString(element_type)
      << " is not supported as a tensor element type";
  const auto negative_dims = GetNegativeValuesIn(size);
  TT_RET_CHECK(negative_dims.empty(), error::kInvalidArgument)
      << "dimension sizes must be >= 0, got [" << absl::StrJoin(size, ", ")
      << "], which contains " << JoinBySerialComma(negative_dims);
  TT_ASSIGN_OR_RETURN(const int64_t extent_product, NumElements(size));
  const auto xla_primitive_type = ConvertTo<xla::PrimitiveType>(element_type);
  TT_ASSIGN_OR_RETURN(
      const int64_t byte_size,
      SafeMultiply(extent_product,
                   xla::ShapeUtil::ByteSizeOfPrimitiveType(xla_primitive_type)),
      _.SetOverride() << "product of dimension sizes ["
                      << absl::StrJoin(size, ", ") << "] and size of "
                      << ToString(element_type) << " ("
                      << xla::ShapeUtil::ByteSizeOfPrimitiveType(
                             xla_primitive_type)
                      << " bytes) overflows as int64");
  return byte_size;
}

int64_t TorchEquivalentBitwidth(mlir::ElementType element_type) {
  switch (element_type) {
    // PyTorch stores booleans as uint8s, even though they're called "I1" in
    // MLIR.
    case mlir::ElementType::PRED:
      return 8;

    // Torch does not actually support 2- or 6-bit integer types, so these
    // cases will never be reached. Return the MLIR bitwidth anyway.
    case mlir::ElementType::I2:
    case mlir::ElementType::UI2:
      return 2;
    case mlir::ElementType::F6E2M3FN:
    case mlir::ElementType::F6E3M2FN:
      return 6;

    // StableHLO's complex32 is equivalent to PyTorch's complex64/cfloat.
    case mlir::ElementType::COMPLEXF32:
      return 64;
    // StableHLO's complex64 is equivalent to PyTorch's complex128/cdouble.
    case mlir::ElementType::COMPLEXF64:
      return 128;

    // For everything else, the torch bitwidth is the same as the MLIR bitwidth.
    case mlir::ElementType::I4:
    case mlir::ElementType::UI4:
    case mlir::ElementType::F4E2M1FN:
      return 4;
    case mlir::ElementType::I8:
    case mlir::ElementType::UI8:
    case mlir::ElementType::F8E3M4:
    case mlir::ElementType::F8E4M3:
    case mlir::ElementType::F8E4M3FN:
    case mlir::ElementType::F8E4M3FNUZ:
    case mlir::ElementType::F8E4M3B11FNUZ:
    case mlir::ElementType::F8E5M2:
    case mlir::ElementType::F8E5M2FNUZ:
    case mlir::ElementType::F8E8M0FNU:
      return 8;
    case mlir::ElementType::I16:
    case mlir::ElementType::UI16:
    case mlir::ElementType::F16:
    case mlir::ElementType::BF16:
      return 16;
    case mlir::ElementType::I32:
    case mlir::ElementType::UI32:
    case mlir::ElementType::F32:
      return 32;
    case mlir::ElementType::I64:
    case mlir::ElementType::UI64:
    case mlir::ElementType::F64:
      return 64;
      // Deliberately no default case, so that the compiler will warn if a new
      // MLIR type is added.
  }
}

mlir::ElementType RealComponentOf(const mlir::ElementType element_type) {
  switch (element_type) {
    case mlir::ElementType::COMPLEXF32:
      return mlir::ElementType::F32;
    case mlir::ElementType::COMPLEXF64:
      return mlir::ElementType::F64;
    default:
      return element_type;
  }
}

// PyTorch uses `at::toAccumulateType` to determine this. We call it with
// `is_cuda=true` to ensure we get the CUDA-aligned accumulation type.
// For full details on PyTorch's accumulation type mapping, see:
// third_party/py/torch/aten/src/ATen/AccumulateType.h
at::ScalarType ToAccumulateType(at::ScalarType type) {
  return at::toAccumulateType(  // AT_TO_ACCUMULATE_TYPE_OK=root usage for the
                                // API.
      type,
      /*is_cuda=*/true);
}

}  // namespace torch_tpu
