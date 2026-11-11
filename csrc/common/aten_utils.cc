/*
 * Copyright 2025 Google LLC
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

#include "csrc/common/aten_utils.h"

#include <cmath>
#include <exception>

#include "ATen/OpMathType.h"
#include "ATen/core/ATen_fwd.h"
#include "absl/status/statusor.h"
#include "c10/core/DefaultDtype.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

bool operator==(const HashableScalar& lhs, const HashableScalar& rhs) {
  // If the target types are different, they are not equal.
  if (lhs.scalar_type != rhs.scalar_type) return false;

  // If the host types are different, they are not equal.
  if (lhs.scalar.type() != rhs.scalar.type()) return false;

  // If the target types are the same, then we check that the host types and
  // values are the same as well.
  switch (lhs.scalar.type()) {
    case c10::ScalarType::ComplexDouble: {
      auto l_val = lhs.scalar.toComplexDouble();
      auto r_val = rhs.scalar.toComplexDouble();
      bool l_real_nan = std::isnan(l_val.real());
      bool r_real_nan = std::isnan(r_val.real());
      bool l_imag_nan = std::isnan(l_val.imag());
      bool r_imag_nan = std::isnan(r_val.imag());

      bool real_equal =
          (l_real_nan && r_real_nan) || (l_val.real() == r_val.real());
      bool imag_equal =
          (l_imag_nan && r_imag_nan) || (l_val.imag() == r_val.imag());
      return real_equal && imag_equal;
    }
    case c10::ScalarType::Double: {
      double l_val = lhs.scalar.toDouble();
      double r_val = rhs.scalar.toDouble();
      bool l_nan = std::isnan(l_val);
      bool r_nan = std::isnan(r_val);
      return (l_nan && r_nan) || (l_val == r_val);
    }
    case c10::ScalarType::UInt64:
      return lhs.scalar.toUInt64() == rhs.scalar.toUInt64();
    case c10::ScalarType::Long:
      return lhs.scalar.toLong() == rhs.scalar.toLong();
    case c10::ScalarType::Bool:
      return lhs.scalar.toBool() == rhs.scalar.toBool();
    default:
      return false;
  }
}

c10::ScalarType InferOutputDtype(const at::Tensor& self) {
  const c10::ScalarType input_dtype = self.scalar_type();
  if (c10::isFloatingType(input_dtype) || c10::isComplexType(input_dtype)) {
    return input_dtype;
  }
  return c10::get_default_dtype_as_scalartype();
}

// TorchTPU kernels must call InferComputationDtype rather than at::toOpMathType
// directly so that all TorchTPU-supported dtypes map to a valid XLA HLO
// computation dtype without crashing on types omitted by upstream ATen.
at::ScalarType InferComputationDtype(at::ScalarType input_dtype) {
  switch (input_dtype) {
    // Upstream at::toOpMathType only promotes BFloat16, Half, and FP8 variants
    // to Float, leaving Float4_e2m1fn_x2 unchanged, whereas XLA HLO arithmetic
    // requires promoting Float4_e2m1fn_x2 to Float (f32).
    case at::ScalarType::Float4_e2m1fn_x2:
      return at::ScalarType::Float;
    // Upstream at::toOpMathType switches over AT_ALL_SCALAR_TYPES_WITH_COMPLEX,
    // which omits barebones unsigned types (UInt16, UInt32, UInt64) and
    // sub-byte integer types (Int2, Int4, UInt2, UInt4) and crashes with
    // TORCH_INTERNAL_ASSERT(false, "Unrecognized ScalarType: ...") on them,
    // even though at::OpMathType<T> is the identity type for all integral
    // types.
    case at::ScalarType::Int2:
    case at::ScalarType::Int4:
    case at::ScalarType::UInt2:
    case at::ScalarType::UInt4:
    case at::ScalarType::UInt16:
    case at::ScalarType::UInt32:
    case at::ScalarType::UInt64:
      return input_dtype;
    default:
      break;
  }
  return at::toOpMathType(  // AT_TO_OP_MATH_TYPE_OK=implementing
                            // InferComputationDtype.
      input_dtype);
}

absl::StatusOr<mlir::ElementType> InferComputationDtype(
    mlir::ElementType input_dtype) {
  const at::ScalarType input_type = ConvertTo<at::ScalarType>(input_dtype);
  const at::ScalarType computation_type = InferComputationDtype(input_type);
  if (computation_type == input_type) {
    return input_dtype;
  }
  return ConvertTo<mlir::ElementType>(computation_type);
}

absl::StatusOr<bool> CanCastScalarWithoutOverflow(const at::Scalar& scalar,
                                                  at::ScalarType scalar_type) {
  try {
    switch (scalar_type) {
#define TT_DEFINE_CASE(_, name) \
  case at::ScalarType::name:    \
    scalar.to##name();          \
    break;

      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(TT_DEFINE_CASE)
      TT_DEFINE_CASE(uint16_t, UInt16)
      TT_DEFINE_CASE(uint32_t, UInt32)
      TT_DEFINE_CASE(uint64_t, UInt64)

#undef TT_DEFINE_CASE

      default:
        return TT_ERROR(error::kInvalidArgument)
               << "unsupported cast of " << ToString(scalar.type())
               << " scalar (" << ToString(scalar) << ") to type "
               << ToString(scalar_type);
    }
  } catch (const std::exception& /*e*/) {
    return false;
  }

  return true;
}

}  // namespace torch_tpu
