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
#include <cstdint>
#include <exception>
#include <sstream>
#include <string>

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

std::string ToString(const at::ScalarType& scalar_type,
                     const std::string& name) {
  std::stringstream os;
  if (!name.empty()) {
    os << name << ": ";
  }
  os << "(type: " << static_cast<int32_t>(scalar_type) << ")";
  return os.str();
}

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

absl::StatusOr<mlir::ElementType> InferComputationDtype(
    mlir::ElementType input_dtype) {
  at::ScalarType input_type = ConvertTo<at::ScalarType>(input_dtype);
  // at::toOpMathType preserves FP8/FP4 types, but XLA HLO operations generally
  // require working on standard FP32 formats. We intercept FP4 here to
  // explicitly specify F32 computation.
  if (input_type == at::kFloat4_e2m1fn_x2) {
    return mlir::ElementType::F32;
  }
  at::ScalarType computation_type = at::toOpMathType(input_type);
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
