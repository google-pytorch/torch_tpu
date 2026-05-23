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

#include "torch_tpu/common/shape.h"

#include <sstream>
#include <string>

#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/utils.h"
#include "xla/shape.h"

namespace torch_tpu {

absl::StatusOr<Shape> MakeShape(const xla::Shape& xla_shape) {
  TT_ASSIGN_OR_RETURN(mlir::ElementType result_dtype,
                      ConvertTo<mlir::ElementType>(xla_shape.element_type()));
  return Shape(CopyIntVector(xla_shape.dimensions()), result_dtype);
}

std::string ToString(const Shape& s) {
  std::ostringstream os;
  for (auto d : s.dimensions()) {
    os << d << "x";
  }
  os << ToShortString(s.dtype());
  if (!s.dynamic_dimensions().empty()) {
    os << "[";
    for (auto d : s.dynamic_dimensions()) {
      os << d.dimension << ":" << d.lower_bound << ":" << d.upper_bound << ",";
    }
    os << "]";
  }
  return os.str();
}

}  // namespace torch_tpu
