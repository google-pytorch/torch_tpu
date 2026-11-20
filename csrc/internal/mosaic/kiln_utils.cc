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

#include "csrc/internal/mosaic/kiln_utils.h"

#include "absl/status/statusor.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace torch_tpu {

absl::StatusOr<kiln::DType> ToKilnDType(const mlir::ElementType element_type) {
  switch (element_type) {
    case mlir::ElementType::PRED:
      return kiln::bool_;
    case mlir::ElementType::I8:
      return kiln::i8;
    case mlir::ElementType::I16:
      return kiln::i16;
    case mlir::ElementType::I32:
      return kiln::i32;
    case mlir::ElementType::I64:
      return kiln::i64;
    case mlir::ElementType::UI8:
      return kiln::u8;
    case mlir::ElementType::UI16:
      return kiln::u16;
    case mlir::ElementType::UI32:
      return kiln::u32;
    case mlir::ElementType::UI64:
      return kiln::u64;
    case mlir::ElementType::F16:
      return kiln::f16;
    case mlir::ElementType::BF16:
      return kiln::bf16;
    case mlir::ElementType::F32:
      return kiln::f32;
    default:
      return TT_ERROR(  // ERROR_COV_INFEASIBLE=cannot be reached through Python
                 error::kInvalidArgument)
             << "unsupported Kiln dtype " << ToString(element_type);
  }
}

}  // namespace torch_tpu
