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

#ifndef TORCH_TPU_OPS_UNARY_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_UNARY_ATEN_KERNELS_H_

#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/DeprecatedTypeProperties.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

struct UnaryOpOptions {
  // The op name for dispatching. If omitted, use the op name from the active
  // TT_KERNEL() context.
  std::optional<OpName> op_name = std::nullopt;
  // Parameters (if any) that were used to construct op_builder and will be used
  // by the compilation cache.
  OpParamCacheKeys op_param_cache_keys;
  // dtype of the output tensor. If not specified, use self's dtype.
  std::optional<mlir::ElementType> out_dtype;
  // Size of the output tensor. If not specified use self's size.
  std::optional<at::IntArrayRef> out_dims;
  // If specified, all inputs will be casted to this dtype before
  // the op_builder is applied.
  std::optional<mlir::ElementType> computation_dtype;
};

// Safely applies a functional unary operation to the input tensor.
// Args:
//    self: The input tensor.
//    op_builder: The unary operation to apply.
//    options: See ApplyOpOptions.
// Returns:
//   A new tensor with the result of the operation.
absl::StatusOr<at::Tensor> UnaryOp(const at::Tensor& self,
                                   MlirUnaryOpBuilder op_builder,
                                   UnaryOpOptions options);

// Like UnaryOpCallback, but the result overwrites the input.
// Because of this overwrite, the output must have the shape and dtype of self.
// Hence .out_dims and .out_dtype in options are ignored.
absl::Status UnaryOpInPlace(at::Tensor& self, MlirUnaryOpBuilder op_builder,
                            UnaryOpOptions options);

// Like UnaryOpCallback, but the result overwrites the provided `out` tensor.
// `out` is resized to `out_dims` and must have dtype `out_dtype`.
// If they are not provided then the values from `self` are used.
absl::Status UnaryOpOut(const at::Tensor& self, at::Tensor& out,
                        MlirUnaryOpBuilder op_builder, UnaryOpOptions options);

// Declare callbacks for a unary ATEN op
#define TT_DECLARE_ATEN_UNARY_OUT(func_name) \
  at::Tensor& func_name##Out(const at::Tensor& self, at::Tensor& out)

// Relu requires both an in-place and out-of-place kernel.
at::Tensor AtenRelu(const at::Tensor& self);
at::Tensor& AtenRelu_(at::Tensor& self);

// go/keep-sorted start
TT_DECLARE_ATEN_UNARY_OUT(AtenAbs);
TT_DECLARE_ATEN_UNARY_OUT(AtenAcos);
TT_DECLARE_ATEN_UNARY_OUT(AtenAcosh);
TT_DECLARE_ATEN_UNARY_OUT(AtenAsin);
TT_DECLARE_ATEN_UNARY_OUT(AtenAsinh);
TT_DECLARE_ATEN_UNARY_OUT(AtenAtan);
TT_DECLARE_ATEN_UNARY_OUT(AtenAtanh);
TT_DECLARE_ATEN_UNARY_OUT(AtenBitwiseNot);
TT_DECLARE_ATEN_UNARY_OUT(AtenCeil);
TT_DECLARE_ATEN_UNARY_OUT(AtenConjPhysical);
TT_DECLARE_ATEN_UNARY_OUT(AtenCos);
TT_DECLARE_ATEN_UNARY_OUT(AtenCosh);
TT_DECLARE_ATEN_UNARY_OUT(AtenErf);
TT_DECLARE_ATEN_UNARY_OUT(AtenErfInv);
TT_DECLARE_ATEN_UNARY_OUT(AtenExp);
TT_DECLARE_ATEN_UNARY_OUT(AtenExp2);
TT_DECLARE_ATEN_UNARY_OUT(AtenExpm1);
TT_DECLARE_ATEN_UNARY_OUT(AtenFloor);
TT_DECLARE_ATEN_UNARY_OUT(AtenLgamma);
TT_DECLARE_ATEN_UNARY_OUT(AtenLog);
TT_DECLARE_ATEN_UNARY_OUT(AtenLog10);
TT_DECLARE_ATEN_UNARY_OUT(AtenLog1p);
TT_DECLARE_ATEN_UNARY_OUT(AtenLog2);
TT_DECLARE_ATEN_UNARY_OUT(AtenNeg);
TT_DECLARE_ATEN_UNARY_OUT(AtenReciprocal);
TT_DECLARE_ATEN_UNARY_OUT(AtenRelu);
TT_DECLARE_ATEN_UNARY_OUT(AtenRsqrt);
TT_DECLARE_ATEN_UNARY_OUT(AtenSgn);
TT_DECLARE_ATEN_UNARY_OUT(AtenSigmoid);
TT_DECLARE_ATEN_UNARY_OUT(AtenSign);
TT_DECLARE_ATEN_UNARY_OUT(AtenSignbit);
TT_DECLARE_ATEN_UNARY_OUT(AtenSilu);
TT_DECLARE_ATEN_UNARY_OUT(AtenSin);
TT_DECLARE_ATEN_UNARY_OUT(AtenSinh);
TT_DECLARE_ATEN_UNARY_OUT(AtenSqrt);
TT_DECLARE_ATEN_UNARY_OUT(AtenTan);
TT_DECLARE_ATEN_UNARY_OUT(AtenTanh);
TT_DECLARE_ATEN_UNARY_OUT(AtenTrunc);
// go/keep-sorted end

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_UNARY_ATEN_KERNELS_H_
