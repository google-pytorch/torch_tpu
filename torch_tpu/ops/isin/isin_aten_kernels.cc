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

#include "torch_tpu/ops/isin/isin_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/native/Resize.h"
#include "ATen/ops/full.h"
#include "ATen/ops/promote_types.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/isin/isin.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace {

absl::StatusOr<DeviceBufferRef> AtenIsInHelper(const at::Tensor& elements,
                                               const at::Tensor& test_elements,
                                               bool assume_unique, bool invert,
                                               OpParamCacheKeys& param_keys) {
  ABSL_VLOG(1) << "AtenIsInHelper assume_unique=" << assume_unique
               << " invert=" << invert;

  at::ScalarType computation_scalar_type =
      at::promote_types(elements.scalar_type(), test_elements.scalar_type());
  TT_ASSIGN_OR_RETURN(const auto computation_dtype,
                      ConvertTo<mlir::ElementType>(computation_scalar_type));

  auto op_builder = [assume_unique,
                     invert](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
    auto& [elements, test_elements] = inputs;
    return BuildIsInShlo(elements, test_elements, assume_unique, invert);
  };

  absl::Span<const int64_t> output_dims = elements.sizes();
  mlir::ElementType output_dtype =
      mlir::ElementType::PRED;  // The output is always Boolean.

  return DispatchOp<2>(std::move(op_builder), {elements, test_elements},
                       {.out_dtype = output_dtype,
                        .out_dims = output_dims,
                        .computation_dtype = computation_dtype,
                        .op_param_cache_keys = std::move(param_keys)});
}

absl::StatusOr<DeviceBufferRef> AtenIsInHelper(const at::Tensor& elements,
                                               const at::Scalar& test_element,
                                               bool assume_unique, bool invert,
                                               OpParamCacheKeys& param_keys) {
  ABSL_VLOG(1) << "AtenIsInHelper assume_unique=" << assume_unique
               << " invert=" << invert;

  at::ScalarType computation_scalar_type =
      at::promote_types(elements.scalar_type(), test_element.type());
  TT_ASSIGN_OR_RETURN(const auto computation_dtype,
                      ConvertTo<mlir::ElementType>(computation_scalar_type));

  auto op_builder = [assume_unique, invert, test_element](
                        mlir::MlirOp elements) -> absl::StatusOr<mlir::MlirOp> {
    mlir::MlirBuilder& builder = elements.getBuilder();
    TT_ASSIGN_OR_RETURN(mlir::MlirOp test_element,
                        MakeConstant(builder, test_element));
    test_element = mlir::stablehlo::Reshape(test_element, {1});
    return BuildIsInShlo(elements, test_element, assume_unique, invert);
  };

  absl::Span<const int64_t> output_dims = elements.sizes();
  mlir::ElementType output_dtype =
      mlir::ElementType::PRED;  // The output is always Boolean.

  return DispatchOp<1>(std::move(op_builder), elements,
                       {.out_dtype = output_dtype,
                        .out_dims = output_dims,
                        .computation_dtype = computation_dtype,
                        .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor& AtenIsInScalarTensorOut(const at::Scalar& element,
                                    const at::Tensor& test_elements,
                                    bool assume_unique, bool invert,
                                    at::Tensor& out) {
  TT_KERNEL(OpName::kIsInScalarTensorOut, _,
            (IgnoreInCacheKey(element, "Legacy usage"), test_elements,
             IgnoreInCacheKey(assume_unique, "Legacy usage"),
             IgnoreInCacheKey(invert, "Legacy usage"), out),
            {
              at::Tensor wrapper_scalar_tensor =
                  at::full({}, element).to(test_elements.device());

              // Resize the output tensor to the shape we expect it to be, an
              // empty tensor for hosting a scalar.
              Dimensions out_dims = {};
              at::native::resize_output(out, out_dims);

              return AtenIsInTensorTensorOut(wrapper_scalar_tensor,
                                             test_elements, assume_unique,
                                             invert, out);
            });
}

at::Tensor& AtenIsInTensorTensorOut(const at::Tensor& elements,
                                    const at::Tensor& test_elements,
                                    bool assume_unique, bool invert,
                                    at::Tensor& out) {
  TT_KERNEL(
      OpName::kIsInTensorTensorOut, param_keys,
      (elements, test_elements, assume_unique, invert, out), {
        TT_ASSIGN_OR_THROW(auto result_buf,
                           AtenIsInHelper(elements, test_elements,
                                          assume_unique, invert, param_keys));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

at::Tensor& AtenIsInTensorScalarOut(const at::Tensor& elements,
                                    const at::Scalar& test_element,
                                    bool assume_unique, bool invert,
                                    at::Tensor& out) {
  TT_KERNEL(
      OpName::kIsInTensorScalarOut, param_keys,
      (elements, test_element, assume_unique, invert, out), {
        TT_ASSIGN_OR_THROW(auto result_buf,
                           AtenIsInHelper(elements, test_element, assume_unique,
                                          invert, param_keys));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
        return out;
      });
}

}  // namespace torch_tpu
