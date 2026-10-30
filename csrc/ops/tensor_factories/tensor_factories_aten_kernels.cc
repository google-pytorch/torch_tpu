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

#include "csrc/ops/tensor_factories/tensor_factories_aten_kernels.h"

#include <optional>
#include <string_view>
#include <utility>

#include "ATen/ExpandUtils.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/full_like_native.h"
#include "ATen/ops/new_full_native.h"
#include "ATen/ops/new_ones_native.h"
#include "ATen/ops/new_zeros_native.h"
#include "ATen/ops/ones_like_native.h"
#include "ATen/ops/zeros_like_native.h"
#include "absl/status/statusor.h"
#include "c10/core/DefaultDtype.h"
#include "c10/core/Device.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorImpl.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/nullary_aten_kernels.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

void ValidateFactoryOptions(at::ScalarType out_dtype,
                            std::optional<at::Layout> layout_opt,
                            std::optional<at::Device> device_opt,
                            std::optional<bool> pin_memory_opt,
                            std::string_view op_name_str) {
  TT_CHECK_THROW(out_dtype != at::ScalarType::ComplexHalf,
                 error::kPythonNotImplementedError)
      << "TorchTPU does not yet support dtype complex32";
  CheckDeviceIsTpu(device_opt, op_name_str);
  const at::Layout layout = layout_opt.value_or(at::Layout::Strided);
  TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=PyTorch catches other layouts first.
      layout == at::Layout::Strided, error::kPythonNotImplementedError)
      << "expected layout to be torch.strided (other layouts are not yet "
         "supported), got "
      << layout;
  TT_CHECK_THROW(!pin_memory_opt.value_or(false), error::kInvalidArgument)
      << "only dense CPU tensors can be pinned";
}

void ValidateLikeMemoryFormat(const at::Tensor& self,
                              at::MemoryFormat memory_format) {
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    TT_CHECK_THROW(self.dim() == 4, error::kInvalidArgument)
        << "expected a 4D tensor for channels_last format, got " << self.dim()
        << "D";
  } else if (memory_format == at::MemoryFormat::ChannelsLast3d) {
    TT_CHECK_THROW(self.dim() == 5, error::kInvalidArgument)
        << "expected a 5D tensor for channels_last_3d format, got "
        << self.dim() << "D";
  } else {
    TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=all valid MemoryFormat enums are
                     // handled above.
        memory_format == at::MemoryFormat::Preserve ||
            memory_format == at::MemoryFormat::Contiguous,
        error::kInvalidArgument)
        << "expected memory_format to be preserve, contiguous, channels_last, "
           "or channels_last_3d, got "
        << memory_format;
  }
}

void ApplyLikeTensorRestride(at::Tensor& result, const at::Tensor& self,
                             at::MemoryFormat memory_format) {
  if (memory_format == at::MemoryFormat::Preserve) {
    if (self.is_non_overlapping_and_dense()) {
      result.unsafeGetTensorImpl()->set_sizes_and_strides(
          self.sizes(), self.strides(), /*storage_offset=*/0);
    } else {
      result.unsafeGetTensorImpl()->set_sizes_and_strides(
          self.sizes(), at::infer_dense_strides(self.sizes(), self.strides()),
          /*storage_offset=*/0);
    }
  } else if (memory_format != at::MemoryFormat::Contiguous) {
    result.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  }
}

at::ScalarType ResolveFullOutputDtype(
    std::optional<at::ScalarType> dtype_opt,
    const PromotedScalar& promoted_fill_value) {
  if (dtype_opt.has_value()) {
    return *dtype_opt;
  }
  const c10::ScalarType fill_type = promoted_fill_value.scalar().type();
  if (fill_type == c10::ScalarType::Bool) {
    return at::ScalarType::Bool;
  }
  if (c10::isIntegralType(fill_type, /*includeBool=*/false)) {
    return at::ScalarType::Long;
  }
  if (c10::isComplexType(fill_type)) {
    return (c10::get_default_dtype() == c10::ScalarType::Double)
               ? at::ScalarType::ComplexDouble
               : at::ScalarType::ComplexFloat;
  }
  return c10::get_default_dtype_as_scalartype();
}

at::Tensor CreateConstantFilledTensor(at::IntArrayRef size, double scalar_val,
                                      at::ScalarType out_dtype,
                                      std::optional<at::Layout> layout_opt,
                                      std::optional<at::Device> device_opt,
                                      std::optional<bool> pin_memory_opt,
                                      std::string_view op_name_str) {
  ValidateFactoryOptions(out_dtype, layout_opt, device_opt, pin_memory_opt,
                         op_name_str);
  auto op_builder =
      [size_vec = CopyIntVector(size), out_dtype,
       scalar_val](mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
    TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=all dtypes are supported.
        const mlir::ElementType mlir_dtype,
        ConvertTo<mlir::ElementType>(out_dtype));
    auto type =
        mlir::makeTensorType(builder.getContext(), size_vec, mlir_dtype);
    return MakeConstant(builder, scalar_val, type);
  };
  return ApplyNullaryOp(std::move(op_builder), out_dtype, size,
                        OpParamCacheKeys::Empty());
}

absl::StatusOr<at::Tensor> CreateScalarFilledTensor(
    at::IntArrayRef size, PromotedScalar& promoted_fill_value,
    at::ScalarType out_dtype, std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt, std::optional<bool> pin_memory_opt,
    std::string_view op_name_str, OpParamCacheKeys param_keys) {
  ValidateFactoryOptions(out_dtype, layout_opt, device_opt, pin_memory_opt,
                         op_name_str);
  TT_RET_CHECK(out_dtype != at::ScalarType::ComplexDouble,
               error::kPythonNotImplementedError)
      << ToString(at::ScalarType::ComplexDouble) << " dtype is not supported";
  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=all dtypes are supported.
      const mlir::ElementType out_mlir_element_type,
      ConvertTo<mlir::ElementType>(out_dtype));
  TT_RETURN_IF_ERROR(  // ERROR_COV_INFEASIBLE=errors inside are covered.
      ValidateTensorByteSize(size, out_mlir_element_type));
  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=errors inside are covered.
      at::Tensor fill_value_tensor, promoted_fill_value.GetTensor(out_dtype));

  auto op_builder =
      [sizes = CopyIntVector(size), out_mlir_element_type](
          mlir::MlirOp fill_value_op) -> absl::StatusOr<mlir::MlirOp> {
    TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=errors inside are covered.
        auto result_op, CastIfNeeded(fill_value_op, out_mlir_element_type));
    TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=errors inside are covered.
        result_op, BroadcastIfNeeded(result_op, sizes));
    return result_op;
  };
  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=errors inside are covered.
      DeviceBufferRef result_buf,
      DispatchOp<1>(std::move(op_builder), /*inputs=*/{fill_value_tensor},
                    {.out_dtype = out_mlir_element_type,
                     .out_dims = size,
                     .op_param_cache_keys = std::move(param_keys)}));
  return MakeTensor(std::move(result_buf));
}

}  // namespace

at::Tensor AtenZeros(at::IntArrayRef size,
                     std::optional<at::ScalarType> dtype_opt,
                     std::optional<at::Layout> layout_opt,
                     std::optional<at::Device> device_opt,
                     std::optional<bool> pin_memory_opt) {
  TT_KERNEL(OpName::kZeros, _,
            (IgnoreInCacheKey(size, "Output shape is in GraphSignature"),
             IgnoreInCacheKey(dtype_opt, "Output dtype is in GraphSignature"),
             IgnoreInCacheKey(layout_opt, "Doesn't affect SHLO"),
             IgnoreInCacheKey(device_opt, "Doesn't affect SHLO"),
             IgnoreInCacheKey(pin_memory_opt, "Doesn't affect SHLO")),
            {
              const at::ScalarType out_dtype =
                  dtype_opt.value_or(c10::get_default_dtype_as_scalartype());
              return CreateConstantFilledTensor(size, 0.0, out_dtype,
                                                layout_opt, device_opt,
                                                pin_memory_opt, "zeros");
            });
}

at::Tensor AtenOnes(at::IntArrayRef size,
                    std::optional<at::ScalarType> dtype_opt,
                    std::optional<at::Layout> layout_opt,
                    std::optional<at::Device> device_opt,
                    std::optional<bool> pin_memory_opt) {
  TT_KERNEL(OpName::kOnes, _,
            (IgnoreInCacheKey(size, "Output shape is in GraphSignature"),
             IgnoreInCacheKey(dtype_opt, "Output dtype is in GraphSignature"),
             IgnoreInCacheKey(layout_opt, "Doesn't affect SHLO"),
             IgnoreInCacheKey(device_opt, "Doesn't affect SHLO"),
             IgnoreInCacheKey(pin_memory_opt, "Doesn't affect SHLO")),
            {
              const at::ScalarType out_dtype =
                  dtype_opt.value_or(c10::get_default_dtype_as_scalartype());
              return CreateConstantFilledTensor(size, 1.0, out_dtype,
                                                layout_opt, device_opt,
                                                pin_memory_opt, "ones");
            });
}

at::Tensor AtenFull(at::IntArrayRef size, const at::Scalar& fill_value,
                    std::optional<at::ScalarType> dtype_opt,
                    std::optional<at::Layout> layout_opt,
                    std::optional<at::Device> device_opt,
                    std::optional<bool> pin_memory_opt) {
  auto promoted_fill_value = PromoteScalar(fill_value);
  TT_KERNEL(
      OpName::kFull, param_keys,
      (IgnoreInCacheKey(size, "Output shape is in GraphSignature"),
       promoted_fill_value,
       IgnoreInCacheKey(dtype_opt, "Output dtype is in GraphSignature"),
       IgnoreInCacheKey(layout_opt, "Doesn't affect SHLO"),
       IgnoreInCacheKey(device_opt, "Doesn't affect SHLO"),
       IgnoreInCacheKey(pin_memory_opt, "Doesn't affect SHLO")),
      {
        const at::ScalarType out_dtype =
            ResolveFullOutputDtype(dtype_opt, promoted_fill_value);
        TT_ASSIGN_OR_THROW(  // ERROR_COV_INFEASIBLE=errors inside are covered.
            at::Tensor result,
            CreateScalarFilledTensor(size, promoted_fill_value, out_dtype,
                                     layout_opt, device_opt, pin_memory_opt,
                                     "full", std::move(param_keys)));
        return result;
      });
}

at::Tensor AtenZerosLike(const at::Tensor& self,
                         std::optional<at::ScalarType> dtype_opt,
                         std::optional<at::Layout> layout_opt,
                         std::optional<at::Device> device_opt,
                         std::optional<bool> pin_memory_opt,
                         std::optional<at::MemoryFormat> memory_format_opt) {
  const at::Device target_device = device_opt.value_or(self.device());
  const at::Layout target_layout = layout_opt.value_or(self.layout());
  if (target_device.type() != GetPrivateUse1DeviceType() ||
      target_layout != at::Layout::Strided) {
    return at::native::zeros_like(self, dtype_opt, layout_opt, device_opt,
                                  pin_memory_opt, memory_format_opt);
  }
  TT_KERNEL(
      OpName::kZerosLike, _,
      (self, IgnoreInCacheKey(dtype_opt, "Output dtype is in GraphSignature"),
       IgnoreInCacheKey(layout_opt, "Doesn't affect SHLO"),
       IgnoreInCacheKey(device_opt, "Doesn't affect SHLO"),
       IgnoreInCacheKey(pin_memory_opt, "Doesn't affect SHLO"),
       IgnoreInCacheKey(memory_format_opt, "Doesn't affect SHLO")),
      {
        const at::ScalarType out_dtype = dtype_opt.value_or(self.scalar_type());
        const at::MemoryFormat memory_format =
            memory_format_opt.value_or(at::MemoryFormat::Preserve);
        ValidateLikeMemoryFormat(self, memory_format);
        at::Tensor result = CreateConstantFilledTensor(
            self.sizes(), 0.0, out_dtype, target_layout, target_device,
            pin_memory_opt, "zeros_like");
        ApplyLikeTensorRestride(result, self, memory_format);
        return result;
      });
}

at::Tensor AtenOnesLike(const at::Tensor& self,
                        std::optional<at::ScalarType> dtype_opt,
                        std::optional<at::Layout> layout_opt,
                        std::optional<at::Device> device_opt,
                        std::optional<bool> pin_memory_opt,
                        std::optional<at::MemoryFormat> memory_format_opt) {
  const at::Device target_device = device_opt.value_or(self.device());
  const at::Layout target_layout = layout_opt.value_or(self.layout());
  if (target_device.type() != GetPrivateUse1DeviceType() ||
      target_layout != at::Layout::Strided) {
    return at::native::ones_like(self, dtype_opt, layout_opt, device_opt,
                                 pin_memory_opt, memory_format_opt);
  }
  TT_KERNEL(
      OpName::kOnesLike, _,
      (self, IgnoreInCacheKey(dtype_opt, "Output dtype is in GraphSignature"),
       IgnoreInCacheKey(layout_opt, "Doesn't affect SHLO"),
       IgnoreInCacheKey(device_opt, "Doesn't affect SHLO"),
       IgnoreInCacheKey(pin_memory_opt, "Doesn't affect SHLO"),
       IgnoreInCacheKey(memory_format_opt, "Doesn't affect SHLO")),
      {
        const at::ScalarType out_dtype = dtype_opt.value_or(self.scalar_type());
        const at::MemoryFormat memory_format =
            memory_format_opt.value_or(at::MemoryFormat::Preserve);
        ValidateLikeMemoryFormat(self, memory_format);
        at::Tensor result = CreateConstantFilledTensor(
            self.sizes(), 1.0, out_dtype, target_layout, target_device,
            pin_memory_opt, "ones_like");
        ApplyLikeTensorRestride(result, self, memory_format);
        return result;
      });
}

at::Tensor AtenFullLike(const at::Tensor& self, const at::Scalar& fill_value,
                        std::optional<at::ScalarType> dtype_opt,
                        std::optional<at::Layout> layout_opt,
                        std::optional<at::Device> device_opt,
                        std::optional<bool> pin_memory_opt,
                        std::optional<at::MemoryFormat> memory_format_opt) {
  const at::Device target_device = device_opt.value_or(self.device());
  const at::Layout target_layout = layout_opt.value_or(self.layout());
  if (target_device.type() != GetPrivateUse1DeviceType() ||
      target_layout != at::Layout::Strided) {
    return at::native::full_like(self, fill_value, dtype_opt, layout_opt,
                                 device_opt, pin_memory_opt, memory_format_opt);
  }
  auto promoted_fill_value = PromoteScalar(fill_value);
  TT_KERNEL(
      OpName::kFullLike, param_keys,
      (self, promoted_fill_value,
       IgnoreInCacheKey(dtype_opt, "Output dtype is in GraphSignature"),
       IgnoreInCacheKey(layout_opt, "Doesn't affect SHLO"),
       IgnoreInCacheKey(device_opt, "Doesn't affect SHLO"),
       IgnoreInCacheKey(pin_memory_opt, "Doesn't affect SHLO"),
       IgnoreInCacheKey(memory_format_opt, "Doesn't affect SHLO")),
      {
        const at::ScalarType out_dtype = dtype_opt.value_or(self.scalar_type());
        const at::MemoryFormat memory_format =
            memory_format_opt.value_or(at::MemoryFormat::Preserve);
        ValidateLikeMemoryFormat(self, memory_format);
        TT_ASSIGN_OR_THROW(  // ERROR_COV_INFEASIBLE=errors inside are covered.
            at::Tensor result, CreateScalarFilledTensor(
                                   self.sizes(), promoted_fill_value, out_dtype,
                                   target_layout, target_device, pin_memory_opt,
                                   "full_like", std::move(param_keys)));
        ApplyLikeTensorRestride(result, self, memory_format);
        return result;
      });
}

at::Tensor AtenNewZeros(const at::Tensor& self, at::IntArrayRef size,
                        std::optional<at::ScalarType> dtype_opt,
                        std::optional<at::Layout> layout_opt,
                        std::optional<at::Device> device_opt,
                        std::optional<bool> pin_memory_opt) {
  const at::Device target_device = device_opt.value_or(self.device());
  const at::Layout target_layout = layout_opt.value_or(self.layout());
  if (target_device.type() != GetPrivateUse1DeviceType() ||
      target_layout != at::Layout::Strided) {
    return at::native::new_zeros(self, size, dtype_opt, layout_opt, device_opt,
                                 pin_memory_opt);
  }
  TT_KERNEL(OpName::kNewZeros, _,
            (self, IgnoreInCacheKey(size, "Output shape is in GraphSignature"),
             IgnoreInCacheKey(dtype_opt, "Output dtype is in GraphSignature"),
             IgnoreInCacheKey(layout_opt, "Doesn't affect SHLO"),
             IgnoreInCacheKey(device_opt, "Doesn't affect SHLO"),
             IgnoreInCacheKey(pin_memory_opt, "Doesn't affect SHLO")),
            {
              const at::ScalarType out_dtype =
                  dtype_opt.value_or(self.scalar_type());
              return CreateConstantFilledTensor(size, 0.0, out_dtype,
                                                target_layout, target_device,
                                                pin_memory_opt, "new_zeros");
            });
}

at::Tensor AtenNewOnes(const at::Tensor& self, at::IntArrayRef size,
                       std::optional<at::ScalarType> dtype_opt,
                       std::optional<at::Layout> layout_opt,
                       std::optional<at::Device> device_opt,
                       std::optional<bool> pin_memory_opt) {
  const at::Device target_device = device_opt.value_or(self.device());
  const at::Layout target_layout = layout_opt.value_or(self.layout());
  if (target_device.type() != GetPrivateUse1DeviceType() ||
      target_layout != at::Layout::Strided) {
    return at::native::new_ones(self, size, dtype_opt, layout_opt, device_opt,
                                pin_memory_opt);
  }
  TT_KERNEL(OpName::kNewOnes, _,
            (self, IgnoreInCacheKey(size, "Output shape is in GraphSignature"),
             IgnoreInCacheKey(dtype_opt, "Output dtype is in GraphSignature"),
             IgnoreInCacheKey(layout_opt, "Doesn't affect SHLO"),
             IgnoreInCacheKey(device_opt, "Doesn't affect SHLO"),
             IgnoreInCacheKey(pin_memory_opt, "Doesn't affect SHLO")),
            {
              const at::ScalarType out_dtype =
                  dtype_opt.value_or(self.scalar_type());
              return CreateConstantFilledTensor(size, 1.0, out_dtype,
                                                target_layout, target_device,
                                                pin_memory_opt, "new_ones");
            });
}

at::Tensor AtenNewFull(const at::Tensor& self, at::IntArrayRef size,
                       const at::Scalar& fill_value,
                       std::optional<at::ScalarType> dtype_opt,
                       std::optional<at::Layout> layout_opt,
                       std::optional<at::Device> device_opt,
                       std::optional<bool> pin_memory_opt) {
  const at::Device target_device = device_opt.value_or(self.device());
  const at::Layout target_layout = layout_opt.value_or(self.layout());
  if (target_device.type() != GetPrivateUse1DeviceType() ||
      target_layout != at::Layout::Strided) {
    return at::native::new_full(self, size, fill_value, dtype_opt, layout_opt,
                                device_opt, pin_memory_opt);
  }
  auto promoted_fill_value = PromoteScalar(fill_value);
  TT_KERNEL(
      OpName::kNewFull, param_keys,
      (self, IgnoreInCacheKey(size, "Output shape is in GraphSignature"),
       promoted_fill_value,
       IgnoreInCacheKey(dtype_opt, "Output dtype is in GraphSignature"),
       IgnoreInCacheKey(layout_opt, "Doesn't affect SHLO"),
       IgnoreInCacheKey(device_opt, "Doesn't affect SHLO"),
       IgnoreInCacheKey(pin_memory_opt, "Doesn't affect SHLO")),
      {
        const at::ScalarType out_dtype = dtype_opt.value_or(self.scalar_type());
        TT_ASSIGN_OR_THROW(  // ERROR_COV_INFEASIBLE=errors inside are covered.
            at::Tensor result, CreateScalarFilledTensor(
                                   size, promoted_fill_value, out_dtype,
                                   target_layout, target_device, pin_memory_opt,
                                   "new_full", std::move(param_keys)));
        return result;
      });
}

}  // namespace torch_tpu
