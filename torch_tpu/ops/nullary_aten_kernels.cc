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

#include "torch_tpu/ops/nullary_aten_kernels.h"

#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/native/Resize.h"
#include "absl/log/absl_log.h"
#include "absl/log/absl_vlog_is_on.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "c10/core/DefaultDtype.h"
#include "c10/core/Device.h"
#include "c10/core/Layout.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/core/TensorImpl.h"
#include "c10/util/Optional.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/device_types.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/as_strided/as_strided_aten_kernels.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

namespace {

void LogKernelStart(std::string_view op_name,
                    const OpParamCacheKeys& op_param_cache_keys) {
  if (ABSL_VLOG_IS_ON(1)) {
    ABSL_VLOG(1) << "[C++ KERNEL " << op_name << "]"
                 << absl::StrJoin(op_param_cache_keys, ",",
                                  absl::PairFormatter(": "));
  }
}

std::string LayoutToString(c10::optional<at::Layout> layout_opt) {
  if (layout_opt == at::Layout::Jagged) {
    return "torch.jagged";
  }
  if (layout_opt.has_value()) {
    std::stringstream ss;
    ss << layout_opt.value();
    return ss.str();
  }
  return "nullopt";  // Should not be reached.
}

}  // namespace

at::Tensor ApplyNullaryOp(MlirNullaryOpBuilder op_builder,
                          c10::ScalarType out_dtype, at::IntArrayRef out_dims,
                          OpParamCacheKeys op_param_cache_keys,
                          OpSplitMode split_mode) {
  TT_ASSIGN_OR_THROW(  // ERROR_COV_INFEASIBLE=all dtypes are supported.
      mlir::ElementType out_mlir_element_type,
      ConvertTo<mlir::ElementType>(out_dtype));
  TT_THROW_IF_ERROR(  // ERROR_COV_INFEASIBLE=errors inside are covered.
      ValidateTensorByteSize(out_dims, out_mlir_element_type));
  TT_ASSIGN_OR_THROW(  // ERROR_COV_INFEASIBLE=errors should be covered inside.
      auto result_buf,
      DispatchOp<0>(std::move(op_builder), /*inputs=*/{},
                    {.out_dtype = out_mlir_element_type,
                     .out_dims = out_dims,
                     .op_param_cache_keys = std::move(op_param_cache_keys),
                     .split_mode = split_mode}));
  return MakeTensor(std::move(result_buf));
}

absl::Status ApplyNullaryOpOut(at::Tensor& out, MlirNullaryOpBuilder op_builder,
                               c10::ScalarType out_dtype,
                               at::IntArrayRef out_dims,
                               OpParamCacheKeys op_param_cache_keys,
                               OpSplitMode split_mode) {
  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=all dtypes are supported.
      mlir::ElementType out_mlir_element_type,
      ConvertTo<mlir::ElementType>(out_dtype));
  TT_RETURN_IF_ERROR(  // ERROR_COV_INFEASIBLE=errors inside are covered.
      ValidateTensorByteSize(out_dims, out_mlir_element_type));
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch catches this first.
      out.device().type() == GetPrivateUse1DeviceType(),
      error::kFailedPrecondition)
      << "the out tensor must be on TPU";

  at::native::resize_output(out, out_dims);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch catches this first.
      out.scalar_type() == out_dtype, error::kInvalidArgument)
      << "expected the out tensor dtype to be " << out_dtype << ", got "
      << out.scalar_type();

  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=errors should be covered inside.
      auto result_buf,
      DispatchOp<0>(std::move(op_builder), /*inputs=*/{},
                    {.out_dtype = out_mlir_element_type,
                     .out_dims = out_dims,
                     .op_param_cache_keys = std::move(op_param_cache_keys),
                     .split_mode = split_mode}));
  return AssignBufferToAtTensor(std::move(result_buf), out);
}

absl::StatusOr<at::Tensor> MakeEmptyMemoryFormat(
    at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<at::MemoryFormat> memory_format_opt) {
  TT_RET_CHECK(!dtype_opt.has_value() ||
                   dtype_opt.value() != at::ScalarType::ComplexHalf,
               error::kUnimplemented)
      << "TorchTPU does not yet support dtype complex32";
  // Check that we support all the provided options.
  CheckDeviceIsTpu(device_opt, ToString(OpName::kEmpty));
  TT_RET_CHECK(layout_opt.value_or(at::Layout::Strided) == at::Layout::Strided,
               error::kUnimplemented)
      << "only layout=torch.strided is supported by TorchTPU for now, "
         "got "
      << LayoutToString(layout_opt);

  // If device_opt is unspecified, we use the global default dtype.
  TT_ASSIGN_OR_RETURN(const auto dtype,
                      ConvertTo<mlir::ElementType>(dtype_opt.value_or(
                          c10::get_default_dtype_as_scalartype())));

  // Create an empty DeviceBufferRef.
  // Note: because XLA doesn't allow for easy construction of
  // uninitialized memory, we follow the behavior of
  // torch.utils.deterministic.fill_uninitialized_memory (even if this
  // flag is False), which is to fill with NaNs or max-value integers. If
  // the empty buffer is overwritten (such as by out=torch.empty(...) or
  // torch.empty(...).fill_(...)), then this DeferredOp node will be
  // dropped and not used.
  TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer, DeviceBufferList::CreateEmpty(
                                                  CopyIntVector(size), dtype));
  at::Tensor result = MakeTensor(std::move(buffer));

  // MakeTensor defaults to a contiguous tensor, and so
  // does aten::empty().
  if (memory_format_opt.value_or(at::MemoryFormat::Contiguous) !=
      at::MemoryFormat::Contiguous) {
    // This may throw an error if the memory format is not supported for
    // the given tensor shape.
    result.unsafeGetTensorImpl()->empty_tensor_restride(
        memory_format_opt.value());
  }
  return result;
}

absl::StatusOr<at::Tensor> MakeEmptyTensor(
    at::IntArrayRef size, c10::ScalarType dtype,
    c10::optional<at::Device> device_opt) {
  return MakeEmptyMemoryFormat(size, dtype, /*layout_opt=*/c10::nullopt,
                               device_opt, /*pin_memory_opt=*/c10::nullopt,
                               /*memory_format_opt=*/c10::nullopt);
}

at::Tensor AtenEmptyMemoryFormat(
    at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<at::MemoryFormat> memory_format_opt) {
  TT_KERNEL(OpName::kEmpty, _,
            (IgnoreInCacheKey(size, "doesn't generate SHLO"),
             IgnoreInCacheKey(dtype_opt, "doesn't generate SHLO"),
             IgnoreInCacheKey(layout_opt, "doesn't generate SHLO"),
             IgnoreInCacheKey(device_opt, "doesn't generate SHLO"),
             IgnoreInCacheKey(pin_memory_opt, "doesn't generate SHLO"),
             IgnoreInCacheKey(memory_format_opt, "doesn't generate SHLO")),
            {
              TT_ASSIGN_OR_THROW(
                  auto result,
                  MakeEmptyMemoryFormat(size, dtype_opt, layout_opt, device_opt,
                                        pin_memory_opt, memory_format_opt));
              return result;
            });
}

at::Tensor AtenEmptyStrided(c10::SymIntArrayRef size_sym,
                            c10::SymIntArrayRef stride_sym,
                            c10::optional<at::ScalarType> dtype_opt,
                            c10::optional<at::Layout> layout_opt,
                            c10::optional<at::Device> device_opt,
                            c10::optional<bool> pin_memory_opt) {
  // Resolve the symbolic shapes and strides before TT_KERNEL for logging.
  Dimensions final_sizes_vec;
  final_sizes_vec.reserve(size_sym.size());
  for (const auto& s : size_sym)
    final_sizes_vec.push_back(s.guard_int(__FILE__, __LINE__));
  Strides final_strides_vec;
  final_strides_vec.reserve(stride_sym.size());
  for (const auto& s : stride_sym)
    final_strides_vec.push_back(s.guard_int(__FILE__, __LINE__));
  TT_KERNEL(
      OpName::kEmptyStrided, _,
      (IgnoreInCacheKey(final_sizes_vec, "Legacy usage"),
       IgnoreInCacheKey(final_strides_vec, "Legacy usage"),
       IgnoreInCacheKey(dtype_opt, "Legacy usage"),
       IgnoreInCacheKey(layout_opt, "Legacy usage"),
       IgnoreInCacheKey(device_opt, "Legacy usage"),
       IgnoreInCacheKey(pin_memory_opt, "Legacy usage")),
      {
        TT_CHECK_THROW(!dtype_opt.has_value() ||
                           dtype_opt.value() != at::ScalarType::ComplexHalf,
                       error::kUnimplemented)
            << "TorchTPU does not yet support dtype complex32";
        TT_CHECK_THROW(final_sizes_vec.size() == final_strides_vec.size(),
                       error::kInvalidArgument)
            << "the dimensionality of sizes must be the same as "
            << "strides, got size [" << final_sizes_vec.size()
            << "] and stride [" << final_strides_vec.size() << "]";

        // Determine how much "storage" we require, in terms of the number of
        // elements.
        int64_t max_index = 0;
        bool is_zero_sized = false;
        for (int64_t i = 0; i < final_sizes_vec.size(); ++i) {
          TT_CHECK_THROW(final_sizes_vec[i] >= 0, error::kInvalidArgument)
              << "size must be nonnegative, got sizes ["
              << absl::StrJoin(final_sizes_vec, ", ") << "]";
          TT_CHECK_THROW(final_strides_vec[i] >= 0, error::kInvalidArgument)
              << "stride must be nonnegative, got strides ["
              << absl::StrJoin(final_strides_vec, ", ") << "]";
          ;
          is_zero_sized |= final_sizes_vec[i] == 0;
          max_index += (final_sizes_vec[i] - 1) * final_strides_vec[i];
        }
        const int64_t element_count = is_zero_sized ? 0 : max_index + 1;

        // Create a 1D contiguous empty (really, deterministically filled)
        // tensor to act as this "storage".
        at::Tensor result = AtenEmptyMemoryFormat(
            {element_count}, dtype_opt, layout_opt, device_opt, pin_memory_opt,
            at::MemoryFormat::Contiguous);

        // Then apply as_strided to give this tensor the correct strided layout.
        c10::SymIntArrayRef final_sizes_sym =
            c10::fromIntArrayRefKnownNonNegative(final_sizes_vec);
        c10::SymIntArrayRef final_strides_sym =
            c10::fromIntArrayRefKnownNonNegative(final_strides_vec);
        return AtenAsStrided(result, final_sizes_sym, final_strides_sym, 0);
      });
}

at::Tensor AtenEfficientZeroTensor(at::IntArrayRef size,
                                   c10::optional<at::ScalarType> dtype_opt,
                                   c10::optional<at::Layout> layout_opt,
                                   c10::optional<at::Device> device_opt,
                                   c10::optional<bool> pin_memory_opt) {
  // Note: _efficientzerotensor is an optimization to avoid materializing
  // a zero tensor. On TPU, we defer the materialization by using a constant
  // in the HLO graph, allowing XLA to optimize it away if possible.
  TT_KERNEL(
      OpName::kEfficientZeroTensor, _,
      (IgnoreInCacheKey(size, "Not needed"),
       IgnoreInCacheKey(dtype_opt, "Not needed"),
       IgnoreInCacheKey(layout_opt, "Not needed"),
       IgnoreInCacheKey(device_opt, "Not needed"),
       IgnoreInCacheKey(pin_memory_opt, "Not needed")),
      {
        c10::ScalarType dtype =
            dtype_opt.value_or(c10::get_default_dtype_as_scalartype());
        auto op_builder =
            [size_vec = CopyIntVector(size), dtype](
                mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
          TT_ASSIGN_OR_RETURN(const mlir::ElementType mlir_dtype,
                              ConvertTo<mlir::ElementType>(dtype));
          auto type =
              mlir::makeTensorType(builder.getContext(), size_vec, mlir_dtype);
          return MakeConstant(builder, 0.0, type);
        };

        return ApplyNullaryOp(std::move(op_builder), dtype, size,
                              OpParamCacheKeys::Empty());
      });
}

}  // namespace torch_tpu
