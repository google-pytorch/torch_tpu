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

#include "csrc/ops/sparse_coo/sparse_coo_aten_kernels.h"

#include <array>
#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/SparseTensorImpl.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/native/SparseTensorUtils.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/result_type.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/DefaultDtype.h"
#include "c10/core/Device.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/DispatchKeySet.h"
#include "c10/core/Scalar.h"
#include "c10/core/ScalarType.h"
#include "c10/core/ScalarTypeToTypeMeta.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/ArrayRef.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/typeid.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/copy_from/copy_common.h"
#include "csrc/ops/copy_from/copy_from_aten_kernels.h"
#include "csrc/ops/copy_from/tpu_to_cpu.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "csrc/ops/sparse_coo/sparse_coo.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

at::Tensor ShallowCopyDetach(const at::Tensor& t) {
  return at::Tensor(t.unsafeGetTensorImpl()->shallow_copy_and_detach(
      t.unsafeGetTensorImpl()->version_counter(),
      /*allow_tensor_metadata_change=*/true));
}

c10::DispatchKey GetSparseDispatchKey(const at::Device& device) {
  return device.is_cpu() ? c10::DispatchKey::SparseCPU
                         : c10::DispatchKey::SparsePrivateUse1;
}

void SetSparseCooTensorInPlace(const at::Tensor& out, const int64_t sparse_dim,
                               const int64_t dense_dim,
                               const at::IntArrayRef sizes,
                               const at::Tensor& indices,
                               const at::Tensor& values,
                               const bool is_coalesced) {
  auto* impl = at::sparse::get_sparse_impl(out);
  impl->raw_resize_(sparse_dim, dense_dim, sizes);
  impl->set_indices_and_values_unsafe(ShallowCopyDetach(indices),
                                      ShallowCopyDetach(values));
  impl->set_coalesced(is_coalesced);
}

Dimensions GetSparseValuesShape(const int64_t nnz, const int64_t sparse_dim,
                                const int64_t dense_dim,
                                const at::IntArrayRef sizes) {
  Dimensions values_shape;
  values_shape.reserve(1 + dense_dim);
  values_shape.push_back(nnz);
  for (int64_t d = 0; d < dense_dim; ++d) {
    values_shape.push_back(sizes[sparse_dim + d]);
  }
  return values_shape;
}

absl::StatusOr<at::Tensor> CopyIndicesTensor(const at::Tensor& src_indices,
                                             const at::Device& target_device,
                                             const bool non_blocking) {
  if (target_device.is_cpu()) {
    at::Tensor dst =
        at::empty(src_indices.sizes(), src_indices.options().device(c10::kCPU));
    CopyTensor(src_indices, dst, non_blocking);
    return dst;
  }
  TT_ASSIGN_OR_RETURN(at::Tensor dst,
                      MakeEmptyTensor(src_indices.sizes(),
                                      c10::ScalarType::Long, target_device));
  CopyTensor(src_indices, dst, non_blocking);
  return dst;
}

absl::StatusOr<at::Tensor> CopyOrCastValuesTensor(
    const at::Tensor& src_values, const at::ScalarType target_dtype,
    const at::Device& target_device, const bool non_blocking) {
  if (target_device.is_cpu()) {
    at::Tensor dst =
        at::empty(src_values.sizes(),
                  src_values.options().device(c10::kCPU).dtype(target_dtype));
    CopyTensor(src_values, dst, non_blocking);
    return dst;
  }
  if (IsPrivateUse1Device(src_values) &&
      src_values.scalar_type() != target_dtype && src_values.numel() > 0) {
    TT_ASSIGN_OR_RETURN(const auto out_dtype,
                        ConvertTo<mlir::ElementType>(target_dtype));
    TT_ASSIGN_OR_RETURN(auto pk, TT_MAKE_OP_PARAM_CACHE_KEYS(target_dtype));
    const auto builder_fn = [out_dtype](mlir::MlirOp val_op) {
      return CastIfNeeded(val_op, out_dtype);
    };
    TT_ASSIGN_OR_RETURN(
        auto res_buf,
        (DispatchOp<1, 1>(builder_fn, src_values,
                          {.out_dtype = out_dtype,
                           .out_dims = src_values.sizes(),
                           .op_param_cache_keys = std::move(pk)})));
    return MakeTensor(res_buf);
  }
  TT_ASSIGN_OR_RETURN(
      at::Tensor dst,
      MakeEmptyTensor(src_values.sizes(), target_dtype, target_device));
  CopyTensor(src_values, dst, non_blocking);
  return dst;
}

absl::StatusOr<at::Tensor> ScaleValuesOnTpu(const at::Tensor& values,
                                            const at::Tensor& alpha_tensor,
                                            const at::ScalarType target_dtype,
                                            OpParamCacheKeys pk) {
  if (values.numel() == 0) {
    return CopyOrCastValuesTensor(values, target_dtype, values.device(),
                                  /*non_blocking=*/false);
  }
  TT_ASSIGN_OR_RETURN(const auto out_dtype,
                      ConvertTo<mlir::ElementType>(target_dtype));
  const auto builder_fn = [out_dtype](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
    return BuildMulSparseScalarShlo(inputs[0], inputs[1], out_dtype);
  };
  TT_ASSIGN_OR_RETURN(
      auto res_buf, (DispatchOp<2, 1>(builder_fn, {values, alpha_tensor},
                                      {.out_dtype = out_dtype,
                                       .out_dims = values.sizes(),
                                       .op_param_cache_keys = std::move(pk)})));
  return MakeTensor(res_buf);
}

absl::StatusOr<int64_t> ExtractOutputSize(const at::Tensor& count_tensor) {
  at::Tensor count_cpu = at::empty(
      {}, at::TensorOptions().dtype(c10::ScalarType::Long).device(at::kCPU));
  TT_RETURN_IF_ERROR(CopyTpuToCpu(count_tensor, count_cpu,
                                  GetTransferSyncMode(/*non_blocking=*/false)));
  return count_cpu.item<int64_t>();
}

at::Tensor& AddOutDenseSparse(const at::Tensor& self, const at::Tensor& other,
                              MaybePromotedScalar& promoted_alpha,
                              OpParamCacheKeys pk, at::Tensor& out) {
  std::optional<at::Tensor> alpha_tensor_opt;
  if (!promoted_alpha.IsOne()) {
    TT_ASSIGN_OR_THROW(alpha_tensor_opt,
                       promoted_alpha.GetTensor(out.scalar_type()));
  }
  TT_CHECK_THROW(!out.is_sparse(), error::kInvalidArgument)
      << "expected dense output tensor when adding dense and sparse tensors, "
         "got sparse";
  TT_CHECK_THROW(self.sizes().equals(other.sizes()), error::kInvalidArgument)
      << "expected 'self' and 'other' to have same size, got "
      << ToString(self.sizes()) << " and " << ToString(other.sizes());
  if (other._nnz() == 0) {
    if (!out.is_same(self)) {
      TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, self.sizes()));
      if (self.scalar_type() != out.scalar_type()) {
        TT_ASSIGN_OR_THROW(
            at::Tensor converted,
            CopyOrCastValuesTensor(self, out.scalar_type(), out.device(),
                                   /*non_blocking=*/false));
        CopyTensor(converted, out, /*non_blocking=*/false);
      } else {
        CopyTensor(self, out, /*non_blocking=*/false);
      }
    }
    return out;
  }
  const at::ScalarType common_dtype = at::result_type(self, other);
  TT_CHECK_THROW(c10::canCast(common_dtype, out.scalar_type()),
                 error::kInvalidArgument)
      << "expected result dtype to be castable to 'out' dtype, got "
      << common_dtype << " and " << out.scalar_type();
  TT_ASSIGN_OR_THROW(const auto out_dtype,
                     ConvertTo<mlir::ElementType>(out.scalar_type()));
  if (alpha_tensor_opt.has_value()) {
    const auto builder_fn = [out_dtype](FixedSizeSpan<mlir::MlirOp, 4> inputs) {
      return BuildAddDenseSparseShlo(inputs[0], inputs[1], inputs[2], inputs[3],
                                     out_dtype);
    };
    TT_THROW_IF_ERROR((DispatchOpOut<4>(
        builder_fn,
        {self, other._indices(), other._values(), *alpha_tensor_opt}, out,
        {.out_dtype = out_dtype,
         .out_dims = self.sizes(),
         .op_param_cache_keys = std::move(pk)})));
  } else {
    const auto builder_fn = [out_dtype](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
      return BuildAddDenseSparseShlo(inputs[0], inputs[1], inputs[2],
                                     std::nullopt, out_dtype);
    };
    TT_THROW_IF_ERROR((DispatchOpOut<3>(
        builder_fn, {self, other._indices(), other._values()}, out,
        {.out_dtype = out_dtype,
         .out_dims = self.sizes(),
         .op_param_cache_keys = std::move(pk)})));
  }
  return out;
}

at::Tensor& AddOutSparseSparse(const at::Tensor& self, const at::Tensor& other,
                               MaybePromotedScalar& promoted_alpha,
                               OpParamCacheKeys pk, at::Tensor& out) {
  std::optional<at::Tensor> alpha_tensor_opt;
  if (!promoted_alpha.IsOne()) {
    TT_ASSIGN_OR_THROW(alpha_tensor_opt,
                       promoted_alpha.GetTensor(out.scalar_type()));
  }
  TT_CHECK_THROW(out.is_sparse(), error::kInvalidArgument)
      << "expected sparse output tensor when adding two sparse tensors, got "
         "dense";
  TT_CHECK_THROW(self.sizes().equals(other.sizes()), error::kInvalidArgument)
      << "expected 'self' and 'other' to have same size, got "
      << ToString(self.sizes()) << " and " << ToString(other.sizes());
  TT_CHECK_THROW(self.sparse_dim() == other.sparse_dim() &&
                     self.dense_dim() == other.dense_dim(),
                 error::kInvalidArgument)
      << "expected matching sparse_dim and dense_dim, got ("
      << self.sparse_dim() << ", " << self.dense_dim() << ") and ("
      << other.sparse_dim() << ", " << other.dense_dim() << ")";
  if (self._nnz() > 0 || other._nnz() > 0) {
    const at::ScalarType common_dtype = at::result_type(self, other);
    TT_CHECK_THROW(c10::canCast(common_dtype, out.scalar_type()),
                   error::kInvalidArgument)
        << "expected result dtype to be castable to 'out' dtype, got "
        << common_dtype << " and " << out.scalar_type();
  }
  if (other._nnz() == 0) {
    if (!out.is_same(self)) {
      TT_ASSIGN_OR_THROW(at::Tensor new_indices,
                         CopyIndicesTensor(self._indices(), out.device(),
                                           /*non_blocking=*/false));
      TT_ASSIGN_OR_THROW(
          at::Tensor new_values,
          CopyOrCastValuesTensor(self._values(), out.scalar_type(),
                                 out.device(), /*non_blocking=*/false));
      SetSparseCooTensorInPlace(out, self.sparse_dim(), self.dense_dim(),
                                self.sizes(), new_indices, new_values,
                                self.is_coalesced());
    }
    return out;
  }
  if (self._nnz() == 0) {
    if (!alpha_tensor_opt.has_value()) {
      if (!out.is_same(other)) {
        TT_ASSIGN_OR_THROW(at::Tensor new_indices,
                           CopyIndicesTensor(other._indices(), out.device(),
                                             /*non_blocking=*/false));
        TT_ASSIGN_OR_THROW(
            at::Tensor new_values,
            CopyOrCastValuesTensor(other._values(), out.scalar_type(),
                                   out.device(), /*non_blocking=*/false));
        SetSparseCooTensorInPlace(out, other.sparse_dim(), other.dense_dim(),
                                  other.sizes(), new_indices, new_values,
                                  other.is_coalesced());
      }
      return out;
    }
    TT_ASSIGN_OR_THROW(at::Tensor new_indices,
                       CopyIndicesTensor(other._indices(), out.device(),
                                         /*non_blocking=*/false));
    TT_ASSIGN_OR_THROW(at::Tensor new_values,
                       ScaleValuesOnTpu(other._values(), *alpha_tensor_opt,
                                        out.scalar_type(), std::move(pk)));
    SetSparseCooTensorInPlace(out, other.sparse_dim(), other.dense_dim(),
                              other.sizes(), new_indices, new_values,
                              other.is_coalesced());
    return out;
  }

  TT_ASSIGN_OR_THROW(const auto val_dtype,
                     ConvertTo<mlir::ElementType>(out.scalar_type()));
  TT_ASSIGN_OR_THROW(const auto i64_dtype,
                     ConvertTo<mlir::ElementType>(c10::ScalarType::Long));
  const std::array<mlir::ElementType, 2> out_dtypes = {i64_dtype, val_dtype};

  const int64_t total_nnz = self._nnz() + other._nnz();
  const Dimensions new_ind_dims = {self.sparse_dim(), total_nnz};
  const Dimensions new_val_dims = GetSparseValuesShape(
      total_nnz, self.sparse_dim(), self.dense_dim(), self.sizes());
  const std::array<absl::Span<const int64_t>, 2> out_dims_list = {new_ind_dims,
                                                                  new_val_dims};

  if (alpha_tensor_opt.has_value()) {
    const auto builder_fn = [val_dtype](FixedSizeSpan<mlir::MlirOp, 5> inputs) {
      return BuildAddSparseSparseShlo(inputs[0], inputs[1], inputs[2],
                                      inputs[3], inputs[4], val_dtype);
    };
    TT_ASSIGN_OR_THROW(
        auto results,
        (DispatchOp<5, 2>(builder_fn,
                          {self._indices(), self._values(), other._indices(),
                           other._values(), *alpha_tensor_opt},
                          {.out_dtypes = out_dtypes,
                           .out_dims_list = out_dims_list,
                           .op_param_cache_keys = std::move(pk)})));
    SetSparseCooTensorInPlace(out, self.sparse_dim(), self.dense_dim(),
                              self.sizes(), MakeTensor(results[0]),
                              MakeTensor(results[1]),
                              /*is_coalesced=*/false);
  } else {
    const auto builder_fn = [val_dtype](FixedSizeSpan<mlir::MlirOp, 4> inputs) {
      return BuildAddSparseSparseShlo(inputs[0], inputs[1], inputs[2],
                                      inputs[3], std::nullopt, val_dtype);
    };
    TT_ASSIGN_OR_THROW(
        auto results,
        (DispatchOp<4, 2>(builder_fn,
                          {self._indices(), self._values(), other._indices(),
                           other._values()},
                          {.out_dtypes = out_dtypes,
                           .out_dims_list = out_dims_list,
                           .op_param_cache_keys = std::move(pk)})));
    SetSparseCooTensorInPlace(out, self.sparse_dim(), self.dense_dim(),
                              self.sizes(), MakeTensor(results[0]),
                              MakeTensor(results[1]),
                              /*is_coalesced=*/false);
  }
  return out;
}

}  // namespace

struct TpuSparseTensorImpl final : public at::SparseTensorImpl {
  TpuSparseTensorImpl(const at::DispatchKeySet key_set,
                      const caffe2::TypeMeta data_type,
                      const c10::Device device)
      : at::SparseTensorImpl(at::DispatchKeySet(c10::DispatchKey::SparseCPU),
                             data_type) {
    key_set_ = key_set;
    device_opt_ = device;
  }
};

at::Tensor MakeSparseCooTensor(const int64_t sparse_dim,
                               const int64_t dense_dim,
                               const at::IntArrayRef sizes,
                               const at::Tensor& indices,
                               const at::Tensor& values,
                               const bool is_coalesced) {
  auto impl = c10::make_intrusive<TpuSparseTensorImpl>(
      at::DispatchKeySet(GetSparseDispatchKey(values.device())),
      c10::scalarTypeToTypeMeta(values.scalar_type()), values.device());
  impl->raw_resize_(sparse_dim, dense_dim, sizes);
  impl->set_indices_and_values_unsafe(ShallowCopyDetach(indices),
                                      ShallowCopyDetach(values));
  impl->set_coalesced(is_coalesced);
  return at::Tensor(std::move(impl));
}

namespace {

at::Tensor SparseCooTensorWithDimsImpl(
    const int64_t sparse_dim, const int64_t dense_dim,
    const at::IntArrayRef size, const std::optional<at::ScalarType> dtype,
    const std::optional<at::Device> device) {
  const at::ScalarType target_dtype =
      dtype.value_or(c10::get_default_dtype_as_scalartype());
  const at::Device target_device =
      device.value_or(at::Device(GetPrivateUse1DeviceType(), 0));
  TT_ASSIGN_OR_THROW(
      at::Tensor empty_indices,
      MakeEmptyTensor({sparse_dim, 0}, c10::ScalarType::Long, target_device));
  const Dimensions values_shape =
      GetSparseValuesShape(/*nnz=*/0, sparse_dim, dense_dim, size);
  TT_ASSIGN_OR_THROW(
      at::Tensor empty_values,
      MakeEmptyTensor(values_shape, target_dtype, target_device));
  return MakeSparseCooTensor(sparse_dim, dense_dim, size, empty_indices,
                             empty_values, /*is_coalesced=*/true);
}

at::Tensor EmptyLikeSparseCooImpl(const at::Tensor& self,
                                  const std::optional<at::ScalarType> dtype,
                                  const std::optional<at::Device> device) {
  return SparseCooTensorWithDimsImpl(
      self.sparse_dim(), self.dense_dim(), self.sizes(),
      dtype.value_or(self.scalar_type()), device.value_or(self.device()));
}

at::Tensor& CopySparseToSparseImpl(at::Tensor& self, const at::Tensor& src,
                                   const bool non_blocking) {
  if (self.is_same(src)) {
    return self;
  }
  TT_ASSIGN_OR_THROW(
      at::Tensor new_indices,
      CopyIndicesTensor(src._indices(), self.device(), non_blocking));
  TT_ASSIGN_OR_THROW(at::Tensor new_values,
                     CopyOrCastValuesTensor(src._values(), self.scalar_type(),
                                            self.device(), non_blocking));
  SetSparseCooTensorInPlace(self, src.sparse_dim(), src.dense_dim(),
                            src.sizes(), new_indices, new_values,
                            src.is_coalesced());
  return self;
}

at::Tensor CloneSparseImpl(
    const at::Tensor& self,
    const std::optional<at::MemoryFormat> memory_format) {
  TT_CHECK_THROW(!memory_format.has_value(), error::kInvalidArgument)
      << "unsupported memory format option for sparse tensor";
  TT_ASSIGN_OR_THROW(at::Tensor new_indices,
                     CopyIndicesTensor(self._indices(), self.device(),
                                       /*non_blocking=*/false));
  TT_ASSIGN_OR_THROW(
      at::Tensor new_values,
      CopyOrCastValuesTensor(self._values(), self.scalar_type(), self.device(),
                             /*non_blocking=*/false));
  return MakeSparseCooTensor(self.sparse_dim(), self.dense_dim(), self.sizes(),
                             new_indices, new_values, self.is_coalesced());
}

const at::Tensor& SparseResizeAndClearImpl(const at::Tensor& self,
                                           const at::IntArrayRef size,
                                           const int64_t sparse_dim,
                                           const int64_t dense_dim) {
  TT_ASSIGN_OR_THROW(
      at::Tensor empty_indices,
      MakeEmptyTensor({sparse_dim, 0}, c10::ScalarType::Long, self.device()));
  const Dimensions values_shape =
      GetSparseValuesShape(/*nnz=*/0, sparse_dim, dense_dim, size);
  TT_ASSIGN_OR_THROW(
      at::Tensor empty_values,
      MakeEmptyTensor(values_shape, self.scalar_type(), self.device()));
  SetSparseCooTensorInPlace(self, sparse_dim, dense_dim, size, empty_indices,
                            empty_values, /*is_coalesced=*/true);
  return self;
}

at::Tensor& AddOutSparseImpl(const at::Tensor& self, const at::Tensor& other,
                             MaybePromotedScalar& promoted_alpha,
                             OpParamCacheKeys pk, at::Tensor& out) {
  if (!self.is_sparse() && other.is_sparse()) {
    return AddOutDenseSparse(self, other, promoted_alpha, std::move(pk), out);
  }
  if (self.is_sparse() && other.is_sparse()) {
    return AddOutSparseSparse(self, other, promoted_alpha, std::move(pk), out);
  }
  if (!promoted_alpha.IsOne()) {
    const at::ScalarType alpha_dtype =
        out.defined() ? out.scalar_type() : at::result_type(self, other);
    TT_ASSIGN_OR_THROW([[maybe_unused]] const at::Tensor alpha_tensor,
                       promoted_alpha.GetTensor(alpha_dtype));
  }
  TT_CHECK_THROW(false, error::kPythonNotImplementedError)
      << "add(sparse, dense) is not supported; use add(dense, sparse) instead";
  return out;
}

at::Tensor AddSparseImpl(const at::Tensor& self, const at::Tensor& other,
                         MaybePromotedScalar& promoted_alpha,
                         OpParamCacheKeys pk) {
  const at::ScalarType out_dtype = at::result_type(self, other);
  at::Tensor out;  // UNINITIALIZED_TENSOR_OK=Assigned below for supported
                   // dense+sparse or sparse+sparse cases
  if (!self.is_sparse() && other.is_sparse()) {
    TT_ASSIGN_OR_THROW(out,
                       MakeEmptyTensor(self.sizes(), out_dtype, self.device()));
  } else if (self.is_sparse() && other.is_sparse()) {
    out = EmptyLikeSparseCooImpl(self, out_dtype, self.device());
  }
  return AddOutSparseImpl(self, other, promoted_alpha, std::move(pk), out);
}

at::Tensor& MulOutSparseImpl(const at::Tensor& self, const at::Tensor& other,
                             OpParamCacheKeys pk, at::Tensor& out) {
  const bool self_is_sparse = self.is_sparse();
  const bool other_is_sparse = other.is_sparse();
  TT_CHECK_THROW((self_is_sparse && !other_is_sparse && other.dim() == 0) ||
                     (!self_is_sparse && self.dim() == 0 && other_is_sparse),
                 error::kPythonNotImplementedError)
      << "only multiplication of a sparse tensor with a 0-dim scalar tensor is "
         "supported on TPU";
  const at::Tensor& sp = self_is_sparse ? self : other;
  const at::Tensor& scalar = self_is_sparse ? other : self;
  if (sp._nnz() == 0) {
    if (!out.is_same(sp)) {
      CopySparseToSparseImpl(out, sp, /*non_blocking=*/false);
    }
    return out;
  }
  const at::ScalarType common_dtype = at::result_type(self, other);
  TT_CHECK_THROW(c10::canCast(common_dtype, out.scalar_type()),
                 error::kInvalidArgument)
      << "expected result dtype to be castable to 'out' dtype, got "
      << common_dtype << " and " << out.scalar_type();

  at::Tensor scalar_dev = scalar;
  if (scalar.device() != sp.device()) {
    TT_ASSIGN_OR_THROW(
        scalar_dev,
        MakeEmptyTensor(scalar.sizes(), scalar.scalar_type(), sp.device()));
    CopyTensor(scalar, scalar_dev, /*non_blocking=*/false);
  }

  const at::Tensor sp_values = sp._values();
  TT_ASSIGN_OR_THROW(const auto out_dtype,
                     ConvertTo<mlir::ElementType>(out.scalar_type()));
  const auto builder_fn = [out_dtype](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
    return BuildMulSparseScalarShlo(inputs[0], inputs[1], out_dtype);
  };

  if (out.is_same(sp)) {
    at::Tensor out_values = out._values();
    TT_THROW_IF_ERROR(
        (DispatchOpOut<2>(builder_fn, {sp_values, scalar_dev}, out_values,
                          {.out_dtype = out_dtype,
                           .out_dims = sp_values.sizes(),
                           .op_param_cache_keys = std::move(pk)})));
    return out;
  }

  TT_ASSIGN_OR_THROW(
      at::Tensor new_indices,
      CopyIndicesTensor(sp._indices(), out.device(), /*non_blocking=*/false));
  TT_ASSIGN_OR_THROW(
      auto res_buf, (DispatchOp<2, 1>(builder_fn, {sp_values, scalar_dev},
                                      {.out_dtype = out_dtype,
                                       .out_dims = sp_values.sizes(),
                                       .op_param_cache_keys = std::move(pk)})));
  SetSparseCooTensorInPlace(out, sp.sparse_dim(), sp.dense_dim(), sp.sizes(),
                            new_indices, MakeTensor(res_buf),
                            sp.is_coalesced());
  return out;
}

at::Tensor& NegOutSparseImpl(const at::Tensor& self, OpParamCacheKeys pk,
                             at::Tensor& out) {
  TT_CHECK_THROW(self.scalar_type() != at::ScalarType::Bool,
                 error::kInvalidArgument)
      << "expected non-bool tensor for negation, got " << self.scalar_type();
  if (self._nnz() == 0) {
    if (!out.is_same(self)) {
      CopySparseToSparseImpl(out, self, /*non_blocking=*/false);
    }
    return out;
  }
  TT_CHECK_THROW(c10::canCast(self.scalar_type(), out.scalar_type()),
                 error::kInvalidArgument)
      << "expected result dtype to be castable to 'out' dtype, got "
      << self.scalar_type() << " and " << out.scalar_type();
  const at::Tensor self_values = self._values();
  TT_ASSIGN_OR_THROW(const auto out_dtype,
                     ConvertTo<mlir::ElementType>(out.scalar_type()));
  const auto builder_fn = [out_dtype](mlir::MlirOp input) {
    return BuildNegSparseShlo(input, out_dtype);
  };

  if (out.is_same(self)) {
    at::Tensor out_values = out._values();
    TT_THROW_IF_ERROR(
        (DispatchOpOut<1>(builder_fn, self_values, out_values,
                          {.out_dtype = out_dtype,
                           .out_dims = self_values.sizes(),
                           .op_param_cache_keys = std::move(pk)})));
    return out;
  }

  TT_ASSIGN_OR_THROW(
      at::Tensor new_indices,
      CopyIndicesTensor(self._indices(), out.device(), /*non_blocking=*/false));
  TT_ASSIGN_OR_THROW(
      auto res_buf, (DispatchOp<1, 1>(builder_fn, self_values,
                                      {.out_dtype = out_dtype,
                                       .out_dims = self_values.sizes(),
                                       .op_param_cache_keys = std::move(pk)})));
  SetSparseCooTensorInPlace(out, self.sparse_dim(), self.dense_dim(),
                            self.sizes(), new_indices, MakeTensor(res_buf),
                            self.is_coalesced());
  return out;
}

void CheckSubNotBool(const at::Tensor& self, const at::Tensor& other) {
  TT_CHECK_THROW(self.scalar_type() != at::ScalarType::Bool &&
                     other.scalar_type() != at::ScalarType::Bool,
                 error::kInvalidArgument)
      << "expected non-bool tensors for subtraction, got " << self.scalar_type()
      << " and " << other.scalar_type();
}

MaybePromotedScalar PromoteNegAlpha(const c10::Scalar& alpha) {
  if (alpha.isBoolean()) {
    return PromoteScalar(alpha).AvoidPromoting(ScalarValue::kOne);
  }
  return PromoteScalar(-alpha).AvoidPromoting(ScalarValue::kOne);
}

}  // namespace

at::Tensor& AtenAddOutSparse(const at::Tensor& self, const at::Tensor& other,
                             const c10::Scalar& alpha, at::Tensor& out) {
  auto promoted_alpha = PromoteScalar(alpha).AvoidPromoting(ScalarValue::kOne);
  TT_KERNEL(OpName::kAddOut, param_keys, (self, other, promoted_alpha, out), {
    return AddOutSparseImpl(self, other, promoted_alpha, std::move(param_keys),
                            out);
  });
}

at::Tensor AtenAddSparse(const at::Tensor& self, const at::Tensor& other,
                         const c10::Scalar& alpha) {
  auto promoted_alpha = PromoteScalar(alpha).AvoidPromoting(ScalarValue::kOne);
  TT_KERNEL(OpName::kAddTensor, param_keys, (self, other, promoted_alpha), {
    return AddSparseImpl(self, other, promoted_alpha, std::move(param_keys));
  });
}

at::Tensor& AtenAdd_Sparse(at::Tensor& self, const at::Tensor& other,
                           const c10::Scalar& alpha) {
  auto promoted_alpha = PromoteScalar(alpha).AvoidPromoting(ScalarValue::kOne);
  TT_KERNEL(OpName::kAdd_Tensor, param_keys, (self, other, promoted_alpha), {
    return AddOutSparseImpl(self, other, promoted_alpha, std::move(param_keys),
                            self);
  });
}

at::Tensor& AtenSubOutSparse(const at::Tensor& self, const at::Tensor& other,
                             const c10::Scalar& alpha, at::Tensor& out) {
  auto promoted_neg_alpha = PromoteNegAlpha(alpha);
  TT_KERNEL(OpName::kSubOut, param_keys, (self, other, promoted_neg_alpha, out),
            {
              CheckSubNotBool(self, other);
              return AddOutSparseImpl(self, other, promoted_neg_alpha,
                                      std::move(param_keys), out);
            });
}

at::Tensor AtenSubSparse(const at::Tensor& self, const at::Tensor& other,
                         const c10::Scalar& alpha) {
  auto promoted_neg_alpha = PromoteNegAlpha(alpha);
  TT_KERNEL(OpName::kSubTensor, param_keys, (self, other, promoted_neg_alpha), {
    CheckSubNotBool(self, other);
    return AddSparseImpl(self, other, promoted_neg_alpha,
                         std::move(param_keys));
  });
}

at::Tensor& AtenSub_Sparse(at::Tensor& self, const at::Tensor& other,
                           const c10::Scalar& alpha) {
  auto promoted_neg_alpha = PromoteNegAlpha(alpha);
  TT_KERNEL(OpName::kSub_Tensor, param_keys, (self, other, promoted_neg_alpha),
            {
              CheckSubNotBool(self, other);
              return AddOutSparseImpl(self, other, promoted_neg_alpha,
                                      std::move(param_keys), self);
            });
}

at::Tensor& AtenMulOutSparse(const at::Tensor& self, const at::Tensor& other,
                             at::Tensor& out) {
  TT_KERNEL(OpName::kMulOut, param_keys, (self, other, out), {
    return MulOutSparseImpl(self, other, std::move(param_keys), out);
  });
}

at::Tensor AtenMulSparse(const at::Tensor& self, const at::Tensor& other) {
  TT_KERNEL(OpName::kMulTensor, param_keys, (self, other), {
    const at::Tensor& sp = self.is_sparse() ? self : other;
    const at::ScalarType out_dtype = at::result_type(self, other);
    at::Tensor out = EmptyLikeSparseCooImpl(sp, out_dtype, sp.device());
    return MulOutSparseImpl(self, other, std::move(param_keys), out);
  });
}

at::Tensor& AtenMul_Sparse(at::Tensor& self, const at::Tensor& other) {
  TT_KERNEL(OpName::kMul_Tensor, param_keys, (self, other), {
    return MulOutSparseImpl(self, other, std::move(param_keys), self);
  });
}

at::Tensor& AtenNegOutSparse(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kNegOut, param_keys, (self, out),
            { return NegOutSparseImpl(self, std::move(param_keys), out); });
}

at::Tensor AtenNegSparse(const at::Tensor& self) {
  TT_KERNEL(OpName::kNeg, param_keys, (self), {
    at::Tensor out =
        EmptyLikeSparseCooImpl(self, self.scalar_type(), self.device());
    return NegOutSparseImpl(self, std::move(param_keys), out);
  });
}

at::Tensor& AtenNeg_Sparse(at::Tensor& self) {
  TT_KERNEL(OpName::kNeg_, param_keys, (self),
            { return NegOutSparseImpl(self, std::move(param_keys), self); });
}

at::Tensor AtenCoalesceSparse(const at::Tensor& self) {
  TT_KERNEL(OpName::kCoalesce, _, (self), {
    if (self.is_coalesced()) {
      return self;
    }
    if (self._nnz() < 2) {
      at::Tensor cloned = CloneSparseImpl(self, std::nullopt);
      cloned._coalesced_(true);
      return cloned;
    }
    const at::Tensor indices = self._indices();
    const at::Tensor values = self._values();
    TT_ASSIGN_OR_THROW(const auto val_dtype,
                       ConvertTo<mlir::ElementType>(values.scalar_type()));
    TT_ASSIGN_OR_THROW(const auto i64_dtype,
                       ConvertTo<mlir::ElementType>(c10::ScalarType::Long));
    const std::array<mlir::ElementType, 3> out_dtypes = {i64_dtype, val_dtype,
                                                         i64_dtype};
    const std::array<absl::Span<const int64_t>, 3> out_dims_list = {
        indices.sizes(), values.sizes(), {}};

    const auto coalesce_builder = [](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
      return BuildCoalesceAndCountShlo(inputs[0], inputs[1]);
    };
    TT_ASSIGN_OR_THROW(
        auto coalesce_results,
        (DispatchOp<2, 3>(coalesce_builder, {indices, values},
                          {.out_dtypes = out_dtypes,
                           .out_dims_list = out_dims_list,
                           .op_param_cache_keys = OpParamCacheKeys::Empty()})));
    const at::Tensor compacted_indices = MakeTensor(coalesce_results[0]);
    const at::Tensor compacted_values = MakeTensor(coalesce_results[1]);
    const at::Tensor count_tensor = MakeTensor(coalesce_results[2]);
    TT_ASSIGN_OR_THROW(const int64_t output_size,
                       ExtractOutputSize(count_tensor));
    if (output_size == self._nnz()) {
      return MakeSparseCooTensor(self.sparse_dim(), self.dense_dim(),
                                 self.sizes(), compacted_indices,
                                 compacted_values, /*is_coalesced=*/true);
    }
    Dimensions sliced_ind_dims = CopyIntVector(indices.sizes());
    sliced_ind_dims[1] = output_size;
    Dimensions sliced_val_dims = CopyIntVector(values.sizes());
    sliced_val_dims[0] = output_size;
    const std::array<mlir::ElementType, 2> slice_dtypes = {i64_dtype,
                                                           val_dtype};
    const std::array<absl::Span<const int64_t>, 2> slice_dims_list = {
        sliced_ind_dims, sliced_val_dims};
    TT_ASSIGN_OR_THROW(auto slice_pk, TT_MAKE_OP_PARAM_CACHE_KEYS(output_size));
    const auto slice_builder =
        [output_size](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
          return BuildSliceCoalescedShlo(inputs[0], inputs[1], output_size);
        };
    TT_ASSIGN_OR_THROW(
        auto slice_results,
        (DispatchOp<2, 2>(slice_builder, {compacted_indices, compacted_values},
                          {.out_dtypes = slice_dtypes,
                           .out_dims_list = slice_dims_list,
                           .op_param_cache_keys = std::move(slice_pk)})));
    return MakeSparseCooTensor(self.sparse_dim(), self.dense_dim(),
                               self.sizes(), MakeTensor(slice_results[0]),
                               MakeTensor(slice_results[1]),
                               /*is_coalesced=*/true);
  });
}

at::Tensor AtenToDenseSparse(const at::Tensor& self,
                             std::optional<c10::ScalarType> dtype,
                             std::optional<bool> masked_grad) {
  TT_KERNEL(
      OpName::kToDense, pk,
      (self, dtype, IgnoreInCacheKey(masked_grad, "Unused")), {
        TT_CHECK_THROW(!dtype.has_value(), error::kInvalidArgument)
            << "dtype argument is not supported by sparse_to_dense";
        const c10::ScalarType out_scalar_type = self.scalar_type();
        TT_ASSIGN_OR_THROW(const auto out_dtype,
                           ConvertTo<mlir::ElementType>(out_scalar_type));
        const Dimensions out_dims = CopyIntVector(self.sizes());
        if (self._nnz() == 0) {
          const auto zero_builder =
              [out_dims, out_dtype](
                  mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
            return MakeConstant(builder, c10::Scalar(0), out_dtype, out_dims);
          };
          TT_ASSIGN_OR_THROW(
              auto zero_buf,
              (DispatchOp<0, 1>(zero_builder, {},
                                {.out_dtype = out_dtype,
                                 .out_dims = out_dims,
                                 .op_param_cache_keys = std::move(pk)})));
          return MakeTensor(zero_buf);
        }
        const auto dense_builder =
            [out_dims, out_dtype](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
              return BuildSparseToDenseShlo(inputs[0], inputs[1], out_dims,
                                            out_dtype);
            };
        TT_ASSIGN_OR_THROW(
            auto dense_buf,
            (DispatchOp<2, 1>(dense_builder, {self._indices(), self._values()},
                              {.out_dtype = out_dtype,
                               .out_dims = out_dims,
                               .op_param_cache_keys = std::move(pk)})));
        return MakeTensor(dense_buf);
      });
}

at::Tensor AtenSparseCooTensorWithDims(
    const int64_t sparse_dim, const int64_t dense_dim,
    const at::IntArrayRef size, const std::optional<c10::ScalarType> dtype,
    const std::optional<c10::Layout> layout,
    const std::optional<c10::Device> device,
    const std::optional<bool> pin_memory) {
  TT_KERNEL(OpName::kSparseCooTensorWithDims, _,
            (IgnoreInCacheKey(sparse_dim, "Doesn't affect SHLO"),
             IgnoreInCacheKey(dense_dim, "Doesn't affect SHLO"),
             IgnoreInCacheKey(size, "Doesn't affect SHLO"),
             IgnoreInCacheKey(dtype, "Metadata allocation only"),
             IgnoreInCacheKey(layout, "Metadata allocation only"),
             IgnoreInCacheKey(device, "Metadata allocation only"),
             IgnoreInCacheKey(pin_memory, "Metadata allocation only")),
            {
              return SparseCooTensorWithDimsImpl(sparse_dim, dense_dim, size,
                                                 dtype, device);
            });
}

at::Tensor AtenEmptySparseSymInt(
    const c10::SymIntArrayRef size, const std::optional<c10::ScalarType> dtype,
    const std::optional<c10::Layout> layout,
    const std::optional<c10::Device> device,
    const std::optional<bool> pin_memory,
    const std::optional<c10::MemoryFormat> memory_format) {
  Dimensions int_size_vec;
  int_size_vec.reserve(size.size());
  for (const auto& s : size) {
    int_size_vec.push_back(s.guard_int(__FILE__, __LINE__));
  }
  TT_KERNEL(
      OpName::kEmptyMemoryFormat, _,
      (IgnoreInCacheKey(int_size_vec, "Doesn't affect SHLO"),
       IgnoreInCacheKey(dtype, "Delegates to SparseCooTensorWithDimsImpl"),
       IgnoreInCacheKey(layout, "Delegates to SparseCooTensorWithDimsImpl"),
       IgnoreInCacheKey(device, "Delegates to SparseCooTensorWithDimsImpl"),
       IgnoreInCacheKey(pin_memory, "Delegates to SparseCooTensorWithDimsImpl"),
       IgnoreInCacheKey(memory_format, "Unused")),
      {
        return SparseCooTensorWithDimsImpl(
            /*sparse_dim=*/static_cast<int64_t>(int_size_vec.size()),
            /*dense_dim=*/0, int_size_vec, dtype, device);
      });
}

at::Tensor AtenEmptyLikeSparseCoo(
    const at::Tensor& self, const std::optional<c10::ScalarType> dtype,
    const std::optional<c10::Layout> layout,
    const std::optional<c10::Device> device,
    const std::optional<bool> pin_memory,
    const std::optional<c10::MemoryFormat> memory_format) {
  TT_KERNEL(
      OpName::kEmptyLike, _,
      (self, IgnoreInCacheKey(dtype, "Delegates to EmptyLikeSparseCooImpl"),
       IgnoreInCacheKey(layout, "Delegates to EmptyLikeSparseCooImpl"),
       IgnoreInCacheKey(device, "Delegates to EmptyLikeSparseCooImpl"),
       IgnoreInCacheKey(pin_memory, "Delegates to EmptyLikeSparseCooImpl"),
       IgnoreInCacheKey(memory_format, "Unused")),
      { return EmptyLikeSparseCooImpl(self, dtype, device); });
}

at::Tensor AtenSparseCooTensorWithDimsAndTensors(
    const int64_t sparse_dim, const int64_t dense_dim,
    const c10::SymIntArrayRef size, const at::Tensor& indices,
    const at::Tensor& values, const std::optional<c10::ScalarType> dtype,
    const std::optional<c10::Layout> layout,
    const std::optional<c10::Device> device,
    const std::optional<bool> pin_memory,
    const std::optional<bool> is_coalesced) {
  Dimensions int_size_vec;
  int_size_vec.reserve(size.size());
  for (const auto& s : size) {
    int_size_vec.push_back(s.guard_int(__FILE__, __LINE__));
  }
  TT_KERNEL(OpName::kSparseCooTensorWithDimsAndTensors, _,
            (IgnoreInCacheKey(sparse_dim, "Doesn't affect SHLO"),
             IgnoreInCacheKey(dense_dim, "Doesn't affect SHLO"),
             IgnoreInCacheKey(int_size_vec, "Doesn't affect SHLO"), indices,
             values, IgnoreInCacheKey(dtype, "Metadata wrapper"),
             IgnoreInCacheKey(layout, "Metadata wrapper"),
             IgnoreInCacheKey(device, "Metadata wrapper"),
             IgnoreInCacheKey(pin_memory, "Metadata wrapper"),
             IgnoreInCacheKey(is_coalesced, "Metadata wrapper")),
            {
              const bool coalesced = is_coalesced.value_or(values.size(0) < 2);
              return MakeSparseCooTensor(sparse_dim, dense_dim, int_size_vec,
                                         indices, values, coalesced);
            });
}

const at::Tensor& AtenSparseResize_(const at::Tensor& self,
                                    const at::IntArrayRef size,
                                    const int64_t sparse_dim,
                                    const int64_t dense_dim) {
  TT_KERNEL(
      OpName::kSparseResize_, _,
      (self, IgnoreInCacheKey(size, "Doesn't affect SHLO"),
       IgnoreInCacheKey(sparse_dim, "Doesn't affect SHLO"),
       IgnoreInCacheKey(dense_dim, "Doesn't affect SHLO")),
      {
        if (self._nnz() == 0) {
          TT_ASSIGN_OR_THROW(
              at::Tensor empty_indices,
              MakeEmptyTensor({sparse_dim, 0}, c10::ScalarType::Long,
                              self.device()));
          const Dimensions values_shape =
              GetSparseValuesShape(/*nnz=*/0, sparse_dim, dense_dim, size);
          TT_ASSIGN_OR_THROW(
              at::Tensor empty_values,
              MakeEmptyTensor(values_shape, self.scalar_type(), self.device()));
          SetSparseCooTensorInPlace(self, sparse_dim, dense_dim, size,
                                    empty_indices, empty_values,
                                    self.is_coalesced());
          return self;
        }
        at::sparse::get_sparse_impl(self)->resize_(sparse_dim, dense_dim, size);
        return self;
      });
}

const at::Tensor& AtenSparseResizeAndClear_(const at::Tensor& self,
                                            const at::IntArrayRef size,
                                            const int64_t sparse_dim,
                                            const int64_t dense_dim) {
  TT_KERNEL(
      OpName::kSparseResizeAndClear_, _,
      (self, IgnoreInCacheKey(size, "Doesn't affect SHLO"),
       IgnoreInCacheKey(sparse_dim, "Doesn't affect SHLO"),
       IgnoreInCacheKey(dense_dim, "Doesn't affect SHLO")),
      { return SparseResizeAndClearImpl(self, size, sparse_dim, dense_dim); });
}

at::Tensor& AtenZero_Sparse(at::Tensor& self) {
  TT_KERNEL(OpName::kZero_, _, (self), {
    SparseResizeAndClearImpl(self, self.sizes(), self.sparse_dim(),
                             self.dense_dim());
    return self;
  });
}

at::Tensor AtenCloneSparse(
    const at::Tensor& self,
    const std::optional<c10::MemoryFormat> memory_format) {
  TT_KERNEL(OpName::kClone, _,
            (self, IgnoreInCacheKey(memory_format, "Unused")),
            { return CloneSparseImpl(self, memory_format); });
}

at::Tensor& AtenCopySparseToSparse_(at::Tensor& self, const at::Tensor& src,
                                    const bool non_blocking) {
  TT_KERNEL(OpName::kCopySparseToSparse_, _,
            (self, src, IgnoreInCacheKey(non_blocking, "Doesn't affect SHLO")),
            { return CopySparseToSparseImpl(self, src, non_blocking); });
}

at::Tensor& AtenCopy_Sparse(at::Tensor& self, const at::Tensor& src,
                            const bool non_blocking) {
  TT_KERNEL(OpName::kCopy_, _,
            (self, src, IgnoreInCacheKey(non_blocking, "Doesn't affect SHLO")),
            {
              TT_CHECK_THROW(self.is_sparse() && src.is_sparse(),
                             error::kInvalidArgument)
                  << "copy_() between dense and sparse tensors is not "
                     "implemented";
              return CopySparseToSparseImpl(self, src, non_blocking);
            });
}

at::Tensor AtenToCopySparse(
    const at::Tensor& self, const std::optional<c10::ScalarType> dtype,
    const std::optional<c10::Layout> layout,
    const std::optional<c10::Device> device,
    const std::optional<bool> pin_memory, const bool non_blocking,
    const std::optional<c10::MemoryFormat> memory_format) {
  TT_KERNEL(
      OpName::kToCopy, _,
      (self, IgnoreInCacheKey(dtype, "Doesn't affect SHLO"),
       IgnoreInCacheKey(layout, "Unused"),
       IgnoreInCacheKey(device, "Doesn't affect SHLO"),
       IgnoreInCacheKey(pin_memory, "Unused"),
       IgnoreInCacheKey(non_blocking, "Doesn't affect SHLO"),
       IgnoreInCacheKey(memory_format, "Unused")),
      {
        const at::Device target_device = device.value_or(self.device());
        const at::ScalarType target_dtype = dtype.value_or(self.scalar_type());
        TT_ASSIGN_OR_THROW(
            at::Tensor new_indices,
            CopyIndicesTensor(self._indices(), target_device, non_blocking));
        TT_ASSIGN_OR_THROW(at::Tensor new_values,
                           CopyOrCastValuesTensor(self._values(), target_dtype,
                                                  target_device, non_blocking));
        return MakeSparseCooTensor(self.sparse_dim(), self.dense_dim(),
                                   self.sizes(), new_indices, new_values,
                                   self.is_coalesced());
      });
}

int64_t AtenSparseDimSparse(const at::Tensor& self) {
  return at::sparse::get_sparse_impl(self)->sparse_dim();
}

int64_t AtenDenseDimSparse(const at::Tensor& self) {
  return at::sparse::get_sparse_impl(self)->dense_dim();
}

int64_t AtenNnzSparse(const at::Tensor& self) {
  return at::sparse::get_sparse_impl(self)->nnz();
}

bool AtenIsCoalescedSparse(const at::Tensor& self) {
  return at::sparse::get_sparse_impl(self)->coalesced();
}

at::Tensor& AtenCoalesced_Sparse(at::Tensor& self, const bool coalesced) {
  at::sparse::get_sparse_impl(self)->set_coalesced(coalesced);
  return self;
}

at::Tensor AtenIndicesInternalSparse(const at::Tensor& self) {
  return at::sparse::get_sparse_impl(self)->indices();
}

at::Tensor AtenValuesInternalSparse(const at::Tensor& self) {
  return at::sparse::get_sparse_impl(self)->values();
}

at::Tensor AtenIndicesSparse(const at::Tensor& self) {
  TT_KERNEL(OpName::kIndices, _, (self), {
    TT_CHECK_THROW(self.is_coalesced(), error::kPythonRuntimeError)
        << "cannot get indices on an uncoalesced tensor, please call "
           ".coalesce() first";
    return ShallowCopyDetach(at::sparse::get_sparse_impl(self)->indices());
  });
}

at::Tensor AtenValuesSparse(const at::Tensor& self) {
  TT_KERNEL(OpName::kValues, _, (self), {
    TT_CHECK_THROW(self.is_coalesced(), error::kPythonRuntimeError)
        << "cannot get values on an uncoalesced tensor, please call "
           ".coalesce() first";
    return ShallowCopyDetach(at::sparse::get_sparse_impl(self)->values());
  });
}

}  // namespace torch_tpu
