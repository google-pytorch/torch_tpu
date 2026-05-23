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
#include "torch_tpu/ops/linalg/lu/linalg_lu_kernels.h"

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/IListRef_inl.h"
#include "ATen/ops/_linalg_check_errors.h"
#include "ATen/ops/add.h"
#include "ATen/ops/any.h"
#include "ATen/ops/argmax.h"
#include "ATen/ops/diagflat.h"
#include "ATen/ops/diagonal.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/empty_like.h"
#include "ATen/ops/eq.h"
#include "ATen/ops/ones.h"
#include "ATen/ops/reshape.h"
#include "ATen/ops/tril.h"
#include "ATen/ops/triu.h"
#include "ATen/ops/where.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/linalg/solve_triangular/linalg_solve_triangular_kernels.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

namespace {

absl::StatusOr<MlirOpResults<2>> LuDecompositionBuilder(mlir::MlirOp input) {
  auto& builder = input.getBuilder();
  auto& op_builder = builder.getOpBuilder();
  auto& ctx = builder.getContext();

  auto a_type = GetTensorTypeOrDie(input);
  auto a_shape = a_type.getShape();
  int rank = a_shape.size();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Dimensions constraints are checked
                 // before Mlir lowering.
      rank >= 2, error::kInvalidArgument)
      << "input tensor expected to have at least 2 dimensions, got " << rank;

  const int n = a_shape[rank - 2];
  const int m = a_shape[rank - 1];
  const int num_batch_dims = rank - 2;
  Dimensions batch_dims(a_shape.begin(), a_shape.begin() + num_batch_dims);
  Dimensions pivot_dims = batch_dims;
  pivot_dims.push_back(std::min(n, m));
  Dimensions perm_dims = batch_dims;
  perm_dims.push_back(n);

  auto pivot_type =
      mlir::RankedTensorType::get(pivot_dims, op_builder.getI32Type());
  auto perm_type =
      mlir::RankedTensorType::get(perm_dims, op_builder.getI32Type());

  auto call_target_attr = op_builder.getNamedAttr(
      "call_target_name", op_builder.getStringAttr("LuDecomposition"));
  auto has_side_effect_attr =
      op_builder.getNamedAttr("has_side_effect", op_builder.getBoolAttr(false));
  auto backend_config_attr = op_builder.getNamedAttr(
      "backend_config", op_builder.getDictionaryAttr({}));
  auto api_version_attr = op_builder.getNamedAttr(
      "api_version",
      mlir::stablehlo::CustomCallApiVersionAttr::get(
          &ctx, mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI));

  mlir::stablehlo::CustomCallOp lu_op = mlir::stablehlo::CustomCallOp::create(
      op_builder, input.getBuilder().getLoc(), /*resultTypes=*/
      {
          a_type,
          pivot_type,
          perm_type,
      },
      /*operands=*/{input.getValue()},
      {call_target_attr, has_side_effect_attr, backend_config_attr,
       api_version_attr});
  mlir::MlirOp lu(builder, lu_op.getResult(0));
  mlir::MlirOp pivots(builder, lu_op.getResult(1));
  mlir::MlirOp perm(builder, lu_op.getResult(2));
  return {{lu, pivots}};
}

// Applies permutation represented by `pivots` in-place to either rows or
// columns of `tensor`. If inverse=true, applies the inverse permutation.
absl::Status ApplyPivotsInPlace(at::Tensor& tensor, const at::Tensor& pivots,
                                bool to_rows, bool inverse) {
  int rank = tensor.dim();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Error is caught before this check.
      rank >= 2, error::kInvalidArgument)
      << "tensor must have at least 2 dimensions, got " << rank;
  TT_RET_CHECK(pivots.dim() + 1 == rank, error::kInvalidArgument)
      << "pivots must have one less dimension than the tensor, got "
      << pivots.dim() << " and " << rank;
  const int size = to_rows ? tensor.size(-2) : tensor.size(-1);
  // TODO: PyTorch actually expects: pivots.size(-1) >= size
  // Otherwise, it accesses out-of-bounds memory.
  //
  // See link below:
  // https://github.com/pytorch/pytorch/blob/b323a6e5a358588d36e6f797ac81d89bf199546c/aten/src/ATen/native/BatchLinearAlgebraKernel.cpp#L1193-L1195
  TT_RET_CHECK(pivots.size(-1) <= size, error::kInvalidArgument)
      << "pivots size must be less than or equal to the size of the matrix, "
         "got "
      << pivots.size(-1) << " and " << size;
  Dimensions tensor_batch_dims(tensor.sizes().begin(),
                               tensor.sizes().begin() + rank - 2);
  Dimensions pivots_batch_dims(pivots.sizes().begin(),
                               pivots.sizes().begin() + pivots.dim() - 1);
  TT_RET_CHECK(tensor_batch_dims == pivots_batch_dims, error::kInvalidArgument)
      << "pivots and tensor must have the same batch dimensions, got "
      << ToString(tensor_batch_dims) << " and " << ToString(pivots_batch_dims);

  // Flatten batch dimensions.
  auto pivots_flat = at::reshape(pivots, {-1, pivots.size(-1)});
  int64_t n_batch = pivots_flat.size(0);
  int64_t n_pivots = pivots_flat.size(1);
  auto tensor_view =
      tensor.reshape({n_batch, tensor.size(-2), tensor.size(-1)});
  for (int i = 0; i < n_pivots; ++i) {
    // Forward swaps apply P^T, reverse swaps apply P.
    int k = inverse ? i : n_pivots - 1 - i;
    auto pivot_indices = pivots_flat.select(1, k).sub(1);
    for (int batch = 0; batch < n_batch; ++batch) {
      int p = pivot_indices[batch].item<int>();
      if (p != k) {
        if (to_rows) {
          auto r1 = tensor_view[batch].select(0, k);
          auto r2 = tensor_view[batch].select(0, p);
          auto tmp = r1.clone();
          r1.copy_(r2);
          r2.copy_(tmp);
        } else {
          auto c1 = tensor_view[batch].select(1, k);
          auto c2 = tensor_view[batch].select(1, p);
          auto tmp = c1.clone();
          c1.copy_(c2);
          c2.copy_(tmp);
        }
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> AtenLinalgLuFactorExOut(
    const at::Tensor& a, bool pivot, bool check_errors, at::Tensor& lu,
    at::Tensor& pivots, at::Tensor& info) {
  TT_KERNEL(OpName::kLinalgLuFactorExOut, param_keys,
            (a, pivot, IgnoreInCacheKey(check_errors, "Doesn't affect SHLO"),
             lu, pivots, info),
            {
              TT_CHECK_THROW(pivot, error::kInvalidArgument)
                  << "non-pivoting decomposition is not supported";
              TT_CHECK_THROW(a.dim() >= 2, error::kInvalidArgument)
                  << "input tensor expected to have at least 2 dimensions, got "
                  << a.dim();
              const int n = a.size(a.dim() - 2);
              const int m = a.size(a.dim() - 1);
              const int num_batch_dims = a.dim() - 2;
              Dimensions batch_dims(a.sizes().begin(),
                                    a.sizes().begin() + num_batch_dims);
              Dimensions pivot_dims = batch_dims;
              pivot_dims.push_back(std::min(n, m));

              TT_ASSIGN_OR_THROW(mlir::ElementType out_mlir_type,
                                 ConvertTo<mlir::ElementType>(a.scalar_type()));
              auto a_f32 = a;
              if (a.scalar_type() == c10::ScalarType::Double) {
                ABSL_VLOG(1)
                    << "linalg.factor_ex.out: lowering is only implemented "
                       "for f32, so casting f64 down";
                a_f32 = a.to(c10::ScalarType::Float);
              }

              TT_ASSIGN_OR_THROW(
                  auto results,
                  (DispatchOp<1, 2>(
                      LuDecompositionBuilder, {a_f32},
                      {.out_dtypes = {out_mlir_type, mlir::ElementType::I32},
                       .out_dims_list = {a.sizes(), pivot_dims},
                       .op_param_cache_keys = std::move(param_keys)})));

              if (lu.sizes() != a.sizes()) {
                lu.resize_(a.sizes());
              }
              TT_THROW_IF_ERROR(AssignBufferToAtTensor(results[0], lu));
              if (!pivots.sizes().equals(pivot_dims)) {
                pivots.resize_(pivot_dims);
              }
              TT_THROW_IF_ERROR(AssignBufferToAtTensor(results[1], pivots));
              // Pivots are 0-based, but torch uses 1-based indexing.
              pivots.add_(1);
              // Find the first non-zero diagonal element.
              auto diagonal = at::diagonal(lu, /*offset=*/0, -2, -1);
              auto zeros = at::eq(diagonal, 0).to(c10::ScalarType::Double);
              auto has_zeros = at::any(zeros, -1);
              info.zero_();
              auto indices = at::add(at::argmax(zeros, -1), 1);
              info = at::where(has_zeros, indices, 0).to(c10::kInt);
              if (check_errors) {
                at::_linalg_check_errors(info, "lu_factor", a.dim() == 2);
              }
              return {lu, pivots, info};
            });
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> AtenLuUnpackOut(
    const at::Tensor& lu_data, const at::Tensor& lu_pivots, bool unpack_data,
    bool unpack_pivots, at::Tensor& p, at::Tensor& l, at::Tensor& u) {
  TT_KERNEL(
      OpName::kLuUnpackOut, _,
      (lu_data, lu_pivots, IgnoreInCacheKey(unpack_data, "Legacy usage"),
       IgnoreInCacheKey(unpack_pivots, "Legacy usage"), p, l, u),
      {
        if (unpack_data || unpack_pivots) {
          TT_CHECK_THROW(lu_data.dim() >= 2, error::kInvalidArgument)
              << "lu_data must have at least 2 dimensions, got "
              << lu_data.dim();
        }

        if (unpack_data) {
          // Check ranks match
          // TODO: b/483972819 resize outputs instead of raising errors.
          TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=PyTorch implementation
                           // resizes the output, instead of raising an error.
              l.dim() == lu_data.dim() && u.dim() == lu_data.dim(),
              error::kInvalidArgument)
              << "l, u, and lu_data must have the same rank, got " << l.dim()
              << ", " << u.dim() << ", and " << lu_data.dim();
          // Check last two dimensions match LU decomposition.
          int64_t m = lu_data.size(-2);
          int64_t n = lu_data.size(-1);
          int64_t k = std::min(m, n);
          // TODO: b/483972819 resize outputs instead of raising errors.
          TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=PyTorch implementation
                           // resizes the output, instead of raising an error.
              l.size(-2) == m && l.size(-1) == k, error::kInvalidArgument)
              << "L must have shape (..., m, k), got " << ToString(l.sizes())
              << " and " << m << " and " << k;
          // TODO: b/483972819 resize outputs instead of raising errors.
          TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=PyTorch implementation
                           // resizes the output, instead of raising an error.
              u.size(-2) == k && u.size(-1) == n, error::kInvalidArgument)
              << "U must have shape (..., k, n), got " << ToString(u.sizes())
              << " and " << k << " and " << n;
          // Check batch dimensions match.
          Dimensions L_batch_dims(l.sizes().begin(), l.sizes().end() - 2);
          Dimensions U_batch_dims(u.sizes().begin(), u.sizes().end() - 2);
          Dimensions LU_data_batch_dims(lu_data.sizes().begin(),
                                        lu_data.sizes().end() - 2);
          // TODO: b/483972819 resize outputs instead of raising errors.
          TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=PyTorch implementation
                           // resizes the output, instead of raising an error.
              L_batch_dims == U_batch_dims &&
                  U_batch_dims == LU_data_batch_dims,
              error::kInvalidArgument)
              << "l, u, and lu_data must have the same batch "
                 "dimensions, got "
              << ToString(L_batch_dims) << ", " << ToString(U_batch_dims)
              << ", and " << ToString(LU_data_batch_dims);
          l.zero_();
          l.add_(at::tril(lu_data.narrow(-1, 0, k), /*diagonal=*/-1));
          // at::eye is not implemented yet.
          l.add_(
              at::diagflat(at::ones({m}, lu_data.options())).narrow(-1, 0, k));
          u.zero_();
          u.add_(at::triu(lu_data.narrow(-2, 0, k), /*diagonal=*/0));
        }

        if (unpack_pivots) {
          TT_CHECK_THROW(lu_pivots.dim() >= 1, error::kInvalidArgument)
              << "lu_pivots must have at least 1 dimension, got "
              << lu_pivots.dim();
          // TODO: b/483972819 resize outputs instead of raising errors.
          TT_CHECK_THROW(p.dim() == lu_pivots.dim() + 1,
                         error::kInvalidArgument)
              << "expected the first output tensor to be a "
              << lu_pivots.dim() + 1 << "D tensor (pivots dimension + 1), got "
              << p.dim() << "D of shape " << ToString(p.sizes());
          // TODO: b/483972819 resize outputs instead of raising errors.
          TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=PyTorch implementation
                           // resizes the output, instead of raising an
                           // error.
              p.size(-1) == lu_data.size(-2) && p.size(-2) == lu_data.size(-2),
              error::kInvalidArgument)
              << "p must be square and of size equal to number of (non-batch) "
                 "rows of LU_data, got "
              << ToString(p.sizes()) << " and " << ToString(lu_data.size(-2));

          // Build permutation matrix from pivots.
          // pivots[..., i] = j means that in the i-th step of the
          // algorithm, the i-th row was swapped with the j-th row. (So j
          // >= i).
          p.zero_();
          p.add_(at::diagflat(at::ones({p.size(-1)}, p.options())));
          TT_THROW_IF_ERROR(ApplyPivotsInPlace(p, lu_pivots, /*to_rows=*/true,
                                               /*inverse=*/false));
        }
        return {p, l, u};
      });
}

at::Tensor& AtenLinalgLuSolveOut(const at::Tensor& lu, const at::Tensor& pivots,
                                 const at::Tensor& b, bool left, bool adjoint,
                                 at::Tensor& out) {
  TT_KERNEL(
      OpName::kLinalgLuSolveOut, param_keys,
      (lu, pivots, b, left, adjoint, out), {
        TT_CHECK_THROW(lu.dim() >= 2, error::kInvalidArgument)
            << "lu must have at least 2 dimensions, got " << lu.dim();
        TT_CHECK_THROW(lu.size(-2) == lu.size(-1), error::kInvalidArgument)
            << "lu must be square, got " << lu.size(-2) << " and "
            << lu.size(-1);
        TT_CHECK_THROW(pivots.dim() >= 1, error::kInvalidArgument)
            << "pivots must have at least 1 dimension, got " << pivots.dim();
        // TODO: PyTorch checks that `b` is able to be broadcasted into `lu`
        // batch dimensions.
        //
        // See link below:
        // https://github.com/pytorch/pytorch/blob/b323a6e5a358588d36e6f797ac81d89bf199546c/aten/src/ATen/native/BatchLinearAlgebra.cpp#L684
        TT_CHECK_THROW(b.dim() == lu.dim(), error::kInvalidArgument)
            << "the rank of b must be equal to the rank of lu, got "
               "rank(b) = "
            << b.dim() << " and rank(lu) = " << lu.dim();
        TT_CHECK_THROW((left && b.size(-2) == lu.size(-2)) ||
                           (!left && b.size(-1) == lu.size(-1)),
                       error::kInvalidArgument)
            << "b must have compatible dimensions with lu, got "
            << "b.shape[-2:]=(" << b.size(-2) << ", " << b.size(-1)
            << ") and lu.shape[-2:]=(" << lu.size(-2) << ", " << lu.size(-1)
            << "), and left=" << left;

        if (lu.numel() == 0 || b.numel() == 0) return out;

        // Note: the four cases we are dealing with here are
        // 1. left=True, adjoint=False: P L U X = B
        // 2. left=False, adjoint=False: X P L U = B
        // 3. left=False, adjoint=True: X (P L U)^H = B <-> X U^H L^H P^T = B
        // 4. left=True, adjoint=True: (P L U)^H X = B <-> U^H L^H P^T X = B
        // In each case, an `inversion` is moving the outer-most matrix from the
        // left to the right side of the equation.
        // Obs: If P B applies a permutation to the rows of B, then B P applies
        // the inverse permutation to the columns of B.

        // This is true in cases 1 and 3.
        const bool inversion_order_is_p_l_u = left ^ adjoint;

        const auto p_inversion = [pivots, inversion_order_is_p_l_u,
                                  left](at::Tensor& t) -> absl::Status {
          TT_RETURN_IF_ERROR(
              ApplyPivotsInPlace(t, pivots, /*to_rows=*/left,
                                 /*inverse=*/inversion_order_is_p_l_u));
          return absl::OkStatus();
        };

        const auto l_inversion = [lu, left, adjoint, &param_keys = param_keys](
                                     at::Tensor& t) -> absl::Status {
          TT_ASSIGN_OR_RETURN(mlir::ElementType out_dtype,
                              ConvertTo<mlir::ElementType>(t.scalar_type()));
          auto step_keys = param_keys.Clone();
          TT_RETURN_IF_ERROR(step_keys.SetParam("step", "L"));
          TT_ASSIGN_OR_RETURN(
              auto buffer,
              DispatchOp<2>(LinalgSolveTriangularBuilder(
                                /*upper=*/false, /*left=*/left,
                                /*unitriangular=*/true, /*adjoint=*/adjoint),
                            {lu, t},
                            {.out_dtype = out_dtype,
                             .out_dims = CopyIntVector(t.sizes()),
                             .op_param_cache_keys = std::move(step_keys)}));

          TT_RETURN_IF_ERROR(AssignBufferToAtTensor(std::move(buffer), t));
          return absl::OkStatus();
        };

        const auto u_inversion = [lu, left, adjoint, &param_keys = param_keys](
                                     at::Tensor& t) -> absl::Status {
          TT_ASSIGN_OR_RETURN(mlir::ElementType out_dtype,
                              ConvertTo<mlir::ElementType>(t.scalar_type()));
          auto step_keys = param_keys.Clone();
          TT_RETURN_IF_ERROR(step_keys.SetParam("step", "U"));
          TT_ASSIGN_OR_RETURN(
              auto buffer,
              DispatchOp<2>(LinalgSolveTriangularBuilder(
                                /*upper=*/true, /*left=*/left,
                                /*unitriangular=*/false, /*adjoint=*/adjoint),
                            {lu, t},
                            {.out_dtype = out_dtype,
                             .out_dims = CopyIntVector(t.sizes()),
                             .op_param_cache_keys = std::move(step_keys)}));

          TT_RETURN_IF_ERROR(AssignBufferToAtTensor(std::move(buffer), t));
          return absl::OkStatus();
        };

        out.copy_(b);
        if (inversion_order_is_p_l_u) {
          TT_THROW_IF_ERROR(p_inversion(out));
          TT_THROW_IF_ERROR(l_inversion(out));
          TT_THROW_IF_ERROR(u_inversion(out));
        } else {
          TT_THROW_IF_ERROR(u_inversion(out));
          TT_THROW_IF_ERROR(l_inversion(out));
          TT_THROW_IF_ERROR(p_inversion(out));
        }

        return out;
      });
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> AtenLinalgLuOut(
    const at::Tensor& a, bool pivot, at::Tensor& p, at::Tensor& l,
    at::Tensor& u) {
  TT_KERNEL(
      OpName::kLinalgLuOut, _,
      (a, IgnoreInCacheKey(pivot, "Legacy usage"), p, l, u), {
        at::Tensor lu = at::empty_like(a);
        Dimensions pivot_dims = CopyIntVector(a.sizes());
        pivot_dims.pop_back();
        int pivot_size = std::min(a.size(-2), a.size(-1));
        pivot_dims[pivot_dims.size() - 1] = pivot_size;
        at::Tensor pivots = at::empty(pivot_dims, a.options().dtype(at::kInt));
        at::Tensor info = at::empty(pivot_dims, a.options().dtype(at::kInt));
        AtenLinalgLuFactorExOut(a, pivot, /*check_errors=*/true, lu, pivots,
                                info);
        AtenLuUnpackOut(lu, pivots, /*unpack_data=*/true,
                        /*unpack_pivots=*/true, p, l, u);
        return {p, l, u};
      });
}

}  // namespace torch_tpu
