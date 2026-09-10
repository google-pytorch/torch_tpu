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
#include "csrc/ops/linalg/lu/linalg_lu_kernels.h"

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
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/copy_from/copy_from_aten_kernels.h"
#include "csrc/ops/linalg/solve_triangular/linalg_solve_triangular_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

absl::Status ValidateLuSupportedDtype(const at::Tensor& tensor) {
  const c10::ScalarType dtype = tensor.scalar_type();
  const bool is_f32_or_f64 = (dtype == c10::kFloat || dtype == c10::kDouble);
  const bool is_c64_or_c128 =
      (dtype == c10::kComplexFloat || dtype == c10::kComplexDouble);
  TT_RET_CHECK(is_f32_or_f64 || is_c64_or_c128, error::kInvalidArgument)
      << "expected the input dtype to be " << ToString(c10::kFloat) << ", "
      << ToString(c10::kDouble) << ", " << ToString(c10::kComplexFloat)
      << ", or " << ToString(c10::kComplexDouble) << ", got "
      << ToString(dtype);
  return absl::OkStatus();
}

absl::StatusOr<MlirOpResults<2>> LuDecompositionBuilder(mlir::MlirOp input) {
  auto& builder = input.getBuilder();
  auto& op_builder = builder.getOpBuilder();
  auto& ctx = builder.getContext();

  const auto a_type = GetTensorTypeOrDie(input);
  const auto a_shape = a_type.getShape();
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
  TT_RET_CHECK(pivots.dim() == rank - 1, error::kInvalidArgument)
      << "expected pivots to have one less dimension than the tensor, got "
      << pivots.dim() << " and " << rank;
  const int size = to_rows ? tensor.size(-2) : tensor.size(-1);
  // TODO: PyTorch actually expects: pivots.size(-1) >= size
  // Otherwise, it accesses out-of-bounds memory.
  //
  // See link below:
  // https://github.com/pytorch/pytorch/blob/b323a6e5a358588d36e6f797ac81d89bf199546c/aten/src/ATen/native/BatchLinearAlgebraKernel.cpp#L1193-L1195
  TT_RET_CHECK(pivots.size(-1) <= size, error::kInvalidArgument)
      << "expected pivots size to be <= " << size
      << " (the size of the matrix), got " << pivots.size(-1) << " and "
      << size;
  Dimensions tensor_batch_dims(tensor.sizes().begin(),
                               tensor.sizes().begin() + rank - 2);
  Dimensions pivots_batch_dims(pivots.sizes().begin(),
                               pivots.sizes().begin() + pivots.dim() - 1);
  TT_RET_CHECK(tensor_batch_dims == pivots_batch_dims, error::kInvalidArgument)
      << "expected pivots and tensor to have the same batch dimensions, got "
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
  TT_KERNEL(
      OpName::kLinalgLuFactorExOut, param_keys,
      (a, pivot, IgnoreInCacheKey(check_errors, "Doesn't affect SHLO"), lu,
       pivots, info),
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

        if (a.numel() == 0) {
          TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(lu, a.sizes()));
          TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(pivots, pivot_dims));
          TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(info, batch_dims));
          info.zero_();
          return std::forward_as_tuple(lu, pivots, info);
        }

        TT_THROW_IF_ERROR(ValidateLuSupportedDtype(a));

        TT_ASSIGN_OR_THROW(mlir::ElementType out_mlir_type,
                           ConvertTo<mlir::ElementType>(a.scalar_type()));

        TT_THROW_IF_ERROR((DispatchOpOut<1, 2>(
            LuDecompositionBuilder, a, {lu, pivots},
            {.out_dtypes = {out_mlir_type, mlir::ElementType::I32},
             .out_dims_list = {a.sizes(), pivot_dims},
             .op_param_cache_keys = std::move(param_keys)})));
        // Pivots are 0-based, but torch uses 1-based indexing.
        pivots.add_(1);
        // Find the first non-zero diagonal element.
        auto diagonal = at::diagonal(lu, /*offset=*/0, -2, -1);
        auto zeros = at::eq(diagonal, 0).to(c10::ScalarType::Double);
        auto has_zeros = at::any(zeros, -1);
        TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(info, batch_dims));
        info.zero_();
        auto indices = at::add(at::argmax(zeros, -1), 1);
        CopyTensor(at::where(has_zeros, indices, 0).to(c10::kInt), info);
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
      (lu_data, lu_pivots, IgnoreInCacheKey(unpack_data, "Doesn't affect SHLO"),
       IgnoreInCacheKey(unpack_pivots, "Doesn't affect SHLO"), p, l, u),
      {
        if (unpack_data || unpack_pivots) {
          TT_CHECK_THROW(lu_data.dim() >= 2, error::kInvalidArgument)
              << "expected lu_data to have at least 2 dimensions, got "
              << lu_data.dim();
        }

        int64_t m = lu_data.size(-2);
        int64_t n = lu_data.size(-1);
        int64_t k = std::min(m, n);
        Dimensions batch_dims(lu_data.sizes().begin(),
                              lu_data.sizes().end() - 2);

        if (unpack_data) {
          Dimensions l_dims = batch_dims;
          l_dims.reserve(l_dims.size() + 2);
          l_dims.push_back(m);
          l_dims.push_back(k);
          TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(l, l_dims));

          Dimensions u_dims = batch_dims;
          u_dims.reserve(u_dims.size() + 2);
          u_dims.push_back(k);
          u_dims.push_back(n);
          TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(u, u_dims));

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
              << "expected lu_pivots to have at least 1 dimension, got "
              << lu_pivots.dim();

          Dimensions p_dims = batch_dims;
          p_dims.reserve(p_dims.size() + 2);
          p_dims.push_back(m);
          p_dims.push_back(m);
          TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(p, p_dims));

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
        TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, b.sizes()));
        TT_CHECK_THROW(lu.dim() >= 2, error::kInvalidArgument)
            << "expected lu to have at least 2 dimensions, got " << lu.dim();
        TT_CHECK_THROW(lu.size(-2) == lu.size(-1), error::kInvalidArgument)
            << "expected lu to be square, got " << lu.size(-2) << " and "
            << lu.size(-1);
        TT_CHECK_THROW(pivots.dim() >= 1, error::kInvalidArgument)
            << "expected pivots to have at least 1 dimension, got "
            << pivots.dim();
        // TODO: PyTorch checks that `b` is able to be broadcasted into `lu`
        // batch dimensions.
        //
        // See link below:
        // https://github.com/pytorch/pytorch/blob/b323a6e5a358588d36e6f797ac81d89bf199546c/aten/src/ATen/native/BatchLinearAlgebra.cpp#L684
        TT_CHECK_THROW(b.dim() == lu.dim(), error::kInvalidArgument)
            << "expected b to have the same number of dimensions as lu ("
            << lu.dim() << "), got " << b.dim();
        if (left) {
          TT_CHECK_THROW(b.size(-2) == lu.size(-2), error::kInvalidArgument)
              << "expected b second-to-last dimension to match the "
                 "second-to-last dimension of lu ("
              << lu.size(-2) << ") when left=True, got " << b.size(-2);
        } else {
          TT_CHECK_THROW(b.size(-1) == lu.size(-1), error::kInvalidArgument)
              << "expected b last dimension to match the last dimension of lu ("
              << lu.size(-1) << ") when left=False, got " << b.size(-1);
        }

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

        const auto solve_triangular_step =
            [lu, left, adjoint, &param_keys = param_keys](
                at::Tensor& t, bool upper, bool unitriangular,
                std::string_view step_name) -> absl::Status {
          TT_ASSIGN_OR_RETURN(mlir::ElementType out_dtype,
                              ConvertTo<mlir::ElementType>(t.scalar_type()));
          auto step_keys = param_keys.Clone();
          TT_RETURN_IF_ERROR(step_keys.SetParam("step", step_name));

          return DispatchOpOut<2>(
              LinalgSolveTriangularBuilder({.upper = upper,
                                            .left = left,
                                            .unitriangular = unitriangular,
                                            .adjoint = adjoint}),
              {lu, t}, t,
              {.out_dtype = out_dtype,
               .out_dims = CopyIntVector(t.sizes()),
               .op_param_cache_keys = std::move(step_keys)});
        };

        const auto l_inversion = [&](at::Tensor& t) {
          return solve_triangular_step(t, /*upper=*/false,
                                       /*unitriangular=*/true, "L");
        };

        const auto u_inversion = [&](at::Tensor& t) {
          return solve_triangular_step(t, /*upper=*/true,
                                       /*unitriangular=*/false, "U");
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
      (a, IgnoreInCacheKey(pivot, "Delegates to AtenLinalgLuFactorExOut"), p, l,
       u),
      {
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
