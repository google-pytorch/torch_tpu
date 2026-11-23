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

#include "csrc/ops/cumsum/cumsum.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "absl/status/statusor.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/scan_builder.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

// Matches `TpuReduceWindowRewriter`'s `base_length = target.LaneCount()` (128
// lanes on TPU). Above 128 elements along the scan axis, `reduce_window`
// decomposes recursively into multi-pass `{L/128, 128}` chunks with extra
// SparseCore/HBM traffic, where `chlo.ScanOp` is faster (`2.98x` at L=2048).
constexpr int64_t kMaxReduceWindowScanLen = 128;

// Below 256K elements (1 MiB in `f32`), `chlo.ScanOp` stays on TPU's fast
// single-tile / row-parallel lane body (`6.56x` faster on `[2,32,32,64]`).
constexpr int64_t kMinReduceWindowElements = 262144;

mlir::MlirOp BuildCumsumReduceWindowShlo(mlir::MlirBuilder& builder,
                                         mlir::MlirOp input,
                                         int64_t normalized_dim,
                                         mlir::ElementType compute_dtype) {
  const mlir::RankedTensorType type = GetTensorTypeOrDie(input);
  const int64_t rank = type.getRank();
  const int64_t scan_len = type.getDimSize(normalized_dim);

  llvm::SmallVector<int64_t> window_dims(rank, 1);
  window_dims[normalized_dim] = scan_len;
  llvm::SmallVector<int64_t> window_strides(rank, 1);
  llvm::SmallVector<int64_t> padding_vals(rank * 2, 0);
  padding_vals[normalized_dim * 2] = scan_len - 1;

  auto* ctx = &builder.getContext();
  auto window_dimensions_attr = mlir::DenseI64ArrayAttr::get(ctx, window_dims);
  auto window_strides_attr = mlir::DenseI64ArrayAttr::get(ctx, window_strides);
  auto padding_attr = mlir::DenseIntElementsAttr::get(
      mlir::RankedTensorType::get({rank, 2},
                                  builder.getOpBuilder().getI64Type()),
      llvm::ArrayRef<int64_t>(padding_vals));

  const mlir::Type elem_type = mlir::getElementType(*ctx, compute_dtype);
  auto init_value = MakeScalarConstant(builder, 0.0, elem_type);
  auto body_builder = [elem_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        elem_type, rb.getRegion(), rb.getOpBuilder());
  };
  return mlir::stablehlo::ReduceWindow(
      builder, {input}, {init_value}, body_builder, window_dimensions_attr,
      window_strides_attr, /*base_dilations=*/{}, /*window_dilations=*/{},
      padding_attr)[0];
}

}  // namespace

CumsumLoweringStrategy SelectCumsumLoweringStrategy(
    const mlir::RankedTensorType input_type, const int64_t normalized_dim) {
  if (!input_type || !input_type.hasStaticShape() || input_type.getRank() < 1 ||
      normalized_dim < 0 || normalized_dim >= input_type.getRank()) {
    return CumsumLoweringStrategy::kChloScan;
  }
  const int64_t scan_len = input_type.getDimSize(normalized_dim);
  if (scan_len <= 1 || scan_len > kMaxReduceWindowScanLen) {
    return CumsumLoweringStrategy::kChloScan;
  }
  bool reached_threshold = false;
  int64_t total_elems = 1;
  for (int64_t dim_size : input_type.getShape()) {
    if (dim_size <= 0) {
      return CumsumLoweringStrategy::kChloScan;
    }
    if (!reached_threshold) {
      if (dim_size >= kMinReduceWindowElements ||
          total_elems >= (kMinReduceWindowElements + dim_size - 1) / dim_size) {
        reached_threshold = true;
      } else {
        total_elems *= dim_size;
      }
    }
  }
  return reached_threshold ? CumsumLoweringStrategy::kReduceWindow
                           : CumsumLoweringStrategy::kChloScan;
}

absl::StatusOr<mlir::MlirOp> BuildCumsumShlo(
    const int64_t normalized_dim,
    const std::optional<mlir::ElementType> out_dtype, mlir::MlirOp input) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  // PyTorch default promotion: when out_dtype is omitted, integral types
  // default to int64 (i64) to prevent overflow; float/complex preserve their
  // input dtype.
  mlir::ElementType target_element_type = out_dtype.value_or(
      input_type.getElementType().isInteger() ? mlir::ElementType::I64
                                              : GetElementTypeOrDie(input));

  mlir::MlirBuilder& builder = input.getBuilder();
  if (GetElementTypeOrDie(input) != target_element_type) {
    input = mlir::stablehlo::ConvertElementType(input, target_element_type);
  }
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_dtype,
                      InferComputationDtype(target_element_type));
  if (GetElementTypeOrDie(input) != compute_dtype) {
    input = mlir::stablehlo::ConvertElementType(input, compute_dtype);
  }

  mlir::MlirOp scan_result;
  if (SelectCumsumLoweringStrategy(GetTensorTypeOrDie(input), normalized_dim) ==
      CumsumLoweringStrategy::kReduceWindow) {
    scan_result = BuildCumsumReduceWindowShlo(builder, input, normalized_dim,
                                              compute_dtype);
  } else {
    TT_ASSIGN_OR_RETURN(
        const mlir::MlirOp carry_init,
        CreateRankReducedConstantLike(input, normalized_dim, 0));

    MultiInputScanBodyBuilder body_builder =
        [](mlir::OpBuilder& op_builder, mlir::Location loc,
           mlir::ValueRange input_slices, mlir::Value /*index*/,
           mlir::ValueRange carries) -> absl::StatusOr<ScanBodyResults> {
      llvm::SmallVector<mlir::Value> sum = {
          mlir::stablehlo::AddOp::create(op_builder, loc, input_slices[0],
                                         carries[0])
              .getResult()};
      // For a cumulative sum the per-position output is the running carry.
      return ScanBodyResults{sum, sum};
    };

    // Associative scan -> chlo.ScanOp (native scan emitter). Results are
    // [carries..., outputs...]; the prefix-scan output is the single output.
    TT_ASSIGN_OR_RETURN(
        const DynamicMlirOpResults results,
        BuildScanShlo(
            builder, {input}, normalized_dim, /*num_scan_inputs=*/1,
            /*carry_inits=*/{carry_init}, /*output_inits=*/{input},
            std::move(body_builder),
            ScanOptions{.should_squeeze = true, .is_associative = true}));
    scan_result = results[/*num_carries=*/1];
  }

  if (GetElementTypeOrDie(scan_result) != target_element_type) {
    return mlir::stablehlo::ConvertElementType(scan_result,
                                               target_element_type);
  }
  return scan_result;
}

}  // namespace torch_tpu
