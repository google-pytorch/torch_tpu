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

#include "torch_tpu/ops/sort_looped/sort_looped_kernels.h"

#include <cstdint>
#include <vector>

#include "absl/log/absl_check.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {
namespace {

namespace stablehlo = mlir::stablehlo;

mlir::stablehlo::CustomCallOp BuildCustomCallOp(
    mlir::MlirBuilder& builder, llvm::StringRef call_target_name,
    llvm::ArrayRef<mlir::Type> result_types,
    llvm::ArrayRef<mlir::Value> operands,
    mlir::DictionaryAttr backend_config = {}) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::NamedAttribute call_target_attr = op_builder.getNamedAttr(
      "call_target_name", op_builder.getStringAttr(call_target_name));
  mlir::NamedAttribute has_side_effect_attr =
      op_builder.getNamedAttr("has_side_effect", op_builder.getBoolAttr(false));
  auto api_version_attr = op_builder.getNamedAttr(
      "api_version",
      mlir::stablehlo::CustomCallApiVersionAttr::get(
          &builder.getContext(),
          mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI));

  std::vector<mlir::NamedAttribute> attrs = {
      call_target_attr, has_side_effect_attr, api_version_attr};
  if (backend_config) {
    attrs.push_back(op_builder.getNamedAttr("backend_config", backend_config));
  }

  return mlir::stablehlo::CustomCallOp::create(op_builder, builder.getLoc(),
                                               result_types, operands, attrs);
}

}  // namespace

// This file does not implement a PyTorch operator kernel directly, so TT_KERNEL
// is not used here. Instead, it provides a helper function
// BuildSortLoopedShlo to build the StableHLO graph, which is called by the
// actual kernel in sort_aten_kernels.cc.
SortLoopedShloOutputs BuildSortLoopedShlo(mlir::MlirOp input_op, bool stable,
                                          int64_t dim, bool descending) {
  const mlir::RankedTensorType inputType = GetTensorTypeOrDie(input_op);
  mlir::MlirBuilder& builder = input_op.getBuilder();

  // Our LLO sort currently only supports 1D tensors and ascending sort.
  // We enforce these constraints here.
  ABSL_CHECK_EQ(  // CRASH_OK=Looped bitonic sort only supports 1D tensors.
      inputType.getRank(), 1)
      << "Looped bitonic sort only supports 1D tensors";
  ABSL_CHECK_EQ(  // CRASH_OK=Looped bitonic sort only supports dim=0.
      dim, 0)
      << "Looped bitonic sort only supports dim=0 for 1D tensors";
  ABSL_CHECK(  // CRASH_OK=Looped bitonic sort only supports ascending order.
      !descending)
      << "Looped bitonic sort only supports ascending order";

  // Build backend_config dictionary to force VMEM allocation.
  mlir::OpBuilder& op_builder = builder.getOpBuilder();

  // Input colors: [{operand_index = 0, color = 1}]
  mlir::NamedAttribute operand_index_attr =
      op_builder.getNamedAttr("operand_index", op_builder.getI64IntegerAttr(0));
  mlir::NamedAttribute input_color_attr =
      op_builder.getNamedAttr("color", op_builder.getI64IntegerAttr(1));
  mlir::DictionaryAttr input_color_dict =
      op_builder.getDictionaryAttr({operand_index_attr, input_color_attr});
  mlir::ArrayAttr input_colors_array =
      op_builder.getArrayAttr({input_color_dict});

  // Output colors: [{shape_index = [], color = 1}]
  mlir::NamedAttribute shape_index_attr =
      op_builder.getNamedAttr("shape_index", op_builder.getI64ArrayAttr({}));
  mlir::NamedAttribute output_color_attr =
      op_builder.getNamedAttr("color", op_builder.getI64IntegerAttr(1));
  mlir::DictionaryAttr output_color_dict =
      op_builder.getDictionaryAttr({shape_index_attr, output_color_attr});
  mlir::ArrayAttr output_colors_array =
      op_builder.getArrayAttr({output_color_dict});

  // custom_call_config: {input_memory_space_colors = ...,
  // output_memory_space_colors = ...}
  mlir::NamedAttribute input_colors_field =
      op_builder.getNamedAttr("input_memory_space_colors", input_colors_array);
  mlir::NamedAttribute output_colors_field = op_builder.getNamedAttr(
      "output_memory_space_colors", output_colors_array);
  mlir::DictionaryAttr custom_call_config_dict =
      op_builder.getDictionaryAttr({input_colors_field, output_colors_field});

  // backend_config: {custom_call_config = ...}
  mlir::NamedAttribute custom_call_config_field =
      op_builder.getNamedAttr("custom_call_config", custom_call_config_dict);
  mlir::DictionaryAttr backend_config_dict =
      op_builder.getDictionaryAttr({custom_call_config_field});

  // 1. Build CustomCall for values
  auto custom_call_op = BuildCustomCallOp(builder, "looped_bitonic_sort",
                                          {inputType},  // result types
                                          {input_op.getValue()},  // operands
                                          backend_config_dict);
  mlir::MlirOp sorted_values(builder, custom_call_op.getResult(0));

  // 2. Build dummy indices (Iota)
  auto mlir_element_type = mlir::ElementType::I64;
  mlir::MlirOp dummy_indices =
      stablehlo::Iota(builder,
                      makeTensorType(builder.getContext(), inputType.getShape(),
                                     mlir_element_type),
                      0);  // dim 0

  return SortLoopedShloOutputs{.values = sorted_values,
                               .indices = dummy_indices};
}

}  // namespace torch_tpu
