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

#include "torch_tpu/ops/dtensor/dtensor_ops.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

namespace {

// Helper to ensure SdyDialect is loaded and define the mesh if needed.
absl::Status EnsureMeshDefined(mlir::MlirBuilder& mlir_builder,
                               mlir::MLIRContext* context,
                               const std::string& mesh_sdy_format) {
  context->loadDialect<mlir::sdy::SdyDialect>();
  auto module_op = GetModuleOp(mlir_builder);
  const std::string mesh_name = "mesh";
  if (!module_op.lookupSymbol<mlir::sdy::MeshOp>(mesh_name)) {
    auto mesh_attr = mlir::parseAttribute(mesh_sdy_format, context);
    TT_RET_CHECK(mesh_attr != nullptr, error::kInvalidArgument)
        << "failed to parse mesh attribute: " << mesh_sdy_format;
    auto sdy_mesh_attr = mlir::dyn_cast<mlir::sdy::MeshAttr>(mesh_attr);

    auto& op_builder = mlir_builder.getOpBuilder();
    mlir::OpBuilder::InsertionGuard guard(op_builder);
    op_builder.setInsertionPointToStart(module_op.getBody());
    op_builder.create<mlir::sdy::MeshOp>(op_builder.getUnknownLoc(), mesh_name,
                                         sdy_mesh_attr);
  }
  return absl::OkStatus();
}

}  // namespace

at::Tensor SdySharding(const at::Tensor& x, std::string_view mesh_sdy_format,
                       std::string_view placements_sdy_format,
                       at::IntArrayRef target_shape, bool is_input) {
  TT_KERNEL(
      OpName::kSdySharding, param_keys,
      (x, mesh_sdy_format, placements_sdy_format, target_shape, is_input), {
        TT_ASSIGN_OR_THROW(const mlir::ElementType mlir_dtype,
                           ConvertTo<mlir::ElementType>(x.scalar_type()));
        const auto out_dims = CopyIntVector(target_shape);

        auto builder =
            [mesh_sdy_format = std::string(mesh_sdy_format),
             placements_sdy_format = std::string(placements_sdy_format),
             is_input,
             out_dims](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          auto& mlir_builder = input.getBuilder();
          mlir::MLIRContext* context = &mlir_builder.getContext();

          TT_RETURN_IF_ERROR(
              EnsureMeshDefined(mlir_builder, context, mesh_sdy_format));

          auto sdy_sharding_attr =
              mlir::parseAttribute(placements_sdy_format, context);
          TT_RET_CHECK(sdy_sharding_attr != nullptr, error::kInvalidArgument)
              << "failed to parse sharding attribute: "
              << placements_sdy_format;
          auto sdy_tensor_sharding_attr =
              mlir::dyn_cast<mlir::sdy::TensorShardingAttr>(sdy_sharding_attr);

          mlir::Value val = input.getValue();
          mlir::OpBuilder& op_builder = mlir_builder.getOpBuilder();

          if (is_input) {
            if (auto block_arg = mlir::dyn_cast<mlir::BlockArgument>(val)) {
              auto func_op = mlir::cast<mlir::func::FuncOp>(
                  block_arg.getOwner()->getParentOp());

              auto old_type =
                  mlir::cast<mlir::RankedTensorType>(block_arg.getType());
              auto global_type = mlir::RankedTensorType::get(
                  out_dims, old_type.getElementType());

              block_arg.setType(global_type);

              auto fn_type = func_op.getFunctionType();
              auto input_types = llvm::to_vector(fn_type.getInputs());
              input_types[block_arg.getArgNumber()] = global_type;
              func_op.setFunctionType(mlir::FunctionType::get(
                  func_op.getContext(), input_types, fn_type.getResults()));

              // Indicate to Shardy/XLA that the input is sharded in the
              // function signature
              func_op.setArgAttr(block_arg.getArgNumber(), "sdy.sharding",
                                 sdy_tensor_sharding_attr);

              auto constraint_op =
                  op_builder.create<mlir::sdy::ShardingConstraintOp>(
                      op_builder.getUnknownLoc(), block_arg,
                      sdy_tensor_sharding_attr);
              return mlir::MlirOp(mlir_builder, constraint_op.getResult());
            } else {
              return TT_ERROR(error::kInternal)
                     << "input sentinel input is not a block argument";
            }
          } else {
            // For output sentinel, we apply sharding constraint on the global
            // output
            auto constraint_op =
                op_builder.create<mlir::sdy::ShardingConstraintOp>(
                    op_builder.getUnknownLoc(), val, sdy_tensor_sharding_attr);

            // Dynamic return mapping: Find which function return index this
            // output maps to, and set the "sdy.sharding" result attribute to
            // compile a local PJRT return.
            if (auto* parent_op = val.getParentBlock()->getParentOp()) {
              if (auto func_op =
                      mlir::dyn_cast<mlir::func::FuncOp>(parent_op)) {
                auto& block = func_op.getBody().front();
                if (!block.empty() &&
                    mlir::isa<mlir::func::ReturnOp>(block.back())) {
                  auto return_op =
                      mlir::cast<mlir::func::ReturnOp>(block.getTerminator());
                  int res_idx = -1;
                  for (unsigned i = 0; i < return_op.getNumOperands(); ++i) {
                    if (return_op.getOperand(i) == constraint_op.getResult() ||
                        return_op.getOperand(i) == val) {
                      res_idx = i;
                      break;
                    }
                  }
                  if (res_idx != -1) {
                    func_op.setResultAttr(res_idx, "sdy.sharding",
                                          sdy_tensor_sharding_attr);
                  }
                }
              }
            }
            return mlir::MlirOp(mlir_builder, constraint_op.getResult());
          }
        };

        TT_ASSIGN_OR_THROW(
            auto result_buf,
            DispatchOp<1>(std::move(builder), {x},
                          {.out_dtype = mlir_dtype,
                           .out_dims = out_dims,
                           .op_param_cache_keys = std::move(param_keys)}));
        return MakeTensor(std::move(result_buf));
      });
}

at::Tensor SdyShardingConstraint(const at::Tensor& x,
                                 std::string_view mesh_sdy_format,
                                 std::string_view placements_sdy_format) {
  TT_KERNEL(
      OpName::kSdyShardingConstraint, param_keys,
      (x, mesh_sdy_format, placements_sdy_format), {
        TT_ASSIGN_OR_THROW(const mlir::ElementType mlir_dtype,
                           ConvertTo<mlir::ElementType>(x.scalar_type()));
        const auto out_dims = CopyIntVector(x.sizes());

        auto builder = [mesh_sdy_format = std::string(mesh_sdy_format),
                        placements_sdy_format =
                            std::string(placements_sdy_format)](
                           mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          auto& mlir_builder = input.getBuilder();
          mlir::MLIRContext* context = &mlir_builder.getContext();

          TT_RETURN_IF_ERROR(
              EnsureMeshDefined(mlir_builder, context, mesh_sdy_format));

          auto sdy_sharding_attr =
              mlir::parseAttribute(placements_sdy_format, context);
          TT_RET_CHECK(sdy_sharding_attr != nullptr, error::kInvalidArgument)
              << "failed to parse sharding attribute: "
              << placements_sdy_format;
          auto sdy_tensor_sharding_attr =
              mlir::dyn_cast<mlir::sdy::TensorShardingAttr>(sdy_sharding_attr);

          mlir::Value val = input.getValue();
          mlir::OpBuilder& op_builder = mlir_builder.getOpBuilder();

          auto constraint_op =
              op_builder.create<mlir::sdy::ShardingConstraintOp>(
                  op_builder.getUnknownLoc(), val, sdy_tensor_sharding_attr);
          return mlir::MlirOp(mlir_builder, constraint_op.getResult());
        };

        TT_ASSIGN_OR_THROW(
            auto result_buf,
            DispatchOp<1>(std::move(builder), {x},
                          {.out_dtype = mlir_dtype,
                           .out_dims = out_dims,
                           .op_param_cache_keys = std::move(param_keys)}));
        return MakeTensor(std::move(result_buf));
      });
}

}  // namespace torch_tpu
