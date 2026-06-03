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

#include "torch_tpu/ops/dtensor/dtensor_ops_kernels.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
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
                               mlir::sdy::MeshAttr mesh_attr) {
  context->loadDialect<mlir::sdy::SdyDialect>();
  auto module_op = GetModuleOp(mlir_builder);
  const std::string mesh_name = "mesh";
  if (!module_op.lookupSymbol<mlir::sdy::MeshOp>(mesh_name)) {
    auto& op_builder = mlir_builder.getOpBuilder();
    mlir::OpBuilder::InsertionGuard guard(op_builder);
    op_builder.setInsertionPointToStart(module_op.getBody());
    op_builder.create<mlir::sdy::MeshOp>(op_builder.getUnknownLoc(), mesh_name,
                                         mesh_attr);
  }
  return absl::OkStatus();
}

mlir::sdy::MeshAttr ParseGenericMesh(std::string_view mesh_spec,
                                     mlir::MLIRContext* context) {
  context->loadDialect<mlir::sdy::SdyDialect>();
  llvm::StringRef spec_ref(mesh_spec.data(), mesh_spec.size());

  // Split spec into axes part and device_ids part
  // e.g. "mesh_dim_0:8[2,3,0,1,5,4,7,6]"
  auto [axes_spec, device_ids_spec] = spec_ref.split('[');

  llvm::SmallVector<mlir::sdy::MeshAxisAttr> axes;
  llvm::SmallVector<llvm::StringRef> dims;
  axes_spec.split(dims, ',');

  for (auto dim : dims) {
    auto [name_ref, size_ref] = dim.split(':');
    int64_t size = 0;
    size_ref.getAsInteger(10, size);
    axes.push_back(mlir::sdy::MeshAxisAttr::get(
        context, mlir::StringAttr::get(context, name_ref), size));
  }

  if (device_ids_spec.empty()) {
    return mlir::sdy::MeshAttr::get(context, axes);
  }

  // Parse device ids
  // Remove trailing ']'
  if (device_ids_spec.back() == ']') {
    device_ids_spec = device_ids_spec.drop_back();
  }

  llvm::SmallVector<llvm::StringRef> ids_strings;
  device_ids_spec.split(ids_strings, ',');

  llvm::SmallVector<int64_t> device_ids;
  for (auto id_str : ids_strings) {
    int64_t id = 0;
    id_str.getAsInteger(10, id);
    device_ids.push_back(id);
  }

  return mlir::sdy::MeshAttr::get(context, axes, device_ids);
}

mlir::sdy::TensorShardingAttr ParseGenericSharding(
    std::string_view sharding_spec, mlir::sdy::MeshAttr mesh,
    mlir::MLIRContext* context) {
  if (sharding_spec.empty()) {
    return nullptr;
  }
  llvm::StringRef spec_ref(sharding_spec.data(), sharding_spec.size());

  auto [dims_spec, partial_spec] = spec_ref.split('|');

  llvm::SmallVector<llvm::StringRef> dims;
  dims_spec.split(dims, ';');

  llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dim_shardings;
  for (auto dim : dims) {
    llvm::SmallVector<mlir::sdy::AxisRefAttr> axes;
    if (!dim.empty()) {
      llvm::SmallVector<llvm::StringRef> axis_names;
      dim.split(axis_names, ',');
      for (auto name : axis_names) {
        if (!name.empty()) {
          axes.push_back(mlir::sdy::AxisRefAttr::get(context, name));
        }
      }
    }
    dim_shardings.push_back(mlir::sdy::DimensionShardingAttr::get(
        context, axes, /*is_closed=*/true));
  }

  llvm::SmallVector<mlir::sdy::AxisRefAttr> replicated_axes;  // empty
  llvm::SmallVector<mlir::sdy::AxisRefAttr> unreduced_axes;
  if (!partial_spec.empty()) {
    llvm::SmallVector<llvm::StringRef> partial_axes;
    partial_spec.split(partial_axes, ',');
    for (auto name : partial_axes) {
      if (!name.empty()) {
        unreduced_axes.push_back(mlir::sdy::AxisRefAttr::get(context, name));
      }
    }
  }

  auto mesh_name_attr = mlir::StringAttr::get(context, "mesh");
  return mlir::sdy::TensorShardingAttr::get(
      context, mesh_name_attr, dim_shardings, replicated_axes, unreduced_axes);
}

Dimensions ComputeLocalShape(at::IntArrayRef global_shape,
                             mlir::sdy::MeshAttr mesh,
                             mlir::sdy::TensorShardingAttr sharding) {
  Dimensions local_shape(global_shape.begin(), global_shape.end());
  if (sharding == nullptr || mesh == nullptr) {
    return local_shape;
  }

  absl::flat_hash_map<std::string_view, int64_t> axis_sizes;
  for (auto axis : mesh.getAxes()) {
    axis_sizes[axis.getName()] = axis.getSize();
  }

  auto dim_shardings = sharding.getDimShardings();
  for (size_t i = 0; i < dim_shardings.size() && i < local_shape.size(); ++i) {
    for (auto axis_ref : dim_shardings[i].getAxes()) {
      std::string_view axis_name = axis_ref.getName();
      auto it = axis_sizes.find(axis_name);
      if (it != axis_sizes.end()) {
        local_shape[i] /= it->second;
      }
    }
  }
  return local_shape;
}

}  // namespace

at::Tensor SpmdMetadata(const at::Tensor& x, std::string_view mesh_spec,
                        std::string_view sharding_spec_in,
                        std::string_view sharding_spec_out,
                        at::IntArrayRef global_shape, std::string_view kind) {
  TT_KERNEL(
      OpName::kSpmdMetadata, param_keys,
      (x, mesh_spec, sharding_spec_in, sharding_spec_out, global_shape, kind), {
        TT_ASSIGN_OR_THROW(const mlir::ElementType mlir_dtype,
                           ConvertTo<mlir::ElementType>(x.scalar_type()));

        mlir::sdy::MeshAttr mesh_attr = nullptr;
        mlir::sdy::TensorShardingAttr sdy_sharding_out = nullptr;
        mlir::sdy::TensorShardingAttr sdy_sharding_in = nullptr;

        mlir::MLIRContext temp_ctx;

        // We only parse specifications if mesh_spec is non-empty
        if (!mesh_spec.empty()) {
          mlir::DialectRegistry registry;
          registry.insert<mlir::sdy::SdyDialect>();
          temp_ctx.appendDialectRegistry(registry);
          mesh_attr = ParseGenericMesh(mesh_spec, &temp_ctx);
          sdy_sharding_in =
              ParseGenericSharding(sharding_spec_in, mesh_attr, &temp_ctx);
          sdy_sharding_out =
              ParseGenericSharding(sharding_spec_out, mesh_attr, &temp_ctx);
        }

        Dimensions out_dims;
        if (kind == "output") {
          out_dims =
              ComputeLocalShape(global_shape, mesh_attr, sdy_sharding_in);
        } else {
          out_dims = CopyIntVector(global_shape);
        }

        auto builder = [mesh_spec = std::string(mesh_spec),
                        sharding_spec_in = std::string(sharding_spec_in),
                        sharding_spec_out = std::string(sharding_spec_out),
                        kind = std::string(kind), out_dims](
                           mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
          auto& mlir_builder = input.getBuilder();
          mlir::MLIRContext* context = &mlir_builder.getContext();

          if (mesh_spec.empty()) {
            return input;
          }

          mlir::sdy::MeshAttr mesh_attr = ParseGenericMesh(mesh_spec, context);
          TT_RETURN_IF_ERROR(
              EnsureMeshDefined(mlir_builder, context, mesh_attr));

          mlir::sdy::TensorShardingAttr sdy_sharding_in =
              ParseGenericSharding(sharding_spec_in, mesh_attr, context);
          mlir::sdy::TensorShardingAttr sdy_sharding_out =
              ParseGenericSharding(sharding_spec_out, mesh_attr, context);

          mlir::Value val = input.getValue();
          mlir::OpBuilder& op_builder = mlir_builder.getOpBuilder();

          // Enforce strict lowering policy
          if (kind == "intermediate") {
            auto constraint_op =
                op_builder.create<mlir::sdy::ShardingConstraintOp>(
                    op_builder.getUnknownLoc(), val, sdy_sharding_out);
            return mlir::MlirOp(mlir_builder, constraint_op.getResult());
          }

          if (kind == "input") {
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

              func_op.setArgAttr(block_arg.getArgNumber(), "sdy.sharding",
                                 sdy_sharding_out);
              return mlir::MlirOp(mlir_builder, block_arg);
            } else {
              return TT_ERROR(error::kInternal)
                     << "input sentinel input is not a block argument";
            }
          }

          if (kind == "output") {
            auto constraint_op =
                op_builder.create<mlir::sdy::ShardingConstraintOp>(
                    op_builder.getUnknownLoc(), val, sdy_sharding_in);
            return mlir::MlirOp(mlir_builder, constraint_op.getResult());
          }

          return TT_ERROR(error::kInvalidArgument)
                 << "unknown spmd_metadata classification kind: " << kind;
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
