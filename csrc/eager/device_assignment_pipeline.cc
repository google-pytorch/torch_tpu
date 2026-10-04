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

#include "csrc/eager/device_assignment_pipeline.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "csrc/common/error_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/ops/op_names.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

std::string VectorToString(absl::Span<const int64_t> vec) {
  return absl::StrCat("[", absl::StrJoin(vec, ", "), "]");
}

std::optional<std::vector<int64_t>>  // INT_VEC_OK
BuildRankToPartition(absl::Span<const int64_t> op_da,
                     absl::Span<const int64_t> resolved_da) {
  const int64_t sub_size = op_da.size();
  const int64_t full_size = resolved_da.size();
  absl::flat_hash_map<int64_t, int64_t> dev_to_partition;
  for (int64_t p = 0; p < full_size; ++p) {
    dev_to_partition[resolved_da[p]] = p;
  }
  std::vector<int64_t> rank_to_partition(sub_size);  // INT_VEC_OK
  for (int64_t r = 0; r < sub_size; ++r) {
    auto it = dev_to_partition.find(op_da[r]);
    if (it == dev_to_partition.end()) return std::nullopt;
    rank_to_partition[r] = it->second;
  }
  return rank_to_partition;
}

mlir::DenseIntElementsAttr RewriteReplicaGroupsAttr(
    mlir::DenseIntElementsAttr attr,
    absl::Span<const int64_t> rank_to_partition) {
  auto shaped_type = mlir::dyn_cast<mlir::RankedTensorType>(attr.getType());
  if (!shaped_type || shaped_type.getRank() != 2) {
    return attr;
  }
  const int64_t num_groups = shaped_type.getDimSize(0);
  const int64_t group_size = shaped_type.getDimSize(1);
  const int64_t sub_size = rank_to_partition.size();
  std::vector<int64_t> mapped_vals;  // INT_VEC_OK
  mapped_vals.reserve(shaped_type.getNumElements());
  for (const llvm::APInt& val : attr.getValues<llvm::APInt>()) {
    int64_t r = val.getSExtValue();
    if (r < 0 || r >= sub_size) {
      return attr;
    }
    mapped_vals.push_back(rank_to_partition[r]);
  }
  if (num_groups <= 0 || group_size <= 0) {
    return mlir::DenseIntElementsAttr::get(shaped_type, mapped_vals);
  }
  std::vector<std::vector<int64_t>> rows(             // INT_VEC_OK
      num_groups, std::vector<int64_t>(group_size));  // INT_VEC_OK
  for (int64_t g = 0; g < num_groups; ++g) {
    // Intra-group order is rank order and must be preserved; do not sort.
    for (int64_t i = 0; i < group_size; ++i) {
      rows[g][i] = mapped_vals[g * group_size + i];
    }
  }
  std::sort(rows.begin(), rows.end(),
            [](const auto& a, const auto& b) { return a[0] < b[0]; });
  std::vector<int64_t> rewritten_vals;  // INT_VEC_OK
  rewritten_vals.reserve(shaped_type.getNumElements());
  for (const auto& row : rows) {
    rewritten_vals.insert(rewritten_vals.end(), row.begin(), row.end());
  }
  return mlir::DenseIntElementsAttr::get(shaped_type, rewritten_vals);
}

mlir::DenseIntElementsAttr RewriteSourceTargetPairsAttr(
    mlir::DenseIntElementsAttr attr,
    absl::Span<const int64_t> rank_to_partition) {
  auto shaped_type = mlir::dyn_cast<mlir::RankedTensorType>(attr.getType());
  if (!shaped_type || shaped_type.getRank() != 2 ||
      shaped_type.getDimSize(1) != 2) {
    return attr;
  }
  const int64_t sub_size = rank_to_partition.size();
  std::vector<int64_t> rewritten_vals;  // INT_VEC_OK
  rewritten_vals.reserve(shaped_type.getNumElements());
  for (const llvm::APInt& val : attr.getValues<llvm::APInt>()) {
    int64_t r = val.getSExtValue();
    if (r < 0 || r >= sub_size) {
      return attr;
    }
    rewritten_vals.push_back(rank_to_partition[r]);
  }
  return mlir::DenseIntElementsAttr::get(shaped_type, rewritten_vals);
}

std::vector<int64_t> ResolvePallasDeviceAssignment(  // INT_VEC_OK
    absl::Span<const PallasDeviceAssignment> pallas_ops) {
  std::vector<int64_t> authoritative_pallas_da;  // INT_VEC_OK
  for (const PallasDeviceAssignment& pallas_op : pallas_ops) {
    if (pallas_op.device_assignment.empty()) {
      continue;
    }
    if (authoritative_pallas_da.empty()) {
      authoritative_pallas_da.assign(pallas_op.device_assignment.begin(),
                                     pallas_op.device_assignment.end());
      continue;
    }
    TT_CHECK_THROW(authoritative_pallas_da == pallas_op.device_assignment,
                   error::kInvalidArgument)
        << "conflicting device assignments detected: differing Pallas mesh "
           "device assignments in the same compilation traversal. Kernel DA "
           "0: "
        << VectorToString(authoritative_pallas_da)
        << " vs Kernel DA: " << VectorToString(pallas_op.device_assignment)
        << ". All Pallas kernels in a traversal must share the identical mesh";
  }
  return authoritative_pallas_da;
}

absl::Status VerifySameDeviceSet(absl::Span<const int64_t> resolved_da,
                                 const OpDeviceAssignment& op, size_t index) {
  ABSL_LOG(WARNING) << "Op [" << index << "] (" << ToString(op.op_name)
                    << ") device assignment "
                    << VectorToString(op.device_assignment)
                    << " differs from resolved device assignment "
                    << VectorToString(resolved_da);
  TT_RET_CHECK(op.device_assignment.size() == resolved_da.size(),
               error::kInvalidArgument)
      << "device assignment mismatch across operations. All operations in "
         "the traversal must share the same world-sized device assignment";
  const absl::flat_hash_set<int64_t> resolved_devs(resolved_da.begin(),
                                                   resolved_da.end());
  for (int64_t dev_id : op.device_assignment) {
    TT_RET_CHECK(resolved_devs.contains(dev_id), error::kInvalidArgument)
        << "device assignment mismatch across operations. All operations in "
           "the traversal must share the same world-sized device assignment";
  }
  return absl::OkStatus();
}

void RewriteCustomCallBackendConfig(
    mlir::Operation* op, absl::Span<const int64_t> rank_to_partition) {
  if (!mlir::isa<mlir::stablehlo::CustomCallOp>(op)) {
    return;
  }
  auto dict_attr = op->getAttrOfType<mlir::DictionaryAttr>("backend_config");
  if (!dict_attr) {
    return;
  }
  auto rg_attr = mlir::dyn_cast_or_null<mlir::DenseIntElementsAttr>(
      dict_attr.get("replica_groups"));
  if (!rg_attr) {
    return;
  }
  llvm::SmallVector<mlir::NamedAttribute> new_attrs;
  for (const mlir::NamedAttribute& named_attr : dict_attr) {
    if (named_attr.getName() == "replica_groups") {
      new_attrs.emplace_back(
          named_attr.getName(),
          RewriteReplicaGroupsAttr(rg_attr, rank_to_partition));
    } else {
      new_attrs.push_back(named_attr);
    }
  }
  op->setAttr("backend_config",
              mlir::DictionaryAttr::get(op->getContext(), new_attrs));
}

}  // namespace

absl::StatusOr<std::vector<int64_t>>  // INT_VEC_OK
ResolveDeviceAssignment(absl::Span<const OpDeviceAssignment> ops,
                        absl::Span<const PallasDeviceAssignment> pallas_ops) {
  std::vector<int64_t> resolved_da =  // INT_VEC_OK
      ResolvePallasDeviceAssignment(pallas_ops);

  for (size_t i = 0; i < ops.size(); ++i) {
    const auto& op = ops[i];
    if (op.device_assignment.empty()) {
      continue;
    }
    if (resolved_da.empty()) {
      resolved_da.assign(op.device_assignment.begin(),
                         op.device_assignment.end());
    } else if (resolved_da != op.device_assignment) {
      TT_RETURN_IF_ERROR(VerifySameDeviceSet(resolved_da, op, i));
    }
  }

  return resolved_da;
}

absl::StatusOr<std::vector<int64_t>>  // INT_VEC_OK
ResolveDeviceAssignmentForTraversal(
    absl::Span<const SharedDeviceBufferList> execution_order,
    absl::Span<const int64_t> pallas_device_assignment) {
  llvm::SmallVector<OpDeviceAssignment, 8> ops;
  llvm::SmallVector<PallasDeviceAssignment, 4> pallas_ops;
  if (!pallas_device_assignment.empty()) {
    pallas_ops.push_back(PallasDeviceAssignment{pallas_device_assignment});
  }
  for (const SharedDeviceBufferList& node : execution_order) {
    const auto& op = node->deferred_op();
    if (!op) {
      continue;
    }
    if (op->op_name() == OpName::kCustomKernel) {
      pallas_ops.push_back(PallasDeviceAssignment{op->device_assignment()});
    } else {
      ops.push_back(OpDeviceAssignment{
          .op_name = op->op_name(),
          .device_assignment = op->device_assignment(),
      });
    }
  }
  return ResolveDeviceAssignment(ops, pallas_ops);
}

absl::Status RewriteAndTileOperationForSubgroup(
    mlir::Operation* absl_nonnull root_op, absl::Span<const int64_t> op_da,
    absl::Span<const int64_t> resolved_da) {
  if (op_da == resolved_da || op_da.empty()) {
    return absl::OkStatus();
  }

  const std::optional<std::vector<int64_t>>  // INT_VEC_OK
      rank_to_partition = BuildRankToPartition(op_da, resolved_da);
  TT_RET_CHECK(rank_to_partition.has_value(), error::kInvalidArgument)
      << "failed to map op rank order to resolved device assignment";

  root_op->walk([&](mlir::Operation* op) {
    if (auto rg_attr =
            op->getAttrOfType<mlir::DenseIntElementsAttr>("replica_groups")) {
      op->setAttr("replica_groups",
                  RewriteReplicaGroupsAttr(rg_attr, *rank_to_partition));
    }
    if (auto st_attr = op->getAttrOfType<mlir::DenseIntElementsAttr>(
            "source_target_pairs")) {
      op->setAttr("source_target_pairs",
                  RewriteSourceTargetPairsAttr(st_attr, *rank_to_partition));
    }
    RewriteCustomCallBackendConfig(op, *rank_to_partition);
  });

  return absl::OkStatus();
}

absl::Status ApplySubgroupTilingOrReplicaRewriting(
    mlir::Operation* absl_nullable first_new_op,
    absl::Span<const mlir::func::FuncOp> new_func_ops,
    const DeferredOp& deferred_op, absl::Span<const int64_t> resolved_da) {
  const absl::Span<const int64_t> op_da = deferred_op.device_assignment();
  if (op_da == resolved_da || op_da.empty()) {
    return absl::OkStatus();
  }

  for (mlir::Operation* curr = first_new_op; curr != nullptr;
       curr = curr->getNextNode()) {
    TT_RETURN_IF_ERROR(
        RewriteAndTileOperationForSubgroup(curr, op_da, resolved_da));
  }
  for (mlir::func::FuncOp func_op : new_func_ops) {
    TT_RETURN_IF_ERROR(RewriteAndTileOperationForSubgroup(
        func_op.getOperation(), op_da, resolved_da));
  }
  return absl::OkStatus();
}

}  // namespace torch_tpu
