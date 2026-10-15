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
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
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
#include "csrc/eager/process_group_partitions.h"
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

// Maps each op rank r to the partition of `resolved_da` that holds device
// `op_da[r]`. Returns std::nullopt unless `op_da` and `resolved_da` have the
// same size and every device of `op_da` is in `resolved_da`, i.e. unless the
// result is a permutation.
std::optional<std::vector<int64_t>>  // INT_VEC_OK
BuildRankToPartition(absl::Span<const int64_t> op_da,
                     absl::Span<const int64_t> resolved_da) {
  if (op_da.size() != resolved_da.size()) {
    return std::nullopt;
  }
  const int64_t size = op_da.size();
  absl::flat_hash_map<int64_t, int64_t> dev_to_partition;
  for (int64_t p = 0; p < size; ++p) {
    dev_to_partition[resolved_da[p]] = p;
  }
  std::vector<int64_t> rank_to_partition(size);  // INT_VEC_OK
  for (int64_t r = 0; r < size; ++r) {
    auto it = dev_to_partition.find(op_da[r]);
    if (it == dev_to_partition.end()) return std::nullopt;
    rank_to_partition[r] = it->second;
  }
  return rank_to_partition;
}

// Remaps each replica id r of a 2D `replica_groups` attribute to
// `rank_to_partition[r]`, keeping the shape and the order of the groups and of
// the ids within each group. Returns `attr` unchanged if it is not 2D or holds
// an id outside `rank_to_partition`.
mlir::DenseIntElementsAttr RewriteReplicaGroupsAttr(
    mlir::DenseIntElementsAttr attr,
    absl::Span<const int64_t> rank_to_partition) {
  auto shaped_type = mlir::dyn_cast<mlir::RankedTensorType>(attr.getType());
  if (!shaped_type || shaped_type.getRank() != 2) {
    return attr;
  }
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
  return mlir::DenseIntElementsAttr::get(shaped_type, mapped_vals);
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
        << "expected Pallas kernels and scans in the same program to have the "
           "same device assignment, got "
        << VectorToString(authoritative_pallas_da) << " and "
        << VectorToString(pallas_op.device_assignment);
  }
  return authoritative_pallas_da;
}

absl::Status VerifySameDeviceSet(absl::Span<const int64_t> resolved_da,
                                 const OpDeviceAssignment& op, size_t index) {
  // A reordering of the same devices is legitimate and is rewritten later.
  ABSL_VLOG(1) << "Op [" << index << "] (" << ToString(op.op_name)
               << ") device assignment " << VectorToString(op.device_assignment)
               << " differs from resolved device assignment "
               << VectorToString(resolved_da);
  const absl::flat_hash_set<int64_t> resolved_devs(resolved_da.begin(),
                                                   resolved_da.end());
  const absl::flat_hash_set<int64_t> op_devs(op.device_assignment.begin(),
                                             op.device_assignment.end());
  // Ops in the same executable must share the same device set. The size check
  // rejects duplicate devices that the set comparison alone would miss.
  TT_RET_CHECK(op.device_assignment.size() == resolved_da.size() &&
                   op_devs == resolved_devs,
               error::kInvalidArgument)
      << "device assignment mismatch across operations: Op [" << index << "] ("
      << ToString(op.op_name) << ") runs on devices "
      << VectorToString(op.device_assignment)
      << " but other ops in the same executable run on devices "
      << VectorToString(resolved_da)
      << ". Collectives over process groups with different members cannot be "
         "compiled into the same executable; dispatch them separately (e.g. "
         "outside a spmd_safe / torch.compile region)";
  return absl::OkStatus();
}

// Tiles a 2D `replica_groups` attribute of an op that runs on one group of a
// partition over every group of that partition: each row of positions in the
// op's group becomes one row per group in `tiles` (the positions of that
// group's members in the program's device assignment). Returns std::nullopt if
// `attr` holds a position outside the group.
std::optional<mlir::DenseIntElementsAttr> TileReplicaGroupsAttr(
    mlir::DenseIntElementsAttr attr,
    absl::Span<const std::vector<int64_t>> tiles) {  // INT_VEC_OK
  auto shaped_type = mlir::dyn_cast<mlir::RankedTensorType>(attr.getType());
  if (!shaped_type || shaped_type.getRank() != 2 || tiles.empty()) {
    return std::nullopt;
  }
  const int64_t num_rows = shaped_type.getDimSize(0);
  const int64_t row_size = shaped_type.getDimSize(1);
  std::vector<int64_t> positions;  // INT_VEC_OK
  for (const llvm::APInt& val : attr.getValues<llvm::APInt>()) {
    positions.push_back(val.getSExtValue());
  }
  std::vector<int64_t> tiled;  // INT_VEC_OK
  tiled.reserve(positions.size() * tiles.size());
  for (const std::vector<int64_t>& tile : tiles) {  // INT_VEC_OK
    for (int64_t p : positions) {
      if (p < 0 || p >= static_cast<int64_t>(tile.size())) {
        return std::nullopt;
      }
      tiled.push_back(tile[p]);
    }
  }
  auto tiled_type = mlir::RankedTensorType::get(
      {num_rows * static_cast<int64_t>(tiles.size()), row_size},
      shaped_type.getElementType());
  return mlir::DenseIntElementsAttr::get(tiled_type, tiled);
}

// The groups of `partition` inside `resolved_da`, each as the positions of its
// members in `resolved_da` in group-rank order. Every group must lie entirely
// inside or entirely outside `resolved_da`, and the groups inside must cover
// it.
absl::StatusOr<std::vector<std::vector<int64_t>>>  // INT_VEC_OK
PartitionTiles(const DeviceIdPartition& partition,
               absl::Span<const int64_t> resolved_da) {
  absl::flat_hash_map<int64_t, int64_t> position;
  for (int64_t p = 0; p < static_cast<int64_t>(resolved_da.size()); ++p) {
    position[resolved_da[p]] = p;
  }
  std::vector<std::vector<int64_t>> tiles;  // INT_VEC_OK
  size_t covered = 0;
  for (const std::vector<int64_t>& group : partition) {  // INT_VEC_OK
    std::vector<int64_t> tile;                           // INT_VEC_OK
    for (int64_t device : group) {
      if (auto it = position.find(device); it != position.end()) {
        tile.push_back(it->second);
      }
    }
    if (tile.empty()) {
      continue;
    }
    TT_RET_CHECK(tile.size() == group.size(), error::kInternal)
        << "process group on devices " << VectorToString(group)
        << " lies partly outside the program's devices "
        << VectorToString(resolved_da);
    covered += tile.size();
    tiles.push_back(std::move(tile));
  }
  TT_RET_CHECK(covered == resolved_da.size(), error::kInternal)
      << "the process groups of an op do not cover the program's devices "
      << VectorToString(resolved_da);
  return tiles;
}

bool IsSameDeviceSet(absl::Span<const int64_t> da,
                     const absl::flat_hash_set<int64_t>& devices) {
  return da.size() == devices.size() &&
         absl::c_all_of(da, [&](int64_t d) { return devices.contains(d); });
}

// Resolves a program whose ops run on different device sets. The program runs
// on the smallest set of devices that holds every op's devices and, for the
// process group of every op, every group of its partition that shares a device
// with the set. Each collective is then tiled over the groups of its partition
// (see RemapReplicaOrderInOperation), as one program per stage of a pipeline
// keeps to that stage. Returns `mismatch` if an op's process group has no
// registered partition.
absl::StatusOr<std::vector<int64_t>>  // INT_VEC_OK
ResolveMixedDeviceAssignment(absl::Span<const OpDeviceAssignment> ops,
                             absl::Span<const int64_t> pallas_da,
                             absl::Status mismatch) {
  absl::flat_hash_set<int64_t> devices(pallas_da.begin(), pallas_da.end());
  std::vector<DeviceIdPartition> partitions;
  absl::flat_hash_set<std::vector<int64_t>> looked_up;  // INT_VEC_OK
  for (const OpDeviceAssignment& op : ops) {
    if (op.device_assignment.empty()) {
      continue;
    }
    devices.insert(op.device_assignment.begin(), op.device_assignment.end());
    if (!looked_up
             .insert(std::vector<int64_t>(  // INT_VEC_OK
                 op.device_assignment.begin(), op.device_assignment.end()))
             .second) {
      continue;
    }
    TT_ASSIGN_OR_RETURN(std::optional<DeviceIdPartition> partition,
                        LookupProcessGroupPartition(op.device_assignment));
    if (!partition.has_value()) {
      return mismatch;
    }
    partitions.push_back(*std::move(partition));
  }
  for (bool grown = true; grown;) {
    grown = false;
    for (const DeviceIdPartition& partition : partitions) {
      for (const std::vector<int64_t>& group : partition) {  // INT_VEC_OK
        if (absl::c_any_of(group,
                           [&](int64_t d) { return devices.contains(d); })) {
          for (int64_t d : group) {
            grown = devices.insert(d).second || grown;
          }
        }
      }
    }
  }
  std::vector<int64_t> resolved_da(devices.begin(),  // INT_VEC_OK
                                   devices.end());
  absl::c_sort(resolved_da);
  if (!pallas_da.empty()) {
    TT_RET_CHECK(IsSameDeviceSet(pallas_da, devices), error::kInvalidArgument)
        << "Pallas kernels in a program that combines process groups must run "
        << "on the program's devices " << VectorToString(resolved_da)
        << ", got " << VectorToString(pallas_da);
    return std::vector<int64_t>(pallas_da.begin(),  // INT_VEC_OK
                                pallas_da.end());
  }
  // Keep the order of an op that already runs on every device, so that it
  // needs no rewriting.
  for (const OpDeviceAssignment& op : ops) {
    if (IsSameDeviceSet(op.device_assignment, devices)) {
      return std::vector<int64_t>(  // INT_VEC_OK
          op.device_assignment.begin(), op.device_assignment.end());
    }
  }
  return resolved_da;
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
  const std::vector<int64_t> pallas_da =  // INT_VEC_OK
      ResolvePallasDeviceAssignment(pallas_ops);
  std::vector<int64_t> resolved_da = pallas_da;  // INT_VEC_OK

  for (size_t i = 0; i < ops.size(); ++i) {
    const auto& op = ops[i];
    if (op.device_assignment.empty()) {
      continue;
    }
    if (resolved_da.empty()) {
      resolved_da.assign(op.device_assignment.begin(),
                         op.device_assignment.end());
    } else if (resolved_da != op.device_assignment) {
      absl::Status same_set = VerifySameDeviceSet(resolved_da, op, i);
      if (!same_set.ok()) {
        return ResolveMixedDeviceAssignment(ops, pallas_da,
                                            std::move(same_set));
      }
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

namespace {

// Rewrites the collectives of an op that runs on one group of a partition so
// that the program, which runs on `resolved_da`, performs them on every group
// of the partition inside `resolved_da`.
absl::Status TileOperationOverPartition(mlir::Operation* absl_nonnull root_op,
                                        absl::Span<const int64_t> op_da,
                                        absl::Span<const int64_t> resolved_da) {
  TT_ASSIGN_OR_RETURN(std::optional<DeviceIdPartition> partition,
                      LookupProcessGroupPartition(op_da));
  TT_RET_CHECK(partition.has_value(), error::kInvalidArgument)
      << "op device assignment " << VectorToString(op_da)
      << " is not a permutation of the resolved device assignment "
      << VectorToString(resolved_da)
      << " and belongs to no known process group";
  TT_ASSIGN_OR_RETURN(
      const std::vector<std::vector<int64_t>> tiles,  // INT_VEC_OK
      PartitionTiles(*partition, resolved_da));
  absl::Status status = absl::OkStatus();
  const auto tile = [&](mlir::Operation* op, mlir::DenseIntElementsAttr attr,
                        std::string_view name) {
    std::optional<mlir::DenseIntElementsAttr> tiled =
        TileReplicaGroupsAttr(attr, tiles);
    if (!tiled.has_value()) {
      status = TT_ERROR(error::kInternal)
               << "cannot tile " << name << " of an op on devices "
               << VectorToString(op_da);
      return attr;
    }
    return *tiled;
  };
  root_op->walk([&](mlir::Operation* op) {
    if (auto rg_attr =
            op->getAttrOfType<mlir::DenseIntElementsAttr>("replica_groups")) {
      op->setAttr("replica_groups", tile(op, rg_attr, "replica_groups"));
    }
    if (auto st_attr = op->getAttrOfType<mlir::DenseIntElementsAttr>(
            "source_target_pairs")) {
      op->setAttr("source_target_pairs",
                  tile(op, st_attr, "source_target_pairs"));
    }
    if (!mlir::isa<mlir::stablehlo::CustomCallOp>(op)) {
      return;
    }
    auto dict_attr = op->getAttrOfType<mlir::DictionaryAttr>("backend_config");
    auto rg_attr = dict_attr
                       ? mlir::dyn_cast_or_null<mlir::DenseIntElementsAttr>(
                             dict_attr.get("replica_groups"))
                       : nullptr;
    if (!rg_attr) {
      return;
    }
    llvm::SmallVector<mlir::NamedAttribute> new_attrs;
    for (const mlir::NamedAttribute& named_attr : dict_attr) {
      if (named_attr.getName() == "replica_groups") {
        new_attrs.emplace_back(named_attr.getName(),
                               tile(op, rg_attr, "replica_groups"));
      } else {
        new_attrs.push_back(named_attr);
      }
    }
    op->setAttr("backend_config",
                mlir::DictionaryAttr::get(op->getContext(), new_attrs));
  });
  return status;
}

}  // namespace

absl::Status RemapReplicaOrderInOperation(
    mlir::Operation* absl_nonnull root_op, absl::Span<const int64_t> op_da,
    absl::Span<const int64_t> resolved_da) {
  if (op_da == resolved_da || op_da.empty()) {
    return absl::OkStatus();
  }

  const std::optional<std::vector<int64_t>>  // INT_VEC_OK
      rank_to_partition = BuildRankToPartition(op_da, resolved_da);
  if (!rank_to_partition.has_value()) {
    return TileOperationOverPartition(root_op, op_da, resolved_da);
  }

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

absl::Status ApplyReplicaOrderRemapping(
    mlir::Operation* absl_nullable first_new_op,
    absl::Span<const mlir::func::FuncOp> new_func_ops,
    const DeferredOp& deferred_op, absl::Span<const int64_t> resolved_da) {
  const absl::Span<const int64_t> op_da = deferred_op.device_assignment();
  if (op_da == resolved_da || op_da.empty()) {
    return absl::OkStatus();
  }

  for (mlir::Operation* curr = first_new_op; curr != nullptr;
       curr = curr->getNextNode()) {
    TT_RETURN_IF_ERROR(RemapReplicaOrderInOperation(curr, op_da, resolved_da));
  }
  for (mlir::func::FuncOp func_op : new_func_ops) {
    TT_RETURN_IF_ERROR(RemapReplicaOrderInOperation(func_op.getOperation(),
                                                    op_da, resolved_da));
  }
  return absl::OkStatus();
}

}  // namespace torch_tpu
