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
#include <utility>
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
#include "csrc/eager/device_assignment.h"
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

// Maps each op rank r to the partition of `resolved_da` that holds device
// `op_da[r]`. Returns std::nullopt unless `op_da` and `resolved_da` have the
// same size and every device of `op_da` is in `resolved_da`.
std::optional<std::vector<int64_t>>  // INT_VEC_OK
BuildRankToPartition(absl::Span<const int64_t> op_da,
                     absl::Span<const int64_t> resolved_da) {
  if (op_da.size() != resolved_da.size()) {
    return std::nullopt;
  }
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
        << "expected Pallas kernels and scans in the same program to have the "
           "same device assignment, got "
        << VectorToString(authoritative_pallas_da) << " and "
        << VectorToString(pallas_op.device_assignment);
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

bool HasSameUniqueDevices(absl::Span<const int64_t> devices,
                          const absl::flat_hash_set<int64_t>& expected) {
  return devices.size() == expected.size() &&
         absl::flat_hash_set<int64_t>(devices.begin(), devices.end()) ==
             expected;
}

bool ContainsSubgroupTile(absl::Span<const int64_t> global_da,
                          absl::Span<const int64_t> op_da) {
  const size_t group_size = op_da.size();
  if (group_size == 0 || global_da.empty() ||
      global_da.size() % group_size != 0) {
    return false;
  }
  for (size_t offset = 0; offset < global_da.size(); offset += group_size) {
    if (global_da.subspan(offset, group_size) == op_da) {
      return true;
    }
  }
  return false;
}

bool ExpandClosedSetUnderOp(const OpDeviceAssignment& op,
                            absl::flat_hash_set<int64_t>& closed_devs) {
  const size_t group_size = op.device_assignment.size();
  const absl::Span<const int64_t> global_da = op.global_device_assignment;
  if (group_size == 0 || global_da.empty() ||
      global_da.size() % group_size != 0) {
    return false;
  }
  const auto in_closed = [&closed_devs](int64_t dev) {
    return closed_devs.contains(dev);
  };
  bool changed = false;
  for (size_t offset = 0; offset < global_da.size(); offset += group_size) {
    const absl::Span<const int64_t> group =
        global_da.subspan(offset, group_size);
    if (absl::c_any_of(group, in_closed) && !absl::c_all_of(group, in_closed)) {
      closed_devs.insert(group.begin(), group.end());
      changed = true;
    }
  }
  return changed;
}

// Returns the smallest superset of `closed_devs` that is a union of complete
// subgroup tiles for every op in `ops` that has a valid
// `global_device_assignment`.
absl::flat_hash_set<int64_t> CloseDeviceSetUnderOpGroups(
    absl::Span<const OpDeviceAssignment> ops,
    absl::flat_hash_set<int64_t> closed_devs) {
  for (bool changed = true; changed;) {
    changed = false;
    for (const OpDeviceAssignment& op : ops) {
      if (ExpandClosedSetUnderOp(op, closed_devs)) {
        changed = true;
      }
    }
  }
  return closed_devs;
}

// Extracts the groups of `global_da` (each of size `group_size`) whose devices
// belong to `closed_devs`, preserving group order and intra-group rank order.
// Returns std::nullopt if `global_da` is not divisible by `group_size`, if any
// group only partially intersects `closed_devs`, or if the extracted groups do
// not cover `closed_devs` with unique devices.
std::optional<std::vector<int64_t>>  // INT_VEC_OK
ExtractGroupsInClosedSet(absl::Span<const int64_t> global_da, size_t group_size,
                         const absl::flat_hash_set<int64_t>& closed_devs) {
  if (group_size == 0 || global_da.empty() ||
      global_da.size() % group_size != 0) {
    return std::nullopt;
  }
  const auto in_closed = [&closed_devs](int64_t dev) {
    return closed_devs.contains(dev);
  };
  std::vector<int64_t> extracted;  // INT_VEC_OK
  extracted.reserve(closed_devs.size());
  for (size_t offset = 0; offset < global_da.size(); offset += group_size) {
    const absl::Span<const int64_t> group =
        global_da.subspan(offset, group_size);
    if (absl::c_all_of(group, in_closed)) {
      extracted.insert(extracted.end(), group.begin(), group.end());
    } else if (!absl::c_none_of(group, in_closed)) {
      return std::nullopt;
    }
  }
  if (!HasSameUniqueDevices(extracted, closed_devs)) {
    return std::nullopt;
  }
  return extracted;
}

std::optional<std::vector<int64_t>>  // INT_VEC_OK
MapLayoutGroupsToPartitions(absl::Span<const int64_t> global_da,
                            size_t group_size,
                            const absl::flat_hash_set<int64_t>& target_devs,
                            absl::Span<const int64_t> target_da) {
  const std::optional<std::vector<int64_t>> active_da =  // INT_VEC_OK
      ExtractGroupsInClosedSet(global_da, group_size, target_devs);
  if (!active_da.has_value()) {
    return std::nullopt;
  }
  return BuildRankToPartition(*active_da, target_da);
}

bool IsOpCompatibleWithClosedSubgroup(
    const OpDeviceAssignment& op,
    const absl::flat_hash_set<int64_t>& closed_devs,
    std::optional<absl::flat_hash_set<int64_t>>& world_devs) {
  if (op.global_device_assignment.empty()) {
    return HasSameUniqueDevices(op.device_assignment, closed_devs);
  }
  if (!ContainsSubgroupTile(op.global_device_assignment,
                            op.device_assignment)) {
    return false;
  }
  if (!world_devs.has_value()) {
    absl::flat_hash_set<int64_t> devs(op.global_device_assignment.begin(),
                                      op.global_device_assignment.end());
    if (devs.size() != op.global_device_assignment.size() ||
        !absl::c_all_of(closed_devs,
                        [&devs](int64_t dev) { return devs.contains(dev); })) {
      return false;
    }
    world_devs = std::move(devs);
    return true;
  }
  return HasSameUniqueDevices(op.global_device_assignment, *world_devs);
}

// Resolves a program whose ops run on different device sets to the smallest
// closed union of subgroup tiles (the least common denominator of the ops'
// layouts, up to the full world) that covers all ops in `ops` and `pallas_da`.
// Each op's collectives are then tiled over the groups of its layout that lie
// in the resolved device assignment (see RewriteAndTileOperationForSubgroup).
// Returns `mismatch` if the ops cannot be reconciled into a common closed
// device set.
absl::StatusOr<std::vector<int64_t>>  // INT_VEC_OK
ResolveClosedSubgroupDeviceAssignment(absl::Span<const OpDeviceAssignment> ops,
                                      absl::Span<const int64_t> pallas_da,
                                      absl::Status mismatch) {
  absl::flat_hash_set<int64_t> seed_devs(pallas_da.begin(), pallas_da.end());
  for (const OpDeviceAssignment& op : ops) {
    seed_devs.insert(op.device_assignment.begin(), op.device_assignment.end());
  }
  const absl::flat_hash_set<int64_t> closed_devs =
      CloseDeviceSetUnderOpGroups(ops, std::move(seed_devs));
  if (!pallas_da.empty() && !HasSameUniqueDevices(pallas_da, closed_devs)) {
    return mismatch;
  }

  std::optional<absl::flat_hash_set<int64_t>> world_devs;
  for (const OpDeviceAssignment& op : ops) {
    if (!op.device_assignment.empty() &&
        !IsOpCompatibleWithClosedSubgroup(op, closed_devs, world_devs)) {
      return mismatch;
    }
  }
  if (!pallas_da.empty()) {
    return std::vector<int64_t>(pallas_da.begin(),  // INT_VEC_OK
                                pallas_da.end());
  }
  const auto first_it = absl::c_find_if(ops, [](const OpDeviceAssignment& op) {
    return !op.device_assignment.empty();
  });
  if (first_it == ops.end()) {
    return std::vector<int64_t>{};  // INT_VEC_OK
  }
  if (first_it->global_device_assignment.empty()) {
    return std::vector<int64_t>(  // INT_VEC_OK
        first_it->device_assignment.begin(), first_it->device_assignment.end());
  }
  std::optional<std::vector<int64_t>> resolved_da =  // INT_VEC_OK
      ExtractGroupsInClosedSet(first_it->global_device_assignment,
                               first_it->device_assignment.size(), closed_devs);
  if (!resolved_da.has_value()) {
    return mismatch;
  }
  return *std::move(resolved_da);
}

const OpDeviceAssignment* absl_nullable FindAnchorOpWithWiderLayout(
    absl::Span<const OpDeviceAssignment> ops, size_t resolved_da_size) {
  const OpDeviceAssignment* anchor = nullptr;
  for (const OpDeviceAssignment& op : ops) {
    if (op.device_assignment.empty()) {
      continue;
    }
    if (!ContainsSubgroupTile(op.global_device_assignment,
                              op.device_assignment)) {
      return nullptr;
    }
    if (anchor == nullptr) {
      if (op.global_device_assignment.size() <= resolved_da_size) {
        return nullptr;
      }
      anchor = &op;
    } else if (op.global_device_assignment.size() !=
               anchor->global_device_assignment.size()) {
      return nullptr;
    }
  }
  return anchor;
}

bool IsCosetIsomorphicAcrossOps(
    absl::Span<const OpDeviceAssignment> ops,
    const absl::flat_hash_set<int64_t>& resolved_devs,
    absl::Span<const int64_t> resolved_da,
    const absl::flat_hash_set<int64_t>& component_devs,
    absl::Span<const int64_t> component_da) {
  for (const OpDeviceAssignment& op : ops) {
    if (op.device_assignment.empty()) {
      continue;
    }
    const size_t group_size = op.device_assignment.size();
    const std::optional<std::vector<int64_t>> ref_map =  // INT_VEC_OK
        MapLayoutGroupsToPartitions(op.global_device_assignment, group_size,
                                    resolved_devs, resolved_da);
    const std::optional<std::vector<int64_t>> comp_map =  // INT_VEC_OK
        MapLayoutGroupsToPartitions(op.global_device_assignment, group_size,
                                    component_devs, component_da);
    if (!ref_map.has_value() || comp_map != ref_map) {
      return false;
    }
  }
  return true;
}

// Resolves the wider layout device assignment across `ops` when the traversal
// resolved to a subgroup `resolved_da` and does not contain a Pallas kernel.
// Returns empty if no wider coset layout applies across all ops.
std::vector<int64_t>  // INT_VEC_OK
ResolveGlobalDeviceAssignmentForTraversal(
    absl::Span<const OpDeviceAssignment> ops,
    absl::Span<const int64_t> resolved_da) {
  const OpDeviceAssignment* absl_nullable anchor =
      FindAnchorOpWithWiderLayout(ops, resolved_da.size());
  if (anchor == nullptr) {
    return {};
  }
  const absl::Span<const int64_t> anchor_global_da =
      anchor->global_device_assignment;
  const size_t anchor_group_size = anchor->device_assignment.size();
  const absl::flat_hash_set<int64_t> resolved_devs(resolved_da.begin(),
                                                   resolved_da.end());
  std::vector<int64_t> layout_da;  // INT_VEC_OK
  layout_da.reserve(anchor_global_da.size());
  absl::flat_hash_set<int64_t> visited;
  visited.reserve(anchor_global_da.size());
  for (int64_t seed_dev : anchor_global_da) {
    if (visited.contains(seed_dev)) {
      continue;
    }
    const absl::flat_hash_set<int64_t> component_devs =
        CloseDeviceSetUnderOpGroups(ops, {seed_dev});
    const std::optional<std::vector<int64_t>> component_da =  // INT_VEC_OK
        component_devs.size() == resolved_da.size()
            ? ExtractGroupsInClosedSet(anchor_global_da, anchor_group_size,
                                       component_devs)
            : std::nullopt;
    if (!component_da.has_value() ||
        !IsCosetIsomorphicAcrossOps(ops, resolved_devs, resolved_da,
                                    component_devs, *component_da)) {
      return {};
    }
    visited.insert(component_devs.begin(), component_devs.end());
    layout_da.insert(layout_da.end(), component_da->begin(),
                     component_da->end());
  }
  return layout_da.size() == anchor_global_da.size()
             ? layout_da
             : std::vector<int64_t>{};  // INT_VEC_OK
}

// Tiles a 2D attribute of positions (`replica_groups` or
// `source_target_pairs`) over `tiles`: each tile maps every position to a
// partition, and the rows are repeated once per tile. If `sort_rows`, the
// rows are then sorted by their first element, as in RewriteReplicaGroupsAttr.
// Returns std::nullopt if `attr` is not 2D or holds a position outside a tile.
std::optional<mlir::DenseIntElementsAttr> TileReplicaGroupsAttr(
    mlir::DenseIntElementsAttr attr,
    absl::Span<const std::vector<int64_t>> tiles,  // INT_VEC_OK
    bool sort_rows) {
  auto shaped_type = mlir::dyn_cast<mlir::RankedTensorType>(attr.getType());
  if (!shaped_type || shaped_type.getRank() != 2 || tiles.empty()) {
    return std::nullopt;
  }
  const int64_t row_size = shaped_type.getDimSize(1);
  std::vector<std::vector<int64_t>> rows;  // INT_VEC_OK
  rows.reserve(shaped_type.getDimSize(0) * tiles.size());
  for (const std::vector<int64_t>& tile : tiles) {  // INT_VEC_OK
    std::vector<int64_t> row;                       // INT_VEC_OK
    for (const llvm::APInt& val : attr.getValues<llvm::APInt>()) {
      const int64_t p = val.getSExtValue();
      if (p < 0 || p >= static_cast<int64_t>(tile.size())) {
        return std::nullopt;
      }
      row.push_back(tile[p]);
      if (static_cast<int64_t>(row.size()) == row_size) {
        rows.push_back(std::move(row));
        row.clear();
      }
    }
  }
  if (sort_rows && row_size > 0) {
    std::sort(rows.begin(), rows.end(),
              [](const auto& a, const auto& b) { return a[0] < b[0]; });
  }
  std::vector<int64_t> tiled;  // INT_VEC_OK
  tiled.reserve(shaped_type.getNumElements() * tiles.size());
  for (const auto& row : rows) {
    tiled.insert(tiled.end(), row.begin(), row.end());
  }
  auto tiled_type = mlir::RankedTensorType::get(
      {shaped_type.getDimSize(0) * static_cast<int64_t>(tiles.size()),
       row_size},
      shaped_type.getElementType());
  return mlir::DenseIntElementsAttr::get(tiled_type, tiled);
}

std::optional<std::vector<std::vector<int64_t>>>  // INT_VEC_OK
BuildSubgroupTiles(absl::Span<const int64_t> op_da,
                   absl::Span<const int64_t> global_da,
                   absl::Span<const int64_t> resolved_da) {
  if (!ContainsSubgroupTile(global_da, op_da)) {
    return std::nullopt;
  }
  const absl::flat_hash_set<int64_t> resolved_devs(resolved_da.begin(),
                                                   resolved_da.end());
  if (resolved_devs.size() != resolved_da.size() ||
      !resolved_devs.contains(op_da[0])) {
    return std::nullopt;
  }
  const size_t group_size = op_da.size();
  const std::optional<std::vector<int64_t>>  // INT_VEC_OK
      rank_to_partition = MapLayoutGroupsToPartitions(
          global_da, group_size, resolved_devs, resolved_da);
  if (!rank_to_partition.has_value()) {
    return std::nullopt;
  }
  const size_t num_groups = rank_to_partition->size() / group_size;
  std::vector<std::vector<int64_t>> tiles(num_groups);  // INT_VEC_OK
  for (size_t g = 0; g < num_groups; ++g) {
    tiles[g].assign(rank_to_partition->begin() + g * group_size,
                    rank_to_partition->begin() + (g + 1) * group_size);
  }
  return tiles;
}

// Rewrites the collectives of an op that runs on one group of a layout, whose
// groups are `global_da` group by group, so that the program, which runs on
// `resolved_da` (a union of one or more groups of `global_da`), performs them
// on every group of the layout within `resolved_da`.
absl::Status TileOperationOverGroups(mlir::Operation* absl_nonnull root_op,
                                     absl::Span<const int64_t> op_da,
                                     absl::Span<const int64_t> global_da,
                                     absl::Span<const int64_t> resolved_da) {
  const std::optional<std::vector<std::vector<int64_t>>>  // INT_VEC_OK
      tiles = BuildSubgroupTiles(op_da, global_da, resolved_da);
  TT_RET_CHECK(tiles.has_value(), error::kInvalidArgument)
      << "op device assignment " << VectorToString(op_da)
      << " is not a permutation of the resolved device assignment "
      << VectorToString(resolved_da);

  absl::Status status = absl::OkStatus();
  const auto tile = [&]  // REF_CAPTURE_OK=called locally
      (mlir::DenseIntElementsAttr attr, std::string_view name, bool sort_rows) {
        std::optional<mlir::DenseIntElementsAttr> tiled =
            TileReplicaGroupsAttr(attr, *tiles, sort_rows);
        if (!tiled.has_value()) {
          status = TT_ERROR(  // ERROR_COV_INFEASIBLE=collectives emit positions
                              // inside their group
                       error::kInternal)
                   << "cannot tile " << name << " of an op on devices "
                   << VectorToString(op_da);
          return attr;
        }
        return *tiled;
      };
  root_op->walk(
      [&](mlir::Operation* op) {  // REF_CAPTURE_OK=called synchronously
        if (auto rg_attr = op->getAttrOfType<mlir::DenseIntElementsAttr>(
                "replica_groups")) {
          op->setAttr("replica_groups",
                      tile(rg_attr, "replica_groups", /*sort_rows=*/true));
        }
        if (auto st_attr = op->getAttrOfType<mlir::DenseIntElementsAttr>(
                "source_target_pairs")) {
          op->setAttr(
              "source_target_pairs",
              tile(st_attr, "source_target_pairs", /*sort_rows=*/false));
        }
        if (!mlir::isa<mlir::stablehlo::CustomCallOp>(op)) {
          return;
        }
        auto dict_attr =
            op->getAttrOfType<mlir::DictionaryAttr>("backend_config");
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
            new_attrs.emplace_back(
                named_attr.getName(),
                tile(rg_attr, "replica_groups", /*sort_rows=*/true));
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
        return ResolveClosedSubgroupDeviceAssignment(ops, pallas_da,
                                                     std::move(same_set));
      }
    }
  }

  return resolved_da;
}

absl::StatusOr<DeviceAssignment> ResolveDeviceAssignmentForTraversal(
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
    if (op->contains_pallas_kernel() && !op->device_assignment().empty()) {
      pallas_ops.push_back(PallasDeviceAssignment{op->device_assignment()});
    } else {
      ops.push_back(OpDeviceAssignment{
          .op_name = op->op_name(),
          .device_assignment = op->device_assignment(),
          .global_device_assignment = op->global_device_assignment(),
      });
    }
  }
  TT_ASSIGN_OR_RETURN(std::vector<int64_t> local_da,  // INT_VEC_OK
                      ResolveDeviceAssignment(ops, pallas_ops));
  const bool contains_pallas_kernel = !pallas_ops.empty();
  std::vector<int64_t> global_da =  // INT_VEC_OK
      (!local_da.empty() && !contains_pallas_kernel)
          ? ResolveGlobalDeviceAssignmentForTraversal(ops, local_da)
          : std::vector<int64_t>{};  // INT_VEC_OK
  return DeviceAssignment(std::move(local_da), std::move(global_da),
                          contains_pallas_kernel);
}

absl::Status RewriteAndTileOperationForSubgroup(
    mlir::Operation* absl_nonnull root_op, absl::Span<const int64_t> op_da,
    absl::Span<const int64_t> resolved_da,
    absl::Span<const int64_t> global_da) {
  if (op_da == resolved_da || op_da.empty()) {
    return absl::OkStatus();
  }

  const std::optional<std::vector<int64_t>>  // INT_VEC_OK
      rank_to_partition = BuildRankToPartition(op_da, resolved_da);
  if (!rank_to_partition.has_value()) {
    return TileOperationOverGroups(root_op, op_da, global_da, resolved_da);
  }

  root_op->walk(
      [&](mlir::Operation* op) {  // REF_CAPTURE_OK=called synchronously
        if (auto rg_attr = op->getAttrOfType<mlir::DenseIntElementsAttr>(
                "replica_groups")) {
          op->setAttr("replica_groups",
                      RewriteReplicaGroupsAttr(rg_attr, *rank_to_partition));
        }
        if (auto st_attr = op->getAttrOfType<mlir::DenseIntElementsAttr>(
                "source_target_pairs")) {
          op->setAttr("source_target_pairs", RewriteSourceTargetPairsAttr(
                                                 st_attr, *rank_to_partition));
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
  const absl::Span<const int64_t> global_da =
      deferred_op.global_device_assignment();

  for (mlir::Operation* curr = first_new_op; curr != nullptr;
       curr = curr->getNextNode()) {
    TT_RETURN_IF_ERROR(RewriteAndTileOperationForSubgroup(
        curr, op_da, resolved_da, global_da));
  }
  for (mlir::func::FuncOp func_op : new_func_ops) {
    TT_RETURN_IF_ERROR(RewriteAndTileOperationForSubgroup(
        func_op.getOperation(), op_da, resolved_da, global_da));
  }
  return absl::OkStatus();
}

}  // namespace torch_tpu
