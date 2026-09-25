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
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "csrc/common/error_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/ops/op_names.h"
#include "csrc/pjrt/pjrt_state.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

constexpr std::string_view kConflictingDeviceAssignmentsErrorPrefix =
    "Disjoint or conflicting device assignments detected in the same "
    "compiled module. Operations belonging to different process groups or "
    "conflicting Pallas meshes cannot be compiled together into a single "
    "XLA executable\n";

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

std::vector<int64_t> BuildCosetShifts(  // INT_VEC_OK
    absl::Span<const int64_t> norm_vals, int64_t full_size) {
  const int64_t sub_size = norm_vals.size();
  if (sub_size == 0 || full_size == 0 || full_size % sub_size != 0) {
    return {};
  }
  std::vector<bool> used(full_size, false);
  std::vector<int64_t> shifts;  // INT_VEC_OK
  shifts.reserve(full_size / sub_size);
  for (int64_t shift = 0; shift < full_size; ++shift) {
    bool can_place = true;
    for (int64_t p : norm_vals) {
      if (p < 0 || p + shift >= full_size || used[p + shift]) {
        can_place = false;
        break;
      }
    }
    if (can_place) {
      for (int64_t p : norm_vals) {
        used[p + shift] = true;
      }
      shifts.push_back(shift);
    }
  }
  if (static_cast<int64_t>(shifts.size()) * sub_size != full_size) {
    return {};
  }
  return shifts;
}

bool IsValidSubgroupCoset(absl::Span<const int64_t> op_da,
                          absl::Span<const int64_t> pre_padded_resolved_da) {
  const int64_t full_size = pre_padded_resolved_da.size();
  if (op_da.empty() || full_size == 0) {
    return false;
  }
  absl::flat_hash_map<int64_t, int64_t> dev_to_partition;
  for (int64_t p = 0; p < full_size; ++p) {
    dev_to_partition[pre_padded_resolved_da[p]] = p;
  }
  std::vector<int64_t> sub_partitions;  // INT_VEC_OK
  sub_partitions.reserve(op_da.size());
  for (int64_t dev_id : op_da) {
    auto it = dev_to_partition.find(dev_id);
    if (it != dev_to_partition.end()) {
      sub_partitions.push_back(it->second);
    }
  }
  if (sub_partitions.empty()) {
    return true;
  }
  std::sort(sub_partitions.begin(), sub_partitions.end());
  const int64_t sub_size = sub_partitions.size();
  if (full_size % sub_size != 0) {
    return false;
  }
  const int64_t min_p = sub_partitions[0];
  std::vector<int64_t> norm_vals(sub_size);  // INT_VEC_OK
  for (int64_t i = 0; i < sub_size; ++i) {
    norm_vals[i] = sub_partitions[i] - min_p;
  }
  const std::vector<int64_t> shifts =  // INT_VEC_OK
      BuildCosetShifts(norm_vals, full_size);
  return !shifts.empty() &&
         std::find(shifts.begin(), shifts.end(), min_p) != shifts.end();
}

mlir::DenseIntElementsAttr RewriteAndTileReplicaGroupsAttr(
    mlir::DenseIntElementsAttr attr, absl::Span<const int64_t> op_da,
    absl::Span<const int64_t> resolved_da) {
  auto shaped_type = mlir::dyn_cast<mlir::RankedTensorType>(attr.getType());
  const int64_t sub_size = op_da.size();
  const int64_t full_size = resolved_da.size();
  if (!shaped_type || shaped_type.getRank() != 2 || sub_size == 0 ||
      full_size == 0 || full_size % sub_size != 0) {
    return attr;
  }
  const int64_t num_groups = shaped_type.getDimSize(0);
  const int64_t group_size = shaped_type.getDimSize(1);
  if (num_groups * group_size != sub_size) {
    return attr;
  }

  // Map each rank r in [0, sub_size) to its partition index in resolved_da.
  const std::optional<std::vector<int64_t>> rank_to_partition =  // INT_VEC_OK
      BuildRankToPartition(op_da, resolved_da);
  if (!rank_to_partition.has_value()) {
    return attr;
  }

  // Rewrite base groups using partition indices.
  std::vector<int64_t> base_vals;  // INT_VEC_OK
  base_vals.reserve(sub_size);
  for (const llvm::APInt& val : attr.getValues<llvm::APInt>()) {
    int64_t r = val.getSExtValue();
    if (r < 0 || r >= sub_size) return attr;
    base_vals.push_back((*rank_to_partition)[r]);
  }

  // Normalize the base pattern by subtracting min_p so coset shifts never wrap
  // modulo full_size (which would invert physical coordinate traversal order
  // and fail XLA's VerifyMeshInfos / ReplicaGroupsStrideable checks).
  const int64_t min_p = *std::min_element(base_vals.begin(), base_vals.end());
  std::vector<int64_t> norm_vals(sub_size);  // INT_VEC_OK
  for (int64_t i = 0; i < sub_size; ++i) {
    norm_vals[i] = base_vals[i] - min_p;
  }

  // Complete the partition of [0 .. full_size - 1] across num_tiles.
  std::vector<std::vector<int64_t>> all_groups;  // INT_VEC_OK
  std::vector<bool> used(full_size, false);

  // First place the exact active base groups so the calling rank's group order
  // is unconditionally preserved.
  for (int64_t g = 0; g < num_groups; ++g) {
    std::vector<int64_t> grp(  // INT_VEC_OK
        base_vals.begin() + g * group_size,
        base_vals.begin() + (g + 1) * group_size);
    for (int64_t p : grp) used[p] = true;
    all_groups.push_back(std::move(grp));
  }

  // Next place non-wrapping translated cosets (norm_vals + shift) so every
  // group shares the exact same physical minor-to-major orientation.
  for (int64_t shift = 0; shift < full_size; ++shift) {
    bool can_place = true;
    for (int64_t p : norm_vals) {
      if (p + shift >= full_size || used[p + shift]) {
        can_place = false;
        break;
      }
    }
    if (can_place) {
      for (int64_t g = 0; g < num_groups; ++g) {
        std::vector<int64_t> grp;  // INT_VEC_OK
        grp.reserve(group_size);
        for (int64_t i = 0; i < group_size; ++i) {
          int64_t p = norm_vals[g * group_size + i] + shift;
          used[p] = true;
          grp.push_back(p);
        }
        all_groups.push_back(std::move(grp));
      }
    }
  }

  // Fallback for any remaining unused partitions if non-uniform.
  std::vector<int64_t> remaining;  // INT_VEC_OK
  for (int64_t p = 0; p < full_size; ++p) {
    if (!used[p]) remaining.push_back(p);
  }
  for (size_t i = 0; i + group_size <= remaining.size(); i += group_size) {
    all_groups.emplace_back(remaining.begin() + i,
                            remaining.begin() + i + group_size);
  }

  // Canonically sort groups by their first element so all ranks emit the same
  // table.
  std::sort(all_groups.begin(), all_groups.end(),
            [](const std::vector<int64_t>& a,    // INT_VEC_OK
               const std::vector<int64_t>& b) {  // INT_VEC_OK
              return a.front() < b.front();
            });

  std::vector<int64_t> flat_vals;  // INT_VEC_OK
  flat_vals.reserve(full_size);
  for (const auto& grp : all_groups) {
    flat_vals.insert(flat_vals.end(), grp.begin(), grp.end());
  }

  const int64_t total_groups = all_groups.size();
  auto new_type = mlir::RankedTensorType::get({total_groups, group_size},
                                              shaped_type.getElementType());
  return mlir::DenseIntElementsAttr::get(new_type, flat_vals);
}

mlir::DenseIntElementsAttr RewriteAndTileSourceTargetPairsAttr(
    mlir::DenseIntElementsAttr attr, absl::Span<const int64_t> op_da,
    absl::Span<const int64_t> resolved_da) {
  auto shaped_type = mlir::dyn_cast<mlir::RankedTensorType>(attr.getType());
  const int64_t sub_size = op_da.size();
  const int64_t full_size = resolved_da.size();
  if (!shaped_type || shaped_type.getRank() != 2 ||
      shaped_type.getDimSize(1) != 2 || sub_size == 0 || full_size == 0 ||
      full_size % sub_size != 0) {
    return attr;
  }
  const int64_t num_pairs = shaped_type.getDimSize(0);
  if (num_pairs == 0 || num_pairs > sub_size) {
    return attr;
  }

  const std::optional<std::vector<int64_t>> rank_to_partition =  // INT_VEC_OK
      BuildRankToPartition(op_da, resolved_da);
  if (!rank_to_partition.has_value()) {
    return attr;
  }

  std::vector<int64_t> base_vals;  // INT_VEC_OK
  base_vals.reserve(num_pairs * 2);
  for (const llvm::APInt& val : attr.getValues<llvm::APInt>()) {
    const int64_t v = val.getSExtValue();
    if (v < 0 || v >= sub_size) {
      return attr;
    }
    base_vals.push_back((*rank_to_partition)[v]);
  }

  if (full_size == sub_size) {
    return mlir::DenseIntElementsAttr::get(shaped_type, base_vals);
  }

  const int64_t min_p =
      *std::min_element(rank_to_partition->begin(), rank_to_partition->end());
  std::vector<int64_t> norm_sub_partitions(sub_size);  // INT_VEC_OK
  for (int64_t i = 0; i < sub_size; ++i) {
    norm_sub_partitions[i] = (*rank_to_partition)[i] - min_p;
  }
  std::vector<int64_t> norm_pair_vals(base_vals.size());  // INT_VEC_OK
  for (size_t i = 0; i < base_vals.size(); ++i) {
    norm_pair_vals[i] = base_vals[i] - min_p;
  }

  struct SourceTargetPair {
    int64_t source;
    int64_t target;
  };
  std::vector<SourceTargetPair> all_pairs;
  std::vector<bool> used(full_size, false);

  for (int64_t p : *rank_to_partition) {
    used[p] = true;
  }
  for (int64_t i = 0; i < num_pairs; ++i) {
    all_pairs.push_back({base_vals[2 * i], base_vals[2 * i + 1]});
  }

  for (int64_t shift = 0; shift < full_size; ++shift) {
    bool can_place = true;
    for (int64_t p : norm_sub_partitions) {
      if (p + shift >= full_size || used[p + shift]) {
        can_place = false;
        break;
      }
    }
    if (can_place) {
      for (int64_t p : norm_sub_partitions) {
        used[p + shift] = true;
      }
      for (int64_t i = 0; i < num_pairs; ++i) {
        all_pairs.push_back(
            {norm_pair_vals[2 * i] + shift, norm_pair_vals[2 * i + 1] + shift});
      }
    }
  }

  std::sort(all_pairs.begin(), all_pairs.end(),
            [](const SourceTargetPair& a, const SourceTargetPair& b) {
              if (a.source != b.source) {
                return a.source < b.source;
              }
              return a.target < b.target;
            });

  std::vector<int64_t> flat_vals;  // INT_VEC_OK
  flat_vals.reserve(all_pairs.size() * 2);
  for (const SourceTargetPair& pair : all_pairs) {
    flat_vals.push_back(pair.source);
    flat_vals.push_back(pair.target);
  }
  const int64_t total_pairs = all_pairs.size();
  auto new_type = mlir::RankedTensorType::get({total_pairs, 2},
                                              shaped_type.getElementType());
  return mlir::DenseIntElementsAttr::get(new_type, flat_vals);
}

}  // namespace

bool IsCompatibleDeviceAssignmentTile(absl::Span<const int64_t> d_sub,
                                      absl::Span<const int64_t> d_full) {
  const size_t m = d_sub.size();
  const size_t n = d_full.size();
  if (m == 0 || n == 0 || n % m != 0) {
    return false;
  }
  for (size_t offset = 0; offset < n; offset += m) {
    if (d_full.subspan(offset, m) == d_sub) {
      return true;
    }
  }
  return false;
}

absl::StatusOr<std::vector<int64_t>>  // INT_VEC_OK
ResolveDeviceAssignment(absl::Span<const OpDeviceAssignment> ops) {
  // --- STAGE 1: Merge Compatible Pallas Kernels (OpName::kCustomKernel) ---
  std::vector<int64_t> pallas_da;  // INT_VEC_OK
  for (const OpDeviceAssignment& op : ops) {
    if (op.op_name != OpName::kCustomKernel || op.device_assignment.empty()) {
      continue;
    }
    if (op.device_assignment.size() > pallas_da.size()) {
      pallas_da.assign(op.device_assignment.begin(),
                       op.device_assignment.end());
    }
  }
  for (const OpDeviceAssignment& op : ops) {
    if (op.op_name != OpName::kCustomKernel || op.device_assignment.empty()) {
      continue;
    }
    if (!IsCompatibleDeviceAssignmentTile(op.device_assignment, pallas_da)) {
      ABSL_LOG(ERROR)
          << "Disjoint or conflicting device assignments in Pallas kernels "
             "within the same compiled module: op_da=["
          << absl::StrJoin(op.device_assignment, ", ") << "] (size "
          << op.device_assignment.size()
          << ") is not a compatible contiguous tile of target Pallas mesh "
             "pallas_da=["
          << absl::StrJoin(pallas_da, ", ") << "] (size " << pallas_da.size()
          << ").";
      return TT_ERROR(error::kInvalidArgument)
             << kConflictingDeviceAssignmentsErrorPrefix
             << "Pallas kernel op_da=["
             << absl::StrJoin(op.device_assignment, ", ") << "] vs pallas_da=["
             << absl::StrJoin(pallas_da, ", ") << "]";
    }
  }

  // --- STAGE 2: Enter StableHLO Space & Merge to Larger Compatible DAs ---
  std::vector<int64_t> resolved_da = pallas_da;  // INT_VEC_OK
  if (resolved_da.empty()) {
    for (const OpDeviceAssignment& op : ops) {
      if (op.op_name == OpName::kCustomKernel || op.device_assignment.empty()) {
        continue;
      }
      if (op.device_assignment.size() > resolved_da.size()) {
        resolved_da.assign(op.device_assignment.begin(),
                           op.device_assignment.end());
      }
    }
  } else {
    for (const OpDeviceAssignment& op : ops) {
      if (op.op_name == OpName::kCustomKernel || op.device_assignment.empty()) {
        continue;
      }
      if (op.device_assignment.size() > resolved_da.size() &&
          IsCompatibleDeviceAssignmentTile(resolved_da, op.device_assignment)) {
        resolved_da.assign(op.device_assignment.begin(),
                           op.device_assignment.end());
      }
    }
  }

  if (resolved_da.empty()) {
    return resolved_da;
  }

  // --- STAGE 3: Validate Smaller / Orthogonal StableHLO Collectives ---
  const auto* const client = PjrtBackend::GetInstance().GetClient();
  bool should_pad_to_client = false;
  if (client != nullptr && resolved_da.size() < client->devices().size()) {
    absl::flat_hash_set<int64_t> client_device_ids;
    client_device_ids.reserve(client->devices().size());
    for (const auto* d : client->devices()) {
      client_device_ids.insert(d->id());
    }
    should_pad_to_client = std::all_of(
        resolved_da.begin(), resolved_da.end(),
        [&](int64_t dev_id) { return client_device_ids.contains(dev_id); });
  }

  std::vector<int64_t> final_da = resolved_da;  // INT_VEC_OK
  if (should_pad_to_client) {
    absl::flat_hash_set<int64_t> present(final_da.begin(), final_da.end());
    for (const auto* d : client->devices()) {
      if (present.insert(d->id()).second) {
        final_da.push_back(d->id());
      }
    }
  }

  const absl::flat_hash_set<int64_t> valid_devices(final_da.begin(),
                                                   final_da.end());
  for (const OpDeviceAssignment& op : ops) {
    if (op.device_assignment.empty()) continue;
    if (final_da.size() % op.device_assignment.size() != 0) {
      return TT_ERROR(error::kInvalidArgument)
             << kConflictingDeviceAssignmentsErrorPrefix << "op_da=["
             << absl::StrJoin(op.device_assignment, ", ")
             << "] size does not divide resolved mesh size " << final_da.size();
    }
    for (int64_t dev_id : op.device_assignment) {
      if (!valid_devices.contains(dev_id)) {
        return TT_ERROR(error::kInvalidArgument)
               << kConflictingDeviceAssignmentsErrorPrefix << "device ID "
               << dev_id << " in op_da=["
               << absl::StrJoin(op.device_assignment, ", ")
               << "] is not present in resolved_da=["
               << absl::StrJoin(final_da, ", ") << "]";
      }
    }
    if (!IsValidSubgroupCoset(op.device_assignment, resolved_da)) {
      return TT_ERROR(error::kInvalidArgument)
             << kConflictingDeviceAssignmentsErrorPrefix << "op_da=["
             << absl::StrJoin(op.device_assignment, ", ")
             << "] is not a valid coset of resolved_da=["
             << absl::StrJoin(resolved_da, ", ") << "]";
    }
  }

  // --- STAGE 4: Final Tail Padding to World Size ---
  resolved_da = std::move(final_da);

  return resolved_da;
}

absl::StatusOr<std::vector<int64_t>>  // INT_VEC_OK
ResolveDeviceAssignmentForTraversal(
    absl::Span<const SharedDeviceBufferList> execution_order) {
  llvm::SmallVector<OpDeviceAssignment, 8> ops;
  ops.reserve(execution_order.size());
  for (const SharedDeviceBufferList& node : execution_order) {
    const auto& op = node->deferred_op();
    if (!op) {
      continue;
    }
    ops.push_back(OpDeviceAssignment{
        .op_name = op->op_name(),
        .device_assignment = op->device_assignment(),
    });
  }
  return ResolveDeviceAssignment(ops);
}

void RewriteAndTileOperationForSubgroup(mlir::Operation* absl_nonnull root_op,
                                        absl::Span<const int64_t> op_da,
                                        absl::Span<const int64_t> resolved_da) {
  const int64_t sub_size = op_da.size();
  const int64_t full_size = resolved_da.size();
  llvm::SmallVector<mlir::stablehlo::PartitionIdOp> partition_id_ops;
  root_op->walk([&](mlir::Operation* op) {
    if (auto rg_attr =
            op->getAttrOfType<mlir::DenseIntElementsAttr>("replica_groups")) {
      op->setAttr("replica_groups",
                  RewriteAndTileReplicaGroupsAttr(rg_attr, op_da, resolved_da));
    }
    if (auto st_attr = op->getAttrOfType<mlir::DenseIntElementsAttr>(
            "source_target_pairs")) {
      op->setAttr("source_target_pairs", RewriteAndTileSourceTargetPairsAttr(
                                             st_attr, op_da, resolved_da));
    }
    if (mlir::isa<mlir::stablehlo::CustomCallOp>(op)) {
      if (auto dict_attr =
              op->getAttrOfType<mlir::DictionaryAttr>("backend_config")) {
        if (auto rg_attr = mlir::dyn_cast_or_null<mlir::DenseIntElementsAttr>(
                dict_attr.get("replica_groups"))) {
          llvm::SmallVector<mlir::NamedAttribute> new_attrs;
          for (const mlir::NamedAttribute& named_attr : dict_attr) {
            if (named_attr.getName() == "replica_groups") {
              new_attrs.emplace_back(
                  named_attr.getName(),
                  RewriteAndTileReplicaGroupsAttr(rg_attr, op_da, resolved_da));
            } else {
              new_attrs.push_back(named_attr);
            }
          }
          op->setAttr("backend_config",
                      mlir::DictionaryAttr::get(op->getContext(), new_attrs));
        }
      }
    }
    if (auto pid_op = mlir::dyn_cast<mlir::stablehlo::PartitionIdOp>(op)) {
      partition_id_ops.push_back(pid_op);
    }
  });

  if (full_size > sub_size && sub_size > 0) {
    for (mlir::stablehlo::PartitionIdOp pid_op : partition_id_ops) {
      mlir::Block::iterator next_it(pid_op);
      ++next_it;
      mlir::OpBuilder builder(pid_op->getBlock(), next_it);
      auto pid_type = mlir::cast<mlir::RankedTensorType>(pid_op.getType());
      auto m_attr = mlir::DenseIntElementsAttr::get(
          pid_type, llvm::APInt(32, static_cast<uint64_t>(sub_size), false));
      auto m_const = mlir::stablehlo::ConstantOp::create(
          builder, pid_op.getLoc(), pid_type, m_attr);
      auto rem_op = mlir::stablehlo::RemOp::create(builder, pid_op.getLoc(),
                                                   pid_type, pid_op.getResult(),
                                                   m_const.getResult());
      pid_op.getResult().replaceAllUsesExcept(rem_op.getResult(), rem_op);
    }
  }
}

void ApplySubgroupTilingOrReplicaRewriting(
    mlir::Operation* absl_nullable first_new_op,
    absl::Span<const mlir::func::FuncOp> new_func_ops,
    const DeferredOp& deferred_op, absl::Span<const int64_t> resolved_da) {
  const absl::Span<const int64_t> op_da = deferred_op.device_assignment();
  const int64_t op_da_size = op_da.size();
  const int64_t full_da_size = resolved_da.size();
  const bool needs_tiling = op_da_size > 0 && full_da_size >= op_da_size &&
                            full_da_size % op_da_size == 0 &&
                            (full_da_size > op_da_size ||
                             deferred_op.device_assignment() != resolved_da);
  if (!needs_tiling) {
    return;
  }

  for (mlir::Operation* curr = first_new_op; curr != nullptr;
       curr = curr->getNextNode()) {
    RewriteAndTileOperationForSubgroup(curr, op_da, resolved_da);
  }
  for (mlir::func::FuncOp func_op : new_func_ops) {
    RewriteAndTileOperationForSubgroup(func_op.getOperation(), op_da,
                                       resolved_da);
  }
}

}  // namespace torch_tpu
