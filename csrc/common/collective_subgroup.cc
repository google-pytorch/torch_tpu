// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "csrc/common/collective_subgroup.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "csrc/common/error_utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "xla/array.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/computation_placer.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

namespace {

constexpr char kReplicaGroupsAttr[] = "replica_groups";
constexpr char kSourceTargetPairsAttr[] = "source_target_pairs";
constexpr char kShardingAttr[] = "mhlo.sharding";
constexpr char kSpmdOutputShardingAttr[] = "mhlo.spmd_output_sharding";
constexpr char kSpmdParametersShardingsAttr[] =
    "mhlo.spmd_parameters_shardings";
constexpr char kCallTargetNameAttr[] = "call_target_name";
// Custom calls whose device groups live inside their backend config, which
// this rewrite does not reach.
constexpr char kRaggedAllToAllTarget[] = "ragged_all_to_all";

std::string GroupString(absl::Span<const int64_t> group) {
  return absl::StrCat("[", absl::StrJoin(group, ", "), "]");
}

// Position of `id` in `group`, or -1.
int64_t IndexInGroup(absl::Span<const int64_t> group, int64_t id) {
  const auto it = absl::c_find(group, id);
  return it == group.end() ? -1 : it - group.begin();
}

// Reads a [rows, cols] i64 attribute into a row-major vector.
absl::StatusOr<std::vector<std::vector<int64_t>>> ReadRows(  // INT_VEC_OK
    mlir::DenseIntElementsAttr attr) {
  const auto type = mlir::cast<mlir::ShapedType>(attr.getType());
  TT_RET_CHECK(type.getRank() == 2, error::kInternal)
      << "expected a rank-2 collective device attribute, got rank "
      << type.getRank();
  const int64_t rows = type.getDimSize(0);
  const int64_t cols = type.getDimSize(1);
  std::vector<std::vector<int64_t>> result(
      rows, std::vector<int64_t>(cols));  // INT_VEC_OK
  auto it = attr.value_begin<int64_t>();
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      result[r][c] = *it++;
    }
  }
  return result;
}

mlir::DenseIntElementsAttr MakeRows(
    mlir::MLIRContext* context,
    const std::vector<std::vector<int64_t>>& rows,  // INT_VEC_OK
    int64_t cols) {
  std::vector<int64_t> flat;  // INT_VEC_OK
  flat.reserve(rows.size() * cols);
  for (const auto& row : rows) {
    flat.insert(flat.end(), row.begin(), row.end());
  }
  const auto type =
      mlir::RankedTensorType::get({static_cast<int64_t>(rows.size()), cols},
                                  mlir::IntegerType::get(context, 64));
  return mlir::DenseIntElementsAttr::get(type, flat);
}

absl::Status RewriteReplicaGroups(mlir::Operation* op,
                                  absl::Span<const int64_t> group) {
  auto attr = op->getAttrOfType<mlir::DenseIntElementsAttr>(kReplicaGroupsAttr);
  if (attr == nullptr) {
    return absl::OkStatus();
  }
  TT_ASSIGN_OR_RETURN(const auto rows, ReadRows(attr));
  bool found = false;
  for (const auto& row : rows) {
    const bool overlaps = absl::c_any_of(
        row, [&](int64_t id) { return IndexInGroup(group, id) >= 0; });
    if (!overlaps) {
      continue;
    }
    TT_RET_CHECK(row.size() == group.size() && absl::c_equal(row, group),
                 error::kInvalidArgument)
        << "collective " << op->getName().getStringRef().str()
        << " runs over devices " << GroupString(row)
        << ", which is not the program's device group " << GroupString(group);
    found = true;
  }
  TT_RET_CHECK(found, error::kInvalidArgument)
      << "collective " << op->getName().getStringRef().str()
      << " involves none of the program's devices " << GroupString(group);
  std::vector<int64_t> local(group.size());  // INT_VEC_OK
  absl::c_iota(local, 0);
  op->setAttr(kReplicaGroupsAttr,
              MakeRows(op->getContext(), {local}, group.size()));
  return absl::OkStatus();
}

absl::Status RewriteSourceTargetPairs(mlir::Operation* op,
                                      absl::Span<const int64_t> group) {
  auto attr =
      op->getAttrOfType<mlir::DenseIntElementsAttr>(kSourceTargetPairsAttr);
  if (attr == nullptr) {
    return absl::OkStatus();
  }
  TT_ASSIGN_OR_RETURN(const auto pairs, ReadRows(attr));
  std::vector<std::vector<int64_t>> local;  // INT_VEC_OK
  for (const auto& pair : pairs) {
    TT_RET_CHECK(pair.size() == 2, error::kInternal)
        << "expected source-target pairs of width 2, got " << pair.size();
    const int64_t source = IndexInGroup(group, pair[0]);
    const int64_t target = IndexInGroup(group, pair[1]);
    if (source < 0 && target < 0) {
      continue;
    }
    TT_RET_CHECK(source >= 0 && target >= 0, error::kInvalidArgument)
        << "collective permute pair (" << pair[0] << ", " << pair[1]
        << ") crosses the program's device group " << GroupString(group);
    local.push_back({source, target});
  }
  TT_RET_CHECK(!local.empty(), error::kInvalidArgument)
      << "collective permute involves none of the program's devices "
      << GroupString(group);
  op->setAttr(kSourceTargetPairsAttr, MakeRows(op->getContext(), local, 2));
  return absl::OkStatus();
}

// The projection of one `mhlo.sharding` string attribute onto `group`.
absl::StatusOr<mlir::StringAttr> ProjectShardingAttr(
    mlir::StringAttr attr, absl::Span<const int64_t> group) {
  TT_ASSIGN_OR_RETURN(const xla::HloSharding sharding,
                      xla::ParseSharding(attr.getValue().str()));
  TT_ASSIGN_OR_RETURN(const xla::HloSharding projected,
                      ProjectShardingToDeviceGroup(sharding, group));
  return mlir::StringAttr::get(attr.getContext(), projected.ToString());
}

// Projects the sharding attributes of `op`: its own, and, for a function,
// those of its arguments and results.
absl::Status RewriteShardings(mlir::Operation* op,
                              absl::Span<const int64_t> group) {
  if (auto attr = op->getAttrOfType<mlir::StringAttr>(kShardingAttr);
      attr != nullptr) {
    TT_ASSIGN_OR_RETURN(const mlir::StringAttr projected,
                        ProjectShardingAttr(attr, group));
    op->setAttr(kShardingAttr, projected);
  }
  auto func = mlir::dyn_cast<mlir::func::FuncOp>(op);
  if (func == nullptr) {
    return absl::OkStatus();
  }
  for (unsigned i = 0; i < func.getNumArguments(); ++i) {
    if (auto attr = func.getArgAttrOfType<mlir::StringAttr>(i, kShardingAttr);
        attr != nullptr) {
      TT_ASSIGN_OR_RETURN(const mlir::StringAttr projected,
                          ProjectShardingAttr(attr, group));
      func.setArgAttr(i, kShardingAttr, projected);
    }
  }
  for (unsigned i = 0; i < func.getNumResults(); ++i) {
    if (auto attr =
            func.getResultAttrOfType<mlir::StringAttr>(i, kShardingAttr);
        attr != nullptr) {
      TT_ASSIGN_OR_RETURN(const mlir::StringAttr projected,
                          ProjectShardingAttr(attr, group));
      func.setResultAttr(i, kShardingAttr, projected);
    }
  }
  return absl::OkStatus();
}

// Projects the module's SPMD entry shardings, when present.
absl::Status RewriteModuleShardings(mlir::ModuleOp module,
                                    absl::Span<const int64_t> group) {
  if (auto attr =
          module->getAttrOfType<mlir::StringAttr>(kSpmdOutputShardingAttr);
      attr != nullptr) {
    TT_ASSIGN_OR_RETURN(const mlir::StringAttr projected,
                        ProjectShardingAttr(attr, group));
    module->setAttr(kSpmdOutputShardingAttr, projected);
  }
  if (auto attr =
          module->getAttrOfType<mlir::ArrayAttr>(kSpmdParametersShardingsAttr);
      attr != nullptr) {
    std::vector<mlir::Attribute> projected;
    projected.reserve(attr.size());
    for (const mlir::Attribute element : attr) {
      auto text = mlir::dyn_cast<mlir::StringAttr>(element);
      TT_RET_CHECK(text != nullptr, error::kInternal)
          << "expected string entries in " << kSpmdParametersShardingsAttr;
      TT_ASSIGN_OR_RETURN(const mlir::StringAttr entry,
                          ProjectShardingAttr(text, group));
      projected.push_back(entry);
    }
    module->setAttr(kSpmdParametersShardingsAttr,
                    mlir::ArrayAttr::get(module.getContext(), projected));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<xla::HloSharding> ProjectShardingToDeviceGroup(
    const xla::HloSharding& sharding, absl::Span<const int64_t> group) {
  TT_RET_CHECK(!group.empty(), error::kInvalidArgument)
      << "cannot project a sharding onto an empty device group";
  if (sharding.IsTuple()) {
    std::vector<xla::HloSharding> elements;
    elements.reserve(sharding.tuple_elements().size());
    for (const xla::HloSharding& element : sharding.tuple_elements()) {
      TT_ASSIGN_OR_RETURN(xla::HloSharding projected,
                          ProjectShardingToDeviceGroup(element, group));
      elements.push_back(std::move(projected));
    }
    return xla::HloSharding::FlatTuple(std::move(elements));
  }
  if (sharding.IsReplicated() || sharding.IsManual() || sharding.IsUnknown()) {
    return sharding;
  }
  if (const std::optional<int64_t> device = sharding.UniqueDevice();
      device.has_value()) {
    const int64_t member = IndexInGroup(group, *device);
    TT_RET_CHECK(member >= 0, error::kInvalidArgument)
        << "sharding " << sharding.ToString()
        << " names a device outside the program's device group "
        << GroupString(group);
    return xla::HloSharding::SingleDevice(member);
  }
  for (const xla::OpSharding::Type type : sharding.subgroup_types()) {
    TT_RET_CHECK(type == xla::OpSharding::REPLICATED, error::kInvalidArgument)
        << "sharding " << sharding.ToString()
        << " has a subgroup type other than replication";
  }
  const xla::Array<int64_t>& tiles = sharding.tile_assignment().array();
  const absl::Span<const int64_t> dims = tiles.dimensions();
  const int64_t tile_rank =
      sharding.ReplicateOnLastTileDim() ? dims.size() - 1 : dims.size();
  int64_t tile_count = 1;
  for (int64_t d = 0; d < tile_rank; ++d) {
    tile_count *= dims[d];
  }
  const auto linear_index = [&](absl::Span<const int64_t> index) {
    int64_t linear = 0;
    for (int64_t d = 0; d < tile_rank; ++d) {
      linear = linear * dims[d] + index[d];
    }
    return linear;
  };
  // The group's members held by each tile, in tile-assignment order.
  std::vector<std::vector<int64_t>> members(tile_count);  // INT_VEC_OK
  tiles.Each([&](absl::Span<const int64_t> index, int64_t device) {
    const int64_t member = IndexInGroup(group, device);
    if (member >= 0) {
      members[linear_index(index)].push_back(member);
    }
  });
  const size_t count = members.empty() ? 0 : members[0].size();
  TT_RET_CHECK(count > 0, error::kInvalidArgument)
      << "sharding " << sharding.ToString()
      << " holds none of the program's devices " << GroupString(group);
  for (const auto& tile : members) {
    TT_RET_CHECK(tile.size() == count, error::kInvalidArgument)
        << "sharding " << sharding.ToString()
        << " splits the program's device group " << GroupString(group)
        << " unevenly across its tiles";
  }
  std::vector<int64_t> new_dims(dims.begin(),
                                dims.begin() + tile_rank);  // INT_VEC_OK
  if (count > 1) {
    new_dims.push_back(static_cast<int64_t>(count));
  }
  xla::Array<int64_t> projected(new_dims);
  projected.Each([&](absl::Span<const int64_t> index, int64_t* value) {
    const std::vector<int64_t>& tile = members[linear_index(index)];
    *value = tile[count > 1 ? index[tile_rank] : 0];
  });
  if (count > 1) {
    return xla::HloSharding::PartialTile(std::move(projected));
  }
  return xla::HloSharding::Tile(std::move(projected));
}

std::vector<int64_t> CommonCollectiveDeviceGroup(     // INT_VEC_OK
    absl::Span<const std::vector<int64_t>> groups) {  // INT_VEC_OK
  std::vector<int64_t> common;                        // INT_VEC_OK
  for (const auto& group : groups) {
    if (group.empty()) {
      continue;
    }
    if (common.empty()) {
      common = group;
      continue;
    }
    if (!absl::c_equal(common, group)) {
      return {};
    }
  }
  return common;
}

absl::Status RestrictBuildOptionsToDeviceGroup(
    xla::ExecutableBuildOptions& options, absl::Span<const int64_t> group,
    absl::Span<const int64_t> global_device_ids, bool replicated) {
  TT_RET_CHECK(!group.empty(), error::kInvalidArgument)
      << "cannot restrict a program to an empty device group";
  const int64_t n = group.size();
  xla::DeviceAssignment assignment(replicated ? n : 1, replicated ? 1 : n);
  for (int64_t j = 0; j < n; ++j) {
    const int64_t logical = group[j];
    TT_RET_CHECK(logical >= 0 && logical < global_device_ids.size(),
                 error::kInvalidArgument)
        << "logical device " << logical << " is outside the slice of "
        << global_device_ids.size() << " devices";
    if (replicated) {
      assignment(j, 0) = global_device_ids[logical];
    } else {
      assignment(0, j) = global_device_ids[logical];
    }
  }
  options.set_num_replicas(replicated ? n : 1);
  options.set_num_partitions(replicated ? 1 : n);
  options.set_device_assignment(assignment);
  return absl::OkStatus();
}

absl::Status RewriteCollectivesForDeviceGroup(mlir::ModuleOp module,
                                              absl::Span<const int64_t> group) {
  TT_RET_CHECK(!group.empty(), error::kInvalidArgument)
      << "cannot rewrite collectives for an empty device group";
  absl::Status status = absl::OkStatus();
  module.walk([&](mlir::Operation* op) {
    if (!status.ok()) {
      return mlir::WalkResult::interrupt();
    }
    if (auto target = op->getAttrOfType<mlir::StringAttr>(kCallTargetNameAttr);
        target != nullptr && target.getValue() == kRaggedAllToAllTarget) {
      status = TT_ERROR(error::kInvalidArgument)
               << "ragged all-to-all is not supported in a program compiled "
                  "for the device group "
               << GroupString(group);
      return mlir::WalkResult::interrupt();
    }
    status.Update(RewriteReplicaGroups(op, group));
    if (status.ok()) {
      status.Update(RewriteSourceTargetPairs(op, group));
    }
    if (status.ok()) {
      status.Update(RewriteShardings(op, group));
    }
    return status.ok() ? mlir::WalkResult::advance()
                       : mlir::WalkResult::interrupt();
  });
  if (status.ok()) {
    status.Update(RewriteModuleShardings(module, group));
  }
  return status;
}

}  // namespace torch_tpu
