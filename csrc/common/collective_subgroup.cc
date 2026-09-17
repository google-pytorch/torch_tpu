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
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "csrc/common/error_utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "xla/client/executable_build_options.h"
#include "xla/service/computation_placer.h"

namespace torch_tpu {

namespace {

constexpr char kReplicaGroupsAttr[] = "replica_groups";
constexpr char kSourceTargetPairsAttr[] = "source_target_pairs";
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

}  // namespace

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
    return status.ok() ? mlir::WalkResult::advance()
                       : mlir::WalkResult::interrupt();
  });
  return status;
}

}  // namespace torch_tpu
