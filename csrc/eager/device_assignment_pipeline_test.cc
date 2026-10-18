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
#include <map>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/Exception.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/ops/op_names.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

std::vector<int64_t> ExtractDenseIntAttrValues(  // INT_VEC_OK
    mlir::Operation* op, std::string_view attr_name) {
  std::vector<int64_t> vals;  // INT_VEC_OK
  if (auto attr = op->getAttrOfType<mlir::DenseIntElementsAttr>(attr_name)) {
    for (const llvm::APInt& v : attr.getValues<llvm::APInt>()) {
      vals.push_back(v.getSExtValue());
    }
  }
  return vals;
}

// Megascale device ids encode the slice as `id / 100000`.
constexpr int64_t kMegascaleSliceStride = 100000;

int64_t SliceOf(int64_t device_id) { return device_id / kMegascaleSliceStride; }

// Groups of `op_da` intended by an op emitting iota replica_groups of size
// `group_size`: group g, rank i is `op_da[g * group_size + i]`.
std::vector<std::vector<int64_t>> IntendedGroups(  // INT_VEC_OK
    absl::Span<const int64_t> op_da, int64_t group_size) {
  std::vector<std::vector<int64_t>> groups;  // INT_VEC_OK
  const int64_t num_groups = op_da.size() / group_size;
  for (int64_t g = 0; g < num_groups; ++g) {
    groups.emplace_back(op_da.begin() + g * group_size,
                        op_da.begin() + (g + 1) * group_size);
  }
  return groups;
}

// Maps each partition index in `rg` to its physical device `resolved_da[p]`,
// preserving member order. Mirrors the flattened-id-to-device-id conversion
// plus the in-order endpoint iteration in the XLA compiler's cross-slice
// replica-group rewrite: member i of a replica group is shard/rank i of the
// collective.
std::vector<std::vector<int64_t>> PhysicalGroups(  // INT_VEC_OK
    absl::Span<const int64_t> rg, int64_t group_size,
    absl::Span<const int64_t> resolved_da) {
  std::vector<std::vector<int64_t>> groups;  // INT_VEC_OK
  const int64_t num_groups = rg.size() / group_size;
  for (int64_t g = 0; g < num_groups; ++g) {
    std::vector<int64_t>& group = groups.emplace_back();  // INT_VEC_OK
    for (int64_t i = 0; i < group_size; ++i) {
      group.push_back(resolved_da[rg[g * group_size + i]]);
    }
  }
  return groups;
}

// Walks `resolved_da` left to right and assigns each device a local partition
// id equal to the running per-slice count. Mirrors XLA's partition-to-replica
// mapping.
std::map<int64_t, int64_t> LocalPartitionIds(
    absl::Span<const int64_t> resolved_da) {
  std::map<int64_t, int64_t> per_slice_count;
  std::map<int64_t, int64_t> local_ids;
  for (int64_t dev : resolved_da) {
    local_ids[dev] = per_slice_count[SliceOf(dev)]++;
  }
  return local_ids;
}

// Mirrors the XLA compiler's same-per-slice-communication-pattern check in its
// cross-slice replica-group rewrite, with replica order significant: localize
// each intra-slice group to local partition ids (member order preserved), sort
// each slice's list of groups (members are not sorted), and require every
// slice to match slice 0. When true, `CollectiveLocalizer` clones slice 0's
// localized collective onto every slice. Only models intra-slice collectives;
// returns false for a group that spans slices.
bool SamePerSliceCommunicationPattern(absl::Span<const int64_t> rg,
                                      int64_t group_size,
                                      absl::Span<const int64_t> resolved_da) {
  const std::map<int64_t, int64_t> local_ids = LocalPartitionIds(resolved_da);
  std::map<int64_t, std::vector<std::vector<int64_t>>>  // INT_VEC_OK
      groups_by_slice;
  for (const std::vector<int64_t>& group :  // INT_VEC_OK
       PhysicalGroups(rg, group_size, resolved_da)) {
    const int64_t slice = SliceOf(group.front());
    std::vector<int64_t> localized;  // INT_VEC_OK
    for (int64_t dev : group) {
      if (SliceOf(dev) != slice) {
        return false;
      }
      localized.push_back(local_ids.at(dev));
    }
    groups_by_slice[slice].push_back(std::move(localized));
  }
  if (groups_by_slice.empty()) {
    return true;
  }
  for (auto& [slice, groups] : groups_by_slice) {
    std::sort(groups.begin(), groups.end());
  }
  const auto& slice0_groups = groups_by_slice.begin()->second;
  for (const auto& [slice, groups] : groups_by_slice) {
    if (groups != slice0_groups) {
      return false;
    }
  }
  return true;
}

TEST(DeviceAssignmentPipelineTest,
     ResolveDeviceAssignmentPermutationRemapsWithWarning) {
  const std::vector<int64_t>  // INT_VEC_OK
      pallas_da = {0, 1, 4, 5, 6, 7, 2, 3};
  const std::vector<int64_t>  // INT_VEC_OK
      collective_da = {0, 1, 2, 3, 4, 5, 6, 7};

  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<int64_t> resolved_da,  // INT_VEC_OK
      ResolveDeviceAssignment(
          {
              OpDeviceAssignment{OpName::kDistributedAllReduce, collective_da},
          },
          {
              PallasDeviceAssignment{pallas_da},
          }));

  EXPECT_EQ(resolved_da, pallas_da);

  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<mlir::stablehlo::StablehloDialect>();
  mlir_context.loadDialect<mlir::func::FuncDialect>();

  mlir::ModuleBuilder mb(mlir_context, "test_permutation_module");
  mlir::func::FunctionBuilder fb(mb, "main");
  auto tensor_type =
      mlir::makeTensorType(mlir_context, {8}, mlir::ElementType::F32);
  mlir::MlirOp arg = mlir::func::Argument(fb, tensor_type);

  mlir::MlirOp collective_op = mlir::stablehlo::Neg(arg);
  mlir::Operation* raw_op = collective_op.getValue().getDefiningOp();
  auto rg_type = mlir::RankedTensorType::get(
      {2, 4}, mlir::IntegerType::get(&mlir_context, 64));
  raw_op->setAttr(
      "replica_groups",
      mlir::DenseIntElementsAttr::get(
          rg_type, llvm::ArrayRef<int64_t>({0, 1, 2, 3, 4, 5, 6, 7})));

  mlir::func::Return(fb, {collective_op});
  auto module = mb.build();

  EXPECT_TRUE(
      RewriteAndTileOperationForSubgroup(raw_op, collective_da, resolved_da)
          .ok());

  EXPECT_THAT(ExtractDenseIntAttrValues(raw_op, "replica_groups"),
              testing::ElementsAre(0, 1, 6, 7, 2, 3, 4, 5));
}

TEST(DeviceAssignmentPipelineTest,
     ResolveDeviceAssignmentPreservesSyntheticNonZeroDeviceIds) {
  const std::vector<int64_t> synthetic_da = {2, 3};             // INT_VEC_OK
  TT_ASSERT_OK_AND_ASSIGN(const std::vector<int64_t> resolved,  // INT_VEC_OK
                          ResolveDeviceAssignment({
                              OpDeviceAssignment{OpName::kAdd, synthetic_da},
                          }));
  EXPECT_THAT(resolved, testing::ElementsAre(2, 3));
}

TEST(DeviceAssignmentPipelineTest,
     ResolveDeviceAssignmentThrowsOnDifferingPallasMeshes) {
  std::vector<int64_t> da1(128);  // INT_VEC_OK
  std::vector<int64_t> da2(128);  // INT_VEC_OK
  for (int i = 0; i < 128; ++i) {
    da1[i] = i;
    da2[i] = 127 - i;
  }
  constexpr std::string_view kExpectedMessage =
      "expected Pallas kernels and scans in the same program to have the same "
      "device assignment, got ";
  bool caught = false;
  try {
    (void)ResolveDeviceAssignment({}, {
                                          PallasDeviceAssignment{da1},
                                          PallasDeviceAssignment{da2},
                                      });
  } catch (const c10::Error& e) {
    caught = true;
    EXPECT_THAT(e.what(), testing::HasSubstr(kExpectedMessage));
  } catch (const torch_tpu::TtError& e) {
    caught = true;
    EXPECT_THAT(e.what(), testing::HasSubstr(kExpectedMessage));
  }
  EXPECT_TRUE(caught);
}

TEST(DeviceAssignmentPipelineTest,
     ResolveDeviceAssignmentLogsWarningAndFailsOnDifferingDAs) {
  const std::vector<int64_t> da1 = {0, 1, 2, 3, 4, 5, 6, 7};  // INT_VEC_OK
  const std::vector<int64_t>                                  // INT_VEC_OK
      da2 = {8, 9, 10, 11, 12, 13, 14, 15};

  const absl::StatusOr<std::vector<int64_t>> resolved =  // INT_VEC_OK
      ResolveDeviceAssignment({
          OpDeviceAssignment{OpName::kDistributedAllReduce, da1},
          OpDeviceAssignment{OpName::kDistributedAllReduce, da2},
      });

  EXPECT_FALSE(resolved.ok());
  EXPECT_EQ(resolved.status().code(), error::kInvalidArgument);
  EXPECT_THAT(
      resolved.status().message(),
      testing::HasSubstr("device assignment mismatch across operations"));
  EXPECT_THAT(
      resolved.status().message(),
      testing::HasSubstr("All operations in the traversal must share the same "
                         "world-sized device assignment"));

  const std::vector<int64_t> da3 = {0, 1, 2, 3};  // INT_VEC_OK
  const absl::StatusOr<std::vector<int64_t>>      // INT_VEC_OK
      resolved_size_mismatch = ResolveDeviceAssignment({
          OpDeviceAssignment{OpName::kDistributedAllReduce, da1},
          OpDeviceAssignment{OpName::kDistributedAllReduce, da3},
      });

  EXPECT_FALSE(resolved_size_mismatch.ok());
  EXPECT_EQ(resolved_size_mismatch.status().code(), error::kInvalidArgument);
  EXPECT_THAT(
      resolved_size_mismatch.status().message(),
      testing::HasSubstr("device assignment mismatch across operations"));
  EXPECT_THAT(
      resolved_size_mismatch.status().message(),
      testing::HasSubstr("All operations in the traversal must share the same "
                         "world-sized device assignment"));

  const absl::StatusOr<std::vector<int64_t>>  // INT_VEC_OK
      resolved_pallas_mismatch = ResolveDeviceAssignment(
          {
              OpDeviceAssignment{OpName::kDistributedAllReduce, da2},
          },
          {
              PallasDeviceAssignment{da1},
          });

  EXPECT_FALSE(resolved_pallas_mismatch.ok());
  EXPECT_EQ(resolved_pallas_mismatch.status().code(), error::kInvalidArgument);
  EXPECT_THAT(
      resolved_pallas_mismatch.status().message(),
      testing::HasSubstr("device assignment mismatch across operations"));
  EXPECT_THAT(
      resolved_pallas_mismatch.status().message(),
      testing::HasSubstr("All operations in the traversal must share the same "
                         "world-sized device assignment"));
}

TEST(DeviceAssignmentPipelineTest, MixedGroupsResolveToTheGlobalDevices) {
  const std::vector<int64_t> pair = {0, 1};              // INT_VEC_OK
  const std::vector<int64_t> pair_layout = {0, 1, 2, 3,  // INT_VEC_OK
                                            4, 5, 6, 7};
  const std::vector<int64_t> stride = {0, 2, 4, 6};        // INT_VEC_OK
  const std::vector<int64_t> stride_layout = {0, 2, 4, 6,  // INT_VEC_OK
                                              1, 3, 5, 7};

  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<int64_t> resolved,  // INT_VEC_OK
      ResolveDeviceAssignment({
          OpDeviceAssignment{OpName::kDistributedAllReduce, pair, pair_layout},
          OpDeviceAssignment{OpName::kDistributedAllReduce, stride,
                             stride_layout},
      }));

  EXPECT_THAT(resolved, testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
}

TEST(DeviceAssignmentPipelineTest,
     MixedGroupsInsideAPipelineStageResolveToTheStage) {
  // Two pipeline stages of four devices, with tensor-parallel pairs and
  // context-parallel pairs inside each stage. A program of stage 1 that
  // combines the two stays on the stage, in the order of the first layout; a
  // Pallas kernel over the stage's mesh anchors the order instead.
  const std::vector<int64_t> tp_pair = {4, 5};  // INT_VEC_OK
  const std::vector<int64_t> tp_layout = {0, 1, 2, 3,
                                          4, 5, 6, 7};  // INT_VEC_OK
  const std::vector<int64_t> pcp_pair = {4, 6};         // INT_VEC_OK
  const std::vector<int64_t> pcp_layout = {0, 2, 1, 3,
                                           4, 6, 5, 7};  // INT_VEC_OK
  const std::vector<int64_t> stage_mesh = {6, 7, 4, 5};  // INT_VEC_OK
  const std::vector<OpDeviceAssignment> ops = {
      OpDeviceAssignment{OpName::kDistributedAllReduce, tp_pair, tp_layout},
      OpDeviceAssignment{OpName::kDistributedAllReduce, pcp_pair, pcp_layout},
  };

  TT_ASSERT_OK_AND_ASSIGN(const std::vector<int64_t> resolved,  // INT_VEC_OK
                          ResolveDeviceAssignment(ops));
  EXPECT_THAT(resolved, testing::ElementsAre(4, 5, 6, 7));

  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<int64_t> resolved_with_mesh,  // INT_VEC_OK
      ResolveDeviceAssignment(ops, {PallasDeviceAssignment{stage_mesh}}));
  EXPECT_THAT(resolved_with_mesh, testing::ElementsAre(6, 7, 4, 5));
}

TEST(DeviceAssignmentPipelineTest, MixedGroupsWithoutGlobalDevicesAreRejected) {
  const std::vector<int64_t> world = {0, 1, 2, 3, 4, 5, 6, 7};  // INT_VEC_OK
  const std::vector<int64_t> half = {0, 1, 2, 3};               // INT_VEC_OK

  const absl::StatusOr<std::vector<int64_t>> resolved =  // INT_VEC_OK
      ResolveDeviceAssignment({
          OpDeviceAssignment{OpName::kDistributedAllReduce, world},
          OpDeviceAssignment{OpName::kDistributedAllReduce, half},
      });

  EXPECT_FALSE(resolved.ok());
  EXPECT_EQ(resolved.status().code(), error::kInvalidArgument);
  EXPECT_THAT(
      resolved.status().message(),
      testing::HasSubstr("device assignment mismatch across operations"));
}

TEST(DeviceAssignmentPipelineTest,
     RewriteAndTileOperationForSubgroupTilesAGroupOverItsLayout) {
  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<mlir::stablehlo::StablehloDialect>();
  mlir_context.loadDialect<mlir::func::FuncDialect>();
  mlir::ModuleBuilder mb(mlir_context, "test_module");
  mlir::func::FunctionBuilder fb(mb, "main");
  auto tensor_type =
      mlir::makeTensorType(mlir_context, {8}, mlir::ElementType::F32);
  mlir::MlirOp arg = mlir::func::Argument(fb, tensor_type);
  mlir::MlirOp reduce_op = mlir::stablehlo::Neg(arg);
  reduce_op.getValue().getDefiningOp()->setAttr(
      "replica_groups",
      mlir::DenseIntElementsAttr::get(
          mlir::RankedTensorType::get(
              {1, 2}, mlir::IntegerType::get(&mlir_context, 64)),
          llvm::ArrayRef<int64_t>({0, 1})));
  mlir::MlirOp permute_op = mlir::stablehlo::Abs(reduce_op);
  permute_op.getValue().getDefiningOp()->setAttr(
      "source_target_pairs",
      mlir::DenseIntElementsAttr::get(
          mlir::RankedTensorType::get(
              {2, 2}, mlir::IntegerType::get(&mlir_context, 64)),
          llvm::ArrayRef<int64_t>({0, 1, 1, 0})));
  mlir::func::Return(fb, {permute_op});
  auto module = mb.build();

  const std::vector<int64_t> op_da = {2, 3};  // INT_VEC_OK
  const std::vector<int64_t>                  // INT_VEC_OK
      resolved_da = {7, 6, 5, 4, 3, 2, 1, 0};
  const std::vector<int64_t>  // INT_VEC_OK
      global_da = {0, 1, 2, 3, 4, 5, 6, 7};
  EXPECT_TRUE(
      RewriteAndTileOperationForSubgroup(reduce_op.getValue().getDefiningOp(),
                                         op_da, resolved_da, global_da)
          .ok());
  EXPECT_TRUE(
      RewriteAndTileOperationForSubgroup(permute_op.getValue().getDefiningOp(),
                                         op_da, resolved_da, global_da)
          .ok());

  // Pair {d0, d1} sits at positions {7 - d0, 7 - d1} of `resolved_da`. The
  // replica groups are sorted by their first element; the source-target pairs
  // keep the group order.
  EXPECT_THAT(ExtractDenseIntAttrValues(reduce_op.getValue().getDefiningOp(),
                                        "replica_groups"),
              testing::ElementsAre(1, 0, 3, 2, 5, 4, 7, 6));
  EXPECT_THAT(
      ExtractDenseIntAttrValues(permute_op.getValue().getDefiningOp(),
                                "source_target_pairs"),
      testing::ElementsAre(7, 6, 6, 7, 5, 4, 4, 5, 3, 2, 2, 3, 1, 0, 0, 1));
}

TEST(DeviceAssignmentPipelineTest,
     RewriteAndTileOperationForSubgroupTilesMultiSliceLayout) {
  // 2 slices x 4 devices: devices 0-3 on slice 0 and 4-7 on slice 1, with
  // topology order equal to device id order. Launcher world ranks are
  // permuted relative to the devices: world rank r runs on device
  // `rank_to_device_id[r]`.
  constexpr int64_t kDevicesPerSlice = 4;
  const std::vector<int64_t>  // INT_VEC_OK
      rank_to_device_id = {5, 4, 7, 6, 1, 0, 3, 2};
  // The TP groups are the device pairs {0, 1}, {2, 3}, {4, 5}, {6, 7}, i.e.
  // world ranks {5, 4}, {7, 6}, {1, 0}, {3, 2}. new_group sorts its ranks, so
  // in PG-rank order they are {4, 5}, {6, 7}, {0, 1}, {2, 3}, whose leaders
  // (PG rank 0) have topology orders 1, 3, 5, 7.
  // PermuteDeviceAssignmentForSubgroups (csrc/distributed, which this target
  // does not depend on) sorts the pairs by their leader's topology order and
  // flattens them, each in PG-rank order.
  const std::vector<int64_t>  // INT_VEC_OK
      flattened_pair_world_ranks = {4, 5, 6, 7, 0, 1, 2, 3};
  std::vector<int64_t> global_da;  // INT_VEC_OK
  for (int64_t world_rank : flattened_pair_world_ranks) {
    global_da.push_back(rank_to_device_id[world_rank]);
  }
  ASSERT_THAT(global_da, testing::ElementsAre(1, 0, 3, 2, 5, 4, 7, 6));
  // The op is a collective of the pair of world ranks {0, 1}, which sits at
  // positions 4 and 5 of `global_da`.
  const std::vector<int64_t> op_da = {5, 4};  // INT_VEC_OK

  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<mlir::stablehlo::StablehloDialect>();
  mlir_context.loadDialect<mlir::func::FuncDialect>();
  mlir::ModuleBuilder mb(mlir_context, "test_multislice_tiling_module");
  mlir::func::FunctionBuilder fb(mb, "main");
  auto tensor_type =
      mlir::makeTensorType(mlir_context, {8}, mlir::ElementType::F32);
  mlir::MlirOp arg = mlir::func::Argument(fb, tensor_type);
  auto i64_type = mlir::IntegerType::get(&mlir_context, 64);
  // The pair's collectives use local positions: replica_groups {{0, 1}} and
  // source_target_pairs {{0, 1}, {1, 0}}. Each case below rewrites its own
  // copy of both ops.
  const auto pair_replica_groups = mlir::DenseIntElementsAttr::get(
      mlir::RankedTensorType::get({1, 2}, i64_type),
      llvm::ArrayRef<int64_t>({0, 1}));
  const auto pair_source_target_pairs = mlir::DenseIntElementsAttr::get(
      mlir::RankedTensorType::get({2, 2}, i64_type),
      llvm::ArrayRef<int64_t>({0, 1, 1, 0}));
  mlir::MlirOp rg_op = mlir::stablehlo::Neg(arg);
  mlir::Operation* raw_rg_op = rg_op.getValue().getDefiningOp();
  raw_rg_op->setAttr("replica_groups", pair_replica_groups);
  mlir::MlirOp st_op = mlir::stablehlo::Abs(rg_op);
  mlir::Operation* raw_st_op = st_op.getValue().getDefiningOp();
  raw_st_op->setAttr("source_target_pairs", pair_source_target_pairs);
  mlir::MlirOp rg_op_2 = mlir::stablehlo::Neg(st_op);
  mlir::Operation* raw_rg_op_2 = rg_op_2.getValue().getDefiningOp();
  raw_rg_op_2->setAttr("replica_groups", pair_replica_groups);
  mlir::MlirOp st_op_2 = mlir::stablehlo::Abs(rg_op_2);
  mlir::Operation* raw_st_op_2 = st_op_2.getValue().getDefiningOp();
  raw_st_op_2->setAttr("source_target_pairs", pair_source_target_pairs);
  mlir::func::Return(fb, {st_op_2});
  auto module = mb.build();

  // Returns the members of the rows of `rg` (positions into `resolved`) that
  // run on `slice`, in row order, relative to the slice's first position.
  // Both slices occupy a contiguous block of positions in the cases below.
  const auto slice_local_rows = [](absl::Span<const int64_t> rg,
                                   absl::Span<const int64_t> resolved,
                                   int64_t slice) {
    const int64_t size = static_cast<int64_t>(resolved.size());
    int64_t first = size;
    for (int64_t p = 0; p < size; ++p) {
      if (resolved[p] / kDevicesPerSlice == slice) {
        first = std::min(first, p);
      }
    }
    std::vector<int64_t> local;  // INT_VEC_OK
    for (int64_t p : rg) {
      if (resolved[p] / kDevicesPerSlice == slice) {
        local.push_back(p - first);
      }
    }
    return local;
  };

  // Case 1: the program runs on the global devices. Every device sits at its
  // own position in `global_da`, so pair g is tiled onto positions
  // {2g, 2g + 1}: IOTA rows into the topology-ordered assignment, not the
  // pairs' world ranks ({4, 5}, {6, 7}, {0, 1}, {2, 3}) or device ids
  // ({1, 0}, {3, 2}, {5, 4}, {7, 6}).
  const std::vector<int64_t>& resolved_da = global_da;  // INT_VEC_OK
  ASSERT_TRUE(RewriteAndTileOperationForSubgroup(raw_rg_op, op_da, resolved_da,
                                                 global_da)
                  .ok());
  ASSERT_TRUE(RewriteAndTileOperationForSubgroup(raw_st_op, op_da, resolved_da,
                                                 global_da)
                  .ok());
  const std::vector<int64_t> rg =  // INT_VEC_OK
      ExtractDenseIntAttrValues(raw_rg_op, "replica_groups");
  EXPECT_THAT(rg, testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
  EXPECT_THAT(
      PhysicalGroups(rg, /*group_size=*/2, resolved_da),
      testing::ElementsAreArray(IntendedGroups(global_da, /*group_size=*/2)));
  // Slice 0 holds positions 0-3 and slice 1 positions 4-7; both localize to
  // rows {0, 1}, {2, 3}.
  EXPECT_THAT(slice_local_rows(rg, resolved_da, /*slice=*/0),
              testing::ElementsAre(0, 1, 2, 3));
  EXPECT_EQ(slice_local_rows(rg, resolved_da, /*slice=*/0),
            slice_local_rows(rg, resolved_da, /*slice=*/1));
  // Source-target pairs are tiled pair by pair and not sorted, as in
  // RewriteSourceTargetPairsAttr.
  EXPECT_THAT(
      ExtractDenseIntAttrValues(raw_st_op, "source_target_pairs"),
      testing::ElementsAre(0, 1, 1, 0, 2, 3, 3, 2, 4, 5, 5, 4, 6, 7, 7, 6));

  // Case 2: the program runs on a different permutation of the same devices,
  // e.g. another op's world assignment listing slice 1 first, in device id
  // order. Device d sits at position (d + 4) % 8, so `global_da`
  // {1, 0, 3, 2, 5, 4, 7, 6} maps to positions {5, 4, 7, 6, 1, 0, 3, 2}: the
  // tiles are {5, 4}, {7, 6}, {1, 0}, {3, 2}. The replica groups are sorted
  // by their first element, {1, 0}, {3, 2}, {5, 4}, {7, 6}, each keeping its
  // PG-rank order; the source-target pairs keep the tile order.
  const std::vector<int64_t>  // INT_VEC_OK
      resolved_da_2 = {4, 5, 6, 7, 0, 1, 2, 3};
  ASSERT_TRUE(RewriteAndTileOperationForSubgroup(raw_rg_op_2, op_da,
                                                 resolved_da_2, global_da)
                  .ok());
  ASSERT_TRUE(RewriteAndTileOperationForSubgroup(raw_st_op_2, op_da,
                                                 resolved_da_2, global_da)
                  .ok());
  const std::vector<int64_t> rg_2 =  // INT_VEC_OK
      ExtractDenseIntAttrValues(raw_rg_op_2, "replica_groups");
  EXPECT_THAT(rg_2, testing::ElementsAre(1, 0, 3, 2, 5, 4, 7, 6));
  // Each row still selects one pair, in PG-rank order: e.g. row {5, 4} is
  // devices {resolved_da_2[5], resolved_da_2[4]} = {1, 0}, world ranks {4, 5}.
  EXPECT_THAT(PhysicalGroups(rg_2, /*group_size=*/2, resolved_da_2),
              testing::UnorderedElementsAreArray(
                  IntendedGroups(global_da, /*group_size=*/2)));
  // Slice 1 holds positions 0-3 and slice 0 positions 4-7; both localize to
  // rows {1, 0}, {3, 2}.
  EXPECT_THAT(slice_local_rows(rg_2, resolved_da_2, /*slice=*/0),
              testing::ElementsAre(1, 0, 3, 2));
  EXPECT_EQ(slice_local_rows(rg_2, resolved_da_2, /*slice=*/0),
            slice_local_rows(rg_2, resolved_da_2, /*slice=*/1));
  EXPECT_THAT(
      ExtractDenseIntAttrValues(raw_st_op_2, "source_target_pairs"),
      testing::ElementsAre(5, 4, 4, 5, 7, 6, 6, 7, 1, 0, 0, 1, 3, 2, 2, 3));
}

TEST(DeviceAssignmentPipelineTest,
     RewriteAndTileOperationForSubgroupRewritesOrthogonalReplicaGroups) {
  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<mlir::stablehlo::StablehloDialect>();
  mlir_context.loadDialect<mlir::func::FuncDialect>();

  mlir::ModuleBuilder mb(mlir_context, "test_module");
  mlir::func::FunctionBuilder fb(mb, "main");
  auto tensor_type =
      mlir::makeTensorType(mlir_context, {8}, mlir::ElementType::F32);
  mlir::MlirOp arg = mlir::func::Argument(fb, tensor_type);

  mlir::MlirOp shard_op = mlir::stablehlo::Neg(arg);
  auto shard_rg_type = mlir::RankedTensorType::get(
      {2, 4}, mlir::IntegerType::get(&mlir_context, 64));
  shard_op.getValue().getDefiningOp()->setAttr(
      "replica_groups",
      mlir::DenseIntElementsAttr::get(
          shard_rg_type, llvm::ArrayRef<int64_t>({0, 1, 2, 3, 4, 5, 6, 7})));

  mlir::MlirOp replicate_op = mlir::stablehlo::Abs(shard_op);
  auto rep_rg_type = mlir::RankedTensorType::get(
      {4, 2}, mlir::IntegerType::get(&mlir_context, 64));
  replicate_op.getValue().getDefiningOp()->setAttr(
      "replica_groups",
      mlir::DenseIntElementsAttr::get(
          rep_rg_type, llvm::ArrayRef<int64_t>({0, 4, 1, 5, 2, 6, 3, 7})));

  mlir::func::Return(fb, {replicate_op});
  auto module = mb.build();

  const std::vector<int64_t> op_da = {0, 1, 2, 3, 4, 5, 6, 7};  // INT_VEC_OK
  const std::vector<int64_t>                                    // INT_VEC_OK
      resolved_da = {2, 3, 0, 1, 6, 7, 4, 5};
  EXPECT_TRUE(RewriteAndTileOperationForSubgroup(
                  shard_op.getValue().getDefiningOp(), op_da, resolved_da)
                  .ok());
  EXPECT_TRUE(RewriteAndTileOperationForSubgroup(
                  replicate_op.getValue().getDefiningOp(), op_da, resolved_da)
                  .ok());

  EXPECT_THAT(ExtractDenseIntAttrValues(shard_op.getValue().getDefiningOp(),
                                        "replica_groups"),
              testing::ElementsAre(2, 3, 0, 1, 6, 7, 4, 5));
  EXPECT_THAT(ExtractDenseIntAttrValues(replicate_op.getValue().getDefiningOp(),
                                        "replica_groups"),
              testing::ElementsAre(0, 4, 1, 5, 2, 6, 3, 7));
}

TEST(DeviceAssignmentPipelineTest,
     RemapOperationForSubgroupRemapsReplicaGroupsAndSourceTargetPairs) {
  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<mlir::stablehlo::StablehloDialect>();
  mlir_context.loadDialect<mlir::func::FuncDialect>();

  mlir::ModuleBuilder mb(mlir_context, "test_tile_module");
  mlir::func::FunctionBuilder fb(mb, "main");
  auto tensor_type =
      mlir::makeTensorType(mlir_context, {8}, mlir::ElementType::F32);
  mlir::MlirOp arg = mlir::func::Argument(fb, tensor_type);

  mlir::MlirOp op = mlir::stablehlo::Neg(arg);
  mlir::Operation* raw_op = op.getValue().getDefiningOp();
  auto rg_type = mlir::RankedTensorType::get(
      {2, 4}, mlir::IntegerType::get(&mlir_context, 64));
  raw_op->setAttr(
      "replica_groups",
      mlir::DenseIntElementsAttr::get(
          rg_type, llvm::ArrayRef<int64_t>({0, 1, 2, 3, 4, 5, 6, 7})));
  auto st_type = mlir::RankedTensorType::get(
      {4, 2}, mlir::IntegerType::get(&mlir_context, 64));
  raw_op->setAttr(
      "source_target_pairs",
      mlir::DenseIntElementsAttr::get(
          st_type, llvm::ArrayRef<int64_t>({0, 1, 2, 3, 4, 5, 6, 7})));

  mlir::func::Return(fb, {op});
  auto module = mb.build();
  mlir::func::FuncOp func_op = module->lookupSymbol<mlir::func::FuncOp>("main");

  const std::vector<int64_t> op_da = {0, 1, 2, 3, 4, 5, 6, 7};  // INT_VEC_OK
  const std::vector<int64_t>                                    // INT_VEC_OK
      resolved_da = {0, 1, 6, 7, 2, 3, 4, 5};
  EXPECT_TRUE(RewriteAndTileOperationForSubgroup(func_op.getOperation(), op_da,
                                                 resolved_da)
                  .ok());

  EXPECT_THAT(ExtractDenseIntAttrValues(raw_op, "replica_groups"),
              testing::ElementsAre(0, 1, 4, 5, 6, 7, 2, 3));
  EXPECT_THAT(ExtractDenseIntAttrValues(raw_op, "source_target_pairs"),
              testing::ElementsAre(0, 1, 4, 5, 6, 7, 2, 3));
}

TEST(DeviceAssignmentPipelineTest,
     RewriteAndTileOperationForSubgroupHandlesPermutedPallasMesh) {
  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<mlir::stablehlo::StablehloDialect>();
  mlir_context.loadDialect<mlir::func::FuncDialect>();

  mlir::ModuleBuilder mb(mlir_context, "test_hsdp_module");
  mlir::func::FunctionBuilder fb(mb, "main");
  auto tensor_type =
      mlir::makeTensorType(mlir_context, {8}, mlir::ElementType::F32);
  mlir::MlirOp arg = mlir::func::Argument(fb, tensor_type);

  mlir::MlirOp replicate_op = mlir::stablehlo::Abs(arg);
  auto rep_rg_type = mlir::RankedTensorType::get(
      {4, 2}, mlir::IntegerType::get(&mlir_context, 64));
  replicate_op.getValue().getDefiningOp()->setAttr(
      "replica_groups",
      mlir::DenseIntElementsAttr::get(
          rep_rg_type, llvm::ArrayRef<int64_t>({0, 4, 1, 5, 2, 6, 3, 7})));

  mlir::func::Return(fb, {replicate_op});
  auto module = mb.build();

  const std::vector<int64_t> op_da = {0, 1, 2, 3, 4, 5, 6, 7};  // INT_VEC_OK
  const std::vector<int64_t>                                    // INT_VEC_OK
      resolved_da = {3, 2, 1, 0, 4, 5, 6, 7};
  EXPECT_TRUE(RewriteAndTileOperationForSubgroup(
                  replicate_op.getValue().getDefiningOp(), op_da, resolved_da)
                  .ok());

  EXPECT_THAT(ExtractDenseIntAttrValues(replicate_op.getValue().getDefiningOp(),
                                        "replica_groups"),
              testing::ElementsAre(0, 7, 1, 6, 2, 5, 3, 4));
}

TEST(DeviceAssignmentPipelineTest,
     RewriteAndTileOperationForSubgroupHsdpReplicateSymmetricAcrossRanks) {
  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<mlir::stablehlo::StablehloDialect>();
  mlir_context.loadDialect<mlir::func::FuncDialect>();

  mlir::ModuleBuilder mb(mlir_context, "test_hsdp_symmetric_module");
  mlir::func::FunctionBuilder fb(mb, "main");
  auto tensor_type =
      mlir::makeTensorType(mlir_context, {8}, mlir::ElementType::F32);
  mlir::MlirOp arg = mlir::func::Argument(fb, tensor_type);

  std::vector<int64_t> rep_rg_vals;  // INT_VEC_OK
  rep_rg_vals.reserve(128);
  for (int64_t i = 0; i < 64; ++i) {
    rep_rg_vals.push_back(i);
    rep_rg_vals.push_back(i + 64);
  }
  auto rep_rg_type = mlir::RankedTensorType::get(
      {64, 2}, mlir::IntegerType::get(&mlir_context, 64));

  mlir::MlirOp replicate_op_rank0 = mlir::stablehlo::Abs(arg);
  replicate_op_rank0.getValue().getDefiningOp()->setAttr(
      "replica_groups", mlir::DenseIntElementsAttr::get(
                            rep_rg_type, llvm::ArrayRef<int64_t>(rep_rg_vals)));

  mlir::MlirOp replicate_op_rank8 = mlir::stablehlo::Abs(replicate_op_rank0);
  replicate_op_rank8.getValue().getDefiningOp()->setAttr(
      "replica_groups", mlir::DenseIntElementsAttr::get(
                            rep_rg_type, llvm::ArrayRef<int64_t>(rep_rg_vals)));

  mlir::func::Return(fb, {replicate_op_rank8});
  auto module = mb.build();

  const std::vector<int64_t>  // INT_VEC_OK
      shard_subgroup_0 = {
          0,  1,  32, 33, 64, 65, 96,  97,  8,  9,  40, 41, 72, 73, 104, 105,
          16, 17, 48, 49, 80, 81, 112, 113, 24, 25, 56, 57, 88, 89, 120, 121,
          2,  3,  34, 35, 66, 67, 98,  99,  10, 11, 42, 43, 74, 75, 106, 107,
          18, 19, 50, 51, 82, 83, 114, 115, 26, 27, 58, 59, 90, 91, 122, 123};

  auto antipodal_partner = [](int64_t d) {
    const int64_t c = d % 2;
    const int64_t chip = d / 2;
    const int64_t z = chip % 4;
    const int64_t y = (chip / 4) % 4;
    const int64_t x = (chip / 16) % 4;
    const int64_t x2 = (x + 2) % 4;
    const int64_t y2 = (y + 2) % 4;
    const int64_t z2 = (z + 2) % 4;
    const int64_t chip2 = x2 * 16 + y2 * 4 + z2;
    return chip2 * 2 + c;
  };

  std::vector<int64_t> canonical_world_da = shard_subgroup_0;  // INT_VEC_OK
  canonical_world_da.reserve(128);
  for (int64_t d : shard_subgroup_0) {
    canonical_world_da.push_back(antipodal_partner(d));
  }

  EXPECT_TRUE(RewriteAndTileOperationForSubgroup(
                  replicate_op_rank0.getValue().getDefiningOp(),
                  /*op_da=*/canonical_world_da, canonical_world_da)
                  .ok());
  EXPECT_TRUE(RewriteAndTileOperationForSubgroup(
                  replicate_op_rank8.getValue().getDefiningOp(),
                  /*op_da=*/canonical_world_da, canonical_world_da)
                  .ok());

  const std::vector<int64_t> rg_rank0 =  // INT_VEC_OK
      ExtractDenseIntAttrValues(replicate_op_rank0.getValue().getDefiningOp(),
                                "replica_groups");
  const std::vector<int64_t> rg_rank8 =  // INT_VEC_OK
      ExtractDenseIntAttrValues(replicate_op_rank8.getValue().getDefiningOp(),
                                "replica_groups");

  EXPECT_EQ(rg_rank0.size(), 128);
  EXPECT_EQ(rg_rank8.size(), 128);
  EXPECT_EQ(
      ExtractDenseIntAttrValues(replicate_op_rank0.getValue().getDefiningOp(),
                                "replica_groups"),
      ExtractDenseIntAttrValues(replicate_op_rank8.getValue().getDefiningOp(),
                                "replica_groups"));

  auto contains_device_pair = [](absl::Span<const int64_t> rg,
                                 absl::Span<const int64_t> resolved_da,
                                 int64_t dev_1, int64_t dev_2) {
    for (size_t i = 0; i + 1 < rg.size(); i += 2) {
      const int64_t d_a = resolved_da[rg[i]];
      const int64_t d_b = resolved_da[rg[i + 1]];
      if ((d_a == dev_1 && d_b == dev_2) || (d_a == dev_2 && d_b == dev_1)) {
        return true;
      }
    }
    return false;
  };
  EXPECT_TRUE(contains_device_pair(rg_rank0, canonical_world_da, 0, 84));
  EXPECT_TRUE(contains_device_pair(rg_rank8, canonical_world_da, 0, 84));

  auto has_consecutive_pair = [](absl::Span<const int64_t> rg, int64_t p1,
                                 int64_t p2) {
    for (size_t i = 0; i + 1 < rg.size(); i += 2) {
      if ((rg[i] == p1 && rg[i + 1] == p2) ||
          (rg[i] == p2 && rg[i + 1] == p1)) {
        return true;
      }
    }
    return false;
  };
  EXPECT_FALSE(has_consecutive_pair(rg_rank0, 44, 45));
  EXPECT_FALSE(has_consecutive_pair(rg_rank8, 44, 45));
}

TEST(DeviceAssignmentPipelineTest,
     ResolveDeviceAssignmentHsdpSymmetricAcrossRanks) {
  const std::vector<int64_t>  // INT_VEC_OK
      shard_subgroup_0 = {
          0,  1,  32, 33, 64, 65, 96,  97,  8,  9,  40, 41, 72, 73, 104, 105,
          16, 17, 48, 49, 80, 81, 112, 113, 24, 25, 56, 57, 88, 89, 120, 121,
          2,  3,  34, 35, 66, 67, 98,  99,  10, 11, 42, 43, 74, 75, 106, 107,
          18, 19, 50, 51, 82, 83, 114, 115, 26, 27, 58, 59, 90, 91, 122, 123};
  const std::vector<int64_t>  // INT_VEC_OK
      shard_subgroup_1 = {
          4,  5,  36, 37, 68, 69, 100, 101, 12, 13, 44, 45, 76, 77, 108, 109,
          20, 21, 52, 53, 84, 85, 116, 117, 28, 29, 60, 61, 92, 93, 124, 125,
          6,  7,  38, 39, 70, 71, 102, 103, 14, 15, 46, 47, 78, 79, 110, 111,
          22, 23, 54, 55, 86, 87, 118, 119, 30, 31, 62, 63, 94, 95, 126, 127};

  std::vector<int64_t> canonical_world_da = shard_subgroup_0;  // INT_VEC_OK
  canonical_world_da.insert(canonical_world_da.end(), shard_subgroup_1.begin(),
                            shard_subgroup_1.end());

  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<int64_t> resolved_rank0,  // INT_VEC_OK
      ResolveDeviceAssignment({
          OpDeviceAssignment{OpName::kDistributedAllReduce, canonical_world_da},
          OpDeviceAssignment{OpName::kDistributedAllReduce, canonical_world_da},
      }));

  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<int64_t> resolved_rank6,  // INT_VEC_OK
      ResolveDeviceAssignment({
          OpDeviceAssignment{OpName::kDistributedAllReduce, canonical_world_da},
          OpDeviceAssignment{OpName::kDistributedAllReduce, canonical_world_da},
      }));

  EXPECT_EQ(resolved_rank0.size(), 128);
  EXPECT_EQ(resolved_rank6.size(), 128);
  EXPECT_EQ(resolved_rank0, canonical_world_da);
  EXPECT_EQ(resolved_rank6, canonical_world_da);
  EXPECT_EQ(resolved_rank0, resolved_rank6);
}

TEST(DeviceAssignmentPipelineTest,
     MultiSlicePermutedLauncherRanksShardIotaAndReplicateRewrite) {
  // In a 2-slice (16-device) HSDP setup:
  // - The FSDP shard PG permutes its device_assignment (`da_shard`) to match
  //   the flattened slice subgroups ordered by topology_aware_mesh on Slice 0
  //   followed by Slice 1, and emits contiguous iota replica_groups
  //   {{0..7}, {8..15}}.
  // - The DCN replicate PG permutes its device_assignment (`da_replicate`) to
  //   match the flattened cross-slice replicate pairs, and emits contiguous
  //   iota replica_groups {{0, 1}, {2, 3}, ..., {14, 15}}.
  const std::vector<int64_t>  // INT_VEC_OK
      da_shard = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  const std::vector<int64_t>  // INT_VEC_OK
      da_replicate = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};

  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<int64_t> resolved_da,  // INT_VEC_OK
      ResolveDeviceAssignment({
          OpDeviceAssignment{OpName::kDistributedReduceScatter, da_shard},
          OpDeviceAssignment{OpName::kDistributedAllReduce, da_replicate},
      }));
  EXPECT_EQ(resolved_da, da_shard);

  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<mlir::stablehlo::StablehloDialect>();
  mlir_context.loadDialect<mlir::func::FuncDialect>();

  mlir::ModuleBuilder mb(mlir_context, "test_multislice_hsdp_module");
  mlir::func::FunctionBuilder fb(mb, "main");
  auto tensor_type =
      mlir::makeTensorType(mlir_context, {16}, mlir::ElementType::F32);
  mlir::MlirOp arg = mlir::func::Argument(fb, tensor_type);

  // FSDP shard collective emitted with iota replica_groups {{0..7}, {8..15}}.
  mlir::MlirOp shard_op = mlir::stablehlo::Neg(arg);
  auto shard_rg_type = mlir::RankedTensorType::get(
      {2, 8}, mlir::IntegerType::get(&mlir_context, 64));
  shard_op.getValue().getDefiningOp()->setAttr(
      "replica_groups",
      mlir::DenseIntElementsAttr::get(
          shard_rg_type, llvm::ArrayRef<int64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                  10, 11, 12, 13, 14, 15})));

  // DCN replicate collective emitted with iota replica_groups
  // {{0, 1}, {2, 3}, ..., {14, 15}}.
  mlir::MlirOp replicate_op = mlir::stablehlo::Abs(shard_op);
  auto rep_rg_type = mlir::RankedTensorType::get(
      {8, 2}, mlir::IntegerType::get(&mlir_context, 64));
  replicate_op.getValue().getDefiningOp()->setAttr(
      "replica_groups",
      mlir::DenseIntElementsAttr::get(
          rep_rg_type, llvm::ArrayRef<int64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                10, 11, 12, 13, 14, 15})));

  mlir::func::Return(fb, {replicate_op});
  auto module = mb.build();

  EXPECT_TRUE(RewriteAndTileOperationForSubgroup(
                  shard_op.getValue().getDefiningOp(), da_shard, resolved_da)
                  .ok());
  EXPECT_TRUE(
      RewriteAndTileOperationForSubgroup(
          replicate_op.getValue().getDefiningOp(), da_replicate, resolved_da)
          .ok());

  EXPECT_THAT(ExtractDenseIntAttrValues(shard_op.getValue().getDefiningOp(),
                                        "replica_groups"),
              testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                   14, 15));
  EXPECT_THAT(ExtractDenseIntAttrValues(replicate_op.getValue().getDefiningOp(),
                                        "replica_groups"),
              testing::ElementsAre(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6,
                                   14, 7, 15));
}

TEST(DeviceAssignmentPipelineTest,
     CrossSlicePermutedSubgroupsRemapsReplicaGroups) {
  const std::vector<int64_t>  // INT_VEC_OK
      op_da = {100001, 200001, 100000, 200000};
  const std::vector<int64_t>  // INT_VEC_OK
      resolved_da = {100000, 100001, 200000, 200001};

  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<mlir::stablehlo::StablehloDialect>();
  mlir_context.loadDialect<mlir::func::FuncDialect>();

  mlir::ModuleBuilder mb(mlir_context, "test_cross_slice_canonical_rg");
  mlir::func::FunctionBuilder fb(mb, "main");
  auto tensor_type =
      mlir::makeTensorType(mlir_context, {4}, mlir::ElementType::F32);
  mlir::MlirOp arg = mlir::func::Argument(fb, tensor_type);

  mlir::MlirOp replicate_op = mlir::stablehlo::Abs(arg);
  auto rep_rg_type = mlir::RankedTensorType::get(
      {2, 2}, mlir::IntegerType::get(&mlir_context, 64));
  replicate_op.getValue().getDefiningOp()->setAttr(
      "replica_groups",
      mlir::DenseIntElementsAttr::get(rep_rg_type,
                                      llvm::ArrayRef<int64_t>({0, 1, 2, 3})));

  mlir::func::Return(fb, {replicate_op});
  auto module = mb.build();

  EXPECT_TRUE(RewriteAndTileOperationForSubgroup(
                  replicate_op.getValue().getDefiningOp(), op_da, resolved_da)
                  .ok());

  EXPECT_THAT(ExtractDenseIntAttrValues(replicate_op.getValue().getDefiningOp(),
                                        "replica_groups"),
              testing::ElementsAre(0, 2, 1, 3));
}

TEST(DeviceAssignmentPipelineTest,
     RewriteAndTileOperationForSubgroupRejectsSmallerOpDeviceAssignment) {
  // An op over a subset of the resolved devices runs on a different device
  // set, so it cannot be remapped into the resolved assignment.
  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<mlir::stablehlo::StablehloDialect>();
  mlir_context.loadDialect<mlir::func::FuncDialect>();

  mlir::ModuleBuilder mb(mlir_context, "test_reject_subset_module");
  mlir::func::FunctionBuilder fb(mb, "main");
  auto tensor_type =
      mlir::makeTensorType(mlir_context, {4}, mlir::ElementType::F32);
  mlir::MlirOp arg = mlir::func::Argument(fb, tensor_type);

  mlir::MlirOp op = mlir::stablehlo::Abs(arg);
  mlir::Operation* raw_op = op.getValue().getDefiningOp();
  raw_op->setAttr("replica_groups",
                  mlir::DenseIntElementsAttr::get(
                      mlir::RankedTensorType::get(
                          {1, 2}, mlir::IntegerType::get(&mlir_context, 64)),
                      llvm::ArrayRef<int64_t>({0, 1})));

  mlir::func::Return(fb, {op});
  auto module = mb.build();

  const std::vector<int64_t> op_da = {2, 3};              // INT_VEC_OK
  const std::vector<int64_t> resolved_da = {0, 1, 2, 3};  // INT_VEC_OK
  const absl::Status status =
      RewriteAndTileOperationForSubgroup(raw_op, op_da, resolved_da);
  EXPECT_EQ(status.code(), error::kInvalidArgument);
  EXPECT_THAT(status.message(), testing::HasSubstr("is not a permutation"));
  EXPECT_THAT(ExtractDenseIntAttrValues(raw_op, "replica_groups"),
              testing::ElementsAre(0, 1));
}

TEST(DeviceAssignmentPipelineTest,
     MultiSliceRewritePreservesRankOrderAndPerSliceSymmetryForMegascale) {
  // 2 slices x 4 devices with megascale ids (slice = id / 100000).
  // `resolved_da` is the dp_shard order from topology_aware_mesh: within each
  // slice it is (x, y, z, core) order, which is not numeric, and both slices
  // share the same per-slice pattern.
  const std::vector<int64_t>  // INT_VEC_OK
      da_shard = {100000, 100002, 100001, 100003,
                  200000, 200002, 200001, 200003};
  // Op B: cross-slice, order-sensitive collective over replicate pairs.
  const std::vector<int64_t>  // INT_VEC_OK
      op_da_b = {100000, 200000, 100001, 200001,
                 100002, 200002, 100003, 200003};
  // Op C: intra-slice, order-sensitive subgroup whose rank order differs from
  // `resolved_da`.
  const std::vector<int64_t>  // INT_VEC_OK
      op_da_c = {100001, 100000, 100003, 100002,
                 200001, 200000, 200003, 200002};

  TT_ASSERT_OK_AND_ASSIGN(
      const std::vector<int64_t> resolved_da,  // INT_VEC_OK
      ResolveDeviceAssignment({
          OpDeviceAssignment{OpName::kDistributedReduceScatter, da_shard},
          OpDeviceAssignment{OpName::kDistributedAllGather, op_da_b},
          OpDeviceAssignment{OpName::kDistributedReduceScatter, op_da_c},
      }));
  ASSERT_EQ(resolved_da, da_shard);

  mlir::MLIRContext mlir_context;
  mlir_context.loadDialect<mlir::stablehlo::StablehloDialect>();
  mlir_context.loadDialect<mlir::func::FuncDialect>();

  mlir::ModuleBuilder mb(mlir_context, "test_multislice_megascale_module");
  mlir::func::FunctionBuilder fb(mb, "main");
  auto tensor_type =
      mlir::makeTensorType(mlir_context, {8}, mlir::ElementType::F32);
  mlir::MlirOp arg = mlir::func::Argument(fb, tensor_type);
  auto i64_type = mlir::IntegerType::get(&mlir_context, 64);

  // Op B with iota replica_groups {{0, 1}, {2, 3}, {4, 5}, {6, 7}}.
  mlir::MlirOp op_b = mlir::stablehlo::Neg(arg);
  mlir::Operation* raw_op_b = op_b.getValue().getDefiningOp();
  raw_op_b->setAttr("replica_groups",
                    mlir::DenseIntElementsAttr::get(
                        mlir::RankedTensorType::get({4, 2}, i64_type),
                        llvm::ArrayRef<int64_t>({0, 1, 2, 3, 4, 5, 6, 7})));

  // Op C (and a fresh copy for the asymmetric case) with iota replica_groups
  // {{0, 1, 2, 3}, {4, 5, 6, 7}}.
  auto c_rg_attr = mlir::DenseIntElementsAttr::get(
      mlir::RankedTensorType::get({2, 4}, i64_type),
      llvm::ArrayRef<int64_t>({0, 1, 2, 3, 4, 5, 6, 7}));
  mlir::MlirOp op_c = mlir::stablehlo::Abs(op_b);
  mlir::Operation* raw_op_c = op_c.getValue().getDefiningOp();
  raw_op_c->setAttr("replica_groups", c_rg_attr);
  mlir::MlirOp op_c_bad = mlir::stablehlo::Neg(op_c);
  mlir::Operation* raw_op_c_bad = op_c_bad.getValue().getDefiningOp();
  raw_op_c_bad->setAttr("replica_groups", c_rg_attr);

  mlir::func::Return(fb, {op_c_bad});
  auto module = mb.build();

  ASSERT_TRUE(
      RewriteAndTileOperationForSubgroup(raw_op_b, op_da_b, resolved_da).ok());
  ASSERT_TRUE(
      RewriteAndTileOperationForSubgroup(raw_op_c, op_da_c, resolved_da).ok());

  // Op B: rank_to_partition = [0, 4, 2, 6, 1, 5, 3, 7]; groups
  // {0,4},{2,6},{1,5},{3,7} are reordered by first element only.
  const std::vector<int64_t> rg_b =  // INT_VEC_OK
      ExtractDenseIntAttrValues(raw_op_b, "replica_groups");
  EXPECT_THAT(rg_b, testing::ElementsAre(0, 4, 1, 5, 2, 6, 3, 7));
  EXPECT_THAT(PhysicalGroups(rg_b, /*group_size=*/2, resolved_da),
              testing::UnorderedElementsAreArray(
                  IntendedGroups(op_da_b, /*group_size=*/2)));

  // Op C: rank_to_partition = [2, 0, 3, 1, 6, 4, 7, 5], so the groups become
  // {2,0,3,1},{6,4,7,5}. The old code sorted members inside each group and
  // produced {0,1,2,3},{4,5,6,7}, whose physical groups are
  // {100000,100002,100001,100003},{200000,200002,200001,200003}: the intended
  // {100001,100000,100003,100002},{200001,200000,200003,200002} with ranks
  // permuted, which fails the PhysicalGroups assertion below.
  const std::vector<int64_t> rg_c =  // INT_VEC_OK
      ExtractDenseIntAttrValues(raw_op_c, "replica_groups");
  EXPECT_THAT(rg_c, testing::ElementsAre(2, 0, 3, 1, 6, 4, 7, 5));
  EXPECT_THAT(PhysicalGroups(rg_c, /*group_size=*/4, resolved_da),
              testing::UnorderedElementsAreArray(
                  IntendedGroups(op_da_c, /*group_size=*/4)));
  // Local partition ids on `resolved_da`: x00000->0, x00002->1, x00001->2,
  // x00003->3 on both slices, so both groups localize to {2, 0, 3, 1}.
  EXPECT_TRUE(
      SamePerSliceCommunicationPattern(rg_c, /*group_size=*/4, resolved_da));

  // Asymmetric device order (e.g. arbitrary Kubernetes-pod/RANK order): the
  // two slices list their devices in different relative orders. The rewrite
  // stays correct, but Megascale cannot clone slice 0's localized collective
  // onto slice 1, so ProcessGroupTpu must order devices consistently per
  // slice.
  const std::vector<int64_t>  // INT_VEC_OK
      resolved_da_bad = {100000, 100002, 100001, 100003,
                         200001, 200003, 200000, 200002};
  ASSERT_TRUE(
      RewriteAndTileOperationForSubgroup(raw_op_c_bad, op_da_c, resolved_da_bad)
          .ok());
  // rank_to_partition = [2, 0, 3, 1, 4, 6, 5, 7].
  const std::vector<int64_t> rg_c_bad =  // INT_VEC_OK
      ExtractDenseIntAttrValues(raw_op_c_bad, "replica_groups");
  EXPECT_THAT(rg_c_bad, testing::ElementsAre(2, 0, 3, 1, 4, 6, 5, 7));
  EXPECT_THAT(PhysicalGroups(rg_c_bad, /*group_size=*/4, resolved_da_bad),
              testing::UnorderedElementsAreArray(
                  IntendedGroups(op_da_c, /*group_size=*/4)));
  // Slice 1 local ids: 200001->0, 200003->1, 200000->2, 200002->3, so slice 0
  // localizes to {2, 0, 3, 1} and slice 1 to {0, 2, 1, 3}.
  EXPECT_FALSE(SamePerSliceCommunicationPattern(rg_c_bad, /*group_size=*/4,
                                                resolved_da_bad));
}

}  // namespace
}  // namespace torch_tpu
