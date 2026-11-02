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

#include "csrc/distributed/alltoall.h"

#include <cstdint>
#include <vector>

#include "csrc/common/status_test_utils.h"
#include "csrc/distributed/types.h"
#include "csrc/ops/op_builder_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {
namespace {

using ::testing::ElementsAre;

constexpr int kBits8 = 8;
constexpr int kBits16 = 16;
constexpr int kBits32 = 32;
constexpr int kBits64 = 64;

DeviceGroupList MakeEightReplicaGroup() {
  return DeviceGroupList{{0, 1, 2, 3, 4, 5, 6, 7}};
}

// Verifies that `custom_call` is a `ragged_all_to_all` with `channel_id = 1`
// whose four metadata operands are compile-time `stablehlo.constant` vectors.
void ExpectConstantMetadataRaggedCustomCall(
    mlir::stablehlo::CustomCallOp custom_call) {
  ASSERT_TRUE(custom_call);
  EXPECT_EQ(custom_call.getCallTargetName(), "ragged_all_to_all");
  ASSERT_EQ(custom_call.getInputs().size(), 6);

  auto backend_config = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
      custom_call->getAttr("backend_config"));
  ASSERT_TRUE(backend_config);
  auto channel_id_attr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
      backend_config.get("channel_id"));
  ASSERT_TRUE(channel_id_attr);
  EXPECT_EQ(channel_id_attr.getInt(), 1);

  for (int operand_idx = 2; operand_idx < 6; ++operand_idx) {
    auto const_op = mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(
        custom_call.getInputs()[operand_idx].getDefiningOp());
    ASSERT_TRUE(const_op) << "operand " << operand_idx << " must be constant";
  }
}

class AllToAllShloBuilderTest : public testing::Test {
 protected:
  AllToAllShloBuilderTest()
      : context_(), module_builder_(context_, mlir::unknownLoc(context_)) {
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
  }

  mlir::MlirBuilder& builder() { return module_builder_; }

 private:
  mlir::MLIRContext context_;
  mlir::ModuleBuilder module_builder_;
};

TEST(AllToAllFactorTest, ComputeLocalGcdAndSelectFactor) {
  EXPECT_EQ(Compute1DRaggedAllToAllLocalGcd(0, 0, {}, {}, {}, {}), 0);

  const std::vector<int32_t> in_offsets = {0, 1024, 1024, 3072};
  const std::vector<int32_t> send_sizes = {1024, 0, 2048, 1024};
  const std::vector<int32_t> out_offsets = {0, 2048, 0, 1024};
  const std::vector<int32_t> recv_sizes = {2048, 1024, 0, 1024};
  EXPECT_EQ(Compute1DRaggedAllToAllLocalGcd(
                /*input_size=*/4096, /*output_size=*/4096, in_offsets,
                send_sizes, out_offsets, recv_sizes),
            1024);

  EXPECT_EQ(Select1DRaggedAllToAllFactor(2048, kBits16), 1024);
  EXPECT_EQ(Select1DRaggedAllToAllFactor(1024, kBits32), 1024);
  EXPECT_EQ(Select1DRaggedAllToAllFactor(4096, kBits8), 1024);
  EXPECT_EQ(Select1DRaggedAllToAllFactor(512, kBits16), 1);
  EXPECT_EQ(Select1DRaggedAllToAllFactor(0, kBits16), 1);
  EXPECT_EQ(Select1DRaggedAllToAllFactor(4096, kBits64), 1);
}

TEST_F(AllToAllShloBuilderTest, UnevenSplits1DFactoredConstantMetadata) {
  constexpr int64_t kFlatFactor = k1DRaggedAllToAllTileElements;
  constexpr int64_t kInputElems = 5 * kFlatFactor;
  constexpr int64_t kOutputElems = 12 * kFlatFactor;
  mlir::MlirOp input =
      BuildFillUninitialized(builder(), mlir::ElementType::F32, {kInputElems});
  mlir::MlirOp output =
      BuildFillUninitialized(builder(), mlir::ElementType::F32, {kOutputElems});

  const std::vector<int32_t> input_offsets = {0,    1024, 2048, 2048,
                                              3072, 4096, 4096, 5120};
  const std::vector<int32_t> send_sizes = {1024, 1024, 0,    1024,
                                           1024, 0,    1024, 0};
  const std::vector<int32_t> output_offsets = {0,    2048, 0,    4096,
                                               1024, 0,    3072, 0};
  const std::vector<int32_t> recv_sizes = {2048, 2048, 1024, 2048,
                                           1024, 2048, 2048, 0};

  TT_ASSERT_OK_AND_ASSIGN(
      mlir::MlirOp result,
      BuildDistributedAllToAllBaseUnevenSplitsShlo(
          input, output, input_offsets, send_sizes, output_offsets, recv_sizes,
          kFlatFactor, MakeEightReplicaGroup()));
  EXPECT_THAT(GetTensorTypeOrDie(result).getShape(), ElementsAre(kOutputElems));

  auto post_reshape = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(
      result.getValue().getDefiningOp());
  ASSERT_TRUE(post_reshape);
  const auto inner_type =
      mlir::cast<mlir::RankedTensorType>(post_reshape.getOperand().getType());
  EXPECT_THAT(inner_type.getShape(), ElementsAre(12, 8, 128));

  auto custom_call = mlir::dyn_cast_or_null<mlir::stablehlo::CustomCallOp>(
      post_reshape.getOperand().getDefiningOp());
  ExpectConstantMetadataRaggedCustomCall(custom_call);

  auto in_reshape = mlir::dyn_cast_or_null<mlir::stablehlo::ReshapeOp>(
      custom_call.getInputs()[0].getDefiningOp());
  ASSERT_TRUE(in_reshape);
  EXPECT_THAT(
      mlir::cast<mlir::RankedTensorType>(in_reshape.getType()).getShape(),
      ElementsAre(5, 8, 128));

  auto out_offsets_const = mlir::cast<mlir::stablehlo::ConstantOp>(
      custom_call.getInputs()[4].getDefiningOp());
  auto out_offsets_values = out_offsets_const.getValue().getValues<int32_t>();
  EXPECT_THAT(std::vector<int32_t>(out_offsets_values.begin(),
                                   out_offsets_values.end()),
              ElementsAre(0, 2, 0, 4, 1, 0, 3, 0));
}

TEST_F(AllToAllShloBuilderTest, UnevenSplits2DUnfactoredConstantMetadata) {
  constexpr int64_t kRows = 16;
  constexpr int64_t kHidden = 32;
  mlir::MlirOp input = BuildFillUninitialized(
      builder(), mlir::ElementType::BF16, {kRows, kHidden});
  mlir::MlirOp output = BuildFillUninitialized(
      builder(), mlir::ElementType::BF16, {kRows, kHidden});

  const std::vector<int32_t> input_offsets = {0, 2, 4, 6, 8, 10, 12, 14};
  const std::vector<int32_t> send_sizes = {2, 2, 2, 2, 2, 2, 2, 2};
  const std::vector<int32_t> output_offsets = {0, 3, 5, 7, 9, 11, 13, 15};
  const std::vector<int32_t> recv_sizes = {3, 2, 2, 2, 2, 2, 2, 1};

  TT_ASSERT_OK_AND_ASSIGN(
      mlir::MlirOp result,
      BuildDistributedAllToAllBaseUnevenSplitsShlo(
          input, output, input_offsets, send_sizes, output_offsets, recv_sizes,
          /*flat_factor=*/1, MakeEightReplicaGroup()));
  EXPECT_THAT(GetTensorTypeOrDie(result).getShape(),
              ElementsAre(kRows, kHidden));

  auto custom_call = mlir::dyn_cast_or_null<mlir::stablehlo::CustomCallOp>(
      result.getValue().getDefiningOp());
  ExpectConstantMetadataRaggedCustomCall(custom_call);

  auto out_offsets_const = mlir::cast<mlir::stablehlo::ConstantOp>(
      custom_call.getInputs()[4].getDefiningOp());
  auto out_offsets_values = out_offsets_const.getValue().getValues<int32_t>();
  EXPECT_THAT(std::vector<int32_t>(out_offsets_values.begin(),
                                   out_offsets_values.end()),
              ElementsAre(0, 3, 5, 7, 9, 11, 13, 15));
}

}  // namespace
}  // namespace torch_tpu
