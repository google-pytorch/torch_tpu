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

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/compilation_cache.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/materialize.h"
#include "torch_tpu/eager/structured_log_buffer.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/eager/tpu_hooks.h"
#include "torch_tpu/experimental/eager/materialize_new.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "xla/tsl/platform/statusor.h"

ABSL_DECLARE_FLAG(bool, torch_tpu_internal_enable_new_materialization);

namespace torch_tpu {
namespace {

class MaterializeRecoveryTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    // Standard CPU PJRT initialization for local testing
    const std::string device_type = "xla_cpu";
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = device_type});
    ASSERT_EQ(AddTpuHooks(), absl::OkStatus());
    RegisterTpuAllocator();
    CompilationCache::GetInstance().SetOptions({});
  }

  void SetUp() override {
    // Enable the experimental new materialization worker
    absl::SetFlag(&FLAGS_torch_tpu_internal_enable_new_materialization, true);
    ResetNewMaterializationState();
  }

  void TearDown() override { ResetNewMaterializationState(); }
};

TEST_F(MaterializeRecoveryTest, IndependentTensorsRecoverOnEagerError) {
  ScopedPythonContextCapturer capturer(OpName::kEmpty);
  const Shape shape(Dimensions{8}, mlir::ElementType::F32);

  // 1. Define a failing builder that triggers a compilation error in the
  // background worker
  auto failing_builder = [](mlir::MlirBuilder& builder,
                            absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    return TT_ERROR(error::kInternal) << "Simulated compilation failure";
  };

  // 2. Define a healthy builder that compiles successfully
  auto healthy_builder = [shape](mlir::MlirBuilder& builder,
                                 absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    if (!inputs.empty()) return DynamicMlirOpResults{inputs[0]};
    return DynamicMlirOpResults{
        BuildFillUninitialized(builder, shape.dtype(), shape.dimensions())};
  };

  // 3. Create a deferred tensor x using the failing builder
  TF_ASSERT_OK_AND_ASSIGN(std::vector<DeviceBufferRef> refs_x,
                          DeviceBufferList::CreateDeferred(
                              OpName::kAdd, failing_builder,
                              /*inputs=*/{}, OpParamCacheKeys::Empty(), {shape},
                              OpSplitMode::kNone, /*donated_indices=*/{},
                              /*skip_subgraph=*/true));
  DeviceBufferRef ref_x = refs_x[0];

  // Verify that the tensor is created successfully on the main thread
  // (deferred)
  ASSERT_TRUE(ref_x.state().ok());
  EXPECT_EQ(ref_x.state().value(), DeviceBufferListState::kDeferred);

  // 4. Trigger materialization on x. This dispatches to the background worker
  // queue.
  EXPECT_EQ(Materialize(ref_x, MaterializationReason::kExplicitSync),
            absl::OkStatus());

  // 5. Block on pending materializations. This must fail with the compiled
  // background error!
  absl::Status status_x = BlockOnPendingMaterializations();
  EXPECT_FALSE(status_x.ok());
  EXPECT_EQ(status_x.code(), error::kInternal);
  EXPECT_THAT(status_x.message(),
              testing::HasSubstr("Simulated compilation failure"));

  // 6. NOW CREATE A BRAND-NEW, INDEPENDENT TENSOR y
  TF_ASSERT_OK_AND_ASSIGN(std::vector<DeviceBufferRef> refs_y,
                          DeviceBufferList::CreateDeferred(
                              OpName::kEmpty, healthy_builder,
                              /*inputs=*/{}, OpParamCacheKeys::Empty(), {shape},
                              OpSplitMode::kNone, /*donated_indices=*/{},
                              /*skip_subgraph=*/true));
  DeviceBufferRef ref_y = refs_y[0];
  ASSERT_TRUE(ref_y.state().ok());
  EXPECT_EQ(ref_y.state().value(), DeviceBufferListState::kDeferred);

  // 7. Materialize y. Under our change, since last_status_ was reset to Ok,
  // this is allowed!
  EXPECT_EQ(Materialize(ref_y, MaterializationReason::kExplicitSync),
            absl::OkStatus());

  // 8. Block on pending materializations for y. This MUST succeed!
  absl::Status status_y = BlockOnPendingMaterializations();
  EXPECT_TRUE(status_y.ok())
      << "Independent tensor failed to materialize: " << status_y;
  ASSERT_TRUE(ref_y.state().ok());
  EXPECT_NE(ref_y.state().value(), DeviceBufferListState::kDeferred);

  // 9. VERIFY NATIVE PROMISE ERROR PROPAGATION
  // Even though the global error was reset, accessing x's data or materializing
  // a child z = x + 1 should still fail natively!

  // a) Direct access to x yields the compilation error:
  auto argument_status = ref_x.AwaitBuffer();
  EXPECT_FALSE(argument_status.ok());
  EXPECT_EQ(argument_status.status().code(), error::kInternal);
  EXPECT_THAT(argument_status.status().message(),
              testing::HasSubstr("Simulated compilation failure"));

  // Repeating materialization on x will no-op and will not block, as X is
  // already in the "materialized error" state
  EXPECT_EQ(Materialize(ref_x, MaterializationReason::kExplicitSync),
            absl::OkStatus());
  absl::Status status_x_retry = BlockOnPendingMaterializations();
  EXPECT_TRUE(status_x_retry.ok());

  // ...but the inner error on ref_x is still the same:
  auto after_retry_status = ref_x.AwaitBuffer();
  EXPECT_FALSE(argument_status.ok());
  EXPECT_EQ(argument_status.status().code(), error::kInternal);
  EXPECT_THAT(argument_status.status().message(),
              testing::HasSubstr("Simulated compilation failure"));

  // b) Downstream operations using x also fail natively:
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> refs_z,
      DeviceBufferList::CreateDeferred(
          OpName::kAdd, healthy_builder, {ref_x}, OpParamCacheKeys::Empty(),
          {shape}, OpSplitMode::kNone, /*donated_indices=*/{},
          /*skip_subgraph=*/true));
  DeviceBufferRef ref_z = refs_z[0];
  ASSERT_TRUE(ref_z.state().ok());
  EXPECT_EQ(ref_z.state().value(), DeviceBufferListState::kDeferred);

  // Materialize z (depends on failed x)
  EXPECT_EQ(Materialize(ref_z, MaterializationReason::kExplicitSync),
            absl::OkStatus());

  // Blocking on materializations of z should fail with the exact same error,
  // as the error successfully propagated from x -> z!
  absl::Status status_z = BlockOnPendingMaterializations();
  EXPECT_FALSE(status_z.ok());
  EXPECT_EQ(status_z.code(), error::kInternal);
  EXPECT_THAT(status_z.message(),
              testing::HasSubstr("Simulated compilation failure"));
}

}  // namespace
}  // namespace torch_tpu
