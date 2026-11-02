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

#include "csrc/ops/copy_from/cpu_to_tpu.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/arange.h"
#include "ATen/ops/equal.h"
#include "ATen/ops/zeros_like.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/compilation_cache.h"
#include "csrc/common/context_manager.h"
#include "csrc/common/context_states.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/materialize.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/eager/tpu_hooks.h"
#include "csrc/ops/copy_from/copy_common.h"
#include "csrc/ops/copy_from/tpu_to_cpu.h"
#include "csrc/pjrt/pjrt_state.h"
#include "csrc/pjrt/pjrt_transfer.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "xla/layout.h"

namespace torch_tpu {
namespace {

using ::absl_testing::IsOk;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::Matcher;
using ::testing::Optional;

// The tests copy 128x128 float32 tensors, whose default layout is row-major,
// {1,0:T(8,128)}.
constexpr int64_t kTensorSize = 128;

// Column-major, which takes as many bytes as the default layout.
xla::Layout ColumnMajorLayout() {
  return xla::Layout({0, 1}, {xla::Tile({8, 128})},
                     /*element_size_in_bits=*/0);
}

// Row-major with 8-element tiles, like SparseCore embedding tables.
xla::Layout SparseCoreLayout() {
  return xla::Layout({1, 0}, {xla::Tile({8})}, /*element_size_in_bits=*/0);
}

// Matches a buffer's layout by its dimension order and tiles. Unlike
// LayoutContext layouts, buffer layouts also record the dtype's width as their
// element size.
Matcher<const std::optional<CustomLayout>&> HasLayout(
    Matcher<const Indices&> minor_to_major,
    Matcher<const absl::InlinedVector<Indices, 1>&> tiles) {
  return Optional(
      AllOf(Field("minor_to_major", &CustomLayout::minor_to_major,
                  std::move(minor_to_major)),
            Field("tiles", &CustomLayout::tiles, std::move(tiles))));
}

// Gives new TPU buffers `layout` while in scope, like LayoutContext in Python.
class ScopedLayoutContext {
 public:
  explicit ScopedLayoutContext(xla::Layout layout) {
    PushContextState<LayoutContextState>(
        std::make_shared<const xla::Layout>(std::move(layout)));
  }
  ScopedLayoutContext(const ScopedLayoutContext&) = delete;
  ScopedLayoutContext& operator=(const ScopedLayoutContext&) = delete;
  ~ScopedLayoutContext() { PopContextState<LayoutContextState>(); }
};

// Returns a TPU tensor with a copy of `cpu`, in `layout` if given.
absl::StatusOr<at::Tensor> ToTpu(
    const at::Tensor& cpu, std::optional<xla::Layout> layout = std::nullopt) {
  std::optional<ScopedLayoutContext> layout_context;
  if (layout.has_value()) layout_context.emplace(*std::move(layout));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType element_type,
                      ConvertTo<mlir::ElementType>(cpu.scalar_type()));
  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef buffer,
      HostToDeviceTransfer::FromTensor(
          cpu.data_ptr(), cpu.storage().data_ptr().get(), element_type,
          absl::MakeConstSpan(cpu.sizes().data(), cpu.sizes().size()))
          .Run());
  TT_RETURN_IF_ERROR(buffer.Synchronize());
  return MakeTensor(std::move(buffer));
}

// Returns the values of the TPU tensor `tpu` on the CPU.
absl::StatusOr<at::Tensor> ToCpu(const at::Tensor& tpu) {
  TT_ASSIGN_OR_RETURN(const DeviceBufferRef buffer, GetBuffer(tpu));
  return CopyDeviceBufferRefToCpuTensor(buffer, TransferSyncMode::kIsolated);
}

class CopyCpuToTpuTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = "tpu"});
    ASSERT_EQ(AddTpuHooks(), absl::OkStatus());
    RegisterTpuAllocator();
    CompilationCache::GetInstance().SetOptions({});
  }

  static void TearDownTestSuite() {
    ShutDownMaterializationState();
    CompilationCache::ShutDown();
    PjrtBackend::GetInstance().Shutdown();
  }

  const at::Tensor src_ = at::arange(kTensorSize * kTensorSize, at::kFloat)
                              .reshape({kTensorSize, kTensorSize});
};

TEST_F(CopyCpuToTpuTest, KeepsNonDefaultLayoutOfDestination) {
  TT_ASSERT_OK_AND_ASSIGN(const at::Tensor dest,
                          ToTpu(at::zeros_like(src_), ColumnMajorLayout()));

  ASSERT_THAT(CopyCpuToTpu(src_, dest, TransferSyncMode::kIsolated), IsOk());

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef buffer, GetBuffer(dest));
  EXPECT_THAT(buffer.shape().layout(),
              HasLayout(ElementsAre(0, 1), ElementsAre(ElementsAre(8, 128))));
  TT_ASSERT_OK_AND_ASSIGN(const at::Tensor copied, ToCpu(dest));
  EXPECT_TRUE(at::equal(copied, src_));
  // The copy doesn't leave its layout behind for later buffers.
  EXPECT_FALSE(GetContextState<LayoutContextState>().has_value());
}

TEST_F(CopyCpuToTpuTest,
       KeepsNonDefaultLayoutOfDestinationWhenConvertingDtype) {
  TT_ASSERT_OK_AND_ASSIGN(const at::Tensor dest,
                          ToTpu(at::zeros_like(src_), ColumnMajorLayout()));

  ASSERT_THAT(
      CopyCpuToTpu(src_.to(at::kDouble), dest, TransferSyncMode::kIsolated),
      IsOk());

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef buffer, GetBuffer(dest));
  EXPECT_THAT(buffer.shape().layout(),
              HasLayout(ElementsAre(0, 1), ElementsAre(ElementsAre(8, 128))));
  TT_ASSERT_OK_AND_ASSIGN(const at::Tensor copied, ToCpu(dest));
  EXPECT_TRUE(at::equal(copied, src_));
}

TEST_F(CopyCpuToTpuTest, KeepsDefaultLayoutOfDestination) {
  TT_ASSERT_OK_AND_ASSIGN(const at::Tensor dest, ToTpu(at::zeros_like(src_)));

  ASSERT_THAT(CopyCpuToTpu(src_, dest, TransferSyncMode::kIsolated), IsOk());

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef buffer, GetBuffer(dest));
  EXPECT_THAT(buffer.shape().layout(),
              HasLayout(ElementsAre(1, 0), ElementsAre(ElementsAre(8, 128))));
  TT_ASSERT_OK_AND_ASSIGN(const at::Tensor copied, ToCpu(dest));
  EXPECT_TRUE(at::equal(copied, src_));
}

TEST_F(CopyCpuToTpuTest, UsesLayoutOfActiveLayoutContext) {
  TT_ASSERT_OK_AND_ASSIGN(const at::Tensor dest,
                          ToTpu(at::zeros_like(src_), ColumnMajorLayout()));
  const ScopedLayoutContext layout_context(SparseCoreLayout());
  const std::optional<LayoutContextState> active_layout =
      GetContextState<LayoutContextState>();

  ASSERT_THAT(CopyCpuToTpu(src_, dest, TransferSyncMode::kIsolated), IsOk());

  TT_ASSERT_OK_AND_ASSIGN(const DeviceBufferRef buffer, GetBuffer(dest));
  EXPECT_THAT(buffer.shape().layout(),
              HasLayout(ElementsAre(1, 0), ElementsAre(ElementsAre(8))));
  TT_ASSERT_OK_AND_ASSIGN(const at::Tensor copied, ToCpu(dest));
  EXPECT_TRUE(at::equal(copied, src_));
  // The LayoutContext is still active.
  EXPECT_EQ(GetContextState<LayoutContextState>(), active_layout);
}

}  // namespace
}  // namespace torch_tpu
