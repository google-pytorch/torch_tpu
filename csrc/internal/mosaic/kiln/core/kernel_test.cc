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

#include "csrc/internal/mosaic/kiln/core/kernel.h"

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "csrc/common/error_utils.h"
#include "csrc/internal/mosaic/kiln/core/context.h"
#include "csrc/internal/mosaic/kiln/core/control_flow.h"
#include "csrc/internal/mosaic/kiln/core/dtype.h"
#include "csrc/internal/mosaic/kiln/core/index_map.h"
#include "csrc/internal/mosaic/kiln/core/ref.h"
#include "csrc/internal/mosaic/kiln/core/scalar.h"
#include "csrc/internal/mosaic/kiln/core/shape.h"
#include "csrc/internal/mosaic/kiln/core/specs.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace torch_tpu::kiln {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(KernelTest, ReturnsErrorReportedInBody) {
  const absl::StatusOr<Kernel> kernel =
      CreateKernel(GridSpec{.name = "body_error"}, [] {
        internal::ReportError(TT_ERROR(error::kInvalidArgument) << "bad body");
      });

  EXPECT_THAT(kernel, StatusIs(error::kInvalidArgument, HasSubstr("bad body")));
}

TEST(KernelTest, ReturnsFirstOfSeveralReportedErrors) {
  const absl::StatusOr<Kernel> kernel =
      CreateKernel(GridSpec{.name = "two_errors"}, [] {
        internal::ReportError(TT_ERROR(error::kInvalidArgument) << "first");
        internal::ReportError(TT_ERROR(error::kInternal) << "second");
      });

  EXPECT_THAT(kernel, StatusIs(error::kInvalidArgument, HasSubstr("first")));
}

TEST(KernelTest, ReturnsErrorReportedInIndexMap) {
  const absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "index_map_error", .grid = {4}},
      InSpec{f32(8, 128), IndexMap([] {
               internal::ReportError(TT_ERROR(error::kInvalidArgument)
                                     << "bad index map");
               return IndexIndices{0, 0};
             })},
      [](VmemRef) {});

  EXPECT_THAT(kernel,
              StatusIs(error::kInvalidArgument, HasSubstr("bad index map")));
}

TEST(KernelTest, IndexMapErrorWinsOverWrongIndexRank) {
  const absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "index_map_error_wrong_rank", .grid = {4}},
      InSpec{f32(8, 128), IndexMap([] {
               internal::ReportError(TT_ERROR(error::kInvalidArgument)
                                     << "bad index map");
               return IndexIndices{};
             })},
      [](VmemRef) {});

  EXPECT_THAT(kernel,
              StatusIs(error::kInvalidArgument, HasSubstr("bad index map")));
}

TEST(KernelTest, FailingIndexMapShortCircuitsLaterIndexMapsAndBody) {
  bool second_map_called = false;
  bool body_called = false;
  const absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "index_map_short_circuit", .grid = {4}},
      InSpec{f32(8, 128), IndexMap([] {
               internal::ReportError(TT_ERROR(error::kInvalidArgument)
                                     << "first index map failed");
               return IndexIndices{0, 0};
             })},
      InSpec{f32(8, 128), IndexMap([&second_map_called] {
               second_map_called = true;
               return IndexIndices{0, 0};
             })},
      [&body_called](VmemRef, VmemRef) { body_called = true; });

  EXPECT_THAT(kernel, StatusIs(error::kInvalidArgument,
                               HasSubstr("first index map failed")));
  EXPECT_FALSE(second_map_called);
  EXPECT_FALSE(body_called);
}

TEST(KernelTest, SucceedingIndexMapFollowedBySecondIndexMapError) {
  const absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "second_index_map_error", .grid = {4}},
      InSpec{f32(8, 128), IndexMap([] { return IndexIndices{0, 0}; })},
      InSpec{f32(8, 128), IndexMap([] {
               internal::ReportError(TT_ERROR(error::kInvalidArgument)
                                     << "second index map failed");
               return IndexIndices{0, 0};
             })},
      [](VmemRef, VmemRef) {});

  EXPECT_THAT(kernel, StatusIs(error::kInvalidArgument,
                               HasSubstr("second index map failed")));
}

TEST(KernelTest, SucceedingIndexMapFollowedByBodyError) {
  const absl::StatusOr<Kernel> kernel = CreateKernel(
      GridSpec{.name = "body_after_index_map_error", .grid = {4}},
      InSpec{f32(8, 128), IndexMap([] { return IndexIndices{0, 0}; })},
      [](VmemRef) {
        internal::ReportError(TT_ERROR(error::kInvalidArgument)
                              << "body failed");
      });

  EXPECT_THAT(kernel,
              StatusIs(error::kInvalidArgument, HasSubstr("body failed")));
}

TEST(KernelTest, ReturnsErrorReportedInsideIfForAndRunScoped) {
  EXPECT_THAT(
      CreateKernel(GridSpec{.name = "if_error"},
                   [] {
                     If(I32(1).ToBool(), [] {
                       internal::ReportError(TT_ERROR(error::kInvalidArgument)
                                             << "inside if");
                     });
                   }),
      StatusIs(error::kInvalidArgument, HasSubstr("inside if")));
  EXPECT_THAT(
      CreateKernel(GridSpec{.name = "for_error"},
                   [] {
                     For(0, 4, 1, [](Scalar) {
                       internal::ReportError(TT_ERROR(error::kInvalidArgument)
                                             << "inside for");
                     });
                   }),
      StatusIs(error::kInvalidArgument, HasSubstr("inside for")));
  EXPECT_THAT(
      CreateKernel(GridSpec{.name = "run_scoped_error"},
                   [] {
                     RunScoped([] {
                       internal::ReportError(TT_ERROR(error::kInvalidArgument)
                                             << "inside run_scoped");
                     });
                   }),
      StatusIs(error::kInvalidArgument, HasSubstr("inside run_scoped")));
}

TEST(KernelTest, ErrorDoesNotLeakIntoNextKernel) {
  ASSERT_THAT(
      CreateKernel(
          GridSpec{.name = "failing"},
          [] { internal::ReportError(TT_ERROR(error::kInternal) << "boom"); }),
      StatusIs(error::kInternal));

  EXPECT_THAT(CreateKernel(GridSpec{.name = "succeeding"}, [] {}), IsOk());
}

TEST(KernelDeathTest, ReportErrorOutsideBuildKernelDies) {
  EXPECT_DEATH(internal::ReportError(TT_ERROR(error::kInternal) << "boom"),
               "Kiln operations may only be called inside BuildKernel");
}

TEST(KernelDeathTest, ReportErrorWithOkStatusDies) {
  EXPECT_DEATH(static_cast<void>(CreateKernel(
                   GridSpec{.name = "ok_error"},
                   [] { internal::ReportError(absl::OkStatus()); })),
               "ReportError\\(\\) requires a non-OK status");
}

TEST(KernelDeathTest, NestedCreateKernelDies) {
  EXPECT_DEATH(
      static_cast<void>(CreateKernel(GridSpec{.name = "outer"},
                                     [] {
                                       static_cast<void>(CreateKernel(
                                           GridSpec{.name = "inner"}, [] {}));
                                     })),
      "BuilderScope cannot be nested");
}

}  // namespace
}  // namespace torch_tpu::kiln
