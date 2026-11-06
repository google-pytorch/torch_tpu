/*
 * Copyright 2025 Google LLC
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

#include "csrc/common/compilation.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/compilation_test_helper.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/excess_precision.h"
#include "csrc/common/mlir_test_modules.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/pjrt/pjrt_state.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/computation_layout.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

using absl_testing::StatusIs;
using testing::Contains;
using testing::ElementsAre;
using testing::Pair;

// Initializes the TPU PjRt client once per test suite.
class TpuClientTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    // This must be done before MakeCompilerOptions() is called, as the latter
    // depends on the PjRt client.
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = "tpu"});
    ABSL_CHECK_OK(PjrtBackend::GetInstance().EnsureInitialized());
  }
};

class MakeCompilerOptionsTest : public TpuClientTest {};

UniqueCompileOptions GetCompileOptions(const CompilationMode mode) {
  return GetCompilationSpec(mode).xla_compile_options;
}

TEST_F(MakeCompilerOptionsTest, DefaultOptionsInEagerMode) {
  UniqueCompileOptions options =
      GetCompileOptions(CompilationMode::kFastCompile);
  EXPECT_EQ(options->executable_build_options.optimization_level(),
            xla::ExecutionOptions::EFFORT_O1);
  EXPECT_THAT(options->env_option_overrides,
              Contains(Pair("xla_tpu_autofdo",
                            xla::CompileOptions::OptionOverride(false))));
  EXPECT_THAT(options->env_option_overrides,
              Contains(Pair("xla_tpu_vmem_scavenging_mode",
                            xla::CompileOptions::OptionOverride(
                                "SAFE_IFF_KERNEL_REQUIRED"))));
}

TEST_F(MakeCompilerOptionsTest, DefaultOptionsInTorchCompileMode) {
  UniqueCompileOptions options =
      GetCompileOptions(CompilationMode::kFastRuntime);
  EXPECT_EQ(options->executable_build_options.optimization_level(),
            xla::ExecutionOptions::EFFORT_UNKNOWN);
  EXPECT_THAT(options->env_option_overrides,
              Contains(Pair("xla_tpu_autofdo",
                            xla::CompileOptions::OptionOverride(false))));
  EXPECT_THAT(options->env_option_overrides,
              Contains(Pair("xla_tpu_vmem_scavenging_mode",
                            xla::CompileOptions::OptionOverride(
                                "SAFE_IFF_KERNEL_REQUIRED"))));
}

TEST_F(MakeCompilerOptionsTest, CompilerOptionOverrides) {
  {
    ScopedCompilerOptionOverrides overrides({{"xla_optimization_level", "O1"}});
    UniqueCompileOptions options =
        GetCompileOptions(CompilationMode::kFastCompile);
    EXPECT_EQ(options->executable_build_options.optimization_level(),
              xla::ExecutionOptions::EFFORT_O1);
  }

  {
    ScopedCompilerOptionOverrides overrides({{"xla_optimization_level", "O2"}});
    UniqueCompileOptions options =
        GetCompileOptions(CompilationMode::kFastCompile);
    EXPECT_EQ(options->executable_build_options.optimization_level(),
              xla::ExecutionOptions::EFFORT_O2);
  }
}

TEST_F(MakeCompilerOptionsTest, LocksExcessPrecision) {
  // Ensure the default compilation context is initialized hermetically.
  GetCompileOptions(CompilationMode::kFastCompile);
  EXPECT_TRUE(IsAllowExcessPrecisionLocked());
  const bool current = GetAllowExcessPrecision();
  SetAllowExcessPrecision(current);
  EXPECT_THROW(SetAllowExcessPrecision(!current), torch_tpu::TtError);
}

class TupleArgumentsTest : public TpuClientTest {
 protected:
  void SetUp() override {
    TT_ASSERT_OK_AND_ASSIGN(client_, PjrtBackend::GetInstance().GetClient());
  }

  // Compiles a program that takes the single parameter `parameter_shape` with
  // `parameter_is_tupled_arguments`.
  absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
  CompileWithTupledArguments(const xla::Shape& parameter_shape) {
    xla::XlaBuilder builder("tupled_arguments");
    const xla::XlaOp parameter =
        xla::Parameter(&builder, 0, parameter_shape, "arguments");
    TT_ASSIGN_OR_RETURN(const xla::XlaComputation computation,
                        builder.Build(xla::GetTupleElement(parameter, 0)));
    xla::CompileOptions options;
    options.parameter_is_tupled_arguments = true;
    return client_->CompileAndLoad(computation, std::move(options));
  }

  xla::PjRtClient* client_ = nullptr;
};

// Checks that `loaded_executable` reports the device shapes of 3 arguments:
// f32[2,3], f32[4,5] and f32[6], in this order.
void ExpectDeviceShapesOfThreeArguments(
    std::unique_ptr<xla::PjRtLoadedExecutable> loaded_executable) {
  TT_ASSERT_OK_AND_ASSIGN(
      SharedLoadedExecutableWithMetadata executable,
      LoadedExecutableWithMetadata::MakeShared(std::move(loaded_executable)));
  const std::vector<std::optional<xla::Shape>>& shapes =
      executable->parameter_device_shapes();
  ASSERT_EQ(shapes.size(), 3);
  for (const std::optional<xla::Shape>& shape : shapes) {
    ASSERT_TRUE(shape.has_value());
    EXPECT_TRUE(shape->has_layout());
  }
  EXPECT_THAT(shapes[0]->dimensions(), ElementsAre(2, 3));
  EXPECT_THAT(shapes[1]->dimensions(), ElementsAre(4, 5));
  EXPECT_THAT(shapes[2]->dimensions(), ElementsAre(6));
}

TEST_F(TupleArgumentsTest, ReportsParameterDeviceShapesOfTupledArguments) {
  // PJRT takes the elements of the tuple as 3 separate arguments.
  TT_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> loaded_executable,
      CompileWithTupledArguments(xla::ShapeUtil::MakeTupleShape(
          {xla::ShapeUtil::MakeShape(xla::F32, {2, 3}),
           xla::ShapeUtil::MakeShape(xla::F32, {4, 5}),
           xla::ShapeUtil::MakeShape(xla::F32, {6})})));
  ExpectDeviceShapesOfThreeArguments(std::move(loaded_executable));
}

// PJRT only supports one level of tupled arguments (see
// xla::GetParameterShapes), so nested tuples never reach
// GetStaticParameterDeviceShapes. If that changes, checks that they are
// flattened in order.
TEST_F(TupleArgumentsTest, RejectsOrFlattensNestedTupledArguments) {
  absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> loaded_executable =
      CompileWithTupledArguments(xla::ShapeUtil::MakeTupleShape(
          {xla::ShapeUtil::MakeShape(xla::F32, {2, 3}),
           xla::ShapeUtil::MakeTupleShape(
               {xla::ShapeUtil::MakeShape(xla::F32, {4, 5}),
                xla::ShapeUtil::MakeShape(xla::F32, {6})})}));
  if (!loaded_executable.ok()) {
    EXPECT_THAT(loaded_executable.status(), StatusIs(error::kInvalidArgument));
    return;
  }
  ExpectDeviceShapesOfThreeArguments(*std::move(loaded_executable));
}

// TPU programs with more than 2000 arguments take them as a single tuple.
constexpr int kFlatArguments = 2000;
constexpr int kTupledArguments = kFlatArguments + 1;

// Returns the parameter shapes of the entry computation of `executable`.
std::vector<xla::Shape> GetEntryParameterShapes(
    const xla::PjRtLoadedExecutable& executable) {
  absl::StatusOr<std::vector<std::shared_ptr<xla::HloModule>>> modules =
      executable.GetHloModules();
  ABSL_CHECK_OK(modules);             // CRASH_OK
  ABSL_CHECK_EQ(modules->size(), 1);  // CRASH_OK
  const xla::ComputationLayout& layout =
      (*modules)[0]->entry_computation_layout();
  std::vector<xla::Shape> shapes;
  for (int i = 0; i < layout.parameter_count(); ++i) {
    shapes.push_back(layout.parameter_shape(i));
  }
  return shapes;
}

TEST_F(TupleArgumentsTest, TuplesArgumentsAboveThreshold) {
  TT_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> executable,
      CompileAndLoadMlirModule(*client_,
                               MakeSumOfArgumentsModule(kTupledArguments),
                               xla::CompileOptions()));

  const std::vector<xla::Shape> parameter_shapes =
      GetEntryParameterShapes(*executable);
  ASSERT_EQ(parameter_shapes.size(), 1);
  ASSERT_TRUE(parameter_shapes[0].IsTuple());
  EXPECT_EQ(parameter_shapes[0].tuple_shapes().size(), kTupledArguments);
}

TEST_F(TupleArgumentsTest, KeepsArgumentsFlatAtThreshold) {
  TT_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> executable,
      CompileAndLoadMlirModule(*client_,
                               MakeSumOfArgumentsModule(kFlatArguments),
                               xla::CompileOptions()));

  const std::vector<xla::Shape> parameter_shapes =
      GetEntryParameterShapes(*executable);
  ASSERT_EQ(parameter_shapes.size(), kFlatArguments);
  for (const xla::Shape& shape : parameter_shapes) {
    EXPECT_TRUE(shape.IsArray());
  }
}

TEST_F(TupleArgumentsTest, TuplesExplicitArgumentLayouts) {
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::F32, {128, 128});
  *shape.mutable_layout() = xla::LayoutUtil::MakeLayout({1, 0});
  xla::CompileOptions options;
  options.argument_layouts = std::vector<xla::Shape>(kTupledArguments, shape);

  TT_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> executable,
      CompileAndLoadMlirModule(*client_,
                               MakeSumOfArgumentsModule(kTupledArguments),
                               std::move(options)));

  const std::vector<xla::Shape> parameter_shapes =
      GetEntryParameterShapes(*executable);
  ASSERT_EQ(parameter_shapes.size(), 1);
  ASSERT_TRUE(parameter_shapes[0].IsTuple());
  EXPECT_EQ(parameter_shapes[0].tuple_shapes().size(), kTupledArguments);
}

TEST_F(TupleArgumentsTest, KeepsArgumentLayoutsOfAlreadyTupledArguments) {
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::F32, {128, 128});
  *shape.mutable_layout() = xla::LayoutUtil::MakeLayout({1, 0});
  xla::CompileOptions options;
  options.parameter_is_tupled_arguments = true;
  options.argument_layouts =
      std::vector<xla::Shape>{xla::ShapeUtil::MakeTupleShape(
          std::vector<xla::Shape>(kTupledArguments, shape))};

  // The layouts aren't wrapped into a nested tuple again.
  TT_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> executable,
      CompileAndLoadMlirModule(*client_,
                               MakeSumOfArgumentsModule(kTupledArguments),
                               std::move(options)));

  const std::vector<xla::Shape> parameter_shapes =
      GetEntryParameterShapes(*executable);
  ASSERT_EQ(parameter_shapes.size(), 1);
  ASSERT_TRUE(parameter_shapes[0].IsTuple());
  EXPECT_EQ(parameter_shapes[0].tuple_shapes().size(), kTupledArguments);
}

}  // namespace
}  // namespace torch_tpu
