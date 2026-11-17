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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/compilation_test_helper.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/excess_precision.h"
#include "csrc/common/mlir_test_modules.h"
#include "csrc/common/shape.h"
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
              Contains(Pair("xla_jf_program_hbm_alignment_in_kib",
                            xla::CompileOptions::OptionOverride(int64_t{1}))));
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
              Contains(Pair("xla_jf_program_hbm_alignment_in_kib",
                            xla::CompileOptions::OptionOverride(int64_t{1}))));
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

  {
    ScopedCompilerOptionOverrides overrides(
        {{"xla_jf_program_hbm_alignment_in_kib", "16"}});
    UniqueCompileOptions options =
        GetCompileOptions(CompilationMode::kFastCompile);
    EXPECT_THAT(
        options->env_option_overrides,
        ElementsAre(
            Pair("xla_jf_program_hbm_alignment_in_kib",
                 xla::CompileOptions::OptionOverride(std::string("16"))),
            Pair("xla_tpu_autofdo", xla::CompileOptions::OptionOverride(false)),
            Pair("xla_tpu_vmem_scavenging_mode",
                 xla::CompileOptions::OptionOverride(
                     std::string("SAFE_IFF_KERNEL_REQUIRED")))));
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
      const SharedLoadedExecutableWithMetadata executable,
      LoadedExecutableWithMetadata::MakeShared(std::move(loaded_executable)));
  const absl::Span<const xla::Shape> shapes =
      executable->parameter_device_shapes();
  ASSERT_EQ(shapes.size(), 3);
  for (const xla::Shape& shape : shapes) {
    EXPECT_TRUE(shape.IsArray());
    EXPECT_TRUE(shape.is_static());
    EXPECT_TRUE(shape.has_layout());
  }
  EXPECT_THAT(shapes[0].dimensions(), ElementsAre(2, 3));
  EXPECT_THAT(shapes[1].dimensions(), ElementsAre(4, 5));
  EXPECT_THAT(shapes[2].dimensions(), ElementsAre(6));
}

class DeviceShapesTest : public TpuClientTest {
 protected:
  void SetUp() override {
    TT_ASSERT_OK_AND_ASSIGN(client_, PjrtBackend::GetInstance().GetClient());
  }

  absl::StatusOr<SharedLoadedExecutableWithMetadata> CompileTuplePassthrough() {
    xla::XlaBuilder builder("tupled_outputs");
    const xla::XlaOp p0 = xla::Parameter(
        &builder, 0, xla::ShapeUtil::MakeShape(xla::F32, {2, 3}), "p0");
    const xla::XlaOp p1 = xla::Parameter(
        &builder, 1, xla::ShapeUtil::MakeShape(xla::F32, {4, 5}), "p1");
    TT_ASSIGN_OR_RETURN(const xla::XlaComputation computation,
                        builder.Build(xla::Tuple(&builder, {p0, p1})));
    TT_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::PjRtLoadedExecutable> loaded_executable,
        client_->CompileAndLoad(computation, xla::CompileOptions()));
    return LoadedExecutableWithMetadata::MakeShared(
        std::move(loaded_executable));
  }

  xla::PjRtClient* client_ = nullptr;
};

TEST_F(DeviceShapesTest, FlattensTupleOutputDeviceShapes) {
  TT_ASSERT_OK_AND_ASSIGN(const SharedLoadedExecutableWithMetadata executable,
                          CompileTuplePassthrough());

  ASSERT_NE(executable->shared_output_device_shapes(), nullptr);
  ASSERT_EQ(executable->output_device_shapes().size(), 2);
  EXPECT_TRUE(executable->output_device_shapes()[0].has_layout());
  EXPECT_TRUE(executable->output_device_shapes()[1].has_layout());
  EXPECT_THAT(executable->output_device_shapes()[0].dimensions(),
              ElementsAre(2, 3));
  EXPECT_THAT(executable->output_device_shapes()[1].dimensions(),
              ElementsAre(4, 5));
  ASSERT_EQ(executable->output_shapes().size(), 2);
  EXPECT_EQ(executable->output_shapes(), executable->output_device_shapes());
  ASSERT_EQ(executable->parameter_device_shapes().size(), 2);
  EXPECT_EQ(executable->output_device_shapes()[0],
            executable->parameter_device_shapes()[0]);
  EXPECT_EQ(executable->output_device_shapes()[1],
            executable->parameter_device_shapes()[1]);
}

TEST_F(DeviceShapesTest,
       BoundedDynamicParameterAndOutputPreserveDynamicDimensionFlags) {
  xla::XlaBuilder dyn_out_builder("bounded_dynamic_output");
  const xla::XlaOp static_in = xla::Parameter(
      &dyn_out_builder, 0, xla::ShapeUtil::MakeShape(xla::F32, {8, 8}), "x");
  const xla::XlaOp dyn_size = xla::Parameter(
      &dyn_out_builder, 1, xla::ShapeUtil::MakeShape(xla::S32, {}), "n");
  TT_ASSERT_OK_AND_ASSIGN(
      const xla::XlaComputation dyn_out_comp,
      dyn_out_builder.Build(xla::SetDimensionSize(static_in, dyn_size, 0)));
  TT_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> raw_dyn_out,
      client_->CompileAndLoad(dyn_out_comp, xla::CompileOptions()));
  TT_ASSERT_OK_AND_ASSIGN(
      const SharedLoadedExecutableWithMetadata dyn_out_exec,
      LoadedExecutableWithMetadata::MakeShared(std::move(raw_dyn_out)));

  ASSERT_EQ(dyn_out_exec->parameter_device_shapes().size(), 2);
  EXPECT_TRUE(dyn_out_exec->parameter_device_shapes()[0].is_static());
  EXPECT_TRUE(dyn_out_exec->parameter_device_shapes()[1].is_static());
  ASSERT_EQ(dyn_out_exec->output_device_shapes().size(), 1);
  EXPECT_TRUE(dyn_out_exec->output_device_shapes()[0].is_dynamic());
  EXPECT_TRUE(dyn_out_exec->output_device_shapes()[0].is_dynamic_dimension(0));
  EXPECT_FALSE(dyn_out_exec->output_device_shapes()[0].is_dynamic_dimension(1));
  ASSERT_EQ(dyn_out_exec->output_shapes().size(), 1);
  EXPECT_THAT(dyn_out_exec->output_shapes()[0].dimensions(), ElementsAre(8, 8));

  xla::XlaBuilder dyn_param_builder("bounded_dynamic_parameter");
  const xla::Shape dyn_param_shape = xla::ShapeUtil::MakeShape(
      xla::F32, {8, 8}, /*dynamic_dimensions=*/{true, false});
  const xla::XlaOp dyn_in =
      xla::Parameter(&dyn_param_builder, 0, dyn_param_shape, "dyn_x");
  TT_ASSERT_OK_AND_ASSIGN(const xla::XlaComputation dyn_param_comp,
                          dyn_param_builder.Build(xla::Add(dyn_in, dyn_in)));
  TT_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> raw_dyn_param,
      client_->CompileAndLoad(dyn_param_comp, xla::CompileOptions()));
  TT_ASSERT_OK_AND_ASSIGN(
      const SharedLoadedExecutableWithMetadata dyn_param_exec,
      LoadedExecutableWithMetadata::MakeShared(std::move(raw_dyn_param)));

  ASSERT_EQ(dyn_param_exec->parameter_device_shapes().size(), 1);
  EXPECT_TRUE(dyn_param_exec->parameter_device_shapes()[0].is_dynamic());
  EXPECT_TRUE(
      dyn_param_exec->parameter_device_shapes()[0].is_dynamic_dimension(0));
  ASSERT_EQ(dyn_param_exec->output_device_shapes().size(), 1);
  EXPECT_TRUE(dyn_param_exec->output_device_shapes()[0].is_dynamic());
  EXPECT_TRUE(
      dyn_param_exec->output_device_shapes()[0].is_dynamic_dimension(0));
}

TEST_F(DeviceShapesTest,
       WithArgumentIndicesPreservesParameterAndOutputDeviceShapes) {
  TT_ASSERT_OK_AND_ASSIGN(const SharedLoadedExecutableWithMetadata executable,
                          CompileTuplePassthrough());
  const SharedLoadedExecutableWithMetadata with_indices =
      LoadedExecutableWithMetadata::WithArgumentIndices(executable, {0, 1});
  EXPECT_EQ(with_indices->parameter_device_shapes(),
            executable->parameter_device_shapes());
  EXPECT_EQ(with_indices->shared_output_device_shapes(),
            executable->shared_output_device_shapes());
}

TEST_F(DeviceShapesTest,
       PreservesParameterAndOutputDeviceShapesAcrossSerialization) {
  // 1. Static tuple-output executable preserves compiled shapes across
  // Tier-2/Tier-3 serialization round-trips.
  TT_ASSERT_OK_AND_ASSIGN(const SharedLoadedExecutableWithMetadata static_exec,
                          CompileTuplePassthrough());
  TT_ASSERT_OK_AND_ASSIGN(
      const std::string serialized_static,
      static_exec->GetLoadedExecutable()->SerializeExecutable());
  TT_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> loaded_static,
      client_->LoadSerializedExecutable(serialized_static,
                                        /*options=*/std::nullopt,
                                        xla::LoadOptions{}));
  TT_ASSERT_OK_AND_ASSIGN(
      const SharedLoadedExecutableWithMetadata deserialized_static,
      LoadedExecutableWithMetadata::MakeShared(std::move(loaded_static)));
  EXPECT_EQ(deserialized_static->parameter_device_shapes(),
            static_exec->parameter_device_shapes());
  EXPECT_EQ(deserialized_static->output_device_shapes(),
            static_exec->output_device_shapes());

  // 2. Bounded-dynamic parameter and output flags also survive serialization
  // rather than turning into static upper-bound shapes.
  xla::XlaBuilder dyn_builder("serialized_bounded_dynamic");
  const xla::Shape dyn_param_shape = xla::ShapeUtil::MakeShape(
      xla::F32, {8, 8}, /*dynamic_dimensions=*/{true, false});
  const xla::XlaOp dyn_in =
      xla::Parameter(&dyn_builder, 0, dyn_param_shape, "dyn_x");
  const xla::XlaOp static_in = xla::Parameter(
      &dyn_builder, 1, xla::ShapeUtil::MakeShape(xla::F32, {8, 8}), "static_x");
  TT_ASSERT_OK_AND_ASSIGN(
      const xla::XlaComputation dyn_comp,
      dyn_builder.Build(xla::Tuple(&dyn_builder, {dyn_in, static_in})));
  TT_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> raw_dyn,
      client_->CompileAndLoad(dyn_comp, xla::CompileOptions()));
  TT_ASSERT_OK_AND_ASSIGN(
      const SharedLoadedExecutableWithMetadata dyn_exec,
      LoadedExecutableWithMetadata::MakeShared(std::move(raw_dyn)));
  TT_ASSERT_OK_AND_ASSIGN(
      const std::string serialized_dyn,
      dyn_exec->GetLoadedExecutable()->SerializeExecutable());
  TT_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtLoadedExecutable> loaded_dyn,
                          client_->LoadSerializedExecutable(
                              serialized_dyn,
                              /*options=*/std::nullopt, xla::LoadOptions{}));
  TT_ASSERT_OK_AND_ASSIGN(
      const SharedLoadedExecutableWithMetadata deserialized_dyn,
      LoadedExecutableWithMetadata::MakeShared(std::move(loaded_dyn)));
  ASSERT_EQ(deserialized_dyn->parameter_device_shapes().size(), 2);
  EXPECT_TRUE(deserialized_dyn->parameter_device_shapes()[0].is_dynamic());
  EXPECT_TRUE(deserialized_dyn->parameter_device_shapes()[1].is_static());
  EXPECT_EQ(deserialized_dyn->parameter_device_shapes(),
            dyn_exec->parameter_device_shapes());
  ASSERT_EQ(deserialized_dyn->output_device_shapes().size(), 2);
  EXPECT_TRUE(deserialized_dyn->output_device_shapes()[0].is_dynamic());
  EXPECT_TRUE(deserialized_dyn->output_device_shapes()[1].is_static());
  EXPECT_EQ(deserialized_dyn->output_device_shapes(),
            dyn_exec->output_device_shapes());
}

TEST_F(DeviceShapesTest,
       DynamicExecutionMetadataGatesOutputDeviceShapesOnOutputTransform) {
  const auto identity_transform =
      +[](std::vector<absl_nonnull std::unique_ptr<xla::PjRtBuffer>> buffers,
          absl::Span<const Shape>)
      -> absl::StatusOr<
          std::vector<absl_nonnull std::unique_ptr<xla::PjRtBuffer>>> {
    return buffers;
  };

  xla::XlaBuilder builder("make_shared_with_transform");
  const xla::XlaOp p0 = xla::Parameter(
      &builder, 0, xla::ShapeUtil::MakeShape(xla::F32, {2, 3}), "p0");
  TT_ASSERT_OK_AND_ASSIGN(const xla::XlaComputation computation,
                          builder.Build(p0));
  TT_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> raw_executable,
      client_->CompileAndLoad(computation, xla::CompileOptions()));
  UnboundedDynamicExecutionMetadata meta_with_transform;
  meta_with_transform.output_transform = identity_transform;
  TT_ASSERT_OK_AND_ASSIGN(
      const SharedLoadedExecutableWithMetadata direct_transformed,
      LoadedExecutableWithMetadata::MakeShared(std::move(raw_executable),
                                               /*argument_indices=*/{},
                                               meta_with_transform));
  EXPECT_EQ(direct_transformed->parameter_device_shapes().size(), 1);
  EXPECT_EQ(direct_transformed->shared_output_device_shapes(), nullptr);
  EXPECT_TRUE(direct_transformed->output_device_shapes().empty());
  EXPECT_EQ(direct_transformed->output_shapes().size(), 1);

  TT_ASSERT_OK_AND_ASSIGN(const SharedLoadedExecutableWithMetadata executable,
                          CompileTuplePassthrough());

  UnboundedDynamicExecutionMetadata meta_without_transform;
  const SharedLoadedExecutableWithMetadata untransformed =
      LoadedExecutableWithMetadata::WithDynamicExecutionMetadata(
          executable, meta_without_transform);
  EXPECT_EQ(untransformed->parameter_device_shapes(),
            executable->parameter_device_shapes());
  EXPECT_EQ(untransformed->shared_output_device_shapes(),
            executable->shared_output_device_shapes());

  const SharedLoadedExecutableWithMetadata transformed =
      LoadedExecutableWithMetadata::WithDynamicExecutionMetadata(
          executable, meta_with_transform);
  EXPECT_EQ(transformed->parameter_device_shapes(),
            executable->parameter_device_shapes());
  EXPECT_EQ(transformed->shared_output_device_shapes(), nullptr);
  EXPECT_TRUE(transformed->output_device_shapes().empty());
  EXPECT_EQ(transformed->output_shapes().size(), 2);
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
// GetCompiledDeviceShapes. If that changes, checks that they are flattened in
// order.
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
