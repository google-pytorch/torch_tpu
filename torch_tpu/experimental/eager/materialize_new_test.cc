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
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/_internal/compile/compiled_mode.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/compilation.h"
#include "torch_tpu/common/compilation_cache.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/eager/tpu_aten_kernels.h"
#include "torch_tpu/eager/tpu_hooks.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/python_context.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "xla/tsl/platform/statusor.h"

ABSL_DECLARE_FLAG(bool, torch_tpu_internal_enable_new_materialization);

namespace torch_tpu {
namespace {

class MaterializeNewTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    absl::SetFlag(&FLAGS_torch_tpu_internal_enable_new_materialization, true);

    // Force linking of eager operators registry.
    (void)IsCpuFallbackEnabled();

    const std::string device_type = "xla_cpu";
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = device_type});
    ASSERT_EQ(AddTpuHooks(), absl::OkStatus());
    RegisterTpuAllocator();
    CompilationCache::GetInstance().SetOptions({});
  }
};

TEST_F(MaterializeNewTest, AsyncMaterializationSynchronization) {
  // 1. Construct a simple MLIR module for a function that takes one argument.
  // We use an identity function that returns its argument.
  auto mlir_builder = [](mlir::MLIRContext& context)
      -> absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> {
    std::string mlir_text = R"(
      module {
        func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
          return %arg0 : tensor<8xf32>
        }
      }
    )";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        llvm::StringRef(mlir_text.data(), mlir_text.size()),
        mlir::ParserConfig{&context});
    if (!module) {
      return TT_ERROR(error::kInvalidArgument) << "Failed to parse MLIR";
    }
    return module;
  };
  TF_ASSERT_OK_AND_ASSIGN(ContextedModule module,
                          ContextedModule::Make(mlir_builder));
  TF_ASSERT_OK_AND_ASSIGN(
      SharedLoadedExecutableWithMetadata executable,
      CompileMlirExecutable(std::move(module).ToMaybeOwningMlirModule()));

  ScopedPythonContextCapturer context_capturer(OpName::kEmpty);

  // 2. Create a deferred argument.
  const Shape shape(Dimensions{8}, mlir::ElementType::F32);
  auto builder = [shape](mlir::MlirBuilder& builder,
                         absl::Span<mlir::MlirOp> inputs)
      -> absl::StatusOr<DynamicMlirOpResults> {
    return DynamicMlirOpResults{
        BuildFillUninitialized(builder, shape.dtype(), shape.dimensions())};
  };
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<DeviceBufferRef> arg_refs,
      DeviceBufferList::CreateDeferred(OpName::kEmpty, builder, {},
                                       OpParamCacheKeys::Empty(), {shape}));
  DeviceBufferRef arg_ref = arg_refs[0];

  // 3. Wrap into at::Tensor.
  at::Tensor arg_tensor = MakeTensor(arg_ref);

  // 4. Run the compiled model.
  // If the change is in place, this will block and succeed.
  // If the change is missing, this will not block, causing the old worker
  // thread to crash.
  std::vector<at::Tensor> result_tensors =
      ExecuteCompiledModel(executable, {arg_tensor}, {{8}});

  // 5. Verify results.
  // We try to get the buffer from the output tensor.
  // If the compiled model failed, this will return the failure status.
  TF_ASSERT_OK_AND_ASSIGN(DeviceBufferRef result_ref,
                          GetBuffer(result_tensors[0]));
  auto status_or = result_ref.AwaitBuffer();
  EXPECT_TRUE(status_or.ok());
}

}  // namespace
}  // namespace torch_tpu
