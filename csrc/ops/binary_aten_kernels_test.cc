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

#include "csrc/ops/binary_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/tensor.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/DefaultDtype.h"
#include "c10/core/ScalarType.h"
#include "c10/util/typeid.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/compilation_cache.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/op_name_stack.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/events_queue.h"
#include "csrc/eager/materialize.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/eager/tpu_hooks.h"
#include "csrc/ops/binary.h"
#include "csrc/ops/copy_from/copy_common.h"
#include "csrc/ops/copy_from/tpu_to_cpu.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "csrc/pjrt/pjrt_state.h"
#include "csrc/pjrt/pjrt_transfer.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch/headeronly/util/BFloat16.h"

namespace torch_tpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::FloatEq;
using ::testing::Pair;
using ::testing::SizeIs;

// Materializes `cpu_tensor` directly onto TPU via `HostToDeviceTransfer`
// instead of `CopyToTpuTensor` so no deferred `AsStrided` HLO is enqueued into
// `CompilationCache` or fused with the `BinaryOp` under test.
absl::StatusOr<at::Tensor> CopyToTpu(const at::Tensor& cpu_tensor) {
  TT_ASSIGN_OR_RETURN(const mlir::ElementType element_type,
                      ConvertTo<mlir::ElementType>(cpu_tensor.scalar_type()));
  TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer,
                      HostToDeviceTransfer::FromTensor(
                          cpu_tensor.data_ptr(),
                          cpu_tensor.storage().data_ptr().get(), element_type,
                          absl::MakeConstSpan(cpu_tensor.sizes().data(),
                                              cpu_tensor.sizes().size()))
                          .Run());
  TT_RETURN_IF_ERROR(buffer.Synchronize());
  return MakeTensor(std::move(buffer));
}

// Dispatches `BinaryOp` with `force_float_inputs = true` and a fixed `PRED`
// (`bool`) output dtype, then copies the single-element boolean result to CPU.
template <typename RhsType>
absl::StatusOr<bool> RunForceFloatGtToPred(const at::Tensor& lhs_tpu,
                                           const RhsType& rhs) {
  const internal::ScopedOpName scoped_op_name(OpName::kGt);
  const ScopedPythonContextCapturer context_capturer(OpName::kGt);
  TT_ASSIGN_OR_RETURN(
      const at::Tensor result_tpu,
      BinaryOp(lhs_tpu, rhs, BuildGtShlo,
               {.force_float_inputs = true,
                .op_param_cache_keys = OpParamCacheKeys::Empty(),
                .result_dtype = mlir::ElementType::PRED}));
  TT_ASSIGN_OR_RETURN(const DeviceBufferRef result_buffer,
                      GetBuffer(result_tpu));
  TT_ASSIGN_OR_RETURN(const at::Tensor result_cpu,
                      CopyDeviceBufferRefToCpuTensor(
                          result_buffer, TransferSyncMode::kEagerBlocking));
  TT_RET_CHECK(result_cpu.scalar_type() == at::kBool, error::kInternal)
      << "Expected bool output tensor";
  return result_cpu.item<bool>();
}

// Dispatches `BinaryOp` with `force_float_inputs = true` and `result_dtype`
// unset (`std::nullopt`), returning the CPU output scalar type and value.
absl::StatusOr<std::pair<at::ScalarType, float>> RunForceFloatDiv(
    const at::Tensor& lhs_tpu, const at::Tensor& rhs_tpu) {
  const internal::ScopedOpName scoped_op_name(OpName::kDiv);
  const ScopedPythonContextCapturer context_capturer(OpName::kDiv);
  TT_ASSIGN_OR_RETURN(
      const at::Tensor result_tpu,
      BinaryOp(lhs_tpu, rhs_tpu, BuildDivShlo,
               {.force_float_inputs = true,
                .op_param_cache_keys = OpParamCacheKeys::Empty()}));
  TT_ASSIGN_OR_RETURN(const DeviceBufferRef result_buffer,
                      GetBuffer(result_tpu));
  TT_ASSIGN_OR_RETURN(const at::Tensor result_cpu,
                      CopyDeviceBufferRefToCpuTensor(
                          result_buffer, TransferSyncMode::kEagerBlocking));
  return std::make_pair(result_cpu.scalar_type(), result_cpu.item<float>());
}

// Dispatches `BinaryOpOut` with `force_float_inputs = true` into a
// pre-allocated `float32` `out` tensor, then copies the single-element result
// to CPU.
template <typename RhsType>
absl::StatusOr<float> RunForceFloatDivOut(const at::Tensor& lhs_tpu,
                                          const RhsType& rhs,
                                          at::Tensor& out_tpu) {
  const internal::ScopedOpName scoped_op_name(OpName::kDiv);
  const ScopedPythonContextCapturer context_capturer(OpName::kDiv);
  TT_RETURN_IF_ERROR(
      BinaryOpOut(lhs_tpu, rhs, out_tpu, BuildDivShlo,
                  {.force_float_inputs = true,
                   .op_param_cache_keys = OpParamCacheKeys::Empty()}));
  TT_ASSIGN_OR_RETURN(const DeviceBufferRef out_buffer, GetBuffer(out_tpu));
  TT_ASSIGN_OR_RETURN(const at::Tensor out_cpu,
                      CopyDeviceBufferRefToCpuTensor(
                          out_buffer, TransferSyncMode::kEagerBlocking));
  TT_RET_CHECK(out_cpu.scalar_type() == at::kFloat, error::kInternal)
      << "Expected float32 output tensor";
  return out_cpu.item<float>();
}

absl::StatusOr<float> CopyFloatScalarToCpu(const at::Tensor& tpu_tensor) {
  TT_ASSIGN_OR_RETURN(const DeviceBufferRef buffer, GetBuffer(tpu_tensor));
  TT_ASSIGN_OR_RETURN(
      const at::Tensor cpu_tensor,
      CopyDeviceBufferRefToCpuTensor(buffer, TransferSyncMode::kEagerBlocking));
  TT_RET_CHECK(cpu_tensor.scalar_type() == at::kFloat, error::kInternal)
      << "Expected float32 output tensor";
  return cpu_tensor.item<float>();
}

class BinaryAtenKernelsTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = "tpu"});
    ABSL_CHECK_OK(AddTpuHooks());
    RegisterTpuAllocator();
    CompilationCache::GetInstance().SetOptions({});
  }

  static void TearDownTestSuite() {
    ShutDownMaterializationState();
    CompilationCache::ShutDown();
    PjrtBackend::GetInstance().Shutdown();
  }

  void SetUp() override {
    ClearAllStreams();
    prev_default_dtype_ = c10::get_default_dtype();
    CompilationCache::GetInstance().EvictAll();
  }

  void TearDown() override {
    c10::set_default_dtype(prev_default_dtype_);
    ClearAllStreams();
  }

 private:
  caffe2::TypeMeta prev_default_dtype_;
};

TEST_F(BinaryAtenKernelsTest,
       ForceFloatInputsWithResultDtypeRecompilesOnDefaultDtypeChange) {
  // 257 rounds to 256.0 in bfloat16, whereas float32 preserves 257.0 exactly.
  // Because `result_dtype` fixes the output shape to `pred[1]`, only recording
  // `default_dtype` in `op_param_cache_keys` distinguishes the two HLO graphs.
  TT_ASSERT_OK_AND_ASSIGN(const at::Tensor lhs_257_tpu,
                          CopyToTpu(at::tensor({int32_t{257}}, at::kInt)));
  TT_ASSERT_OK_AND_ASSIGN(const at::Tensor lhs_259_tpu,
                          CopyToTpu(at::tensor({int32_t{259}}, at::kInt)));
  TT_ASSERT_OK_AND_ASSIGN(const at::Tensor rhs_256_tpu,
                          CopyToTpu(at::tensor({int32_t{256}}, at::kInt)));

  c10::set_default_dtype(caffe2::TypeMeta::Make<c10::BFloat16>());
  EXPECT_THAT(RunForceFloatGtToPred(lhs_257_tpu, rhs_256_tpu),
              IsOkAndHolds(false));
  EXPECT_THAT(CompilationCache::GetInstance().GetCacheStats().per_entry_stats,
              SizeIs(1));

  c10::set_default_dtype(caffe2::TypeMeta::Make<float>());
  EXPECT_THAT(RunForceFloatGtToPred(lhs_257_tpu, rhs_256_tpu),
              IsOkAndHolds(true));
  EXPECT_THAT(CompilationCache::GetInstance().GetCacheStats().per_entry_stats,
              SizeIs(2));

  // Switching back to bfloat16 reuses the first cached executable.
  c10::set_default_dtype(caffe2::TypeMeta::Make<c10::BFloat16>());
  EXPECT_THAT(RunForceFloatGtToPred(lhs_257_tpu, rhs_256_tpu),
              IsOkAndHolds(false));
  EXPECT_THAT(RunForceFloatGtToPred(lhs_259_tpu, rhs_256_tpu),
              IsOkAndHolds(true));
  EXPECT_THAT(CompilationCache::GetInstance().GetCacheStats().per_entry_stats,
              SizeIs(2));
}

TEST_F(BinaryAtenKernelsTest,
       ForceFloatInputsWithoutResultDtypeRecompilesOnDefaultDtypeChange) {
  TT_ASSERT_OK_AND_ASSIGN(const at::Tensor lhs_257_tpu,
                          CopyToTpu(at::tensor({int32_t{257}}, at::kInt)));
  TT_ASSERT_OK_AND_ASSIGN(const at::Tensor rhs_256_tpu,
                          CopyToTpu(at::tensor({int32_t{256}}, at::kInt)));

  c10::set_default_dtype(caffe2::TypeMeta::Make<c10::BFloat16>());
  EXPECT_THAT(RunForceFloatDiv(lhs_257_tpu, rhs_256_tpu),
              IsOkAndHolds(Pair(at::kBFloat16, FloatEq(1.0f))));
  EXPECT_THAT(CompilationCache::GetInstance().GetCacheStats().per_entry_stats,
              SizeIs(1));

  c10::set_default_dtype(caffe2::TypeMeta::Make<float>());
  EXPECT_THAT(RunForceFloatDiv(lhs_257_tpu, rhs_256_tpu),
              IsOkAndHolds(Pair(at::kFloat, FloatEq(257.0f / 256.0f))));
  EXPECT_THAT(CompilationCache::GetInstance().GetCacheStats().per_entry_stats,
              SizeIs(2));
}

TEST_F(BinaryAtenKernelsTest,
       ForceFloatInputsWithScalarOperandRecompilesOnDefaultDtypeChange) {
  TT_ASSERT_OK_AND_ASSIGN(const at::Tensor lhs_257_tpu,
                          CopyToTpu(at::tensor({int32_t{257}}, at::kInt)));

  c10::set_default_dtype(caffe2::TypeMeta::Make<c10::BFloat16>());
  EXPECT_THAT(RunForceFloatGtToPred(lhs_257_tpu, at::Scalar(256)),
              IsOkAndHolds(false));
  EXPECT_THAT(CompilationCache::GetInstance().GetCacheStats().per_entry_stats,
              SizeIs(1));

  c10::set_default_dtype(caffe2::TypeMeta::Make<float>());
  EXPECT_THAT(RunForceFloatGtToPred(lhs_257_tpu, at::Scalar(256)),
              IsOkAndHolds(true));
  EXPECT_THAT(CompilationCache::GetInstance().GetCacheStats().per_entry_stats,
              SizeIs(2));
}

TEST_F(BinaryAtenKernelsTest,
       ForceFloatInputsBinaryOpOutRecompilesOnDefaultDtypeChange) {
  // With a fixed `float32[1]` `out` tensor, intermediate computation in
  // bfloat16 rounds 257 to 256.0 (`256.0 / 256.0 = 1.0f`), whereas float32
  // computes `257.0f / 256.0f` exactly.
  TT_ASSERT_OK_AND_ASSIGN(const at::Tensor lhs_257_tpu,
                          CopyToTpu(at::tensor({int32_t{257}}, at::kInt)));
  TT_ASSERT_OK_AND_ASSIGN(const at::Tensor rhs_256_tpu,
                          CopyToTpu(at::tensor({int32_t{256}}, at::kInt)));
  TT_ASSERT_OK_AND_ASSIGN(at::Tensor out_tpu,
                          CopyToTpu(at::tensor({0.0f}, at::kFloat)));

  c10::set_default_dtype(caffe2::TypeMeta::Make<c10::BFloat16>());
  EXPECT_THAT(RunForceFloatDivOut(lhs_257_tpu, rhs_256_tpu, out_tpu),
              IsOkAndHolds(FloatEq(1.0f)));
  EXPECT_THAT(CompilationCache::GetInstance().GetCacheStats().per_entry_stats,
              SizeIs(1));

  c10::set_default_dtype(caffe2::TypeMeta::Make<float>());
  EXPECT_THAT(RunForceFloatDivOut(lhs_257_tpu, rhs_256_tpu, out_tpu),
              IsOkAndHolds(FloatEq(257.0f / 256.0f)));
  EXPECT_THAT(CompilationCache::GetInstance().GetCacheStats().per_entry_stats,
              SizeIs(2));

  // Switching back to bfloat16 reuses the first cached executable.
  c10::set_default_dtype(caffe2::TypeMeta::Make<c10::BFloat16>());
  EXPECT_THAT(RunForceFloatDivOut(lhs_257_tpu, rhs_256_tpu, out_tpu),
              IsOkAndHolds(FloatEq(1.0f)));
  EXPECT_THAT(CompilationCache::GetInstance().GetCacheStats().per_entry_stats,
              SizeIs(2));
}

TEST_F(BinaryAtenKernelsTest,
       AtenDivOutIntegralInputsRecompilesOnDefaultDtypeChange) {
  TT_ASSERT_OK_AND_ASSIGN(const at::Tensor lhs_257_tpu,
                          CopyToTpu(at::tensor({int32_t{257}}, at::kInt)));
  TT_ASSERT_OK_AND_ASSIGN(const at::Tensor rhs_256_tpu,
                          CopyToTpu(at::tensor({int32_t{256}}, at::kInt)));
  TT_ASSERT_OK_AND_ASSIGN(at::Tensor out_tpu,
                          CopyToTpu(at::tensor({0.0f}, at::kFloat)));

  c10::set_default_dtype(caffe2::TypeMeta::Make<c10::BFloat16>());
  AtenDivOut(lhs_257_tpu, rhs_256_tpu, out_tpu);
  EXPECT_THAT(CopyFloatScalarToCpu(out_tpu), IsOkAndHolds(FloatEq(1.0f)));
  EXPECT_THAT(CompilationCache::GetInstance().GetCacheStats().per_entry_stats,
              SizeIs(1));

  c10::set_default_dtype(caffe2::TypeMeta::Make<float>());
  AtenDivOut(lhs_257_tpu, rhs_256_tpu, out_tpu);
  EXPECT_THAT(CopyFloatScalarToCpu(out_tpu),
              IsOkAndHolds(FloatEq(257.0f / 256.0f)));
  EXPECT_THAT(CompilationCache::GetInstance().GetCacheStats().per_entry_stats,
              SizeIs(2));
}

TEST_F(BinaryAtenKernelsTest,
       ForceFloatInputsWithFloatOperandsIgnoresDefaultDtype) {
  TT_ASSERT_OK_AND_ASSIGN(const at::Tensor lhs_tpu,
                          CopyToTpu(at::tensor({257.0f}, at::kFloat)));
  TT_ASSERT_OK_AND_ASSIGN(const at::Tensor rhs_tpu,
                          CopyToTpu(at::tensor({256.0f}, at::kFloat)));

  c10::set_default_dtype(caffe2::TypeMeta::Make<c10::BFloat16>());
  EXPECT_THAT(RunForceFloatGtToPred(lhs_tpu, rhs_tpu), IsOkAndHolds(true));
  EXPECT_THAT(CompilationCache::GetInstance().GetCacheStats().per_entry_stats,
              SizeIs(1));

  c10::set_default_dtype(caffe2::TypeMeta::Make<float>());
  EXPECT_THAT(RunForceFloatGtToPred(lhs_tpu, rhs_tpu), IsOkAndHolds(true));
  EXPECT_THAT(CompilationCache::GetInstance().GetCacheStats().per_entry_stats,
              SizeIs(1));
}

}  // namespace
}  // namespace torch_tpu
