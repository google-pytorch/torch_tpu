// Copyright 2025 Google LLC
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

#include "torch_tpu/ops/macros/kernel.h"

#include <optional>
#include <string>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/ones.h"
#include "absl/status/statusor.h"
#include "c10/util/Exception.h"
#include "c10/util/StringUtil.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace {

using testing::ElementsAre;
using testing::Pair;
using testing::StartsWith;

// Per
// https://docs.pytorch.org/docs/stable/debugging_environment_variables.html,
// this disables C++ context in pytorch errors. This can be done only once per
// process as pytorch caches the value of this variable.
static const auto kInitShowCppContext =
    setenv("TORCH_SHOW_CPP_STACKTRACES", "0", /*overwrite=*/1);

void Kernel1(int ndim) {
  TT_KERNEL(OpName::kAdd, _, (IgnoreInCacheKey(ndim, "testing")), {
    throw  // For testing error API.
        TtError(TT_ERROR(error::kInvalidArgument) << "test error",
                c10::SourceLocation({"foo()", "bar.cc", 42}));
  });
}

TEST(TtKernel, PrependOpName) {
  bool thrown_c10_error = false;
  try {
    Kernel1(1);
  } catch (const c10::Error& e) {
    thrown_c10_error = true;
    EXPECT_EQ(std::string(e.what_without_backtrace()), "add(): test error");
    EXPECT_THAT(std::string(e.what()),
                StartsWith("add(): test error\n"
                           "Exception raised from foo() at bar.cc:42 "));
  }
  EXPECT_TRUE(thrown_c10_error);
}

// A delegated-to op.
at::Tensor AtenOp2(bool cond) {
  TT_KERNEL(OpName::kBmm, _, (IgnoreInCacheKey(cond, "testing")), {
    // Even though this error is thrown in op2, the root op is op3 as we
    // are calling op2 from op3.
    throw  // For testing error API.
        TtError(TT_ERROR(error::kInvalidArgument) << "some error",
                c10::SourceLocation({"foo()", "bar.cc", 43}));
  });
}

void NewKernel1(int ndim) {
  TT_KERNEL(OpName::kCatOut, _, (IgnoreInCacheKey(ndim, "testing")),
            { AtenOp2(true); });
}

TEST(TtKernel, PrependRootOpName) {
  bool thrown_c10_error = false;
  const int ndim = 1;
  try {
    // op3 calls op2, which throws an error.
    NewKernel1(ndim);
  } catch (const c10::Error& e) {
    thrown_c10_error = true;
    // The error thrown by op2 is associated with op3, the root op.
    EXPECT_EQ(std::string(e.what_without_backtrace()), "cat(): some error");
    EXPECT_THAT(std::string(e.what()),
                StartsWith("cat(): some error\n"
                           "Exception raised from foo() at bar.cc:43 "));
  }
  EXPECT_TRUE(thrown_c10_error);
}

void Kernel5(int ndim, double alpha, std::optional<int> seed,
             std::optional<bool> cond, std::optional<std::string> name) {
  TT_KERNEL(OpName::kAdd, param_keys, (ndim, alpha, seed, cond, name), {
    EXPECT_THAT(param_keys, ElementsAre(
                                // go/keep-sorted start
                                Pair("alpha", "2.5"),  //
                                Pair("name", "<>"),    //
                                Pair("ndim", "3"),     //
                                Pair("seed", "<42>")   //
                                // go/keep-sorted end
                                ));
  });
}

TEST(TtKernel, ComputesCacheKeysWithNonTensors) {
  const int ndim = 3;
  const double alpha = 2.5;
  const std::optional<int> seed = 42;
  Kernel5(ndim, alpha, seed, std::nullopt, "");
}

void Kernel4(const at::Tensor& self, int ndim, bool expand,
             std::optional<int> seed) {
  TT_KERNEL(OpName::kAdd, param_keys, (self, ndim, expand, seed), {
    // `self` is a Tensor and thus shouldn't be in the cache keys.
    // `seed` is nullopt and thus should be omitted from the cache keys.
    EXPECT_THAT(param_keys, ElementsAre(
                                // go/keep-sorted start
                                Pair("expand", "t"),  //
                                Pair("ndim", "3")     //
                                // go/keep-sorted end
                                ));
  });
}

TEST(TtKernel, ComputesCacheKeysWithTensorAndNonTensors) {
  const at::Tensor self = at::ones(1);
  const int ndim = 3;
  const bool expand = true;
  const std::optional<int> seed = std::nullopt;
  Kernel4(self, ndim, expand, seed);
}

void Kernel0() {
  TT_KERNEL(OpName::kAdd, _, (),
            {
                // Do nothing.
            });
}

TEST(TtKernel, SupportsNullaryOps) { Kernel0(); }

void Kernel6(int ndim, int size, int step) {
  TT_KERNEL(
      OpName::kAdd, param_keys,
      (ndim, IgnoreInCacheKey(size, "Ignore."),
       IgnoreInCacheKey(
           step,
           "This reason contains a comma, which should be handled correctly.")),
      { EXPECT_THAT(param_keys, ElementsAre(Pair("ndim", "1"))); });
}

TEST(TtKernel, SupportsIgnoreInCacheKeyWithReason) { Kernel6(1, 2, 3); }

// _promoted_scalars is defined only in debug mode. The "tensors promoted from
// scalars are used" check is also done only in debug mode.
#ifndef NDEBUG

void KernelWithPromotedScalars(const at::Tensor& t, const at::Scalar& s,
                               const std::optional<at::Scalar>& os,
                               const at::ArrayRef<at::Scalar>& as, int ndim) {
  auto promoted_s = PromoteScalar(s);
  auto promoted_os = PromoteScalar(os);
  auto promoted_as = PromoteScalar(as);
  TT_KERNEL(OpName::kRelu, _,
            (t, promoted_s, IgnoreInCacheKey(promoted_os, "testing"),
             promoted_as, IgnoreInCacheKey(ndim, "testing")),
            {
              EXPECT_EQ(_promoted_scalars.size(),
                        1 + (os.has_value() ? 1 : 0) + promoted_as.size());
              // Mark all scalars as used to avoid crash.
              promoted_s.GetTensor().IgnoreError();
              if (promoted_os.has_value()) {
                promoted_os->GetTensor().IgnoreError();
              }
              for (auto& a : promoted_as) {
                a.GetTensor().IgnoreError();
              }
            });
}

// Verifies that TT_KERNEL() collects pointers to all PromotedScalar-typed
// arguments.
TEST(TtKernel, CollectsPromotedScalarPointers) {
  const at::Tensor t = at::ones(1);
  const at::Scalar s_val = 1.0;
  const std::optional<at::Scalar> os_val = 2.0;
  std::vector<at::Scalar> as_vals = {3.0, 4.0};
  KernelWithPromotedScalars(t, s_val, os_val, as_vals, 1);
}

// Verifies that TT_KERNEL() collects pointers to all PromotedScalar-typed
// arguments, even if some of them are nullopt.
TEST(TtKernel, CollectsPromotedScalarPointersWithNullopt) {
  const at::Tensor t = at::ones(1);
  const at::Scalar s_val = 1.0;
  const std::optional<at::Scalar> os_val;
  std::vector<at::Scalar> as_vals = {3.0, 4.0};
  KernelWithPromotedScalars(t, s_val, os_val, as_vals, 1);
}

void TestCheckTensorsUsedCrash(at::Scalar s) {
  auto promoted_s = PromoteScalar(s);
  TT_KERNEL(OpName::kRelu, _, (promoted_s),
            {
                // Scalar s is not used here.
            });
}

// Verifies that TT_KERNEL() crashes if any of the PromotedScalar-typed
// arguments is not used in the kernel when the kernel doesn't throw an error.
TEST(TtKernelDeathTest, CheckTensorsUsedCrashesIfNotUsed) {
  EXPECT_DEATH(TestCheckTensorsUsedCrash(1.0), "GetTensor");
}

#endif  // NDEBUG

void TestCheckTensorsUsedWithException(at::Scalar s) {
  auto promoted_s = PromoteScalar(s);
  TT_KERNEL(OpName::kRelu, _, (promoted_s), {
    // Throw without calling promoted_s.GetTensor() first.
    TT_CHECK_THROW(false, error::kInvalidArgument) << "kernel error";
  });
}

TEST(TtKernel, CheckTensorsUsedDoesNotCrashIfKernelThrows) {
  bool thrown_c10_error = false;
  try {
    // This should throw.
    TestCheckTensorsUsedWithException(1.0);
  } catch (const c10::Error& e) {
    thrown_c10_error = true;
  }
  // Since the kernel threw, TT_KERNEL() should not crash even though the
  // PromotedScalar-typed argument was not used.
  EXPECT_TRUE(thrown_c10_error);
}

}  // namespace
}  // namespace torch_tpu
