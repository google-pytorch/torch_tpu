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

#include "torch_tpu/common/op_name_stack.h"

#include <optional>
#include <stack>
#include <string>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include "torch_tpu/common/flags.h"
#include "torch_tpu/ops/op_names.h"

ABSL_FLAG(bool, torch_tpu_internal_ban_composite_ops, true,
          "If set, crash when a composite TPU op (i.e. one that is implemented "
          "by TorchTPU by delegating to other ops) is executed.");

namespace torch_tpu {
namespace internal {

// ClangTidy is wrong: a static and global variable is the right choice here
// even though it's not trivially destructible. This is a thread-local
// variable, so it needs to be destructed when the thread ends to avoid memory
// leaks. Hence we shouldn't use NoDestructor<> here.
thread_local  // CPP_THREAD_LOCAL_OK=needed only in the dispatching thread.
    std::stack<OpName>
        OpNameStack::stack_;  // NOLINT

void OpNameStack::Push(OpName op_name) {
  static absl::NoDestructor<const absl::flat_hash_set<OpName>>
      known_composite_ops({
          // DO NOT ADD NEW ENTRIES TO THIS LIST! Our goal is to shrink the list
          // and eventually remove it.
          // go/keep-sorted start
          OpName::kAdaptiveAvgPool2d,
          OpName::kAdaptiveAvgPool2dBackward,
          OpName::kAdaptiveAvgPool3d,
          OpName::kAdaptiveAvgPool3dBackward,
          OpName::kAddRelu_Scalar,
          OpName::kAddRelu_Tensor,
          OpName::kAllOut,
          OpName::kAnyOut,
          OpName::kArangeStartOut,
          OpName::kBaddbmmOut,
          OpName::kBernoulli_Float,
          OpName::kBinCount,
          OpName::kCatOut,
          OpName::kComplexOut,
          OpName::kCopyFrom,
          OpName::kCopyFromAndResize,
          OpName::kCopy_,
          OpName::kCtcLoss,
          OpName::kCtcLossBackward,
          OpName::kCtcLossBackwardTensor,
          OpName::kCtcLossTensor,
          OpName::kCumprodOut,
          OpName::kCumsumOut,
          OpName::kDistributedAllGatherIntoTensor,
          OpName::kDistributedBarrier,
          OpName::kDistributedBroadcast,
          OpName::kDistributedGather,
          OpName::kDistributedReduceScatter,
          OpName::kDistributedReduceScatterTensor,
          OpName::kDistributedReduceScatterTensorCoalesced,
          OpName::kDistributedScatter,
          OpName::kDropout,
          OpName::kEmbeddingBagForwardOnly,
          OpName::kEmbeddingRenorm_,
          OpName::kEmptyStrided,
          OpName::kEqual,
          OpName::kExponential_,
          OpName::kEyeMOut,
          OpName::kEyeOut,
          OpName::kFakeQuantizePerTensorAffineCachemask,
          OpName::kForeachAdd_List,
          OpName::kForeachAdd_Scalar,
          OpName::kForeachAdd_ScalarList,
          OpName::kForeachAdd_Tensor,
          OpName::kForeachAddcdiv_Scalar,
          OpName::kForeachAddcdiv_ScalarList,
          OpName::kForeachAddcmul_Scalar,
          OpName::kForeachAddcmul_ScalarList,
          OpName::kForeachClampMaxList,
          OpName::kForeachClampMaxScalar,
          OpName::kForeachClampMaxScalarList,
          OpName::kForeachClampMax_List,
          OpName::kForeachClampMax_Scalar,
          OpName::kForeachClampMax_ScalarList,
          OpName::kForeachClampMinList,
          OpName::kForeachClampMinScalar,
          OpName::kForeachClampMinScalarList,
          OpName::kForeachClampMin_List,
          OpName::kForeachClampMin_Scalar,
          OpName::kForeachClampMin_ScalarList,
          OpName::kForeachCopy_,
          OpName::kForeachDiv_List,
          OpName::kForeachDiv_Scalar,
          OpName::kForeachDiv_ScalarList,
          OpName::kForeachDiv_Tensor,
          OpName::kForeachLerp_List,
          OpName::kForeachLerp_Scalar,
          OpName::kForeachLerp_ScalarList,
          OpName::kForeachLgamma,
          OpName::kForeachMax,
          OpName::kForeachMaximumList,
          OpName::kForeachMaximumScalar,
          OpName::kForeachMaximumScalarList,
          OpName::kForeachMaximum_List,
          OpName::kForeachMaximum_Scalar,
          OpName::kForeachMaximum_ScalarList,
          OpName::kForeachMinimumList,
          OpName::kForeachMinimumScalar,
          OpName::kForeachMinimumScalarList,
          OpName::kForeachMinimum_List,
          OpName::kForeachMinimum_Scalar,
          OpName::kForeachMinimum_ScalarList,
          OpName::kForeachMul_List,
          OpName::kForeachMul_Scalar,
          OpName::kForeachMul_ScalarList,
          OpName::kForeachMul_Tensor,
          OpName::kForeachNormScalar,
          OpName::kForeachPowList,
          OpName::kForeachPowScalar,
          OpName::kForeachPowScalarAndTensor,
          OpName::kForeachPowScalarList,
          OpName::kForeachPow_List,
          OpName::kForeachPow_Scalar,
          OpName::kForeachPow_ScalarList,
          OpName::kForeachSub_List,
          OpName::kForeachSub_Scalar,
          OpName::kForeachSub_ScalarList,
          OpName::kForeachZero_,
          OpName::kGridSampler2d,
          OpName::kGridSampler2dBackward,
          OpName::kGridSampler3d,
          OpName::kGridSampler3dBackward,
          OpName::kHistc,
          OpName::kHistcOut,
          OpName::kIndexPutImpl_,
          OpName::kIndexTensorOut,
          OpName::kLayerNorm,
          OpName::kLerpScalarOut,
          OpName::kLinalgInvExInverse,
          OpName::kLinalgLuFactorExOut,
          OpName::kLinalgLuOut,
          OpName::kLinalgLuSolveOut,
          OpName::kLinalgSolveExOut,
          OpName::kLinalgVectorNormOut,
          OpName::kLocalScalarDense,
          OpName::kLuUnpackOut,
          OpName::kMaskedScatter_,
          OpName::kMaskedSelect,
          OpName::kMaxPool2d,
          OpName::kMaxPool2dBackward,
          OpName::kMaxPool3dWithIndices,
          OpName::kMaxPool3dWithIndicesBackward,
          OpName::kMseLossBackward,
          OpName::kMseLossOut,
          OpName::kNativeBatchNormLegit,
          OpName::kNativeBatchNormLegitNoStats,
          OpName::kNativeBatchNormLegitNoStatsOut,
          OpName::kNativeBatchNormLegitOut,
          OpName::kNllLoss2dForward,
          OpName::kNonzero,
          OpName::kNonzeroOut,
          OpName::kNormalFloatTensor,
          OpName::kNormalFloatTensorOut,
          OpName::kNormalTensorFloat,
          OpName::kNormalTensorFloatOut,
          OpName::kNormalTensorTensor,
          OpName::kNormalTensorTensorOut,
          OpName::kPolarOut,
          OpName::kProd,
          OpName::kProdDimOut,
          OpName::kRaggedAllToAll,
          OpName::kRaggedAllToAllOut,
          OpName::kRandom_,
          OpName::kRandom_From,
          OpName::kRandom_To,
          OpName::kRandpermGeneratorOut,
          OpName::kReflectionPad2d,
          OpName::kReflectionPad2dBackward,
          OpName::kRepeatInterleaveSelfTensor,
          OpName::kReplicationPad2dBackward,
          OpName::kReplicationPad3dBackward,
          OpName::kReshapeAlias,
          OpName::kRsub,
          OpName::kScaledDotProductEfficientAttention,
          OpName::kScaledDotProductFlashAttention,
          OpName::kScaledDotProductFusedAttentionOverrideable,
          OpName::kScaledDotProductFusedAttentionOverrideableBackward,
          OpName::kScatterAddOut,
          OpName::kScatterValueOut,
          OpName::kScatterValueReduceOut,
          OpName::kSplitWithSizesCopyOut,
          OpName::kSumIntListOut,
          OpName::kTake,
          OpName::kTakeOut,
          OpName::kThresholdBackwardGradInput,
          OpName::kThresholdOut,
          OpName::kToCopy,
          OpName::kTorchTpuInternalGatherAllSubgroups,
          OpName::kUnfold,
          OpName::kUniform_,
          OpName::kUnique2,
          OpName::kUntypedStorageResize_,
          OpName::kVar,
          OpName::kVarOut,
          OpName::kView,
          OpName::kZero_,
          // go/keep-sorted end
          // DO NOT ADD NEW ENTRIES TO THE ABOVE LIST! Our goal is to shrink the
          // list and eventually remove it.
      });
  // Only check known_composite_ops size once as it's a run-time constant.
  static const bool check_once = [] {
    ABSL_CHECK_EQ(  // CRASH_OK
        known_composite_ops->size(), 176 /* DO NOT increase this! */)
        << "The size of known_composite_ops MUST NOT go up. "
           "If you are removing entries from known_composite_ops, please LOWER "
           "the expected size in the comparison to match the new size and "
           "prevent regression. If you are adding entries, please REVERT the "
           "addition and implement the ops by lowering them to SHLO directly, "
           "without delegating to other ops. DO NOT fix this crash by "
           "increasing the expected size!";
    return true;
  }();
  static_cast<void>(check_once);  // VOID_CAST_OK=dummy value

  if (!stack_.empty()) {
    // This op is delegated from the op at the top of the stack.
    const OpName composite = stack_.top();
    if (!known_composite_ops->contains(composite)) {
      const std::string msg = absl::StrCat(
          "CompositeOpCheck: Op ", ToString(composite), " is decomposed into ",
          ToString(op_name),
          ". This is unusually not allowed - please implement the ",
          ToString(composite),
          " op by lowering it to SHLO instead. This is a TorchTPU bug.");
      if (GetFlagOnce<bool, &FLAGS_torch_tpu_internal_ban_composite_ops>()) {
        ABSL_LOG(FATAL) << msg;  // CRASH_OK
      } else {
        ABSL_LOG(ERROR) << msg;
      }
    }
  }
  stack_.push(op_name);
}

void OpNameStack::Pop() {
  ABSL_CHECK(!stack_.empty());  // CRASH_OK
  stack_.pop();
}

OpName OpNameStack::Top() {
  ABSL_CHECK(!stack_.empty());  // CRASH_OK
  return stack_.top();
}

std::optional<OpName> OpNameStack::MaybeTop() {
  if (stack_.empty()) {
    return std::nullopt;
  }
  return stack_.top();
}

}  // namespace internal
}  // namespace torch_tpu
