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

#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
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
  static absl::Mutex mutex(absl::kConstInit);  // Protects known_composite_ops.
  static absl::NoDestructor<absl::flat_hash_set<OpName>> known_composite_ops({
      // go/keep-sorted start
      OpName::kAbsOut,
      OpName::kAdaptiveAvgPool2d,
      OpName::kAdaptiveAvgPool2dBackward,
      OpName::kAdaptiveAvgPool3d,
      OpName::kAdaptiveAvgPool3dBackward,
      OpName::kAdd,
      OpName::kAddRelu_Scalar,
      OpName::kAddRelu_Tensor,
      OpName::kAllOut,
      OpName::kAmaxOut,
      OpName::kAminOut,
      OpName::kAminmaxOut,
      OpName::kAnyOut,
      OpName::kArangeStartOut,
      OpName::kArgMaxOut,
      OpName::kArgMinOut,
      OpName::kBaddbmmOut,
      OpName::kBernoulli_Float,
      OpName::kBinCount,
      OpName::kBitwiseLeftShiftTensorOut,
      OpName::kBitwiseRightShiftTensorOut,
      OpName::kBmm,
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
      OpName::kFftR2c,
      OpName::kFloorDivide,
      OpName::kForeachAbs_,
      OpName::kForeachAcos_,
      OpName::kForeachAdd_List,
      OpName::kForeachAdd_Scalar,
      OpName::kForeachAdd_ScalarList,
      OpName::kForeachAddcdiv_Scalar,
      OpName::kForeachAddcdiv_ScalarList,
      OpName::kForeachAddcmul_Scalar,
      OpName::kForeachAddcmul_ScalarList,
      OpName::kForeachAsin_,
      OpName::kForeachAtan_,
      OpName::kForeachCeil_,
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
      OpName::kForeachCos_,
      OpName::kForeachCosh_,
      OpName::kForeachDiv_List,
      OpName::kForeachDiv_Scalar,
      OpName::kForeachDiv_ScalarList,
      OpName::kForeachDiv_Tensor,
      OpName::kForeachErf_,
      OpName::kForeachErfc_,
      OpName::kForeachExp_,
      OpName::kForeachExpm1_,
      OpName::kForeachFloor_,
      OpName::kForeachFrac_,
      OpName::kForeachLerp_List,
      OpName::kForeachLerp_Scalar,
      OpName::kForeachLerp_ScalarList,
      OpName::kForeachLgamma,
      OpName::kForeachLgamma_,
      OpName::kForeachLog10_,
      OpName::kForeachLog1p_,
      OpName::kForeachLog2_,
      OpName::kForeachLog_,
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
      OpName::kForeachNeg_,
      OpName::kForeachNormScalar,
      OpName::kForeachPowList,
      OpName::kForeachPowScalar,
      OpName::kForeachPowScalarAndTensor,
      OpName::kForeachPowScalarList,
      OpName::kForeachPow_List,
      OpName::kForeachPow_Scalar,
      OpName::kForeachPow_ScalarList,
      OpName::kForeachReciprocal_,
      OpName::kForeachRound_,
      OpName::kForeachRsqrt_,
      OpName::kForeachSigmoid_,
      OpName::kForeachSign_,
      OpName::kForeachSin_,
      OpName::kForeachSinh_,
      OpName::kForeachSqrt_,
      OpName::kForeachSub_List,
      OpName::kForeachSub_Scalar,
      OpName::kForeachSub_ScalarList,
      OpName::kForeachTan_,
      OpName::kForeachTanh_,
      OpName::kForeachTrunc_,
      OpName::kForeachZero_,
      OpName::kGridSampler2d,
      OpName::kGridSampler2dBackward,
      OpName::kGridSampler3d,
      OpName::kGridSampler3dBackward,
      OpName::kHistc,
      OpName::kHistcOut,
      OpName::kIlshiftScalar,
      OpName::kIlshiftTensor,
      OpName::kIndexPutImpl_,
      OpName::kIndexTensorOut,
      OpName::kIrshiftScalar,
      OpName::kIrshiftTensor,
      OpName::kIsInScalarTensorOut,
      OpName::kLayerNorm,
      OpName::kLerpScalarOut,
      OpName::kLinalgInvExInverse,
      OpName::kLinalgLuFactorExOut,
      OpName::kLinalgLuOut,
      OpName::kLinalgLuSolveOut,
      OpName::kLinalgSolveExOut,
      OpName::kLinalgVectorNormOut,
      OpName::kLocalScalarDense,
      OpName::kLogicalAndOut,
      OpName::kLogicalNotOut,
      OpName::kLogicalOrOut,
      OpName::kLogicalXorOut,
      OpName::kLshiftScalar,
      OpName::kLshiftTensor,
      OpName::kLuUnpackOut,
      OpName::kMaskedScatter_,
      OpName::kMaskedSelect,
      OpName::kMax,
      OpName::kMaxPool2d,
      OpName::kMaxPool2dBackward,
      OpName::kMaxPool3dWithIndices,
      OpName::kMaxPool3dWithIndicesBackward,
      OpName::kMin,
      OpName::kMmOut,
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
      OpName::kNormal_,
      OpName::kPolarOut,
      OpName::kProd,
      OpName::kProdDimOut,
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
      OpName::kResize_,
      OpName::kRngSeed,
      OpName::kRshiftScalar,
      OpName::kRshiftTensor,
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
  });
  if (!stack_.empty()) {
    // This op is delegated from the op at the top of the stack.
    const OpName composite = stack_.top();
    absl::MutexLock lock(mutex);
    if (!known_composite_ops->contains(composite)) {
      known_composite_ops->insert(composite);
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
