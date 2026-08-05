# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom presubmit convention checker for TorchTPU OSS repository.

This tool validates coding conventions (such as banned Bazel rules or C++
patterns) that mirror presubmit rules in METADATA.

It supports incremental checking (`check_delta_only=True`) by parsing unified
diffs from either `git diff -U0` (in Git/GitHub Actions CI) or `hg diff -U0`
(in CitC/Fig workspaces), ensuring developers and PRs are only alerted to
violations on new or modified lines.
"""

import argparse
import collections
from collections.abc import Mapping, Sequence
import dataclasses
import os
import pathlib
import re
import subprocess
import sys


@dataclasses.dataclass(frozen=True)
class Rule:
  """A presubmit check rule modeled after presubmit check_contents."""

  id: str
  path_regex: str
  prohibited_regex: str
  failure_message: str
  suppression_regex: str | None = None
  path_exclusion_regex: str | None = None
  check_delta_only: bool = True
  disable_tags: Sequence[str] = dataclasses.field(default_factory=list)

  def matches_path(self, rel_path: str) -> bool:
    """Returns True if the rule applies to rel_path."""
    if not re.search(self.path_regex, rel_path):
      return False
    if self.path_exclusion_regex and re.search(
        self.path_exclusion_regex, rel_path
    ):
      return False
    return True


@dataclasses.dataclass(frozen=True)
class CandidateFile:
  """A candidate file paired with matching rules to avoid redundant evaluation."""

  file_path: pathlib.Path
  rel_path: str
  matching_rules: Sequence[Rule]


@dataclasses.dataclass(frozen=True)
class Violation:
  """Represents a rule violation in a specific file and line."""

  rule_id: str
  file_path: str
  line_number: int
  line_content: str
  message: str


RULES: Sequence[Rule] = [
    Rule(
        id='CheckCopybaraTags',
        path_regex='.*',
        prohibited_regex=(
            '\\bcopybara:(strip|strip_begin|insert|insert_begin)\\b'
        ),
        failure_message=(
            "Found irreversible 'copybara:' tags (strip or insert). Please use"
            ' reversible alternatives instead. See'
            ' go/ml-velocity-copybara#reversibility for instructions on how to'
            ' use reversible alternatives. If you must use these tags, add'
            ' ALLOW_IRREVERSIBLE_TRANSFORM=<reason> to the CL description.'
        ),
        path_exclusion_regex='copy.bara.sky',
        disable_tags=['ALLOW_IRREVERSIBLE_TRANSFORM', 'ROLLBACK_OF'],
    ),
    Rule(
        id='CheckC10ToStringUsage',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\bc10::toString\\b',
        failure_message=(
            'Please do not use c10::toString in torch_tpu codebase. Instead,'
            ' use torch_tpu::ToString (defined in common/to_string.h). If you'
            ' have a legitimate reason to use it, add C10_TOSTRING_OK=<reason>'
            ' to the same line.'
        ),
        suppression_regex='\\bC10_TOSTRING_OK\\b',
        disable_tags=['C10_TOSTRING_OK', 'ROLLBACK_OF'],
    ),
    Rule(
        id='CheckDeprecatedMutexLock',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\b(MutexLock|ReaderMutexLock|WriterMutexLock)\\b.*&',
        failure_message=(
            'Please use `absl::MutexLock lock(mutex)` instead of'
            ' `absl::MutexLock lock(&mutex)`. The latter is deprecated. If you'
            ' have a legitimate reason to use the deprecated syntax, add'
            ' DEPRECATED_MUTEX_LOCK_OK=<reason> to the same line as the'
            ' MutexLock call.'
        ),
        disable_tags=['ROLLBACK_OF', 'DEPRECATED_MUTEX_LOCK_OK'],
    ),
    Rule(
        id='CheckIgnoreInCacheKeyLegacy',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\bIgnoreInCacheKey\\([^,]+,.*\\b(Legacy|legacy)\\b',
        failure_message=(
            'Please do not use \\"Legacy usage\\" as the reason for'
            ' IgnoreInCacheKey() in *new* code. This reason is only used to get'
            ' old code to compile. Instead, state the actual reason this'
            ' parameter should be excluded from the cache key computation. If'
            ' this is a false alarm (e.g. you are moving or reformatting'
            ' existing code), add IGNORE_IN_CACHE_KEY_LEGACY_OK=<reason> to the'
            ' CL description to suppress this check.'
        ),
        disable_tags=['IGNORE_IN_CACHE_KEY_LEGACY_OK', 'ROLLBACK_OF'],
    ),
    Rule(
        id='CheckThreadLocalNoDestructor',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\bthread_local\\s+.*\\babsl::NoDestructor\\b',
        failure_message=(
            'Do not use `thread_local` with `absl::NoDestructor<>` variables as'
            ' the combination leads to leaked memory when a thread finishes. If'
            ' you have a legitimate reason to use them, add a comment `//'
            ' THREAD_LOCAL_NO_DESTRUCTOR_OK=<reason> ` to the same line as the'
            ' variable declaration.'
        ),
        suppression_regex='\\bTHREAD_LOCAL_NO_DESTRUCTOR_OK=.+',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckThreadLocalNoTag',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\bthread_local\\b',
        failure_message=(
            'C++ threads and user Python threads are not 1:1 (see'
            ' go/tt-context-managers). If you need per-python-thread state, use'
            ' the ThreadLocalDebugInfo API described in the doc above. Use'
            ' thread_local only if the state should be per-C++-thread.\\n\\nIf'
            ' thread_local is the right choice, add a comment `//'
            ' CPP_THREAD_LOCAL_OK=<reason>` to the same line as the'
            ' `thread_local`.'
        ),
        suppression_regex='\\bCPP_THREAD_LOCAL_OK=.+|//.*\\bthread_local\\b',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckMaybeWrapDim',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\bat::maybe_wrap_dim\\b',
        failure_message=(
            'Please use SafeWrapDim() instead of at::maybe_wrap_dim(..., 0) as'
            ' the latter may throw. If you have a legitimate reason to use the'
            ' latter, add a comment `// MAYBE_WRAP_DIM_OK=<reason> ` to the'
            ' same line as the at::maybe_wrap_dim call.'
        ),
        suppression_regex='\\bMAYBE_WRAP_DIM_OK=.+',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckDisallowedLogMacros',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\b(LOG|PLOG|DLOG|VLOG|DVLOG|LOG_IF|PLOG_IF|DLOG_IF|LOG_EVERY_N|LOG_FIRST_N|LOG_EVERY_POW_2|LOG_EVERY_N_SEC|PLOG_EVERY_N|PLOG_FIRST_N|PLOG_EVERY_POW_2|PLOG_EVERY_N_SEC|DLOG_EVERY_N|DLOG_FIRST_N|DLOG_EVERY_POW_2|DLOG_EVERY_N_SEC|VLOG_EVERY_N|VLOG_FIRST_N|VLOG_EVERY_POW_2|VLOG_EVERY_N_SEC|LOG_IF_EVERY_N|LOG_IF_FIRST_N|LOG_IF_EVERY_POW_2|LOG_IF_EVERY_N_SEC|PLOG_IF_EVERY_N|PLOG_IF_FIRST_N|PLOG_IF_EVERY_POW_2|PLOG_IF_EVERY_N_SEC|DLOG_IF_EVERY_N|DLOG_IF_FIRST_N|DLOG_IF_EVERY_POW_2|DLOG_IF_EVERY_N_SEC|CHECK|QCHECK|PCHECK|DCHECK|CHECK_EQ|CHECK_NE|CHECK_LE|CHECK_LT|CHECK_GE|CHECK_GT|QCHECK_EQ|QCHECK_NE|QCHECK_LE|QCHECK_LT|QCHECK_GE|QCHECK_GT|DCHECK_EQ|DCHECK_NE|DCHECK_LE|DCHECK_LT|DCHECK_GE|DCHECK_GT|CHECK_OK|QCHECK_OK|DCHECK_OK|CHECK_STREQ|CHECK_STRNE|CHECK_STRCASEEQ|CHECK_STRCASENE|QCHECK_STREQ|QCHECK_STRNE|QCHECK_STRCASEEQ|QCHECK_STRCASENE|DCHECK_STREQ|DCHECK_STRNE|DCHECK_STRCASEEQ|DCHECK_STRCASENE)\\(',
        failure_message=(
            'Please use the ABSL_* version of the macro for logging/checking to'
            ' avoid macro clashes between libraries like absl, glogs, and'
            ' TensorFlow. If you have a legitimate reason to bypass this check,'
            ' add ALLOW_LOG_MACROS=<reason> to the CL description or'
            ' (preferably) a comment `// ALLOW_LOG_MACROS=<reason>` on the same'
            ' line as the macro.'
        ),
        suppression_regex='\\bALLOW_LOG_MACROS=.+',
        disable_tags=['ALLOW_LOG_MACROS', 'ROLLBACK_OF'],
    ),
    Rule(
        id='CheckStdEndl',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='<<\\s*std::endl;',
        failure_message=(
            'You can remove an std::endl streamed to the end of a log message'
            ' as the logging library already adds that automatically. If you'
            ' have a legitimate reason to use std::endl, add a `// ENDL_OK`'
            ' comment on the same line as the std::endl.'
        ),
        suppression_regex='\\bENDL_OK\\b',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckUninitializedTensor',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\bat::Tensor\\s+\\w+;',
        failure_message=(
            'Please avoid declaring uninitialized at::Tensor variables like'
            ' `at::Tensor foo;`. An uninitialized tensor can cause errors if'
            ' passed as an out parameter to another function. Instead,'
            ' initialize it using MakeEmptyTensor (e.g.,'
            ' `TT_ASSIGN_OR_THROW(at::Tensor foo, MakeEmptyTensor(sizes, dtype,'
            ' device))`), or by calling a function that returns an at::Tensor'
            ' (e.g. at::sub(...) instead of AtenSubOut(...,'
            ' uninitialized_tensor)). Otherwise, you will get errors like'
            ' `RuntimeError: tensor does not have a device` or `RuntimeError:'
            ' sym_strides() called on an undefined Tensor`. If you have a'
            ' legitimate reason to use an uninitialized tensor, add a `//'
            ' UNINITIALIZED_TENSOR_OK` comment on the same line.'
        ),
        suppression_regex='\\bUNINITIALIZED_TENSOR_OK\\b',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckErrorApi',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\b(ASSIGN_OR_RETURN|RETURN_IF_ERROR|RET_CHECK.*|TF_ASSIGN_OR_RETURN|TF_RETURN_IF_ERROR|TF_RET_CHECK.*|TORCH_CHECK.*|absl::\\w+Error|throw)\\b',
        failure_message=(
            'This construct is not allowed in torch_tpu. See'
            ' go/torch-tpu-error-api for which alternatives are allowed. If you'
            ' have a legitimate reason to use it, add a comment on the same'
            ' line as the construct name to explain why.'
        ),
        suppression_regex='//|/\\*',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckMacroNamePrefix',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='^\\s*#define\\s',
        failure_message=(
            'Macros defined in torch_tpu must either be a header guard or start'
            ' with TT_. This reduces the risk of macro name collisions. Please'
            ' rename the macro to follow this convention. If you have a'
            ' legitimate reason to suppress this check, add a'
            ' `NON_TT_MACRO_OK=<reason>` comment on the same line as the'
            ' #define line. E.g.\\n  #define FOO_BAR /* NON_TT_MACRO_OK=name'
            ' required by the xyz library. */ \\\\\\n    ...'
        ),
        suppression_regex=(
            '^\\s*#define\\s+(TT_|TORCH_TPU_)|\\bNON_TT_MACRO_OK\\b'
        ),
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckIntVec',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\bstd::vector<int64_t>',
        failure_message=(
            'Please use `Dimensions`, `Strides`, or `Indices` instead of'
            ' std::vector<int64_t> for tensor dimensions, strides, or indices.'
            ' The former is more efficient for the common case of <= 6'
            ' elements. If you have a legitimate reason to use the latter, add'
            ' a `// INT_VEC_OK` comment on the same line as the'
            ' std::vector<int64_t>.'
        ),
        suppression_regex='\\bINT_VEC_OK\\b|//.*\\bstd::vector<int64_t>',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckVecMethod',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\bvec\\(\\)',
        failure_message=(
            'Please use CopyIntVector() instead of .vec() for creating a vector'
            ' of tensor dimensions. The former is more efficient for the common'
            ' case of <= 6 dimensions. If you have a legitimate reason to use'
            ' .vec(), add a `// VEC_OK` comment on the same line as the .vec().'
        ),
        suppression_regex='\\bVEC_OK\\b|//.*\\bvec\\(\\)',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckErrorSpace',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\babsl::StatusCode::',
        failure_message=(
            'Please use error::kFoo instead of absl::StatusCode::kFoo. The'
            ' former is more consistent with the rest of the torch_tpu error'
            ' handling library, and is more readable. If you have a legitimate'
            ' reason to use absl::StatusCode, add a `// STATUS_CODE_OK` comment'
            ' on the same line as the absl::StatusCode.'
        ),
        suppression_regex='\\bSTATUS_CODE_OK\\b|//.*\\babsl::StatusCode::',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckLeadingColons',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='(^|[^\\w])::(absl|absl_testing|std|proto2|testing|llvm|mlir|pjrt|xla)::',
        failure_message=(
            'Please omit the leading :: in well-known top-level namespaces'
            ' (e.g. absl, absl_testing, std, proto2, testing, llvm, mlir, pjrt,'
            ' xla, etc.). See go/totw/151 for the latest recommendation. If'
            ' you have a legitimate reason to suppress this check, add a `//'
            ' LEADING_COLONS_OK` comment on the same line as the leading ::.'
        ),
        suppression_regex='\\bLEADING_COLONS_OK\\b|//.*::(absl|absl_testing|std|proto2|testing|llvm|mlir|pjrt|xla)::',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckUsingDecls',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='^\\s*using\\s+(absl|std|llvm|mlir|stablehlo|xla)::',
        failure_message=(
            "Please don't add using declarations for symbols in common/standard"
            " packages. It's clearer to use the qualified names (e.g."
            ' `mlir::ElementType` as opposed to `ElementType`) at all places.'
            ' If you have a legitimate reason to suppress this check, add a `//'
            ' USING_DECL_OK` comment on the same line as the using declaration.'
        ),
        suppression_regex='\\bUSING_DECL_OK\\b',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckMlirSignedInt',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\b(isSigned|isSignedInteger)',
        failure_message=(
            'Please use `isSignless` methods instead of `isSigned` methods. For'
            ' historic reasons most MLIR dialects including StableHLO use'
            ' signless integers types to represent signed values. If this call'
            ' is needed, add a comment `// MLIR_SIGNED_INT_OK=<reason>` on the'
            ' same line as the `isSigned` call.'
        ),
        suppression_regex='\\bMLIR_SIGNED_INT_OK\\b|//.*\\b(isSigned|isSignedInteger)|\\/\\*.*\\b(isSigned|isSignedInteger)',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckLambdaCapture',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\[=(\\]|,)',
        failure_message=(
            "Please avoid [=] and [=, ...] captures in lambdas. It's clearer"
            ' and safer to explicitly list the captured variables one-by-one.'
            ' This reduces bugs where one accidentally captures a'
            ' reference-semantics variable (e.g. a string_view or ArrayRef) and'
            ' the lambda outlives the original object. E.g. if the lambda needs'
            ' to be called after the current function returns, [=]{ ... sizes'
            ' ...} would be an undefined behavior bug when `sizes` is an'
            ' ArrayRef of a temporary object, and the fix is to write [sizes ='
            ' MakeIntVec(sizes)]{ ... sizes ... }, which captures a copy of the'
            ' array by value. If you have a legitimate reason to use them, add'
            ' a `// DEFAULT_CAPTURE_OK` comment on the same line as the capture'
            ' list.'
        ),
        suppression_regex='\\bDEFAULT_CAPTURE_OK\\b|//.*\\[=(\\]|,)',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckAtInferSize',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\bat::infer_size',
        failure_message=(
            'Please use `InferSize()` instead of `at::infer_size()`.'
            ' `InferSize()` is an efficiency wrapper that avoids heap'
            ' allocations for small vectors, which is a common case for tensor'
            ' dimensions. If you have a legitimate reason to use'
            ' `at::infer_size()`, add a `// AT_INFER_SIZE_OK` comment on the'
            ' same line as the call.'
        ),
        suppression_regex='\\bAT_INFER_SIZE_OK\\b|//.*\\bat::infer_size',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckStdFunction',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\bstd::function<',
        failure_message=(
            'Please use absl::AnyInvocable instead of std::function. See'
            ' go/totw/191 for the rationale. If you have a legitimate reason to'
            ' use std::function, add a `// STD_FUNCTION_OK` comment on the same'
            ' line as the std::function.'
        ),
        suppression_regex='\\bSTD_FUNCTION_OK\\b|//.*\\bstd::function<',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckStringView',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\babsl::string_view\\b',
        failure_message=(
            'Please use std::string_view instead of absl::string_view for'
            ' consistency. If you have a legitimate reason to use'
            ' absl::string_view, add a `// ABSL_STRING_VIEW_OK` comment on the'
            ' same line as the absl::string_view.'
        ),
        suppression_regex='\\bABSL_STRING_VIEW_OK\\b|//.*\\babsl::string_view',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckHashVsFingerprint',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\b(std::hash|absl::Hash)',
        failure_message=(
            'Hash functions (like std::hash, absl::Hash) are not guaranteed to'
            ' be stable across processes, runs, or library versions. If this'
            ' hash value is used for persistence, cache keys, or inter-process'
            ' communication, please use fingerprinting instead to ensure'
            ' stability and prevent correctness bugs.\\n\\nSee'
            ' go/hash-vs-fingerprint for a detailed explanation on the'
            ' rationale of this check. If you have a legitimate reason to use a'
            ' hash function, add a comment `// HASH_OK=<reason>` on the same'
            ' line.'
        ),
        suppression_regex='\\bHASH_OK=\\S|//.*\\b(std::hash|absl::Hash)',
        path_exclusion_regex='.*_test\\.(cc|cpp|h)',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckNoVoidCast',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\(void\\)|\\bstatic_cast<void>',
        failure_message=(
            'Please do not cast an expression or variable to void to suppress'
            " 'unused result' or 'unused variable' compiler errors - these"
            ' compiler checks catch real bugs.\\n\\nFor example, if an'
            ' expression returns a Status, make sure the result is checked and'
            ' handled. For another example, if a param_keys variable is'
            " created, make sure it's used to call DispatchOp instead of"
            ' dropped.\\n\\nIf you have a legitimate reason to cast to void,'
            ' add a comment `// VOID_CAST_OK=<reason>` on the same line.'
        ),
        suppression_regex=(
            '\\bVOID_CAST_OK\\b|//.*(\\(void\\)|\\bstatic_cast<void>)'
        ),
        path_exclusion_regex='.*_test\\.(cc|cpp|h)',
        disable_tags=['ROLLBACK_OF', 'VOID_CAST_OK'],
    ),
    Rule(
        id='CheckIgnoreError',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='(\\.|->)IgnoreError\\(',
        failure_message=(
            'Please do not call .IgnoreError() on StatusOr objects. Usually you'
            ' should handle the error. If you have a legitimate reason to'
            ' ignore the error, add a `// IGNORE_ERROR_OK=<reason>` comment on'
            ' the same line.'
        ),
        suppression_regex=(
            '\\bIGNORE_ERROR_OK\\b|(//|/\\*).*(\\.|->)IgnoreError\\('
        ),
        path_exclusion_regex='.*_test\\.(cc|cpp|h)',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckCppFlagNames',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\bABSL_FLAG\\(',
        failure_message=(
            'Please prefix all TorchTPU C++ flag names with'
            ' `torch_tpu_internal_` as they are not meant for the users (they'
            " have no easy ways to pass flags to TorchTPU and shouldn't depend"
            ' on TorchTPU internals anyway).\\n\\nMake sure the flag name is on'
            ' the same line as the ABSL_FLAG call.\\n\\nIf you have a'
            ' legitimate reason to not follow this rule, add a'
            ' NONSTANDARD_FLAG_NAME_OK=<reason> comment on the same line as the'
            ' ABSL_FLAG call.'
        ),
        suppression_regex=(
            '(\\btorch_tpu_internal_|\\bNONSTANDARD_FLAG_NAME_OK=.+)'
        ),
        path_exclusion_regex='.*_test\\.(cc|cpp|h)',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckEnvVarName',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\b(setenv|SetEnv)\\(\\"',
        failure_message=(
            'Please do not use a string literal as the env var name. All env'
            ' vars used by TorchTPU must be defined in'
            ' torch_tpu/common/env_vars.h and documented'
            ' if they are not introduced by TorchTPU. If you have a legitimate'
            ' reason to suppress this check, add a `//'
            ' STRING_LITERAL_ENV_VAR_NAME_OK=<reason>` comment on the same'
            ' line.'
        ),
        suppression_regex='\\bSTRING_LITERAL_ENV_VAR_NAME_OK=.+',
        path_exclusion_regex='.*_test\\.(cc|cpp|h)',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckGetenv',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\bgetenv\\(',
        failure_message=(
            'Please do not use getenv. Use GetEnvOnce<>() instead to read an'
            ' env var once and memoize it. This ensures that no one can change'
            ' a global knob in the middle of a program (this is consistent with'
            ' PyTorch). If you have a legitimate reason to use getenv, add a'
            ' `// GETENV_OK=<reason>` comment on the same line as getenv().'
        ),
        suppression_regex='\\bGETENV_OK=.+|//.*\\bgetenv\\(',
        path_exclusion_regex='.*_test\\.(cc|cpp|h)',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckGetFlag',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\babsl::GetFlag\\(',
        failure_message=(
            'Please use GetFlagOnce<>() instead of absl::GetFlag. This ensures'
            ' that no one can change a global flag in the middle of a program'
            ' (this is consistent with PyTorch). If you have a legitimate'
            ' reason to use absl::GetFlag, add a `// GET_FLAG_OK=<reason>`'
            ' comment on the same line as absl::GetFlag().'
        ),
        suppression_regex='\\bGET_FLAG_OK=.+|//.*\\babsl::GetFlag\\(',
        path_exclusion_regex='.*_test\\.(cc|cpp|h)',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckCrashingMacros',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\b(FATAL|ABSL_CHECK|ABSL_QCHECK|ABSL_PCHECK|ABSL_CHECK_EQ|ABSL_CHECK_NE|ABSL_CHECK_LE|ABSL_CHECK_LT|ABSL_CHECK_GE|ABSL_CHECK_GT|ABSL_QCHECK_EQ|ABSL_QCHECK_NE|ABSL_QCHECK_LE|ABSL_QCHECK_LT|ABSL_QCHECK_GE|ABSL_QCHECK_GT|ABSL_CHECK_OK|ABSL_QCHECK_OK|ABSL_CHECK_STREQ|ABSL_CHECK_STRNE|ABSL_CHECK_STRCASEEQ|ABSL_CHECK_STRCASENE|ABSL_QCHECK_STREQ|ABSL_QCHECK_STRNE|ABSL_QCHECK_STRCASEEQ|ABSL_QCHECK_STRCASENE|TT_ASSIGN_OR_CRASH)\\b',
        failure_message=(
            'A crashing macro should be used for validating critical invariants'
            ' only. See go/when-to-die-torch-tpu for when this is allowed. If'
            ' this is a legitimate use of the macro, add a comment `//'
            ' CRASH_OK=<reason>` on the same line as the macro to document why'
            ' this is for catching torch_tpu bugs instead of user errors.'
            " E.g.\\n  ABSL_CHECK_EQ(  // CRASH_OK=If the sizes don't match,"
            ' the object state is messed up.\\n      x.size(), y.size()) <<'
            " ...;\\n\\nIf it's self-evident why the macro is appropriate, you"
            ' can omit the reason in the comment. E.g.\\n  ABSL_CHECK(  //'
            ' CRASH_OK\\n      IsGlobalRegistryValid()) << ...;\\n'
        ),
        suppression_regex='\\bCRASH_OK\\b|((//|/\\*).*\\b(FATAL|ABSL_CHECK|ABSL_QCHECK|ABSL_PCHECK|ABSL_CHECK_EQ|ABSL_CHECK_NE|ABSL_CHECK_LE|ABSL_CHECK_LT|ABSL_CHECK_GE|ABSL_CHECK_GT|ABSL_QCHECK_EQ|ABSL_QCHECK_NE|ABSL_QCHECK_LE|ABSL_QCHECK_LT|ABSL_QCHECK_GE|ABSL_QCHECK_GT|ABSL_CHECK_OK|ABSL_QCHECK_OK|ABSL_CHECK_STREQ|ABSL_CHECK_STRNE|ABSL_CHECK_STRCASEEQ|ABSL_CHECK_STRCASENE|ABSL_QCHECK_STREQ|ABSL_QCHECK_STRNE|ABSL_QCHECK_STRCASEEQ|ABSL_QCHECK_STRCASENE|TT_ASSIGN_OR_CRASH)\\b)',
        path_exclusion_regex='.*_test\\.(cc|cpp|h)',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckRawOpName',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\"[\\w\\.]+\\(\\):\\s',
        failure_message=(
            'Please do not include the op name when constructing end-user error'
            ' messages. Instead, let TT_KERNEL add that automatically. See'
            ' go/torch-tpu-error-guide for more details.'
        ),
        suppression_regex='\\bRAW_OP_NAME_OK\\b|//.*\\"[\\w\\.]+\\(\\):\\s',
        path_exclusion_regex='.*_test\\.(cc|cpp|h)',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckMImpl',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\bm\\.impl\\(',
        failure_message=(
            'Please do not use `m.impl(...)` directly in C++ source files. Use'
            ' ImplExperimental, ImplStable, or ImplDeprecated instead. If you'
            ' have a legitimate reason to use m.impl(...) directly, add a'
            ' comment `// M_IMPL_OK=<reason>` on the same line as the'
            ' `m.impl(...)` call.'
        ),
        suppression_regex='\\bM_IMPL_OK=.+|//.*\\bm\\.impl\\(',
        path_exclusion_regex='.*_test\\.(cc|cpp|h)',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckStdPair',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\bstd::pair\\b',
        failure_message=(
            'Please avoid using std::pair in non-test C++ code. Use custom'
            ' structs with descriptive field names instead to improve code'
            ' readability and prevent subtle bugs caused by using .first and'
            ' .second (see go/cstyle#Structs_vs._Tuples).\\n\\nIf you have a'
            " legitimate reason to use std::pair (e.g. you're writing generic"
            ' code or interfacing with APIs that dictate std::pair), add a `//'
            ' STD_PAIR_OK=<reason>` comment on the same line.'
        ),
        suppression_regex='\\bSTD_PAIR_OK=.+|//.*\\bstd::pair\\b',
        path_exclusion_regex='.*_test\\.(cc|cpp|h)',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckPyFunctionSuffix',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\w*Py\\(',
        failure_message=(
            "For consistency, functions whose name ends with 'Py'"
            ' (case-sensitive) are prohibited in non-test C++ code. Please use'
            " 'Py' as a prefix of the function name instead (e.g."
            ' `PyFunctionName` instead of `FunctionNamePy`) to indicate that'
            ' the function is a Python binding.\\n\\nIf you have a legitimate'
            ' reason to use the *Py suffix, add a `// PY_SUFFIX_OK=<reason>`'
            ' comment on the same line.'
        ),
        suppression_regex='\\bPY_SUFFIX_OK=.+|//.*\\w*Py\\(',
        path_exclusion_regex='.*_test\\.(cc|cpp|h)',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckTorchThreadLocalStateGuards',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex=(
            '\\b(c10::DebugInfoGuard|at::ThreadLocalStateGuard)\\b'
        ),
        failure_message=(
            'Please do not use c10::DebugInfoGuard or at::ThreadLocalStateGuard'
            ' in non-test C++ code. In general, TorchTPU marks a clear boundary'
            ' between the dispatch thread and background worker threads, such'
            ' that only the former is aware of user Python logic.\\n\\nWorker'
            ' threads should not access or mutate Python thread-local states'
            ' directly. Instead, the dispatch thread is responsible for'
            ' resolving any necessary states or configurations and explicitly'
            ' passing them down to worker threads.\\n\\nIf you have a'
            ' legitimate reason to use these guards, add a `//'
            ' TORCH_THREAD_LOCAL_STATE_GUARD_OK=<reason>` comment on the same'
            ' line.'
        ),
        suppression_regex='\\bTORCH_THREAD_LOCAL_STATE_GUARD_OK=.+|//.*\\b(c10::DebugInfoGuard|at::ThreadLocalStateGuard)\\b',
        path_exclusion_regex='.*_test\\.(cc|cpp|h)',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckExplicitPrecisionUsage',
        path_regex='.*\\.(cc|h|hpp)$',
        prohibited_regex='\\bPrecision::(DEFAULT|HIGH|HIGHEST)\\b',
        failure_message=(
            'Please avoid using Precision::DEFAULT, Precision::HIGH or'
            ' Precision::HIGHEST as this should be set by the framework user.'
            ' Explicitly setting this may be detrimental to compilation'
            ' performance. If you have a legitimate reason to use it, please'
            ' add a `// EXPLICIT_PRECISION_OK=<reason>` comment on the same'
            ' line.'
        ),
        suppression_regex='\\bEXPLICIT_PRECISION_OK\\b|//.*\\bPrecision::(DEFAULT|HIGH|HIGHEST)',
        path_exclusion_regex='oss/shims/.*',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckBannedApiDependency',
        path_regex='.*(BUILD|\\.bzl)',
        prohibited_regex='"torch_tpu/api',
        failure_message=(
            'Please do not depend on torch_tpu/api. The api'
            ' package has been removed.'
        ),
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckNoSan',
        path_regex='.*(BUILD|\\.bzl)',
        prohibited_regex='\\b(nomsan|notsan|noasan|nosan)\\b',
        failure_message=(
            'Please document the reason for skipping the *san build by'
            ' appending a comment on the same line as the no*san tag.'
        ),
        suppression_regex='#\\s[^\\s]+',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckNoTapTag',
        path_regex='.*(BUILD|\\.bzl)',
        prohibited_regex='"notap"',
        failure_message=(
            'Please try to avoid the \\"notap\\" tag as it leads to bit rot. If'
            ' the test fails in a particular build mode and a fix is not'
            ' trivial, just disable it in that build mode (e.g. by adding a'
            ' \\"nofastbuild\\" or \\"noopt\\" tag). If you have a legitimate'
            ' reason to use \\"notap\\", add a `# NOTAP_OK=<reason>` comment on'
            ' the same line as the \\"notap\\" tag, and add a build_test to'
            ' ensure that the test is at least buildable.'
        ),
        suppression_regex='\\bNOTAP_OK=.+',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckCcTargetUsage',
        path_regex='.*(BUILD|\\.bzl)',
        prohibited_regex='\\b(cc_library|cc_test)\\(',
        failure_message=(
            'Please use torch_tpu_cc_library instead of cc_library, and use'
            ' torch_tpu_cc_test instead of cc_test. This enforces that the code'
            ' follows best practices consistently. If you have a legitimate'
            ' reason to bypass this check, add ALLOW_CC_TARGETS=<reason> to the'
            ' CL description or (preferably) a comment `#'
            ' ALLOW_CC_TARGETS=<reason>` on the same line or on the first line'
            ' after the `cc_library(` or `cc_test(` line.'
        ),
        suppression_regex='\\bALLOW_CC_TARGETS=.+',
        disable_tags=['ALLOW_CC_TARGETS', 'ROLLBACK_OF'],
    ),
    Rule(
        id='CheckPyLibraryUsage',
        path_regex='.*(BUILD|\\.bzl)',
        prohibited_regex=(
            '\\b(py_library|pytype_library|pytype_strict_library)\\('
        ),
        failure_message=(
            'Please use torch_tpu_py_library instead of py_library,'
            ' pytype_library, or pytype_strict_library. For example:\\n '
            ' load(\\"//shims/build_files:build_defs.bzl\\",'
            ' \\"torch_tpu_py_library\\")\\n\\n  torch_tpu_py_library(\\n     '
            ' name = \\"foo\\",\\n      ...\\n  )\\n\\nThis enforces best'
            ' practices. If you have a legitimate reason to bypass this check,'
            ' add a comment `# PY_LIBRARY_OK=<reason>` on the first line after'
            ' the `py_library(`, etc, line, e.g.\\n  py_library(\\n    #'
            ' PY_LIBRARY_OK=<reason>\\n    name = \\"foo\\",\\n    ...\\n  )\\n'
        ),
        suppression_regex=(
            '\\b(py_library|pytype_library|pytype_strict_library)\\(?\\s*.*'
            'PY_LIBRARY_OK=.+'
        ),
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckPyTestUsage',
        path_regex='.*(BUILD|\\.bzl)',
        prohibited_regex='\\b(py_test|py_strict_test|py_platform_test)\\(\\n',
        failure_message=(
            'Please use torch_tpu_py_test instead of py_test, py_strict_test,'
            ' or py_platform_test. This enforces that the code follows best'
            ' practices consistently. If you have a legitimate reason to bypass'
            ' this check, add a comment `# PY_TEST_OK=<reason>` on the first'
            ' line after the `py_test(` or `py_strict_test(` line, e.g.\\n '
            ' py_test(\\n    # PY_TEST_OK=<reason>\\n    name = \\"foo\\",\\n  '
            '  ...\\n  )\\n'
        ),
        suppression_regex=(
            '\\b(py_test|py_strict_test|py_platform_test)\\(\\n\\s*#'
            ' PY_TEST_OK=.+'
        ),
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckPythonErrorTestArgs',
        path_regex='.*error.*test.*\\.py',
        prohibited_regex='\\b(tpu|gpu)\\s*=\\s*("[^"]|\'[^\'])',
        failure_message=(
            'Please use triple-quoted strings for TPU and GPU error messages in'
            ' Python error tests. This makes the error messages easier to read'
            ' and update via copy-n-paste.'
        ),
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckPythonErrorTestArgsParentheses',
        path_regex='.*error.*test.*\\.py',
        prohibited_regex='\\b(tpu|gpu)\\s*=\\s*\\(',
        failure_message=(
            'Please avoid using parentheses around TPU and CPU error messages'
            ' in Python error tests. This makes the error assertion formats'
            ' more consistent and easier to update by agents.'
        ),
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckPythonErrorTestBackslashes',
        path_regex='.*error.*test.*\\.py',
        prohibited_regex='\\b(tpu|gpu)=.*\\\\.*',
        failure_message=(
            'Please avoid using backslashes in TPU and GPU error messages in'
            ' error tests. They are unnecessary in triple-quoted strings and'
            ' make the error messages harder to read and update by agents.'
        ),
        suppression_regex='\\br"',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckRelativeImports',
        path_regex='\\.py$',
        prohibited_regex='^\\s*from \\.',
        failure_message=(
            'Please use absolute imports instead of relative imports in'
            ' non-example Python files. Relative imports are banned by'
            ' go/pythonstyle and break tools like build_cleaner. They can be'
            ' useful in the examples/ and prototypes/ directories, which may'
            " need to run in a user's environment, but not in other"
            ' directories.\\n\\nIf you'
            ' have a legitimate reason to use relative imports, add a'
            ' RELATIVE_IMPORT_OK=<reason> comment on the same line as the'
            ' import.'
        ),
        suppression_regex='\\bRELATIVE_IMPORT_OK=.+',
        path_exclusion_regex='(?:examples|google/prototypes)/.*',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckPythonFlags',
        path_regex='\\.py$',
        prohibited_regex='\\bflags\\.DEFINE_',
        failure_message=(
            'Please start TorchTPU flags with `torch_tpu_internal_` as they are'
            ' not meant for the users (they have no easy ways to pass flags to'
            " TorchTPU and shouldn't depend on TorchTPU internals"
            ' anyway).\\n\\nIf you have a legitimate reason to not follow this'
            ' rule, add a NONSTANDARD_FLAG_NAME_OK=<reason> comment on the same'
            ' line as the flags.DEFINE_* call.'
        ),
        suppression_regex='(\\bNONSTANDARD_FLAG_NAME_OK=.+|\\bflags.DEFINE_\\w+\\([\\s|\\n]*[\'"]torch_tpu_internal_)',
        path_exclusion_regex='(?:google/.*|ops/scaled_dot_product_attention/kernels/scaled_dot_product_attention_generate.py|.*benchmark.*|.*example.*|.*test.*|.*custom_lint.*)',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckModulePath',
        path_regex='\\.ipynb$|\\.py$',
        prohibited_regex='^\\s*(from|import)\\s+torch_tpu\\.',
        failure_message=(
            "Please don't introduce public Python modules without a design"
            ' review. See go/tt-api-policy for details.'
        ),
        suppression_regex='\\btorch_tpu\\.(_internal\\b|ops\\b|examples\\b|google\\b|tests\\b)',
        disable_tags=['ROLLBACK_OF', 'DISABLE_CHECK_MODULE_PATH'],
    ),
    Rule(
        id='CheckPublicPySymbol',
        path_regex='\\.ipynb$|\\.py$',
        prohibited_regex='(^|[^<\\`])torch\\.tpu\\.[a-zA-Z]',
        failure_message=(
            "Please don't introduce public Python symbols in torch.tpu without"
            ' a design review. See go/tt-api-policy for details.'
        ),
        suppression_regex='\\btorch\\.tpu\\.(Event|Precision|Stream|current_device|current_stream|default_generators|default_stream|device_count|get_amp_supported_dtype|get_rng_state|initial_seed|is_available|is_gpu|is_initialized|is_triton_capable|manual_seed|manual_seed_all|precision|set_device|set_rng_state|set_stream|stream|synchronize|topology_aware_mesh|_)\\b',
        disable_tags=['ROLLBACK_OF', 'DISABLE_CHECK_PUBLIC_PY_SYMBOL'],
    ),
    Rule(
        id='CheckBannedApiImport',
        path_regex='\\.ipynb$|\\.py$',
        prohibited_regex='\\b(import\\s+torch_tpu\\.api|from\\s+torch_tpu(\\.|\\s+import\\s+)api\\b)',
        failure_message=(
            'Please do not import torch_tpu.api or its members. The'
            ' torch_tpu.api module has been removed. Accelerator devices like'
            " `torch.device('tpu')` are now automatically available when"
            ' importing `torch`.'
        ),
        path_exclusion_regex='.*custom_lint.*',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckAssertClose',
        path_regex='\\.ipynb$|\\.py$',
        prohibited_regex=(
            '\\b(testing\\.assert_close|assert'
            ' torch\\.allclose|assertTrue\\(torch\\.allclose)\\('
        ),
        failure_message=(
            'For internal torch_tpu tests, please prefer assert_close'
            ' implemented in _internal/utils/utils.py over'
            ' torch.testing.assert_close or asserting `torch.allclose()`. The'
            ' internal implementation provides more detailed information on'
            ' failure. Examples and tutorials are exempt from this check and'
            ' should use torch.testing.assert_close for consistency with'
            ' PyTorch. If you have a legitimate reason to use one of the'
            ' prohibited patterns in internal tests, add a comment'
            ' `TORCH_ASSERT_CLOSE_OK=<reason>` on the same line.'
        ),
        suppression_regex='\\bTORCH_ASSERT_CLOSE_OK\\b',
        path_exclusion_regex='(?:docs/.*|examples/.*)',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckPythonLogging',
        path_regex='\\.ipynb$|\\.py$',
        prohibited_regex=(
            '^\\s*(import\\s+logging|from\\s+logging\\s+import)\\b'
        ),
        failure_message=(
            'Please use `from absl import logging` instead of the standard'
            ' `logging` module to avoid configuration clashes, swallowing of'
            ' flags like --verbosity, and inconsistent log formatting across'
            ' the codebase. If you have a legitimate reason to use the standard'
            ' `logging` module, add a `# PYTHON_LOGGING_OK=<reason>` comment on'
            ' the same line.'
        ),
        suppression_regex='\\bPYTHON_LOGGING_OK\\b|#.*\\b(import\\s+logging|from\\s+logging\\s+import)\\b',
        path_exclusion_regex='examples/.*',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckToAccumulateTypeUsage',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\bat::toAccumulateType\\b',
        failure_message=(
            'Please do not use at::toAccumulateType directly. Instead, use'
            ' ToAccumulateType (defined in common/dtype.h) and set'
            ' options.computation_dtype in DispatchOp, or explicitly cast to'
            ' the proper dtype inside StableHLO op builders.\\nIf you have a'
            ' legitimate reason to use it, add a comment `//'
            ' AT_TO_ACCUMULATE_TYPE_OK=<reason>` on the same line.'
        ),
        suppression_regex='AT_TO_ACCUMULATE_TYPE_OK',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckAbslTestCase',
        path_regex='\\.py$',
        prohibited_regex=r'\b(absltest|unittest)\.TestCase',
        failure_message=(
            'Instead of absltest.TestCase or unittest.TestCase, please use'
            ' RepeatableTest or VaryingSeedInPostsubmitTest defined in'
            ' seed_test_utils (or one of their subclasses) to ensure that tests'
            ' have reproducible RNG seeds.\n\nIf you have a legitimate reason'
            ' to use it, add a `ABSLTEST_OK=<reason>` or `UNITTEST_OK=<reason>`'
            ' comment on the same line as it.'
        ),
        suppression_regex=r'\b(ABSLTEST_OK|UNITTEST_OK)=.+',
        disable_tags=['ROLLBACK_OF'],
    ),
]


def parse_unified_diff(diff_text: str) -> Mapping[str, set[int]]:
  """Parses unified diff (-U0) and maps file relative path to modified line numbers."""
  modified_lines: dict[str, set[int]] = collections.defaultdict(set)
  current_file: str | None = None

  for line in diff_text.splitlines():
    if line.startswith('+++ '):
      path = line[4:].strip().removeprefix('b/').removeprefix('a/')
      if path in ('/dev/null', 'dev/null', ''):
        current_file = None
      else:
        current_file = path
    elif line.startswith('@@ ') and current_file is not None:
      # Match hunk header: @@ -old,count +new,count @@
      match = re.search(r'\+(\d+)(?:,(\d+))?', line)
      if match:
        start_line = int(match.group(1))
        count = int(match.group(2)) if match.group(2) is not None else 1
        for line_num in range(start_line, start_line + count):
          modified_lines[current_file].add(line_num)
  return modified_lines


def get_modified_lines_from_vcs(
    base_sha: str | None,
) -> Mapping[str, set[int]] | None:
  """Returns mapping of modified line numbers via git/hg diff, or None if VCS diff fails."""
  # First try git diff against base_sha
  git_cmd = ['git', 'diff', '-U0', '--no-color']
  if base_sha:
    git_cmd.append(base_sha)
  try:
    result = subprocess.run(
        git_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode == 0:
      return parse_unified_diff(result.stdout)
  except FileNotFoundError:
    pass

  # If git failed or is not available, try hg diff -U0 (CitC/Fig environments)
  try:
    result = subprocess.run(
        ['hg', 'diff', '-U0'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode == 0:
      return parse_unified_diff(result.stdout)
  except FileNotFoundError:
    pass

  return None


def normalize_path(path: pathlib.Path, root_dir: pathlib.Path) -> str:
  """Returns path relative to root_dir as POSIX string, or str(path) if not relative."""
  try:
    return path.relative_to(root_dir).as_posix()
  except ValueError:
    return path.as_posix()


def get_commit_message(base_sha: str | None) -> str:
  """Retrieves current CL/commit messages to check for disable tags."""
  cmds: list[list[str]] = []
  if base_sha:
    cmds.append(['git', 'log', f'{base_sha}..HEAD', '--format=%B'])
  cmds.extend([
      ['git', 'log', '-1', '--format=%B'],
      ['hg', 'log', '-l', '1', '--template', '{desc}'],
  ])

  for cmd in cmds:
    try:
      result = subprocess.run(cmd, capture_output=True, text=True, check=False)
      if result.returncode == 0 and result.stdout.strip():
        return result.stdout
    except FileNotFoundError:
      continue

  return ''


def is_rule_disabled(rule: Rule, commit_message: str) -> bool:
  """Returns True if any of the rule's disable_tags are present in the commit message."""
  return any(tag in commit_message for tag in rule.disable_tags)


DEFAULT_EXCLUDED_PATH_REGEX = (
    r'(^|/)(custom_lint\.py|custom_lint\.sh|custom_lint_test\.py|'
    r'custom_lint_presubmit\.py|custom_lint_presubmit_test\.py|copy\.bara\.sky)$|'
    r'.*\.(orig|bak)$|.*\.del\..*'
)
DEFAULT_EXCLUDED_PATH_RE = re.compile(DEFAULT_EXCLUDED_PATH_REGEX)


def is_path_excluded(rel_path: str) -> bool:
  """Returns True if the file path is globally excluded from custom lint checks."""
  if DEFAULT_EXCLUDED_PATH_RE.search(rel_path):
    return True
  if rel_path.startswith('examples/') or '/examples/' in rel_path:
    return True
  return False


def check_file_content(
    content: str,
    lines: Sequence[str],
    rel_path: str,
    rule: Rule,
    modified_lines: Mapping[str, set[int]] | None,
) -> Sequence[Violation]:
  """Checks file content against a rule and returns any violations."""
  if is_path_excluded(rel_path):
    return []
  if not rule.matches_path(rel_path):
    return []

  # Fast path: if prohibited pattern doesn't occur anywhere in content, exit early.
  if not re.search(rule.prohibited_regex, content):
    return []

  violations: list[Violation] = []
  delta_lines = (
      modified_lines.get(rel_path) if modified_lines is not None else None
  )

  for line_idx, line in enumerate(lines, start=1):
    if rule.check_delta_only and delta_lines is not None:
      if line_idx not in delta_lines:
        continue

    match = re.search(rule.prohibited_regex, line)
    window = None
    if not match and line_idx < len(lines):
      window = f'{line}\n{lines[line_idx]}'
      m = re.search(rule.prohibited_regex, window)
      if m and m.start() <= len(line):
        match = m

    if match:
      if rule.suppression_regex:
        if re.search(rule.suppression_regex, line):
          continue
        if window is None and line_idx < len(lines):
          window = f'{line}\n{lines[line_idx]}'
        if window and re.search(rule.suppression_regex, window):
          continue

      violations.append(
          Violation(
              rule_id=rule.id,
              file_path=rel_path,
              line_number=line_idx,
              line_content=line.strip(),
              message=rule.failure_message,
          )
      )
  return violations


def check_file(
    file_path: pathlib.Path,
    rel_path: str,
    rule: Rule,
    modified_lines: Mapping[str, set[int]] | None,
) -> Sequence[Violation]:
  """Checks a single file against a rule and returns any violations."""
  try:
    content = file_path.read_text(encoding='utf-8')
  except (IOError, UnicodeDecodeError):
    return []
  lines = content.splitlines()
  return check_file_content(content, lines, rel_path, rule, modified_lines)


def find_candidate_files(
    root_dir: pathlib.Path, rules: Sequence[Rule]
) -> Sequence[CandidateFile]:
  """Finds all files under root_dir that match at least one rule's path_regex."""
  candidates: list[CandidateFile] = []
  ignore_dirs = {
      '.git',
      '.hg',
      '.nox',
      'bazel-bin',
      'bazel-out',
      'bazel-testlogs',
      'blaze-bin',
      'blaze-out',
      'blaze-testlogs',
      'blaze-genfiles',
      'venv',
      '.venv',
  }

  for dirpath, dirnames, filenames in os.walk(root_dir):
    dirnames[:] = [
        d
        for d in dirnames
        if d not in ignore_dirs
        and not d.startswith('.')
        and not d.startswith('bazel-')
        and not d.startswith('blaze-')
    ]

    for filename in filenames:
      full_path = pathlib.Path(dirpath) / filename
      rel_path = normalize_path(full_path, root_dir)
      if is_path_excluded(rel_path):
        continue
      matching = [r for r in rules if r.matches_path(rel_path)]
      if matching:
        candidates.append(
            CandidateFile(
                file_path=full_path,
                rel_path=rel_path,
                matching_rules=matching,
            )
        )
  return candidates


def find_torch_tpu_root(
    start_path: pathlib.Path | None = None,
) -> pathlib.Path:
  """Finds the root directory of the torch_tpu package/repository."""
  # 1. First try relative to __file__ location
  current = pathlib.Path(__file__).resolve()
  for parent in current.parents:
    if parent.name == 'torch_tpu' and (parent / 'METADATA').exists():
      return parent
    if (parent / 'pyproject.toml').exists() and (parent / 'oss').exists():
      return parent

  # 2. Try start_path or current working directory
  candidate = (start_path or pathlib.Path.cwd()).resolve()
  torch_tpu_sub = candidate / 'third_party/py/torch_tpu'
  if torch_tpu_sub.exists() and (torch_tpu_sub / 'METADATA').exists():
    return torch_tpu_sub

  for p in [candidate] + list(candidate.parents):
    if p.name == 'torch_tpu' and (p / 'METADATA').exists():
      return p
    if (p / 'pyproject.toml').exists():
      return p

  return candidate


def run_checks(
    root_dir: pathlib.Path,
    files: Sequence[str] | None = None,
    base_sha: str | None = None,
    check_all: bool = False,
) -> int:
  """Runs convention checks and prints violations."""
  root_dir = find_torch_tpu_root(root_dir)
  rules = RULES
  commit_message = get_commit_message(base_sha)
  active_rules = [r for r in rules if not is_rule_disabled(r, commit_message)]

  if not active_rules:
    print(
        'INFO: All custom convention rules disabled by CL/commit description.'
    )
    return 0

  modified_lines = None if check_all else get_modified_lines_from_vcs(base_sha)

  if files:
    candidate_files: list[CandidateFile] = []
    for f in files:
      p = pathlib.Path(f)
      if not p.is_absolute():
        cwd_path = pathlib.Path.cwd() / p
        root_path = root_dir / p
        p = (
            root_path
            if not cwd_path.exists() and root_path.exists()
            else cwd_path
        )
      p = p.resolve()
      if p.is_dir():
        candidate_files.extend(find_candidate_files(p, active_rules))
      elif p.is_file():
        rel_path = normalize_path(p, root_dir)
        if is_path_excluded(rel_path):
          continue
        matching = [r for r in active_rules if r.matches_path(rel_path)]
        if matching:
          candidate_files.append(
              CandidateFile(
                  file_path=p,
                  rel_path=rel_path,
                  matching_rules=matching,
              )
          )
  else:
    candidate_files = list(find_candidate_files(root_dir, active_rules))

  all_violations: list[Violation] = []
  for candidate in candidate_files:
    if not candidate.file_path.is_file():
      continue

    try:
      content = candidate.file_path.read_text(encoding='utf-8')
    except (IOError, UnicodeDecodeError):
      continue
    lines = content.splitlines()

    for rule in candidate.matching_rules:
      all_violations.extend(
          check_file_content(
              content, lines, candidate.rel_path, rule, modified_lines
          )
      )

  if not all_violations:
    print('INFO: Custom TorchTPU convention checks passed successfully.')
    return 0

  print('=' * 65)
  print(
      f'ERROR: {len(all_violations)} custom convention check violation(s)'
      ' found:'
  )
  print('=' * 65)
  for v in all_violations:
    print(f'[{v.rule_id}] {v.file_path}:{v.line_number}')
    print(f'  >   {v.line_content}')
    print(f'  {v.message}')
    print('-' * 65)
  print(f'Total violations: {len(all_violations)}')
  print('=' * 65)
  return 1


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      description='Custom TorchTPU OSS presubmit convention checker.'
  )
  parser.add_argument(
      'files',
      nargs='*',
      help='Optional specific files or directories to check.',
  )
  parser.add_argument(
      '--base-sha',
      default='origin/main',
      help='Base git SHA or tracking branch for incremental diffs.',
  )
  parser.add_argument(
      '--all',
      action='store_true',
      dest='check_all',
      help='Check all lines regardless of git/hg diffs.',
  )
  args = parser.parse_args(argv)

  working_dir = os.environ.get('BUILD_WORKING_DIRECTORY')
  root_dir = pathlib.Path(working_dir) if working_dir else pathlib.Path.cwd()
  return run_checks(
      root_dir=root_dir,
      files=args.files,
      base_sha=args.base_sha,
      check_all=args.check_all,
  )


if __name__ == '__main__':
  sys.exit(main())
