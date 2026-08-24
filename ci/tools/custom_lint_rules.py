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

"""Rules definitions for custom_lint presubmit checker."""

from collections.abc import Sequence
import dataclasses
import re


@dataclasses.dataclass(frozen=True)
class Rule:
  """A presubmit check rule modeled after presubmit check_contents."""

  id: str
  path_regex: str
  prohibited_regex: str = r'^\b$'
  failure_message: str = ''
  suppression_regex: str | None = None
  path_exclusion_regex: str | None = None
  check_delta_only: bool = True
  disable_tags: Sequence[str] = dataclasses.field(default_factory=list)
  required_regex: str | None = None
  match_entire_file: bool = False
  operations: Sequence[str] = dataclasses.field(default_factory=list)

  def matches_path(self, rel_path: str) -> bool:
    """Returns True if the rule applies to rel_path."""
    if not re.search(self.path_regex, rel_path):
      return False
    if self.path_exclusion_regex and re.search(
        self.path_exclusion_regex, rel_path
    ):
      return False
    return True


RULES: Sequence[Rule] = [
    Rule(
        id='CheckLicenseHeader',
        path_regex='(^|/)BUILD(\\.bazel)?$|.*\\.(h|hpp|c|cc|cpp|py|sh|bzl)$',
        failure_message=(
            'All source files must include a license header. Please ensure the'
            ' file has the appropriate copyright and license information. See'
            ' go/thirdparty/licenses and go/copyright for details. Add'
            ' SKIP_LICENSE_HEADER_CHECK=<reason> to the CL description to'
            ' disable this check.'
        ),
        path_exclusion_regex='^google/.*',
        check_delta_only=False,
        disable_tags=['SKIP_LICENSE_HEADER_CHECK'],
        required_regex=(
            'Copyright \\d{4} Google LLC(.|\\n)*Licensed under the Apache'
            ' License'
        ),
        match_entire_file=True,
    ),
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
        path_exclusion_regex='(^|/)copy\\.bara\\.sky$',
        disable_tags=['ALLOW_IRREVERSIBLE_TRANSFORM', 'ROLLBACK_OF'],
    ),
    Rule(
        id='CheckPythonEnvAccess',
        path_regex='.*\\.py$',
        prohibited_regex='\\bos\\.(environ|getenv|putenv|unsetenv)\\b',
        failure_message=(
            'Please do not read or write environment variables directly from'
            ' Python code that is linked into the torch_tpu library. Use'
            ' torch_tpu._internal.env, whose get_*_env_once() readers are bound'
            ' to the C++ GetEnvOnce() family: they check the env var against'
            ' the registry in'
            ' /torch_tpu/csrc/common/env_vars.h and'
            ' memoize it, so that no one can change a global knob in the middle'
            ' of a program. For writes, which have no C++ equivalent, use'
            ' torch_tpu._internal.env_utils.set_str_env(). If you have a'
            ' legitimate reason to use os.environ, add a `#'
            ' PY_ENVIRON_OK=<reason>` comment on the same line.'
        ),
        suppression_regex='\\bPY_ENVIRON_OK=.+|#.*\\bos\\.(environ|getenv|putenv|unsetenv)\\b|`os\\.(environ|getenv|putenv|unsetenv)`',
        path_exclusion_regex='(^|/)(benchmarks|examples|docs|prototypes|tests)/.*|(^|/)google/(prototypes|tools)/.*|(^|/)oss/(bazel|ci|tools)/.*|(^|/)oss/noxfile\\.py$|.*_test\\.py$',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckKernelMacro',
        path_regex='(^|/)csrc/ops/.*kernels\\.cc$',
        failure_message=(
            'All *kernels.cc files must use TT_KERNEL() to implement a kernel'
            ' (see go/torch-tpu-op-easy for the rationale and more details. If'
            ' you have a legitimate reason to not use TT_KERNEL(), add a'
            ' comment in the file to explain why TT_KERNEL is not used (as long'
            ' as the comment mentions TT_KERNEL, this check passes). You can'
            ' also suppress this check for all files in the CL by adding'
            ' NO_TT_KERNEL_OK=<reason> to the CL description.'
        ),
        path_exclusion_regex='(^|/)csrc/ops/custom_kernels\\.cc$',
        check_delta_only=False,
        disable_tags=['ROLLBACK_OF', 'NO_TT_KERNEL_OK'],
        required_regex='\\bTT_KERNEL\\b',
    ),
    Rule(
        id='CheckTolerancesUpdatedByTool',
        path_regex='(^|/)tests/ops_test\\.py$',
        prohibited_regex='"(rtol|atol)":',
        failure_message=(
            "You are adding or modifying a tolerance override ('rtol' or"
            " 'atol') in ops_test.py. Tolerances must be calculated and updated"
            ' by an automated tool rather than tweaked by hand. Please run the'
            ' update tolerances workflow (`update_tolerances_workflow`) to'
            ' compute reliable bounds. See go/tt-optimal-error-bounds and the'
            ' skill documentation at'
            ' http://torch_tpu/google/tools/_agent/plugins/coding/skills/update_tolerances/SKILL.md'
            ' for exact instructions.\n\nAdd the tag'
            ' TOLERANCES_UPDATED_BY_TOOL=<sponge_link> to your CL description'
            ' pointing to the Sponge invocation demonstrating optimal tolerance'
            ' calculation.'
        ),
        suppression_regex='\\bTOLERANCE_OVERRIDE_OK\\b',
        disable_tags=['TOLERANCES_UPDATED_BY_TOOL', 'ROLLBACK_OF'],
    ),
    Rule(
        id='CheckAssertOkAndAssign',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\b(TF_ASSERT_OK_AND_ASSIGN|ASSERT_OK_AND_ASSIGN)\\b',
        failure_message=(
            'Please use TT_ASSERT_OK_AND_ASSIGN instead of'
            ' TF_ASSERT_OK_AND_ASSIGN (deprecated) or ASSERT_OK_AND_ASSIGN (not'
            ' available in the OSS build).'
        ),
        suppression_regex='//.*ASSERT_OK_AND_ASSIGN\\b',
        disable_tags=['ROLLBACK_OF'],
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
            'Please do not use "Legacy usage" as the reason for'
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
        id='CheckIncludeBeforeCopyright',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\A(\\s|\\n)*#include',
        failure_message=(
            'Please ensure that all #includes are after the copyright notice.'
        ),
        check_delta_only=False,
        disable_tags=['ROLLBACK_OF'],
        match_entire_file=True,
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
            ' thread_local only if the state should be per-C++-thread.\n\nIf'
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
            ' have a legimitate reason to use it, add a comment on the same'
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
            ' #define line. E.g.\n  #define FOO_BAR /* NON_TT_MACRO_OK=name'
            ' required by the xyz library. */ \\\n    ...'
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
            ' xla, etc.).See go/totw/151 for the latest recommendation. If you'
            ' have a legitimate reason to suppress this check, add a `//'
            ' LEADING_COLONS_OK` comment on thesame line as the leading ::.'
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
            ' signlessintegers types to represent signed values. If this call'
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
        id='CheckCheckFunctionName',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\babsl::(Status|StatusOr<.*?>)\\s+Check\\w*\\(',
        failure_message=(
            'Function name starting with `Check` should be used only if it'
            ' crashes when the condition is not met. Otherwise, please use a'
            ' different name, e.g. `Validate*`. If you have a legitimate reason'
            ' to use the `Check` prefix, add a `// CHECK_NAME_OK=<reason>`'
            ' comment on the same line as the function name.'
        ),
        suppression_regex='//.*\\bCHECK_NAME_OK=.+',
        path_exclusion_regex='.*_test\\.(cc|cpp|h)$',
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
            ' stability and prevent correctness bugs.\n\nSee'
            ' go/hash-vs-fingerprint for a detailed explanation on the'
            ' rationale of this check. If you have a legitimate reason to use a'
            ' hash function, add a comment `// HASH_OK=<reason>` on the same'
            ' line.'
        ),
        suppression_regex='\\bHASH_OK=\\S|//.*\\b(std::hash|absl::Hash)',
        path_exclusion_regex='.*_test\\.(cc|cpp|h)$',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckNoVoidCast',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\(void\\)|\\bstatic_cast<void>',
        failure_message=(
            'Please do not cast an expression or variable to void to suppress'
            " 'unused result' or 'unused variable' compiler errors - these"
            ' compiler checks catch real bugs.\n\nFor example, if an expression'
            ' returns a Status, make sure the result is checked and handled.'
            ' For another example, if a param_keys variable is created, make'
            " sure it's used to call DispatchOp instead of dropped.\n\nIf you"
            ' have a legitimate reason to cast to void, add a comment `//'
            ' VOID_CAST_OK=<reason>` on the same line.'
        ),
        suppression_regex=(
            '\\bVOID_CAST_OK\\b|//.*(\\(void\\)|\\bstatic_cast<void>)'
        ),
        path_exclusion_regex='.*_test\\.(cc|cpp|h)$',
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
        path_exclusion_regex='.*_test\\.(cc|cpp|h)$',
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
            ' on TorchTPU internals anyway).\n\nMake sure the flag name is on'
            ' the same line as the ABSL_FLAG call.\n\nIf you have a legitimate'
            ' reason to not follow this rule, add a'
            ' NONSTANDARD_FLAG_NAME_OK=<reason> comment on the same line as the'
            ' ABSL_FLAG call.'
        ),
        suppression_regex=(
            '(\\btorch_tpu_internal_|\\bNONSTANDARD_FLAG_NAME_OK=.+)'
        ),
        path_exclusion_regex='.*_test\\.(cc|cpp|h)$',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckEnvVarName',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\bsetenv\\(\\"',
        failure_message=(
            'Please do not use a string literal as the env var name. All env'
            ' vars used by TorchTPU must be defined in'
            ' torch_tpu/csrc/common/env_vars.h and'
            ' documented if they are not introduced by TorchTPU. If you have a'
            ' legitimate reason to suppress this check, add a `//'
            ' STRING_LITERAL_ENV_VAR_NAME_OK=<reason>` comment on the same'
            ' line.'
        ),
        suppression_regex='\\bSTRING_LITERAL_ENV_VAR_NAME_OK=.+',
        path_exclusion_regex='.*_test\\.(cc|cpp|h)$',
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
        path_exclusion_regex='.*_test\\.(cc|cpp|h)$',
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
        path_exclusion_regex='.*_test\\.(cc|cpp|h)$',
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
            " E.g.\n  ABSL_CHECK_EQ(  // CRASH_OK=If the sizes don't match, the"
            ' object state is messed up.\n      x.size(), y.size()) <<'
            " ...;\n\nIf it's self-evident why the macro is appropriate, you"
            ' can omit the reason in the comment. E.g.\n  ABSL_CHECK(  //'
            ' CRASH_OK\n      IsGlobalRegistryValid()) << ...;\n'
        ),
        suppression_regex='\\bCRASH_OK\\b|((//|/\\*).*\\b(FATAL|ABSL_CHECK|ABSL_QCHECK|ABSL_PCHECK|ABSL_CHECK_EQ|ABSL_CHECK_NE|ABSL_CHECK_LE|ABSL_CHECK_LT|ABSL_CHECK_GE|ABSL_CHECK_GT|ABSL_QCHECK_EQ|ABSL_QCHECK_NE|ABSL_QCHECK_LE|ABSL_QCHECK_LT|ABSL_QCHECK_GE|ABSL_QCHECK_GT|ABSL_CHECK_OK|ABSL_QCHECK_OK|ABSL_CHECK_STREQ|ABSL_CHECK_STRNE|ABSL_CHECK_STRCASEEQ|ABSL_CHECK_STRCASENE|ABSL_QCHECK_STREQ|ABSL_QCHECK_STRNE|ABSL_QCHECK_STRCASEEQ|ABSL_QCHECK_STRCASENE|TT_ASSIGN_OR_CRASH)\\b)',
        path_exclusion_regex='.*_test\\.(cc|cpp|h)$',
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
        path_exclusion_regex='.*_test\\.(cc|cpp|h)$',
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
        path_exclusion_regex='.*_test\\.(cc|cpp|h)$',
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
            ' .second (see go/cstyle#Structs_vs._Tuples).\n\nIf you have a'
            " legitimate reason to use std::pair (e.g. you're writing generic"
            ' code or interfacing with APIs that dictate std::pair), add a `//'
            ' STD_PAIR_OK=<reason>` comment on the same line.'
        ),
        suppression_regex='\\bSTD_PAIR_OK=.+|//.*\\bstd::pair\\b',
        path_exclusion_regex='.*_test\\.(cc|cpp|h)$',
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
            ' the function is a Python binding.\n\nIf you have a legitimate'
            ' reason to use the *Py suffix, add a `// PY_SUFFIX_OK=<reason>`'
            ' comment on the same line.'
        ),
        suppression_regex='\\bPY_SUFFIX_OK=.+|//.*\\w*Py\\(',
        path_exclusion_regex='.*_test\\.(cc|cpp|h)$',
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
            ' that only the former is aware of user Python logic.\n\nWorker'
            ' threads should not access or mutate Python thread-local states'
            ' directly. Instead, the dispatch thread is responsible for'
            ' resolving any necessary states or configurations and explicitly'
            ' passing them down to worker threads.\n\nIf you have a legitimate'
            ' reason to use these guards, add a `//'
            ' TORCH_THREAD_LOCAL_STATE_GUARD_OK=<reason>` comment on the same'
            ' line.'
        ),
        suppression_regex='\\bTORCH_THREAD_LOCAL_STATE_GUARD_OK=.+|//.*\\b(c10::DebugInfoGuard|at::ThreadLocalStateGuard)\\b',
        path_exclusion_regex='.*_test\\.(cc|cpp|h)$',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckRawPyBindUsage',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='(\\bPYBIND11_MODULE\\b|\\b(py|pybind11)::(class_|module|module_)\\b)',
        failure_message=(
            'Please do not use raw pybind11 module or class registrations'
            ' directly. Instead, use the error-handling wrappers from'
            ' pybind_error_utils.h:\n\n  - For modules:'
            ' TT_PYBIND11_MODULE(name, m) { ... }\n  - For classes:'
            ' PyBindClass<T>(m, "ClassName").def(...)\n\nThis ensures that'
            ' errors are properly formatted and translated to PyTorch'
            ' c10::Error. If you have a legitimate reason to use raw pybind11,'
            ' add a comment `// RAW_PYBIND_OK=<reason>` on the same line.'
        ),
        suppression_regex='\\bRAW_PYBIND_OK=.+',
        path_exclusion_regex='.*_test\\.(cc|cpp|h)$',
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
        path_exclusion_regex='(^|/)oss/shims/.*',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckNotestOss',
        path_regex='(^|/)(BUILD(\\.bazel)?|.*\\.bzl)$',
        prohibited_regex='\\bnotest_oss\\s*=',
        failure_message=(
            "It's important to test TorchTPU thoroughly in OSS builds. Please"
            " do not use notest_oss unless there's a good reason. If you have a"
            ' legitimate reason to use notest_oss, add a comment `#'
            ' NOTEST_OSS_OK=<reason>` on the same line.'
        ),
        suppression_regex='#.*\\bnotest_oss\\s*=|\\bNOTEST_OSS_OK=.+',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckNobuildOss',
        path_regex='(^|/)(BUILD(\\.bazel)?|.*\\.bzl)$',
        prohibited_regex='\\bnobuild_oss\\s*=',
        failure_message=(
            "It's important to build and test TorchTPU thoroughly in OSS"
            " builds. Please do not use nobuild_oss unless there's a good"
            ' reason (and prefer notest_oss if the target can still be built).'
            ' If you have a legitimate reason to use nobuild_oss, add a comment'
            ' `# NOBUILD_OSS_OK=<reason>` on the same line.'
        ),
        suppression_regex='#.*\\bnobuild_oss\\s*=|\\bNOBUILD_OSS_OK=.+',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckBannedApiDependency',
        path_regex='(^|/)(BUILD(\\.bazel)?|.*\\.bzl)$',
        prohibited_regex='"torch_tpu/api',
        failure_message=(
            'Please do not depend on torch_tpu/api. The api'
            ' package has been removed.'
        ),
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckNoSan',
        path_regex='(^|/)(BUILD(\\.bazel)?|.*\\.bzl)$',
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
        path_regex='(^|/)(BUILD(\\.bazel)?|.*\\.bzl)$',
        prohibited_regex='"notap"',
        failure_message=(
            'Please try to avoid the "notap" tag as it leads to bit rot. If the'
            ' test fails in a particular build mode and a fix is not trivial,'
            ' just disable it in that build mode (e.g. by adding a'
            ' "nofastbuild" or "noopt" tag). If you have a legitimate reason to'
            ' use "notap", add a `# NOTAP_OK=<reason>` comment on the same line'
            ' as the "notap" tag, and add a build_test to ensure that the test'
            ' is at least buildable.'
        ),
        suppression_regex='\\bNOTAP_OK=.+',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckCcTargetUsage',
        path_regex='(^|/)(BUILD(\\.bazel)?|.*\\.bzl)$',
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
        path_regex='(^|/)(BUILD(\\.bazel)?|.*\\.bzl)$',
        prohibited_regex=(
            '\\b(py_library|pytype_library|pytype_strict_library)\\('
        ),
        failure_message=(
            'Please use torch_tpu_py_library instead of py_library,'
            ' pytype_library, or pytype_strict_library. For example:\n '
            ' load("//shims/build_files:build_defs.bzl",'
            ' "torch_tpu_py_library")\n\n  torch_tpu_py_library(\n      name ='
            ' "foo",\n      ...\n  )\n\nThis enforces best practices. If you'
            ' have a legitimate reason to bypass this check, add a comment `#'
            ' PY_LIBRARY_OK=<reason>` on the first line after the'
            ' `py_library(`, etc, line, e.g.\n  py_library(\n    #'
            ' PY_LIBRARY_OK=<reason>\n    name = "foo",\n    ...\n  )\n'
        ),
        suppression_regex=(
            '\\b(py_library|pytype_library|pytype_strict_library)\\(\\n\\s*#'
            ' PY_LIBRARY_OK=.+'
        ),
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckPyTestUsage',
        path_regex='(^|/)(BUILD(\\.bazel)?|.*\\.bzl)$',
        prohibited_regex='\\b(py_test|py_strict_test|py_platform_test)\\(\\n',
        failure_message=(
            'Please use torch_tpu_py_test instead of py_test, py_strict_test,'
            ' or py_platform_test. This enforces that the code follows best'
            ' practices consistently. If you have a legitimate reason to bypass'
            ' this check, add a comment `# PY_TEST_OK=<reason>` on the first'
            ' line after the `py_test(` or `py_strict_test(` line, e.g.\n '
            ' py_test(\n    # PY_TEST_OK=<reason>\n    name = "foo",\n    ...\n'
            '  )\n'
        ),
        suppression_regex=(
            '\\b(py_test|py_strict_test|py_platform_test)\\(\\n\\s*#'
            ' PY_TEST_OK=.+'
        ),
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckTorchTpuPackageEnd',
        path_regex='(^|/)BUILD$',
        failure_message=(
            'All BUILD files must include the following lines to define some'
            ' special build targets needed by TorchTPU.\nAdd this to your BUILD'
            ' file:\n\nload("//shims/build_files:build_defs.bzl",'
            ' "torch_tpu_package_end")\n...\n# This must be the last line in'
            ' the BUILD file.\ntorch_tpu_package_end()\n\nIf you have a'
            ' legitimate reason to not include them, add'
            ' NO_TORCH_TPU_PACKAGE_END_OK=<reason> to the CL description.'
        ),
        path_exclusion_regex='(^|/)oss/.*',
        check_delta_only=False,
        disable_tags=['ROLLBACK_OF', 'NO_TORCH_TPU_PACKAGE_END_OK'],
        required_regex='\\btorch_tpu_package_end\\(\\)[\\s|\\n]*$',
        match_entire_file=True,
    ),
    Rule(
        id='CheckPythonErrorTestArgs',
        path_regex='.*error.*test.*\\.py$',
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
        path_regex='.*error.*test.*\\.py$',
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
        path_regex='.*error.*test.*\\.py$',
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
        id='CheckAssertRaises',
        path_regex='.*_test\\.py$',
        prohibited_regex='\\b(assertRaises\\w*|assert_raises_message)\\b',
        failure_message=(
            'Please move tests for error handling to a *errors_test.py file so'
            ' that error coverage can be computed correctly. Most of such tests'
            ' should be in tests/errors_test.py. If they are TPU-specific'
            ' (uncommon), they should be in tests/tpu_errors_test.py. Error'
            ' tests for distributed programs should be in one of the other'
            ' *_errors_test.py files.\n\nIf you have a legitimate reason to'
            ' bypass this check, add ASSERT_RAISES_OK=<reason> to the same line'
            ' as the assert call.'
        ),
        suppression_regex='\\bASSERT_RAISES_OK=.+',
        path_exclusion_regex=(
            '(^|/)oss/.*|.*/error_testing_test\\.py$|.*errors_test\\.py$'
        ),
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckRelativeImports',
        path_regex='.*\\.py$',
        prohibited_regex='^\\s*from \\.',
        failure_message=(
            'Please use absolute imports instead of relative imports in'
            ' non-example Python files. Relative imports are banned by'
            ' go/pythonstyle and break tools like build_cleaner. They can be'
            ' useful in the examples/ and prototypes/ directories, which may'
            " need to run in a user's environment, but not in other"
            ' directories.\n\nIf you have a legitimate reason to use relative'
            ' imports, add a RELATIVE_IMPORT_OK=<reason> comment on the same'
            ' line as the import.'
        ),
        suppression_regex='\\bRELATIVE_IMPORT_OK=.+',
        path_exclusion_regex='(^|/)(examples|google/prototypes)/.*',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckPythonFlags',
        path_regex='.*\\.py$',
        prohibited_regex='\\bflags\\.DEFINE_',
        failure_message=(
            'Please do not define Python flags in torch_tpu except in tests,'
            ' benchmarks, and examples. They lead to crashes when client'
            ' programs skip flag parsing (many of them do).\n\nIf you have a'
            ' legitimate reason to not follow this rule, add a'
            ' PYTHON_FLAG_OK=<reason> comment on the same line as the'
            ' flags.DEFINE_* call.'  # PYTHON_FLAG_OK=rule_definition
        ),
        suppression_regex='\\bPYTHON_FLAG_OK=.+',
        path_exclusion_regex='(^|/)(_internal/performance/micro|google)/.*|.*(benchmark|example|test).*',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckAbslTestCase',
        path_regex='\\.(ipynb|py)$',
        prohibited_regex='\\b(absltest|unittest|parameterized)\\.TestCase',
        failure_message=(
            'Instead of absltest.TestCase, unittest.TestCase, or'
            ' parameterized.TestCase, please use RepeatableTest or'
            ' VaryingSeedInPostsubmitTest defined in seed_test_utils (or one of'
            ' their subclasses) to ensure that tests have reproducible RNG'
            ' seeds.\n\nIf you have a legitimate reason to use it, add a'
            ' `ABSLTEST_OK=<reason>` or `UNITTEST_OK=<reason>` comment on the'
            ' same line as it.'
        ),
        suppression_regex='\\b(ABSLTEST_OK|UNITTEST_OK)=.+',
        path_exclusion_regex='(^|/)oss/ci/.*',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckAssertClose',
        path_regex='\\.(ipynb|py)$',
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
        path_exclusion_regex='(^|/)(docs|examples)/.*',
        check_delta_only=False,
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckPythonLogging',
        path_regex='\\.(ipynb|py)$',
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
        path_exclusion_regex='(^|/)(examples|google/prototypes)/.*',
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckInitDeviceUsage',
        path_regex='\\.(ipynb|py)$',
        prohibited_regex='\\b_init_device\\b',
        failure_message=(
            '`_init_device` is a legacy API that is no longer needed because'
            ' the TPU backend is automatically initialized when PyTorch loads'
            ' torch_tpu. Please remove calls to `_init_device`. If you have a'
            ' legitimate reason to call it, add an `# INIT_DEVICE_OK=<reason>`'
            ' comment on the same line.'
        ),
        suppression_regex=(
            '\\bdef'
            ' _init_device\\b|\\bINIT_DEVICE_OK\\b|#.*\\b_init_device\\b|`_init_device`'
        ),
        disable_tags=['ROLLBACK_OF'],
    ),
    Rule(
        id='CheckCopyrightYearUpdate',
        path_regex='(^|/)BUILD(\\.bazel)?$|.*\\.(h|hpp|c|cc|cpp|py|sh|bzl)$',
        prohibited_regex='Copyright \\d{4}',
        failure_message=(
            'Copyright year should be the year the file was first published.'
            ' Please do not update the copyright year in existing files. See'
            ' go/copyright#the-year for more details.\n\nIf you have a'
            ' legitimate reason to update the year, add'
            ' ALLOW_COPYRIGHT_YEAR_UPDATE=<reason> to your CL description.'
        ),
        disable_tags=['ALLOW_COPYRIGHT_YEAR_UPDATE'],
        operations=['EDIT'],
    ),
    Rule(
        id='CheckToAccumulateTypeUsage',
        path_regex='.*\\.(cc|cpp|h|hpp)$',
        prohibited_regex='\\bat::toAccumulateType\\b',
        failure_message=(
            'Please do not use at::toAccumulateType directly. Instead, use'
            ' ToAccumulateType (defined in common/dtype.h) and set'
            ' options.computation_dtype in DispatchOp, or explicitly cast to'
            ' the proper dtype inside StableHLO op builders.\nIf you have a'
            ' legitimate reason to use it, add a comment `//'
            ' AT_TO_ACCUMULATE_TYPE_OK=<reason>` on the same line.'
        ),
        suppression_regex='AT_TO_ACCUMULATE_TYPE_OK',
        disable_tags=['ROLLBACK_OF'],
    ),
]
