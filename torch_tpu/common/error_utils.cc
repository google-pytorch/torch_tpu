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

#include "torch_tpu/common/error_utils.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/log/absl_check.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "c10/core/Device.h"
#include "c10/core/WrapDimMinimal.h"
#include "c10/util/ArrayRef.h"  // IWYU pragma: keep for IntArrayRef
#include "c10/util/Exception.h"
#include "c10/util/Optional.h"
#include "c10/util/StringUtil.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/common/status_builder.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/eager/device_types.h"
#include "torch_tpu/eager/eager_mode.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

bool GetEnableDebugChecks() {
  const auto& env_var = GetEnvOnce<kTorchTpuInternalEnableDebugChecksEnvVar>();
  // If the env var is set, respect the user's choice.
  if (env_var.has_value()) {
    return *env_var == "1";
  }
  // Otherwise, enable debug checks if we are in debug eager mode.
  return GetEagerMode() == EagerMode::kDeferNeverAndLaunchBlocking;
}

absl::StatusOr<int64_t> SafeMultiply(int64_t x, int64_t y) {
  const auto product_i128 = absl::int128(x) * y;  // Guaranteed to not overflow.
  const auto product_i64 = static_cast<int64_t>(product_i128);
  TT_RET_CHECK(product_i64 == product_i128, error::kInvalidArgument)
      << "product of " << x << " and " << y << " overflows as int64";
  return product_i64;
}

absl::StatusOr<int64_t> NumElements(absl::Span<const int64_t> dims) {
  int64_t n = 1;
  for (auto d : dims) {
    TT_RET_CHECK(d >= 0, error::kInvalidArgument)
        << "dimension size must be >= 0, got " << d;
    TT_ASSIGN_OR_RETURN(n, SafeMultiply(n, d),
                        _.SetOverride() << "product of dimension sizes ["
                                        << absl::StrJoin(dims, ", ")
                                        << "] overflows as int64");
  }
  return n;
}

absl::StatusOr<int64_t> NumElements(const at::Tensor& tensor) {
  return NumElements(tensor.sizes());
}

absl::StatusOr<int64_t> SafeWrapDim(int64_t dim, int64_t dim_bound) {
  try {
    return at::maybe_wrap_dim(  // MAYBE_WRAP_DIM_OK=implementing SafeWrapDim.
        dim, dim_bound);
  } catch (const c10::Error& e) {
    return TT_ERROR(error::kIndexError) << e.what_without_backtrace();
  }
}

void CheckDeviceIsTpu(const c10::optional<at::Device> device_opt,
                      const std::string_view op_name) {
  ABSL_CHECK(device_opt.has_value());       // CRASH_OK
  ABSL_CHECK_EQ(device_opt.value().type(),  // CRASH_OK
                GetPrivateUse1DeviceType());
}

bool IsXlaOomError(const absl::Status& status) {
  return absl::IsResourceExhausted(status) ||
         (absl::IsInternal(status) &&
          // Some XLA OOM errors are reported as internal errors with this
          // word in the message.
          absl::StrContains(status.message(), "allocation_size"));
}

enum class ExceptionType {
  kTtError,
  kC10Error,
};

// Returns the error message for a Status error.
// Arguments:
// - status: the error Status.
// - thrown_from: the source location where the exception is thrown.
// - exception_type: the type of the exception to be thrown.
static std::string MakeErrorMessage(const absl::Status& status,
                                    const c10::SourceLocation& thrown_from,
                                    const ExceptionType exception_type) {
  if (status.ok()) {
    return "";
  }
  const auto root_op_name = internal::GetRootOpNamePayload(status);
  const std::optional<bool> show_cpp_trace_env = TorchShowCppStacktraces();
  const bool add_cpp_trace =
      exception_type == ExceptionType::kTtError
          ?
          // When throwing a TtError to PyTorch, since TtError is not a subclass
          // of c10::Error, PyTorch won't automatically add the C++ stack trace
          // to the error message even if TORCH_SHOW_CPP_STACKTRACES is set
          // to 1. Therefore we include the C++ stack trace in the error message
          // ourselves when needed. In particular:
          //
          // - For internal torch_tpu, we always add the C++ trace unless
          //   TORCH_SHOW_CPP_STACKTRACES is set to 0.
          // - For external torch_tpu, we add the C++ trace only if
          //   TORCH_SHOW_CPP_STACKTRACES is not set.
          (TT_IS_INTERNAL_TORCH_TPU ? show_cpp_trace_env != false
                                    : show_cpp_trace_env == true)
          // When throwing a c10::Error, PyTorch will add the C++ stack trace
          // if TORCH_SHOW_CPP_STACKTRACES is set to 1. Therefore we don't need
          // to add the C++ stack trace ourselves normally. However, for
          // internal torch_tpu, we add the C++ trace when
          // TORCH_SHOW_CPP_STACKTRACES is not set (PyTorch won't add it
          // automatically in this case), so that Google developers can easily
          // debug torch_tpu bugs.
          : TT_IS_INTERNAL_TORCH_TPU && show_cpp_trace_env == std::nullopt;
  const std::string message =
      absl::StrCat(internal::GetMessageWithCppErrorTrace(status),
                   add_cpp_trace ?
                                 // Leverage c10::Error to generate a string
                                 // with the C++ stack trace.
                       c10::Error(thrown_from, "").what()
                                 : "");
  return root_op_name.empty() ? message
                              : absl::StrCat(root_op_name, "(): ", message);
}

TtError::TtError(absl::Status status, c10::SourceLocation thrown_from)
    // The std::runtime_error message is used when the TtError is passed to
    // PyTorch without being intercepted by TT_KERNEL (e.g. for ops that aren't
    // migrated to TT_KERNEL yet).
    : std::runtime_error(
          MakeErrorMessage(status, thrown_from, ExceptionType::kTtError)),
      status_(std::move(status)),
      thrown_from_(std::move(thrown_from)) {
  ABSL_CHECK(!status_.ok())  // CRASH_OK
      << "TtError must be constructed from a non-OK status.";
}

void TranslateToC10ErrorAndThrow(const TtError& e) {
  const std::string message =
      MakeErrorMessage(e.status(), e.thrown_from(), ExceptionType::kC10Error);
  // Map the error code to the corresponding c10::Error subclass to be
  // consistent with PyTorch CPU's error handling. These different c10::Error
  // subclasses will be translated to different Python exception types by
  // PyTorch (e.g. RuntimeError, IndexError, ValueError, etc).
  // TODO(wan): handle more error codes.
  switch (e.status().code()) {
    case error::kOutOfRange:
      throw  // Translates to Python IndexError.
          c10::IndexError(e.thrown_from(), message);
    case error::kUnimplemented:
      throw  // Translates to Python NotImplementedError.
          c10::NotImplementedError(e.thrown_from(), message);
    default:
      throw  // Translates to Python RuntimeError.
          c10::Error(e.thrown_from(), message);
  }
}

}  // namespace torch_tpu
