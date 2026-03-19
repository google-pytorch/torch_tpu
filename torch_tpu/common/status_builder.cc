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

#include "torch_tpu/common/status_builder.h"

#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "torch/csrc/utils/cpp_stacktraces.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/common/utils.h"

namespace torch_tpu {

std::optional<bool> TorchShowCppStacktraces() {
  static const auto state = []() -> std::optional<bool> {
    const auto& env_var = GetEnvOnce<kTorchShowCppStacktracesEnvVar>();
    if (!env_var.has_value()) {
      return std::nullopt;
    }
    // The env var is set. Let PyTorch parse it.
    return torch::get_cpp_stacktraces_enabled();
  }();
  return state;
}

// Returns true if C++ stack traces are enabled. This is controlled by the
// environment variable `TORCH_SHOW_CPP_STACKTRACES`. See
// https://docs.pytorch.org/docs/stable/debugging_environment_variables.html
// If the environment variable is not set, the default is false in the
// open source version and true in the google3 version. This lets Google
// developers see the C++ stack traces so that they can easily fix torch_tpu
// bugs.
//
// For efficiency, this function only checks the environment variable once
// and caches the result. Therefore it will always return the same value
// during the lifetime of the process.
[[nodiscard]] static bool CppStackTracesEnabled() {
  static const bool enabled =
      TorchShowCppStacktraces().value_or(TT_IS_INTERNAL_TORCH_TPU);
  return enabled;
}

// These constructors are not-inlined and defined in the .cc file to reduce
// binary size. See cl/354351433 for a quantification.
StatusBuilder::StatusBuilder() = default;

StatusBuilder::~StatusBuilder() = default;

StatusBuilder::StatusBuilder(const absl::Status& original_status)
    : rep_(Rep::InitRep(original_status)) {}

StatusBuilder::StatusBuilder(absl::Status&& original_status)
    : rep_(Rep::InitRep(std::move(original_status))) {}

StatusBuilderWithMessage::operator absl::Status() && {
  if (rep_ == nullptr) return absl::Status();
  return std::move(*this).CreateStatus();
}

StatusBuilder::Rep::Rep(const absl::Status& s) : status(s) {}
StatusBuilder::Rep::Rep(absl::Status&& s) : status(std::move(s)) {}
StatusBuilder::Rep::~Rep() = default;

StatusBuilder::Rep::Rep(const Rep& r)
    : status(r.status), message_join_style(r.message_join_style) {
  if (r.stream.has_value()) {
    std::stringstream ss;
    ss << r.stream->str();
    stream = std::move(ss);
  }
}

void StatusBuilder::Rep::InitStream() { stream = std::stringstream(); }

absl::Status StatusBuilder::JoinMessageToStatus(absl::Status s,
                                                const std::string_view msg,
                                                const MessageJoinStyle style) {
  ABSL_CHECK(!s.ok())  // CRASH_OK
      << "StatusBuilder::JoinMessageToStatus() can only be called with a "
         "non-OK status.";
  if (msg.empty() && style != MessageJoinStyle::kOverride) return s;
  const std::string new_msg =
      style == MessageJoinStyle::kAppend    ? absl::StrCat(s.message(), msg)
      : style == MessageJoinStyle::kPrepend ? absl::StrCat(msg, s.message())
                                            : std::string(msg);
  absl::Status new_status(s.code(), new_msg);

  if (CppStackTracesEnabled()) {
    // Preserve the C++ context in the original Status if any.
    auto old_context = s.GetPayload(internal::kCppErrorTraceUrl);
    if (old_context.has_value()) {
      new_status.SetPayload(internal::kCppErrorTraceUrl,
                            std::move(old_context).value());
    }
  }

  // Preserve the root op name in the original Status if any.
  auto old_root_op_name = s.GetPayload(internal::kRootOpNameUrl);
  if (old_root_op_name.has_value()) {
    new_status.SetPayload(internal::kRootOpNameUrl,
                          std::move(old_root_op_name).value());
  }

  return new_status;
}

absl::Status StatusBuilder::CreateStatus() && {
  absl::Status result =
      JoinMessageToStatus(std::move(rep_->status),
                          rep_->stream.has_value() ? rep_->stream->str() : "",
                          rep_->message_join_style);
  ABSL_CHECK(!result.ok())  // CRASH_OK
      << "StatusBuilder can only hold a non-OK status.";
  ABSL_CHECK(!result.message().empty())  // CRASH_OK
      << "StatusBuilder must have a non-empty error message. Did you forget to "
         "stream anything into the TT_* macro?";
  rep_.reset();
  return result;
}

namespace internal {

absl::Status SetRootOpNamePayload(absl::Status status,
                                  std::string_view root_op_name) {
  status.SetPayload(kRootOpNameUrl, absl::Cord(root_op_name));
  return status;
}

std::string GetRootOpNamePayload(const absl::Status& status) {
  auto payload = status.GetPayload(kRootOpNameUrl);
  if (!payload.has_value()) return "";
  return std::string(payload.value());
}

absl::Status MaybeAddCppSourceLoc(absl::Status status,
                                  const std::string_view file, const int line,
                                  const std::string_view function) {
  if (!CppStackTracesEnabled() || status.ok()) return status;

  auto old_context = status.GetPayload(kCppErrorTraceUrl);
  absl::Cord new_context(absl::StrCat(file, ":", line, ": ", function, "()\n"));
  if (old_context.has_value()) {
    new_context.Prepend(std::move(old_context).value());
  }
  status.SetPayload(kCppErrorTraceUrl, new_context);
  return status;
}

std::string GetMessageWithCppErrorTrace(const absl::Status& status) {
  auto context = status.GetPayload(kCppErrorTraceUrl);
  if (!context.has_value()) return std::string(status.message());
  return absl::StrCat(status.message(),
                      "\n\nC++ error trace (starting from the origin):\n",
                      std::string(context.value()));
}

}  // namespace internal

}  // namespace torch_tpu
