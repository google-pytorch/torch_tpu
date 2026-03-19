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

#ifndef TORCH_TPU_COMMON_STATUS_BUILDER_H_
#define TORCH_TPU_COMMON_STATUS_BUILDER_H_

// A builder for absl::Status. The API mostly mirrors util::StatusBuilder at
// Google. This is mainly used for implementing status macros in error_utils.h.

#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"

namespace torch_tpu {

// A status builder that may not yet contain a message.
class StatusBuilder;
// A status builder that already contains a message.
class StatusBuilderWithMessage;

// Creates a Status based on an original_status, but enriched with additional
// information.
//
// This class cannot be implicitly converted to absl::Status or
// absl::StatusOr<T>, but the result of `builder << message` can, allowing for
// it to be returned directly. This forces the user to stream at least one
// message.
//
//   StatusBuilder builder(original);
//   return builder << " - info about error";
//
// It provides method chaining to simplify typical usage:
//
//   return StatusBuilder(original)
//       .SetPrepend() << "oh no! ";
//
// In more detail:
// - When the original status is OK, all methods become no-ops.
// - Messages streamed into the status builder are collected into a single
//   additional message string.
// - The original Status's message and the additional message are joined
//   together when the result status is built.
class [[nodiscard]] StatusBuilder {
 public:
  explicit StatusBuilder();
  ~StatusBuilder();

  // Creates a `StatusBuilder` based on an original status.
  explicit StatusBuilder(const absl::Status& original_status);
  explicit StatusBuilder(absl::Status&& original_status);

  // StatusBuilder is copyable and movable.
  StatusBuilder(const StatusBuilder& sb);
  StatusBuilder& operator=(const StatusBuilder& sb);
  StatusBuilder(StatusBuilder&&) = default;
  StatusBuilder& operator=(StatusBuilder&&) = default;

  // Mutates the builder so that the final additional message is prepended to
  // the original error message in the status.  A convenience separator is not
  // placed between the messages.
  //
  // Returns `*this` to allow method chaining.
  StatusBuilder& SetPrepend() &;
  [[nodiscard]] StatusBuilder&& SetPrepend() &&;

  // Mutates the builder so that the final additional message overrides the
  // original error message in the status.
  //
  // Returns `*this` to allow method chaining.
  StatusBuilder& SetOverride() &;
  [[nodiscard]] StatusBuilder&& SetOverride() &&;

  // Appends to the extra message that will be added to the original status.  By
  // default, the extra message is appended to the original message.
  //
  // The return type is StatusBuilderWithMessage to allow implicitly converting
  // to absl::Status or absl::StatusOr<T>.
  template <typename T>
  StatusBuilderWithMessage& operator<<(const T& value) &;
  template <typename T>
  [[nodiscard]] StatusBuilderWithMessage&& operator<<(const T& value) &&;

  // Returns true if the Status created by this builder will be ok().
  [[nodiscard]] bool ok() const;

  // This operator prevents the StatusBuilder from being converted to an
  // absl::Status before any error message has been added using the << operator.
  // It enforces the contract that macros like TT_ERROR or TT_RET_CHECK must be
  // followed by at least one streamed message.
  //
  // How it works:
  // - The template is instantiated only when T is absl::Status.
  // - When user code triggers this instantiation, it means that the
  //   TT_ERROR() or TT_RET_CHECK() macro is used without any streamed message.
  template <typename T = absl::Status,
            typename U = std::enable_if_t<std::is_same_v<T, absl::Status>>>
  operator T() const {  // NOLINT: Builder converts implicitly.
    // Name the static_assert condition to make the compiler error message
    // more readable.
    constexpr bool must_provide_error_message = false;
    static_assert(
        must_provide_error_message,
        "\nYou must stream at least one message into the TT_ERROR() or "
        "TT_RET_CHECK() macro or the result of .SetPrepend() or "
        ".SetOverride(). E.g.\n\n"
        "  return TT_ERROR(error::kInternal) << \"message\";\n\n"
        "See the compiler notes below for where you need to make the change."
        " See http://go/enforce-tt-error if you need more details.");
    return absl::OkStatus();
  }

 private:
  friend class StatusBuilderWithMessage;

  // Specifies how to join the error message in the original status and any
  // additional message that has been streamed into the builder.
  enum class MessageJoinStyle {
    kAppend,
    kPrepend,
    kOverride,
  };

  // Infrequently set builder options, instantiated lazily. This reduces
  // average construction/destruction time (e.g. the `stream` is fairly
  // expensive). Stacks can also be blown if StatusBuilder grows too large.
  // This is primarily an issue for debug builds, which do not necessarily
  // re-use stack space within a function across the sub-scopes used by
  // status macros.
  struct Rep {
    explicit Rep(const absl::Status& s);
    explicit Rep(absl::Status&& s);
    Rep(const Rep& r);
    ~Rep();
    void InitStream();

    static std::unique_ptr<Rep> InitRep(const absl::Status& s) {
      if (s.ok()) {
        return nullptr;
      }
      return std::make_unique<Rep>(s);
    }

    static std::unique_ptr<Rep> InitRep(absl::Status&& s) {
      if (s.ok()) {
        return nullptr;
      }
      return std::make_unique<Rep>(std::move(s));
    }

    // The status that the result will be based on.
    absl::Status status;
    // Gathers additional messages added with `<<` for use in the final status.
    // Lazily initialized as it's expensive to construct.
    std::optional<std::stringstream> stream;
    MessageJoinStyle message_join_style = MessageJoinStyle::kAppend;
  };

  // Creates a new status based on an old one by joining the message from the
  // original to an additional message. Requires that the original status is not
  // OK.
  static absl::Status JoinMessageToStatus(absl::Status s, std::string_view msg,
                                          MessageJoinStyle style);

  // Creates a Status from this builder.
  absl::Status CreateStatus() &&;

  // nullptr if the result status will be OK.  Extra fields moved to the heap to
  // minimize stack space.
  std::unique_ptr<Rep> rep_;
};

// A status builder that already contains a message. It can be implicitly
// converted to absl::Status or absl::StatusOr<T>.
// This class is copyable and movable.
class [[nodiscard]] StatusBuilderWithMessage : public StatusBuilder {
 public:
  using StatusBuilder::StatusBuilder;

  // Implicit conversion to Status.
  //
  // Careful: this operator has side effects, so it should be called at
  // most once. In particular, do NOT use this conversion operator to inspect
  // the status from an adapter object passed into With().
  //
  // Style guide exception for using Ref qualified methods and for implicit
  // conversions (cl/124566728).  This override allows us to implement
  // RETURN_IF_ERROR with 2 move operations in the common case.
  operator absl::Status() const&;  // NOLINT: Builder converts implicitly.
  operator absl::Status() &&;      // NOLINT: Builder converts implicitly.
};

// Implicitly converts `builder` to `Status` and write it to `os`.
inline std::ostream& operator<<(std::ostream& os,
                                const StatusBuilderWithMessage& builder) {
  return os << absl::Status(builder);
}
inline std::ostream& operator<<(std::ostream& os,
                                StatusBuilderWithMessage&& builder) {
  return os << absl::Status(std::move(builder));
}

inline StatusBuilder::StatusBuilder(const StatusBuilder& sb) {
  if (sb.rep_ != nullptr) {
    rep_ = std::make_unique<Rep>(*sb.rep_);
  }
}

inline StatusBuilder& StatusBuilder::operator=(const StatusBuilder& sb) {
  if (sb.rep_ != nullptr) {
    rep_ = std::make_unique<Rep>(*sb.rep_);
  } else {
    rep_ = nullptr;
  }
  return *this;
}

inline StatusBuilder& StatusBuilder::SetPrepend() & {
  if (rep_ == nullptr) return *this;
  rep_->message_join_style = MessageJoinStyle::kPrepend;
  return *this;
}
inline StatusBuilder&& StatusBuilder::SetPrepend() && {
  return std::move(SetPrepend());
}

inline StatusBuilder& StatusBuilder::SetOverride() & {
  if (rep_ == nullptr) return *this;
  rep_->message_join_style = MessageJoinStyle::kOverride;
  return *this;
}
inline StatusBuilder&& StatusBuilder::SetOverride() && {
  return std::move(SetOverride());
}

template <typename T>
StatusBuilderWithMessage& StatusBuilder::operator<<(const T& value) & {
  if (rep_ != nullptr) {
    if (!rep_->stream.has_value()) {
      rep_->InitStream();
    }
    *rep_->stream << value;
  }
  return *static_cast<StatusBuilderWithMessage*>(this);
}

template <typename T>
StatusBuilderWithMessage&& StatusBuilder::operator<<(const T& value) && {
  return std::move(operator<<(value));
}

inline bool StatusBuilder::ok() const {
  return rep_ == nullptr ? true : rep_->status.ok();
}

inline StatusBuilderWithMessage::operator absl::Status() const& {
  if (rep_ == nullptr) return absl::OkStatus();
  return StatusBuilder(*this).CreateStatus();
}

// Returns the state of the TORCH_SHOW_CPP_STACKTRACES environment variable:
// - std::nullopt: The environment variable is not set.
// - true: The environment variable is set to "1" or "true".
// - false: The environment variable is set to other values.
//
// For efficiency, this function only checks the environment variable once
// and caches the result. Therefore it will always return the same value
// during the lifetime of the process.
[[nodiscard]] std::optional<bool> TorchShowCppStacktraces();

namespace internal {

// Key for the payload in absl::Status that stores the C++ error trace,
// which is a multi-line string where each line is in the format of
//   <file>:<line>: <function>()
//
// This trace records the history of the absl::Status being created by TT_ERROR
// or TT_RET_CHECK and bubbled up by TT_ASSIGN_OR_RETURN or TT_RETURN_IF_ERROR.
// It is ordered by the time of the event. For example:
//   foo.cc:123: bar()
//   qux.cc:456: baz()
//   qux.cc:789: qux()
//   ...
// means that the absl::Status is originated in function bar() in foo.cc:123,
// then propagated up to baz() in qux.cc:456, then propagated up to qux() in
// qux.cc:789, and so on.
//
// This trace can be very helpful in debugging how an error is generated and
// propagated across the C++ call stack.
inline constexpr std::string_view kCppErrorTraceUrl =
    "type.googleapis.com/torch_tpu.cpp_error_trace";

// Key for the payload in absl::Status that stores the root op name.
inline constexpr std::string_view kRootOpNameUrl =
    "type.googleapis.com/torch_tpu.root_op_name";

// Sets the root op name in the absl::Status payload.
absl::Status SetRootOpNamePayload(absl::Status status,
                                  std::string_view root_op_name);

// Returns the root op name in the absl::Status payload.
[[nodiscard]] std::string GetRootOpNamePayload(const absl::Status& status);

// Appends the given C++ source location to the absl::Status payload if
// requested by the user and status is not OK.
absl::Status MaybeAddCppSourceLoc(absl::Status status, std::string_view file,
                                  int line, std::string_view function);

// Returns the message of the given status, with the C++ error trace in its
// payload if it is present.
[[nodiscard]] std::string GetMessageWithCppErrorTrace(
    const absl::Status& status);

}  // namespace internal
}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_STATUS_BUILDER_H_
