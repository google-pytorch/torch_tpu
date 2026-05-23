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

#ifndef TORCH_TPU_OPS_MACROS_LOGGING_H_
#define TORCH_TPU_OPS_MACROS_LOGGING_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Generator.h"
#include "ATen/core/IListRef.h"
#include "ATen/core/TensorBody.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/absl_vlog_is_on.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "c10/core/Device.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/macro_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

// In dbg mode, checks that the arguments passed to TT_KERNEL() match the
// arguments for the kernel function that encloses the TT_KERNEL() call.
// Also, at verbosity level 1, logs the arguments to a C++ kernel, along with
// their variable names. Example:
//
//    TT_CHECK_AND_LOG_KERNEL_ARGS_(
//        OpName::kFoo, lhs, rhs, value, dims, predicate);
//
//  When enabled with absl logging verbosity level 1, the following is logged:
//   [C++ KERNEL foo]
//   lhs: <tensor debug>
//   rhs: <tensor debug>
//   value: <scalar debug>
//   dims: [1, 2, 3]
//   predicate: true
//
// Requirements:
//   - All arguments except the first are variable names.
//   - All argument types are printable via ToString().
//
// Do not use this macro directly. It's an implementation detail of TT_KERNEL().
//
// Implementation note: __VA_OPT__(,) expands to nothing if the variadic
// argument list is empty, and expands to a comma otherwise.
#define TT_CHECK_AND_LOG_KERNEL_ARGS_(op_name, ...) \
  TT_CHECK_AND_LOG_KERNEL_ARGS_IMPL_(op_name __VA_OPT__(, ) __VA_ARGS__)

// Implements TT_CHECK_AND_LOG_KERNEL_ARGS_. The redirection is needed to allow
// TT_CHECK_AND_LOG_KERNEL_ARGS_() to be used in a macro (e.g. TT_KERNEL).
//
// Implementation note: __VA_OPT__(,) expands to nothing if the variadic
// argument list is empty, and expands to a comma otherwise.
#define TT_CHECK_AND_LOG_KERNEL_ARGS_IMPL_(op_name, ...)                       \
  do {                                                                         \
    static_assert(::torch_tpu::internal::ArgsAreIdentifiers(#__VA_ARGS__),     \
                  "The op parameter list passed to TT_KERNEL() must contain "  \
                  "only identifier names.");                                   \
    constexpr std::string_view _func_sig = __PRETTY_FUNCTION__;                \
    static_assert(                                                             \
        ::torch_tpu::internal::CountFunctionParams(_func_sig) ==               \
            std::tuple_size_v<decltype(std::forward_as_tuple(__VA_ARGS__))>,   \
        "TT_KERNEL() called with mismatching number of arguments to "          \
        "the kernel function signature.");                                     \
    if (ABSL_VLOG_IS_ON(1)) {                                                  \
      std::ostringstream _ss;                                                  \
      ::torch_tpu::internal::LogKernelName(_ss, op_name);                      \
      const std::vector<std::string_view> _arg_names =                         \
          TT_ARGS_AS_STRINGS_(__VA_ARGS__);                                    \
      ::torch_tpu::internal::LogKernelArgs(                                    \
          _ss, _arg_names __VA_OPT__(, ) __VA_ARGS__);                         \
      ABSL_LOG(INFO) << _ss.str();                                             \
    }                                                                          \
    if constexpr (::torch_tpu::internal::kDebugMode) {                         \
      /* Only check the argument types once per kernel function. */            \
      static const bool _checked = [&] {                                       \
        ::torch_tpu::internal::CheckTtKernelArgList<op_name>(                  \
            ::torch_tpu::internal::KernelArgCheckerContext{                    \
                TT_NORMALIZED_FILE, __LINE__, _func_sig,                       \
                ::torch_tpu::internal::ParseArgTypesOrEmpty(_func_sig),        \
                TT_ARGS_AS_STRINGS_(__VA_ARGS__)} __VA_OPT__(, ) __VA_ARGS__); \
        return true;                                                           \
      }();                                                                     \
      static_cast<void>(_checked); /* VOID_CAST_OK=dummy result. */            \
    }                                                                          \
  } while (false)

namespace internal {

inline void LogKernelName(std::ostream& ss, std::string_view op_name) {
  ss << "[C++ KERNEL " << op_name << "]\n";
}

inline void LogKernelName(std::ostream& ss, OpName op_name) {
  LogKernelName(ss, ToString(op_name));
}

// SupportsStreaming<T>::value is true if and only if T supports streaming
// to std::ostream.
template <typename T, typename = void>
struct SupportsStreaming : std::false_type {};
template <typename T>
struct SupportsStreaming<
    T, std::enable_if_t<!std::is_same_v<
           decltype(std::declval<std::ostream&>() << std::declval<T>()), void>>>
    : std::true_type {};

// Variadic function LogKernelArgs() logs the arguments one by one.

// Base case: 0 arguments to log.
inline void LogKernelArgs(std::ostream& ss,
                          const absl::Span<const std::string_view> arg_names) {
  ABSL_CHECK(arg_names.empty())  // CRASH_OK
      << "INTERNAL: LogKernelArgs called with too many arg_names.";
}

// Recursive case: log the next argument and then call the function recursively
// with the remaining arguments.
template <typename T, typename... Args>
inline void LogKernelArgs(std::ostream& ss,
                          const absl::Span<const std::string_view> arg_names,
                          const T& arg, const Args&... rest_args) {
  ABSL_CHECK(!arg_names.empty())  // CRASH_OK
      << "INTERNAL: LogKernelArgs called with too few arg_names.";
  ss << arg_names.front() << ": " << ToString(arg) << "\n";
  LogKernelArgs(ss, arg_names.subspan(1), rest_args...);
}

// Parses and normalizes the argument types from a C++ function signature
// generated by __PRETTY_FUNCTION__. Returns an empty vector if parsing fails.
[[nodiscard]] std::vector<std::string> ParseArgTypesOrEmpty(
    std::string_view func_sig);

// Returns the number of parameters in a function signature string.
constexpr size_t CountFunctionParams(std::string_view func_sig) {
  size_t start_idx = func_sig.find_last_of('(');
  size_t end_idx = func_sig.find_last_of(')');
  if (start_idx == std::string_view::npos ||
      end_idx == std::string_view::npos || end_idx <= start_idx + 1) {
    return 0;
  }

  std::string_view args_str =
      func_sig.substr(start_idx + 1, end_idx - start_idx - 1);

  size_t num_params = 1;
  int open_brackets = 0;
  int open_parentheses = 0;
  for (size_t j = 0; j < args_str.size(); ++j) {
    const char c = args_str[j];
    if (c == '<') {
      ++open_brackets;
    } else if (c == '>') {
      --open_brackets;
    } else if (c == '(') {
      ++open_parentheses;
    } else if (c == ')') {
      --open_parentheses;
    } else if (c == ',' && open_brackets == 0 && open_parentheses == 0) {
      ++num_params;
    }
  }
  return num_params;
}

// Information needed for checking kernel argument types.
struct KernelArgCheckerContext {
  std::string_view file;  // Path to the file where TT_KERNEL() is called.
  int line = -1;          // Line number of the TT_KERNEL() call.
  // Signature of the kernel function, as generated by __PRETTY_FUNCTION__.
  std::string_view func_sig;
  // Argument types of the kernel function, as parsed and normalized from
  // func_sig.
  std::vector<std::string> arg_types;
  // Argument names passed to TT_KERNEL().
  std::vector<std::string_view> tt_kernel_arg_names;
};

// Trait to check T is std::array<bool, N> for some N.
template <typename T>
struct is_std_array_bool : std::false_type {};
template <size_t kSize>
struct is_std_array_bool<std::array<bool, kSize>> : std::true_type {};

// Trait to check T is absl::InlinedVector<int64_t, N> for some N.
template <typename T>
struct is_inlined_vector_int64 : std::false_type {};
template <size_t kSize>
struct is_inlined_vector_int64<absl::InlinedVector<int64_t, kSize>>
    : std::true_type {
  static constexpr size_t kCapacity = kSize;
};

// Does an op use Scalar inputs in TT_KERNEL()?
enum class UsesScalarInput {
  kNo,
  kYes,
};

// Reports a compiler error if the given op uses Scalar inputs in
// TT_KERNEL().
template <OpName kOpName, UsesScalarInput kUse>
void CheckScalarInput() {
  // Is this op an existing violation of the scalar promotion pattern?
  constexpr bool kIsLegacyViolation =
      false ||  // The `false` case is for nice code formatting.
                // DO NOT ADD NEW ENTRIES TO THIS LIST.
                // TODO: make this list empty.
                // go/keep-sorted start
      kOpName == OpName::kAdd ||                         //
      kOpName == OpName::kAddRelu_Scalar ||              //
      kOpName == OpName::kAddmmDtype ||                  //
      kOpName == OpName::kAddmmDtypeOut ||               //
      kOpName == OpName::kAddmmOut ||                    //
      kOpName == OpName::kArangeStartOut ||              //
      kOpName == OpName::kBaddbmmDtype ||                //
      kOpName == OpName::kBaddbmmDtypeOut ||             //
      kOpName == OpName::kBaddbmmOut ||                  //
      kOpName == OpName::kCdistForward ||                //
      kOpName == OpName::kEluBackwardGradInput ||        //
      kOpName == OpName::kEluOut ||                      //
      kOpName == OpName::kEq ||                          //
      kOpName == OpName::kExponential_ ||                //
      kOpName == OpName::kFill_Scalar ||                 //
      kOpName == OpName::kForeachAddList ||              //
      kOpName == OpName::kForeachAddTensor ||            //
      kOpName == OpName::kForeachAdd_List ||             //
      kOpName == OpName::kForeachAdd_Tensor ||           //
      kOpName == OpName::kForeachClampMaxScalar ||       //
      kOpName == OpName::kForeachClampMaxScalarList ||   //
      kOpName == OpName::kForeachClampMax_Scalar ||      //
      kOpName == OpName::kForeachClampMax_ScalarList ||  //
      kOpName == OpName::kForeachClampMinScalar ||       //
      kOpName == OpName::kForeachClampMinScalarList ||   //
      kOpName == OpName::kForeachClampMin_Scalar ||      //
      kOpName == OpName::kForeachClampMin_ScalarList ||  //
      kOpName == OpName::kForeachNormScalar ||           //
      kOpName == OpName::kForeachPowScalar ||            //
      kOpName == OpName::kForeachPowScalarAndTensor ||   //
      kOpName == OpName::kForeachPowScalarList ||        //
      kOpName == OpName::kForeachPow_Scalar ||           //
      kOpName == OpName::kForeachPow_ScalarList ||       //
      kOpName == OpName::kForeachSubList ||              //
      kOpName == OpName::kForeachSub_List ||             //
      kOpName == OpName::kGe ||                          //
      kOpName == OpName::kGt ||                          //
      kOpName == OpName::kHardtanh ||                    //
      kOpName == OpName::kHardtanhOut ||                 //
      kOpName == OpName::kHardtanh_ ||                   //
      kOpName == OpName::kHistc ||                       //
      kOpName == OpName::kIndexAddOut ||                 //
      kOpName == OpName::kIsInScalarTensorOut ||         //
      kOpName == OpName::kIsInTensorScalarOut ||         //
      kOpName == OpName::kLe ||                          //
      kOpName == OpName::kLeakyReluBackward ||           //
      kOpName == OpName::kLeakyReluOut ||                //
      kOpName == OpName::kLinalgVectorNormOut ||         //
      kOpName == OpName::kLinspaceOut ||                 //
      kOpName == OpName::kLocalScalarDense ||            //
      kOpName == OpName::kLogSoftmaxBackwardDataOut ||   //
      kOpName == OpName::kLt ||                          //
      kOpName == OpName::kMaskedFill_Scalar ||           //
      kOpName == OpName::kNativeBatchNormBackward ||     //
      kOpName == OpName::kNe ||                          //
      kOpName == OpName::kPdistForward ||                //
      kOpName == OpName::kPow ||                         //
      kOpName == OpName::kRemainder ||                   //
      kOpName == OpName::kRsub ||                        //
      kOpName == OpName::kScatterValueOut ||             //
      kOpName == OpName::kScatterValueReduceOut ||       //
      kOpName == OpName::kSoftmaxBackwardDataOut ||      //
      kOpName == OpName::kSoftplusBackwardGradInput ||   //
      kOpName == OpName::kSoftplusOut ||                 //
      kOpName == OpName::kSub ||                         //
      kOpName == OpName::kThresholdBackwardGradInput ||  //
      kOpName == OpName::kThresholdOut ||                //
      kOpName == OpName::kVar ||                         //
      kOpName == OpName::kVarOut ||                      //
      // go/keep-sorted end
      false;  // The `false` case is for nice code formatting.
  if constexpr (kUse == UsesScalarInput::kYes) {
    static_assert(
        kIsLegacyViolation,
        "Don't use Scalar inputs with TT_KERNEL(). Promote them to "
        "Tensors via PromoteScalar() first. See AtenLerpScalarOut() for "
        "an example.");
  } else {
    static_assert(!kIsLegacyViolation,
                  "This op already follows the best practice of promoting "
                  "Scalar inputs to Tensors via PromoteScalar(). Please remove "
                  "its OpName from the list of OpNames in CheckScalarInput() "
                  "in logging.h.");
  }
}

// Crashes if the type T does not match the given argument's type string.
template <OpName kOpName, typename T>
void CheckKernelArgType(  // NOLINT: cognitive complexity
    const KernelArgCheckerContext& context, const int arg_idx) {
  const std::string_view arg_name = context.tt_kernel_arg_names[arg_idx];
  const std::string_view normalized_arg_type_in_func_sig =
      context.arg_types[arg_idx];
  const auto message = [&]() {
    return absl::StrCat("\n", context.file, ":", context.line,
                        ": INTERNAL: expected the '", arg_name,
                        "' argument in TT_KERNEL()'s inner parenthesized list "
                        "to have normalized type ",
                        normalized_arg_type_in_func_sig,
                        " to match the kernel function signature");
  };
  if constexpr (std::is_same_v<T, at::Tensor>) {
    ABSL_CHECK(  // CRASH_OK
        normalized_arg_type_in_func_sig == "at::Tensor")
        << message();
  } else if constexpr (std::is_same_v<T, at::Scalar>) {
    CheckScalarInput<kOpName, UsesScalarInput::kYes>();
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "at::Scalar")
        << message();
  } else if constexpr (std::is_same_v<T, int>) {
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig, "int")  // CRASH_OK
        << message();
  } else if constexpr (std::is_same_v<T, std::string>) {
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig, "std::string")  // CRASH_OK
        << message();
  } else if constexpr (std::is_same_v<T, at::OptionalIntArrayRef>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "at::OptionalIntArrayRef")
        << message();
  } else if constexpr (std::is_same_v<T, at::TensorList>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "at::TensorList")
        << message();
  } else if constexpr (std::is_same_v<T, std::optional<at::ScalarType>>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "std::optional<at::ScalarType>")
        << message();
  } else if constexpr (std::is_same_v<T, std::string_view>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "std::string_view")
        << message();
  } else if constexpr (std::is_same_v<T, std::optional<std::string_view>>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "std::optional<std::string_view>")
        << message();
  } else if constexpr (std::is_same_v<T, std::optional<at::Scalar>>) {
    CheckScalarInput<kOpName, UsesScalarInput::kYes>();
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "std::optional<at::Scalar>")
        << message();
  } else if constexpr (std::is_same_v<T, at::ScalarType>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "at::ScalarType")
        << message();
  } else if constexpr (std::is_same_v<T, at::IntArrayRef>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "at::IntArrayRef")
        << message();
  } else if constexpr (std::is_same_v<T, bool>) {
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig, "bool")  // CRASH_OK
        << message();
  } else if constexpr (std::is_same_v<T, double>) {
    ABSL_CHECK(  // CRASH_OK
        normalized_arg_type_in_func_sig == "double" ||
        // Sometimes a kernel normalizes an optional<double> to double before
        // invoking TT_KERNEL().
        normalized_arg_type_in_func_sig == "std::optional<double>")
        << message();
  } else if constexpr (std::is_same_v<T, at::Device>) {
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig, "at::Device")  // CRASH_OK
        << message();
  } else if constexpr (std::is_same_v<T, std::optional<at::Device>>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "std::optional<at::Device>")
        << message();
  } else if constexpr (std::is_same_v<T, at::Layout>) {
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig, "at::Layout")  // CRASH_OK
        << message();
  } else if constexpr (std::is_same_v<T, std::optional<at::Layout>>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "std::optional<at::Layout>")
        << message();
  } else if constexpr (std::is_same_v<T, at::MemoryFormat>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "at::MemoryFormat")
        << message();
  } else if constexpr (std::is_same_v<T, std::optional<at::MemoryFormat>>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "std::optional<at::MemoryFormat>")
        << message();
  } else if constexpr (std::is_same_v<T, std::optional<at::Tensor>>) {
    ABSL_CHECK(  // CRASH_OK
        normalized_arg_type_in_func_sig == "std::optional<at::Tensor>")
        << message();
  } else if constexpr (std::is_same_v<T, std::optional<bool>>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "std::optional<bool>")
        << message();
  } else if constexpr (std::is_same_v<T, std::optional<double>>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "std::optional<double>")
        << message();
  } else if constexpr (std::is_same_v<T, std::optional<int64_t>>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "std::optional<int64_t>")
        << message();
  } else if constexpr (std::is_same_v<T, std::optional<int>>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "std::optional<int>")
        << message();
  } else if constexpr (std::is_same_v<T, std::optional<std::string>>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "std::optional<std::string>")
        << message();
  } else if constexpr (std::is_same_v<T, at::SymInt>) {
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig, "at::SymInt")  // CRASH_OK
        << message();
  } else if constexpr (std::is_same_v<T, at::DimnameList>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "at::DimnameList")
        << message();
  } else if constexpr (std::is_same_v<T, at::Generator>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "at::Generator")
        << message();
  } else if constexpr (std::is_same_v<T, std::optional<at::Generator>>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "std::optional<at::Generator>")
        << message();
  } else if constexpr (is_std_array_bool<T>::value) {
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig,  // CRASH_OK
                  absl::StrCat("std::array<bool, ", std::tuple_size_v<T>, ">"))
        << message();
  } else if constexpr (is_inlined_vector_int64<T>::value) {
    // Some kernels normalize integer/SymInt arrays to InlinedVector<int64_t,
    // N> before invoking TT_KERNEL().
    ABSL_CHECK(  // CRASH_OK
        normalized_arg_type_in_func_sig == "at::IntArrayRef" ||
        normalized_arg_type_in_func_sig == "at::SymIntArrayRef" ||
        normalized_arg_type_in_func_sig == "at::OptionalIntArrayRef" ||
        normalized_arg_type_in_func_sig == "at::OptionalSymIntArrayRef" ||
        normalized_arg_type_in_func_sig ==
            absl::StrCat("absl::InlinedVector<int64_t, ",
                         is_inlined_vector_int64<T>::kCapacity, ">"))
        << message();
  } else if constexpr (std::is_same_v<T,
                                      c10::List<std::optional<at::Tensor>>>) {
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig,  // CRASH_OK
                  "c10::List<std::optional<at::Tensor>>")
        << message();
  } else if constexpr (std::is_same_v<T, std::vector<at::Tensor>>) {
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig,  // CRASH_OK
                  "std::vector<at::Tensor>")
        << message();
  } else if constexpr (                                           //
      std::is_same_v<T, std::unordered_map<int64_t, int64_t>>) {  // NOLINT
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig,                // CRASH_OK
                  "std::unordered_map<int64_t, int64_t>")
        << message();
  } else if constexpr (std::is_same_v<T,
                                      std::vector<std::vector<at::Tensor>>>) {
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig,  // CRASH_OK
                  "std::vector<std::vector<at::Tensor>>")
        << message();
  } else if constexpr (                        //
      std::is_same_v<T, std::vector<int64_t>>  // INT_VEC_OK=generic code
  ) {
    // Some kernels normalize an integer/SymInt array to std::vector<int64_t>
    // before invoking TT_KERNEL().
    ABSL_CHECK(  // CRASH_OK
        normalized_arg_type_in_func_sig == "at::IntArrayRef" ||
        normalized_arg_type_in_func_sig == "at::SymIntArrayRef" ||
        normalized_arg_type_in_func_sig ==  //
            "std::vector<int64_t>")         // INT_VEC_OK=generic code
        << message();
  } else if constexpr (std::is_same_v<T, c10d::AllreduceOptions>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "c10d::AllreduceOptions")
        << message();
  } else if constexpr (std::is_same_v<T, c10d::BroadcastOptions>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "c10d::BroadcastOptions")
        << message();
  } else if constexpr (std::is_same_v<T, c10d::AllgatherOptions>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "c10d::AllgatherOptions")
        << message();
  } else if constexpr (std::is_same_v<T, c10d::GatherOptions>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "c10d::GatherOptions")
        << message();
  } else if constexpr (std::is_same_v<T, c10d::ScatterOptions>) {
    ABSL_CHECK_EQ(  // CRASH_OK
        normalized_arg_type_in_func_sig, "c10d::ScatterOptions")
        << message();
  } else if constexpr (std::is_same_v<T, c10d::ReduceScatterOptions>) {
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig,  // CRASH_OK
                  "c10d::ReduceScatterOptions")
        << message();
  } else if constexpr (std::is_same_v<T, c10d::AllToAllOptions>) {
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig,  // CRASH_OK
                  "c10d::AllToAllOptions")
        << message();
  } else if constexpr (std::is_same_v<T, c10d::BarrierOptions>) {
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig,  // CRASH_OK
                  "c10d::BarrierOptions")
        << message();
  } else if constexpr (std::is_same_v<T, c10::ArrayRef<c10::SymInt>>) {
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig,  // CRASH_OK
                  "at::SymIntArrayRef")
        << message();
  } else if constexpr (std::is_same_v<T, at::Storage>) {
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig, "at::Storage")  // CRASH_OK
        << message();
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig, "uint64_t")  // CRASH_OK)
        << message();
  } else if constexpr (std::is_same_v<T, int64_t>) {
    // Some kernels normalize a SymInt, optional<SymInt>, or optional<int64_t>
    // to int64_t before invoking TT_KERNEL().
    ABSL_CHECK(normalized_arg_type_in_func_sig == "int64_t" ||  // CRASH_OK
               normalized_arg_type_in_func_sig == "at::SymInt" ||
               normalized_arg_type_in_func_sig == "std::optional<at::SymInt>" ||
               normalized_arg_type_in_func_sig == "std::optional<int64_t>")
        << message();
  } else if constexpr (std::is_same_v<T, at::ITensorListRef>) {
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig,  // CRASH_OK
                  "at::ITensorListRef")
        << message();
  } else if constexpr (std::is_same_v<T,
                                      c10::IListRef<at::OptionalTensorRef>>) {
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig,  // CRASH_OK
                  "c10::IListRef<at::OptionalTensorRef>")
        << message();
  } else if constexpr (std::is_same_v<T, at::ArrayRef<at::Scalar>>) {
    CheckScalarInput<kOpName, UsesScalarInput::kYes>();
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig,  // CRASH_OK
                  "at::ArrayRef<at::Scalar>")
        << message();
  } else if constexpr (std::is_same_v<T, at::OptionalSymIntArrayRef>) {
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig,  // CRASH_OK
                  "at::OptionalSymIntArrayRef")
        << message();
  } else if constexpr (is_ignored_in_cache_key<T>::value) {
    // The argument is IgnoreInCacheKey(x), so we check the type of x.
    CheckKernelArgType<kOpName, typename T::value_type>(context, arg_idx);
  } else if constexpr (std::is_same_v<T, PromotedScalar>) {
    CheckScalarInput<kOpName, UsesScalarInput::kNo>();
    // The argument is a PromotedScalar, so we check it matches an at::Scalar
    // parameter in the kernel function signature.
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig, "at::Scalar")  // CRASH_OK
        << message();
  } else if constexpr (std::is_same_v<T, MaybePromotedScalar>) {
    CheckScalarInput<kOpName, UsesScalarInput::kNo>();
    // The argument is a MaybePromotedScalar, so we check it matches an
    // at::Scalar parameter in the kernel function signature.
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig, "at::Scalar")  // CRASH_OK
        << message();
  } else if constexpr (std::is_same_v<T, std::optional<PromotedScalar>>) {
    CheckScalarInput<kOpName, UsesScalarInput::kNo>();
    // The argument is an optional PromotedScalar, so we check it matches an
    // optional at::Scalar parameter in the kernel function signature.
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig,  // CRASH_OK
                  "std::optional<at::Scalar>")
        << message();
  } else if constexpr (std::is_same_v<T, std::vector<PromotedScalar>>) {
    CheckScalarInput<kOpName, UsesScalarInput::kNo>();
    // The argument is a vector of PromotedScalar, so we check it matches an
    // ArrayRef of at::Scalar parameter in the kernel function signature.
    ABSL_CHECK_EQ(normalized_arg_type_in_func_sig,  // CRASH_OK
                  "at::ArrayRef<at::Scalar>")
        << message();
  } else {
    static_assert(false, "Unsupported argument type.");
  }
}

// Variadic function CheckKernelArgTypes() validates the arguments one by
// one.
//
// Base case: 0 arguments to validate.
template <OpName kOpName>
inline void CheckKernelArgTypes(const KernelArgCheckerContext& context,
                                const int arg_idx) {}

// Recursive case: validate the next argument and then call the function
// recursively with the remaining arguments.
template <OpName kOpName, typename T, typename... Args>
inline void CheckKernelArgTypes(const KernelArgCheckerContext& context,
                                const int arg_idx, const T& arg,
                                const Args&... rest_args) {
  CheckKernelArgType<kOpName, T>(context, arg_idx);
  CheckKernelArgTypes<kOpName>(context, arg_idx + 1, rest_args...);
}

// Checks that the arguments passed to TT_KERNEL() match the arguments for the
// kernel function that encloses the TT_KERNEL() call.
template <OpName kOpName, typename... Args>
inline void CheckTtKernelArgList(const KernelArgCheckerContext& context,
                                 const Args&... args) {
  // 1. Check that the arguments contain no duplicates.
  std::set<std::string_view> arg_names_seen;
  for (const std::string_view arg_name : context.tt_kernel_arg_names) {
    ABSL_CHECK(  // CRASH_OK
        arg_names_seen.find(arg_name) == arg_names_seen.end())
        << "\n"
        << context.file << ":" << context.line << ": argument '" << arg_name
        << "' was passed to TT_KERNEL() multiple times.";
    arg_names_seen.insert(arg_name);
  }

  // 2. Check that the argument types match the kernel function signature.
  CheckKernelArgTypes<kOpName>(context, 0, args...);
}

}  // namespace internal

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_MACROS_LOGGING_H_
