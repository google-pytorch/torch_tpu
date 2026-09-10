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

#ifndef TORCH_TPU_CSRC_COMMON_PYBIND_ERROR_UTILS_H_
#define TORCH_TPU_CSRC_COMMON_PYBIND_ERROR_UTILS_H_

#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/strings/str_cat.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/python_context.h"
#include "pybind11/pybind11.h"
#include "torch/csrc/Exceptions.h"
#include "torch/headeronly/util/Metaprogramming.h"

namespace torch_tpu {
namespace internal {

// Detects if a type is pybind11::class_.
//
// Used to distinguish between a pybind11::module_ and a pybind11::class_ when
// wrapping.
template <typename T>
struct IsPyBindClass : std::false_type {};

template <typename... Args>
struct IsPyBindClass<pybind11::class_<Args...>> : std::true_type {};

// Helper struct to wrap functions bound to Python APIs with TorchTPU error
// handling.
//
// Specifically, it adds the following actions to the given function:
//   1. Captures the Python API name, so that it's prepended to the error.
//   2. Calls the `TranslateToC10ErrorAndThrow()` function.
//     - Passes them through the error message checks.
//     - Translates them into PyTorch `c10::Error`.
//   3. Translates the `c10::Error` into their corresponding Python errors.
template <typename FuncType>
struct ErrorHandlingHelper;

template <typename Ret, typename... Args>
struct ErrorHandlingHelper<Ret(Args...)> {
  // TorchTPU specific logic in the wrapper.
  //
  // Captures the Python API name, and calls the `TranslateToC10ErrorAndThrow()`
  // function.
  //
  // In theory, this we could inline this function onto `wrap()` definition.
  // However, we split it so that we can test it separately.
  template <typename F>
  static Ret impl(const std::string_view api_name, const F& f, Args... args) {
    ScopedPythonContextCapturer capturer(PythonApiName{api_name});
    try {
      return f(std::forward<Args>(args)...);
    } catch (const TtError& e) {
      TranslateToC10ErrorAndThrow(e);
    }
  }

  // Wraps the Python API callable in a lambda that executes with proper
  // TorchTPU error handling.
  template <typename F>
  static auto wrap(const std::string_view api_name, const F& f) {
    return [api_name = std::string(api_name), f](Args... args) -> Ret {
      // Use of the pair HANDLE_TH_ERRORS and END_HANDLE_TH_ERRORS_PYBIND are
      // needed for correctly translating `c10::Error` into Python errors. They
      // are macros that add the required surrounding try-catch statements. In
      // summary, they will handle any error thrown in-between them.
      HANDLE_TH_ERRORS
      return impl(api_name, f, std::forward<Args>(args)...);
      END_HANDLE_TH_ERRORS_PYBIND
    };
  }
};

// Wrapper around pybind11 objects to automatically wrap `def()` calls with
// error handling.
//
// This should not be used directly. Use the `PyBindWrapWithErrorHandling()`
// function below, instead.
template <typename PyBindType>
class PyBindErrorHandlingWrapper {
  static_assert(std::is_same_v<PyBindType, pybind11::module_> ||
                    internal::IsPyBindClass<PyBindType>::value,
                "PyBindErrorHandlingWrapper can only wrap pybind11::module_ "
                "or pybind11::class_ types.");

 public:
  // If `PyBindType` is a class, it sets the `prefix_` to "<class-name>."
  // (ending in a dot). Otherwise, it leaves it empty.
  //
  // The error message should have the following format:
  // `<class-name>.<api-name>(): <error-message>`
  explicit PyBindErrorHandlingWrapper(PyBindType& wrapped) : wrapped_(wrapped) {
    if constexpr (internal::IsPyBindClass<PyBindType>::value) {
      auto class_name = wrapped_.attr("__name__").template cast<std::string>();
      ABSL_CHECK(!class_name.empty());  // CRASH_OK
      prefix_ = absl::StrCat(class_name, ".");
    }
  }

  // Forward other common pybind11 class methods we use.
  template <typename... Args>
  PyBindErrorHandlingWrapper& def_readonly(Args&&... args) {
    wrapped_.def_readonly(std::forward<Args>(args)...);
    return *this;
  }

  // Overload for single-argument def (like py::init): forwards directly.
  template <typename Func>
  PyBindErrorHandlingWrapper& def(Func&& f) {
    wrapped_.def(std::forward<Func>(f));
    return *this;
  }

  // Registers the Python API `name` to the function `f` wrapped by the
  // `internal::ErrorHandlingHelper<FuncType>::wrap()` function.
  //
  // In this context, `FuncType` is the inferred function type `Ret(Args...)`
  // from `F` (actual type of `f`).
  template <typename F, typename... Extra>
  PyBindErrorHandlingWrapper& def(const char* name, const F& f,
                                  const Extra&... extra) {
    using InferredTraits = c10::guts::infer_function_traits<F>::type;
    using FuncType = typename InferredTraits::func_type;

    // API name. Either of:
    //   - <class-name>.<name>
    //   - <name>
    const std::string api_name = absl::StrCat(prefix_, name);

    // Actually registers the API into the wrapped object.
    wrapped_.def(name, ErrorHandlingHelper<FuncType>::wrap(api_name, f),
                 extra...);
    return *this;
  }

 private:
  // The pybind11 module or class object being wrapped.
  PyBindType& wrapped_;

  // The prefix string prepended to the API name.
  //
  // This is needed in cases where the wrapped object is a class registration.
  // It allows us to prepend the error message with the class name, so that the
  // error message format looks like:
  //
  // `<class-name>.<api-name>(): <error-message>`.
  std::string prefix_;
};

}  // namespace internal

// Creates a PyBindErrorHandlingWrapper for a pybind11 object.
//
// Convenient function for wrapping pybind11 objects (modules or classes)
// without needing to specify the template type.
template <typename T>
internal::PyBindErrorHandlingWrapper<T> PyBindWrapWithErrorHandling(
    T& wrapped) {
  return internal::PyBindErrorHandlingWrapper<T>(wrapped);
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_PYBIND_ERROR_UTILS_H_
