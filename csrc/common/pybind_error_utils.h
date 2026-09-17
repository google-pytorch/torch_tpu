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
#include "pybind11/gil.h"
#include "pybind11/pybind11.h"
#include "torch/csrc/Exceptions.h"
#include "torch/headeronly/util/Metaprogramming.h"

namespace torch_tpu {
namespace internal {

// Detects if a type is pybind11 class_.
//
// Used to distinguish between a pybind11 module_ and a pybind11 class_ when
// wrapping.
template <typename T>
struct IsPyBindClass : std::false_type {};

template <typename... Args>
struct IsPyBindClass<           //
    pybind11::class_<Args...>>  // RAW_PYBIND_OK=Wrapper implementation.
    : std::true_type {};

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
  // In theory, we could inline this function into `wrap()` definition.
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
      try {
        return impl(api_name, f, std::forward<Args>(args)...);
      } catch (...) {
        // If the caller of this function explicitly releases the GIL before
        // calling this lambda, e.g. passes `call_guard<gil_scoped_release>()`
        // as argument to the registration function, e.g. `def()`, we need to
        // re-acquire it before PyTorch error handling. The reason being that
        // PyTorch error handling calls the Python C API for setting the
        // exception information.
        //
        // Otherwise, i.e. if the caller doesn't explicitly releases the GIL, we
        // should still be holding the GIL at this point. In this case, the RAII
        // below just increments a counter without trying to re-acquire the
        // lock.
        pybind11::gil_scoped_acquire gil;

        // Use of the pair HANDLE_TH_ERRORS and END_HANDLE_TH_ERRORS_PYBIND are
        // needed for correctly translating `c10::Error` into Python errors.
        // They are macros that add the required surrounding try-catch
        // statements. In summary, they will handle any error thrown in-between
        // them.
        HANDLE_TH_ERRORS
        throw;  // Re-throwing the exception within PyTorch error handling.
        END_HANDLE_TH_ERRORS_PYBIND
      }
    };
  }
};

// Unified helper that extracts function signature traits using PyTorch's
// c10::guts::infer_function_traits and wraps the callable with error handling.
template <typename F>
auto WrapWithErrorHandling(const std::string_view api_name, const F& f) {
  using InferredTraits = typename c10::guts::infer_function_traits<F>::type;
  using FuncType = typename InferredTraits::func_type;
  return ErrorHandlingHelper<FuncType>::wrap(api_name, f);
}

// Wrapper around pybind11 objects to automatically wrap `def()` calls with
// error handling.
//
// This should not be used directly. Use the `PyBindWrapWithErrorHandling()`
// function below, instead.
template <typename PyBindType>
class  //
    [[deprecated(
        "TODO(b/556294340): use the newer abstractions PyBindModule and "
        "PyBindClass.")]]  //
    PyBindErrorHandlingWrapper {
  static_assert(
      std::is_same_v<PyBindType, pybind11::module_>  // RAW_PYBIND_OK=Wrapper
                                                     // implementation.
          || internal::IsPyBindClass<PyBindType>::value,
      "PyBindErrorHandlingWrapper can only wrap a pybind11 module_ or a "
      "pybind11 class_ type.");

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
    // API name. Either of:
    //   - <class-name>.<name>
    //   - <name>
    const std::string api_name = absl::StrCat(prefix_, name);

    // Actually registers the API into the wrapped object.
    wrapped_.def(name, internal::WrapWithErrorHandling(api_name, f), extra...);
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

// Wrapper around pybind11 module_ to automatically wrap `def()` calls with
// error handling.
class PyBindModule {
 public:
  explicit PyBindModule(
      pybind11::module_ wrapped)  // RAW_PYBIND_OK=Wrapper implementation.
      : wrapped_(wrapped) {}

  // Registers the Python API `name` to the function `f` wrapped with TorchTPU
  // guidelines conforming error handling.
  template <typename F, typename... Extra>
  PyBindModule& def(const char* name, const F& f, const Extra&... extra) {
    wrapped_.def(name, internal::WrapWithErrorHandling(name, f), extra...);
    return *this;
  }

  // Creates and returns a new submodule wrapped in a `PyBindModule`.
  template <typename... Extra>
  PyBindModule def_submodule(const char* name, const Extra&... extra) {
    return PyBindModule(wrapped_.def_submodule(name, extra...));
  }

  // Imports an existing Python module and wraps it in a `PyBindModule`.
  static PyBindModule Import(const char* name) {
    return PyBindModule(
        pybind11::module_::import(  // RAW_PYBIND_OK=Wrapper implementation.
            name));
  }

  auto doc() { return wrapped_.doc(); }
  auto attr(const char* name) { return wrapped_.attr(name); }

  friend pybind11::handle GetPyBindHandle(const PyBindModule& m);

 private:
  pybind11::module_ wrapped_;  // RAW_PYBIND_OK=Wrapper implementation.
};

// Returns the underlying pybind11 handle from a `PyBindModule` instance.
//
// Its use should be avoided whenever possible, and only used when interfacing
// with pybind11 APIs that require the raw handle (such as `pybind11::enum_`).
inline pybind11::handle GetPyBindHandle(const PyBindModule& m) {
  return m.wrapped_;
}

// Wrapper around pybind11 class_ constructor to automatically wrap `def()`
// calls with error handling.
template <typename T, typename... Options>
class PyBindClass {
 public:
  using RawClass =
      pybind11::class_<T, Options...>;  // RAW_PYBIND_OK=Wrapper implementation.

  template <typename... Extra>
  PyBindClass(PyBindModule& m, const char* name, const Extra&... extra)
      : wrapped_(GetPyBindHandle(m), name, extra...) {
    auto class_name = wrapped_.attr("__name__").template cast<std::string>();
    ABSL_CHECK(!class_name.empty());  // CRASH_OK
    prefix_ = absl::StrCat(class_name, ".");
  }

  // Overload for init or other non-named defs (e.g., py::init with py::arg):
  // forwards directly.
  template <
      typename Init, typename... Extra,
      typename = std::enable_if_t<!std::is_convertible_v<Init, const char*>>>
  PyBindClass& def(Init&& init, Extra&&... extra) {
    wrapped_.def(std::forward<Init>(init), std::forward<Extra>(extra)...);
    return *this;
  }

  // Registers the Python API `name` to the function `f` wrapped with TorchTPU
  // guidelines conforming error handling.
  template <typename F, typename... Extra>
  PyBindClass& def(const char* name, const F& f, const Extra&... extra) {
    const std::string api_name = absl::StrCat(prefix_, name);
    wrapped_.def(name, internal::WrapWithErrorHandling(api_name, f), extra...);
    return *this;
  }

  // Forwards `def_readonly()` to the wrapped class_ without wrapping it with
  // error handling.
  //
  // If TorchTPU guidelines conforming error handling is needed, consider
  // changing this into a read-only property.
  template <typename... Args>
  PyBindClass& def_readonly(Args&&... args) {
    wrapped_.def_readonly(std::forward<Args>(args)...);
    return *this;
  }

  // Forwards `def_readwrite()` to the wrapped pybind11 class_ without
  // wrapping it with error handling.
  //
  // If TorchTPU guidelines conforming error handling is needed, consider
  // changing this into a property.
  template <typename... Args>
  PyBindClass& def_readwrite(Args&&... args) {
    wrapped_.def_readwrite(std::forward<Args>(args)...);
    return *this;
  }

  // Registers a read-only property `name` using getter function `f` wrapped
  // with TorchTPU guidelines conforming error handling.
  template <typename Getter, typename... Extra>
  PyBindClass& def_property_readonly(const char* name, const Getter& f,
                                     const Extra&... extra) {
    const std::string api_name = absl::StrCat(prefix_, name);
    wrapped_.def_property_readonly(
        name, internal::WrapWithErrorHandling(api_name, f), extra...);
    return *this;
  }

  // Registers a read-write property `name` using `getter` and `setter` wrapped
  // with TorchTPU guidelines conforming error handling.
  template <typename Getter, typename Setter, typename... Extra>
  PyBindClass& def_property(const char* name, const Getter& getter,
                            const Setter& setter, const Extra&... extra) {
    const std::string api_name = absl::StrCat(prefix_, name);
    wrapped_.def_property(
        name, internal::WrapWithErrorHandling(api_name, getter),
        internal::WrapWithErrorHandling(api_name, setter), extra...);
    return *this;
  }

 private:
  RawClass wrapped_;
  std::string prefix_;
};

// Defines a Python module whose module parameter is automatically wrapped with
// TorchTPU error handling.
//
// Inside the module definition, `variable` is an instance of
// `torch_tpu::PyBindModule&`. Calling `variable.def(...)` automatically
// attaches TorchTPU error handling.
//
// Example usage:
//
//   TT_PYBIND11_MODULE(my_module, m) {
//     m.def("my_func", &MyFunc);
//   }
//
// Expands to:
//
//   static void TorchTpuPyBindInitModule_my_module(
//       ::torch_tpu::PyBindModule& m);
//
//   PYBIND11_MODULE(  /* RAW_PYBIND_OK=Wrapper implementation. */
//     my_module, raw_my_module) {
//     ::torch_tpu::PyBindModule m_my_module(raw_my_module);
//     TorchTpuPyBindInitModule_my_module(m_my_module);
//   }
//
//   void TorchTpuPyBindInitModule_my_module(
//       ::torch_tpu::PyBindModule& m) {
//     m.def("my_func", &MyFunc);
//   }
#define TT_PYBIND11_MODULE(name, variable)                    \
  static void TorchTpuPyBindInitModule_##name(                \
      ::torch_tpu::PyBindModule& variable);                   \
                                                              \
  PYBIND11_MODULE(/* RAW_PYBIND_OK=Wrapper implementation. */ \
                  name, raw_##name) {                         \
    ::torch_tpu::PyBindModule m_##name(raw_##name);           \
    TorchTpuPyBindInitModule_##name(m_##name);                \
  }                                                           \
                                                              \
  void TorchTpuPyBindInitModule_##name(::torch_tpu::PyBindModule& variable)

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_PYBIND_ERROR_UTILS_H_
