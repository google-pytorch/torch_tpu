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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_ERROR_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_ERROR_H_

#include <utility>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "csrc/common/error_utils.h"

namespace torch_tpu::kiln::internal {

// Reports `error` to the active scope of the current thread, which holds the
// first error reported to it; later errors are ignored, and `error` must not
// be OK. `BuildKernel()` returns the held error. Kiln handle types signal an
// error with a null `ValueHandle`; the error itself is reported here.
// Must be called within an active BuildKernel() or IndexMap execution.
void ReportError(absl::Status error);

// Implementation detail of `TT_KILN_RET_CHECK`: reports the error assigned to
// it, then calls `result_fn` for the value that the current function returns.
template <typename ResultFn>
class ErrorReturner {
 public:
  explicit ErrorReturner(ResultFn result_fn)
      : result_fn_(std::move(result_fn)) {}

  auto operator=(absl::Status error) && {  // NOLINT: see `TT_KILN_RET_CHECK`.
    ReportError(std::move(error));
    return result_fn_();
  }

 private:
  ResultFn result_fn_;
};

// Opts `ErrorReturner` in to class template argument deduction.
template <typename ResultFn>
ErrorReturner(ResultFn) -> ErrorReturner<ResultFn>;

// TT_KILN_RET_CHECK(cond, error_code[, result]) << message;
//
// Checks a condition that the kernel being built must satisfy. Does nothing if
// `cond` holds. Otherwise reports an error with `error_code` and the streamed
// message to the active scope (see `ReportError()`), and then returns `result`
// from the current function, or returns nothing if `result` is omitted.
// `result` and the message are only evaluated if `cond` doesn't hold; the error
// is reported before `result` is evaluated, so it precedes any error that
// `result` reports.
//
// `result` can't be a braced list, and the form without `result` can't be used
// in a constructor or destructor.
//
// Example:
//   TT_KILN_RET_CHECK(axis < rank(), error::kInvalidArgument,
//                     CreateNullValue<Tile>(dtype()))
//       << "axis " << axis << " out of bounds for rank " << rank();
//
// Implementation notes:
// - `switch (0) case 0: default:` avoids a dangling-else warning when the macro
//   is the unbraced body of an `if`.
// - `result` is wrapped in a lambda so that it is evaluated after the error is
//   reported.
#define TT_KILN_RET_CHECK(cond, error_code, ...)                 \
  switch (0)                                                     \
  case 0:                                                        \
  default:                                                       \
    if (ABSL_PREDICT_TRUE((cond)))                               \
      /* do nothing */;                                          \
    else                                                         \
      return ::torch_tpu::kiln::internal::ErrorReturner(         \
                 [&] { /* REF_CAPTURE_OK=called synchronously */ \
                       return __VA_ARGS__;                       \
                 }) = TT_ERROR(error_code)

}  // namespace torch_tpu::kiln::internal

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_KILN_CORE_ERROR_H_
