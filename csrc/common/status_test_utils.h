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

#ifndef TORCH_TPU_CSRC_COMMON_STATUS_TEST_UTILS_H_
#define TORCH_TPU_CSRC_COMMON_STATUS_TEST_UTILS_H_

#include <utility>  // IWYU pragma: keep for std::move in macro body.

#include "gtest/gtest.h"

#define TT_STATUS_TEST_MACROS_CONCAT_NAME_INNER_(x, y) x##y
#define TT_STATUS_TEST_MACROS_CONCAT_NAME_(x, y) \
  TT_STATUS_TEST_MACROS_CONCAT_NAME_INNER_(x, y)

// Evaluates an expression `rexpr` that returns an `absl::StatusOr<T>`.
// On OK, moves its value into `lhs`. Otherwise, causes a fatal test failure
// and returns from the current test function.
//
// Drop-in replacement for `TF_ASSERT_OK_AND_ASSIGN` and `ASSERT_OK_AND_ASSIGN`
// that works in both google3 and OSS environments.
//
// Example usage:
//   TT_ASSERT_OK_AND_ASSIGN(auto val, ReturnsStatusOr());
//   TT_ASSERT_OK_AND_ASSIGN(const Type& val, ReturnsStatusOr());
//   TT_ASSERT_OK_AND_ASSIGN(val, ReturnsStatusOr());
#define TT_ASSERT_OK_AND_ASSIGN(lhs, rexpr)                                   \
  TT_ASSERT_OK_AND_ASSIGN_IMPL_(                                              \
      TT_STATUS_TEST_MACROS_CONCAT_NAME_(_status_or_value, __COUNTER__), lhs, \
      rexpr)

#define TT_ASSERT_OK_AND_ASSIGN_IMPL_(statusor, lhs, rexpr)      \
  auto statusor = (rexpr);                                       \
  ASSERT_TRUE(statusor.ok()) << statusor.status();               \
  /* This redundant check is to silence the "unchecked access */ \
  /* 'absl::StatusOr' value" clang-tidy warning. */              \
  if (!statusor.ok()) return;                                    \
  lhs = std::move(statusor).value()

#endif  // TORCH_TPU_CSRC_COMMON_STATUS_TEST_UTILS_H_
