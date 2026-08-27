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

// Custom test main function for OSS C++ unit tests in torch_tpu.
//
// Background & Motivation:
// Inside Google, C++ tests typically depend on a library that provides a
// `main()` function that automatically initializes GoogleTest AND parses
// command-line flags.
//
// In OSS, GoogleTest's default `gtest_main` only calls
// `testing::InitGoogleTest` and runs the tests; it does NOT call
// `absl::ParseCommandLine`. Consequently, any ABSL flags defined in
// test source files (e.g., using `ABSL_FLAG`) and passed via command-line
// arguments in BUILD rules are ignored in OSS builds.
//
// To ensure consistent behavior between Google and OSS, we use this library
// to provide `main()` for OSS C++ tests, ensuring that command-line flags are
// parsed correctly.

#include "absl/flags/parse.h"
#include "gtest/gtest.h"

int main(int argc, char** argv) {
  // Step 1: Initialize GoogleTest first. This strips GoogleTest-specific
  // command-line flags (such as `--gtest_filter` and `--gtest_repeat`) from
  // `argv` and adjusts `argc` accordingly.
  testing::InitGoogleTest(&argc, argv);

  // Step 2: Parse remaining command-line flags using Abseil's flag parser.
  // This populates any ABSL_FLAG definitions in the test binary from `argv`
  // (for example, flags passed via `args = [...]` in cc_test rules).
  absl::ParseCommandLine(argc, argv);

  // Step 3: Execute all registered unit tests and return the result code.
  return RUN_ALL_TESTS();
}
