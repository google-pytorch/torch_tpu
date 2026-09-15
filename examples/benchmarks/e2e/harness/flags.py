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

"""Common command-line flags for benchmark tests."""

from absl import flags

SKIP_BEHAVIOR = flags.DEFINE_enum(
    "skip_behavior",
    "skip",
    ["skip", "assert_raise", "ignore", "run_skipped"],
    "Behavior for run modes that are marked as skipped in the config."
    " Options:\n  skip: Skips the test if the benchmark is marked as skipped.\n"
    "  assert_raise: Runs the test and expects it to raise an exception if"
    " skipped.\n  ignore: Runs the test normally even if it is marked as"
    " skipped.\n  run_skipped: Only runs skipped tests (skips those not marked"
    " as skipped).",
)

ENABLE_XPROF = flags.DEFINE_bool(
    "enable_xprof",
    False,
    "Whether to enable xprof profiling.",
    allow_override=True,
)

DRY_RUN = flags.DEFINE_bool(
    "dry_run",
    False,
    "Whether to run in dry-run mode. If True, actual benchmark execution is"
    " skipped, and the list of targets that would have run are logged and"
    " written to the test outputs directory specified by environment variable"
    " TEST_UNDECLARED_OUTPUTS_DIR.",
)


RUN_IN_SUBPROCESS = flags.DEFINE_bool(
    "run_in_subprocess",
    True,
    "Whether to run each benchmark case in its own child process. A case that"
    " crashes the interpreter (a CUDA device-side assert, a SIGSEGV in a"
    " kernel) or poisons the device context then fails alone instead of"
    " taking every later case in the shard with it. Disable to run cases"
    " in-process, e.g. to attach a debugger or use --pdb_post_mortem.",
)
