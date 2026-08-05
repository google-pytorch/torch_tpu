# Copyright 2025 Google LLC
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

"""Utilities for testing error messages."""

import contextlib
import enum
import functools
from multiprocessing import connection as mp_connection
import re
import sys
import traceback
import unittest

from absl import flags
from absl.testing import absltest
import torch
from torch.distributed.elastic.multiprocessing import errors
import torch.multiprocessing as mp
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils

ChildFailedError = errors.ChildFailedError


TEST_MODE = flags.DEFINE_string(
    name="test_mode",
    required=True,
    default=None,
    help=(
        "Mode of the test. Valid options are 'tpu' (verify the error messages"
        " on TPU), 'gpu' (verify the error messages on GPU), and 'cov' (measure"
        " the error message coverage on TPU). Note that the 'cov' mode does NOT"
        " verify the correctness of the error messages (it only collects and"
        " prints the coverage stats). A cov-mode test only fails when the"
        " tested code fails to raise an exception or raises an exception of an"
        " unexpected type. Therefore running the test in the 'tpu' mode  is"
        " still necessary."
    ),
)


@functools.lru_cache(maxsize=1)
def device():
  """Returns the device to use for testing the error messages."""

  if TEST_MODE.value in ("tpu", "cov"):
    return torch.device("tpu")
  if TEST_MODE.value == "gpu":
    return torch.device("cuda")
  raise ValueError(f"Unsupported test mode: {TEST_MODE.value}")


def set_up_module() -> None:
  """Called by absltest after flags are parsed and before tests are run."""
  pass


@functools.lru_cache(maxsize=1)
def is_on_tpu() -> bool:
  """Returns True if the test is running on TPU."""
  return device().type == "tpu"


@functools.lru_cache(maxsize=1)
def is_on_gpu() -> bool:
  """Returns True if the test is running on GPU."""
  return device().type == "cuda"


# Matches a source location in the format of "file:line: ...".
_SOURCE_LOC_RE = re.compile(r"([/\w\.]+:\d+):.*")

# Matches a line in the format of "Exception raised from ... at file:line ...".
_EXCEPTION_RAISED_RE = re.compile(
    r"Exception raised from .* at ([/\w\.]+:\d+).*"
)

# Whether to allow `gpu=...` in `assert_raises_message()`.
# If False, passing `gpu` will raise an error. This prevents accidental
# misuse of the function in a TPU-only error test.
_allow_gpu_parameter = True


def _parse_error(msg: str):
  """Parses the message and prints the error macro locations it covers."""
  # An error message looks like:
  #
  #   empty(): product of dimension sizes ...
  #
  #   C++ error trace (starting from the origin):
  #   third_party/py/torch_tpu/common/error_utils.cc:53: SafeMultiply()
  #   third_party/py/torch_tpu/common/error_utils.cc:101:
  #   ValidateTensorByteSize()
  #   third_party/py/torch_tpu/device_buffer.cc:361: CreateEmpty()
  #   third_party/py/torch_tpu/ops/nullary_aten_kernels.cc:90: MakeEmptyBuffer()
  #
  #   Exception raised from operator() at
  #   third_party/py/torch_tpu/ops/nullary_aten_kernels.cc:155
  #   (most recent call first):
  #   ...
  #
  # We need to extract the source locations from the C++ error trace and the
  # "Exception raised from ..." line.
  for line in msg.splitlines():
    m = _SOURCE_LOC_RE.match(line)
    if m:
      print(f"\nCovered {m.group(1)}")
    else:
      m = _EXCEPTION_RAISED_RE.match(line)
      if m:
        print(f"\nCovered {m.group(1)}")
        break


class _MatchType(enum.Enum):
  """The type of match to perform on the error message."""

  EXACT = "exact"  # Verify that the message is exactly equal to the expected.
  SUFFIX = "suffix"  # Verify that the message ends with the expected message.


def _check_error_message(
    msg: str,
    expected_msg: str | re.Pattern[str],
    device_type: str,
    match_type: _MatchType = _MatchType.EXACT,
):
  """Checks that the error message matches the expected message.

  Or, in the coverage mode and when `device_type` is "tpu", parses the error
  message and prints the covered error macro locations to stdout.

  Args:
    msg: The error message to check.
    expected_msg: The expected error message. Can be either a string or a
      regular expression (the result of re.compile()). Prefer using a string. A
      regular expression should only be used when the error message contains
      unstable information such as a source line number, which should be
      extremely rare as we strive at not leaking implementation details in the
      error messages.
    device_type: The device the error message is for (either "tpu" or "gpu").
    match_type: The type of match to perform on the error message.
  """

  if TEST_MODE.value == "cov":
    if device_type == "tpu":
      _parse_error(msg)
    return

  if match_type == _MatchType.SUFFIX:
    assert isinstance(
        expected_msg, str
    ), "The expected message must be a string for suffix match."
    assert msg.endswith(expected_msg), (
        f"Expected the {device_type} message to end with:\n{expected_msg}\nbut"
        f" got:\n{msg}"
    )
    return

  if isinstance(expected_msg, str):
    assert (
        expected_msg == msg
    ), f"Expected the {device_type} message:\n{expected_msg}\nbut got:\n{msg}"
  else:  # `expected_msg` is a regex.
    # We don't use assertRegex as it does a substring match, while we want
    # a full match.
    assert expected_msg.fullmatch(msg), (
        f"Expected the {device_type} message to match:\n{expected_msg}\nbut"
        f" got:\n{msg}"
    )


def _append_error_test_failure_protocol(assert_fn):
  """Appends the test failure protocol to the error raised by `assert_fn`.

  Whenever `assert_fn` fails (either because of an error mismatch, or the fact
  that no errors were raised), this function makes sure to append the errors
  test protocol to the error message.

  As an example, the following note is appended:

  ```
  AssertionError: Expected the tpu message to match:
  ...
  but got:
  ...

  ----------------------------------------------------------------------

  NOTE: This test might fail depending on which PyTorch version is being used.

  If this is a failure from an `*errors_test` target, you might be seeing this
  because one of the scenarios below:
      1. You are updating the version of PyTorch used by TorchTPU.
      2. You are building against a different PyTorch version than the one used
         by the main TorchTPU project.
  In either case, just ignore the failure (the test is not currently enforced
  for new changes) - the TorchTPU team will fix it later.

  ----------------------------------------------------------------------
  Ran X test in X.XXXs
  ```

  Args:
    assert_fn: either one of `assert_raises_message` or
      `assert_subprocess_raises_message`.

  Returns:
    A context manager that internally uses the assertion function, appending
    the
    note if any exception is raised.
  """

  @contextlib.contextmanager
  @functools.wraps(assert_fn)
  def wrapper(*args, **kwargs):
    try:
      with assert_fn(*args, **kwargs):
        yield
    except Exception as e:
      raise type(e)(
          f"""{e}

----------------------------------------------------------------------

NOTE: This test might fail depending on which PyTorch version is being used.

If this is a failure from an `*errors_test` target, you might be seeing this because one of the scenarios below:
    1. You are updating the version of PyTorch used by TorchTPU.
    2. You are building against a different PyTorch version than the one used by the main TorchTPU project.
In either case, just ignore the failure (the test is not currently enforced for new changes) - the TorchTPU team will fix it later."""
      )

  return wrapper


@_append_error_test_failure_protocol
@contextlib.contextmanager
def assert_raises_message(
    exception_type: type[Exception],
    *,
    tpu: str | re.Pattern[str],
    gpu: str | re.Pattern[str] | None = None,
    message_reviewed_by: str | None = None,
):
  """Asserts that a specific exception and message is raised.

  The expected error message can be either a string or a regular expression
  (the result of re.compile()). Prefer using a string. A regular expression
  should only be used when the error message contains unstable information
  such as a source line number, which should be extremely rare as we strive
  at not leaking implementation details in the error messages.

  Args:
    exception_type: The expected exception type.
    tpu: The expected error message on TPU.
    gpu: The expected error message on GPU (defaults to `tpu` if omitted).
    message_reviewed_by: The LDAP of the engineer who reviewed the error
      message. This will be used by a future tool to find all error messages not
      reviewed by an engineer.
  """

  # Check that exception_type is a class derived from BaseException.
  assert issubclass(exception_type, BaseException), (
      "exception_type must be a class derived from BaseException. "
      "It cannot be anything else, including a tuple."
  )

  del message_reviewed_by  # Unused for now.

  if not _allow_gpu_parameter and gpu is not None:
    raise AssertionError("gpu=... is not allowed in TPU-only error tests.")

  if gpu is None:
    gpu = tpu

  try:
    yield
  except exception_type as e:
    msg = str(e)
    if TEST_MODE.value == "gpu":
      _check_error_message(msg, gpu, "gpu")
    else:
      _check_error_message(msg, tpu, "tpu")
  except BaseException as e:
    raise AssertionError(
        f"Expected {exception_type.__name__} but got {type(e).__name__}: {e}"
    ) from e
  else:
    raise AssertionError(
        f"Expected {exception_type.__name__} to be raised, but no exception was"
        " raised."
    )


@_append_error_test_failure_protocol
@contextlib.contextmanager
def assert_subprocess_raises_message(exception_type, expected_msg: str):
  """Asserts that a specific exception and message is raised by a subprocess.

  Args:
    exception_type: The expected exception type.
    expected_msg: The expected error message.
  """
  try:
    yield
  # When a subprocess raises an exception of type FooError, the parent process
  # translates it to a ProcessRaisedException with a message like:
  #
  #   -- Process 0 terminated with the following error:
  #   Traceback (most recent call last):
  #     ...
  #   FooError: <exception message>
  #
  # Therefore we verify that the parent process message ends with
  #   FooError: <expected message>
  except (mp.ProcessRaisedException, ChildFailedError) as e:
    msg = str(e)
    # ChildFailedError includes a large report. Extract the traceback part for
    # matching.
    if isinstance(e, ChildFailedError):
      # Find the first observed failure traceback.
      match = re.search(r"traceback : (.*?)(?:\n\s*==+|$)", msg, re.DOTALL)
      if match:
        msg = match.group(1).strip()

    # Both mp.spawn and torchrun include Tracebacks. We want to find the
    # final line which should be "ExceptionType: Expected Message"
    # Or for RuntimeError specifically, it often just ends with the message.

    # Let's extract the last non-empty line of the traceback.
    lines = [l.strip() for l in msg.strip().splitlines() if l.strip()]
    if not lines:
      raise AssertionError(f"Empty error message from subprocess: {msg}") from e

    # Check if the last line matches "ExceptionType: Expected Message"
    # OR if we are using ChildFailedError, it might be in a different format.

    # Try to find a line matching the pattern: "ErrorName: message"
    # We look from the bottom up.
    found_match = False
    expected_full = f"{exception_type.__name__}: {expected_msg}"

    for line in reversed(lines):
      if line == expected_full or line == expected_msg:
        found_match = True
        break
      if (
          line.startswith(exception_type.__name__ + ":")
          and expected_msg in line
      ):
        found_match = True
        break

    if not found_match:
      raise AssertionError(
          f"Could not find expected message '{expected_full}' in subprocess"
          f" output:\n{msg}"
      ) from e
  except BaseException as e:
    raise AssertionError(
        "Expected ProcessRaisedException or ChildFailedError but got"
        f" {type(e).__name__}: {e}"
    ) from e
  else:
    raise AssertionError(
        "Expected ProcessRaisedException or ChildFailedError to be raised, but"
        " no exception was raised."
    )


# Attribute name to store the @why_tpu_only reason in the test method.
_WHY_TPU_ONLY_ATTR = "_why_tpu_only"

# Explanation for why we need the @why_tpu_only decorator on a test method
# in TpuOnlyErrorTestBase.
_EXPLAIN_WHY_TPU_ONLY = (
    "TpuOnlyErrorTestBase is for verifying errors that occur on TPU but not on"
    " GPU, which usually mean a bug in TPU code. If the code under test fails"
    " on GPU too, the test case should be moved to errors_test.py instead to"
    " verify that the code fails on both TPU and GPU."
)


def why_tpu_only(reason: str):
  """Decorates a TPU-only error test method with why it only applies to TPU."""
  if not reason or not isinstance(reason, str) or not reason.strip():
    raise ValueError(
        "why_tpu_only requires a non-empty explanation of why this test case"
        f" only applies to TPU: {_EXPLAIN_WHY_TPU_ONLY}"
    )

  def decorator(func):
    func._why_tpu_only = reason  # pylint: disable=protected-access
    return func

  return decorator


def _get_why_tpu_only_reason(
    test_case: seed_test_utils.RepeatableTest,
) -> str | None:
  """Retrieves the @why_tpu_only reason for the active test method.

  When test methods are wrapped by decorators or parameterized test runners
  (e.g. absl.testing.parameterized), the generated test methods (e.g.
  test_foo_0) do not directly inherit custom attributes from the underlying
  function. This function inspects the method and unwraps any standard Python
  __wrapped__ chains to locate the @why_tpu_only reason.

  Args:
    test_case: The active test case instance.

  Returns:
    The explanation string if the active method is decorated with @why_tpu_only,
    or None otherwise.
  """
  method = getattr(
      test_case, test_case._testMethodName  # pylint: disable=protected-access
  )
  reason = getattr(method, _WHY_TPU_ONLY_ATTR, None)
  if reason is None:
    # Traverse standard Python __wrapped__ chains (set by functools.wraps) to
    # find the underlying decorated function. This avoids relying on private
    # test runner internals (like _test_params_reprs) or fragile test name
    # prefix matching.
    curr = method
    while hasattr(curr, "__wrapped__"):
      curr = curr.__wrapped__
      r = getattr(curr, _WHY_TPU_ONLY_ATTR, None)
      if r is not None:
        reason = r
        # Set the attribute in the original method for fast future lookups.
        setattr(method, _WHY_TPU_ONLY_ATTR, r)
        break
  return reason


_OUTCOME_SUCCESS = "success"
_OUTCOME_SKIPPED = "skipped"
_OUTCOME_FAILURE = "failure"
_OUTCOME_ERROR = "error"


def _subprocess_test_worker(
    cls: type[absltest.TestCase],  # ABSLTEST_OK=Reference to absltest.TestCase
    method_name: str,
    conn: mp_connection.Connection,
) -> None:
  """Executes a single TestCase method in an isolated child process.

  Must be called in a child process.

  Args:
    cls: The test case class.
    method_name: The name of the test method to run.
    conn: The connection to send the result to.
  """
  try:
    instance = cls(method_name)
    res = unittest.TestResult()  # UNITTEST_OK=TestResult object for subprocess
    absltest.TestCase.run(  # ABSLTEST_OK=Subprocess runner execution
        instance, res
    )
    if res.failures:
      conn.send((_OUTCOME_FAILURE, res.failures[0][1]))
    elif res.errors:
      conn.send((_OUTCOME_ERROR, res.errors[0][1]))
    elif res.skipped:
      conn.send((_OUTCOME_SKIPPED, res.skipped[0][1]))
    else:
      conn.send((_OUTCOME_SUCCESS, None))
  except Exception:  # pylint: disable=broad-exception-caught
    conn.send((_OUTCOME_ERROR, traceback.format_exc()))
  finally:
    conn.close()


class ErrorTestBase(seed_test_utils.RepeatableTest):
  """Base class for error tests."""

  def assertRaisesRegex(self, *args, **kwargs):
    """Bans the default assertRaisesRegex() in favor of assert_raises_message()."""

    self.fail(
        "You must use et.assert_raises_message() instead of assertRaisesRegex()"
        " to check the error on GPU and TPU. Using the same API guarantees "
        "consistency."
    )

  def setUp(self):
    super().setUp()
    # Error tests sometimes force op dispatch failures. We reset the forced
    # op dispatch failures in setUp() and tearDown() to avoid influencing
    # other tests.
    tt_testing.set_op_dispatch_failure("", "")
    tt_testing.reset_eager_state()

  def tearDown(self):
    # Error tests sometimes force op dispatch failures. We reset the forced
    # op dispatch failures in setUp() and tearDown() to avoid influencing
    # other tests.
    tt_testing.set_op_dispatch_failure("", "")
    super().tearDown()

  def run(self, result=None):
    """Executes the test case, isolating GPU tests inside clean subprocesses.

    Args:
      result: The TestResult object to populate with test outcome data.

    Returns:
      The populated TestResult object.

    IPC Protocol:
      The parent process spawns an isolated child process running
      `_subprocess_test_worker` and communicates via a one-way pipe (`Pipe`).
      Before exiting, the child process transmits a single 2-tuple payload
      `(outcome, detail)` over the pipe:
      - (_OUTCOME_SUCCESS, None): Test passed.
      - (_OUTCOME_SKIPPED, reason_str): Test skipped.
      - (_OUTCOME_FAILURE, traceback_str): Test failed an assertion.
      - (_OUTCOME_ERROR, traceback_str): Test raised an unexpected exception.

      If the child process terminates abruptly (e.g., CUDA device-side assert
      or SIGSEGV) without transmitting a payload, the parent detects the
      non-zero exit code and records a test failure.
    """
    if TEST_MODE.value != "gpu":
      return super().run(result)

    if result is None:
      result = self.defaultTestResult()
      start_test_run = getattr(result, "startTestRun", None)
      if start_test_run is not None:
        start_test_run()

    # Rationale for choosing `multiprocessing` over `subprocess` or
    # standard `mp`:
    # 1. Standard `fork` is banned in Google3 (go/python-tips/018) because
    #    multithreaded C++ runtimes (Borg logging, gRPC, CUDA) deadlock or
    #    SIGSEGV when forked.
    # 2. Standard `spawn` (and `resource_tracker` threads launched by
    #    `Queue`) fails in self-contained Bazel `.par` zip binaries where
    #    `sys.executable` evaluates to None.
    # 3. Standard `subprocess.run` lacks object pickling / IPC exception
    #    transport needed to execute single parameterized `TestCase` class
    #    methods mid-flight.
    ctx = multiprocessing.get_context("forkserver")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    # Create a subprocess to run the test method.
    p = ctx.Process(
        target=_subprocess_test_worker,
        args=(self.__class__, self._testMethodName, child_conn),
    )
    p.start()
    child_conn.close()
    p.join()

    if p.exitcode != 0 and not parent_conn.poll():
      parent_conn.close()
      try:
        raise RuntimeError(
            "CUDA device-side assert triggered or subprocess crashed with exit"
            f" code {p.exitcode}"
        )
      except RuntimeError:
        result.addFailure(self, sys.exc_info())
      return

    outcome, detail = parent_conn.recv()
    parent_conn.close()
    if outcome == _OUTCOME_SUCCESS:
      result.addSuccess(self)
    elif outcome == _OUTCOME_SKIPPED:
      result.addSkip(self, detail)
    elif outcome == _OUTCOME_FAILURE:
      try:
        raise AssertionError(f"Subprocess test failed:\n{detail}")
      except AssertionError:
        result.addFailure(self, sys.exc_info())
    elif outcome == _OUTCOME_ERROR:
      try:
        raise RuntimeError(f"Subprocess test errored:\n{detail}")
      except RuntimeError:
        result.addError(self, sys.exc_info())


class TpuOnlyErrorTestBaseNoCheckingWhy(ErrorTestBase):
  """Base class for error tests that are only relevant for TPU.

  This class does not enforce the @why_tpu_only decorator on test methods.
  """

  def setUp(self):
    super().setUp()

    if TEST_MODE.value == "gpu":
      self.fail(
          "This test is only relevant for TPU. Please run it with only"
          " --test_mode=tpu or --test_mode=cov."
      )

    # Disable GPU error testing for TPU-only error tests.
    global _allow_gpu_parameter
    self.old_allow_gpu_parameter = _allow_gpu_parameter
    _allow_gpu_parameter = False

  def tearDown(self):
    # Restore the original value of _allow_gpu_parameter.
    global _allow_gpu_parameter
    _allow_gpu_parameter = self.old_allow_gpu_parameter

    super().tearDown()


class TpuOnlyErrorTestBase(TpuOnlyErrorTestBaseNoCheckingWhy):
  """Base class for error tests that are only relevant for TPU.

  This class enforces that each test method has a @why_tpu_only decorator.
  """

  def setUp(self):
    super().setUp()

    # Enforce that the test method has a @why_tpu_only decorator.
    if _get_why_tpu_only_reason(self) is None:
      self.fail(
          'This test method must be decorated with @why_tpu_only("reason")'
          f" to explain why it only applies to TPU: {_EXPLAIN_WHY_TPU_ONLY}"
      )


class TpuOnlyDistributedErrorTestBase(TpuOnlyErrorTestBaseNoCheckingWhy):
  """Base class for distributed error tests that are only relevant for TPU.

  This class does not enforce the @why_tpu_only decorator on test methods.
  """

  seed_in_setup = False


def get_scaled_mm_v2_default_inputs():
  """Returns default input tensors for torch._scaled_mm_v2 error tests."""
  m, n, k = 16, 16, 16
  dtype = torch.float8_e4m3fn
  dev = device()
  self_fp8 = torch.randn(m, k, dtype=torch.float32).to(dtype).to(dev)
  mat2_fp8 = torch.randn(k, n, dtype=torch.float32).to(dtype).to(dev)
  scale_a = [torch.tensor([1.5], dtype=torch.float32, device=dev)]
  scale_b = [torch.tensor([2.0], dtype=torch.float32, device=dev)]
  recipe_a = [0]
  recipe_b = [0]
  swizzle_a = [0]
  swizzle_b = [0]
  return (
      self_fp8,
      mat2_fp8,
      scale_a,
      recipe_a,
      swizzle_a,
      scale_b,
      recipe_b,
      swizzle_b,
  )
