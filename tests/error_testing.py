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
import re

from absl import flags
from absl.testing import absltest
import torch
from torch.distributed.elastic.multiprocessing import errors
import torch.multiprocessing as mp
from torch_tpu import api
from torch_tpu._internal import testing as tt_testing


ChildFailedError = errors.ChildFailedError
TEST_MODE = flags.DEFINE_string(
    name="test_mode",
    required=True,
    default=None,
    help=(
        "Mode of the test. Valid options are 'tpu' (verify the error messages"
        " on TPU), 'cpu' (verify the error messages on CPU), and 'cov' (measure"
        " the error message coverage on TPU). Note that the 'cov' mode does"
        " NOT verify the correctness of the error messages (it only collects"
        " the coverage stats). Therefore running the test in the 'tpu' mode "
        " is still necessary."
    ),
)


@functools.lru_cache(maxsize=1)
def device():
  """Returns the device to use for testing the error messages."""

  if TEST_MODE.value in ("tpu", "cov"):
    return api.tpu_device()
  if TEST_MODE.value == "cpu":
    return torch.device("cpu")
  raise ValueError(f"Unsupported test mode: {TEST_MODE.value}")


def set_up_module(init_torch_tpu: bool = True):
  """Called by absltest after flags are parsed and before tests are run.

  Args:
    init_torch_tpu: Whether to initialize torch_tpu. This should be set to False
      for distributed tests, which need to initialize torch_tpu by itself.
  """

  if TEST_MODE.value in ("tpu", "cov") and init_torch_tpu:
    api.tpu_device()  # Initialize torch_tpu.


# Matches a source location in the format of "file:line: ...".
_SOURCE_LOC_RE = re.compile(r"([/\w\.]+:\d+):.*")

# Matches a line in the format of "Exception raised from ... at file:line ...".
_EXCEPTION_RAISED_RE = re.compile(
    r"Exception raised from .* at ([/\w\.]+:\d+).*"
)


def _parse_error(msg: str):
  """Parses the message and prints the error macro locations it covers."""
  # An error message looks like:
  #
  #   empty(): product of dimension sizes ...
  #
  #   C++ error trace (starting from the origin):
  #   third_party/py/torch_tpu/common/error_utils.cc:53: SafeMultiply()
  #   third_party/py/torch_tpu/common/error_utils.cc:101: ValidateTensorByteSize()
  #   third_party/py/torch_tpu/device_buffer.cc:361: CreateEmpty()
  #   third_party/py/torch_tpu/ops/nullary_aten_kernels.cc:90: MakeEmptyBuffer()
  #
  #   Exception raised from operator() at third_party/py/torch_tpu/ops/nullary_aten_kernels.cc:155 (most recent call first):
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
    device_type: The device the error message is for (either "tpu" or "cpu").
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
    exception_type,
    *,
    tpu: str | re.Pattern[str],
    cpu: str | re.Pattern[str] | None = None,
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
    cpu: The expected error message on CPU. If not provided, it is assumed to be
      the same as the TPU error message.
    message_reviewed_by: The LDAP of the engineer who reviewed the error
      message. This will be used by a future tool to find all error messages not
      reviewed by an engineer.
  """

  del message_reviewed_by  # Unused for now.

  if cpu is None:
    cpu = tpu

  try:
    yield
  except exception_type as e:
    msg = str(e)
    if TEST_MODE.value == "cpu":
      _check_error_message(msg, cpu, "cpu")
    else:
      _check_error_message(msg, tpu, "tpu")
  except BaseException as e:
    raise AssertionError(
        f"Expected {exception_type.__name__} but got {type(e).__name__}: {e}"
    ) from e
  else:
    raise AssertionError(
        f"Expected {exception_type.__name__} to be raised, but no exception"
        " was raised."
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


class ErrorTestBase(absltest.TestCase):
  """Base class for error tests."""

  def setUp(self):
    super().setUp()
    # Error tests sometimes force op dispatch failures. We reset the forced
    # op dispatch failures in setUp() and tearDown() to avoid influencing
    # other tests.
    tt_testing.set_op_dispatch_failure("", "")

  def tearDown(self):
    # Error tests sometimes force op dispatch failures. We reset the forced
    # op dispatch failures in setUp() and tearDown() to avoid influencing
    # other tests.
    tt_testing.set_op_dispatch_failure("", "")
    super().tearDown()


class TpuOnlyErrorTestBase(ErrorTestBase):
  """Base class for error tests that are only relevant for TPU."""

  def setUp(self):
    super().setUp()
    if TEST_MODE.value == "cpu":
      self.skipTest("Skipping test on CPU as it is only relevant for TPU.")
