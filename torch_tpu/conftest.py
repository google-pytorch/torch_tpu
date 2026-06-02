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

"""Pytest configuration and global fixtures for TorchTPU tests."""

import os

import pytest


def pytest_addoption(parser):
  """Registers custom command-line options for TorchTPU pytest suite."""
  parser.addoption(
      "--torch-tpu-device",
      action="store",
      default=None,
      help="Type of device to run tests on. Valid values are tpu, cpu and gpu",
  )
  parser.addoption(
      "--max_samples_per_op_dtype",
      action="store",
      default=None,
      help="Max samples per op dtype",
  )
  parser.addoption(
      "--compute_grad",
      action="store",
      default=None,
      help="Compute grad",
  )
  parser.addoption(
      "--use_compiled",
      action="store",
      default=None,
      help="Use compiled",
  )
  parser.addoption(
      "--test_mode",
      action="store",
      default=None,
      help="Test mode",
  )
  parser.addoption(
      "--check_dynamism_using_seed",
      action="store",
      default=None,
      help="Check dynamism using seed",
  )
  parser.addoption(
      "--alsologtostderr",
      action="store_true",
      default=False,
      help="Alsologtostderr",
  )
  parser.addoption(
      "--suppress_failure_output",
      action="store_true",
      default=False,
      help="Suppress failure output",
  )
  parser.addoption(
      "--dtypes",
      action="store",
      default=None,
      help="Comma-separated list of dtypes to test",
  )
  parser.addoption(
      "--opt_level",
      action="store",
      default=None,
      help="Optimization level to run the test with.",
  )
  parser.addoption(
      "--update_perf_data",
      action="store",
      default=None,
      help="Update the output perf data files.",
  )
  parser.addoption(
      "--analyze",
      action="store",
      default=None,
      help="Analyze the performance results stored in --perf_dir.",
  )
  parser.addoption(
      "--perf_dir",
      action="store",
      default=None,
      help="Directory for storing the performance results.",
  )


@pytest.fixture(scope="session", autouse=True)
def device_type(request) -> str | None:
  """Returns the device_type provided via command line.

  Since it's autouse, it's available to the entire suite and only runs once.

  Args:
    request: The pytest request object.
  """
  device_type_val: str | None = request.config.getoption("--torch-tpu-device")
  valid_values = ("cpu", "xla_cuda", "tpu", None)
  if device_type_val not in valid_values:
    raise ValueError(
        f"Invalid device type: {device_type_val}, must be one of:\n  "
        + "\n  ".join(valid_values[:-1])
    )

  return device_type_val


@pytest.fixture(scope="session", autouse=True)
def initialize_debug_checks():
  """Initializes debug checks flags for session."""
  os.environ["TORCH_TPU_INTERNAL_ENABLE_DEBUG_CHECKS"] = "0"


def pytest_configure(config) -> None:
  """Configures pytest options and registers custom markers."""
  # register an additional marker
  config.addinivalue_line(
      "markers",
      "requires_device(device_type): "
      + "Skip tests when the specified device type doesn't match the required"
      " type",
  )


# pylint: disable=redefined-outer-name
@pytest.fixture(autouse=True)
def requires_device(request, device_type: str | None) -> None:
  """Automatically skip tests that require a device type we don't have.

  Args:
    request: The pytest request object.
    device_type: The device type fixture.
  """
  if request.node.get_closest_marker("requires_device"):
    if device_type is not None and (
        request.node.get_closest_marker("requires_device").args[0]
        != device_type
    ):
      pytest.skip(f"Test only runs on {device_type}")


@pytest.fixture(scope="session", autouse=True)
def configure_test_env(request) -> None:
  """Synchronizes pytest command-line options with environment variables."""
  # Synchronize test_mode
  test_mode_val = request.config.getoption("--test_mode")
  if test_mode_val:
    os.environ["TORCH_TPU_TEST_MODE"] = test_mode_val

  # Synchronize max_samples_per_op_dtype
  max_samples = request.config.getoption("--max_samples_per_op_dtype")
  if max_samples is not None:
    os.environ["TORCH_TPU_MAX_SAMPLES"] = str(max_samples)

  # Synchronize compute_grad
  compute_grad = request.config.getoption("--compute_grad")
  if compute_grad is not None:
    os.environ["TORCH_TPU_COMPUTE_GRAD"] = str(
        compute_grad.lower() in ("true", "1", "yes")
    )

  # Synchronize use_compiled
  use_compiled = request.config.getoption("--use_compiled")
  if use_compiled is not None:
    os.environ["TORCH_TPU_USE_COMPILED"] = str(
        use_compiled.lower() in ("true", "1", "yes")
    )

  # Synchronize check_dynamism_using_seed
  dynamism_seed = request.config.getoption("--check_dynamism_using_seed")
  if dynamism_seed is not None:
    os.environ["TORCH_TPU_CHECK_DYNAMISM_SEED"] = str(dynamism_seed)

  # Synchronize dtypes
  dtypes = request.config.getoption("--dtypes")
  if dtypes is not None:
    os.environ["TORCH_TPU_TEST_DTYPES"] = dtypes

  # Synchronize opt_level
  opt_level = request.config.getoption("--opt_level")
  if opt_level is not None:
    os.environ["TORCH_TPU_OPT_LEVEL"] = opt_level

  # Synchronize update_perf_data
  update_perf_data = request.config.getoption("--update_perf_data")
  if update_perf_data is not None:
    os.environ["TORCH_TPU_UPDATE_PERF_DATA"] = str(
        update_perf_data.lower() in ("true", "1", "yes")
    )

  # Synchronize analyze
  analyze = request.config.getoption("--analyze")
  if analyze is not None:
    os.environ["TORCH_TPU_ANALYZE"] = str(
        analyze.lower() in ("true", "1", "yes")
    )

  # Synchronize perf_dir
  perf_dir = request.config.getoption("--perf_dir")
  if perf_dir is not None:
    os.environ["TORCH_TPU_PERF_DIR"] = perf_dir

  # Mark ABSL flags as parsed if ABSL is present, to prevent warnings
  try:
    # pylint: disable=g-import-not-at-top
    from absl import flags

    if not flags.FLAGS.is_parsed():
      flags.FLAGS.mark_as_parsed()
  except ImportError:
    pass
