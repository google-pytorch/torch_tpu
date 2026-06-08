# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import pathlib
import shutil
import threading
import time

from absl import logging
from absl.testing import absltest
import torch
from torch.autograd import profiler
from torch_tpu._internal import sync as tpu_sync
from torch_tpu._internal.profiler.profiler_config import TpuProfilerConfig

# pylint: disable=g-direct-tensorflow-import
from tsl.profiler.protobuf import profiler_options_pb2
from tsl.profiler.protobuf import xplane_pb2

# pylint: enable=g-direct-tensorflow-import


def _get_profile_dir() -> pathlib.Path:
  tmpdir = os.environ.get("TEST_TMPDIR") or os.environ.get("TMPDIR") or "/tmp"
  return pathlib.Path(tmpdir) / "plugins" / "profile"


class ComplexLoopedModel(torch.nn.Module):

  def __init__(self, num_loops=5):
    super(ComplexLoopedModel, self).__init__()
    self.num_loops = num_loops
    self.fc_in = torch.nn.Linear(1000, 4096)
    self.relu_in = torch.nn.ReLU()

    # Block to be repeated
    self.loop_fc1 = torch.nn.Linear(4096, 8192)
    self.loop_relu1 = torch.nn.ReLU()
    self.loop_fc2 = torch.nn.Linear(8192, 8192)
    self.loop_relu2 = torch.nn.ReLU()
    self.loop_fc3 = torch.nn.Linear(8192, 4096)
    self.loop_relu3 = torch.nn.ReLU()

    self.fc_out = torch.nn.Linear(4096, 1000)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.fc_in(x)
    x = self.relu_in(x)

    for i in range(self.num_loops):
      with profiler.record_function(f"loop_{i}"):
        x_res = x  # Residual connection
        x = self.loop_fc1(x)
        x = self.loop_relu1(x)
        x = self.loop_fc2(x)
        x = self.loop_relu2(x)
        x = self.loop_fc3(x)
        x = self.loop_relu3(x)
        if i > 0:  # Add residual connection after the first loop
          x = x + x_res
    return self.fc_out(x)


def _cleanup_profile_dir() -> None:
  try:
    shutil.rmtree(_get_profile_dir())
  except FileNotFoundError:
    pass


def _get_profiler_options_bytes(xspace: xplane_pb2.XSpace) -> bytes:
  """Traverses the XSpace to find the profile_options stat as bytes."""
  for plane in xspace.planes:
    if plane.name != "Task Environment":
      continue
    for stat in plane.stats:
      metadata = plane.stat_metadata.get(stat.metadata_id)
      if metadata and metadata.name == "profile_options":
        return stat.bytes_value
  return b""


class ProfilerIntegrationTest(absltest.TestCase):

  def _get_and_copy_xplane(self, destination_path: pathlib.Path) -> None:
    profile_dir = _get_profile_dir()
    xplane_files = list(profile_dir.glob("**/*.xplane.pb"))
    self.assertNotEmpty(xplane_files, f"No xplane file found in {profile_dir}")

    shutil.copy(xplane_files[0], destination_path)
    logging.info("Copied XPlane to %s", destination_path)

  def test_native_profiler(self):
    """Tests the StartTrace and StopTrace functionality integrated with PyTorch Kineto."""
    device = torch.device("tpu")

    a = torch.ones((16, 16)).to(device)
    b = torch.ones((16, 16)).to(device)
    # Warmup
    c = a @ b
    tpu_sync.synchronize(c)

    output_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if output_dir is None:
      output_dir = self.create_tempdir("trace_integration_tpu").full_path

    tpu_xplane_path = pathlib.Path(output_dir) / "xplane.pb"

    _cleanup_profile_dir()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
    ) as prof:
      c = a @ b
      tpu_sync.synchronize(c)
      prof.step()

    self._get_and_copy_xplane(tpu_xplane_path)

    # Data processing and setup logic
    traces = list(pathlib.Path(output_dir).glob("*.pt.trace.json"))

    trace_data = {}
    if traces:
      trace_path = traces[0]
      with open(trace_path, "r") as f:
        trace_data = json.load(f)

    events = trace_data.get("traceEvents", [])
    tpu_events = [e for e in events if e.get("cat") == "Trace"]
    cats = set(e.get("cat") for e in events if "cat" in e)

    xplane_exists = tpu_xplane_path.exists()
    xplane_contents = (
        list(pathlib.Path(output_dir).glob("*.*")) if not xplane_exists else []
    )

    # Subtests containing only assertions
    with self.subTest(msg="Kineto JSON Trace"):
      self.assertNotEmpty(traces, "Trace file should be created")
      self.assertIn("traceEvents", trace_data)
      self.assertNotEmpty(
          tpu_events,
          msg=f"Missing kernel tpu events. Found categories: {cats}",
      )

    with self.subTest(msg="TPU XPlane Output"):
      self.assertTrue(
          xplane_exists,
          f"XPlane missing! contents: {xplane_contents}",
      )

  def test_automatic_xplane_path(self):
    device = torch.device("tpu")

    a = torch.ones((16, 16)).to(device)
    b = torch.ones((16, 16)).to(device)
    c = a @ b
    tpu_sync.synchronize(c)

    output_dir = self.create_tempdir("auto_xplane").full_path

    _cleanup_profile_dir()
    profile_dir = _get_profile_dir()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    ) as prof:
      c = a @ b
      tpu_sync.synchronize(c)
      prof.step()

    self.assertTrue(
        profile_dir.exists(),
        "plugins/profile directory should exist under the base directory",
    )

    timestamp_dirs = list(profile_dir.glob("*"))
    self.assertNotEmpty(timestamp_dirs, "Should have a timestamp directory")

    xplane_files = list(timestamp_dirs[0].glob("*.xplane.pb"))
    self.assertNotEmpty(
        xplane_files, "Should have a .xplane.pb file in timestamp directory"
    )

  def test_full_pipeline_profiling(self):
    cpu_device = torch.device("cpu")
    tpu_device = torch.device("tpu")

    model_tpu = ComplexLoopedModel().to(tpu_device)
    model_cpu = ComplexLoopedModel().to(cpu_device)
    inputs = torch.ones(1000, 1000).to(tpu_device)

    output_dir = (
        pathlib.Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp"))
        / "debug_profiler_output"
    )
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Output directory: %s", output_dir)

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.PrivateUse1,
    ]

    _cleanup_profile_dir()

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,  # If True, might require torch.autograd.
        profile_memory=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    ):
      with profiler.record_function("full_pipeline"):
        with profiler.record_function("model1_inference_tpu"):
          output1_tpu = model_tpu(inputs)
          tpu_sync.synchronize(output1_tpu)

        with profiler.record_function("copy_to_cpu"):
          output1_cpu = output1_tpu.cpu()

        with profiler.record_function("model2_inference_cpu"):
          model_cpu(output1_cpu)

    self._get_and_copy_xplane(output_dir / "xplane.pb")

    self.assertTrue(
        (output_dir / "xplane.pb").exists(),
        f"XPlane missing! contents: {list(output_dir.glob('*.*'))}",
    )

    content = (output_dir / "xplane.pb").read_bytes()
    self.assertIn(
        b"model1_inference_tpu",
        content,
        "Python annotation not found in XPlane!",
    )

  def test_tpu_profiler_config_override(self):
    """Verifies that TPU-specific profiler options are correctly applied."""
    device = torch.device("tpu")
    base_output_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp")
    output_dir = pathlib.Path(base_output_dir) / "tpu_config_override"
    os.makedirs(output_dir, exist_ok=True)

    _cleanup_profile_dir()

    config = TpuProfilerConfig(device_tracer_level=3)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        experimental_config=config,
    ):
      a = torch.ones((16, 16)).to(device)
      b = torch.ones((16, 16)).to(device)
      c = a @ b
      tpu_sync.synchronize(c)

    tpu_xplane_path = pathlib.Path(output_dir) / "xplane.pb"
    self._get_and_copy_xplane(tpu_xplane_path)

    xspace = xplane_pb2.XSpace.FromString(tpu_xplane_path.read_bytes())
    options_bytes = _get_profiler_options_bytes(xspace)

    options = profiler_options_pb2.ProfileOptions.FromString(options_bytes)

    self.assertEqual(
        options.device_tracer_level,
        3,
        f"expected device_tracer_level=3, got {options.device_tracer_level}",
    )

  def test_tpu_profiler_config_run_dir(self):
    device = torch.device("tpu")
    custom_run_dir = self.create_tempdir("custom_run_dir_path").full_path

    _cleanup_profile_dir()

    config = TpuProfilerConfig(run_dir=pathlib.Path(custom_run_dir))

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(custom_run_dir),
        experimental_config=config,
    ):
      a = torch.ones((16, 16)).to(device)
      b = torch.ones((16, 16)).to(device)
      c = a @ b
      tpu_sync.synchronize(c)

    expected_profile_dir = pathlib.Path(custom_run_dir) / "plugins" / "profile"
    self.assertTrue(
        expected_profile_dir.exists(),
        "plugins/profile directory should exist under the configured run_dir",
    )

    timestamp_dirs = list(expected_profile_dir.glob("*"))
    self.assertLen(timestamp_dirs, 1)

    (timestamp_dir,) = timestamp_dirs
    xplane_files = list(timestamp_dir.glob("*.xplane.pb"))
    self.assertNotEmpty(
        xplane_files, "Should have a .xplane.pb file in custom run_dir"
    )

  def test_tpu_profiler_config_run_dir_none(self):
    device = torch.device("tpu")
    output_dir = self.create_tempdir("run_dir_none").full_path

    _cleanup_profile_dir()
    profile_dir = _get_profile_dir()

    config = TpuProfilerConfig(run_dir=None)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        experimental_config=config,
    ):
      a = torch.ones((16, 16)).to(device)
      b = torch.ones((16, 16)).to(device)
      c = a @ b
      tpu_sync.synchronize(c)

    self.assertTrue(
        profile_dir.exists(),
        "plugins/profile directory should exist under Kineto's fallback path",
    )

    timestamp_dirs = list(profile_dir.glob("*"))
    self.assertLen(timestamp_dirs, 1)

    (timestamp_dir,) = timestamp_dirs
    xplane_files = list(timestamp_dir.glob("*.xplane.pb"))
    self.assertNotEmpty(
        xplane_files, "Should have a .xplane.pb file in fallback directory"
    )

  def test_invalid_profiler_config(self):
    """Tests that invalid profiler configuration raises a Python exception."""
    device = torch.device("tpu")
    output_dir = self.create_tempdir("invalid_config").full_path

    _cleanup_profile_dir()

    bad_config = torch.profiler._ExperimentalConfig(
        custom_profiler_config="invalid_option_missing_colon"
    )

    with self.assertRaisesRegex(
        RuntimeError,
        r"expected the config item to be in the 'key:value' format, got"
        r" 'invalid_option_missing_colon'",
    ):
      with torch.profiler.profile(
          activities=[
              torch.profiler.ProfilerActivity.CPU,
              torch.profiler.ProfilerActivity.PrivateUse1,
          ],
          on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
          experimental_config=bad_config,
      ):
        c = torch.ones((16, 16)).to(device) @ torch.ones((16, 16)).to(device)
        tpu_sync.synchronize(c)

  def test_concurrent_profiling_and_execution(self):
    stop_workers = False

    def worker():
      while not stop_workers:
        _ = torch.randn(10, 10) + torch.randn(10, 10)

    threads = []
    for _ in range(8):
      t = threading.Thread(target=worker)
      t.start()
      threads.append(t)

    output_dir = self.create_tempdir("stress_test").full_path
    handler = torch.profiler.tensorboard_trace_handler(dir_name=output_dir)

    try:
      for _ in range(10):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            on_trace_ready=handler,
        ):
          time.sleep(0.01)
    finally:
      stop_workers = True
      for t in threads:
        t.join()


if __name__ == "__main__":
  absltest.main()
