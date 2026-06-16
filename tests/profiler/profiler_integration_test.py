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
import unittest

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch.autograd import profiler
import torch_tpu  # pylint: disable=unused-import
from torch_tpu._internal import sync as tpu_sync
from torch_tpu._internal.profiler import profiler_config

# pylint: disable=g-direct-tensorflow-import
from tsl.profiler.protobuf import profiler_options_pb2
from tsl.profiler.protobuf import xplane_pb2
# pylint: enable=g-direct-tensorflow-import


def _get_profile_dir() -> pathlib.Path:
  tmpdir = os.environ.get("TEST_TMPDIR") or os.environ.get("TMPDIR") or "/tmp"
  return pathlib.Path(tmpdir) / "plugins" / "profile"

TpuProfilerConfig = profiler_config.TpuProfilerConfig


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


class ProfilerIntegrationTest(parameterized.TestCase):

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
            torch.profiler.ProfilerActivity.TPU,  # type: ignore
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

  def test_kineto_key_averages_eager(self):
    device = torch.device("tpu")

    a = torch.ones((16, 16)).to(device)
    b = torch.ones((16, 16)).to(device)
    # Warmup
    c = a @ b
    tpu_sync.synchronize(c)

    _cleanup_profile_dir()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.TPU,  # type: ignore
        ],
        record_shapes=True,
    ) as prof:
      c = a @ b
      tpu_sync.synchronize(c)
      prof.step()

    key_averages = prof.key_averages()
    table_str = key_averages.table()
    print(table_str, flush=True)

    # Verify that the table output includes TPU time columns
    self.assertIn("TPU time", table_str)

    # Verify there are events with non-zero tpu time
    has_tpu_time = any(
        getattr(evt, "device_time_total", 0) > 0 for evt in key_averages
    )
    self.assertTrue(
        has_tpu_time,
        "Expected non-zero TPU time in key_averages for eager mode",
    )

  def test_kineto_key_averages_compile(self):
    device = torch.device("tpu")

    def my_func(a, b):
      return a @ b

    compiled_func = torch.compile(my_func, backend="tpu")

    a = torch.ones((16, 16)).to(device)
    b = torch.ones((16, 16)).to(device)
    # Warmup
    c = compiled_func(a, b)
    tpu_sync.synchronize(c)

    _cleanup_profile_dir()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.TPU,  # type: ignore
        ],
        record_shapes=True,
    ) as prof:
      c = compiled_func(a, b)
      tpu_sync.synchronize(c)
      prof.step()

    table_str = prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=10
    )
    print(table_str, flush=True)

    self.assertIn("TPU time", table_str)

    has_tpu_time = any(
        getattr(evt, "device_time_total", 0) > 0 for evt in prof.key_averages()
    )
    self.assertTrue(
        has_tpu_time,
        "Expected non-zero TPU time in key_averages for torch.compile mode",
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
            torch.profiler.ProfilerActivity.TPU,  # type: ignore
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
        torch.profiler.ProfilerActivity.TPU,  # type: ignore
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
            torch.profiler.ProfilerActivity.TPU,  # type: ignore
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
            torch.profiler.ProfilerActivity.TPU,  # type: ignore
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
            torch.profiler.ProfilerActivity.TPU,  # type: ignore
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
              torch.profiler.ProfilerActivity.TPU,  # type: ignore
          ],
          on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
          experimental_config=bad_config,
      ):
        c = torch.ones((16, 16)).to(device) @ torch.ones((16, 16)).to(device)
        tpu_sync.synchronize(c)

  def test_tpu_profiler_config_experimental_options(self):
    device = torch.device("tpu")
    output_dir = pathlib.Path(
        self.create_tempdir("experimental_options").full_path
    )

    config = TpuProfilerConfig(
        host_tracer_level=2,
        experimental_options=dict(
            tpu_num_sparse_cores_to_trace=4,
            tpu_trace_mode="TRACE_ONLY_HOST",
            tpu_perf_counters=True,
        ),
        run_dir=output_dir,
    )

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.TPU,  # type: ignore
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(output_dir)
        ),
        experimental_config=config,
    ):
      a = torch.ones((16, 16)).to(device)
      b = torch.ones((16, 16)).to(device)
      c = a @ b
      tpu_sync.synchronize(c)

    xplane_files = list(output_dir.glob("**/plugins/profile/**/*.xplane.pb"))
    self.assertLen(xplane_files, 1)
    (tpu_xplane_path,) = xplane_files

    xspace = xplane_pb2.XSpace.FromString(tpu_xplane_path.read_bytes())
    options_bytes = _get_profiler_options_bytes(xspace)
    options = profiler_options_pb2.ProfileOptions.FromString(options_bytes)

    advanced_config = options.advanced_configuration
    actual_subset = {
        "tpu_num_sparse_cores_to_trace": (
            advanced_config["tpu_num_sparse_cores_to_trace"].int64_value
        ),
        "tpu_trace_mode": advanced_config["tpu_trace_mode"].string_value,
        "tpu_perf_counters": advanced_config["tpu_perf_counters"].bool_value,
    }
    expected = {
        "tpu_num_sparse_cores_to_trace": 4,
        "tpu_trace_mode": "TRACE_ONLY_HOST",
        "tpu_perf_counters": True,
    }
    self.assertEqual(expected, actual_subset)

  @unittest.skip(
      "Skipped until new libtpu wheel containing BrokenProfiler is rolled into"
      " google3 (bootstrap dependency)."
  )
  def test_tpu_profiler_config_invalid_experimental_option(self):
    device = torch.device("tpu")
    output_dir = pathlib.Path(
        self.create_tempdir("invalid_experimental_option").full_path
    )

    config = TpuProfilerConfig(
        host_tracer_level=2,
        experimental_options={
            "invalid_option_key": "some_value",
        },
        run_dir=output_dir,
    )

    with self.assertRaisesRegex(
        RuntimeError,
        r"Parsing advanced_configuration failed\. The following keys "
        r"were not recognized: invalid_option_key",
    ):
      with torch.profiler.profile(
          activities=[
              torch.profiler.ProfilerActivity.CPU,
              torch.profiler.ProfilerActivity.TPU,  # type: ignore
          ],
          on_trace_ready=torch.profiler.tensorboard_trace_handler(
              str(output_dir)
          ),
          experimental_config=config,
      ):
        a = torch.ones((16, 16)).to(device)
        b = torch.ones((16, 16)).to(device)
        c = a @ b
        tpu_sync.synchronize(c)

  def test_tpu_profiler_config_invalid_experimental_option_ignored(self):
    device = torch.device("tpu")
    output_dir = pathlib.Path(
        self.create_tempdir("invalid_experimental_option_ignored").full_path
    )

    config = TpuProfilerConfig(
        host_tracer_level=2,
        experimental_options={
            "invalid_option_key": "some_value",
        },
        check_experimental_options=False,
        run_dir=output_dir,
    )

    # Should not raise any error.
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.TPU,  # type: ignore
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(output_dir)
        ),
        experimental_config=config,
    ):
      a = torch.ones((16, 16)).to(device)
      b = torch.ones((16, 16)).to(device)
      c = a @ b
      tpu_sync.synchronize(c)

  @parameterized.named_parameters(
      dict(testcase_name="colon", key="invalid:key", value="value"),
      dict(testcase_name="comma", key="invalid,key", value="value"),
  )
  def test_tpu_profiler_config_invalid_characters_in_experimental_options(
      self, key, value
  ):
    with self.assertRaisesRegex(
        ValueError,
        r"Experimental option keys cannot contain ':' or ','",
    ):
      TpuProfilerConfig(experimental_options={key: value})

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

  def test_timestamps(self):

    tpu_device = torch.device("tpu")

    def model_tpu(x):
      return x @ x

    def model_cpu(x):
      return x @ x

    inputs = torch.ones(10, 10).to(tpu_device)

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.TPU,
    ]

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
    ) as prof:
      output1_tpu = model_tpu(inputs)
      tpu_sync.synchronize(output1_tpu)
      output1_cpu = output1_tpu.cpu()
      model_cpu(output1_cpu)
      prof.step()

    print("\n--- EVENTS ---", flush=True)
    print("\n--- EVENTS ---", flush=True)
    tpu_min = float("inf")
    tpu_max = 0
    cpu_min = float("inf")
    cpu_max = 0

    tpu_starts = []

    for evt in prof.events():
      start = evt.time_range.start
      end = evt.time_range.end
      dtype = str(getattr(evt, "device_type", None))
      if "CPU" in dtype or "cpu" in dtype:
        cpu_min = min(cpu_min, start)
        cpu_max = max(cpu_max, end)
      elif "TPU" in dtype or "tpu" in dtype or "PrivateUse1" in dtype:
        tpu_min = min(tpu_min, start)
        tpu_max = max(tpu_max, end)
        tpu_starts.append(start)
      else:
        # Fallback based on name?
        pass

    print(f"CPU Times: {cpu_min} to {cpu_max}", flush=True)
    print(f"TPU Times: {tpu_min} to {tpu_max}", flush=True)
    print(f"Difference (TPU - CPU): {tpu_min - cpu_min}", flush=True)
    print(f"TPU Starts (first 5): {tpu_starts[:5]}", flush=True)


if __name__ == "__main__":
  absltest.main()
