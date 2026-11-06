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
from torch_tpu._internal import execution_mode
from torch_tpu._internal import sync as tpu_sync
from torch_tpu._internal.profiler import named_scope
from torch_tpu._internal.profiler.profiler_config import TpuProfilerConfig
from tests import seed_test_utils

# pylint: disable=g-direct-tensorflow-import
from tsl.profiler.protobuf import profiler_options_pb2
from tsl.profiler.protobuf import xplane_pb2

# pylint: enable=g-direct-tensorflow-import

# Set TMPDIR to TEST_TMPDIR so TorchTPU's profiler writes artifacts into the
# test's temporary directory instead of falling back to system /tmp.
if "TEST_TMPDIR" in os.environ:
  os.environ["TMPDIR"] = os.environ["TEST_TMPDIR"]


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


class SimpleAttention(torch.nn.Module):

  def __init__(self, d_model: int):
    super().__init__()
    self.q_proj = torch.nn.Linear(d_model, d_model, bias=False)
    self.k_proj = torch.nn.Linear(d_model, d_model, bias=False)
    self.v_proj = torch.nn.Linear(d_model, d_model, bias=False)
    self.out_proj = torch.nn.Linear(d_model, d_model, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)
    scores = torch.matmul(q, k.transpose(-2, -1))
    probs = torch.softmax(scores, dim=-1)
    return self.out_proj(torch.matmul(probs, v))


class SimpleMLP(torch.nn.Module):

  def __init__(self, d_model: int):
    super().__init__()
    self.fc1 = torch.nn.Linear(d_model, d_model * 4, bias=False)
    self.act = torch.nn.GELU()
    self.fc2 = torch.nn.Linear(d_model * 4, d_model, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.fc2(self.act(self.fc1(x)))


class SimpleTransformerBlock(torch.nn.Module):

  def __init__(self, d_model: int):
    super().__init__()
    self.norm1 = torch.nn.LayerNorm(d_model)
    self.attn = SimpleAttention(d_model)
    self.norm2 = torch.nn.LayerNorm(d_model)
    self.mlp = SimpleMLP(d_model)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.attn(self.norm1(x))
    return x + self.mlp(self.norm2(x))


class SimpleTransformerModel(torch.nn.Module):

  def __init__(self, d_model: int, num_layers: int):
    super().__init__()
    self.layers = torch.nn.ModuleList(
        [SimpleTransformerBlock(d_model) for _ in range(num_layers)]
    )
    self.head = torch.nn.Linear(d_model, d_model, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for layer in self.layers:
      x = layer(x)
    return self.head(x)


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


def _device_tf_ops(xspace: xplane_pb2.XSpace) -> set[str]:
  """Returns the `tf_op` stat (Framework Name Scope) of TPU device events.

  On TPU the stat is attached to the event metadata of each HLO op (shared by
  all occurrences of the op), so both the per-event and the per-metadata stats
  are inspected.
  """
  tf_ops = set()
  for plane in xspace.planes:
    if "TPU" not in plane.name or plane.name.startswith("/host"):
      continue
    tf_op_ids = {
        stat_id
        for stat_id, metadata in plane.stat_metadata.items()
        if metadata.name == "tf_op"
    }
    if not tf_op_ids:
      continue

    def stat_value(stat):
      if stat.HasField("ref_value"):
        return plane.stat_metadata[stat.ref_value].name
      return stat.str_value

    for metadata in plane.event_metadata.values():
      for stat in metadata.stats:
        if stat.metadata_id in tf_op_ids:
          tf_ops.add(stat_value(stat))
    for line in plane.lines:
      for event in line.events:
        for stat in event.stats:
          if stat.metadata_id in tf_op_ids:
            tf_ops.add(stat_value(stat))
  return tf_ops


def _describe_xspace(xspace: xplane_pb2.XSpace) -> str:
  """Summarizes plane names and their stat names, for assertion messages."""
  parts = []
  for plane in xspace.planes:
    stat_names = sorted({m.name for m in plane.stat_metadata.values()})
    parts.append(f"{plane.name}: {stat_names[:40]}")
  return "\n".join(parts)


class ProfilerIntegrationTest(seed_test_utils.RepeatableTest):

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
      _ = a @ b
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
    cats = {e.get("cat") for e in events if "cat" in e}

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

  def _count_tpu_events(self, profile_dir: pathlib.Path) -> int:
    xplane_files = list(profile_dir.glob("**/*.xplane.pb"))
    if not xplane_files:
      return 0
    xspace = xplane_pb2.XSpace()
    xspace.ParseFromString(xplane_files[0].read_bytes())
    count = 0
    for plane in xspace.planes:
      if plane.name.startswith("/device:TPU"):
        for line in plane.lines:
          count += len(line.events)
    return count

  def test_native_profiler_without_explicit_sync(self):
    """Tests that profiling without script sync captures all TPU events."""
    device = torch.device("tpu")
    a = torch.ones((16, 16), device=device)
    b = torch.ones((16, 16), device=device)

    # Warmup using native PyTorch accelerator synchronization
    for _ in range(3):
      _ = a @ b
    torch.accelerator.synchronize()

    base_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if base_dir is None:
      base_dir = self.create_tempdir("trace_auto_sync_tpu").full_path

    dir_no_sync = os.path.join(base_dir, "no_sync")
    dir_with_sync = os.path.join(base_dir, "with_sync")

    _cleanup_profile_dir()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.TPU,  # type: ignore
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_no_sync),
        record_shapes=True,
    ) as prof_no_sync:
      for _ in range(5):
        _ = a @ b
      prof_no_sync.step()
    events_no_sync = self._count_tpu_events(_get_profile_dir())

    _cleanup_profile_dir()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.TPU,  # type: ignore
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_with_sync),
        record_shapes=True,
    ) as prof_with_sync:
      for _ in range(5):
        _ = a @ b
        torch.accelerator.synchronize()
      prof_with_sync.step()
    events_with_sync = self._count_tpu_events(_get_profile_dir())

    self.assertGreater(
        events_no_sync,
        0,
        msg=(
            "Expected TPU trace events without explicit script sync after our"
            " change"
        ),
    )
    self.assertEqual(
        events_no_sync,
        events_with_sync,
        msg="Expected identical TPU event count with and without script sync",
    )

    # Check key_averages time attribution without script sync
    key_averages = prof_no_sync.key_averages()
    self.assertIn("TPU time", key_averages.table())
    has_tpu_time = any(
        getattr(evt, "device_time_total", 0) > 0 for evt in key_averages
    )
    self.assertTrue(has_tpu_time, "Expected non-zero TPU time without sync")

  def test_async_workload_captures_all_in_flight_events(self):
    """Verifies that TpuKinetoProfilerSession::stop() automatically synchronizes

    addressable TPU devices and captures trace events for asynchronous TPU
    workloads that are actively executing in flight when prof.step() is called,
    without requiring explicit user script synchronization calls.
    """
    device = torch.device("tpu")
    a = torch.ones((4096, 4096), device=device)
    b = torch.ones((4096, 4096), device=device)

    # Warmup so kernel is compiled
    _ = a @ b
    torch.accelerator.synchronize()

    _cleanup_profile_dir()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.TPU,  # type: ignore
        ],
        record_shapes=True,
    ) as prof:
      # Launch async TPU workload in a background thread so operations
      # are actively in flight when prof.step() is called on main thread.
      ready_event = threading.Event()

      def _async_workload():
        for i in range(500):
          _ = a @ b
          if i == 150:
            ready_event.set()

      worker = threading.Thread(target=_async_workload)
      worker.start()
      ready_event.wait(timeout=10.0)
      prof.step()
      worker.join()

    events_captured = self._count_tpu_events(_get_profile_dir())
    logging.info(
        "TPU events captured for in-flight asynchronous workload: %d",
        events_captured,
    )

    self.assertGreater(
        events_captured,
        100,
        msg=(
            "Expected TPU trace events to be captured automatically when"
            " stopping the profiler during an active asynchronous workload,"
            " without explicit user script synchronization"
        ),
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
      _ = a @ b
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
      _ = compiled_func(a, b)
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

  def test_tpu_profiler_with_stack(self):
    device = torch.device("tpu")
    output_dir = self.create_tempdir("with_stack").full_path

    _cleanup_profile_dir()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.TPU,  # type: ignore
        ],
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    ) as prof:
      a = torch.ones((16, 16)).to(device)
      b = torch.ones((16, 16)).to(device)
      _ = a @ b
      prof.step()

    tpu_xplane_path = pathlib.Path(output_dir) / "xplane.pb"
    self._get_and_copy_xplane(tpu_xplane_path)

    xspace = xplane_pb2.XSpace.FromString(tpu_xplane_path.read_bytes())
    options_bytes = _get_profiler_options_bytes(xspace)
    options = profiler_options_pb2.ProfileOptions.FromString(options_bytes)

    with self.subTest(msg="ProfileOptions tracer level check"):
      self.assertEqual(
          options.python_tracer_level,
          1,
          "expected python_tracer_level=1 when with_stack=True, got"
          f" {options.python_tracer_level}",
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
      _ = a @ b
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
        with_stack=True,
        profile_memory=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    ):
      with profiler.record_function("full_pipeline"):
        with profiler.record_function("model1_inference_tpu"):
          output1_tpu = model_tpu(inputs)

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

  def test_compiled_module_name_scopes(self):
    """Checks that the module hierarchy of a compiled model reaches the trace."""
    device = torch.device("tpu")
    d_model = 256
    model = SimpleTransformerModel(d_model, num_layers=2).to(device)
    compiled_model = torch.compile(model, backend="tpu")
    x = torch.randn((4, 128, d_model), device=device, requires_grad=True)

    def train_step():
      loss = compiled_model(x).sum()
      loss.backward()
      tpu_sync.synchronize(loss)
      torch.accelerator.synchronize()

    # Compile both the initial gradient creation and the gradient accumulation
    # before tracing.
    for _ in range(2):
      train_step()

    # Keep the trace in the test outputs, from where it can be opened in XProf.
    undeclared_outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if undeclared_outputs_dir:
      output_dir = pathlib.Path(undeclared_outputs_dir) / "module_name_scopes"
      output_dir.mkdir(parents=True, exist_ok=True)
    else:
      output_dir = pathlib.Path(self.create_tempdir().full_path)

    _cleanup_profile_dir()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.TPU,  # type: ignore
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    ) as prof:
      for _ in range(2):
        train_step()
        prof.step()

    self._get_and_copy_xplane(output_dir / "xplane.pb")

    # The module scopes reach the trace through the HLO op metadata.
    content = (output_dir / "xplane.pb").read_bytes()
    for scope in (
        b"layers.0[SimpleTransformerBlock]",
        b"layers.1[SimpleTransformerBlock]",
        b"attn[SimpleAttention]",
        b"mlp[SimpleMLP]",
        b"q_proj[Linear]",
    ):
      self.assertIn(scope, content)

  def test_named_scope_in_xprof(self):
    """Checks that `named_scope` annotations reach the XProf trace."""

    @named_scope.named_scope("rope")
    def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
      return torch.matmul(x, freqs)

    class ScopedAttention(torch.nn.Module):

      def __init__(self, d_model: int):
        super().__init__()
        self.q_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.k_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.v_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("freqs", torch.eye(d_model))

      def forward(self, x: torch.Tensor) -> torch.Tensor:
        with named_scope.named_scope("qkv_proj"):
          q = apply_rope(self.q_proj(x), self.freqs)
          k = apply_rope(self.k_proj(x), self.freqs)
          v = self.v_proj(x)
        with named_scope.named_scope("sdpa"):
          scores = torch.matmul(q, k.transpose(-2, -1))
          probs = torch.softmax(scores, dim=-1)
          attn_out = torch.matmul(probs, v)
        return self.out_proj(attn_out)

    class ScopedBlock(torch.nn.Module):

      def __init__(self, d_model: int):
        super().__init__()
        self.attn = ScopedAttention(d_model)

      def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x)

    device = torch.device("tpu")
    d_model = 192
    model = ScopedBlock(d_model).to(device)
    compiled_model = torch.compile(model, backend="tpu")
    x = torch.randn((4, 64, d_model), device=device, requires_grad=True)

    def train_step():
      out = compiled_model(x)
      # Default eager mode (DEFER_NEVER): each op is compiled on its own.
      with named_scope.named_scope("eager_loss"):
        loss = (out * out).sum()
      loss.backward()
      tpu_sync.synchronize(loss)
      # DEFER_AND_FUSE: the scoped ops are fused into one executable.
      with execution_mode.set_eager_mode(
          execution_mode.EagerMode.DEFER_AND_FUSE
      ):
        with named_scope.named_scope("eager_fused"):
          fused = (out.detach() + 1) * 2
        tpu_sync.synchronize(fused)
      torch.accelerator.synchronize()

    # Compile both the initial gradient creation and the gradient accumulation
    # before tracing.
    for _ in range(2):
      train_step()

    output_dir = pathlib.Path(self.create_tempdir("named_scopes").full_path)

    _cleanup_profile_dir()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.TPU,  # type: ignore
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    ) as prof:
      for _ in range(2):
        train_step()
        prof.step()

    self._get_and_copy_xplane(output_dir / "xplane.pb")

    # Verify the outer-to-inner `/`-joined scope hierarchy in the `tf_op` stat
    # (XProf's Framework Name Scope, stored as `<op_name>:<op_type>`) of the
    # TPU device events, not merely in the HLO module proto that the XPlane
    # also carries.
    xplane_bytes = (output_dir / "xplane.pb").read_bytes()
    xspace = xplane_pb2.XSpace()
    xspace.ParseFromString(xplane_bytes)
    tf_ops = _device_tf_ops(xspace)
    self.assertNotEmpty(tf_ops, _describe_xspace(xspace))
    self.assertContainsSubset(
        (
            "attn[ScopedAttention]/qkv_proj/q_proj[Linear]/mm/mm:",
            "attn[ScopedAttention]/qkv_proj/k_proj[Linear]/mm/mm:",
            "attn[ScopedAttention]/qkv_proj/v_proj[Linear]/mm/mm:",
            "attn[ScopedAttention]/qkv_proj/rope/mm/mm:",
            "attn[ScopedAttention]/sdpa/bmm/bmm:",
            "attn[ScopedAttention]/sdpa/softmax/softmax:",
            "attn[ScopedAttention]/out_proj[Linear]/mm/mm:",
            # Default eager mode (DEFER_NEVER).
            "eager_loss/mul/mul:",
            "eager_loss/sum/sum:",
            # DEFER_AND_FUSE: the fusion carries the scope of its root op.
            "eager_fused/mul/mul:",
        ),
        tf_ops,
    )
    # Backward ops keep their scopes in the HLO metadata, but XLA may fuse them
    # into a fusion whose device event reports the root op's `tf_op` only, so
    # the backward softmax is checked in the embedded HLO module instead.
    # (`named_scope_test` asserts the backward StableHLO before fusion.)
    self.assertIn(
        b"attn[ScopedAttention]/sdpa/softmax_backward_data/softmax_backward_data",
        xplane_bytes,
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
      _ = a @ b

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
      _ = a @ b

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
      _ = a @ b

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
        _ = torch.ones((16, 16)).to(device) @ torch.ones((16, 16)).to(device)

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
      _ = a @ b

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
        _ = a @ b

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
      _ = a @ b

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

  @parameterized.parameters(
      ("3", "_3.xplane.pb"),
      ("worker_A", "_worker_A.xplane.pb"),
  )
  def test_tpu_profiler_config_worker_rank(self, worker_rank, expected_suffix):
    device = torch.device("tpu")
    custom_run_dir = self.create_tempdir("custom_run_dir_worker_rank").full_path

    _cleanup_profile_dir()

    config = TpuProfilerConfig(
        run_dir=pathlib.Path(custom_run_dir), worker_rank=worker_rank
    )

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
    xplane_files = list(timestamp_dir.glob(f"*{expected_suffix}"))
    self.assertNotEmpty(
        xplane_files, f"Should have a {expected_suffix} file in custom run_dir"
    )

  def test_tpu_profiler_config_worker_rank_validation(self):
    # Valid values should not raise errors:
    _ = TpuProfilerConfig(worker_rank="3")
    _ = TpuProfilerConfig(worker_rank="worker_A")
    _ = TpuProfilerConfig(worker_rank=None)

    # Invalid values should raise TypeError or ValueError:
    with self.assertRaises(TypeError):
      TpuProfilerConfig(worker_rank=0)
    with self.assertRaises(TypeError):
      TpuProfilerConfig(worker_rank=42)
    with self.assertRaises(TypeError):
      TpuProfilerConfig(worker_rank=-1)
    with self.assertRaises(TypeError):
      TpuProfilerConfig(worker_rank=3.5)
    with self.assertRaises(TypeError):
      TpuProfilerConfig(worker_rank=True)
    with self.assertRaises(TypeError):
      TpuProfilerConfig(worker_rank={"rank": 1})
    with self.assertRaises(ValueError):
      TpuProfilerConfig(worker_rank="worker:1")
    with self.assertRaises(ValueError):
      TpuProfilerConfig(worker_rank="worker,1")


if __name__ == "__main__":
  absltest.main()
