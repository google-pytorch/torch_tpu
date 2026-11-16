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

"""Tests for device_utils."""

import dataclasses
import json
import types
from unittest import mock

from absl.testing import absltest
import torch
from torch_tpu._internal.utils import test_utils
from torch_tpu._internal.benchmarks import xprof_adapter
from perflab import device_utils
from tests import oss_utils
from tests import seed_test_utils

_MB = 1024 * 1024
_PS_PER_S = 1000000000000


# Plain value types mirroring the XSpace proto fields that device_utils reads.
# Unlike MagicMock trees, reading any other field raises AttributeError.
@dataclasses.dataclass(frozen=True, slots=True)
class _XEvent:
  duration_ps: int
  offset_ps: int = 0
  metadata_id: int | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class _XLine:
  name: str
  events: list[_XEvent]
  timestamp_ns: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class _XEventMetadata:
  name: str
  display_name: str = ''


@dataclasses.dataclass(frozen=True, slots=True)
class _XPlane:
  name: str
  lines: list[_XLine]
  event_metadata: dict[int, _XEventMetadata] = dataclasses.field(
      default_factory=dict
  )


@dataclasses.dataclass(frozen=True, slots=True)
class _XSpace:
  planes: list[_XPlane]


def _make_plane(name, durations_ps, offsets_ps=None):
  if offsets_ps is None:
    offsets_ps = list(range(len(durations_ps)))
  events = [
      _XEvent(duration_ps=d, offset_ps=o)
      for o, d in zip(offsets_ps, durations_ps)
  ]
  return _XPlane(name=name, lines=[_XLine('XLA Modules:...', events)])


def _make_host_plane(step_windows_ps, timestamp_ns=100):
  """Returns a host plane with a `post_warmup_step` TraceMe per window."""
  events = [
      _XEvent(duration_ps=d, offset_ps=o, metadata_id=1)
      for o, d in step_windows_ps
  ]
  return _XPlane(
      name='/host:CPU',
      lines=[_XLine('Thread 1', events, timestamp_ns=timestamp_ns)],
      event_metadata={
          1: _XEventMetadata(
              name='post_warmup_step',
              display_name='post_warmup_step#step_num=0#',
          )
      },
  )


def _make_device_plane(name, events_ps, timestamp_ns=100):
  """Returns a device plane with XLA module events at (offset, duration)."""
  events = [_XEvent(duration_ps=d, offset_ps=o) for o, d in events_ps]
  return _XPlane(
      name=name,
      lines=[_XLine('XLA Modules:...', events, timestamp_ns=timestamp_ns)],
  )


def _xprof_client(xspaces_by_host=None, peak_bytes=None):
  """Returns an autospecced XProf client serving the given profile data."""
  client = mock.create_autospec(xprof_adapter.XprofAnalysisClient)
  xspaces_by_host = xspaces_by_host or {}
  client.get_hosts.return_value = list(xspaces_by_host)
  client.get_xspace.side_effect = lambda session_id, host: xspaces_by_host.get(
      host
  )
  if peak_bytes is not None:
    profile = {
        'memoryProfilePerAllocator': {
            '0': {
                'profileSummary': {'peakStats': {'peakBytesInUse': peak_bytes}}
            }
        }
    }
    client.get_profile_data.return_value = (
        'application/json',
        json.dumps(profile).encode('utf-8'),
    )
  else:
    client.get_profile_data.return_value = ('application/json', b'')
  return client


class _FakeCudaMemory:
  """Stateful stand-in for the torch.cuda.memory peak-tracking counters."""

  def __init__(self):
    self.allocated = 0
    self.peak = 0

  def allocate(self, num_bytes):
    self.allocated += num_bytes
    self.peak = max(self.peak, self.allocated)

  def free(self, num_bytes):
    self.allocated -= num_bytes

  def max_memory_allocated(self):
    return self.peak

  def reset_max_memory_allocated(self):
    self.peak = self.allocated


class _FakeJaxDevice:
  """Stateful stand-in for a jax.Device's memory statistics."""

  def __init__(self, stats):
    self._stats = stats

  def memory_stats(self):
    return None if self._stats is None else dict(self._stats)

  def reset_memory_stats(self):
    self._stats['peak_bytes_in_use'] = self._stats['bytes_in_use']


class DeviceUtilsTest(seed_test_utils.RepeatableTest):

  @oss_utils.skip_in_oss('gcontain compilation container is internal-only.')
  def test_get_peak_host_compilation_memory_mb(self):
    class SimpleModel(torch.nn.Module):

      def forward(self, x):
        return x + 1

    device = torch.device('tpu')
    model = SimpleModel().to(device)
    x = torch.randn(10, device=device)

    # We need to use torch.compile with the tpu backend
    compiled_model = torch.compile(model, backend='tpu')

    # Run it to trigger compilation
    _ = compiled_model(x)
    torch.accelerator.synchronize()

    peak_mem = device_utils.get_peak_host_compilation_memory_mb('tpu')
    self.assertIsNotNone(peak_mem)
    self.assertGreater(peak_mem, 0.0)

  def test_get_max_total_device_time_success(self):
    client = _xprof_client({
        'host1': _XSpace([_make_plane('/device:0', [1 * _PS_PER_S])]),
        'host2': _XSpace([_make_plane('/device:1', [2 * _PS_PER_S])]),
    })

    max_time = device_utils.get_max_total_device_time(
        session_id='session_123', client=client
    )

    # We expect max of 1s and 2s, which is 2s.
    self.assertEqual(max_time, 2.0)

  def test_get_max_total_device_time_sums_every_event(self):
    # A single step frequently executes more than one XLA module (forward,
    # backward, optimizer, ...), so the event count is not the step count and
    # events must never be truncated to a step count.
    durations_ps = [
        1 * _PS_PER_S,
        1 * _PS_PER_S,
        2 * _PS_PER_S,
        3 * _PS_PER_S,
    ]
    client = _xprof_client({
        'host1': _XSpace([
            _make_plane(
                '/device:0', durations_ps, offsets_ps=[1000, 2000, 3000, 4000]
            )
        ])
    })

    total_time = device_utils.get_max_total_device_time(
        session_id='session_123', client=client
    )
    self.assertEqual(total_time, 7.0)

  def test_get_total_device_time_from_xspace(self):
    xspace = _XSpace([_make_plane('/device:0', [1 * _PS_PER_S, 2 * _PS_PER_S])])
    self.assertEqual(
        device_utils.get_total_device_time_from_xspace(xspace), 3.0
    )

  def test_get_total_device_time_from_xspace_expected_steps_does_not_truncate(
      self,
  ):
    # A step routinely executes more than one XLA module. On TPU, torch_tpu
    # emits a tiny prologue module plus the main module per step, so 2 profiled
    # steps produce 4 events. Truncating to the step count would report only
    # the last 2 events and halve the measured device time.
    xspace = _XSpace([
        _make_plane(
            '/device:0',
            durations_ps=[
                1 * _PS_PER_S,  # step 1, module A
                1 * _PS_PER_S,  # step 1, module B
                2 * _PS_PER_S,  # step 2, module A
                3 * _PS_PER_S,  # step 2, module B
            ],
            offsets_ps=[1000, 2000, 3000, 4000],
        )
    ])
    self.assertEqual(
        device_utils.get_total_device_time_from_xspace(
            xspace, expected_steps=2
        ),
        7.0,
    )

  def test_get_total_device_time_from_xspace_warns_on_partial_step(self):
    # 3 events over 2 steps means the capture window does not line up with the
    # measured steps. The total is still reported in full, but a warning flags
    # the trace as suspect.
    xspace = _XSpace([
        _make_plane(
            '/device:0',
            durations_ps=[1 * _PS_PER_S, 2 * _PS_PER_S, 3 * _PS_PER_S],
        )
    ])
    with self.assertLogs(level='WARNING') as logs:
      total = device_utils.get_total_device_time_from_xspace(
          xspace, expected_steps=2
      )
    self.assertEqual(total, 6.0)
    self.assertRegex('\n'.join(logs.output), 'not a multiple')

  def test_get_total_device_time_from_xspace_takes_max_over_local_devices(self):
    xspace = _XSpace([
        _make_plane('/device:0', [1 * _PS_PER_S]),
        _make_plane('/device:1', [4 * _PS_PER_S]),
    ])
    self.assertEqual(
        device_utils.get_total_device_time_from_xspace(xspace), 4.0
    )

  def test_get_total_device_time_from_xspace_ignores_non_device_planes(self):
    host_plane = _make_plane('/host:CPU', [9 * _PS_PER_S])
    device_plane = _make_plane('/device:0', [1 * _PS_PER_S])
    xspace = _XSpace([host_plane, device_plane])
    self.assertEqual(
        device_utils.get_total_device_time_from_xspace(xspace), 1.0
    )

  def test_get_total_device_time_from_xspace_none(self):
    self.assertEqual(device_utils.get_total_device_time_from_xspace(None), -1.0)

  def test_get_total_device_time_from_xspace_no_xla_modules(self):
    plane = _XPlane(name='/device:0', lines=[_XLine(name='Steps', events=[])])
    self.assertEqual(
        device_utils.get_total_device_time_from_xspace(_XSpace([plane])),
        -1.0,
    )

  def test_get_max_total_device_time_missing_args(self):
    self.assertEqual(
        device_utils.get_max_total_device_time(session_id=None, client=None),
        -1.0,
    )
    self.assertEqual(
        device_utils.get_max_total_device_time(
            session_id='session_123', client=None
        ),
        -1.0,
    )
    self.assertEqual(
        device_utils.get_max_total_device_time(
            session_id=None, client=_xprof_client()
        ),
        -1.0,
    )

  def test_get_max_total_device_time_no_hosts(self):
    self.assertEqual(
        device_utils.get_max_total_device_time(
            session_id='session_123', client=_xprof_client()
        ),
        -1.0,
    )

  def test_get_max_total_device_time_no_devices(self):
    client = _xprof_client(
        {'host1': _XSpace([_XPlane(name='/cpu:0', lines=[])])}
    )
    self.assertEqual(
        device_utils.get_max_total_device_time(
            session_id='session_123', client=client
        ),
        -1.0,
    )

  def test_tpu_peak_memory_reset_and_query(self):
    device = torch.device('tpu')
    device_utils.reset_peak_memory_stats('tpu')

    initial_peak = device_utils.get_peak_memory_hbm('tpu')
    self.assertGreaterEqual(initial_peak, 0.0)

    # Allocate 16MB on TPU (4M float32 elements = 16MB)
    size_in_bytes = 16 * 1024 * 1024
    num_elements = size_in_bytes // 4
    t = torch.randn(num_elements, device=device)
    torch.accelerator.synchronize()

    peak_after_alloc = device_utils.get_peak_memory_hbm('tpu')
    self.assertGreaterEqual(peak_after_alloc, 16.0)

    # Free tensor and verify that before reset, the high watermark remains
    del t
    torch.accelerator.synchronize()

    peak_before_reset = device_utils.get_peak_memory_hbm('tpu')
    self.assertEqual(peak_before_reset, peak_after_alloc)

    # Reset peak memory stats and verify that peak memory is cleared
    device_utils.reset_peak_memory_stats('tpu')
    peak_after_reset = device_utils.get_peak_memory_hbm('tpu')
    self.assertLess(peak_after_reset, peak_after_alloc)

  def test_reset_peak_memory_stats_unsupported_device(self):
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=benchmark
      device_utils.reset_peak_memory_stats('unsupported_device')

  def test_reset_peak_memory_stats_cpu(self):
    # CPU should be a safe no-op
    device_utils.reset_peak_memory_stats('cpu')

  def test_get_peak_memory_hbm_zero_bytes(self):
    with mock.patch.object(
        torch.accelerator, 'max_memory_allocated', return_value=0
    ):
      # Verifies that 0 bytes allocated is accurately reported as 0.0 MB rather
      # than falling back to XProf (-1.0 MB).
      self.assertEqual(device_utils.get_peak_memory_hbm('tpu'), 0.0)

  def test_get_peak_memory_hbm_tpu_fallback_when_accelerator_raises(self):
    client = _xprof_client(peak_bytes=96 * _MB)
    with mock.patch.object(
        torch.accelerator,
        'max_memory_allocated',
        side_effect=RuntimeError('Accelerator query failed'),
    ):
      result = device_utils.get_peak_memory_hbm(
          'tpu', session_id='s1', xprof_client=client
      )
    self.assertEqual(result, 96.0)

  def test_get_peak_memory_hbm_tpu_fallback_without_accelerator_api(self):
    client = _xprof_client(peak_bytes=99 * _MB)
    with mock.patch.object(torch, 'accelerator', create=True, spec=[]):
      result = device_utils.get_peak_memory_hbm(
          'tpu', session_id='s1', xprof_client=client
      )
    self.assertEqual(result, 99.0)

  def test_get_peak_memory_hbm_tpu_fallback_without_session(self):
    with mock.patch.object(torch, 'accelerator', create=True, spec=[]):
      self.assertEqual(device_utils.get_peak_memory_hbm('tpu'), -1.0)

  def test_get_peak_memory_hbm_tpu_fallback_empty_profile(self):
    with mock.patch.object(torch, 'accelerator', create=True, spec=[]):
      result = device_utils.get_peak_memory_hbm(
          'tpu', session_id='s1', xprof_client=_xprof_client()
      )
    self.assertEqual(result, -1.0)

  def test_get_peak_memory_hbm_cpu(self):
    mem = device_utils.get_peak_memory_hbm('cpu')
    self.assertGreater(mem, 0.0)

  def test_get_peak_memory_hbm_jax(self):
    client = _xprof_client(peak_bytes=42 * _MB)

    def peak_mb(device, **kwargs):
      with mock.patch('jax.local_devices', return_value=[device]):
        return device_utils.get_peak_memory_hbm('jax', **kwargs)

    # Devices without memory stats fall back to the XProf memory profile.
    self.assertEqual(
        peak_mb(_FakeJaxDevice(None), session_id='s1', xprof_client=client),
        42.0,
    )
    self.assertEqual(
        peak_mb(_FakeJaxDevice({'peak_bytes_in_use': 256 * _MB})), 256.0
    )
    # Compiled XLA executable stack reservations count towards the peak.
    self.assertEqual(
        peak_mb(
            _FakeJaxDevice({
                'peak_bytes_in_use': 256 * _MB,
                'peak_bytes_reserved': 512 * _MB,
            })
        ),
        768.0,
    )
    # peak_allocated_bytes takes precedence when reported.
    self.assertEqual(
        peak_mb(
            _FakeJaxDevice({
                'peak_allocated_bytes': 1024 * _MB,
                'peak_bytes_in_use': 256 * _MB,
                'peak_bytes_reserved': 512 * _MB,
            })
        ),
        1024.0,
    )
    self.assertEqual(
        peak_mb(
            _FakeJaxDevice(
                {'bytes_in_use': 128 * _MB, 'bytes_reserved': 64 * _MB}
            )
        ),
        192.0,
    )
    # A negative peak is invalid, so bytes_in_use is used instead.
    self.assertEqual(
        peak_mb(
            _FakeJaxDevice({'peak_bytes_in_use': -1, 'bytes_in_use': 8 * _MB})
        ),
        8.0,
    )
    self.assertEqual(
        peak_mb(
            _FakeJaxDevice({'peak_bytes_in_use': -1}),
            session_id='s1',
            xprof_client=client,
        ),
        42.0,
    )
    with mock.patch(
        'jax.local_devices', side_effect=RuntimeError('JAX unavailable')
    ):
      self.assertEqual(
          device_utils.get_peak_memory_hbm(
              'jax', session_id='s1', xprof_client=client
          ),
          42.0,
      )

  def test_get_peak_memory_hbm_unsupported_device(self):
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=benchmark
      device_utils.get_peak_memory_hbm('unsupported_device')

  def test_cuda_peak_memory_reset_and_query(self):
    # There is no CUDA device under test, so the CUDA allocator counters are
    # replaced by a stateful fake.
    fake = _FakeCudaMemory()
    with mock.patch.object(
        torch.cuda.memory, 'max_memory_allocated', fake.max_memory_allocated
    ), mock.patch.object(
        torch.cuda.memory,
        'reset_max_memory_allocated',
        fake.reset_max_memory_allocated,
    ):
      fake.allocate(64 * _MB)
      fake.free(48 * _MB)
      self.assertEqual(device_utils.get_peak_memory_hbm('cuda'), 64.0)

      device_utils.reset_peak_memory_stats('cuda')
      self.assertEqual(device_utils.get_peak_memory_hbm('cuda'), 16.0)

  def test_jax_peak_memory_reset_and_query(self):
    devices = [
        _FakeJaxDevice(
            {'peak_bytes_in_use': 256 * _MB, 'bytes_in_use': 32 * _MB}
        ),
        _FakeJaxDevice(
            {'peak_bytes_in_use': 512 * _MB, 'bytes_in_use': 64 * _MB}
        ),
    ]
    with mock.patch('jax.local_devices', return_value=devices):
      self.assertEqual(device_utils.get_peak_memory_hbm('jax'), 256.0)

      device_utils.reset_peak_memory_stats('jax')

      self.assertEqual(device_utils.get_peak_memory_hbm('jax'), 32.0)
    # Every local device is reset, not just the one that is queried.
    self.assertEqual(devices[1].memory_stats()['peak_bytes_in_use'], 64 * _MB)

  def test_configure_inductor_for_benchmarks(self):
    from torch._inductor.codegen.triton import TritonOverrides  # pylint: disable=g-import-not-at-top

    fake_masking_utils = types.ModuleType('masking_utils')
    with mock.patch('importlib.import_module', return_value=fake_masking_utils):
      device_utils.configure_inductor_for_benchmarks()
    self.assertEqual(torch._inductor.config.compile_threads, 1)
    self.assertIn('tl.float32', TritonOverrides.isinf('x'))
    self.assertIn('tl.float32', TritonOverrides.isnan('x'))
    self.assertFalse(fake_masking_utils.is_tracing(torch.zeros(1)))

  def test_torch_compile_cpu_configures_inductor_and_forwards_fullgraph(self):
    # The test sandbox has no C++ compiler, so inductor codegen cannot run.
    # Dynamo rejects a graph break under fullgraph=True while tracing, before
    # the backend is invoked, which makes fullgraph observable without codegen.
    # TODO(wan): revisit DECISION (b/568593385): backend='inductor' itself is
    # not asserted, since that needs a working C++ toolchain under test.
    def add_one_with_graph_break(x):
      torch._dynamo.graph_break()
      return x + 1

    torch._dynamo.reset()
    with torch._inductor.config.patch(compile_threads=4):
      compiled = device_utils.torch_compile(
          add_one_with_graph_break, 'cpu', fullgraph=True
      )
      # Inductor is configured for benchmarking before compiling.
      self.assertEqual(torch._inductor.config.compile_threads, 1)
    self.assertIs(
        torch._dynamo.eval_frame.innermost_fn(compiled),
        add_one_with_graph_break,
    )
    with self.assertRaises(torch._dynamo.exc.Unsupported):
      compiled(torch.arange(4.0))

  def test_collect_tensors_on_keeps_only_tensors_on_the_device(self):
    tpu_a = torch.ones(2, device='tpu')
    tpu_b = torch.zeros(3, device='tpu')
    cpu = torch.ones(4)
    tree = {'a': tpu_a, 'nested': (cpu, [tpu_b, 3, None]), 's': 'text'}

    tpu_tensors = device_utils.collect_tensors_on('tpu', tree)
    cpu_tensors = device_utils.collect_tensors_on('cpu', tree)

    self.assertLen(tpu_tensors, 2)
    self.assertIs(tpu_tensors[0], tpu_a)
    self.assertIs(tpu_tensors[1], tpu_b)
    self.assertLen(cpu_tensors, 1)
    self.assertIs(cpu_tensors[0], cpu)

  def test_collect_tensors_on_returns_empty_without_matching_tensors(self):
    for tree in (None, 3, {'n': 3}, torch.ones(2)):
      with self.subTest(tree=tree):
        self.assertEmpty(device_utils.collect_tensors_on('tpu', tree))

  def test_collect_tensors_on_expands_sparse_tensors(self):
    sparse = torch.sparse_coo_tensor([[0, 2]], [1.0, 2.0], (3,))

    tensors = device_utils.collect_tensors_on('cpu', {'s': sparse, 'n': 3})

    self.assertLen(tensors, 2)
    test_utils.assert_close(tensors[0], torch.tensor([[0, 2]]))
    test_utils.assert_close(tensors[1], torch.tensor([1.0, 2.0]))

  def test_get_total_device_time_from_xspace_filters_post_warmup_step_windows(
      self,
  ):
    # Two post_warmup_step host TraceMes: [1000, 3500] and [6000, 8500].
    host_plane = _make_host_plane([(1000, 2500), (6000, 2500)])
    # Device plane has:
    #   - step 0 module (offset 1500, duration 1000 ps)
    #   - post_warmup_hook cache-reset module (offset 4500, duration 500 ps)
    #   - step 1 module (offset 6500, duration 2000 ps)
    dev_plane = _make_device_plane(
        '/device:TPU:0', [(1500, 1_000), (4500, 500), (6500, 2_000)]
    )

    total = device_utils.get_total_device_time_from_xspace(
        _XSpace(planes=[host_plane, dev_plane]), expected_steps=2
    )
    # Only step 0 (1000 ps) and step 1 (2000 ps) modules are summed; the
    # interstep hook module (500 ps) is excluded.
    self.assertAlmostEqual(total, 3_000 / 1e12, places=18)

  def test_get_total_device_time_from_xspace_excludes_non_overlapping_plane(
      self,
  ):
    host_plane = _make_host_plane([(1000, 2500)])
    # Active device plane has a 1000 ps module in step window [1000, 3500].
    active_plane = _make_device_plane('/device:TPU:0', [(1500, 1000)])
    # Auxiliary device plane has a longer 5000 ps module entirely outside the
    # step window. It must not fall back to all module time and be selected as
    # max device time.
    aux_plane = _make_device_plane('/device:TPU:1', [(5000, 5000)])

    total = device_utils.get_total_device_time_from_xspace(
        _XSpace(planes=[host_plane, active_plane, aux_plane])
    )
    self.assertAlmostEqual(total, 1_000 / 1e12, places=18)

    # When no device plane overlaps the step window,
    # get_total_device_time_from_xspace returns -1.0.
    self.assertEqual(
        device_utils.get_total_device_time_from_xspace(
            _XSpace(planes=[host_plane, aux_plane])
        ),
        -1.0,
    )


if __name__ == '__main__':
  absltest.main()
