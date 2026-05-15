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

"""Microbenchmarks comparing PyPadViaHostDma/PySliceViaHostDma vs XLA compilation."""

import collections
import gc
import os
import queue
import resource
import time
from typing import Any

from absl import flags
from absl import logging
from absl.testing import absltest
import torch
from torch_tpu._internal import sync
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.utils import log_utils

from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

log_utils.log_to_stderr()

_NUM_WARMUP_STEPS = flags.DEFINE_integer(
    "num_warmup_steps", 2, "Number of warmup steps."
)
_NUM_STEPS = flags.DEFINE_integer("num_steps", 10, "Number of benchmark steps.")


def get_host_rss_mb():
  # ru_maxrss is in kilobytes on Linux
  return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def get_device_max_mem_mb():
  return torch.accelerator.max_memory_allocated() / (1024.0 * 1024.0)


def _run_xla_pad_worker(
    q: queue.Queue[dict[str, Any] | Exception],
    seq_len: int,
    padded_seq_len: int,
    num_heads: int,
    head_dim: int,
):
  try:
    dev_idx = os.environ.get("TPU_VISIBLE_DEVICES")
    if dev_idx:
      flags.FLAGS.deepsea_hal_included_devs = str(dev_idx)
    print(
        "Worker _run_xla_pad_worker started with"
        f" TPU_VISIBLE_DEVICES={os.environ.get('TPU_VISIBLE_DEVICES')},"
        f" deepsea_hal_included_devs={flags.FLAGS.deepsea_hal_included_devs}",
        flush=True,
    )
    torch.manual_seed(123)
    device = torch.device("tpu")
    x = torch.randn(
        1, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    sync.synchronize(x)

    tensor_info = [([1, seq_len, num_heads, head_dim], x.dtype)]
    bounds_list = [([1], [padded_seq_len])]

    for _ in range(_NUM_WARMUP_STEPS.value):
      mlir = tpu_torch_compile.get_pad_module_mlir(tensor_info, bounds_list)
      executable = tpu_torch_compile.compile_mlir(mlir)
      pad_outputs = tpu_torch_compile.execute(executable, [x])
      sync.synchronize(pad_outputs[0])

    gc.collect()
    start_time = time.time()
    for _ in range(_NUM_STEPS.value):
      mlir = tpu_torch_compile.get_pad_module_mlir(tensor_info, bounds_list)
      executable = tpu_torch_compile.compile_mlir(mlir)
      pad_outputs = tpu_torch_compile.execute(executable, [x])
      sync.synchronize(pad_outputs[0])
    xla_pad_latency = ((time.time() - start_time) / _NUM_STEPS.value) * 1000
    xla_host_rss_peak = get_host_rss_mb()
    xla_device_peak = get_device_max_mem_mb()

    transfer_size_bytes = x.numel() * x.element_size()
    padded_size_bytes = pad_outputs[0].numel() * pad_outputs[0].element_size()

    result = {
        "in_size_bytes": transfer_size_bytes,
        "out_size_bytes": padded_size_bytes,
        "latency_ms": xla_pad_latency,
        "host_rss_mb": xla_host_rss_peak,
        "tpu_peak_mb": xla_device_peak,
    }
    q.put(result)
  except Exception as e:  # pylint: disable=broad-exception-caught
    q.put(e)


def _run_dma_pad_worker(
    q: queue.Queue[dict[str, Any] | Exception],
    seq_len: int,
    padded_seq_len: int,
    num_heads: int,
    head_dim: int,
):
  try:
    dev_idx = os.environ.get("TPU_VISIBLE_DEVICES")
    if dev_idx:
      flags.FLAGS.deepsea_hal_included_devs = str(dev_idx)
    print(
        "Worker _run_dma_pad_worker started with"
        f" TPU_VISIBLE_DEVICES={os.environ.get('TPU_VISIBLE_DEVICES')},"
        f" deepsea_hal_included_devs={flags.FLAGS.deepsea_hal_included_devs}",
        flush=True,
    )
    torch.manual_seed(123)
    device = torch.device("tpu")
    x = torch.randn(
        1, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    sync.synchronize(x)

    padded_shape = [1, padded_seq_len, num_heads, head_dim]
    for _ in range(_NUM_WARMUP_STEPS.value):
      padded_tensor = tpu_torch_compile.pad_via_host_dma(x, padded_shape)
      sync.synchronize(padded_tensor)

    gc.collect()
    start_time = time.time()
    for _ in range(_NUM_STEPS.value):
      padded_tensor = tpu_torch_compile.pad_via_host_dma(x, padded_shape)
      sync.synchronize(padded_tensor)
    dma_pad_latency = ((time.time() - start_time) / _NUM_STEPS.value) * 1000
    dma_host_rss_peak = get_host_rss_mb()
    dma_device_peak = get_device_max_mem_mb()

    transfer_size_bytes = x.numel() * x.element_size()
    padded_size_bytes = padded_tensor.numel() * padded_tensor.element_size()

    result = {
        "in_size_bytes": transfer_size_bytes,
        "out_size_bytes": padded_size_bytes,
        "latency_ms": dma_pad_latency,
        "host_rss_mb": dma_host_rss_peak,
        "tpu_peak_mb": dma_device_peak,
    }
    q.put(result)
  except Exception as e:  # pylint: disable=broad-exception-caught
    q.put(e)


def _run_xla_slice_worker(
    q: queue.Queue[dict[str, Any] | Exception],
    seq_len: int,
    padded_seq_len: int,
    num_heads: int,
    head_dim: int,
):
  try:
    dev_idx = os.environ.get("TPU_VISIBLE_DEVICES")
    if dev_idx:
      flags.FLAGS.deepsea_hal_included_devs = str(dev_idx)
    print(
        "Worker _run_xla_slice_worker started with"
        f" TPU_VISIBLE_DEVICES={os.environ.get('TPU_VISIBLE_DEVICES')},"
        f" deepsea_hal_included_devs={flags.FLAGS.deepsea_hal_included_devs}",
        flush=True,
    )
    torch.manual_seed(123)
    device = torch.device("tpu")
    padded_tensor = torch.randn(
        1,
        padded_seq_len,
        num_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    sync.synchronize(padded_tensor)

    target_shape = [1, seq_len, num_heads, head_dim]
    padded_shape = [1, padded_seq_len, num_heads, head_dim]

    for _ in range(_NUM_WARMUP_STEPS.value):
      mlir = tpu_torch_compile.get_slice_module_mlir(
          [target_shape], [padded_shape], [padded_tensor.dtype]
      )
      executable = tpu_torch_compile.compile_mlir(mlir)
      slice_outputs = tpu_torch_compile.execute(executable, [padded_tensor])
      sync.synchronize(slice_outputs[0])

    gc.collect()
    start_time = time.time()
    for _ in range(_NUM_STEPS.value):
      mlir = tpu_torch_compile.get_slice_module_mlir(
          [target_shape], [padded_shape], [padded_tensor.dtype]
      )
      executable = tpu_torch_compile.compile_mlir(mlir)
      slice_outputs = tpu_torch_compile.execute(executable, [padded_tensor])
      sync.synchronize(slice_outputs[0])
    xla_slice_latency = ((time.time() - start_time) / _NUM_STEPS.value) * 1000
    xla_host_rss_peak = get_host_rss_mb()
    xla_device_peak = get_device_max_mem_mb()

    transfer_size_bytes = (
        slice_outputs[0].numel() * slice_outputs[0].element_size()
    )
    padded_size_bytes = padded_tensor.numel() * padded_tensor.element_size()

    result = {
        "in_size_bytes": padded_size_bytes,
        "out_size_bytes": transfer_size_bytes,
        "latency_ms": xla_slice_latency,
        "host_rss_mb": xla_host_rss_peak,
        "tpu_peak_mb": xla_device_peak,
    }
    q.put(result)
  except Exception as e:  # pylint: disable=broad-exception-caught
    q.put(e)


def _run_dma_slice_worker(
    q: queue.Queue[dict[str, Any] | Exception],
    seq_len: int,
    padded_seq_len: int,
    num_heads: int,
    head_dim: int,
):
  try:
    dev_idx = os.environ.get("TPU_VISIBLE_DEVICES")
    if dev_idx:
      flags.FLAGS.deepsea_hal_included_devs = str(dev_idx)
    print(
        "Worker _run_dma_slice_worker started with"
        f" TPU_VISIBLE_DEVICES={os.environ.get('TPU_VISIBLE_DEVICES')},"
        f" deepsea_hal_included_devs={flags.FLAGS.deepsea_hal_included_devs}",
        flush=True,
    )
    torch.manual_seed(123)
    device = torch.device("tpu")
    padded_tensor = torch.randn(
        1,
        padded_seq_len,
        num_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    sync.synchronize(padded_tensor)

    target_shape = [1, seq_len, num_heads, head_dim]

    for _ in range(_NUM_WARMUP_STEPS.value):
      sliced_tensor = tpu_torch_compile.slice_via_host_dma(
          padded_tensor, target_shape
      )
      sync.synchronize(sliced_tensor)

    gc.collect()
    start_time = time.time()
    for _ in range(_NUM_STEPS.value):
      sliced_tensor = tpu_torch_compile.slice_via_host_dma(
          padded_tensor, target_shape
      )
      sync.synchronize(sliced_tensor)
    dma_slice_latency = ((time.time() - start_time) / _NUM_STEPS.value) * 1000
    dma_host_rss_peak = get_host_rss_mb()
    dma_device_peak = get_device_max_mem_mb()

    transfer_size_bytes = sliced_tensor.numel() * sliced_tensor.element_size()
    padded_size_bytes = padded_tensor.numel() * padded_tensor.element_size()

    result = {
        "in_size_bytes": padded_size_bytes,
        "out_size_bytes": transfer_size_bytes,
        "latency_ms": dma_slice_latency,
        "host_rss_mb": dma_host_rss_peak,
        "tpu_peak_mb": dma_device_peak,
    }
    q.put(result)
  except Exception as e:  # pylint: disable=broad-exception-caught
    q.put(e)


class PadSliceMicrobenchmarkTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.results = []

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    if not cls.results:
      return

    headers = [
        "Op",
        "Shape",
        "In -> Out (KB)",
        "XLA (ms)",
        "XLA RSS (MB)",
        "XLA TPU (MB)",
        "DMA (ms)",
        "DMA RSS (MB)",
        "DMA TPU (MB)",
        "Speedup",
    ]

    row_format = (
        "{:<6} | {:<28} | {:<21} | {:>10} | {:>12} | {:>12} | {:>10} | {:>12} |"
        " {:>12} | {:>8}"
    )

    logging.info("=" * 157)
    logging.info(" " * 60 + "PAD / SLICE MICROBENCHMARK RESULTS")
    logging.info("=" * 157)
    logging.info(row_format.format(*headers))
    logging.info("-" * 157)

    for r in cls.results:
      if r["op"] == "PAD":
        shape_str = (
            f"[1, {r['seq_len']}->{r['padded_seq_len']}, {r['num_heads']},"
            f" {r['head_dim']}]"
        )
      else:
        shape_str = (
            f"[1, {r['padded_seq_len']}->{r['seq_len']}, {r['num_heads']},"
            f" {r['head_dim']}]"
        )

      in_kb = r["in_size_bytes"] / 1024.0
      out_kb = r["out_size_bytes"] / 1024.0
      size_str = f"{in_kb:.1f} -> {out_kb:.1f}"

      speedup_str = f"{r['speedup']:.1f}x"

      logging.info(
          row_format.format(
              r["op"],
              shape_str,
              size_str,
              f"{r['xla_latency_ms']:.2f}",
              f"{r['xla_host_rss_mb']:.1f}",
              f"{r['xla_tpu_peak_mb']:.1f}",
              f"{r['dma_latency_ms']:.2f}",
              f"{r['dma_host_rss_mb']:.1f}",
              f"{r['dma_tpu_peak_mb']:.1f}",
              speedup_str,
          )
      )
    logging.info("=" * 157)

  def test_microbenchmarks(self):
    shapes = [
        (12, 128, 2, 128),
        (128, 512, 2, 128),
        (512, 1024, 2, 128),
        (1024, 2048, 2, 128),
        (2048, 4096, 2, 128),
        (4096, 8192, 2, 128),
        # Extreme shapes for large models (>100B parameters)
        # Llama-3-405B / Qwen-110B GQA (16 KV heads, 16K context)
        (8192, 16384, 16, 128),
        # Falcon-180B MQA (8 KV heads, 64K context)
        (32768, 65536, 8, 128),
        # 100B+ MHA (64 KV heads, 16K context)
        (8192, 16384, 64, 128),
        # Llama-3-405B 128K extreme context window (16 KV heads, 128K context)
        (65536, 131072, 16, 128),
    ]

    # We have 40 tasks: 10 shapes * 2 ops * 2 backends
    # Task format: (func, args, op, shape)
    tasks = []
    for shape in shapes:
      seq_len, padded_seq_len, num_heads, head_dim = shape
      tasks.append((
          _run_xla_pad_worker,
          (seq_len, padded_seq_len, num_heads, head_dim),
          "PAD",
          shape,
      ))
      tasks.append((
          _run_dma_pad_worker,
          (seq_len, padded_seq_len, num_heads, head_dim),
          "PAD",
          shape,
      ))
      tasks.append((
          _run_xla_slice_worker,
          (seq_len, padded_seq_len, num_heads, head_dim),
          "SLICE",
          shape,
      ))
      tasks.append((
          _run_dma_slice_worker,
          (seq_len, padded_seq_len, num_heads, head_dim),
          "SLICE",
          shape,
      ))

    parent_visible_chips = os.environ.get("TPU_VISIBLE_CHIPS")
    parent_visible_devices = os.environ.get("TPU_VISIBLE_DEVICES")
    logging.info(
        "Parent TPU_VISIBLE_CHIPS: %s, TPU_VISIBLE_DEVICES: %s",
        parent_visible_chips,
        parent_visible_devices,
    )
    parent_visible = parent_visible_chips or parent_visible_devices
    if parent_visible:
      available_devices = [d.strip() for d in parent_visible.split(",")]
    else:
      available_devices = [str(i) for i in range(8)]
    active_processes = []

    # worker_results: maps (op, shape) -> { "xla": result, "dma": result }
    worker_results = collections.defaultdict(dict)

    # Track tasks that failed
    failures = []

    ctx = g3_multiprocessing.get_context(g3_multiprocessing.ABSL_SPAWN)

    while tasks or active_processes:
      # Spawn new processes if we have available devices and tasks
      while tasks and available_devices:
        dev_idx = available_devices.pop(0)
        func, args, op, shape = tasks.pop(0)

        # Set env vars in parent before spawn
        os.environ["TPU_VISIBLE_DEVICES"] = str(dev_idx)
        os.environ["TPU_VISIBLE_CHIPS"] = str(dev_idx)

        q = ctx.Queue()
        p = ctx.Process(
            target=func,
            args=(q, *args),
        )
        p.start()

        # Clean up parent env immediately after start
        if "TPU_VISIBLE_DEVICES" in os.environ:
          del os.environ["TPU_VISIBLE_DEVICES"]
        if "TPU_VISIBLE_CHIPS" in os.environ:
          del os.environ["TPU_VISIBLE_CHIPS"]

        active_processes.append((p, q, dev_idx, func, op, shape))
        logging.info(
            "Started %s worker for %s on TPU %s (PID %d)",
            func.__name__,
            shape,
            dev_idx,
            p.pid,
        )

      # Check for finished processes
      still_active = []
      for p, q, dev_idx, func, op, shape in active_processes:
        if not p.is_alive():
          p.join()
          available_devices.append(dev_idx)  # Free the device immediately

          if p.exitcode != 0:
            msg = (
                f"Worker {func.__name__} for {op} {shape} failed with exitcode"
                f" {p.exitcode}"
            )
            logging.error(msg)
            failures.append(msg)
            continue

          try:
            result = q.get_nowait()
            if isinstance(result, Exception):
              msg = (
                  f"Worker {func.__name__} for {op} {shape} raised exception:"
                  f" {result}"
              )
              logging.error(msg)
              failures.append(msg)
            else:
              backend = "xla" if "xla" in func.__name__ else "dma"
              worker_results[(op, shape)][backend] = result
          except queue.Empty:
            msg = (
                f"Worker {func.__name__} for {op} {shape} finished but queue"
                " was empty"
            )
            logging.error(msg)
            failures.append(msg)
        else:
          still_active.append((p, q, dev_idx, func, op, shape))
      active_processes = still_active
      time.sleep(0.1)

    if failures:
      self.fail("\n".join(failures))

    # Combine results
    for (op, shape), backends in worker_results.items():
      xla_res = backends.get("xla")
      dma_res = backends.get("dma")
      if not xla_res or not dma_res:
        logging.warning("Missing results for %s %s", op, shape)
        continue

      seq_len, padded_seq_len, num_heads, head_dim = shape
      speedup = (
          xla_res["latency_ms"] / dma_res["latency_ms"]
          if dma_res["latency_ms"] > 0
          else 0
      )

      combined_result = {
          "op": op,
          "seq_len": seq_len,
          "padded_seq_len": padded_seq_len,
          "num_heads": num_heads,
          "head_dim": head_dim,
          "in_size_bytes": xla_res["in_size_bytes"],
          "out_size_bytes": xla_res["out_size_bytes"],
          "xla_latency_ms": xla_res["latency_ms"],
          "xla_host_rss_mb": xla_res["host_rss_mb"],
          "xla_tpu_peak_mb": xla_res["tpu_peak_mb"],
          "dma_latency_ms": dma_res["latency_ms"],
          "dma_host_rss_mb": dma_res["host_rss_mb"],
          "dma_tpu_peak_mb": dma_res["tpu_peak_mb"],
          "speedup": speedup,
      }
      self.results.append(combined_result)


if __name__ == "__main__":
  g3_multiprocessing.handle_test_main(absltest.main)
