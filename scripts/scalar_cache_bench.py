"""Benchmark the eager scalar device-buffer cache (torch_tpu #6036/#6037).

Run against an installed torch_tpu wheel:  PYTHONPATH=. python scalar_cache_bench.py LABEL
Prints per-op dispatch latency for cache hits, a stream of distinct scalars (misses), a cycle
larger than the cache, a fused-region scenario that reuses a hot scalar around an eviction
(compile count), and an AdamW + cosine-LR loop (compile count, step time, live allocations).
"""
import gc
import sys
import time

import torch
import torch_tpu  # noqa: F401  (registers the device)
from torch_tpu._internal import execution_mode

EagerMode = execution_mode.EagerMode
LABEL = sys.argv[1] if len(sys.argv) > 1 else "wheel"
DEV = torch.device("tpu")
MODES = [EagerMode.DEFER_NEVER, EagerMode.DEFER_AND_FUSE]


def live():
  torch.accelerator.synchronize()
  gc.collect()
  return torch.accelerator.memory_stats()["allocation.all.current"]


def compiles():
  return torch.tpu._get_cache_stats().num_compilations  # pylint: disable=protected-access


def timed_loop(mode, fn, n, label, warmup=64):
  with execution_mode.set_eager_mode(mode):
    x = torch.zeros(8, 128, device=DEV)
    for i in range(warmup):
      fn(x, -1 - i)
    torch.accelerator.synchronize()
    base_live, base_comp = live(), compiles()
    lat = []
    t0 = time.perf_counter()
    for i in range(n):
      s = time.perf_counter()
      fn(x, i)
      lat.append(time.perf_counter() - s)
    torch.accelerator.synchronize()
    total = time.perf_counter() - t0
    grew, comp = live() - base_live, compiles() - base_comp
  lat.sort()
  print(
      f"[{LABEL}] {label:34s} {mode.name:14s} {total * 1e6 / n:8.1f} us/op"
      f"  dispatch p50 {lat[n // 2] * 1e6:7.1f}  p99 {lat[int(n * 0.99)] * 1e6:7.1f}"
      f"  max {lat[-1] * 1e6:8.1f} us  live {grew:+d}  compiles {comp:+d}",
      flush=True,
  )


def fused_hot_scalar_step(x, i):
  # One fused region per step in DEFER_AND_FUSE: 0.9 is used before and after
  # the new per-step scalar, which can trigger a cache eviction in between.
  y = x.mul(0.9)
  y = y.add(10.0 + i / 4096)
  y = y.mul(0.9)
  x.copy_(y - y)  # keep x at zero; ends the region at synchronize()
  torch.accelerator.synchronize()


def adamw_cosine(mode, steps=1500):
  torch.manual_seed(0)
  with execution_mode.set_eager_mode(mode):
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 128), torch.nn.Tanh(), torch.nn.Linear(128, 8)
    ).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    inp = torch.randn(32, 64, device=DEV)
    tgt = torch.randn(32, 8, device=DEV)

    def step():
      opt.zero_grad(set_to_none=True)
      loss = torch.nn.functional.mse_loss(model(inp), tgt)
      loss.backward()
      opt.step()
      sched.step()

    for _ in range(20):
      step()
    torch.accelerator.synchronize()
    base_live, base_comp = live(), compiles()
    t0 = time.perf_counter()
    for _ in range(steps):
      step()
    torch.accelerator.synchronize()
    total = time.perf_counter() - t0
    print(
        f"[{LABEL}] {'adamw+cosine ' + str(steps) + ' steps':34s} {mode.name:14s}"
        f" {total * 1e3 / steps:8.2f} ms/step  live {live() - base_live:+d}"
        f"  compiles {compiles() - base_comp:+d}",
        flush=True,
    )


def main():
  print(f"[{LABEL}] torch {torch.__version__} torch_tpu {getattr(torch_tpu, '__version__', '?')}", flush=True)
  for mode in MODES:
    timed_loop(mode, lambda x, i: x.add_(0.5), 4000, "hit: same scalar")
    timed_loop(mode, lambda x, i: x.add_(1.0 + i / 4096), 4000, "miss: distinct scalar stream")
    timed_loop(mode, lambda x, i: x.add_(1.0 + (i % 1500) / 4096), 6000, "cycle of 1500 distinct scalars")
    timed_loop(mode, lambda x, i: x.add_(1.0 + (i % 512) / 4096), 6000, "cycle of 512 distinct scalars")
  timed_loop(EagerMode.DEFER_AND_FUSE, fused_hot_scalar_step, 3000, "fused region, hot 0.9 + new scalar")
  for mode in MODES:
    adamw_cosine(mode)
  print(f"[{LABEL}] done", flush=True)


if __name__ == "__main__":
  main()
