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

"""Compilation benchmark for cummax/cumsum."""

import os
import re
import statistics
import sys
import tempfile
import time
from typing import Any

from absl import flags
from absl import logging
from absl.testing import absltest
from etils import epath
import tabulate
import torch
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.compile import tpu_torch_compile

_NUM_TRIALS = flags.DEFINE_integer(
    "num_trials", 5, "Number of trials for each measurement."
)

_LENGTHS = flags.DEFINE_list(
    "lengths",
    ["1", "10", "100", "1000", "10000", "100000"],
    "List of tensor lengths to benchmark.",
)

_DUMP_DIR = (
    epath.Path(os.environ.get("TEST_TMPDIR", tempfile.gettempdir()))
    / "llo_dump"
)

_USE_ASSOCIATIVE_SCAN = flags.DEFINE_bool(
    "use_associative_scan",
    False,
    "Whether to enable associative scan emitter.",
)


def _get_early_flag(name: str, default: Any) -> Any:
  """Extracts flag value from sys.argv before absl parsing."""
  for arg in sys.argv:
    if arg.startswith(f"--{name}="):
      val = arg.split("=", 1)[1]
      if isinstance(default, bool):
        return val.lower() in ("true", "1", "t", "y", "yes")
      return val
  return default


def _set_dump_env() -> None:
  """Configures XLA/TPU environment variables for IR dumping."""
  dump_flags = [
      f"--xla_jf_dump_to={_DUMP_DIR}",
      "--xla_jf_dump_llo_text=true",
      "--xla_jf_dump_hlo_text=true",
      "--xla_jf_debug_level=2",
  ]
  if _get_early_flag("use_associative_scan", False):
    dump_flags.append("--xla_tpu_use_associative_scan_emitter=true")

  os.environ["LIBTPU_INIT_ARGS"] = (
      os.environ.get("LIBTPU_INIT_ARGS", "") + " " + " ".join(dump_flags)
  )
  sys.argv.extend(dump_flags)

  xla_flags = [
      f"--xla_dump_to={_DUMP_DIR}",
      "--xla_dump_hlo_as_text",
  ]
  os.environ["XLA_FLAGS"] = (
      os.environ.get("XLA_FLAGS", "") + " " + " ".join(xla_flags)
  )


# Initialize environment immediately upon module load.
_set_dump_env()


class _XlaArtifacts:
  """Scans and parses XLA dump artifacts."""

  @staticmethod
  def get_snapshot() -> set[epath.Path]:
    """Returns a set of current file paths in the dump directory."""
    paths = set()
    if not _DUMP_DIR.exists():
      return paths

    for root, _, files in os.walk(_DUMP_DIR):
      for f in files:
        path = epath.Path(root) / f
        if path.suffix in {".txt", ".hlo", ".llo"}:
          paths.add(path)
    return paths

  def count_logical_instructions(
      self, path: epath.Path
  ) -> tuple[int, dict[str, int]]:
    """Parses IR file and counts logical instructions.

    Args:
      path: Path to the .llo.txt or .hlo.txt dump file.

    Returns:
      A tuple containing:
        - Total instruction count.
        - Dictionary mapping opcode names to their occurrence counts.
    """
    try:
      content = path.read_text()
    except (IOError, OSError):
      return 0, {}

    ops = {}
    total = 0

    # Words that match the opcode pattern but aren't instructions.
    exclude = {
        "hlo",
        "operation",
        "emitted",
        "module",
        "computation",
        "entry",
        "root",
        "func",
        "return",
        "primitive",
        "bundle",
        "to",
        "at",
        "pre",
        "post",
        "after",
        "before",
        "metadata",
        "fingerprint",
        "details",
        "all",
        "finalize",
    }

    for line in content.splitlines():
      line = line.strip()
      # Only process lines that start with [ (LLO bundled/unbundled) or % (HLO).
      if not (line.startswith("[") or line.startswith("%")):
        continue

      # Regex: Matches optional [id] or %name = prefix, then captures mnemonic.
      m = re.search(
          r"^(?:(?:\[\d+\]\s+)|(?:%\S+\s+=\s+(?:[a-z0-9<>\[\]]+\s+)?))?([a-z][a-z0-9._-]{1,30})[\(\s]",
          line,
      )
      if m:
        op = m.group(1).strip()
        if op not in exclude:
          total += 1
          ops[op] = ops.get(op, 0) + 1

    return total, ops

  def get_ir_metrics(self, baseline: set[epath.Path]) -> dict[str, Any]:
    """Analyzes new dump files to find peak and final instruction counts.

    Args:
      baseline: Snapshot of files before the compilation trial.

    Returns:
      Dictionary with statistics about the compiled artifacts.
    """
    new_files = self.get_snapshot() - baseline
    if not new_files:
      return {
          "peak_instr": 0,
          "peak_stage": "N/A",
          "final_instr": 0,
          "dump_mb": 0,
          "top_ops": {},
      }

    total_bytes = sum(f.stat().length for f in new_files if f.is_file())
    peak_count = 0
    peak_file = "N/A"
    peak_ops = {}
    final_count = 0

    for f in new_files:
      # Exclude non-instruction files.
      if not f.is_file() or any(
          x in f.name
          for x in (
              "tpu_comp_env",
              "flagfile",
              "config",
              "details",
              "fingerprints",
          )
      ):
        continue

      count, ops = self.count_logical_instructions(f)
      if count > peak_count:
        peak_count = count
        peak_file = f.name
        peak_ops = ops

      # Use the 'final-top-level-llo' stage for final instruction count.
      if "final-top-level-llo.txt" in f.name:
        final_count = count

    return {
        "peak_instr": peak_count,
        "peak_stage": peak_file,
        "final_instr": final_count,
        "dump_mb": total_bytes / (1024 * 1024),
        "top_ops": peak_ops,
    }

  def normalize_hlo(self, hlo: str) -> str:
    """Masks shape/metadata to allow structural comparison."""
    hlo = re.sub(r"tensor<[\d?x]+x", "tensor<?x", hlo)
    hlo = re.sub(r"tensor<[\d?x]+f", "tensor<?f", hlo)
    hlo = re.sub(r"tensor<[\d?x]+i", "tensor<?i", hlo)
    hlo = re.sub(r"module @tt_jit_[a-zA-Z0-9_]+", "module @masked", hlo)
    hlo = re.sub(r"fingerprint:[a-f0-9]+", "fingerprint:masked", hlo)
    hlo = re.sub(r"=\s*\[[\d\s,\[\]]*\]", "= [masked]", hlo)
    hlo = re.sub(r"\[\d+:\d+[^\]]*\]", "[masked]", hlo)
    hlo = re.sub(r"\{[^\}]+\}", "{masked}", hlo)
    hlo = re.sub(r"dense<[^>]+>", "dense<masked>", hlo)
    hlo = re.sub(r"#[a-zA-Z0-9_.]+", "#masked", hlo)
    return hlo


class CumsumCummaxBenchmarkTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    os.environ["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"] = "disabled"
    os.environ["TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT"] = ""
    tt_testing.reset_eager_state()
    _DUMP_DIR.mkdir(parents=True, exist_ok=True)
    self._artifacts = _XlaArtifacts()

  def _reset(self) -> None:
    torch.compiler.reset()
    torch.tpu._clear_cache()

  def _run_trial(
      self, op: Any, x: torch.Tensor, trial: int, baseline: set[epath.Path]
  ) -> dict[str, Any]:
    """Runs one measurement trial.

    Args:
      op: The torch operation to compile and run.
      x: Input tensor.
      trial: Current trial index.
      baseline: IR file snapshot before run.

    Returns:
      Dictionary of timings and IR metrics.
    """
    self._reset()
    backend = _backend.TpuBackend(debug=True)
    compiled = torch.compile(op, backend=backend)

    # First run (Cold): Compilation + Lowering.
    torch.accelerator.synchronize()
    start = time.perf_counter()
    result = None
    if op == torch.cumsum:
      result = compiled(x, dim=0)
    elif op == torch.cummax:
      result, _ = compiled(x, dim=0)
    torch.accelerator.synchronize()
    compile_and_run_time = time.perf_counter() - start

    metrics = {}
    if trial == 0:
      binary_size = 0
      hlo = ""
      if backend._compiled_executables:
        executable_metadata = backend._compiled_executables[0]
        hlo = executable_metadata.mlir_text
        try:
          binary_size = len(
              tpu_torch_compile.serialize_executable(
                  executable_metadata._executable
              )
          )
        except (AttributeError, RuntimeError):
          pass
      metrics = self._artifacts.get_ir_metrics(baseline)
      metrics["binary_size"] = binary_size
      metrics["hlo"] = hlo

    # Second run (Warm): Cached execution.
    start = time.perf_counter()
    if op == torch.cumsum:
      _ = compiled(x, dim=0)
    elif op == torch.cummax:
      _, _ = compiled(x, dim=0)
    torch.accelerator.synchronize()
    cached_run_time = time.perf_counter() - start

    return {
        "compile_and_run_time": compile_and_run_time,
        "cached_run_time": cached_run_time,
        "metrics": metrics,
        "tpu_output": result,
    }

  def test_scaling(self) -> None:
    results = []
    refs = {}

    for op_fn in [torch.cumsum, torch.cummax]:
      for length in [int(l) for l in _LENGTHS.value]:
        compile_run_times, cached_run_times = [], []
        ir_metrics = {}
        logging.info("Benchmarking %s(n=%d)", op_fn.__name__, length)

        x_cpu = torch.randn(length).float()
        x = x_cpu.to("tpu")

        # Golden reference on CPU.
        golden = None
        if op_fn == torch.cumsum:
          golden = torch.cumsum(x_cpu, dim=0)
        elif op_fn == torch.cummax:
          golden, _ = torch.cummax(x_cpu, dim=0)

        tpu_output = None
        for i in range(_NUM_TRIALS.value):
          baseline = self._artifacts.get_snapshot()
          run = self._run_trial(op_fn, x, i, baseline)
          compile_run_times.append(run["compile_and_run_time"])
          cached_run_times.append(run["cached_run_time"])
          if i == 0:
            ir_metrics = run["metrics"]
            tpu_output = run["tpu_output"].cpu()

        # HLO Match check.
        hlo_match = "N/A"
        if ir_metrics.get("hlo"):
          norm = self._artifacts.normalize_hlo(ir_metrics["hlo"])
          if op_fn.__name__ not in refs:
            refs[op_fn.__name__] = norm
            hlo_match = "Ref"
          else:
            hlo_match = "YES" if norm == refs[op_fn.__name__] else "NO"

        sorted_ops = sorted(
            ir_metrics.get("top_ops", {}).items(),
            key=lambda x: x[1],
            reverse=True,
        )
        ops_str = ", ".join([f"{k}({v})" for k, v in sorted_ops[:3]])

        # Numerical accuracy check.
        max_diff = (tpu_output - golden).abs().max().item()
        rel_diff = max_diff / (golden.abs().max().item() + 1e-8)

        results.append({
            "Op": op_fn.__name__,
            "N": length,
            "Median Compile+Run (s)": (
                f"{statistics.median(compile_run_times):.4f}"
            ),
            "Median Cached Run (s)": (
                f"{statistics.median(cached_run_times):.4f}"
            ),
            "HLO Match": hlo_match,
            "Peak Instr": f"{ir_metrics.get('peak_instr', 0):,}",
            "Final Instr": f"{ir_metrics.get('final_instr', 0):,}",
            "Binary (KB)": f"{ir_metrics.get('binary_size', 0) / 1024:.2f}",
            "Peak Stage": ir_metrics.get("peak_stage", "N/A"),
            "Max Diff": f"{max_diff:.2e}",
            "Rel Diff": f"{rel_diff:.2e}",
            "Dump (MB)": f"{ir_metrics.get('dump_mb', 0):.2f}",
            "Top Opcodes": ops_str,
        })

    print(f"\nResults (Trials={_NUM_TRIALS.value}):")
    print(
        tabulate.tabulate(results, headers="keys", tablefmt="github"),
        flush=True,
    )


if __name__ == "__main__":
  absltest.main()
