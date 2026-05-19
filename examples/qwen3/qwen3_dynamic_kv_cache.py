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

"""Example use of Qwen3 model with KV Cache."""

from collections.abc import Mapping
import copy
import dataclasses
import enum
import pathlib
import time
from typing import Any

from absl import app
from absl import flags
from absl import logging
import torch
import torch._inductor.config as inductor_config
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.utils import log_utils
from examples import paths
import transformers

from torch_tpu._internal.shims.xprof import traceme

log_utils.log_to_stderr()

_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    "Qwen/Qwen3-0.6B",
    "Name of the model to run. For ex: 'Qwen/Qwen3-0.6B', 'Qwen/Qwen3-1.7B',"
    " 'Qwen/Qwen3-4B', 'Qwen/Qwen3-8B', or"
    " 'Qwen/Qwen3-Coder-30B-A3B-Instruct-MINI'.",
)

_DEVICE = flags.DEFINE_string(
    "device",
    "tpu",
    "Device to run the model on. Can be 'tpu' or 'cuda'.",
)

_COMPILE_ONLY = flags.DEFINE_boolean(
    "compile_only",
    True,
    "If True, only do compile mode execution, else also do eager mode"
    " execution.",
)
_USE_RANDOM_WEIGHTS = flags.DEFINE_boolean(
    "use_random_weights",
    False,
    "Whether to initialize model with random weights instead of loading from"
    " checkpoint.",
)
_PREFILL_SEQ_LEN = flags.DEFINE_integer(
    "prefill_seq_len",
    None,
    "If set, overrides the input text with a random tensor of this length (for"
    " benchmark).",
)
_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    16,
    "Batch size for the input tensor when prefill_seq_len is set.",
)
_WARMUP_STEPS = flags.DEFINE_integer(
    "warmup_steps",
    3,
    "Number of decode steps to skip for average latency calculation.",
)


@enum.unique
class Device(enum.Enum):
  """A supported device for running the model.

  Attributes:
    TPU: TPU device.
    CUDA: CUDA device.
  """

  TPU = "tpu"
  CUDA = "cuda"


@enum.unique
class Mode(enum.Enum):
  """An execution mode for running the model.

  Attributes:
    CPU: CPU mode.
    EAGER: Eager mode.
    COMPILED_DYNAMIC: Compiled dynamic mode.
    COMPILED_STATIC: Compiled static mode.
  """

  CPU = "cpu"
  EAGER = "eager"
  COMPILED_DYNAMIC = "compiled_dynamic"
  COMPILED_STATIC = "compiled_static"


def model_generate(
    model: transformers.PreTrainedModel,
    initial_inputs: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizer,
    max_decode_steps: int,
    prefix: str = "",
) -> tuple[str, Mapping[str, Any]]:
  """Generates text using a model with dynamic KV caching.

  Args:
    model: The language model to use for generation.
    initial_inputs: The initial input tensor for the prefill stage.
    tokenizer: The tokenizer to decode the generated token IDs.
    max_decode_steps: The maximum number of decoding steps to perform.
    prefix: A prefix string to prepend to log messages.

  Returns:
    A tuple of (decoded text, latency metrics dictionary).
  """
  with torch.inference_mode():
    with traceme.TraceMe(f"[{prefix}] Prefill"):
      start_time = time.time()
      output = model(input_ids=initial_inputs, use_cache=True)
      logits = output.logits.to("cpu")
      past_key_values = output.past_key_values
      end_time = time.time()

    prefill_time_ms = (end_time - start_time) * 1000
    print(f"{prefix} Prefill time: {prefill_time_ms:.2f} ms")
    output_tokens = initial_inputs.to("cpu")
    next_token = None
    start_time = time.time()
    steps_completed = 0
    decode_step_times_ms = []

    for i in range(max_decode_steps):
      if isinstance(next_token, torch.Tensor):
        if torch.eq(next_token, 0).all():
          break
      # greedy sampling
      next_token_cpu = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
      output_tokens = torch.cat([output_tokens, next_token_cpu], dim=1)

      next_token = next_token_cpu.to(model.device)
      with traceme.TraceMe(f"[{prefix}] Decode step {i + 1}"):
        prev_time = time.time()
        output = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = output.logits.to("cpu")
        current_time = time.time()

      step_time_ms = (current_time - prev_time) * 1000
      print(
          f"{prefix} Decode step {i + 1}:"
          f" {step_time_ms:.2f} ms,"
          f" device={model.device}"
      )
      decode_step_times_ms.append(step_time_ms)
      past_key_values = output.past_key_values
      steps_completed += 1

    end_time = time.time()
    decode_time = end_time - start_time

    warmup = _WARMUP_STEPS.value
    avg_decode_time_no_warmup_ms = None
    if steps_completed > warmup:
      non_warmup_times = decode_step_times_ms[warmup:]
      avg_decode_time_no_warmup_ms = sum(non_warmup_times) / len(
          non_warmup_times
      )
      print(
          f"{prefix} Average Decode time per token (excluding first {warmup}"
          f" steps): {avg_decode_time_no_warmup_ms:.2f} ms"
      )

    avg_decode_time_ms = (
        decode_time * 1000 / steps_completed if steps_completed > 0 else None
    )
    if steps_completed > 0:
      print(
          f"{prefix} Decode time per token (including warmup):"
          f" {avg_decode_time_ms:.2f} ms"
      )
    else:
      print(f"{prefix} No decode steps were run.")
  output = output_tokens
  output_list = output.tolist()
  output_text = tokenizer.decode(output_list[0], skip_special_tokens=True)
  metrics = {
      "prefill_time_ms": prefill_time_ms,
      "decode_step_times_ms": decode_step_times_ms,
      "avg_decode_time_per_token_ms": avg_decode_time_ms,
      "avg_decode_time_per_token_no_warmup_ms": avg_decode_time_no_warmup_ms,
  }
  return output_text, metrics


def _print_summary_table(
    all_metrics: Mapping[str, Mapping[str, Any]], max_decode_steps: int
) -> None:
  """Prints a summary table of latency metrics across different runs.

  Args:
    all_metrics: A dictionary where keys are strings representing different
      execution modes (e.g., "CPU", "TPU", "Compiled TPU"). Each value is
      another dictionary containing latency metrics for that mode, with the
      following expected keys:
      - "prefill_time_ms": The time taken for the prefill stage in milliseconds
        (float).
      - "decode_step_times_ms": A list of floats, where each element is the time
        taken for a single decode step in milliseconds.
      - "avg_decode_time_per_token_ms": The average time per token during
        decoding in milliseconds (float).
      - "avg_decode_time_per_token_no_warmup_ms": The average time per token
        during decoding, excluding warmup steps, in milliseconds (float).
    max_decode_steps: The maximum number of decode steps performed, used to
      structure the table rows.
  """  # fmt: skip
  headers = ["Metric"] + list(all_metrics.keys())
  table_data = []

  prefill_row = ["Prefill time (ms)"]
  for mode, metrics in all_metrics.items():
    prefill_row.append(metrics["prefill_time_ms"])
  table_data.append(prefill_row)

  # Rows for decode steps
  for i in range(max_decode_steps):
    step_row = [f"Decode step {i + 1} (ms)"]
    for mode, metrics in all_metrics.items():
      step_times = metrics["decode_step_times_ms"]
      if i < len(step_times):
        step_row.append(step_times[i])
      else:
        step_row.append(None)
    table_data.append(step_row)

  avg_row = ["Avg decode time / token (ms)"]
  for mode, metrics in all_metrics.items():
    avg_row.append(metrics["avg_decode_time_per_token_ms"])
  table_data.append(avg_row)

  # New Row: Avg decode time per token (no warmup)
  avg_row = ["Avg decode time / token (no warmup) (ms)"]
  for mode, metrics in all_metrics.items():
    avg_row.append(metrics["avg_decode_time_per_token_no_warmup_ms"])
  table_data.append(avg_row)

  print("\n=== Latency Summary Table ===")

  def format_cell(val):
    if val is None:
      return ""
    if isinstance(val, float):
      return f"{val:.2f}"
    return str(val)

  formatted_headers = [format_cell(h) for h in headers]
  formatted_rows = [[format_cell(cell) for cell in row] for row in table_data]

  col_widths = [len(h) for h in formatted_headers]
  for row in formatted_rows:
    for i, cell in enumerate(row):
      col_widths[i] = max(col_widths[i], len(cell))

  header_line = (
      "| "
      + " | ".join(h.ljust(w) for h, w in zip(formatted_headers, col_widths))
      + " |"
  )
  print(header_line)

  # Separator
  separator_parts = []
  for i, w in enumerate(col_widths):
    if i == 0:
      separator_parts.append(":" + "-" * (w - 1) if w > 0 else "")
    else:
      separator_parts.append("-" * (w - 1) + ":" if w > 0 else "")
  separator_line = "| " + " | ".join(separator_parts) + " |"
  print(separator_line)

  # Data rows
  for row in formatted_rows:
    row_cells = []
    for i, cell in enumerate(row):
      if i == 0:
        row_cells.append(cell.ljust(col_widths[i]))
      else:
        row_cells.append(cell.rjust(col_widths[i]))
    print("| " + " | ".join(row_cells) + " |")
  print()


@dataclasses.dataclass(frozen=True)
class CompiledConfig:
  """Configuration for torch.compile.

  Attributes:
    backend: The backend to use for compilation.
    prefix: The prefix for log messages.
    key: The key for storing metrics.
  """

  backend: Any
  prefix: str
  key: str


def _get_compiled_config(device: Device, mode: Mode) -> CompiledConfig:
  """Gets the backend, prefix, and key for torch.compile based on device and mode."""
  if mode == Mode.COMPILED_DYNAMIC:
    if device == Device.CUDA:
      return CompiledConfig(
          backend="inductor",
          prefix="[Compiled CUDA] ",
          key="Compiled CUDA",
      )
    return CompiledConfig(
        backend=_backend.TpuBackend(dynamism=True),
        prefix="[Compiled TPU (Dynamic)] ",
        key="Compiled TPU (Dynamic)",
    )

  if mode == Mode.COMPILED_STATIC:
    if device == Device.CUDA:
      raise ValueError("Static compilation not supported for CUDA")
    return CompiledConfig(
        backend=_backend.TpuBackend(dynamism=False),
        prefix="[Compiled TPU (Static)] ",
        key="Compiled TPU (Static)",
    )

  raise ValueError(f"Invalid compiled mode: {mode}")


def _run_with_actual_weights(
    model_cpu: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    inputs: torch.Tensor,
    max_decode_steps: int,
    *,
    device: Device,
    mode: Mode,
) -> tuple[str, Mapping[str, Any]]:
  """Runs benchmark for a single mode with actual weights.

  Args:
    model_cpu: The HuggingFace model loaded on CPU.
    tokenizer: The tokenizer.
    inputs: The initial input tensor.
    max_decode_steps: The maximum number of decoding steps.
    device: The device to run on.
    mode: The execution mode.

  Returns:
    A tuple (output_text, metrics), where output_text is the decoded output
    text, and metrics is a dictionary of metrics for the executed mode.
  """
  with torch.no_grad():
    if mode == Mode.CPU:
      result_output, metrics_cpu = model_generate(
          model_cpu,
          inputs,
          tokenizer,
          max_decode_steps,
          prefix="[CPU] ",
      )
      logging.info("output_cpu=%s", result_output)
      return result_output, {"CPU": metrics_cpu}

    model_device = copy.deepcopy(model_cpu).to(device.value)

    if mode == Mode.EAGER:
      output_device, metrics_device = model_generate(
          model_device,
          inputs.to(device.value),
          tokenizer,
          max_decode_steps,
          prefix=f"[{device.value.upper()}] ",
      )
      logging.info("output_%s=%s", device.value, output_device)
      return output_device, {device.value.upper(): metrics_device}

    if mode == Mode.COMPILED_DYNAMIC:
      config = _get_compiled_config(device, mode)
      model_device_compiled = torch.compile(
          model_device, backend=config.backend
      )

      output_device_compiled, metrics_compiled = model_generate(
          model_device_compiled,
          inputs.to(device.value),
          tokenizer,
          max_decode_steps,
          prefix=config.prefix,
      )
      logging.info(
          "output_%s_compiled=%s", device.value, output_device_compiled
      )
      return output_device_compiled, {config.key: metrics_compiled}

    if mode == Mode.COMPILED_STATIC:
      config = _get_compiled_config(device, mode)
      model_device_compiled = torch.compile(
          model_device, backend=config.backend, dynamic=False
      )

      output_device_compiled, metrics_compiled = model_generate(
          model_device_compiled,
          inputs.to(device.value),
          tokenizer,
          max_decode_steps,
          prefix=config.prefix,
      )
      logging.info(
          "output_%s_compiled_static=%s", device.value, output_device_compiled
      )
      return output_device_compiled, {config.key: metrics_compiled}

    raise ValueError(f"Invalid mode: {mode}")


def _run_with_random_weights(
    config: transformers.PretrainedConfig,
    tokenizer: transformers.PreTrainedTokenizer,
    inputs: torch.Tensor,
    max_decode_steps: int,
    *,
    device: Device,
    mode: Mode,
) -> Mapping[str, Any]:
  """Runs benchmark for a single mode with random weights.

  Args:
    config: The HuggingFace model configuration.
    tokenizer: The tokenizer.
    inputs: The initial input tensor.
    max_decode_steps: The maximum number of decoding steps.
    device: The device to run on.
    mode: The execution mode.

  Returns:
    A dictionary of metrics for the executed mode.
  """
  with torch.device(device.value):
    model_device = transformers.AutoModelForCausalLM.from_config(config).to(
        torch.bfloat16
    )

  if mode == Mode.COMPILED_DYNAMIC:
    compiled_config = _get_compiled_config(device, mode)
    model_device_compiled = torch.compile(
        model_device, backend=compiled_config.backend
    )

    _, metrics_compiled = model_generate(
        model_device_compiled,
        inputs.to(device.value),
        tokenizer,
        max_decode_steps,
        prefix=compiled_config.prefix,
    )
    return {compiled_config.key: metrics_compiled}

  if mode == Mode.COMPILED_STATIC:
    compiled_config = _get_compiled_config(device, mode)
    model_device_compiled = torch.compile(
        model_device, backend=compiled_config.backend, dynamic=False
    )

    _, metrics_compiled = model_generate(
        model_device_compiled,
        inputs.to(device.value),
        tokenizer,
        max_decode_steps,
        prefix=compiled_config.prefix,
    )
    return {compiled_config.key: metrics_compiled}

  raise ValueError(f"Invalid mode: {mode}")


# pylint: disable=unused-argument
def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  torch.manual_seed(123)

  device = Device(_DEVICE.value)

  if device == Device.CUDA:
    inductor_config.compile_threads = 1

  # All Qwen3 models (from 0.6B up to 30B MoE) share the exact same
  # tokenizer vocabulary. Also, not every model has its own separate
  # tokenizer files, hence using the base 0.6B model for tokenizer.
  tokenizer_path = (
      pathlib.Path(paths.XM_HOME)
      / "weights"
      / "huggingface"
      / "Qwen/Qwen3-0.6B"
  )

  model_path = (
      pathlib.Path(paths.XM_HOME)
      / "weights"
      / "huggingface"
      / _MODEL_NAME.value
  )

  tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

  if _PREFILL_SEQ_LEN.value is not None:
    inputs = torch.randint(
        0,
        tokenizer.vocab_size,
        (_BATCH_SIZE.value, _PREFILL_SEQ_LEN.value),
        dtype=torch.long,
    )
  else:
    text = "Who are you?"
    messages = [{"role": "user", "content": text}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).input_ids

  max_decode_steps = 10
  all_metrics = {}

  if _USE_RANDOM_WEIGHTS.value:
    config = transformers.AutoConfig.from_pretrained(model_path)
    config._attn_implementation = "eager"

    if device == Device.TPU:
      modes = [Mode.COMPILED_DYNAMIC, Mode.COMPILED_STATIC]
    else:
      modes = [Mode.COMPILED_DYNAMIC]

    for mode in modes:
      metrics = _run_with_random_weights(
          config,
          tokenizer,
          inputs,
          max_decode_steps,
          device=device,
          mode=mode,
      )
      all_metrics.update(metrics)
  else:
    model_cpu = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )

    modes = (
        [Mode.CPU]
        + ([Mode.EAGER] if not _COMPILE_ONLY.value else [])
        + (
            [Mode.COMPILED_DYNAMIC, Mode.COMPILED_STATIC]
            if device == Device.TPU
            else [Mode.COMPILED_DYNAMIC]
        )
    )

    output_cpu = None
    if Mode.CPU in modes:
      output_cpu, metrics = _run_with_actual_weights(
          model_cpu,
          tokenizer,
          inputs,
          max_decode_steps,
          device=device,
          mode=Mode.CPU,
      )
      all_metrics.update(metrics)

    for mode in modes:
      if mode == Mode.CPU:
        continue
      output, metrics = _run_with_actual_weights(
          model_cpu,
          tokenizer,
          inputs,
          max_decode_steps,
          device=device,
          mode=mode,
      )
      if output_cpu is not None:
        assert output_cpu == output, f"{output_cpu=} != {output=}"
      all_metrics.update(metrics)

  _print_summary_table(all_metrics, max_decode_steps)


if __name__ == "__main__":
  app.run(main)
