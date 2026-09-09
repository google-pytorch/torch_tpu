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
from etils import epath
import torch
import torch._inductor.config as inductor_config
from torch_tpu._internal.utils import log_utils
from torch_tpu._internal.utils import utils
from examples import paths
from torch_tpu._internal.benchmarks import xprof_adapter
import transformers

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
_MAX_DECODE_STEPS = flags.DEFINE_integer(
    "max_decode_steps",
    10,
    "The maximum number of decoding steps.",
)
_MAX_BUFFER_SIZE = flags.DEFINE_integer(
    "max_buffer_size",
    None,
    "The maximum buffer size for KV cache. Used as upper bound for dynamo"
    " mark_dynamic.",
)
_HF_DATA_PATH = epath.DEFINE_path(
    "hf_data_path",
    pathlib.Path(paths.XM_HOME) / "weights" / "huggingface",
    "Path to HuggingFace artifacts (e.g. tokenizer config, weights).",
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


def _mark_dynamic_cache(
    cache: transformers.DynamicCache | None, min_val: int, max_val: int
) -> None:
  """Marks the key and value tensors within a KV cache as dynamic.

  This function uses `torch._dynamo.mark_dynamic` to inform torch.compile
  that the second dimension (sequence length) of the key and value tensors
  in each layer of the `cache` can vary between `min_val` and `max_val`.

  Args:
    cache: The past_key_values cache object, typically from HuggingFace
      transformers.
    min_val: The minimum expected size for the dynamic dimension.
    max_val: The maximum expected size for the dynamic dimension.
  """
  if cache is None:
    return
  # pylint: disable=protected-access
  for layer in cache.layers:
    if hasattr(layer, "keys") and layer.keys is not None:
      torch._dynamo.mark_dynamic(layer.keys, 2, min=min_val, max=max_val)
    if hasattr(layer, "values") and layer.values is not None:
      torch._dynamo.mark_dynamic(layer.values, 2, min=min_val, max=max_val)
  # pylint: enable=protected-access


def model_generate(
    model: transformers.PreTrainedModel,
    initial_inputs: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizer,
    max_decode_steps: int,
    prefix: str = "",
    use_static_cache: bool = False,
    mark_dynamic: bool = False,
    max_buffer_size: int | None = None,
) -> tuple[str, Mapping[str, Any]]:
  """Generates text using a model with dynamic KV caching.

  Args:
    model: The language model to use for generation.
    initial_inputs: The initial input tensor for the prefill stage.
    tokenizer: The tokenizer to decode the generated token IDs.
    max_decode_steps: The maximum number of decoding steps to perform.
    prefix: A prefix string to prepend to log messages.
    use_static_cache: If True, uses pre-allocated StaticCache to prevent shape
      recompilation.
    mark_dynamic: If True, uses torch._dynamo.mark_dynamic to specify dynamic
      shapes for the KV cache.
    max_buffer_size: The maximum buffer size for KV cache. Required if
      use_static_cache or mark_dynamic is True.

  Returns:
    A tuple of (decoded text, latency metrics dictionary).

  Raises:
    ValueError: If `max_buffer_size` is None when `use_static_cache` or
      `mark_dynamic` is True.
  """
  batch_size, seq_len = initial_inputs.shape[:2]

  if (use_static_cache or mark_dynamic) and max_buffer_size is None:
    raise ValueError("max_buffer_size must be provided for compiled modes")

  with torch.inference_mode():
    past_key_values = None
    if use_static_cache:
      past_key_values = transformers.StaticCache(
          config=model.config,
          max_cache_len=max_buffer_size + 1,
      )
      num_heads = getattr(
          model.config, "num_key_value_heads", model.config.num_attention_heads
      )
      head_dim = getattr(
          model.config,
          "head_dim",
          model.config.hidden_size // model.config.num_attention_heads,
      )
      past_key_values.early_initialization(
          batch_size=batch_size,
          num_heads=num_heads,
          head_dim=head_dim,
          dtype=next(model.parameters()).dtype,
          device=initial_inputs.device,
      )

    with xprof_adapter.TraceMe(f"[{prefix}] Prefill"):
      prefill_start_time = time.time()
      output = model(
          input_ids=initial_inputs,
          past_key_values=past_key_values,
          use_cache=True,
      )
      logits = output.logits
      past_key_values = output.past_key_values
      if torch.accelerator.is_available():
        torch.accelerator.synchronize()
      prefill_end_time = time.time()
    logits = logits.to("cpu")

    prefill_time_ms = (prefill_end_time - prefill_start_time) * 1000
    print(f"{prefix} Prefill time: {prefill_time_ms:.2f} ms")
    output_tokens = initial_inputs.to("cpu")
    next_token = None
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
      if mark_dynamic and past_key_values is not None:
        min_val = seq_len
        max_val = max_buffer_size
        _mark_dynamic_cache(past_key_values, min_val, max_val)
      with xprof_adapter.TraceMe(f"[{prefix}] Decode step {i + 1}"):
        step_start_time = time.time()
        output = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = output.logits
        if torch.accelerator.is_available():
          torch.accelerator.synchronize()
        step_end_time = time.time()

      logits = logits.to("cpu")
      step_time_ms = (step_end_time - step_start_time) * 1000
      decode_step_times_ms.append(step_time_ms)
      past_key_values = output.past_key_values
      steps_completed += 1

    warmup = _WARMUP_STEPS.value
    avg_decode_time_no_warmup_ms = None
    if steps_completed > warmup:
      non_warmup_times = decode_step_times_ms[warmup:]
      avg_decode_time_no_warmup_ms = sum(non_warmup_times) / len(
          non_warmup_times
      )

    avg_decode_time_ms = None
    if steps_completed > 0:
      avg_decode_time_ms = sum(decode_step_times_ms) / len(decode_step_times_ms)

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
    options: The compiler options dict (optional).
  """

  backend: Any
  prefix: str
  key: str
  options: dict[str, Any] | None = None


def _get_compiled_config(device: Device, mode: Mode) -> CompiledConfig:
  """Gets the backend, options, prefix, and key for torch.compile based on device and mode."""
  if mode == Mode.COMPILED_DYNAMIC:
    if device == Device.CUDA:
      return CompiledConfig(
          backend="inductor",
          prefix="[Compiled CUDA] ",
          key="Compiled CUDA",
      )
    return CompiledConfig(
        backend="tpu",
        prefix="[Compiled TPU (Dynamic)] ",
        key="Compiled TPU (Dynamic)",
        options={"bounded_dynamism": True},
    )

  if mode == Mode.COMPILED_STATIC:
    if device == Device.CUDA:
      raise ValueError("Static compilation not supported for CUDA")
    return CompiledConfig(
        backend="tpu",
        prefix="[Compiled TPU (Static)] ",
        key="Compiled TPU (Static)",
        options={"bounded_dynamism": False},
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
    max_buffer_size: int | None = None,
) -> tuple[str, Mapping[str, Any]]:
  """Runs benchmark for a single mode with actual weights.

  Args:
    model_cpu: The HuggingFace model loaded on CPU.
    tokenizer: The tokenizer.
    inputs: The initial input tensor.
    max_decode_steps: The maximum number of decoding steps.
    device: The device to run on.
    mode: The execution mode.
    max_buffer_size: The maximum buffer size for KV cache, used when compiling.

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
          use_static_cache=False,
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
          use_static_cache=False,
      )
      logging.info("output_%s=%s", device.value, output_device)
      return output_device, {device.value.upper(): metrics_device}

    if mode == Mode.COMPILED_DYNAMIC:
      config = _get_compiled_config(device, mode)
      compile_kwargs = {"backend": config.backend}
      if config.options:
        compile_kwargs["options"] = config.options
      model_device_compiled = torch.compile(model_device, **compile_kwargs)

      output_device_compiled, metrics_compiled = model_generate(
          model_device_compiled,
          inputs.to(device.value),
          tokenizer,
          max_decode_steps,
          prefix=config.prefix,
          use_static_cache=False,
          mark_dynamic=True,
          max_buffer_size=max_buffer_size,
      )
      logging.info(
          "output_%s_compiled=%s", device.value, output_device_compiled
      )
      return output_device_compiled, {config.key: metrics_compiled}

    if mode == Mode.COMPILED_STATIC:
      config = _get_compiled_config(device, mode)
      compile_kwargs = {"backend": config.backend, "dynamic": False}
      if config.options:
        compile_kwargs["options"] = config.options
      model_device_compiled = torch.compile(model_device, **compile_kwargs)

      output_device_compiled, metrics_compiled = model_generate(
          model_device_compiled,
          inputs.to(device.value),
          tokenizer,
          max_decode_steps,
          prefix=config.prefix,
          use_static_cache=True,
          max_buffer_size=max_buffer_size,
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
    max_buffer_size: int | None = None,
) -> Mapping[str, Any]:
  """Runs benchmark for a single mode with random weights.

  Args:
    config: The HuggingFace model configuration.
    tokenizer: The tokenizer.
    inputs: The initial input tensor.
    max_decode_steps: The maximum number of decoding steps.
    device: The device to run on.
    mode: The execution mode.
    max_buffer_size: The maximum buffer size for KV cache, used when compiling.

  Returns:
    A dictionary of metrics for the executed mode.
  """
  with torch.device(device.value):
    model_device = transformers.AutoModelForCausalLM.from_config(config).to(
        torch.bfloat16
    )

  if mode == Mode.COMPILED_DYNAMIC:
    compiled_config = _get_compiled_config(device, mode)
    compile_kwargs = {"backend": compiled_config.backend}
    if compiled_config.options:
      compile_kwargs["options"] = compiled_config.options
    model_device_compiled = torch.compile(model_device, **compile_kwargs)

    _, metrics_compiled = model_generate(
        model_device_compiled,
        inputs.to(device.value),
        tokenizer,
        max_decode_steps,
        prefix=compiled_config.prefix,
        use_static_cache=False,
        mark_dynamic=True,
        max_buffer_size=max_buffer_size,
    )
    return {compiled_config.key: metrics_compiled}

  if mode == Mode.COMPILED_STATIC:
    compiled_config = _get_compiled_config(device, mode)
    compile_kwargs = {
        "backend": compiled_config.backend,
        "dynamic": False,
    }
    if compiled_config.options:
      compile_kwargs["options"] = compiled_config.options
    model_device_compiled = torch.compile(model_device, **compile_kwargs)

    _, metrics_compiled = model_generate(
        model_device_compiled,
        inputs.to(device.value),
        tokenizer,
        max_decode_steps,
        prefix=compiled_config.prefix,
        use_static_cache=True,
        max_buffer_size=max_buffer_size,
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
  tokenizer_path = _HF_DATA_PATH.value / "Qwen/Qwen3-0.6B"

  model_path = _HF_DATA_PATH.value / _MODEL_NAME.value

  tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

  # SINGLE CONFIG MODE (In-process)
  batch_size = _BATCH_SIZE.value
  prefill_seq_len = _PREFILL_SEQ_LEN.value

  if _USE_RANDOM_WEIGHTS.value:
    config = transformers.AutoConfig.from_pretrained(model_path)
    model_cpu = None
  else:
    model_cpu = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )

  if prefill_seq_len is not None:
    inputs = torch.randint(
        0,
        tokenizer.vocab_size,
        (batch_size, prefill_seq_len),
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

  prefill_len = inputs.shape[1]
  initial_max_decode_steps = _MAX_DECODE_STEPS.value
  explicit_max_buffer_size = _MAX_BUFFER_SIZE.value

  if explicit_max_buffer_size is not None:
    if prefill_len > explicit_max_buffer_size:
      raise ValueError(
          f"prefill_seq_len ({prefill_len}) exceeds max_buffer_size "
          f"({explicit_max_buffer_size})"
      )
    potential_max_decode_steps = explicit_max_buffer_size - prefill_len + 1
    if initial_max_decode_steps > potential_max_decode_steps:
      logging.warning(
          "Adjusting max_decode_steps to %d because prefill_seq_len (%d) + "
          "max_decode_steps exceeds max_buffer_size (%d)",
          potential_max_decode_steps,
          prefill_len,
          explicit_max_buffer_size,
      )
      max_decode_steps = potential_max_decode_steps
    else:
      max_decode_steps = initial_max_decode_steps
    max_buffer_size = explicit_max_buffer_size
  else:
    max_decode_steps = initial_max_decode_steps
    max_buffer_size = prefill_len + max_decode_steps

  logging.info(
      "Running benchmark for batch_size=%d, prefill_seq_len=%d,"
      " max_decode_steps=%d, max_buffer_size=%d",
      batch_size,
      inputs.shape[1],
      max_decode_steps,
      max_buffer_size,
  )
  all_metrics = {}

  if _USE_RANDOM_WEIGHTS.value:
    if device == Device.TPU:
      modes = [Mode.COMPILED_DYNAMIC, Mode.COMPILED_STATIC]
    else:
      modes = [Mode.COMPILED_DYNAMIC]

    if device == Device.TPU and not utils.libtpu_at_least((0, 0, 42)):
      logging.warning(
          "Skipping Mode.COMPILED_DYNAMIC: requires libtpu >= 0.0.42 with"
          " dynamic buffer size relaxation."
      )
      if Mode.COMPILED_DYNAMIC in modes:
        modes.remove(Mode.COMPILED_DYNAMIC)

    for mode in modes:
      metrics = _run_with_random_weights(
          config,
          tokenizer,
          inputs,
          max_decode_steps,
          device=device,
          mode=mode,
          max_buffer_size=max_buffer_size,
      )
      all_metrics.update(metrics)
  else:
    modes = (
        [Mode.CPU]
        + ([Mode.EAGER] if not _COMPILE_ONLY.value else [])
        + (
            [Mode.COMPILED_DYNAMIC, Mode.COMPILED_STATIC]
            if device == Device.TPU
            else [Mode.COMPILED_DYNAMIC]
        )
    )

    if device == Device.TPU and not utils.libtpu_at_least((0, 0, 42)):
      logging.warning(
          "Skipping Mode.COMPILED_DYNAMIC: requires libtpu >= 0.0.42 with"
          " dynamic buffer size relaxation."
      )
      if Mode.COMPILED_DYNAMIC in modes:
        modes.remove(Mode.COMPILED_DYNAMIC)

    output_cpu = None
    if Mode.CPU in modes:
      output_cpu, metrics = _run_with_actual_weights(
          model_cpu,
          tokenizer,
          inputs,
          max_decode_steps,
          device=device,
          mode=Mode.CPU,
          max_buffer_size=max_buffer_size,
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
          max_buffer_size=max_buffer_size,
      )
      if output_cpu is not None:
        assert output_cpu == output, f"{output_cpu=} != {output=}"
      all_metrics.update(metrics)

  print(
      f"\n=== Results for batch_size={batch_size},"
      f" prefill_seq_len={prefill_seq_len} ==="
  )
  _print_summary_table(all_metrics, max_decode_steps)


if __name__ == "__main__":
  app.run(main)
