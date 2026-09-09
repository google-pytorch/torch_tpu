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

"""Example use of Qwen3-VL model with KV Cache."""

from collections.abc import Mapping
import copy
import dataclasses
import enum
import os
import pathlib
import time
import traceback
from typing import Any

from absl import app
from absl import flags
from absl import logging
from PIL import Image
import torch
import torch._inductor.config as inductor_config
from torch_tpu._internal import dynamism
from torch_tpu._internal import sync as tpu_sync
from torch_tpu._internal.utils import log_utils
from examples import paths
from torch_tpu._internal.benchmarks import xprof_adapter
import transformers
from transformers import Qwen3VLForConditionalGeneration

from rules_python.python.runfiles import runfiles

log_utils.log_to_stderr()


def patched_deepstack_process(
    self, hidden_states, visual_pos_masks, visual_embeds
):
  """Uses masked_scatter to inject visual features without data-dependent nonzero."""
  if visual_embeds is None or visual_pos_masks is None:
    return hidden_states
  zeros = torch.zeros_like(hidden_states)
  visual_features = zeros.masked_scatter(
      visual_pos_masks.unsqueeze(-1), visual_embeds
  )
  return hidden_states + visual_features


from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

Qwen3VLTextModel._deepstack_process = patched_deepstack_process


_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    "Qwen/Qwen3-VL-2B-Instruct",
    "Name of the model to run. Currently only 'Qwen/Qwen3-VL-2B-Instruct'"
    " config is supported locally.",
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
    True,
    "Whether to initialize model with random weights instead of loading from"
    " checkpoint.",
)
_PREFILL_SEQ_LEN = flags.DEFINE_integer(
    "prefill_seq_len",
    None,
    "If set, overrides the input text length (for benchmark).",
)
_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    1,
    "Batch size for the input tensor.",
)
_WARMUP_STEPS = flags.DEFINE_integer(
    "warmup_steps",
    2,
    "Number of decode steps to skip for average latency calculation.",
)
_PRECOMPILE_STEPS = flags.DEFINE_integer(
    "precompile_steps",
    0,
    "Number of steps to precompile dynamic adapters for.",
)
_MAX_DECODE_STEPS = flags.DEFINE_integer(
    "max_decode_steps",
    4,
    "The maximum number of decoding steps.",
)
_MAX_BUFFER_SIZE = flags.DEFINE_integer(
    "max_buffer_size",
    None,
    "The maximum buffer size for KV cache. Used as upper bound for dynamo"
    " mark_dynamic.",
)
_VERIFY_NUMERICAL_CORRECTNESS = flags.DEFINE_boolean(
    "verify_numerical_correctness",
    True,
    "Whether to run CPU reference execution and verify numerical output"
    " equivalence between CPU and TPU with identical random weights.",
)


@enum.unique
class Device(enum.Enum):
  TPU = "tpu"
  CUDA = "cuda"


@enum.unique
class Mode(enum.Enum):
  CPU = "cpu"
  EAGER = "eager"
  COMPILED_DYNAMIC = "compiled_dynamic"
  COMPILED_STATIC = "compiled_static"


def _mark_dynamic_cache(
    cache: transformers.DynamicCache | None, min_val: int, max_val: int
) -> None:
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
    initial_inputs: dict[str, torch.Tensor],
    processor: transformers.AutoProcessor,
    max_decode_steps: int,
    prefix: str = "",
    use_static_cache: bool = False,
    mark_dynamic: bool = False,
    max_buffer_size: int | None = None,
) -> tuple[str, Mapping[str, Any]]:
  """Generates text using a model with dynamic KV caching."""
  input_ids = initial_inputs["input_ids"]
  batch_size, seq_len = input_ids.shape[:2]

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
          model.config.text_config,
          "num_key_value_heads",
          model.config.text_config.num_attention_heads,
      )
      head_dim = getattr(
          model.config.text_config,
          "head_dim",
          model.config.text_config.hidden_size
          // model.config.text_config.num_attention_heads,
      )
      past_key_values.early_initialization(
          batch_size=batch_size,
          num_heads=num_heads,
          head_dim=head_dim,
          dtype=next(model.parameters()).dtype,
          device=input_ids.device,
      )

    with xprof_adapter.TraceMe(f"[{prefix}] Prefill"):
      prefill_start_time = time.time()
      output = model(
          **initial_inputs,
          past_key_values=past_key_values,
          use_cache=True,
      )
      logits = output.logits
      past_key_values = output.past_key_values
      tpu_sync.synchronize(logits, wait=True)
      prefill_end_time = time.time()
    logits = logits.to("cpu")

    prefill_time_ms = (prefill_end_time - prefill_start_time) * 1000
    print(f"{prefix} Prefill time: {prefill_time_ms:.2f} ms")

    if mark_dynamic and past_key_values is not None:
      min_val = 1
      max_val = max_buffer_size
      _mark_dynamic_cache(past_key_values, min_val, max_val)

    output_tokens = input_ids.to("cpu")
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

      decode_inputs = {
          "input_ids": next_token,
          "past_key_values": past_key_values,
          "use_cache": True,
      }

      try:
        with xprof_adapter.TraceMe(f"[{prefix}] Decode step {i + 1}"):
          step_start_time = time.time()
          output = model(**decode_inputs)
          logits = output.logits
          tpu_sync.synchronize(logits, wait=True)
          step_end_time = time.time()
      except Exception as e:
        print(f"=== DECODE STEP {i+1} EXCEPTION: {e} ===", flush=True)
        if hasattr(e, "__cause__") and e.__cause__ is not None:
          print(f"=== CAUSE: {e.__cause__} ===", flush=True)

        traceback.print_exc()
        raise e

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
  output_text = processor.decode(output_list[0], skip_special_tokens=True)
  metrics = {
      "prefill_time_ms": prefill_time_ms,
      "decode_step_times_ms": decode_step_times_ms,
      "avg_decode_time_per_token_ms": avg_decode_time_ms,
      "avg_decode_time_per_token_no_warmup_ms": avg_decode_time_no_warmup_ms,
  }
  return output_text, metrics, output_tokens


def _print_summary_table(
    all_metrics: Mapping[str, Mapping[str, Any]], max_decode_steps: int
) -> None:
  headers = ["Metric"] + list(all_metrics.keys())
  table_data = []

  prefill_row = ["Total prefill time (ms)"]
  for mode, metrics in all_metrics.items():
    prefill_row.append(metrics["prefill_time_ms"])
  table_data.append(prefill_row)

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

  separator_parts = []
  for i, w in enumerate(col_widths):
    if i == 0:
      separator_parts.append(":" + "-" * (w - 1) if w > 0 else "")
    else:
      separator_parts.append("-" * (w - 1) + ":" if w > 0 else "")
  separator_line = "| " + " | ".join(separator_parts) + " |"
  print(separator_line)

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
  backend: Any
  prefix: str
  key: str
  options: dict[str, Any] | None = None


def _get_compiled_config(device: Device, mode: Mode) -> CompiledConfig:
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
    processor: transformers.AutoProcessor,
    inputs: dict[str, torch.Tensor],
    max_decode_steps: int,
    *,
    device: Device,
    mode: Mode,
    max_buffer_size: int | None = None,
) -> tuple[str, Mapping[str, Any]]:
  """Runs benchmark for a single mode with actual weights."""
  # We don't support actual weights for now since 2B is not in CNS.
  raise NotImplementedError(
      "Running with actual weights is not supported yet for Qwen3-VL-2B."
  )


def _run_with_random_weights(
    config: transformers.PretrainedConfig,
    processor: transformers.AutoProcessor,
    multi_inputs: list[tuple[str, dict[str, torch.Tensor]]],
    max_decode_steps: int,
    *,
    device: Device,
    mode: Mode,
    max_buffer_size: int | None = None,
    verify_correctness: bool = False,
) -> Mapping[str, Any]:
  """Runs benchmark for a single mode with random weights across image resolutions."""
  torch.manual_seed(123)
  model_cpu = Qwen3VLForConditionalGeneration(config).to(torch.bfloat16)

  with torch.device(device.value):
    model_device = copy.deepcopy(model_cpu).to(device.value)

  compiled_config = _get_compiled_config(device, mode)

  if mode == Mode.COMPILED_DYNAMIC:
    compile_kwargs = {"backend": compiled_config.backend}
    if compiled_config.options:
      compile_kwargs["options"] = compiled_config.options
    model_device.model.language_model = torch.compile(
        model_device.model.language_model,
        **compile_kwargs,
    )
    model_device_compiled = model_device

  elif mode == Mode.COMPILED_STATIC:
    compile_kwargs = {
        "backend": compiled_config.backend,
        "dynamic": False,
    }
    if compiled_config.options:
      compile_kwargs["options"] = compiled_config.options
    model_device.model.language_model = torch.compile(
        model_device.model.language_model,
        **compile_kwargs,
    )
    model_device_compiled = model_device

  mode_metrics = {}
  for label, inputs in multi_inputs:
    device_inputs = {}
    for k, v in inputs.items():
      if isinstance(v, torch.Tensor):
        device_inputs[k] = v.to(device.value)
      else:
        device_inputs[k] = v

    prefill_len = device_inputs["input_ids"].shape[1]
    buf_size = (
        max_buffer_size
        if max_buffer_size is not None
        else prefill_len + max_decode_steps
    )

    output_tokens_cpu = None
    output_text_cpu = None
    if verify_correctness:
      cpu_inputs = {
          k: v.to("cpu") if isinstance(v, torch.Tensor) else v
          for k, v in inputs.items()
      }
      output_text_cpu, metrics_cpu, output_tokens_cpu = model_generate(
          model_cpu,
          cpu_inputs,
          processor,
          max_decode_steps,
          prefix=f"[CPU Eager] [{label}] ",
          use_static_cache=False,
          mark_dynamic=False,
          max_buffer_size=buf_size,
      )
      mode_metrics[f"CPU Eager [{label}]"] = metrics_cpu

    if mode == Mode.COMPILED_DYNAMIC:
      if "pixel_values" in device_inputs and isinstance(
          device_inputs["pixel_values"], torch.Tensor
      ):
        # Qwen3-VL flattens image patches into shape (num_patches, patch_dim),
        # where num_patches = (T * H * W) / (patch_size^2). We mark dim 0
        # dynamic to support variable image resolutions across iterations.
        torch._dynamo.mark_dynamic(
            device_inputs["pixel_values"], 0, min=16, max=16384
        )

    run_label = f"{compiled_config.key} [{label}]"
    prefix = f"{compiled_config.prefix}[{label}] "

    output_text_tpu, metrics_compiled, output_tokens_tpu = model_generate(
        model_device_compiled,
        device_inputs,
        processor,
        max_decode_steps,
        prefix=prefix,
        use_static_cache=(mode == Mode.COMPILED_STATIC),
        mark_dynamic=(mode == Mode.COMPILED_DYNAMIC),
        max_buffer_size=buf_size,
    )
    mode_metrics[run_label] = metrics_compiled

    if verify_correctness and output_tokens_cpu is not None:
      assert torch.equal(output_tokens_cpu, output_tokens_tpu), (
          f"Mismatch between CPU and TPU generated tokens for {label}:\n"
          f"  CPU tokens: {output_tokens_cpu.tolist()}\n"
          f"  TPU tokens: {output_tokens_tpu.tolist()}\n"
          f"  CPU text: {output_text_cpu!r}\n"
          f"  TPU text: {output_text_tpu!r}"
      )
      print(f"[Correctness] Verified: CPU and TPU outputs match for {label}!")
      print(f"  CPU text: {output_text_cpu!r}\n  TPU text: {output_text_tpu!r}")

  return mode_metrics


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  artifact_path = os.path.join(
      os.environ.get("TEST_TMPDIR", "/tmp"), "captured_dynamo_fx_graphs.md"
  )
  with open(artifact_path, "w") as f:
    f.write("# Captured PyTorch Dynamo FX Graphs Breakdown\n\n")

  torch.manual_seed(123)

  device = Device(_DEVICE.value)

  if device == Device.CUDA:
    inductor_config.compile_threads = 1

  # Load config from resource
  config_file = runfiles.Create().Rlocation(
      "__main__/examples/huggingface_transformers/model_configs/Qwen/Qwen3-VL-2B-Instruct/config.json"
  )
  model_path = os.path.dirname(config_file)
  config = transformers.AutoConfig.from_pretrained(model_path)

  # Load tokenizer from CNS (same base tokenizer as Qwen3-0.6B)
  tokenizer_path = (
      pathlib.Path(paths.XM_HOME)
      / "weights"
      / "huggingface"
      / "Qwen"
      / "Qwen3-0.6B"
  )
  tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
  image_processor = transformers.AutoImageProcessor.from_pretrained(model_path)
  try:
    video_processor = transformers.AutoVideoProcessor.from_pretrained(
        model_path
    )
  except Exception:
    video_processor = None

  chat_template_file = os.path.join(model_path, "chat_template.json")
  if os.path.exists(chat_template_file):
    with open(chat_template_file, "r") as f:
      chat_template = f.read()
  else:
    chat_template = None

  processor = transformers.Qwen3VLProcessor(
      image_processor=image_processor,
      video_processor=video_processor,
      tokenizer=tokenizer,
      chat_template=chat_template,
  )

  test_resolutions = [
      ("Iteration 1 (224x224 Warmup)", (224, 224)),
      ("Iteration 2 (224x224 Exec)", (224, 224)),
  ]

  multi_inputs = []
  for label, size in test_resolutions:
    image = Image.new("RGB", size, color="red")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image."},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inp = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    patches = inp["pixel_values"].shape[0]
    multi_inputs.append((f"{label} ({patches} patches)", inp))

  dummy_model = Qwen3VLForConditionalGeneration(config)
  for _, inp in multi_inputs:
    mrope_pos, _ = dummy_model.model.get_rope_index(
        input_ids=inp["input_ids"],
        mm_token_type_ids=inp["mm_token_type_ids"],
        image_grid_thw=inp["image_grid_thw"],
    )
    seq_len = inp["input_ids"].shape[1]
    text_pos = torch.arange(
        seq_len, dtype=torch.long, device=inp["input_ids"].device
    ).view(1, 1, -1)
    inp["position_ids"] = torch.cat([text_pos, mrope_pos], dim=0)

  initial_max_decode_steps = _MAX_DECODE_STEPS.value
  explicit_max_buffer_size = _MAX_BUFFER_SIZE.value
  if explicit_max_buffer_size is None:
    max_prefill_len = max(inp["input_ids"].shape[1] for _, inp in multi_inputs)
    explicit_max_buffer_size = max_prefill_len + initial_max_decode_steps + 128
  all_metrics = {}

  if _USE_RANDOM_WEIGHTS.value:
    if device == Device.TPU:
      modes = [Mode.COMPILED_DYNAMIC]
    else:
      modes = []

    for mode in modes:
      metrics = _run_with_random_weights(
          config,
          processor,
          multi_inputs,
          initial_max_decode_steps,
          device=device,
          mode=mode,
          max_buffer_size=explicit_max_buffer_size,
          verify_correctness=_VERIFY_NUMERICAL_CORRECTNESS.value,
      )
      all_metrics.update(metrics)
  else:
    raise ValueError("Only random weights are supported for now.")

  _print_summary_table(all_metrics, initial_max_decode_steps)


if __name__ == "__main__":
  app.run(main)
