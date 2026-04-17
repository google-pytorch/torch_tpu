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

# This is a development script for testing Qwen3 KV cache on TPU.
# It is not intended to be a production-quality example.

from contextlib import contextmanager
import time
import types
from typing import Any, Optional, Sequence

from absl import app
from absl import flags
import torch
from torch_tpu._internal import dynamism
from torch_tpu._internal import sync
from torch_tpu._internal.utils import utils
import transformers
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.models.qwen3 import modeling_qwen3

from torch_tpu._internal.shims.xprof import traceme
from rules_python.python.runfiles import runfiles

BASE_MODEL_CONFIG_PATH = "__main__/examples/huggingface_transformers/model_configs"

_RUN_MODE = flags.DEFINE_enum(
    "run_mode",
    "static",
    ["static", "bd", "both"],
    "The run mode for inference.",
)
_COMPILE = flags.DEFINE_boolean(
    "compile", False, "Whether to compile the model."
)
_DECODE_STEPS = flags.DEFINE_integer(
    "decode_steps", 4, "Number of decode steps to run."
)
_PADDING_LENGTH = flags.DEFINE_integer(
    "padding_length", 10, "Padding length for the dynamic tensors."
)
_SEQ_LEN = flags.DEFINE_integer(
    "seq_len", 2048, "Sequence length for the input."
)

_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    "Qwen/Qwen3-0.6B",
    "Name of the model to run. Can be 'Qwen/Qwen3-0.6B', 'Qwen/Qwen3-1.7B', "
    " or 'Qwen/Qwen3-4B'.",
)

_DEVICE = flags.DEFINE_string(
    "device",
    "tpu",
    "Device to run the model on. Can be 'tpu' or 'cuda'.",
)


def sync_device(
    tensors_to_sync: torch.Tensor | list[torch.Tensor], wait: bool = True
) -> None:
  if _DEVICE.value == "tpu":
    # Wait for the compilation and execution of model output to complete.
    sync.synchronize(tensors_to_sync, wait=wait)
  elif _DEVICE.value == "cuda":
    torch.cuda.synchronize()


@contextmanager
def measure_cache_misses(run_type: str, step_name: str):
  """Measures TPU cache misses during a code block."""
  if _DEVICE.value == "tpu":
    prev_cache_misses = torch.tpu._get_cache_misses()
  yield
  if _DEVICE.value == "tpu":
    current_cache_misses = torch.tpu._get_cache_misses()
    print(
        f"[{run_type}] Cache misses during {step_name}: ",
        current_cache_misses - prev_cache_misses,
        flush=True,
    )


# --- Monkey Patching Logic ---
# This is a hack to inject dynamism.mark_dynamic into the DynamicLayer.update.
# We need to patch the function in order to mark the key/value tensors before
# they are resized by the torch.cat in the original update.


def _create_wrapped_bounded_dynamic_update(
    dim: int, lower_bound: int, upper_bound: int
):
  """Creates a wrapped update function for DynamicLayer with specific dynamism parameters."""

  def _bounded_dynamic_update(
      self,
      key_states: torch.Tensor,
      value_states: torch.Tensor,
      cache_kwargs: Optional[dict[str, Any]] = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrapped version of DynamicLayer.update to inject dynamism.mark_dynamic.

    This function replaces the original DynamicLayer.update. It calls
    dynamism.mark_dynamic on the existing self.keys and self.values tensors
    before the original update logic is executed.
    """

    # Dynamism should be marked on the tensors *before* they are potentially
    # resized by the torch.cat in the original update.
    # We only mark if the layer has been initialized and contains tensors.
    if self.is_initialized:
      try:
        # The sequence dimension is dim 2 for key/value tensors.
        if self.keys is not None and self.keys.numel() > 0:
          dynamism.mark_dynamic(
              self.keys,
              dimension=dim,
              lower_bound=lower_bound,
              upper_bound=upper_bound,
          )
          print("dynamism.mark_dynamic on keys: ", self.keys.shape, flush=True)
        if self.values is not None and self.values.numel() > 0:
          dynamism.mark_dynamic(
              self.values,
              dimension=dim,
              lower_bound=lower_bound,
              upper_bound=upper_bound,
          )
          print(
              "dynamism.mark_dynamic on values: ", self.values.shape, flush=True
          )
      except Exception as e:
        print(
            "Error during dynamism.mark_dynamic in"
            f" wrapped_dynamic_layer_update: {e}",
            flush=True,
        )

    # Call the original DynamicLayer.update.
    return DynamicLayer.update(self, key_states, value_states, cache_kwargs)

  return _bounded_dynamic_update


def _dynamic_patch_cache(
    cache: DynamicCache, bounded_dim: int, lower_bound: int, upper_bound: int
) -> DynamicCache:
  """Applies a dynamic patch to the DynamicLayer instances within a DynamicCache.

  The update method of each DynamicLayer within the provided DynamicCache
  is replaced to include dynamism.mark_dynamic calls.
  """
  print(
      f"\n--- Applying dynamic patch to {len(cache.layers)} layers within"
      f" DynamicCache on key/value tensors dimension {bounded_dim} with"
      f" lower_bound {lower_bound} and upper_bound {upper_bound}---",
      flush=True,
  )
  # Create the wrapped update function for this specific set of dynamism parameters
  wrapped_update_fn = _create_wrapped_bounded_dynamic_update(
      bounded_dim, lower_bound, upper_bound
  )

  for _, layer in enumerate(cache.layers):
    if isinstance(layer, DynamicLayer):
      # Replace the 'update' method for this specific DynamicLayer instance.
      layer.update = types.MethodType(wrapped_update_fn, layer)
  return cache


# --- End Monkey Patching Logic ---


def _create_decode_attention_mask(
    initial_seq_len, past_seq_len, dtype, native_device
):
  """Creates full attention mask for decode step."""
  # attention mask for decode with q_len=1 is all zeros
  # shape is (batch_size, 1, q_len, kv_len)
  # here batch_size=1, q_len=1, kv_len = past_seq_len + 1
  kv_len = past_seq_len + 1
  attention_mask_tensor = torch.zeros(
      (1, 1, 1, kv_len), dtype=dtype, device=native_device
  )
  attention_mask_dict = {"full_attention": attention_mask_tensor}
  if not _COMPILE.value:
    dynamism.mark_dynamic(
        attention_mask_tensor,
        dimension=3,
        lower_bound=initial_seq_len,
        upper_bound=initial_seq_len + _PADDING_LENGTH.value + 1,
    )

  return attention_mask_dict


def _create_position_ids(past_seq_len, native_device):
  """Creates position_ids for decode step."""
  position_ids = torch.tensor(
      [[past_seq_len]], dtype=torch.long, device=native_device
  )
  return position_ids


def model_generate(
    model,
    seq_len: int,
    initial_inputs: torch.Tensor,
    decode_steps: int = 10,
    use_bounded_dynamism: bool = False,
):
  """Generate from a model."""
  native_device = model.device
  print(f"Running model on {native_device}", flush=True)

  past_key_values = DynamicCache(
      config=model.config,
  )

  run_type = "Static"
  if use_bounded_dynamism:
    if not _COMPILE.value:
      past_key_values = _dynamic_patch_cache(
          past_key_values,
          bounded_dim=2,
          lower_bound=seq_len,
          upper_bound=seq_len + _PADDING_LENGTH.value,
      )
    run_type = "BD"

  model_kwargs = {
      "past_key_values": past_key_values,
  }

  output_tokens = []
  all_logits = []
  with torch.no_grad():
    # Prefill
    start_time = time.time()
    with traceme.TraceMe(f"[{run_type}] Prefill"):
      with measure_cache_misses(run_type, "prefill"):
        output = model(input_ids=initial_inputs, **model_kwargs)
        sync_device(output.logits, wait=True)
    end_time = time.time()
    prefill_time = end_time - start_time
    print(
        f"[{run_type}] Prefill time: {prefill_time * 1000:.2f} ms", flush=True
    )
    next_token = torch.argmax(output.logits[:, -1, :], dim=-1).unsqueeze(-1)
    if _RUN_MODE.value == "both":
      logits = output.logits.to("cpu")
      all_logits.append(logits)
    output_tokens.append(next_token.to("cpu"))

    if _COMPILE.value:
      model = torch.compile(model, backend="tpu")

    # Decode
    for i in range(decode_steps):
      decode_input_ids = next_token.to(native_device)

      past_seq_len = past_key_values.get_seq_length()
      model_kwargs = {
          "past_key_values": output.past_key_values,
      }
      bd_overrides = {}
      if use_bounded_dynamism:
        bd_overrides["attention_mask"] = _create_decode_attention_mask(
            seq_len,
            past_seq_len,
            model.dtype,
            native_device,
        )
        bd_overrides["position_ids"] = _create_position_ids(
            past_seq_len, native_device
        )
        model_kwargs.update(bd_overrides)

      start_time = time.time()
      with traceme.TraceMe(f"[{run_type}] Decode step {i + 1}"):
        with measure_cache_misses(run_type, f"decode step {i + 1}"):
          output = model(input_ids=decode_input_ids, **model_kwargs)
          sync_device(output.logits, wait=True)
      end_time = time.time()
      decode_time = end_time - start_time
      print(
          f"[{run_type}] Decode step {i + 1} time: {decode_time * 1000:.2f} ms",
          flush=True,
      )
      next_token = torch.argmax(output.logits[:, -1, :], dim=-1).unsqueeze(-1)
      if _RUN_MODE.value == "both":
        logits = output.logits.to("cpu")
        all_logits.append(logits)
      output_tokens.append(next_token.to("cpu"))

  output_ids = torch.cat(output_tokens, dim=1)
  output_list = output_ids.tolist()
  return output_list, all_logits


def main(argv: Sequence[str]) -> None:

  if _DEVICE.value == "tpu":
    device = torch.device("tpu")
  else:
    device = "cuda"

  torch.manual_seed(1234)

  model_config_path = runfiles.Create().Rlocation(
      f"{BASE_MODEL_CONFIG_PATH}/{_MODEL_NAME.value}/config.json"
  )
  config = transformers.AutoConfig.from_pretrained(model_config_path)
  config._attn_implementation = "eager"

  with torch.device(device):
    model = modeling_qwen3.Qwen3ForCausalLM(config)

  inputs = torch.randint(
      0, config.vocab_size, (1, _SEQ_LEN.value), device=device
  )
  # Materialize inputs tensor.
  sync_device(inputs, wait=True)

  output_text_without_bd, all_logits_without_bd = None, None
  if _RUN_MODE.value in ["static", "both"]:
    output_text_without_bd, all_logits_without_bd = model_generate(
        model,
        _SEQ_LEN.value,
        inputs,
        decode_steps=_DECODE_STEPS.value,
        use_bounded_dynamism=False,
    )
    print("-" * 10, "[Static] Model Output", "-" * 10)
    print(output_text_without_bd, flush=True)

  output_text_with_bd, all_logits_with_bd = None, None
  if _RUN_MODE.value in ["bd", "both"]:
    # Clear cache before running with bounded dynamism.
    if _DEVICE.value == "tpu":
      torch.tpu._clear_cache()

    output_text_with_bd, all_logits_with_bd = model_generate(
        model,
        _SEQ_LEN.value,
        inputs,
        decode_steps=_DECODE_STEPS.value,
        use_bounded_dynamism=True,
    )
    print("-" * 10, "[BD] Model Output", "-" * 10)
    print(output_text_with_bd, flush=True)

  if _RUN_MODE.value == "both":
    assert output_text_with_bd == output_text_without_bd
    for i in range(len(all_logits_without_bd)):
      utils.assert_close(
          all_logits_without_bd[i], all_logits_with_bd[i], rtol=10, atol=1e-2
      )


if __name__ == "__main__":
  app.run(main)
