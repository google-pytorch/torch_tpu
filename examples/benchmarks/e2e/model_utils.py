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

from collections.abc import Sequence
import contextlib
import dataclasses
import enum
import functools
from importlib import resources
import math
import re
from typing import Any, Callable

from fairscale.nn.model_parallel import initialize as fairscale_init
from fairscale.nn.model_parallel import layers as fairscale_layers
import llama_models.llama3.model as m
import numpy as np
import torch
from torch.distributed import fsdp
import torch.distributed.tensor as dt
from torch.nn import parallel
from examples.benchmarks.e2e import benchmark_utils
from examples.benchmarks.e2e import ragged_moe
from tests import module_registry
import transformers
from transformers import activations
from transformers.models.bert import modeling_bert
from transformers.models.mamba2 import configuration_mamba2
from transformers.models.mamba2 import modeling_mamba2
from transformers.models.nemotron_h import configuration_nemotron_h
from transformers.models.nemotron_h import modeling_nemotron_h
from transformers.models.qwen3 import configuration_qwen3
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe
from transformers.models.qwen3_moe import modeling_qwen3_moe


def get_vocab_size(config: Any, default: int = 256_000) -> int:
  """Gets the vocab size from a config, with a fallback default."""
  direct_vocab_size = getattr(config, "vocab_size", None)
  if direct_vocab_size is not None:
    return direct_vocab_size

  text_config = getattr(config, "text_config", None)
  # this change is specific to Gemma 4
  if text_config is not None:
    text_vocab_size = getattr(text_config, "vocab_size", None)
    if text_vocab_size is not None:
      return text_vocab_size

  return default


@dataclasses.dataclass(frozen=True)
class DynamicDimension:
  min_value: int
  max_value: int
  values: Sequence[int] | None = None


def _get_shape_iterator(
    dim: int | DynamicDimension, num_steps: int, mode: str
) -> Sequence[int]:
  if isinstance(dim, int):
    return [dim] * num_steps

  if dim.values:
    return [dim.values[i % len(dim.values)] for i in range(num_steps)]

  if mode == "warmup":
    if num_steps < 2:
      return [dim.min_value]
    values = [dim.min_value, dim.max_value]
    if num_steps > 2:
      values.extend(
          np.linspace(
              dim.min_value, dim.max_value, num_steps - 2, dtype=int
          ).tolist()
      )
    return values[:num_steps]
  else:
    return np.random.randint(
        dim.min_value, dim.max_value + 1, size=num_steps
    ).tolist()


def _generate_inputs(
    batch_size: int | DynamicDimension,
    sequence_length: int | DynamicDimension,
    shape_fn: Callable[[int, int], Any],
) -> Any:
  if not isinstance(batch_size, DynamicDimension) and not isinstance(
      sequence_length, DynamicDimension
  ):
    return shape_fn(batch_size, sequence_length)

  warmup_steps = benchmark_utils.MAX_WARMUP_STEPS.value
  post_warmup_steps = benchmark_utils.POST_WARMUP_STEPS.value

  warmup_bs = _get_shape_iterator(batch_size, warmup_steps, "warmup")
  warmup_seq = _get_shape_iterator(sequence_length, warmup_steps, "warmup")
  post_bs = _get_shape_iterator(batch_size, post_warmup_steps, "post_warmup")
  post_seq = _get_shape_iterator(
      sequence_length, post_warmup_steps, "post_warmup"
  )

  bs_iterator = list(warmup_bs) + list(post_bs)
  seq_iterator = list(warmup_seq) + list(post_seq)

  return [shape_fn(bs, seq) for bs, seq in zip(bs_iterator, seq_iterator)]


@contextlib.contextmanager
def set_default_dtype(dtype: torch.dtype):
  """Temporarily sets the default torch dtype."""
  old_dtype = torch.get_default_dtype()
  torch.set_default_dtype(dtype)
  try:
    yield
  finally:
    torch.set_default_dtype(old_dtype)


def _get_base_wan_transformer_config():
  # Reference config:
  # third_party/py/torch_tpu/examples/huggingface_diffusers/model_configs/Wan-AI/Wan2.2-TI2V-5B-Diffusers/transformer/config.json
  return {
      "attention_head_dim": 128,
      "cross_attn_norm": True,
      "eps": 1e-06,
      "ffn_dim": 14336,
      "freq_dim": 256,
      "in_channels": 48,
      "num_attention_heads": 24,
      "num_layers": 30,
      "out_channels": 48,
      "patch_size": [1, 2, 2],
      "qk_norm": "rms_norm_across_heads",
      "rope_max_seq_len": 1024,
      "text_dim": 4096,
  }


def _get_base_bert_config():
  cfg = modeling_bert.BertConfig(
      vocab_size=30522,
      hidden_size=768,
      num_hidden_layers=12,
      num_attention_heads=12,
      intermediate_size=3072,
      hidden_dropout_prob=0.1,
      attention_probs_dropout_prob=0.1,
      max_position_embeddings=512,
  )
  cfg._attn_implementation = "eager"  # pylint: disable=protected-access
  return cfg


def _get_base_qwen3_config():
  # https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
  config = configuration_qwen3.Qwen3Config.from_dict({
      "architectures": ["Qwen3ForCausalLM"],
      "attention_bias": False,
      "attention_dropout": 0.0,
      "bos_token_id": 151643,
      "eos_token_id": 151645,
      "head_dim": 128,
      "hidden_act": "silu",
      "hidden_size": 4096,
      "initializer_range": 0.02,
      "intermediate_size": 12288,
      "max_position_embeddings": 40960,
      "max_window_layers": 36,
      "model_type": "qwen3",
      "num_attention_heads": 32,
      "num_hidden_layers": 36,
      "num_key_value_heads": 8,
      "rms_norm_eps": 1e-06,
      "rope_scaling": None,
      "rope_theta": 1000000,
      "sliding_window": None,
      "tie_word_embeddings": False,
      "torch_dtype": "bfloat16",
      "transformers_version": "4.51.0",
      "use_cache": True,
      "use_sliding_window": False,
      "vocab_size": 151936,
  })
  config._attn_implementation = "eager"  # pylint: disable=protected-access
  return config


@dataclasses.dataclass
class ModelAndInput:
  """Dataclass to hold the benchmarking model and its example inputs.

  Attributes:
      model: The model to run the benchmark on.
      example_inputs: The example inputs to run the model with. The model should
        be able to accept these inputs.
  """

  model: torch.nn.Module
  example_inputs: Any


class DistStrat(enum.Enum):
  NONE = "none"
  DDP = "ddp"
  FSDP = "fsdp"


def _load_fsdp_model(
    model: torch.nn.Module,
    model_cpu: torch.nn.Module,
    device: torch.device,
) -> torch.nn.Module:
  """Loads an FSDP-sharded on-device model from a CPU model.

  This function assumes a typical HF Transformer structure.

  Args:
      model: The model to be initialized. Initially it should be on the meta
        device.
      model_cpu: A CPU copy of model with full weights.
      device: The target device for the sharded model (e.g., 'tpu').

  Returns:
      The FSDP-sharded model with weights loaded on the specified device.
  """
  for layer in model.model.layers:
    fsdp.fully_shard(layer)
  fsdp.fully_shard(model)
  model.to_empty(device=device)

  empty_sharded_sd = model.state_dict()
  sharded_sd = {}
  for param_name, full_cpu_tensor in model_cpu.state_dict().items():
    sharded_empty_param = empty_sharded_sd.get(param_name)
    if sharded_empty_param is None:
      continue
    sharded_tensor = dt.distribute_tensor(
        full_cpu_tensor,
        device_mesh=sharded_empty_param.device_mesh,
        placements=sharded_empty_param.placements,
        src_data_rank=None,
    )
    sharded_sd[param_name] = torch.nn.Parameter(sharded_tensor)

  model.load_state_dict(sharded_sd, assign=True)
  return model


@functools.cache
def get_module_registry():
  return module_registry.ModuleRegistry()


def huggingface_llm_model_builder(
    model_and_input_args: Any,
    device: torch.device,
    weights_dtype: torch.dtype,
    is_training: bool,
) -> ModelAndInput:
  """Returns the huggingface LLM model.

  See supported models in
  `third_party/py/torch_tpu/examples/huggingface_transformers/model_configs`.

  Args:
      model_and_input_args: The model and input args.
      device: The device to load the model and inputs on (e.g., 'tpu', 'cuda').
      weights_dtype: The data type for the model weights (e.g., torch.float32,
        torch.bfloat16).
      is_training: Whether the model is in training mode or eval mode.

  Returns:
      A ModelAndInput dataclass containing the loaded Hugging Face model
      and example inputs suitable for benchmarking.
  """
  model_name = model_and_input_args.model_name
  sequence_length = model_and_input_args.sequence_length
  batch_size = model_and_input_args.batch_size
  dist_strat_str = model_and_input_args.custom_kwargs.get("dist_strat", "none")
  dist_strat = DistStrat(dist_strat_str)
  modify_config_hook = model_and_input_args.custom_kwargs.get(
      "modify_config_hook", None
  )

  registry = get_module_registry()
  module_spec = registry.get_module_spec(
      "transformers",
      model_name,
      load_weights=False,
      modify_config_hook=modify_config_hook,
  )

  with torch.device("cpu"):
    model_cpu = module_spec.module_factory().to(weights_dtype)

  _, example_inputs = module_spec.sample_inputs_factory(
      (batch_size, sequence_length), str(device)
  )
  # Pop attention_mask to trigger transformers model-internal fully static
  # causal attention mask fallback. This avoids JAX/XLA JIT compilation
  # control-flow tracing errors in masking_utils.py while keeping identical
  # benchmark workload/math.
  example_inputs.pop("attention_mask", None)
  if model_and_input_args.custom_kwargs.get("disable_vision_inputs", False):
    example_inputs.pop("pixel_values", None)
    example_inputs.pop("image_position_ids", None)

  for k, v in example_inputs.items():
    if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
      example_inputs[k] = v.to(weights_dtype)

  if is_training:
    vocab_size = get_vocab_size(model_cpu.config)

    example_inputs["labels"] = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length),
        device=device,
        dtype=torch.int64,
        requires_grad=False,
    )

  model: torch.nn.Module
  if dist_strat == DistStrat.NONE:
    model = model_cpu.to(device)
  elif dist_strat == DistStrat.DDP:
    model = model_cpu.to(device)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = parallel.DistributedDataParallel(
        model,
        init_sync=False,
        broadcast_buffers=False,
        gradient_as_bucket_view=False,
    )
  elif dist_strat == DistStrat.FSDP:
    with torch.device("meta"):
      model_meta = module_spec.module_factory().to(weights_dtype)
    model_meta.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = _load_fsdp_model(
        model=model_meta, model_cpu=model_cpu, device=device
    )
  else:
    raise ValueError(f"Distributed strategy {dist_strat} unrecognized")

  if is_training:
    model.train()
  else:
    model.eval()

  return ModelAndInput(model=model, example_inputs=example_inputs)


def huggingface_detr_resnet_model_builder(
    model_and_input_args: Any,
    device: torch.device,
    weights_dtype: torch.dtype,
    is_training: bool,
) -> ModelAndInput:
  """Returns a ModelAndInput for the specified Hugging Face vision model.

  Args:
      model_and_input_args: The model and input args. Must contain 'input_shape'
        in custom_kwargs.
      device: The device to load the model and inputs on.
      weights_dtype: The data type for the model weights.
      is_training: Whether the model is in training mode or eval mode.

  Returns:
      A ModelAndInput dataclass containing the loaded Hugging Face vision model
      and example inputs suitable for benchmarking.
  """
  model_name = model_and_input_args.model_name
  input_shape = model_and_input_args.custom_kwargs.get(
      "input_shape", (1, 3, 800, 800)
  )

  registry = get_module_registry()
  module_spec = registry.get_module_spec(
      "transformers",
      model_name,
      load_weights=False,
  )

  with torch.device(device), set_default_dtype(weights_dtype):
    model = module_spec.module_factory()
    example_inputs = {
        "pixel_values": torch.randn(input_shape, dtype=weights_dtype),
    }

  if is_training:
    model.train()
  else:
    model.eval()

  return ModelAndInput(model=model, example_inputs=example_inputs)


def _init_model_weights(model):
  """Initializes model weights to small random values."""

  if hasattr(model, "weight") and model.weight is not None:
    tensor = model.weight.data
    tensor.normal_(std=0.01)

  if hasattr(model, "bias") and model.bias is not None:
    tensor = model.bias.data
    tensor.fill_(0.0)


def whisper_model_builder(
    model_and_input_args: Any,
    device: torch.device,
    weights_dtype: torch.dtype,
    is_training: bool,
) -> ModelAndInput:
  """Returns a ModelAndInput for the Whisper speech-to-text model."""
  model_name = model_and_input_args.model_name
  batch_size = model_and_input_args.batch_size
  sequence_length = model_and_input_args.sequence_length

  registry = get_module_registry()
  module_spec = registry.get_module_spec(
      "transformers", model_name, load_weights=False
  )
  config = module_spec.config

  with torch.device(device), set_default_dtype(weights_dtype):
    model = module_spec.module_factory()

  if is_training:
    model.train()
  else:
    model.eval()

  num_mel_bins = getattr(config, "num_mel_bins", 128)
  feature_length = (
      3000  # Whisper convolution layers strictly require 3000 frames
  )
  vocab_size = getattr(config, "vocab_size", 51866)
  decoder_start = getattr(config, "decoder_start_token_id", 50258)

  input_features = torch.randn(
      (batch_size, num_mel_bins, feature_length),
      dtype=weights_dtype,
      device=device,
  )
  decoder_input_ids = torch.full(
      (batch_size, sequence_length),
      decoder_start,
      dtype=torch.long,
      device=device,
  )

  example_inputs = {
      "input_features": input_features,
      "decoder_input_ids": decoder_input_ids,
  }

  return ModelAndInput(model=model, example_inputs=example_inputs)


def meta_llama_model_builder(
    model_and_input_args: Any,
    device: torch.device,
    weights_dtype: torch.dtype,
    is_training: bool,
) -> ModelAndInput:
  """Returns a ModelAndInput for the specified Meta Llama model.

  This function loads or initializes a Meta Llama model with random weights
  and prepares example inputs suitable for benchmarking. Model parallel
  initialization is handled internally.

  Args:
      model_and_input_args: The model and input args.
      device: The device to load the model and inputs on (e.g., 'tpu', 'cuda').
      weights_dtype: The data type for the model weights (e.g., torch.float32,
        torch.bfloat16).
      is_training: Unused.

  Returns:
      A ModelAndInput dataclass containing the loaded Meta Llama model and
      example inputs.
  """
  del is_training  # Unused for forward
  model_name = model_and_input_args.model_name
  sequence_length = model_and_input_args.sequence_length
  batch_size = model_and_input_args.batch_size

  if model_name == "Llama-3.2-8B":
    args = m.ModelArgs(
        dim=4096,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        n_heads=32,
        n_kv_heads=8,
        n_layers=32,
        norm_eps=1e-05,
        rope_theta=500000.0,
        use_scaled_rope=True,
        vocab_size=128256,
        max_seq_len=2048,
        max_batch_size=batch_size,
    )
  elif model_name == "Llama-3.2-70B":
    args = m.ModelArgs(
        dim=8192,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        n_heads=64,
        n_kv_heads=8,
        n_layers=80,
        norm_eps=1e-05,
        rope_theta=500000.0,
        use_scaled_rope=True,
        vocab_size=128256,
        max_seq_len=2048,
        max_batch_size=batch_size,
    )
  else:
    raise ValueError(f"Unknown model name: {model_name}")

  world_size = (
      torch.distributed.get_world_size()
      if torch.distributed.is_initialized()
      else 1
  )

  # Set model parallel size to gcd(world_size, n_kv_heads), this ensures that
  # n_kv_heads is divisible by the mp_size.
  #
  # NOTE: For forward-only benchmarks, if mp_size fits within a single host
  # (e.g., 8 for Llama-3.2-8B), all tensor parallel communication will be
  # restricted to that host. Since Data Parallelism requires no
  # communication in a forward pass, this means no cross-host communication
  # will occur in such cases.
  mp_size = math.gcd(world_size, args.n_kv_heads)  # pyrefly: ignore[bad-argument-type]

  # Ensure model parallel is initialized
  if not fairscale_init.model_parallel_is_initialized():
    fairscale_init.initialize_model_parallel(mp_size)

  prev_dtype = torch.get_default_dtype()
  torch.set_default_dtype(weights_dtype)

  with torch.device(device):
    # We currently only support random weights for benchmarking
    # TODO(b/461516258): investigate need to initialize model with
    # inference_mode()
    with torch.inference_mode():
      model = m.Transformer(args)

    model.apply(_init_model_weights)

  input_ids = torch.randint(
      0, args.vocab_size, (batch_size, sequence_length), device=device
  )
  # The model expects (tokens, start_pos)
  example_inputs = (input_ids, 0)

  torch.set_default_dtype(prev_dtype)
  return ModelAndInput(model=model, example_inputs=example_inputs)


def ml_layer_model_builder(
    model_and_input_args: Any,
    device: torch.device,
    weights_dtype: torch.dtype,
    is_training: bool,
) -> ModelAndInput:
  """Returns a ModelAndInput for the specified ML layer.

  Args:
      model_and_input_args: The model and input args.
      device: The device to load the model and inputs on (e.g., 'tpu', 'cuda').
      weights_dtype: The data type for the model weights (e.g., torch.float32,
        torch.bfloat16).
      is_training: Whether the model is in training mode or eval mode.

  Returns:
      ModelAndInput dataclass.
  """
  is_bounded_dynamic = model_and_input_args.is_bounded_dynamic
  model_name = model_and_input_args.model_name
  sequence_length = model_and_input_args.sequence_length
  batch_size = model_and_input_args.batch_size
  kwargs = model_and_input_args.custom_kwargs

  if model_name == "nn.Linear":
    if "in_features" not in kwargs:
      raise ValueError("nn.Linear requires in_features to be specified.")
    if "out_features" not in kwargs:
      raise ValueError("nn.Linear requires out_features to be specified.")

    in_features = kwargs["in_features"]
    out_features = kwargs["out_features"]

    class LinearModel(torch.nn.Module):

      def __init__(self, in_features, out_features, dtype):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, dtype=dtype)

      def forward(self, x):
        return self.linear(x)

    model = LinearModel(in_features, out_features, dtype=weights_dtype)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            (bs, seq, in_features),
            dtype=weights_dtype,
            device=device,
        ),
    )
  elif model_name == "nn.Embedding":
    num_embeddings = kwargs["num_embeddings"]
    embedding_dim = kwargs["embedding_dim"]

    class EmbeddingModel(torch.nn.Module):

      def __init__(self, num_embeddings, embedding_dim, dtype):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings, embedding_dim, dtype=dtype
        )

      def forward(self, x):
        return self.embedding(x)

    model = EmbeddingModel(num_embeddings, embedding_dim, dtype=weights_dtype)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randint(
            0,
            num_embeddings,
            (bs, seq),
            dtype=torch.int32,  # Using int32 as per the reference
            device=device,
        ),
    )
  elif model_name == "nn.Dropout":
    p = kwargs["p"]
    input_shape = kwargs["shape"]

    class DropoutModel(torch.nn.Module):

      def __init__(self, p):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=p)

      def forward(self, x):
        return self.dropout(x)

    model = DropoutModel(p)
    example_inputs = torch.randn(
        input_shape,
        dtype=weights_dtype,
        device=device,
        requires_grad=is_training,
    )
  elif model_name == "nn.Tanh":
    input_shape = kwargs["shape"]

    class TanhModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()

      def forward(self, x):
        return self.tanh(x)

    model = TanhModel()
    example_inputs = torch.randn(
        input_shape,
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "nn.BatchNorm1d":
    num_features = kwargs["num_features"]

    class BatchNormModel(torch.nn.Module):

      def __init__(self, num_features, dtype):
        super().__init__()
        self.batchnorm = torch.nn.BatchNorm1d(num_features, dtype=dtype)

      def forward(self, x):
        return self.batchnorm(x)

    model = BatchNormModel(num_features, dtype=weights_dtype)
    example_inputs = torch.randn(
        (batch_size, num_features, sequence_length),
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "nn.BatchNorm2d":
    num_features = kwargs["num_features"]
    height = kwargs["height"]
    width = kwargs["width"]

    class BatchNorm2dModel(torch.nn.Module):

      def __init__(self, num_features, dtype):
        super().__init__()
        self.batchnorm = torch.nn.BatchNorm2d(num_features, dtype=dtype)

      def forward(self, x):
        return self.batchnorm(x)

    model = BatchNorm2dModel(num_features, dtype=weights_dtype)
    example_inputs = torch.randn(
        (batch_size, num_features, height, width),
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "nn.LayerNorm":
    normalized_shape = kwargs["normalized_shape"]
    # shape argument for input generation
    input_shape = kwargs["shape"]

    class LayerNormModel(torch.nn.Module):

      def __init__(self, normalized_shape, dtype):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(normalized_shape, dtype=dtype)

      def forward(self, x):
        return self.layernorm(x)

    model = LayerNormModel(normalized_shape, dtype=weights_dtype)
    example_inputs = torch.randn(
        input_shape,
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "nn.Conv2d":
    in_channels = kwargs["in_channels"]
    out_channels = kwargs["out_channels"]
    kernel_size = kwargs["kernel_size"]
    stride = kwargs["stride"]
    padding = kwargs["padding"]
    height = kwargs["height"]
    width = kwargs["width"]

    class Conv2dModel(torch.nn.Module):

      def __init__(
          self,
          in_channels,
          out_channels,
          kernel_size,
          stride,
          padding,
          dtype,
      ):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dtype=dtype,
        )

      def forward(self, x):
        return self.conv(x)

    model = Conv2dModel(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dtype=weights_dtype,
    )
    example_inputs = torch.randn(
        (batch_size, in_channels, height, width),
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "nn.RMSNorm":
    num_features = kwargs["num_features"]

    class RMSNormModel(torch.nn.Module):

      def __init__(self, num_features, dtype):
        super().__init__()
        self.rmsnorm = torch.nn.RMSNorm(num_features, dtype=dtype)

      def forward(self, x):
        return self.rmsnorm(x)

    model = RMSNormModel(num_features, dtype=weights_dtype)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            (bs, seq, num_features),
            dtype=weights_dtype,
            device=device,
        ),
    )
  elif model_name == "fft.fft":
    dim = kwargs.get("dim", -1)
    norm = kwargs.get("norm", "backward")
    hidden_size = kwargs["hidden_size"]

    class FftModel(torch.nn.Module):

      def __init__(self, dim, norm):
        super().__init__()
        self.dim = dim
        self.norm = norm

      def forward(self, x):
        return torch.fft.fft(x, dim=self.dim, norm=self.norm)

    model = FftModel(dim, norm)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.complex(
            torch.randn(
                (bs, seq, hidden_size), dtype=torch.float32, device=device
            ),
            torch.randn(
                (bs, seq, hidden_size), dtype=torch.float32, device=device
            ),
        ),
    )

  elif model_name == "slice_scatter":
    dim = kwargs["dim"]
    start = kwargs["start"]
    end = kwargs["end"]
    step = kwargs["step"]
    input_shape = kwargs["input_shape"]
    src_shape = kwargs["src_shape"]

    class SliceScatterModel(torch.nn.Module):

      def __init__(self, dim, start, end, step):
        super().__init__()
        self.dim = dim
        self.start = start
        self.end = end
        self.step = step

      def forward(self, x, src):
        return torch.slice_scatter(
            x,
            src,
            dim=self.dim,
            start=self.start,
            end=self.end,
            step=self.step,
        )

    model = SliceScatterModel(dim, start, end, step)
    example_inputs = (
        torch.randn(input_shape, dtype=weights_dtype, device=device),
        torch.randn(src_shape, dtype=weights_dtype, device=device),
    )

  elif model_name == "nn.AvgPool2d":
    kernel_size = kwargs["kernel_size"]
    stride = kwargs["stride"]
    padding = kwargs["padding"]
    channels = kwargs["channels"]
    height = kwargs["height"]
    width = kwargs["width"]

    class AvgPool2dModel(torch.nn.Module):

      def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.pool = torch.nn.AvgPool2d(
            kernel_size, stride=stride, padding=padding, ceil_mode=True
        )

      def forward(self, x):
        return self.pool(x)

    model = AvgPool2dModel(kernel_size, stride, padding)
    example_inputs = torch.randn(
        (batch_size, channels, height, width),
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "nn.AdaptiveAvgPool2d":
    output_size = kwargs["output_size"]
    channels = kwargs["channels"]
    height = kwargs["height"]
    width = kwargs["width"]

    class AdaptiveAvgPool2dModel(torch.nn.Module):

      def __init__(self, output_size):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(output_size)

      def forward(self, x):
        return self.pool(x)

    model = AdaptiveAvgPool2dModel(output_size)
    example_inputs = torch.randn(
        (batch_size, channels, height, width),
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "nn.Flatten":
    start_dim = kwargs["start_dim"]
    shape = kwargs["shape"]

    class FlattenModel(torch.nn.Module):

      def __init__(self, start_dim):
        super().__init__()
        self.flatten = torch.nn.Flatten(start_dim=start_dim)

      def forward(self, x):
        return self.flatten(x)

    model = FlattenModel(start_dim)
    example_inputs = torch.randn(
        shape,
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "nn.MaxPool2d":
    kernel_size = kwargs["kernel_size"]
    stride = kwargs["stride"]
    padding = kwargs["padding"]
    channels = kwargs["channels"]
    height = kwargs["height"]
    width = kwargs["width"]

    class MaxPool2dModel(torch.nn.Module):

      def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(
            kernel_size, stride=stride, padding=padding
        )

      def forward(self, x):
        return self.pool(x)

    model = MaxPool2dModel(kernel_size, stride, padding)
    example_inputs = torch.randn(
        (batch_size, channels, height, width),
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "nn.ReLU":
    shape = kwargs["shape"]

    class ReLUModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

      def forward(self, x):
        return self.relu(x)

    model = ReLUModel()
    example_inputs = torch.randn(
        shape,
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "nn.PReLU":
    channels = kwargs["channels"]
    height = kwargs["height"]
    width = kwargs["width"]
    channel_wise = kwargs.get("channel_wise", True)

    class PreluModel(torch.nn.Module):

      def __init__(self, dtype):
        super().__init__()
        self.prelu = torch.nn.PReLU(
            num_parameters=channels if channel_wise else 1, dtype=dtype
        )

      def forward(self, x):
        return self.prelu(x)

    model = PreluModel(dtype=weights_dtype)

    def shape_fn(bs, seq):
      return torch.randn(
          (bs, channels, height, width),
          dtype=weights_dtype,
          device=device,
          requires_grad=is_training,
      )

    example_inputs = _generate_inputs(batch_size, sequence_length, shape_fn)
  elif model_name == "nn.GLU":
    dim = kwargs.get("dim", -1)
    shape = kwargs["shape"]
    num_features = shape[-1]

    class GLUModel(torch.nn.Module):

      def __init__(self, dim, features, is_training, dtype):
        super().__init__()
        self.dim = dim
        self.is_training = is_training
        if is_training:
          self.linear = torch.nn.Linear(features, features, dtype=dtype)

      def forward(self, x):
        if self.is_training:
          x = self.linear(x)
        return torch.nn.functional.glu(x, dim=self.dim)

    model = GLUModel(dim, num_features, is_training, dtype=weights_dtype)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            (bs, seq, num_features),
            dtype=weights_dtype,
            device=device,
        ),
    )
  elif model_name == "nn.Conv1d":
    in_channels = kwargs["in_channels"]
    out_channels = kwargs["out_channels"]
    kernel_size = kwargs["kernel_size"]
    stride = kwargs.get("stride", 1)
    padding = kwargs.get("padding", 0)
    dilation = kwargs.get("dilation", 1)
    groups = kwargs.get("groups", 1)
    bias = kwargs.get("bias", True)

    class Conv1dModel(torch.nn.Module):

      def __init__(
          self,
          in_channels,
          out_channels,
          kernel_size,
          stride,
          padding,
          dilation,
          groups,
          bias,
          dtype,
      ):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            dtype=dtype,
        )

      def forward(self, x):
        return self.conv(x)

    model = Conv1dModel(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        dtype=weights_dtype,
    )
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            (bs, in_channels, seq),
            dtype=weights_dtype,
            device=device,
        ),
    )
  elif model_name == "nn.MultiheadAttention":
    embed_dim = kwargs["embed_dim"]
    num_heads = kwargs["num_heads"]
    dropout = kwargs.get("dropout", 0.0)
    bias = kwargs.get("bias", True)

    class MultiheadAttentionModel(torch.nn.Module):

      def __init__(self, embed_dim, num_heads, dropout, bias, dtype):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            dtype=dtype,
        )

      def forward(self, query, key, value, key_padding_mask=None):
        return self.mha(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]

    model = MultiheadAttentionModel(
        embed_dim, num_heads, dropout, bias, dtype=weights_dtype
    )
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: (
            torch.randn(seq, bs, embed_dim, dtype=weights_dtype, device=device),
            torch.randn(seq, bs, embed_dim, dtype=weights_dtype, device=device),
            torch.randn(seq, bs, embed_dim, dtype=weights_dtype, device=device),
            None,
        ),
    )

  elif model_name == "SelectAdaptivePool2d":
    import timm.layers

    output_size = kwargs["output_size"]
    pool_type = kwargs["pool_type"]
    flatten = kwargs["flatten"]
    input_fmt = kwargs["input_fmt"]
    channels = kwargs["channels"]
    height = kwargs["height"]
    width = kwargs["width"]

    class SelectAdaptivePool2dModel(torch.nn.Module):

      def __init__(self, output_size, pool_type, flatten, input_fmt):
        super().__init__()
        self.pool = timm.layers.SelectAdaptivePool2d(
            output_size=output_size,
            pool_type=pool_type,
            flatten=flatten,
            input_fmt=input_fmt,
        )

      def forward(self, x):
        return self.pool(x)

    model = SelectAdaptivePool2dModel(
        output_size, pool_type, flatten, input_fmt
    )
    example_inputs = torch.randn(
        (batch_size, channels, height, width),
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "Bottleneck":
    from timm.models import resnet

    inplanes = kwargs["inplanes"]
    planes = kwargs["planes"]
    stride = kwargs["stride"]
    height = kwargs["height"]
    width = kwargs["width"]

    class BottleneckModel(torch.nn.Module):

      def __init__(self, inplanes, planes, stride):
        super().__init__()
        self.bottleneck = resnet.Bottleneck(
            inplanes=inplanes, planes=planes, stride=stride
        )

      def forward(self, x):
        return self.bottleneck(x)

    model = BottleneckModel(inplanes, planes, stride).to(dtype=weights_dtype)
    example_inputs = torch.randn(
        (batch_size, inplanes, height, width),
        dtype=weights_dtype,
        device=device,
    )

  elif model_name == "BertLayer":
    cfg = _get_base_bert_config()
    model = modeling_bert.BertLayer(cfg)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: (
            torch.randn(
                bs,
                seq,
                cfg.hidden_size,
                dtype=weights_dtype,
                device=device,
            ),
            None,
        ),
    )

    class BertLayerWrapper(torch.nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, hidden_states, attention_mask):
        return self.m(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            output_attentions=False,
            cache_position=None,
        )[0]

    model = BertLayerWrapper(model).to(dtype=weights_dtype)

  elif model_name == "BertSelfOutput":
    cfg = _get_base_bert_config()
    model = modeling_bert.BertSelfOutput(cfg)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: (
            torch.randn(
                bs,
                seq,
                cfg.hidden_size,
                dtype=weights_dtype,
                device=device,
            ),
            torch.randn(
                bs,
                seq,
                cfg.hidden_size,
                dtype=weights_dtype,
                device=device,
            ),
        ),
    )

    class BertSelfOutputWrapper(torch.nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, hidden_states, input_tensor):
        return self.m(hidden_states=hidden_states, input_tensor=input_tensor)

    model = BertSelfOutputWrapper(model).to(dtype=weights_dtype)

  elif model_name == "BertIntermediate":
    cfg = _get_base_bert_config()
    model = modeling_bert.BertIntermediate(cfg)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            bs,
            seq,
            cfg.hidden_size,
            dtype=weights_dtype,
            device=device,
        ),
    )

    class BertIntermediateWrapper(torch.nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, hidden_states):
        return self.m(hidden_states)

    model = BertIntermediateWrapper(model).to(dtype=weights_dtype)

  elif model_name == "BertOutput":
    cfg = _get_base_bert_config()
    model = modeling_bert.BertOutput(cfg)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: (
            torch.randn(
                bs,
                seq,
                cfg.intermediate_size,
                dtype=weights_dtype,
                device=device,
            ),
            torch.randn(
                bs,
                seq,
                cfg.hidden_size,
                dtype=weights_dtype,
                device=device,
            ),
        ),
    )

    class BertOutputWrapper(torch.nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, hidden_states, input_tensor):
        return self.m(hidden_states=hidden_states, input_tensor=input_tensor)

    model = BertOutputWrapper(model).to(dtype=weights_dtype)

  elif model_name == "GELUActivation":
    cfg = _get_base_bert_config()
    model = activations.GELUActivation(use_gelu_python=False)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            bs,
            seq,
            cfg.intermediate_size,
            dtype=weights_dtype,
            device=device,
            requires_grad=is_training,
        ),
    )
  elif model_name == "BertPooler":
    cfg = _get_base_bert_config()
    model = modeling_bert.BertPooler(cfg).to(dtype=weights_dtype)
    example_inputs = torch.randn(
        batch_size,
        sequence_length,
        cfg.hidden_size,
        dtype=weights_dtype,
        device=device,
    )

  elif model_name == "BertEmbeddings":
    cfg = _get_base_bert_config()
    model = modeling_bert.BertEmbeddings(cfg).to(dtype=weights_dtype)
    input_ids = torch.randint(
        0,
        cfg.vocab_size,
        (batch_size, sequence_length),
        dtype=torch.long,
        device=device,
    )
    example_inputs = input_ids
  elif model_name == "Qwen3Attention":
    config = _get_base_qwen3_config()
    model = modeling_qwen3.Qwen3Attention(config, layer_idx=0)

    head_dim = config.hidden_size // config.num_attention_heads

    def _qwen3_shape_fn(bs, seq):
      cos = torch.randn(1, seq, head_dim, device=device, dtype=weights_dtype)
      sin = torch.randn(1, seq, head_dim, device=device, dtype=weights_dtype)
      return (
          torch.randn(
              bs,
              seq,
              config.hidden_size,
              dtype=weights_dtype,
              device=device,
          ),
          (cos, sin),
          None,
      )

    example_inputs = _generate_inputs(
        batch_size, sequence_length, _qwen3_shape_fn
    )

    class Qwen3AttentionWrapper(torch.nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, hidden_states, position_embeddings, attention_mask):
        return self.m(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )[0]

    model = Qwen3AttentionWrapper(model)
    model = model.to(dtype=weights_dtype)

  elif model_name == "Qwen3RMSNorm":
    hidden_size = kwargs["hidden_size"]
    model = modeling_qwen3.Qwen3RMSNorm(hidden_size=hidden_size)
    model = model.to(dtype=weights_dtype)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            bs,
            seq,
            hidden_size,
            dtype=weights_dtype,
            device=device,
        ),
    )

  elif model_name == "Qwen3MLP":
    config = configuration_qwen3.Qwen3Config(
        hidden_size=kwargs["hidden_size"],
        intermediate_size=kwargs["intermediate_size"],
        hidden_act="silu",
    )
    model = modeling_qwen3.Qwen3MLP(config)
    model = model.to(dtype=weights_dtype)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            bs,
            seq,
            config.hidden_size,
            dtype=weights_dtype,
            device=device,
        ),
    )

  elif model_name == "SiLUActivation":
    model = activations.SiLUActivation()
    input_shape = kwargs["shape"]
    example_inputs = torch.randn(
        input_shape,
        dtype=weights_dtype,
        device=device,
    )

  elif model_name == "Qwen3RotaryEmbedding":
    config = configuration_qwen3.Qwen3Config(
        max_position_embeddings=kwargs["max_position_embeddings"],
        rope_theta=kwargs["rope_theta"],  # pyrefly: ignore[unexpected-keyword]
    )
    model = modeling_qwen3.Qwen3RotaryEmbedding(config)
    head_dim = kwargs["head_dim"]

    position_ids = (
        torch.arange(0, sequence_length, dtype=torch.long, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    example_inputs = (
        torch.randn(
            batch_size,
            sequence_length,
            head_dim,
            dtype=weights_dtype,
            device=device,
        ),
        position_ids,
    )

    class Qwen3RotaryEmbeddingWrapper(torch.nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, x, position_ids):
        return self.m(x, position_ids)  # returns cos, sin

    model = Qwen3RotaryEmbeddingWrapper(model)

  elif model_name == "nn.f.scaled_dot_product_attention":
    embed_dim = kwargs["embed_dim"]
    q_num_heads = kwargs["q_num_heads"]
    kv_num_heads = kwargs["kv_num_heads"]
    qk_head_dim = kwargs["qk_head_dim"]
    v_head_dim = kwargs["v_head_dim"]
    is_causal = kwargs["is_causal"]
    enable_gqa = kwargs["enable_gqa"]
    backend = kwargs["backend"]

    class AttentionLayer(torch.nn.Module):

      def __init__(self, dtype):
        super().__init__()
        self.q_proj = torch.nn.Linear(
            embed_dim,
            q_num_heads * qk_head_dim,
            bias=False,
            dtype=dtype,
        )
        self.k_proj = torch.nn.Linear(
            embed_dim,
            kv_num_heads * qk_head_dim,
            bias=False,
            dtype=dtype,
        )
        self.v_proj = torch.nn.Linear(
            embed_dim,
            kv_num_heads * v_head_dim,
            bias=False,
            dtype=dtype,
        )
        self.out_proj = torch.nn.Linear(
            q_num_heads * v_head_dim,
            embed_dim,
            bias=False,
            dtype=dtype,
        )

      def forward(self, x):
        bsz, q_len, _ = x.size()
        q = (
            self.q_proj(x)
            .view(bsz, q_len, q_num_heads, qk_head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(bsz, q_len, kv_num_heads, qk_head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(bsz, q_len, kv_num_heads, v_head_dim)
            .transpose(1, 2)
        )

        with torch.nn.attention.sdpa_kernel(backend):
          attn_output = torch.nn.functional.scaled_dot_product_attention(
              q, k, v, is_causal=is_causal, enable_gqa=enable_gqa
          )
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(bsz, q_len, q_num_heads * v_head_dim)
        )
        return self.out_proj(attn_output)

    model = AttentionLayer(dtype=weights_dtype)
    example_inputs = torch.randn(
        (batch_size, kwargs["q_seq_len"], embed_dim),
        dtype=weights_dtype,
        device=device,
    )
  elif model_name == "aten._masked_softmax":
    dim = kwargs.get("dim", -1)
    mask_type = kwargs.get("mask_type", 2)
    num_heads = kwargs["num_heads"]
    q_seq_len = kwargs["q_seq_len"]
    kv_seq_len = kwargs["kv_seq_len"]

    class MaskedSoftmaxModel(torch.nn.Module):

      def __init__(self, dim, mask_type):
        super().__init__()
        self.dim = dim
        self.mask_type = mask_type

      def forward(self, x, mask):
        return torch.ops.aten._masked_softmax(
            x, mask, dim=self.dim, mask_type=self.mask_type
        )

    model = MaskedSoftmaxModel(dim, mask_type)
    model = model.to(dtype=weights_dtype)

    def shape_fn(bs, seq):
      cur_q_seq_len = (
          seq if isinstance(sequence_length, DynamicDimension) else q_seq_len
      )
      cur_kv_seq_len = (
          seq if isinstance(sequence_length, DynamicDimension) else kv_seq_len
      )

      x = torch.randn(
          (bs, num_heads, cur_q_seq_len, cur_kv_seq_len),
          dtype=weights_dtype,
          device=device,
          requires_grad=is_training,
      )

      if mask_type == 0:
        mask = torch.randint(
            0,
            2,
            (cur_q_seq_len, cur_kv_seq_len),
            dtype=torch.bool,
            device=device,
        )
      elif mask_type == 1:
        mask = torch.randint(
            0, 2, (bs, cur_kv_seq_len), dtype=torch.bool, device=device
        )
      else:
        mask = torch.randint(
            0,
            2,
            (bs, num_heads, cur_q_seq_len, cur_kv_seq_len),
            dtype=torch.bool,
            device=device,
        )
      return (x, mask)

    example_inputs = _generate_inputs(batch_size, sequence_length, shape_fn)
  elif model_name == "Mamba2Block":
    config = configuration_mamba2.Mamba2Config(
        hidden_size=kwargs["hidden_size"],
        state_size=kwargs["state_size"],
        conv_kernel=kwargs["conv_kernel"],
        expand=kwargs["expand"],
        num_heads=kwargs["num_heads"],
        head_dim=kwargs["head_dim"],
        n_groups=kwargs["n_groups"],
        chunk_size=kwargs["chunk_size"],
    )
    model = modeling_mamba2.Mamba2Block(config, layer_idx=0)

    class Mamba2BlockWrapper(torch.nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, hidden_states):
        return self.m(hidden_states)

    model = Mamba2BlockWrapper(model).to(dtype=weights_dtype)

    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            bs, seq, kwargs["hidden_size"], dtype=weights_dtype, device=device
        ),
    )

  elif model_name == "NemotronHMamba2Block":
    config = configuration_nemotron_h.NemotronHConfig(
        hidden_size=kwargs["hidden_size"],
        ssm_state_size=kwargs["state_size"],
        conv_kernel=kwargs["conv_kernel"],
        expand=kwargs["expand"],
        mamba_num_heads=kwargs["num_heads"],
        mamba_head_dim=kwargs["head_dim"],
        n_groups=kwargs["n_groups"],
        chunk_size=kwargs["chunk_size"],
    )
    config.use_mamba_kernels = False
    config.layers_block_type = ["linear_attention"]
    model = modeling_nemotron_h.NemotronHBlock(config, layer_idx=0)

    class NemotronHMamba2BlockWrapper(torch.nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, hidden_states):
        return self.m(hidden_states)

    model = NemotronHMamba2BlockWrapper(model).to(dtype=weights_dtype)

    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            bs, seq, kwargs["hidden_size"], dtype=weights_dtype, device=device
        ),
    )

  elif model_name == "torch.topk":
    k = kwargs["k"]
    dim = kwargs.get("dim", None)
    largest = kwargs.get("largest", True)
    sorted_out = kwargs.get("sorted", True)
    num_features = kwargs["num_features"]

    class TopKModel(torch.nn.Module):

      def __init__(self, k, dim, largest, sorted_out):
        super().__init__()
        self.k = k
        self.dim = dim
        self.largest = largest
        self.sorted_out = sorted_out

      def forward(self, x):
        if self.dim is None:
          values, indices = torch.topk(
              x,
              self.k,
              largest=self.largest,
              sorted=self.sorted_out,
          )
        else:
          values, indices = torch.topk(
              x,
              self.k,
              dim=self.dim,
              largest=self.largest,
              sorted=self.sorted_out,
          )
        return values, indices

    model = TopKModel(k, dim, largest, sorted_out)
    example_inputs = _generate_inputs(
        batch_size,
        sequence_length,
        lambda bs, seq: torch.randn(
            (bs, seq, num_features),
            dtype=weights_dtype,
            device=device,
            requires_grad=is_training,
        ),
    )

  else:
    raise ValueError(f"Unknown ML layer model: {model_name}")

  model = model.to(device)
  if is_training:
    model.train()
  else:
    model.eval()

  if (
      is_bounded_dynamic
      and not isinstance(batch_size, DynamicDimension)
      and not isinstance(sequence_length, DynamicDimension)
  ):
    raise ValueError(
        "At least one of batch_size or sequence_length must be a"
        " DynamicDimension for bounded dynamic benchmarks."
    )

  return ModelAndInput(model=model, example_inputs=example_inputs)


def timm_model_builder(
    model_and_input_args: Any,
    device: torch.device,
    weights_dtype: torch.dtype,
    is_training: bool,
) -> ModelAndInput:
  """Returns the TIMM model.

  Args:
      model_and_input_args: The model and input args.
      device: The device to load the model and inputs on (e.g., 'tpu', 'cuda').
      weights_dtype: The data type for the model weights (e.g., torch.float32,
        torch.bfloat16).
      is_training: Whether the model is in training mode or eval mode.
      use_torch_compile: Whether to wrap the model with `torch.compile`.

  Returns:
      A ModelAndInput dataclass containing the loaded TIMM model
      and example inputs suitable for benchmarking.
  """
  model_name = model_and_input_args.model_name
  input_shape = model_and_input_args.custom_kwargs["input_shape"]

  registry = get_module_registry()
  module_spec = registry.get_module_spec("timm", model_name, load_weights=False)

  with torch.device("cpu"):
    model_cpu = module_spec.module_factory().to(weights_dtype)

  input_args, _ = module_spec.sample_inputs_factory(input_shape, str(device))
  example_inputs = input_args[0].to(weights_dtype)
  model = model_cpu.to(device)
  if is_training:
    model.train()
  else:
    model.eval()

  return ModelAndInput(model=model, example_inputs=example_inputs)


# TODO(b/505851863): use model registry to get the model and inputs for all categories.
def huggingface_diffuser_model_builder(
    model_and_input_args: Any,
    device: torch.device,
    weights_dtype: torch.dtype,
    is_training: bool,
) -> ModelAndInput:
  """Returns a ModelAndInput for the specified Hugging Face Diffuser model.

  Args:
      model_and_input_args: The model and input args.
      device: The device to load the model and inputs on (e.g., 'tpu', 'cuda').
      weights_dtype: The data type for the model weights (e.g., torch.float32,
        torch.bfloat16).
      is_training: Whether the model is in training mode or eval mode.
      use_torch_compile: Whether to wrap the model with `torch.compile`.

  Returns:
      A ModelAndInput dataclass containing the loaded Hugging Face Diffuser
      model
      and example inputs.
  """
  model_name = model_and_input_args.model_name
  if model_name == "Wan-AI/Wan2.2-TI2V-5B-Diffusers":
    registry = get_module_registry()
    module_spec = registry.get_module_spec(
        "diffusers", model_name, load_weights=False, subfolder="transformer"
    )

    with torch.device("cpu"):
      model_cpu = module_spec.module_factory()

    model = model_cpu.to(weights_dtype).to(device)
    model.apply(_init_model_weights)

    # Dimensions derived from pipeline settings (default: height=704, width=1280, num_frames=5):
    text_seq_len = model_and_input_args.custom_kwargs.get("text_seq_len", 512)
    batch_size = model_and_input_args.custom_kwargs.get("batch_size", 1)
    #   (VAE temporal downscale factor is 4)
    num_frames = model_and_input_args.custom_kwargs.get("num_frames", 5)
    latent_frames = model_and_input_args.custom_kwargs.get(
        "latent_frames", (num_frames - 1) // 4 + 1
    )
    #   (VAE spatial downscale factor is 16 for Wan2.2)
    height = model_and_input_args.custom_kwargs.get("height", 704)
    width = model_and_input_args.custom_kwargs.get("width", 1280)
    latent_height = model_and_input_args.custom_kwargs.get(
        "latent_height", height // 16
    )
    latent_width = model_and_input_args.custom_kwargs.get(
        "latent_width", width // 16
    )
    _, example_inputs = module_spec.sample_inputs_factory(
        (batch_size, text_seq_len, latent_frames, latent_height, latent_width),
        str(device),
    )

    for k, v in example_inputs.items():

      if isinstance(v, torch.Tensor) and v.is_floating_point():
        example_inputs[k] = v.to(weights_dtype)

  else:
    # Generic fallback for other diffusers using registry default shapes
    registry = get_module_registry()
    subfolder = model_and_input_args.custom_kwargs.get("subfolder", None)
    module_spec = registry.get_module_spec(
        "diffusers", model_name, load_weights=False, subfolder=subfolder
    )

    with torch.device("cpu"):
      model_cpu = module_spec.module_factory().to(weights_dtype)

    # Generate inputs on CPU, passing None for shape to use registry defaults
    _, example_inputs = module_spec.sample_inputs_factory(None, "cpu")  # pytype: disable=wrong-arg-count

    # Move inputs to device and weights_dtype
    device_inputs = {}
    for k, v in example_inputs.items():
      if isinstance(v, torch.Tensor):
        if v.is_floating_point():
          device_inputs[k] = v.to(device).to(weights_dtype)
        else:
          device_inputs[k] = v.to(device)
      else:
        device_inputs[k] = v

    model = model_cpu.to(device)
    example_inputs = device_inputs

  if is_training:
    model.train()
  else:
    model.eval()

  return ModelAndInput(model=model, example_inputs=example_inputs)


def vjepa_2_model_builder(
    model_and_input_args: Any,
    device: torch.device,
    weights_dtype: torch.dtype,
    is_training: bool,
) -> ModelAndInput:
  """Returns a ModelAndInput for the specified VJEPA-2 model.

  Args:
      model_and_input_args: The model and input args.
      device: The device to load the model and inputs on (e.g., 'tpu', 'cuda').
      weights_dtype: The data type for the model weights (e.g., torch.float32,
        torch.bfloat16).
      is_training: Whether the model is in training mode or eval mode.

  Returns:
      A ModelAndInput dataclass containing the loaded VJEPA-2 model and example
      inputs.
  """
  model_name = model_and_input_args.model_name

  registry = get_module_registry()
  module_spec = registry.get_module_spec(
      "transformers",
      model_name,
      load_weights=False,
  )
  config = module_spec.config
  input_shape = (
      model_and_input_args.batch_size,
      getattr(config, "frames_per_clip", 64),
      getattr(config, "in_chans", 3),
      getattr(config, "crop_size", 256),
      getattr(config, "crop_size", 256),
  )
  with torch.device(device), set_default_dtype(weights_dtype):
    model = module_spec.module_factory()
    model.apply(_init_model_weights)
    # Skip module registry for inputs
    example_inputs = {
        "pixel_values_videos": torch.randn(input_shape, dtype=weights_dtype),
    }
  if not is_training:
    example_inputs["skip_predictor"] = True

  if is_training:
    model.train()
  else:
    model.eval()

  return ModelAndInput(model=model, example_inputs=example_inputs)


def _apply_tensor_parallel_plan(
    module,
    name_prefix="",
    tp_plan=None,
    world_size=1,
    rank=0,
):
  """Applies a tensor parallel plan to a model."""
  for name, child in module.named_children():
    full_name = f"{name_prefix}.{name}" if name_prefix else name
    full_name = full_name.removeprefix("model.")

    if isinstance(child, ragged_moe.RaggedMoeQwen3):
      # Handled during MoE replacement
      continue

    if isinstance(child, torch.nn.Linear):
      for pattern, tp_type in tp_plan.items():  # pyrefly: ignore[missing-attribute]
        if re.fullmatch(pattern, full_name):
          original_linear = child
          new_linear = None

          # TODO(mkkhanna): gather_output=False for ColumnParallelLinear layer
          # does not work for self_attn linear layer. It requires world_size to
          # be the divisor of num_attention_heads (32) and
          # num_key_value_heads (4). Hence, for it to work we need world_size
          # to be 4, which is currently not supported.
          if "self_attn" in full_name:
            gather_output = True
          else:
            gather_output = False

          if tp_type == "colwise":
            new_linear = fairscale_layers.ColumnParallelLinear(
                original_linear.in_features,
                original_linear.out_features,
                bias=original_linear.bias is not None,
                gather_output=gather_output,
                init_method=lambda w: w,
            )
          elif tp_type == "rowwise":
            new_linear = fairscale_layers.RowParallelLinear(
                original_linear.in_features,
                original_linear.out_features,
                bias=original_linear.bias is not None,
                input_is_parallel=not gather_output,
                init_method=lambda w: w,
            )

          if new_linear:
            setattr(module, name, new_linear)
          break  # Found a match, move to the next child

    # Recurse for submodules
    _apply_tensor_parallel_plan(child, full_name, tp_plan, world_size, rank)


def _replace_qwen_moe_with_ragged_moe(model, config):
  for layer in model.model.layers:
    assert isinstance(layer.mlp, modeling_qwen3_moe.Qwen3MoeSparseMoeBlock)
    layer.mlp = ragged_moe.RaggedMoeQwen3(config, is_tensor_parallel=True)


def _replace_gemma_moe_with_ragged_moe(model, config):
  layers = model.model.language_model.layers
  for layer in layers:
    if hasattr(layer, "enable_moe_block") and layer.enable_moe_block:
      layer.experts = ragged_moe.RaggedExpertsGemma4(
          config.text_config if hasattr(config, "text_config") else config
      )


def qwen_ragged_moe_model_builder(
    model_and_input_args: Any,
    device: torch.device,
    weights_dtype: torch.dtype,
    is_training: bool,
) -> ModelAndInput:
  """Returns a ModelAndInput for the specified Qwen Ragged MoE model.

  Args:
      model_and_input_args: The model and input args.
      device: The device to load the model and inputs on (e.g., 'tpu', 'cuda').
      weights_dtype: The data type for the model weights (e.g., torch.float32,
        torch.bfloat16).
      is_training: Unused.

  Returns:
      A ModelAndInput dataclass containing the loaded Qwen Ragged MoE model and
      example inputs.
  """
  del is_training  # Unused
  model_name = model_and_input_args.model_name
  sequence_length = model_and_input_args.sequence_length
  batch_size = model_and_input_args.batch_size
  registry = get_module_registry()
  module_spec = registry.get_module_spec(
      "transformers", model_name, load_weights=False
  )
  module_config = module_spec.config

  rank = (
      torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
  )
  world_size = (
      torch.distributed.get_world_size()
      if torch.distributed.is_initialized()
      else 1
  )

  # Directly initialize the model on the device.
  with torch.device(device), set_default_dtype(weights_dtype):
    model = module_spec.module_factory()
    if world_size > 1:
      # Fairscale library setup for distributed model parallelism.
      if not fairscale_init.model_parallel_is_initialized():
        fairscale_init.initialize_model_parallel(world_size)

      _replace_qwen_moe_with_ragged_moe(model, module_config)

      _apply_tensor_parallel_plan(
          model,
          name_prefix="",
          tp_plan=module_config.base_model_tp_plan,
          world_size=world_size,
          rank=rank,
      )
    model.apply(_init_model_weights)

  _, example_inputs = module_spec.sample_inputs_factory(
      (batch_size, sequence_length), str(device)
  )
  return ModelAndInput(model=model, example_inputs=example_inputs)


def huggingface_resnet_model_builder(
    model_and_input_args: Any,
    device: torch.device,
    weights_dtype: torch.dtype,
    is_training: bool,
) -> ModelAndInput:
  """Returns a Hugging Face ResNet model with custom vision inputs.

  See supported models in
  `third_party/py/torch_tpu/examples/huggingface_transformers/model_configs`.

  Args:
      model_and_input_args: The model and input args.
      device: The device to load the model and inputs on (e.g., "tpu", "cuda").
      weights_dtype: The data type for the model weights (e.g., torch.float32,
        torch.bfloat16).
      is_training: Whether the model is in training mode or eval mode.

  Returns:
      A ModelAndInput dataclass containing the loaded Hugging Face model
      and custom inputs suitable for benchmarking.
  """
  model_name = model_and_input_args.model_name
  batch_size = model_and_input_args.batch_size

  registry = get_module_registry()
  module_spec = registry.get_module_spec(
      "transformers",
      model_name,
      load_weights=False,
  )

  with torch.device("cpu"):
    model_cpu = module_spec.module_factory().to(weights_dtype)

  example_inputs = {
      "pixel_values": torch.randn(
          batch_size, 3, 224, 224, device=device, dtype=weights_dtype
      )
  }

  model = model_cpu.to(device)
  if is_training:
    model.train()
  else:
    model.eval()

  return ModelAndInput(model=model, example_inputs=example_inputs)


def gemma_ragged_moe_model_builder(
    model_and_input_args: Any,
    device: torch.device,
    weights_dtype: torch.dtype,
    is_training: bool,
) -> ModelAndInput:
  """Returns a ModelAndInput for the specified Gemma 4 Ragged MoE model."""
  del is_training  # Unused
  model_name = model_and_input_args.model_name
  sequence_length = model_and_input_args.sequence_length
  batch_size = model_and_input_args.batch_size
  modify_config_hook = model_and_input_args.custom_kwargs.get(
      "modify_config_hook", None
  )
  registry = get_module_registry()
  module_spec = registry.get_module_spec(
      "transformers",
      model_name,
      load_weights=False,
      modify_config_hook=modify_config_hook,
  )
  module_config = module_spec.config

  with torch.device(device), set_default_dtype(weights_dtype):
    model = module_spec.module_factory()
    _replace_gemma_moe_with_ragged_moe(model, module_config)

    model.apply(_init_model_weights)

  _, example_inputs = module_spec.sample_inputs_factory(
      (batch_size, sequence_length), str(device)
  )
  example_inputs.pop("attention_mask", None)
  if model_and_input_args.custom_kwargs.get("disable_vision_inputs", False):
    example_inputs.pop("pixel_values", None)
    example_inputs.pop("image_position_ids", None)
  return ModelAndInput(model=model, example_inputs=example_inputs)


def gemma4_custom_standalone_model_builder(
    model_and_input_args: Any,
    device: torch.device,
    weights_dtype: torch.dtype,
    is_training: bool = False,
) -> ModelAndInput:
  """Builder for the Gemma 4 custom standalone SWA model."""
  from examples.gemma4 import model as gemma4_standalone_model

  model_preset = model_and_input_args.custom_kwargs.get("preset", "e2b")
  num_layers = model_and_input_args.custom_kwargs.get("num_layers", 35)
  vocab_size = model_and_input_args.custom_kwargs.get("vocab_size", 262144)

  if model_preset == "e2b":
    config = gemma4_standalone_model.Gemma4Config(
        vocab_size=vocab_size,
        hidden_size=1536,
        intermediate_size=6144,
        num_hidden_layers=num_layers,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=256,
        sliding_window=min(512, model_and_input_args.sequence_length),
        use_bidirectional_attention=None,
    )
  else:
    config = gemma4_standalone_model.Gemma4Config(
        vocab_size=vocab_size,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=num_layers,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        sliding_window=1024,
        use_bidirectional_attention=None,
    )

  torch.manual_seed(1234)
  model = (
      gemma4_standalone_model.Gemma4ForCausalLM(config)
      .to(weights_dtype)
      .to(device)
  )

  bs = model_and_input_args.batch_size
  seq_len = model_and_input_args.sequence_length

  def _create_input(b: int, s: int):
    torch.manual_seed(1234)
    input_ids = torch.randint(0, vocab_size, (b, s), device=device)
    if is_training:
      labels = torch.randint(0, vocab_size, (b, s), device=device)
      return {"input_ids": input_ids, "labels": labels}
    return input_ids

  example_inputs = _generate_inputs(bs, seq_len, _create_input)
  return ModelAndInput(model=model, example_inputs=example_inputs)
