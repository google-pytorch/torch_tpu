# Copyright 2025 Google LLC
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

"""Basic training loop with llama3.2 model distributed by DDP or FSDPv2."""

from absl import app
from absl import logging
from etils import epath
import torch
from torch import distributed as dist
from torch import nn
from torch.distributed import fsdp
import torch.distributed.tensor as dt
from torch.nn import parallel
from torch_tpu._internal import compile as torch_tpu_compile
from torch_tpu._internal.utils import log_utils
from examples import paths
import transformers


log_utils.log_to_stderr()

# Configuration Variables
MODEL_SIZE = "1B"
DIST_STRAT = "fsdp"
MODEL_COMPILE = False
USE_RANDOM_WEIGHTS = True
BATCH_SIZE = 16
SEQ_LEN = 2056
NUM_TRAIN_STEPS = 10
BUCKET_CAP_MB = 25

WEIGHTS_DIR = f"{paths.XM_HOME}/weights/huggingface"
MODEL_ID_DICT = {
    "1B": "meta-llama/Llama-3.2-1B",
    "8B": "meta-llama/Meta-Llama-3-8B-Instruct",
}


def _torch_compile_model(model):
  return torch.compile(
      model, dynamic=False, backend=torch_tpu_compile.TpuBackend()
  )


def _verify_all_parameters_initialized(empty_sharded_sd, sharded_sd):
  all_fsdp_keys = set(empty_sharded_sd.keys())
  matched_keys = set(sharded_sd.keys())
  missing_keys = all_fsdp_keys - matched_keys
  if missing_keys:
    raise RuntimeError(
        f"FSDP materialization failed. The following {len(missing_keys)}"
        f" parameters were left uninitialized after to_empty(): {missing_keys}"
    )


def _shard_and_materialize_model(
    model: nn.Module, device: str, model_dir: epath.Path, load_weights: bool
) -> nn.Module:
  """Shards the meta model and materializes it on the target device."""
  for layer in model.model.layers:
    fsdp.fully_shard(layer)
  fsdp.fully_shard(model.model)
  fsdp.fully_shard(model)
  model.to_empty(device=device)

  # Load weights/config onto CPU model first.
  if load_weights:
    logging.info("Loading model weights from %s", model_dir)
    cpu_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16
    )
  else:
    logging.info("Loading model config from %s", model_dir)
    with torch.device("cpu"):
      config = transformers.AutoConfig.from_pretrained(model_dir)
      cpu_model = transformers.AutoModelForCausalLM.from_config(
          config, dtype=torch.bfloat16
      )

  # Re-init RoPE buffers (persistent=False) since to_empty wipes them.
  for mod, cpu_mod in zip(model.modules(), cpu_model.modules()):
    if hasattr(cpu_mod, "inv_freq"):
      mod.register_buffer(
          "inv_freq", cpu_mod.inv_freq.to(device), persistent=False
      )
      if hasattr(cpu_mod, "attention_scaling"):
        mod.attention_scaling = cpu_mod.attention_scaling

  # Shard the weights and load them onto the meta model.
  empty_sharded_sd = model.state_dict()
  sharded_sd = {}
  for param_name, full_cpu_tensor in cpu_model.state_dict().items():
    sharded_empty_param = empty_sharded_sd.get(param_name)
    if sharded_empty_param is None:
      continue
    sharded_tensor = dt.distribute_tensor(
        full_cpu_tensor,
        device_mesh=sharded_empty_param.device_mesh,
        placements=sharded_empty_param.placements,
        src_data_rank=None,
    )
    sharded_sd[param_name] = nn.Parameter(sharded_tensor)

  _verify_all_parameters_initialized(empty_sharded_sd, sharded_sd)

  model.load_state_dict(sharded_sd, assign=True)
  logging.info("Loaded model weights from %s", model_dir)

  torch.tpu.synchronize()

  return model


def worker_fn(argv=None):
  del argv  # Unused
  device = torch.device("tpu")
  dist.init_process_group(backend="tpu_dist")

  rank = dist.get_rank()
  world_size = dist.get_world_size()
  batch_size = BATCH_SIZE
  global_batch_size = batch_size * world_size
  logging.info(
      "Using batch_size=%d (global=%d, world=%d)",
      batch_size,
      global_batch_size,
      world_size,
  )

  model_id = MODEL_ID_DICT[MODEL_SIZE]
  model_dir = epath.Path(WEIGHTS_DIR) / model_id
  if not model_dir.exists():
    logging.warning(
        f"Local model directory not found: {model_dir}. Falling back to"
        " HuggingFace Hub."
    )
    model_dir = model_id  # Use HF hub ID directly

  config = transformers.AutoConfig.from_pretrained(model_dir)

  # Shard the model using FSDP strategy.
  if DIST_STRAT == "fsdp":
    logging.info("Instantiating meta-model for FSDP")
    with torch.device("meta"):
      model: nn.Module = transformers.AutoModelForCausalLM.from_config(
          config, dtype=torch.bfloat16
      )
    model = _shard_and_materialize_model(
        model, device, model_dir, not USE_RANDOM_WEIGHTS
    )
  else:
    if USE_RANDOM_WEIGHTS:
      logging.info("Instantiating model with random weights")
      with torch.device(device):
        model: nn.Module = transformers.AutoModelForCausalLM.from_config(
            config, dtype=torch.bfloat16
        )
    else:
      logging.info("Reading model weights from %s", model_dir)
      model: nn.Module = transformers.AutoModelForCausalLM.from_pretrained(
          model_dir, dtype=torch.bfloat16
      )
      model.to(device)
      logging.info("Loaded model on device %s (rank %d)", device, rank)
  # A bug in the interaction between DDP and HF's LlamaDecoderLayer on
  # newer GPUs results in multiple hooks marking some parameters as ready.
  model.gradient_checkpointing_enable(
      gradient_checkpointing_kwargs={"use_reentrant": False}
  )
  vocab_size = model.config.vocab_size

  # Shard the model using DDP strategy.
  if DIST_STRAT == "ddp":
    model = parallel.DistributedDataParallel(
        model,
        broadcast_buffers=False,
        gradient_as_bucket_view=False,
        bucket_cap_mb=BUCKET_CAP_MB,
    )

  if MODEL_COMPILE:
    logging.info("Compiling model on rank %d", rank)
    model = _torch_compile_model(model)

  seq_len = SEQ_LEN
  torch.manual_seed(rank)

  # Generate random input and target tokens.
  data = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=device)
  input_tokens = data[:, :seq_len]
  target_tokens = data[:, 1:]

  optimizer = torch.optim.Adam(
      model.parameters(), lr=1e-4
  )  # pytype: disable=attribute-error
  model.train()

  def model_step_func():
    losses = []
    for _ in range(NUM_TRAIN_STEPS):
      optimizer.zero_grad()
      output = model(input_ids=input_tokens, labels=target_tokens)
      loss = output.loss
      losses.append(loss.detach())
      loss.backward()
      optimizer.step()
    return losses

  # No mode specified. Runs a single step and syncs.
  losses = model_step_func()

  # Sync losses across all ranks to log them.
  torch.tpu.synchronize()
  logging.info("[Rank %d] Losses: %s", rank, torch.stack(losses))


if __name__ == "__main__":
  app.run(worker_fn)
