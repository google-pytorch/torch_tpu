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

# %%
"""Tests of Llama3 1B training based on llama3_1B_training_nb.py."""

from absl import logging
import os
import sys
import time
from typing import Any, Dict

from absl import flags
from absl.testing import absltest
import datasets
from etils import epath
import torch
import torch._inductor.config as inductor_config
from torch.utils import tensorboard
from torch_tpu._internal import compile as torch_tpu_compile
from torch_tpu._internal import execution_mode
from torch_tpu._internal.utils import benchmarking
from examples import paths
import tqdm
import transformers

from torch_tpu._internal.shims.xprof import traceme


from torch_tpu._internal.utils import log_utils

log_utils.log_to_stderr()


# %%
_DEVICE = flags.DEFINE_enum(
    "device",
    None,
    ["tpu", "cuda", "xla_cuda"],
    required=True,
    help="Accelerator to test.",
)

_EAGER_MODE = flags.DEFINE_enum(
    "eager_mode",
    "DEFER_AND_FUSE",
    [
        "DEFER_AND_FUSE",
        "DEFER_NEVER",
        "DEFER_NEVER_AND_LAUNCH_BLOCKING",
    ],
    "Eager mode for the model.",
)

_USE_TORCH_COMPILE = flags.DEFINE_bool(
    "use_torch_compile",
    False,
    "Whether to use torch.compile to run the model.",
)

BASE_PATH = epath.Path(paths.XM_HOME)
MODEL_PATH = BASE_PATH / "weights/huggingface/meta-llama/Llama-3.2-1B"
TOK_PATH = (
    BASE_PATH / "weights/huggingface/meta-llama/Meta-Llama-3-8B-Instruct/"
)
_ENABLE_TENSORBOARD_LOGGING = flags.DEFINE_bool(
    "enable_tensorboard_logging",
    False,
    "Whether to enable TensorBoard logging.",
)
_TB_SUMMARY_LOGGING_DIR = flags.DEFINE_string(
    "tb_summary_logging_dir",
    default=os.environ.get("TB_SUMMARY_LOGGING_DIR", None),
    help="TensorBoard summary logging directory.",
)


# %%
def get_torch_device() -> torch.device:
  if _DEVICE.value == "tpu":
    return torch.device("tpu")
  elif _DEVICE.value == "cuda":
    return torch.device("cuda")
  elif _DEVICE.value == "xla_cuda":
    return torch.device("xla_cuda")
  else:
    raise ValueError(f"Unsupported device: {_DEVICE.value}")


# %%
def get_eager_mode() -> execution_mode.EagerMode:
  if _EAGER_MODE.value == "DEFER_AND_FUSE":
    return execution_mode.EagerMode.DEFER_AND_FUSE
  elif _EAGER_MODE.value == "DEFER_NEVER":
    return execution_mode.EagerMode.DEFER_NEVER
  elif _EAGER_MODE.value == "DEFER_NEVER_AND_LAUNCH_BLOCKING":
    return execution_mode.EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING
  else:
    raise ValueError(f"Unsupported defer mode: {_EAGER_MODE.value}")


# %%
def torch_compile_model(model):
  if _DEVICE.value == "cuda":
    model = torch.compile(model)
  elif _DEVICE.value in ("tpu", "xla_cuda"):
    model = torch.compile(
        model, dynamic=False, backend=torch_tpu_compile.TpuBackend()
    )
  return model


# %%
def _generate_text(
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    device: torch.device,
    max_new_tokens: int,
) -> None:
  """Generates text from the model."""
  streamer = transformers.TextIteratorStreamer(tokenizer)
  chat = [[{"role": "user", "content": "What is the capital of Uganda?"}]]
  text = tokenizer.apply_chat_template(chat, tokenize=False)
  inputs = tokenizer(text, return_tensors="pt")
  inputs = {k: v.to(device) for k, v in inputs.items()}
  model.generate(**inputs, streamer=streamer, max_new_tokens=max_new_tokens)
  for token_idx, new_text in enumerate(streamer):
    print(new_text, end="", flush=True)
    if (token_idx + 1) % 20 == 0:
      print()
  print()


# %%
def _train(
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    device: torch.device,
    epochs: int,
    local_batch_size: int,
    steps: int,
    global_batch_size: int | None,
    dp_degree: int = 1,
) -> Dict[str, Any]:
  """Trains the model using a TorchTitan-style microbatching loop.

  This function accumulates gradients over several "microbatches"
  (defined by `local_batch_size`) before performing a single optimizer step.
  Mimics the structure of
  https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/train.py

  The number of microbatches to accumulate is calculated as:
  `gradient_accumulation_steps = global_batch_size // (local_batch_size *
  dp_degree)`

  Args:
    model: The PyTorch model to train.
    tokenizer: The tokenizer, used for formatting the dataset.
    device: The torch.device to run training on (e.g., 'cuda', 'tpu').
    epochs: The total number of training epochs.
    local_batch_size: The batch size for a single microbatch (this is the size
      passed to the DataLoader).
    steps: The total number of microbatches to run per epoch.
    global_batch_size: The target effective batch size. If set to -1 (default),
      it defaults to `local_batch_size * dp_degree`, resulting in
      `gradient_accumulation_steps = 1`.
    dp_degree: The data parallel world size (defaults to 1).

  Returns:
    'step_train_loss' (per optimizer step), and 'val_loss'.

  Raises:
    ValueError: If `global_batch_size` is not divisible by
      `(local_batch_size * dp_degree)`, or if `steps` (total microbatches)
      is not divisible by `gradient_accumulation_steps`.
  """
  if global_batch_size is None:
    # If global_batch_size is not specified, default to no accumulation
    global_batch_size = local_batch_size * dp_degree
    logging.info(
        "global_batch_size not set, defaulting to "
        f"local_batch_size * dp_degree ({global_batch_size}). "
        "gradient_accumulation_steps will be 1."
    )

  if global_batch_size % (local_batch_size * dp_degree) != 0:
    raise ValueError(
        f"global_batch_size ({global_batch_size}) must be divisible by "
        f"local_batch_size ({local_batch_size}) * dp_degree ({dp_degree})"
    )

  gradient_accumulation_steps = global_batch_size // (
      local_batch_size * dp_degree
  )
  # Total number of optimizer steps per epoch.
  num_optimizer_steps = steps // gradient_accumulation_steps

  if steps % gradient_accumulation_steps != 0:
    logging.warning(
        f"Number of steps ({steps}) is not a multiple of "
        f"gradient_accumulation_steps ({gradient_accumulation_steps}). "
        "The dataloader will be truncated."
    )

  dataset_path = BASE_PATH / "datasets/huggingface/tatsu-lab/alpaca"
  raw_dataset = datasets.load_from_disk(str(dataset_path))["train"]

  def format(example):
    chat = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(chat, tokenize=False)
    inputs = tokenizer(
        text, padding="max_length", max_length=128, truncation=True
    )
    return inputs

  dataset = raw_dataset.map(
      format,
      load_from_cache_file=False,
      cache_file_name="/tmp/cache.arrow",
      remove_columns=raw_dataset.column_names,
  )

  # Dataloader uses the local_batch_size.
  train_dataloader = torch.utils.data.DataLoader(
      dataset.select(range(steps * local_batch_size)),
      collate_fn=transformers.DataCollatorForLanguageModeling(
          tokenizer=tokenizer, mlm=False
      ),
      batch_size=local_batch_size,
  )

  val_dataloader = torch.utils.data.DataLoader(
      dataset.select(
          range(steps * local_batch_size, (steps + 100) * local_batch_size)
      ),
      collate_fn=transformers.DataCollatorForLanguageModeling(
          tokenizer=tokenizer, mlm=False
      ),
      batch_size=local_batch_size,
  )

  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
  stats = {
      "epoch": [],
      "train_loss": [],
      "val_loss": [],
      "step_train_loss": [],
      "step_times": [],
  }

  for epoch in range(1, epochs + 1):
    model.train()
    data_iterator = iter(train_dataloader)
    pbar = tqdm.tqdm(
        range(num_optimizer_steps), unit="optimizer step", desc=f"Epoch {epoch}"
    )
    train_loss = 0.0

    # Run training steps
    with traceme.TraceMe(f"Epoch_{epoch}_Train"):
      for _ in pbar:
        step_start_time = time.time()
        accumulated_losses = []
        optimizer.zero_grad()
        # If data runs out during gradient accumulation, that entire step will not
        # be executed.
        for _microbatch in range(gradient_accumulation_steps):
          inputs = next(data_iterator)
          inputs = {k: v.to(device) for k, v in inputs.items()}
          outputs = model(**inputs)
          loss = outputs.loss
          # We must manually scale the loss here
          # in TorchTitan this is done by 'rescale_accumulated_loss' wrapper
          loss = loss / gradient_accumulation_steps
          loss.backward()
          # Store detached loss for logging
          accumulated_losses.append(loss.detach())

        optimizer.step()
        step_train_loss = torch.sum(torch.stack(accumulated_losses)).item()

        step_end_time = time.time()
        step_time = step_end_time - step_start_time
        stats["step_times"].append(step_time)

        pbar.set_postfix(step_loss=f"{step_train_loss:.4f}")
        stats["step_train_loss"].append(step_train_loss)
        train_loss += step_train_loss

    # Validation loop
    with traceme.TraceMe(f"Epoch_{epoch}_Eval"):
      model.eval()
      val_losses = []
      with torch.no_grad():
        for val_inputs in val_dataloader:
          val_inputs = {k: v.to(device) for k, v in val_inputs.items()}
          outputs = model(**val_inputs)
          # Append the detached tensor instead of syncing.
          # Move to CPU to prevent out-of-memory (OOM) errors during testing.
          val_losses.append(outputs.loss.detach().to("cpu"))
      avg_val_loss = torch.stack(val_losses).mean().item()
      avg_train_loss = train_loss / num_optimizer_steps
      print(
          f"\nEpoch {epoch}, Train loss: {avg_train_loss:.4f}, Val loss:"
          f" {avg_val_loss:.4f}\n"
      )
      stats["epoch"].append(epoch)
      stats["train_loss"].append(avg_train_loss)
      stats["val_loss"].append(avg_val_loss)

  return stats


# %%
def _plot_stats(stats: Dict[str, Any]):
  """Plots the training and validation loss."""
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

  plt.figure()
  plt.title("Per-epoch loss")
  plt.plot(stats["epoch"], stats["train_loss"], label="Train Loss")
  plt.plot(stats["epoch"], stats["val_loss"], label="Val Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()
  plt.show()
  plt.close()

  plt.figure()
  plt.title("Per-step training loss")
  plt.plot(stats["step_train_loss"])
  plt.xlabel("Step")
  plt.ylabel("Loss")
  plt.show()
  plt.close()

  plt.figure()
  plt.title("Per-step training time")
  plt.plot(stats["step_times"])
  plt.xlabel("Step")
  plt.ylabel("Time (seconds)")
  plt.show()
  plt.close()


# %%
def _record_metrics(
    stats: Dict[str, Any],
    writer: "tensorboard.SummaryWriter",
    epochs: int,
    steps_per_epoch: int,
):
  """Calculates and records training metrics with TensorBoard.

  Args:
    stats: A dictionary containing training statistics.
    writer: An optional TensorBoard SummaryWriter for logging.
    epochs: The number of training epochs.
    steps_per_epoch: The number of steps per epoch.
  """

  # Retrieve the final validation loss
  final_val_loss = stats["val_loss"][-1]
  stable_step_times = None
  if epochs > 1:
    stable_step_times = stats["step_times"][steps_per_epoch:]
  else:
    logging.warning(
        "Cannot calculate training step time with only one epoch."
        " Run with epochs > 1 for a stable performance metric."
    )

  logging.info("Exporting metrics to TensorBoard...")
  benchmarking.record_tensorboard_metrics(
      writer,
      benchmarking.METRIC_PROFILES["Training/ValidationLoss"],
      [final_val_loss],
  )
  if stable_step_times is not None:
    benchmarking.record_tensorboard_metrics(
        writer,
        benchmarking.METRIC_PROFILES["Training/StepTime"],
        stable_step_times,
    )


# %%
class Llama31BTrainingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    if "ipykernel" in sys.modules:
      seed = 301
      logging.info("Using default seed for colab: %d", seed)
    else:
      seed = absltest.FLAGS.test_random_seed
      if seed is None or not isinstance(seed, int):
        raise ValueError(
            "absltest.FLAGS.test_random_seed not an int: %s" % seed
        )
      logging.info("Using absltest.FLAGS.test_random_seed: %d", seed)

    # TODO(gunhyun): Figure out why inductor multiprocessing library is causing
    # issues with GPU.
    if _DEVICE.value == "cuda":
      inductor_config.compile_threads = 1
    torch.manual_seed(seed)

    self.writer = None
    if _ENABLE_TENSORBOARD_LOGGING.value:
      log_dir = _TB_SUMMARY_LOGGING_DIR.value
      if log_dir:
        logging.info("TensorBoard logging enabled. Writing to: %s", log_dir)
        self.writer = tensorboard.SummaryWriter(log_dir)
      else:
        logging.warning(
            "TensorBoard logging is enabled but --tb_summary_logging_dir is"
            " not set. No logs will be written."
        )

    # Initialize device
    self.device = get_torch_device()

    # Set defer mode for torch_tpu
    if _DEVICE.value in ("tpu", "xla_cuda"):
      execution_mode.set_eager_mode(get_eager_mode())

    # Initialize model
    self.model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype="auto", device_map="auto"
    ).to(self.device)

    if _USE_TORCH_COMPILE.value:
      self.model = torch_compile_model(self.model)

    # Initialize tokenizer
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(TOK_PATH)
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def tearDown(self):
    super().tearDown()
    if self.writer:
      logging.info("Flushing and closing TensorBoard SummaryWriter.")
      self.writer.flush()
      self.writer.close()

  def test_training(self):
    """Test llama3.1B model training with instruct SFT dataset."""
    epochs = 3
    local_batch_size = 20
    global_batch_size = 80
    steps = 200  # Number of microbatches per epoch
    dp_degree = 1  # Test is for single TPU

    # Generate tokens before training
    _generate_text(self.model, self.tokenizer, self.device, max_new_tokens=20)

    # Train the model
    stats = _train(
        self.model,
        self.tokenizer,
        self.device,
        epochs,
        local_batch_size,
        steps,
        global_batch_size,
        dp_degree,
    )

    if _DEVICE.value == "cuda":
      peak_memory_usage_mb = torch.cuda.memory.max_memory_allocated() / (
          1024 * 1024
      )
      print(
          f"ACTUAL CUDA MEMORY USAGE: {peak_memory_usage_mb:.2f} MB",
          flush=True,
      )

    # After training
    _generate_text(self.model, self.tokenizer, self.device, max_new_tokens=100)

    # Report metrics to TensorBoard (using the number of optimizer steps per epoch)
    if self.writer:
      gradient_accumulation_steps = global_batch_size // (
          local_batch_size * dp_degree
      )
      # Total number of optimizer steps per epoch.
      num_optimizer_steps = steps // gradient_accumulation_steps
      _record_metrics(
          stats=stats,
          writer=self.writer,
          epochs=epochs,
          steps_per_epoch=num_optimizer_steps,
      )

    return stats


# %%
if __name__ == "__main__":
  if "ipykernel" not in sys.modules:
    absltest.main()
  else:
    test = Llama31BTrainingTest()
    test.setUp()
    stats = test.test_training()
    _plot_stats(stats)
