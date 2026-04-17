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

# %% [markdown]
# # Llama3 1B Training on TPU with HuggingFace Transformers
#
# This notebook demonstrates:
#
# 1. A pre-trained LLM is not a very good chatbot
# 2. An example of training an LLM with instruction fine-tuning.
# 3. Demonstration that after fine tuning, the LLM is a better chatbot.

# %% [markdown]
# ## Setup
#
# Install and import necessary libraries.

# %%
from threading import Thread
import tqdm

import datasets
from etils import epath
import torch
import transformers
from examples import paths
# %% [markdown]
# ## Set up the TPU device for PyTorch
# %%
# device = torch.device("tpu")  # Workaround. Will become `torch.device("tpu")`
device = torch.device("cuda")

torch.set_default_device(device)
# %% [markdown]
# ## Load Model and Tokenizer
#

# %%
base_path = epath.Path(paths.XM_HOME)

model_path = base_path / "weights/huggingface/meta-llama/Llama-3.2-1B"
model = transformers.AutoModelForCausalLM.from_pretrained(model_path, dtype="auto")
model = model.to(device)
# %%
tok_path = (
    base_path / "weights/huggingface/meta-llama/Meta-Llama-3-8B-Instruct/"
)
tokenizer = transformers.AutoTokenizer.from_pretrained(tok_path)
tokenizer.pad_token = tokenizer.eos_token
# %% [markdown]
# ## Try to chat with a pretrained model.
#
# Trying to chat with a pretrained model before an instruct-SFT phase results
# in a non-functional "chatbot".
# %%
pipe = transformers.pipeline(
    "text-generation", model=model, tokenizer=tokenizer, do_sample=False
)
streamer = transformers.TextIteratorStreamer(tokenizer)

t = Thread(
    target=pipe,
    args=[[{"role": "user", "content": "What is the capital of Uganda?"}]],
    kwargs={"streamer": streamer, "max_new_tokens":100},
)
t.start()
for token_idx, new_text in enumerate(streamer):
  print(new_text, end="", flush=True)
  if (token_idx + 1) % 20 == 0:
    print()
t.join()
# %% [markdown]
# ## Load instruct SFT dataset
# %%
dataset_path = base_path / "datasets/huggingface/tatsu-lab/alpaca"
raw_dataset = datasets.load_from_disk(str(dataset_path))["train"]
# %% [markdown]
# ## Apply chat template and tokenize the dataset.
#
#
#
#

# %%
def format(example):
  chat = [
      {"role": "user", "content": example["instruction"]},
      {"role": "assistant", "content": example["output"]},
  ]
  text = tokenizer.apply_chat_template(chat, tokenize=False)
  inputs = tokenizer(
      text,
      padding="max_length",
      max_length=256,
      truncation=True,
  )
  return inputs


dataset = raw_dataset.map(
    format,
    load_from_cache_file=False,
    cache_file_name="/tmp/cache.arrow",  # Workaround
    remove_columns=raw_dataset.column_names
)
# %% [markdown]
# # Instruct SFT

# %%
EPOCHS = 1
BATCH_SIZE = 4

# TODO: Set labels of user content to -100.
train_dataloader = torch.utils.data.DataLoader(
    dataset.select(range(100 * BATCH_SIZE)),
    collate_fn=transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    ),
    batch_size=BATCH_SIZE,
)

val_dataloader = torch.utils.data.DataLoader(
    dataset.select(range(100 * BATCH_SIZE, 101 * BATCH_SIZE)),
    collate_fn=transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    ),
    batch_size=BATCH_SIZE,
)
val_input = next(iter(val_dataloader))
val_input = {k: v.to(device) for k, v in val_input.items()}

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(1, EPOCHS + 1):
  pbar = tqdm.tqdm(enumerate(train_dataloader), unit="batch", desc=f"Epoch {epoch}")
  for step, inputs in pbar:
    model.train()
    optimizer.zero_grad()
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    pbar.set_postfix(loss=f"{loss.item():.4f}")

    if (step + 1) % 10 == 0:
      model.eval()
      with torch.no_grad():
        outputs = model(**val_input)
      print(f"\nEpoch {epoch}, Step {step + 1}, Val loss: {outputs.loss:.4f}\n")

# %%
pipe = transformers.pipeline(
    "text-generation", model=model, tokenizer=tokenizer, do_sample=False
)
streamer = transformers.TextIteratorStreamer(tokenizer)

t = Thread(
    target=pipe,
    args=[[{"role": "user", "content": "What is the capital of Uganda?"}]],
    kwargs={"streamer": streamer, "max_new_tokens": 100},
)
t.start()
for token_idx, new_text in enumerate(streamer):
  print(new_text, end="", flush=True)
  if (token_idx + 1) % 20 == 0:
    print()
t.join()
# %% [markdown]
# ## Example of profiling
# %%
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
  for epoch in range(1, EPOCHS + 1):
    pbar = tqdm.tqdm(enumerate(train_dataloader), unit="batch", desc=f"Epoch {epoch}")
    for step, inputs in pbar:
      model.train()
      optimizer.zero_grad()
      inputs = {k: v.to(device) for k, v in inputs.items()}

      outputs = model(**inputs)
      loss = outputs.loss
      loss.backward()
      optimizer.step()
      pbar.set_postfix(loss=f"{loss.item():.4f}")

      if step >= 3:
         break

prof.export_chrome_trace("/tmp/trace.json")

# %%
import subprocess

source_path = "/tmp/trace.json"
destination_path = paths.TRACES_HOME

subprocess.run(['fileutil', 'cp', '-f', source_path, destination_path], check=True)
