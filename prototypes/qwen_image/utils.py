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

import glob
import json
import os
import datasets
import pandas as pd
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset


class GenericImageCaptionDataset(Dataset):
  """Extensible PyTorch Dataset that loads images and captions from: 1.

  A folder containing images and corresponding .txt caption files. 2. A folder
  containing metadata.jsonl or metadata.csv. 3. A list of dict items [{'image':
  PIL.Image, 'text': str}].
  """

  def __init__(
      self,
      data_source,
      image_column="image",
      caption_column="text",
      trigger_phrase="",
      transform=None,
  ):
    self.items = []
    self.transform = transform
    self.trigger_phrase = trigger_phrase.strip()

    if isinstance(data_source, str) and os.path.isdir(data_source):
      jsonl_path = os.path.join(data_source, "metadata.jsonl")
      csv_path = os.path.join(data_source, "metadata.csv")

      if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
          for line in f:
            entry = json.loads(line.strip())
            img_p = os.path.join(data_source, entry[image_column])
            caption = entry.get(caption_column, "")
            self.items.append({"image_path": img_p, "caption": caption})
      elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
          img_p = os.path.join(data_source, str(row[image_column]))
          caption = str(row.get(caption_column, ""))
          self.items.append({"image_path": img_p, "caption": caption})
      else:
        exts = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
        img_paths = []
        for ext in exts:
          img_paths.extend(glob.glob(os.path.join(data_source, ext)))
        for img_p in sorted(img_paths):
          base, _ = os.path.splitext(img_p)
          txt_p = base + ".txt"
          caption = ""
          if os.path.exists(txt_p):
            with open(txt_p, "r", encoding="utf-8") as tf:
              caption = tf.read().strip()
          self.items.append({"image_path": img_p, "caption": caption})
    elif isinstance(data_source, list):
      for entry in data_source:
        self.items.append(entry)
    else:
      raise ValueError(
          f"Unsupported data source type or directory not found: {data_source}"
      )

  def __len__(self):
    return len(self.items)

  def __getitem__(self, idx):
    item = self.items[idx]
    if "image_path" in item:
      image = Image.open(item["image_path"]).convert("RGB")
    elif "image" in item and isinstance(item["image"], Image.Image):
      image = item["image"].convert("RGB")
    else:
      image = item["image"].convert("RGB")

    caption = item.get("caption", item.get("text", ""))
    if self.trigger_phrase:
      caption = f"{self.trigger_phrase} {caption}".strip()

    if self.transform:
      image = self.transform(image)

    return {"image": image, "caption": caption}


def load_extensible_dataset(
    dataset_name_or_path,
    image_column="image",
    caption_column="text",
    trigger_phrase="",
    split="train",
):
  """Universal dataset loader function.

  Loads from Hugging Face Hub or Local Directory.
  """
  if os.path.isdir(dataset_name_or_path):
    print(
        "[Utils] Loading custom local dataset from directory:"
        f" {dataset_name_or_path}"
    )
    return GenericImageCaptionDataset(
        data_source=dataset_name_or_path,
        image_column=image_column,
        caption_column=caption_column,
        trigger_phrase=trigger_phrase,
    )
  else:
    print(
        f"[Utils] Loading dataset from Hugging Face Hub: {dataset_name_or_path}"
        f" (split='{split}')"
    )
    hf_dataset = datasets.load_dataset(dataset_name_or_path, split=split)

    items = []
    for row in hf_dataset:
      img = row[image_column]
      caption = row[caption_column]
      if trigger_phrase:
        caption = f"{trigger_phrase} {caption}".strip()
      items.append({"image": img, "caption": caption})

    return GenericImageCaptionDataset(
        data_source=items,
        image_column=image_column,
        caption_column=caption_column,
        trigger_phrase=trigger_phrase,
    )


def convert_peft_to_diffusers_adapter(adapter_dir):
  """Normalizes raw PEFT LoRA adapter keys into clean Diffusers transformer format

  and exports 'clean_diffusers_adapter.safetensors' for direct weight matrix
  injection.
  """
  peft_file = os.path.join(adapter_dir, "adapter_model.safetensors")
  if not os.path.exists(peft_file):
    print(f"[Utils] Warning: {peft_file} does not exist. Skipping conversion.")
    return None

  module_map = {
      "to_q": "attn.to_q",
      "to_k": "attn.to_k",
      "to_v": "attn.to_v",
      "to_out": "attn.to_out.0",
      "add_q_proj": "attn.add_q_proj",
      "add_k_proj": "attn.add_k_proj",
      "add_v_proj": "attn.add_v_proj",
      "to_add_out": "attn.to_add_out",
      "img_mlp_net_2": "img_mlp.net.2",
  }

  peft_weights = load_file(peft_file)
  clean_peft_weights = {}
  diffusers_weights = {}

  for k, v in peft_weights.items():
    if k.startswith("transformer.base_model.model."):
      peft_key = k.replace("transformer.base_model.model.", "base_model.model.")
    elif k.startswith("base_model.model."):
      peft_key = k
    else:
      peft_key = "base_model.model." + k

    peft_key = peft_key.replace(
        "transformer_blocks.blocks.", "transformer_blocks."
    )
    for src_mod, dst_mod in module_map.items():
      peft_key = peft_key.replace(
          f".{src_mod}_lora_A.", f".{dst_mod}.lora_A."
      ).replace(f".{src_mod}_lora_B.", f".{dst_mod}.lora_B.")

    peft_key = peft_key.replace(".kernel", "")
    if not peft_key.endswith(".weight"):
      peft_key += ".weight"

    clean_peft_weights[peft_key] = v

    if "lora" in peft_key:
      diff_key = peft_key.replace("base_model.model.", "transformer.").replace(
          ".default.weight", ".weight"
      )
      if not diff_key.endswith(".weight"):
        diff_key += ".weight"
      diffusers_weights[diff_key] = v

  save_file(clean_peft_weights, peft_file)
  diff_file = os.path.join(adapter_dir, "clean_diffusers_adapter.safetensors")
  save_file(diffusers_weights, diff_file)
  print(
      f"[Utils] Cleaned and exported {len(diffusers_weights)} tensors to:"
      f" {diff_file}"
  )
  return diff_file
