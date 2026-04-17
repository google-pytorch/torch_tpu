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

"""Example of fine-tuning a model with PEFT LoRA on TPU."""

from absl import flags
from absl import logging
from absl.testing import absltest
import datasets
from etils import epath
import peft
import torch
from torch.utils.data import DataLoader
from torch_tpu._internal.utils import log_utils
from examples import paths
import transformers

log_utils.log_to_stderr()


FLAGS = flags.FLAGS
BASE_PATH = epath.Path(paths.XM_HOME)
MODEL_PATH = BASE_PATH / "weights/huggingface/meta-llama/Llama-3.2-1B"
TOK_PATH = (
    BASE_PATH / "weights/huggingface/meta-llama/Meta-Llama-3-8B-Instruct/"
)
flags.DEFINE_integer("batch_size", 4, "Training batch size.")
flags.DEFINE_integer("num_epochs", 1, "Number of training epochs.")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate.")


class PeftLoraTrainingTest(absltest.TestCase):

  def _create_training_setup(self):
    device = torch.device("tpu")
    logging.info("Using TPU device: %s", device)

    dataset_path = BASE_PATH / "datasets/huggingface/tatsu-lab/alpaca"
    peft_save_path = epath.Path(self.create_tempdir().full_path)

    cache_file = self.create_tempfile("cache.arrow")
    tokenizer = transformers.AutoTokenizer.from_pretrained(TOK_PATH)
    if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token
    logging.info("Tokenizer pad token ID: %d", tokenizer.pad_token_id)

    raw_dataset = datasets.load_from_disk(str(dataset_path))["train"]

    def format_example(example):
      chat = [
          {"role": "user", "content": example["instruction"]},
          {"role": "assistant", "content": example["output"]},
      ]
      text = tokenizer.apply_chat_template(chat, tokenize=False)
      return tokenizer(
          text,
          padding="max_length",
          max_length=256,
          truncation=True,
          return_tensors=None,
      )

    dataset = raw_dataset.map(
        format_example,
        load_from_cache_file=False,
        cache_file_name=cache_file.full_path,
        remove_columns=["instruction", "input", "output", "text"],
    )

    train_dataloader = DataLoader(
        dataset,
        shuffle=False,
        collate_fn=transformers.DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        ),
        batch_size=FLAGS.batch_size,
    )

    # Apply LoRA
    peft_config = peft.LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype="auto"
    )
    model = peft.get_peft_model(model, peft_config)
    # Log number of trainable parameters
    trainable_params, all_param = model.get_nb_trainable_parameters()
    logging.info(
        "trainable params: %s || all params: %s || trainable%%: %f",
        f"{trainable_params:,}",
        f"{all_param:,}",
        100 * trainable_params / all_param,
    )
    model = model.to(device)
    return device, model, tokenizer, train_dataloader, peft_save_path

  def test_peft_lora_training(self):
    device, model, tokenizer, train_dataloader, peft_save_path = (
        self._create_training_setup()
    )

    with self.subTest("check_only_lora_weights_are_trainable"):
      # Check that only LoRA parameters are trainable
      trainable_params = {
          name
          for name, param in model.named_parameters()
          if param.requires_grad
      }

      # Check for and report all trainable non-LoRA parameters.
      non_lora_trainable = {
          name for name in trainable_params if "lora" not in name
      }
      self.assertEmpty(
          non_lora_trainable,
          f"Found trainable non-LoRA parameters: {list(non_lora_trainable)}",
      )
      self.assertNotEmpty(
          trainable_params, "No trainable LoRA parameters found."
      )

    with self.subTest("train_and_check_parameter_updates"):
      initial_params = {n: p.clone().cpu() for n, p in model.named_parameters()}

      # Train model
      optimizer = torch.optim.AdamW(model.parameters(), lr=FLAGS.learning_rate)
      logging.info("Starting training...")
      model.train()
      for epoch in range(FLAGS.num_epochs):
        logging.info("Starting Epoch %d", epoch)
        for step, batch in enumerate(train_dataloader):
          batch_on_device = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch_on_device)
          loss = outputs.loss
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()

          if step % 10 == 0:
            logging.info(
                "Epoch %d, Step %d, Loss: %f", epoch, step, loss.item()
            )
          # Run only one step to verify params change and keep test runtime short.
          break
        logging.info("Finished Epoch %d", epoch)
      logging.info("Training finished.")

      # Assert that base parameters are unchanged and LoRA parameters have changed
      lora_changed = False
      for name, param in model.named_parameters():
        initial_p = initial_params[name]
        current_p = param.cpu()
        if param.requires_grad:
          if not torch.equal(initial_p, current_p):
            lora_changed = True
        else:
          self.assertTrue(
              torch.equal(initial_p, current_p),
              f"Base parameter {name} changed during training",
          )
      self.assertTrue(
          lora_changed, "LoRA parameters did not change during training"
      )

    with self.subTest("save_and_load_model"):
      # Save model
      logging.info("Saving model...")
      model.save_pretrained(peft_save_path)
      tokenizer.save_pretrained(peft_save_path)
      logging.info("Model saved to %s", peft_save_path)

      # Load model
      logging.info("Loading model from %s...", peft_save_path)
      base_model_for_loading = (
          transformers.AutoModelForCausalLM.from_pretrained(
              MODEL_PATH, dtype="auto"
          )
      )
      loaded_model = peft.PeftModel.from_pretrained(
          base_model_for_loading, str(peft_save_path)
      )

      # Assert loaded state dict matches saved state dict
      model_sd = model.state_dict()
      loaded_sd = loaded_model.state_dict()
      self.assertEqual(model_sd.keys(), loaded_sd.keys())
      for name, param in model_sd.items():
        self.assertTrue(
            torch.equal(param.cpu(), loaded_sd[name].cpu()),
            f"Loaded parameter {name} does not match saved",
        )


if __name__ == "__main__":
  absltest.main()
