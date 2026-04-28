"""The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

From the original minGPT implementation:
https://github.com/karpathy/minGPT/blob/master/demo.ipynb
"""

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

# Note: This file has two licenses because it is a derivative work of an
# MIT-licensed file. We have left the original license notice intact and added
# an Apache license header to cover modifications.

from collections.abc import Sequence
import pickle

from absl import app
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch_tpu._internal import execution_mode
from examples.mingpt.impl.mingpt import GPT
# create a Trainer object
from examples.mingpt.impl.trainer import Trainer
from examples.mingpt.impl.utils import set_seed


class SortDataset(Dataset):
  """Dataset for the Sort problem. E.g.

  for problem length 6: Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2 Which will
  feed into the transformer concatenated as: input:  0 0 2 1 0 1 0 0 0 1 1
  output: I I I I I 0 0 0 1 1 2 where I is "ignore", as the transformer is
  reading the input sequence
  """

  def __init__(self, split, length=6, num_digits=3):
    assert split in {'train', 'test'}
    self.split = split
    self.length = length
    self.num_digits = num_digits

  def __len__(self):
    return 10000  # ...

  def get_vocab_size(self):
    return self.num_digits

  def get_block_size(self):
    # the length of the sequence that will feed into transformer,
    # containing concatenated input and the output, but -1 because
    # the transformer starts making predictions at the last input element
    return self.length * 2 - 1

  def __getitem__(self, idx):
    # use rejection sampling to generate an input example from the desired split
    while True:
      # generate some random integers
      inp = torch.randint(
          self.num_digits, size=(self.length,), dtype=torch.long
      )
      # half of the time let's try to boost the number of examples that
      # have a large number of repeats, as this is what the model seems to struggle
      # with later in training, and they are kind of rate
      if torch.rand(1).item() < 0.5:
        if inp.unique().nelement() > self.length // 2:
          # too many unique digits, re-sample
          continue
      # figure out if this generated example is train or test based on its hash
      h = hash(pickle.dumps(inp.tolist()))
      inp_split = (
          'test' if h % 4 == 0 else 'train'
      )  # designate 25% of examples as test
      if inp_split == self.split:
        break  # ok

    # solve the task: i.e. sort
    sol = torch.sort(inp)[0]

    # concatenate the problem specification and the solution
    cat = torch.cat((inp, sol), dim=0)

    # the inputs to the transformer will be the offset sequence
    x = cat[:-1].clone()
    y = cat[1:].clone()
    # we only want to predict at output locations, mask out the loss at the input locations
    y[: self.length - 1] = -1
    return x, y


def main(_: Sequence[str]) -> None:
  set_seed(3407)
  model_config = GPT.get_default_config()

  train_dataset = SortDataset('train')
  x, y = train_dataset[0]
  for a, b in zip(x, y):
    print(int(a), int(b))

  model_config.model_type = 'gpt-nano'
  model_config.vocab_size = train_dataset.get_vocab_size()
  model_config.block_size = train_dataset.get_block_size()

  device = torch.accelerator.current_accelerator()

  model = GPT(model_config, device=device)

  train_config = Trainer.get_default_config()
  train_config.learning_rate = (
      5e-4  # the model we're using is so small that we can go a bit faster
  )
  train_config.max_iters = 3
  train_config.num_workers = 0
  trainer = Trainer(train_config, model, train_dataset, device)

  def batch_end_callback(trainer):
    if trainer.iter_num % 2 == 0:
      print(
          f'iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}:'
          f' train loss {trainer.loss.item():.5f}'
      )

  trainer.set_callback('on_batch_end', batch_end_callback)
  # TODO: Remove callback mode once we implement all the missing ops.
  with execution_mode.cpu_fallback_mode(enabled=True):
    trainer.run()


if __name__ == '__main__':
  app.run(main)
