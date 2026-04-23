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

"""Simple torch.nn.Embedding and EmbeddingBag in sparse model."""

from absl import app
from absl import flags
import torch
from torch import nn
from torch_tpu._internal.utils import utils

_TPU = flags.DEFINE_bool("tpu", False, "Also run on TPU.")
_DIM = flags.DEFINE_integer("dim", 128, "Embeddings Dimension.")
_VOCAB_SIZE = flags.DEFINE_integer("vocab_size", 100000, "Vocabulary Size.")
_BSZ = flags.DEFINE_integer("bsz", 256, "batch size.")
_SEQ_LEN = flags.DEFINE_integer("seq_len", 128, "sequence length.")


class EmbLayerDense(nn.Module):

  def __init__(self, vocab_size: int, emb_dim: int):
    super().__init__()
    self.eps = 1e-5
    self.embedding = nn.Embedding(vocab_size, emb_dim, sparse=False)

  def forward(self, x: torch.Tensor):
    return self.embedding(x)


class EmbLayerSparse(nn.Module):

  def __init__(self, vocab_size: int, emb_dim: int):
    super().__init__()
    self.eps = 1e-5
    self.embedding = nn.Embedding(vocab_size, emb_dim, sparse=True)

  def forward(self, x: torch.Tensor):
    return self.embedding(x)


def main(argv):
  emb_dim = _DIM.value
  print(f"Embbedding layers with inputs of size {_VOCAB_SIZE.value, emb_dim}")

  input_tensor = torch.randint(
      0, _VOCAB_SIZE.value, (_BSZ.value, _SEQ_LEN.value), dtype=torch.int32
  )
  dense_model = EmbLayerDense(_VOCAB_SIZE.value, emb_dim).eval()
  dense_output = dense_model(input_tensor)
  sparse_model = EmbLayerSparse(_VOCAB_SIZE.value, emb_dim).eval()
  sparse_model.embedding.weight = dense_model.embedding.weight
  sparse_output = sparse_model(input_tensor)

  utils.assert_close(
      actual=dense_output,
      expected=sparse_output,
      rtol=1e-3,
      atol=1e-5,
      preamble="Comparing sparse and dense results on CPU",
  )

  if _TPU.value:
    try:
      print("Evaluating on TPU vs CPU ...")
      device = torch.device("tpu")
      sparse_model.to(device)
      output_tpu = sparse_model(input_tensor.to(device)).to("cpu")
      utils.assert_close(
          actual=output_tpu,
          expected=sparse_output,
          rtol=1e-3,
          atol=1e-5,
          preamble="Sparse results",
      )
      utils.assert_close(
          actual=output_tpu,
          expected=dense_output,
          rtol=1e-3,
          atol=1e-5,
          preamble="Dense results",
      )
    except Exception as e:
      print("TPU model not fully supported yet.", e)


if __name__ == "__main__":
  app.run(main)
