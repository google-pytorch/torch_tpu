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

"""A simple implementation of DLRM.

Currently, this file only contains the model instantiation.
Weights and other configurations will come later.

///////////////////////////////////////////////////////////////
NOTE: This file/model is meant to DEBUG and TRIAGE remaining
      issues in the torch_tpu implementation. It will contain
      some experimental things and lots of TODOs to bypass
      existing problems.
//////////////////////////////////////////////////////////////
"""

from absl import app
from absl import flags
import torch
from torch import nn
from torch_tpu._internal.utils import utils

_TPU = flags.DEFINE_bool("tpu", False, "Also run on TPU.")

class DLRM(nn.Module):
  """A minimal implementation of the Deep Learning Recommendation Model."""

  def __init__(
      self,
      embedding_dim: int,
      num_dense_features: int,
      vocab_sizes: list[int],
      mlp_dims: list[int],
  ):
    super().__init__()
    self.num_dense_features = num_dense_features
    self.num_sparse_features = len(vocab_sizes)

    # Bottom MLP for dense features
    # Input: (batch_size, num_dense_features)
    # Output: (batch_size, embedding_dim)
    dense_layers = []
    input_dim = self.num_dense_features
    for dim in mlp_dims:
      dense_layers.append(nn.Linear(input_dim, dim))
      dense_layers.append(nn.ReLU())
      input_dim = dim
    dense_layers.append(nn.Linear(input_dim, embedding_dim))
    self.bottom_mlp = nn.Sequential(*dense_layers)

    # Embedding layers for sparse features
    self.embedding_layers = nn.ModuleList([
        nn.Embedding(num_embeddings=vs, embedding_dim=embedding_dim)
        for vs in vocab_sizes
    ])

    # Top MLP for combined features
    # The total number of features to interact is num_sparse + 1
    # (for the dense MLP output)
    num_total_features = self.num_sparse_features + 1
    # Number of pairwise interactions is nC2 = n * (n - 1) / 2
    num_interactions = (num_total_features * (num_total_features - 1)) // 2

    # Input to top MLP is the concatenation of interaction features
    # and the dense MLP output
    top_mlp_input_dim = num_interactions + embedding_dim

    top_layers = []
    input_dim = top_mlp_input_dim
    for dim in mlp_dims:
      top_layers.append(nn.Linear(input_dim, dim))
      top_layers.append(nn.ReLU())
      input_dim = dim
    top_layers.append(nn.Linear(input_dim, 1))  # Final output layer
    self.top_mlp = nn.Sequential(*top_layers)

  def forward(self, dense_x: torch.Tensor, sparse_x: torch.Tensor):
    # dense_x shape: (batch_size, num_dense_features)
    # sparse_x shape: (batch_size, num_sparse_features)

    # 1. Process dense features with bottom MLP
    # Output shape: (batch_size, embedding_dim)
    dense_out = self.bottom_mlp(dense_x)

    # 2. Process sparse features with embedding layers
    # This creates a list of tensors, each of shape (batch_size, embedding_dim)
    sparse_embeddings = [
        self.embedding_layers[i](sparse_x[:, i])
        for i in range(self.num_sparse_features)
    ]

    # 3. Feature Interaction
    # Combine dense and sparse features for interaction
    # We start with the dense output and add the sparse embeddings to a list
    all_features = [dense_out] + sparse_embeddings

    # Stack all features to form a tensor of shape:
    # (batch_size, num_total_features, embedding_dim)
    interaction_input = torch.stack(all_features, dim=1)

    # Perform the dot product interaction
    # `bmm` is batch matrix multiplication
    # (batch, n, d) @ (batch, d, n) -> (batch, n, n)
    dot_products = torch.bmm(
        interaction_input, interaction_input.transpose(1, 2)
    )

    # We only need the lower triangular part of the interaction
    # matrix (excluding the diagonal)
    # This extracts the unique pairs of interactions
    tril_indices = torch.tril_indices(
        row=dot_products.size(1),
        col=dot_products.size(1),
        offset=-1,
        device=dot_products.device,
    )
    interaction_features = dot_products[:, tril_indices[0], tril_indices[1]]
    # Output shape: (batch_size, num_interactions)

    # 4. Final Combination and Top MLP
    # Concatenate the dense features (post-MLP) and the interaction features
    # Input shape: (batch_size, embedding_dim + num_interactions)
    top_mlp_input = torch.cat([dense_out, interaction_features], dim=1)

    # Process through the top MLP to get the final prediction
    # Output shape: (batch_size, 1)
    logits = self.top_mlp(top_mlp_input)

    # Apply sigmoid to get a probability
    return torch.sigmoid(logits)


# pylint: disable=unused-argument
def main(argv):
  torch.manual_seed(123)
  batch_size = 16
  embedding_dim = 64  # Dimension for all feature embeddings
  num_dense_features = 10  # e.g., age, price, etc.
  # Vocabulary sizes for 5 different categorical features
  vocab_sizes = [100, 500, 20, 8000, 300]
  mlp_dims = [512, 256]  # Hidden layer dimensions for both MLPs

  # --- Create the Model ---
  model = DLRM(
      embedding_dim=embedding_dim,
      num_dense_features=num_dense_features,
      vocab_sizes=vocab_sizes,
      mlp_dims=mlp_dims,
  ).eval()

  # --- Create Dummy Input Data ---
  # Dense features (continuous values)
  dense_input = torch.randn(batch_size, num_dense_features)

  # Sparse features (categorical indices)
  # Each value must be within the range of its corresponding vocabulary size
  sparse_input = torch.cat(
      [
          torch.randint(0, vocab_size, (batch_size, 1))
          for vocab_size in vocab_sizes
      ],
      dim=1,
  )

  print("Model Input Shapes:")
  print(f"  Dense features:  {dense_input.shape}")
  print(f"  Sparse features: {sparse_input.shape}\n")

  print(utils.format_model(model, [dense_input, sparse_input], pt=True))
  # --- Forward Pass ---
  print("CPU")
  prediction = model(dense_input, sparse_input)
  print(prediction)

  print("TPU")
  tpu_device = torch.device("tpu")
  model = model.to(tpu_device)
  dense_input_tpu = dense_input.to(tpu_device)
  sparse_input_tpu = sparse_input.to(tpu_device)
  output_tpu = model(dense_input_tpu, sparse_input_tpu).to("cpu")
  print(output_tpu)

  utils.assert_close(output_tpu, prediction, rtol=1e-3, atol=4e-4)


if __name__ == "__main__":
  app.run(main)
