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

"""Pallas kernel and JAX wrapper for tensor addition."""

import jax
from jax.experimental import pallas as pl


def add_vectors_kernel(x_ref, y_ref, o_ref):
  """Pallas kernel body that adds two tensors in TPU memory."""
  o_ref[...] = x_ref[...] + y_ref[...]


def add_vectors_jax(x: jax.Array, y: jax.Array) -> jax.Array:
  """JAX wrapper calling the Pallas add_vectors kernel via pallas_call."""
  out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
  return pl.pallas_call(add_vectors_kernel, out_shape=out_shape)(x, y)
