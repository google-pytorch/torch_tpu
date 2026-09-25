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
# pylint: skip-file

import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
  import marimo as mo

  return (mo,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    # Custom Kernels with Pallas

    Pallas allows you to write custom TPU kernels in Python. This is essential for implementing new research ideas or where you need more fine-grained control over the code executed on the TPU hardware.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 1. What is Pallas?

    [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) is a JAX-based kernel language for TPUs. It provides direct control over the TPU's memory hierarchy and vector units while remaining in pure Python.

    `torch.tpu` integrates with Pallas via `torch.tpu.pallas.jax_op`, allowing you to write custom hardware-accelerated logic and call it directly within your PyTorch models.

    **Prerequisites:** JAX must be installed (`pip install jax`).
    """)
  return


@app.cell
def _():
  import jax
  from jax.experimental import pallas as pl
  import torch

  device = torch.device("tpu")

  print(f"Pallas environment ready on {device}.")
  return device, jax, pl, torch


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 2. Writing Your First Kernel

    The `torch.tpu.pallas.jax_op` function is the primary way to define a custom TPU kernel. By using it directly, you can bind your kernel logic to statically known output shapes.

    ### Key Concepts

    This notebook is only focused on showing how to use Pallas in the context of TorchTPU. It is not meant to be a full tutorial on Pallas. If you are new to Pallas be sure to read the For more technical details on Pallas memory management, see the [Pallas Quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html).

    - **`_ref` arguments**: In these examples, Pallas manages the transfer of data from HBM to local VMEM. These parameters act as references to that local memory.
    - **`[...]` syntax**: Since Pallas handles the tile transfers here, we use `[...]` to load the full contents of the local reference into registers for computation, or to store them back.
    - **`output_shapes`**: You provide example tensors (or shapes) to tell the PyTorch dispatcher how to allocate the output buffer statically.
    """)
  return


@app.cell
def _(jax, pl, torch):
  # Define a simple vector addition kernel
  def add_vectors_kernel(x_ref, y_ref, o_ref):
    """x_ref, y_ref: Input references

    o_ref: Output reference to be populated
    """
    x, y = x_ref[...], y_ref[...]
    o_ref[...] = x + y

  x = torch.tensor([1.0, 2.0, 3.0], device="tpu")
  y = torch.tensor([4.0, 5.0, 6.0], device="tpu")

  # NEW WAY: Wrap the Pallas kernel in a JAX function
  def add_vectors_jax(x: jax.Array, y: jax.Array) -> jax.Array:
    return pl.pallas_call(
        add_vectors_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
    )(x, y)

  # Wrap it using the functional interface
  add_vectors_fn = torch.tpu.pallas.jax_op(
      "pallas::add_vectors",
      add_vectors_jax,
  )

  # Use it like a standard PyTorch function
  z = add_vectors_fn(x, y)

  print(f"Input x:  {x.cpu()}")
  print(f"Input y:  {y.cpu()}")
  print(f"Result z: {z.cpu()}")
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ### 2.1 Multiple Outputs

    If your kernel produces multiple results, simply pass a list of multiple example tensors to the `output_shapes` argument.
    """)
  return


@app.cell
def _(jax, pl, torch):
  def add_subtract_kernel(x_ref, y_ref, out_add_ref, out_sub_ref):
    x, y = x_ref[...], y_ref[...]
    out_add_ref[...] = x + y
    out_sub_ref[...] = x - y

  a = torch.tensor([10.0, 20.0, 30.0], device="tpu")
  b = torch.tensor([1.0, 2.0, 3.0], device="tpu")

  def add_subtract_jax(
      x: jax.Array, y: jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    return pl.pallas_call(
        add_subtract_kernel,
        out_shape=[
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            jax.ShapeDtypeStruct(x.shape, x.dtype),
        ],
    )(x, y)

  add_subtract_fn = torch.tpu.pallas.jax_op(
      "pallas::add_subtract",
      add_subtract_jax,
  )

  res_add, res_sub = add_subtract_fn(a, b)

  print(f"Input a:    {a.cpu()}")
  print(f"Input b:    {b.cpu()}")
  print(f"a + b:      {res_add.cpu()}")
  print(f"a - b:      {res_sub.cpu()}")
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 3. Integration: `torch.compile`

    Pallas kernels can be seamlessly compiled into larger graphs using `torch.compile(backend="tpu")`. Because `torch.tpu.pallas.jax_op` automatically registers the function as a `torch.library.custom_op` and handles the `register_fake` implementation internally, you can compile the resulting op directly.
    """)
  return


@app.cell
def _(jax, pl, torch):
  # 1. Define the Pallas kernel and its JAX wrapper
  def mul_kernel_fn(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] * y_ref[...]

  def mul_jax_fn(x: jax.Array, y: jax.Array) -> jax.Array:
    return pl.pallas_call(
        mul_kernel_fn, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
    )(x, y)

  mul_vectors_op = torch.tpu.pallas.jax_op("pallas::mul_vectors", mul_jax_fn)

  # 4. Compile with the TPU backend
  @torch.compile(backend="tpu")
  def compiled_mul(x, y):
    return mul_vectors_op(x, y)

  # --- Run it ---
  x_inf = torch.tensor([1.0, 2.0, 3.0], device="tpu")
  y_inf = torch.tensor([4.0, 5.0, 6.0], device="tpu")

  res_compiled = compiled_mul(x_inf, y_inf)

  print(f"Input x:  {x_inf.cpu()}")
  print(f"Input y:  {y_inf.cpu()}")
  print(f"Result:   {res_compiled.cpu()}")
  return (mul_vectors_op,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 4. Integration: Autograd

    If you are using custom kernels in a training loop, you must manually define the backward pass and register it with the forward op using `register_autograd`.

    Both the forward and backward passes should be defined using `torch.tpu.pallas.jax_op`. This ensures they are registered as custom ops with their own `register_fake` implementations, which is required because AOTAutograd traces both the forward and backward graphs.
    """)
  return


@app.cell
def _(device, jax, mul_vectors_op, pl, torch):
  # 1. Define the backward Pallas kernel and its JAX wrapper
  def mul_backward_kernel_fn(g_ref, x_ref, y_ref, g_x_ref, g_y_ref):
    g, x, y = g_ref[...], x_ref[...], y_ref[...]
    g_x_ref[...] = g * y
    g_y_ref[...] = g * x

  def mul_backward_jax_fn(
      grad: jax.Array, x: jax.Array, y: jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    return pl.pallas_call(
        mul_backward_kernel_fn,
        out_shape=[
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            jax.ShapeDtypeStruct(y.shape, y.dtype),
        ],
    )(grad, x, y)

  mul_vectors_backward_op = torch.tpu.pallas.jax_op(
      "pallas::mul_vectors_backward", mul_backward_jax_fn
  )

  # 3. Register fake for the backward op
  mul_vectors_backward_op.register_fake(
      lambda g, x, y: (torch.empty_like(x), torch.empty_like(y))
  )

  # 4. Connect forward and backward via autograd
  def setup_context(ctx, inputs, output):
    x, y = inputs
    ctx.save_for_backward(x, y)

  def backward_impl(ctx, grad):
    x, y = ctx.saved_tensors
    return mul_vectors_backward_op(grad, x, y)

  mul_vectors_op.register_autograd(backward_impl, setup_context=setup_context)

  # --- Run Training Loop ---
  @torch.compile(dynamic=False, backend="tpu")
  def compiled_mul_train(x, y):
    return mul_vectors_op(x, y).sum()

  x_train = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
  y_train = torch.tensor([4.0, 5.0, 6.0], device=device, requires_grad=True)

  loss_compiled = compiled_mul_train(x_train, y_train)
  loss_compiled.backward()

  print(f"Loss:    {loss_compiled.detach().cpu().item()}")
  print(f"x grad:  {x_train.grad.cpu()}")
  print(f"y grad:  {y_train.grad.cpu()}")
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 5. Verification

    A quick sanity check that the Pallas integration is working correctly.
    """)
  return


@app.cell
def _(jax, pl, torch):
  def double_kernel(i_ref, o_ref):
    o_ref[...] = i_ref[...] * 2.0

  test_input = torch.tensor([10.0, 20.0, 30.0], device="tpu")

  def double_jax_fn(x: jax.Array) -> jax.Array:
    return pl.pallas_call(
        double_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
    )(x)

  double_fn = torch.tpu.pallas.jax_op(
      "pallas::double",
      double_jax_fn,
  )

  test_output = double_fn(test_input)

  expected = torch.tensor([20.0, 40.0, 60.0])
  torch.testing.assert_close(
      test_output.cpu(), expected
  )  # TORCH_ASSERT_CLOSE_OK=<Using standard PyTorch assert for user facing documentation only.>

  print(f"Input:    {test_input.cpu()}")
  print(f"Output:   {test_output.cpu()}")
  print(f"Expected: {expected}")
  print("\n✅ Verification complete: Pallas integration functional")
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ### 🛠️ Summary

    | Concept | Key Takeaway |
    | :--- | :--- |
    | **`torch.tpu.pallas.jax_op`** | Decorator to define a TPU kernel as a PyTorch custom operation by wrapping a JAX function |
    | **`pl.pallas_call`** | Used to wrap a Pallas kernel function into a JAX function |
    | **`_ref[...]` syntax** | Used in these examples to read/write local memory references managed by Pallas |
    | **Multiple outputs** | Provide a list of `jax.ShapeDtypeStruct` to `out_shape` in `pl.pallas_call`. |
    | **Autograd integration** | Register a backward kernel on the forward `CustomOpDef` returned by `jax_op` using `register_autograd` |
    | **`torch.compile`** | Use `backend="tpu"` for full graph fusion with custom kernels |
    """)
  return


if __name__ == "__main__":
  app.run()
