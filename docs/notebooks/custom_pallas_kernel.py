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

    `torch_tpu` integrates with Pallas via the `@pallas.jax_op` decorator, allowing you to write custom hardware-accelerated logic and call it directly within your PyTorch models.

    **Prerequisites:** JAX must be installed (`pip install jax`).
    """)
  return


@app.cell
def _():
  import torch
  import jax
  from jax.experimental import pallas as pl
  from torch_tpu._internal import pallas  # This will likely change from _internal to experimental

  device = torch.device("tpu")

  print(f"Pallas environment ready on {device}.")
  return device, jax, pallas, pl, torch


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 2. Writing Your First Kernel

    The `pallas.jax_op` function is the primary way to define a custom TPU kernel. By using it directly, you can bind your kernel logic to statically known output shapes.

    ### Key Concepts

    This notebook is only focused on showing how to use Pallas in the context of TorchTPU. It is not meant to be a full tutorial on Pallas. If you are new to Pallas be sure to read the For more technical details on Pallas memory management, see the [Pallas Quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html).

    - **`_ref` arguments**: In these examples, Pallas manages the transfer of data from HBM to local VMEM. These parameters act as references to that local memory.
    - **`[...]` syntax**: Since Pallas handles the tile transfers here, we use `[...]` to load the full contents of the local reference into registers for computation, or to store them back.
    - **`output_shapes`**: You provide example tensors (or shapes) to tell the PyTorch dispatcher how to allocate the output buffer statically.
    """)
  return


@app.cell
def _(jax, pallas, pl, torch):
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
  add_vectors_fn = pallas.jax_op(
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
def _(jax, pallas, pl, torch):
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

  add_subtract_fn = pallas.jax_op(
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

    Pallas kernels can be seamlessly compiled into larger graphs using `torch.compile(backend="tpu")`. Because `pallas.jax_op` automatically registers the function as a `torch.library.custom_op` and handles the `register_fake` implementation internally, you can compile the resulting op directly.
    """)
  return


@app.cell
def _(jax, pallas, pl, torch):
  from torch_tpu._internal import compile

  # 1. Define the Pallas kernel and its JAX wrapper
  def mul_kernel_fn(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] * y_ref[...]

  def mul_jax_fn(x: jax.Array, y: jax.Array) -> jax.Array:
    return pl.pallas_call(
        mul_kernel_fn, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
    )(x, y)

  mul_vectors_op = pallas.jax_op("pallas::mul_vectors", mul_jax_fn)

  # 4. Compile with the TPU backend
  @torch.compile(backend="tpu", dynamic=False, fullgraph=True)
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

    Training requires a reusable backward operator, not only a forward kernel. The example below demonstrates the complete integration pattern:

    - the forward and backward are separate Pallas kernels wrapped with `pallas.jax_op`;
    - the forward returns the result plus two auxiliary tensors needed by backward;
    - `setup_context` marks those auxiliaries non-differentiable and saves them instead of saving the original inputs;
    - `register_autograd` connects the two namespaced custom operators; and
    - `torch.compile(backend="tpu", dynamic=False, fullgraph=True)` captures the forward and AOTAutograd backward without a graph break.

    JAX array arguments and return values must be annotated. Plain-data arguments such as `negate: bool` and `block_size: int` are static specializations, so keep their values and the tensor shapes stable within a compiled graph. Register the operators before entering the function passed to `torch.compile`; lazy imports or registration during Dynamo tracing are unsupported.
    """)
  return


@app.cell
def _(device, jax, pallas, pl, torch):
  from functools import partial

  block_size = 128

  def validate_shape(x, block_size):
    if block_size <= 0 or block_size % 128:
      raise ValueError("block_size must be a positive multiple of 128")
    if x.ndim != 1 or x.shape[0] % block_size:
      raise ValueError("input must be 1-D and divisible by block_size")

  # Forward returns out plus local derivatives used by backward.
  def mul_forward_kernel(
      x_ref, y_ref, out_ref, local_dx_ref, local_dy_ref, *, negate
  ):
    sign = -1.0 if negate else 1.0
    x, y = x_ref[...], y_ref[...]
    out_ref[...] = sign * x * y
    local_dx_ref[...] = sign * y
    local_dy_ref[...] = sign * x

  def mul_forward_jax(
      x: jax.Array, y: jax.Array, negate: bool, block_size: int
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    validate_shape(x, block_size)
    if y.shape != x.shape:
      raise ValueError("x and y must have the same shape")
    spec = pl.BlockSpec((block_size,), lambda i: (i,))
    shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return pl.pallas_call(
        partial(mul_forward_kernel, negate=negate),
        out_shape=(shape, shape, shape),
        grid=(x.shape[0] // block_size,),
        in_specs=(spec, spec),
        out_specs=(spec, spec, spec),
    )(x, y)

  def mul_backward_kernel(
      grad_out_ref, local_dx_ref, local_dy_ref, grad_x_ref, grad_y_ref
  ):
    grad_out = grad_out_ref[...]
    grad_x_ref[...] = grad_out * local_dx_ref[...]
    grad_y_ref[...] = grad_out * local_dy_ref[...]

  def mul_backward_jax(
      grad_out: jax.Array,
      local_dx: jax.Array,
      local_dy: jax.Array,
      block_size: int,
  ) -> tuple[jax.Array, jax.Array]:
    validate_shape(grad_out, block_size)
    spec = pl.BlockSpec((block_size,), lambda i: (i,))
    shape = jax.ShapeDtypeStruct(grad_out.shape, grad_out.dtype)
    return pl.pallas_call(
        mul_backward_kernel,
        out_shape=(shape, shape),
        grid=(grad_out.shape[0] // block_size,),
        in_specs=(spec, spec, spec),
        out_specs=(spec, spec),
    )(grad_out, local_dx, local_dy)

  # Namespaces avoid collisions in PyTorch's process-wide operator registry.
  mul_forward_op = pallas.jax_op(
      "pallas_autograd_example::mul_forward", mul_forward_jax
  )
  mul_backward_op = pallas.jax_op(
      "pallas_autograd_example::mul_backward", mul_backward_jax
  )

  def setup_context(ctx, inputs, output):
    _, _, _, static_block_size = inputs
    _, local_dx, local_dy = output
    ctx.mark_non_differentiable(local_dx, local_dy)
    ctx.save_for_backward(local_dx, local_dy)
    ctx.block_size = static_block_size

  def backward_impl(ctx, grad_out, _grad_local_dx, _grad_local_dy):
    local_dx, local_dy = ctx.saved_tensors
    grad_x, grad_y = mul_backward_op(
        grad_out.contiguous(), local_dx, local_dy, ctx.block_size
    )
    return grad_x, grad_y, None, None

  mul_forward_op.register_autograd(
      backward_impl, setup_context=setup_context
  )

  # The public wrapper hides implementation-only auxiliary outputs.
  def mul(x, y, *, negate=False, block_size=128):
    out, _, _ = mul_forward_op(x, y, negate, block_size)
    return out

  @torch.compile(backend="tpu", dynamic=False, fullgraph=True)
  def compiled_mul_train(x, y, weight, negate: bool):
    return (mul(x, y, negate=negate, block_size=block_size) * weight).sum()

  x_host = torch.linspace(-2.0, 2.0, 256)
  y_host = torch.linspace(3.0, -1.0, 256)
  weight_host = torch.linspace(-0.5, 1.5, 256)
  x_train = x_host.to(device).requires_grad_()
  y_train = y_host.to(device).requires_grad_()
  weight = weight_host.to(device)

  loss_compiled = compiled_mul_train(x_train, y_train, weight, True)
  loss_compiled.backward()

  expected_loss = (-(x_host * y_host) * weight_host).sum()
  expected_dx = -(weight_host * y_host)
  expected_dy = -(weight_host * x_host)
  torch.testing.assert_close(
      loss_compiled.detach().cpu(), expected_loss
  )  # TORCH_ASSERT_CLOSE_OK=<Using standard PyTorch assert for user facing documentation only.>
  torch.testing.assert_close(
      x_train.grad.cpu(), expected_dx
  )  # TORCH_ASSERT_CLOSE_OK=<Using standard PyTorch assert for user facing documentation only.>
  torch.testing.assert_close(
      y_train.grad.cpu(), expected_dy
  )  # TORCH_ASSERT_CLOSE_OK=<Using standard PyTorch assert for user facing documentation only.>

  print(f"Loss:    {loss_compiled.detach().cpu().item()}")
  print(f"x grad finite: {torch.isfinite(x_train.grad).all().cpu().item()}")
  print(f"y grad finite: {torch.isfinite(y_train.grad).all().cpu().item()}")
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 5. Verification

    A quick sanity check that the Pallas integration is working correctly.
    """)
  return


@app.cell
def _(jax, pallas, pl, torch):
  def double_kernel(i_ref, o_ref):
    o_ref[...] = i_ref[...] * 2.0

  test_input = torch.tensor([10.0, 20.0, 30.0], device="tpu")

  def double_jax_fn(x: jax.Array) -> jax.Array:
    return pl.pallas_call(
        double_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
    )(x)

  double_fn = pallas.jax_op(
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
  print("\n? Verification complete: Pallas integration functional")
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ### ??? Summary

    | Concept | Key Takeaway |
    | :--- | :--- |
    | **`pallas.jax_op`** | Decorator to define a TPU kernel as a PyTorch custom operation by wrapping a JAX function |
    | **`pl.pallas_call`** | Used to wrap a Pallas kernel function into a JAX function |
    | **`_ref[...]` syntax** | Used in these examples to read/write local memory references managed by Pallas |
    | **Multiple outputs** | Provide a list of `jax.ShapeDtypeStruct` to `out_shape` in `pl.pallas_call`. |
    | **Static arguments** | Annotate plain-data arguments such as `bool` and `int`; they specialize the compiled graph. |
    | **Autograd integration** | Register a separately wrapped Pallas backward on the forward `CustomOpDef`, save only required tensors, and mark auxiliary outputs non-differentiable. |
    | **`torch.compile`** | Register operators before tracing, then use `backend="tpu", dynamic=False, fullgraph=True` for the demonstrated contract. |

    `torch_tpu._internal.pallas` is a preview/internal API and may move or change before a public release.
    """)
  return


if __name__ == "__main__":
  app.run()
