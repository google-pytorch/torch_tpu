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

"""TorchAx stepper wrapper and conversion utilities for benchmarking."""

from collections.abc import Callable
import functools
import inspect
from typing import Any, Tuple

import jax
import optax
import torch
from perflab import compile as compile_lib
from perflab import context as context_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab.torchax import optimizer as optimizer_lib
import torchax
from torchax import interop  # pylint: disable=unused-import # noqa: F401
from torchax import train  # pylint: disable=unused-import # noqa: F401


def _prepare_inputs_for_torchax(inputs: Any) -> Any:
  """Recursively converts all torch tensors in a PyTree to 'jax' device."""
  default_jax_device = jax.devices()[0]
  with jax.default_device(default_jax_device):

    def _move_tensor(x):
      if isinstance(x, torch.Tensor):
        res = x.to("jax")
        assert isinstance(res, torch.Tensor) and isinstance(
            res, torchax.tensor.Tensor
        )
        return res
      return x

    inputs_jax = jax.tree_util.tree_map(_move_tensor, inputs)
  return inputs_jax


# TODO(b/568763915): Remove once `JittableModule` stops collecting property
# values as buffers.
def _drop_property_buffers(
    model: torch.nn.Module, buffers: dict[str, Any]
) -> dict[str, Any]:
  """Returns `buffers` without the tensors that are computed by properties.

  `JittableModule` collects buffers by calling `getattr()` on every name in
  `dir(module)`, so it also records tensors returned by properties. For
  example, `torch.nn.utils.parametrizations.weight_norm` (used by the
  positional convolution of Wav2Vec2 and HuBERT) turns `weight` into a property
  that recomputes it from `parametrizations.weight.original0` and `original1`.
  These tensors are derived values, not module state. Passing them to
  `functional_call()` as buffers makes `_reparametrize_module()` assign to the
  property, which for a parametrization calls `right_inverse()` and
  `Tensor.set_()`, an op that TorchAX does not implement. Leaving them out lets
  the property recompute them from the swapped-in parametrization inputs.

  Args:
    model: The module that `buffers` was extracted from.
    buffers: Buffers keyed by fully qualified attribute name.

  Returns:
    A new dict with every entry that names a property on its owning module's
    class removed.
  """
  result = {}
  for name, value in buffers.items():
    module_path, _, attr_name = name.rpartition(".")
    owner = model.get_submodule(module_path)
    if not isinstance(
        inspect.getattr_static(type(owner), attr_name, None), property
    ):
      result[name] = value
  return result


def _extract_jax_compatible_output(out: Any) -> Any:
  """Extracts a plain PyTree of tensors from model outputs.

  JAX functional transformations (e.g., jax.jit, jax.grad) require function
  return values to be standard JAX PyTrees (tuples, dicts, or Tensors).
  Custom output containers (such as HuggingFace ModelOutput dataclasses) cannot
  be traced natively by JAX. This helper extracts raw tensors or plain
  containers (tuples/dicts) from model outputs.
  """
  if out is None:
    return None
  if isinstance(out, (torch.Tensor, torchax.tensor.Tensor)):
    return out
  # Check `loss` before `logits` because training outputs (such as HuggingFace
  # CausalLMOutputWithPast when `labels` are provided) populate both `.loss` and
  # `.logits`. Extracting `.logits` first would discard the computed loss and
  # cause `_default_loss_fn` to fall back to `logits.mean()`.
  if hasattr(out, "loss") and out.loss is not None:
    return _extract_jax_compatible_output(out.loss)
  if hasattr(out, "logits") and out.logits is not None:
    return _extract_jax_compatible_output(out.logits)
  if hasattr(out, "to_tuple"):
    res = tuple(
        _extract_jax_compatible_output(x)
        for x in out.to_tuple()
        if x is not None
    )
    res = tuple(x for x in res if x is not None)
    return res if len(res) > 1 else (res[0] if len(res) == 1 else None)
  if isinstance(out, (list, tuple)):
    res = tuple(_extract_jax_compatible_output(x) for x in out if x is not None)
    res = tuple(x for x in res if x is not None)
    return res if len(res) > 1 else (res[0] if len(res) == 1 else None)
  if isinstance(out, dict):
    res = {
        k: _extract_jax_compatible_output(v)
        for k, v in out.items()
        if v is not None
    }
    return {k: v for k, v in res.items() if v is not None}
  return None


def _call_functional_model(model_jittable, params, buffers, args, kwargs):
  """Calls functional_call on JittableModule with dynamically unpacked inputs."""
  if isinstance(args, dict):
    call_args = ()
    call_kwargs = args
  elif isinstance(args, (list, tuple)):
    call_args = tuple(args)
    call_kwargs = dict(kwargs) if kwargs else {}
  else:
    call_args = (args,) if args is not None else ()
    call_kwargs = dict(kwargs) if kwargs else {}

  out = model_jittable.functional_call(
      "forward", params, buffers, *call_args, **call_kwargs
  )
  return _extract_jax_compatible_output(out)


def _default_loss_fn(outputs: Any) -> torch.Tensor:
  """Top-level default loss evaluator for training step function."""
  if hasattr(outputs, "loss") and outputs.loss is not None:
    loss = outputs.loss
  elif isinstance(outputs, dict) and "loss" in outputs:
    loss = outputs["loss"]
  elif (
      isinstance(outputs, (list, tuple))
      and outputs
      and torch.is_tensor(outputs[0])
  ):
    loss = outputs[0]
  elif torch.is_tensor(outputs):
    loss = outputs
  else:
    raise ValueError(
        "Could not extract a valid loss tensor from model output:"
        f" {type(outputs)}"
    )

  if not isinstance(loss, torch.Tensor):
    raise TypeError(
        f"TorchAx stepper wrapper loss is {type(loss).__name__}, not a Tensor"
    )

  if loss.ndim == 0:
    return loss
  else:
    return loss.mean()


def _train_step_fn(
    model_jittable: Any,
    optax_optimizer: Any,
    weights: Any,
    buffers: Any,
    opt_state: Any,
    args: Any,
    kwargs: Any,
    compute_loss: Callable[..., torch.Tensor] | None = None,
) -> Tuple[Any, Any, Any]:
  def loss_func(params):
    if compute_loss is not None:
      # The loss must be computed from the traced `params`, or it would not
      # depend on them and every gradient would be zero. `functional_call`
      # swaps `params` and `buffers` into the original module and calls
      # `compute_loss(module, args, kwargs)`, the same contract as in the
      # TorchTPU training steppers.
      return model_jittable.functional_call(
          compute_loss, params, buffers, args, kwargs
      )
    out = _call_functional_model(model_jittable, params, buffers, args, kwargs)
    return _default_loss_fn(out)

  grad_fn = torchax.interop.jax_value_and_grad(loss_func)
  loss_val, grads = grad_fn(weights)
  opt_res = torchax.interop.call_jax(
      optax_optimizer.update, grads, opt_state, weights
  )
  updates, new_opt_state = opt_res  # pyrefly: ignore[not-iterable]
  new_weights = torchax.interop.call_jax(optax.apply_updates, weights, updates)
  return loss_val, new_weights, new_opt_state


class _TorchaxDeviceKindProxy:
  """Proxy for target_lib.DeviceKind that returns 'cpu' for .value so model/sample factories use cpu."""

  def __init__(self, orig_device_kind):
    self._orig = orig_device_kind
    self.value = "cpu"

  def __eq__(self, other):
    return self._orig == other

  def __getattr__(self, name):
    return getattr(self._orig, name)


class _TorchaxFactoryContext:
  """Context proxy used during model factory invocation under TorchAx."""

  def __init__(self, orig_ctx: context_lib.Context):
    self._orig_ctx = orig_ctx
    self.device_kind = _TorchaxDeviceKindProxy(orig_ctx.device_kind)

  def __getattr__(self, name):
    return getattr(self._orig_ctx, name)


class _BaseTorchaxStepper:
  """Base class for TorchAx steppers."""

  def __init__(self):
    self._weights = None
    self._buffers = None
    self._input_args = ()
    self._input_kwargs = {}
    self._runnable_step: Any = None

  def _call_factory(
      self, spec: registry_lib.BenchmarkSpec, ctx: context_lib.Context
  ) -> Any:
    """Runs `spec.factory` on CPU with TorchAX globally disabled.

    Benchmark factories build their models and sample inputs with standard
    PyTorch on CPU (via `_TorchaxFactoryContext`, which overrides
    `device_kind.value` to `"cpu"`), and `_init_model_and_inputs()` moves them
    to `"jax"` afterward.

    `torchax.enable_globally()` is already active when
    `init_with_benchmark_args`
    is called. If TorchAX's `XLAFunctionMode` and `XLADispatchMode` remain on
    PyTorch's mode stacks while `spec.factory` runs, legacy tensor constructors
    such as `torch.Tensor(size)` (used by Wav2Vec2, HuBERT, and Wav2Vec2-BERT to
    initialize `masked_spec_embed`) bypass `TENSOR_CONSTRUCTORS` in
    `XLAFunctionMode` and hit `XLADispatchMode` on `aten::empty.memory_format`.
    That makes `torch.Tensor.__new__` return a `torchax.tensor.Tensor`, and
    Python then invokes `Tensor.__init__(size)`, failing with
    `TypeError: Tensor.__init__() missing 1 required positional argument:
    'env'`.

    We call `torchax.disable_globally()` / `torchax.enable_globally()` directly
    instead of `with torchax.disable_temporarily():` because
    `disable_temporarily()` only acts when `default_env().enabled` is `True`,
    and every internal `with self:` block inside TorchAX (such as in
    `Environment.dispatch()` and `Environment._move_one_value()`) resets
    `enabled` to `False` on exit even though the modes pushed by
    `torchax.enable_globally()` are still active on the mode stacks.
    """
    factory_ctx = _TorchaxFactoryContext(ctx)
    torchax.disable_globally()
    try:
      return spec.factory(factory_ctx)
    finally:
      torchax.enable_globally()

  def _init_model_and_inputs(
      self, model: torch.nn.Module, input_args: Any, input_kwargs: Any
  ) -> torchax.interop.JittableModule:
    self._input_args, self._input_kwargs = _prepare_inputs_for_torchax(
        (input_args, input_kwargs)
    )

    model.to("jax")
    model_jittable = torchax.interop.JittableModule(model)
    self._weights = {
        k: v.data if isinstance(v, torch.nn.Parameter) else v
        for k, v in model_jittable.params.items()
    }
    buffers = _drop_property_buffers(model, model_jittable.buffers)
    # `model.to("jax")` only moves registered parameters and buffers, but
    # `JittableModule` also collects plain tensor attributes as buffers.
    self._buffers = _prepare_inputs_for_torchax({
        k: v.data if isinstance(v, torch.nn.Parameter) else v
        for k, v in buffers.items()
    })
    return model_jittable

  def pre_warmup_init(self) -> None:
    pass

  def post_warmup_hook(self) -> None:
    pass


class TorchaxForwardStepper(_BaseTorchaxStepper):
  """Wraps a benchmark forward-only model and inputs into a TorchAx / JAX functional step."""

  def init_with_benchmark_args(
      self,
      spec: registry_lib.BenchmarkSpec,
      ctx: context_lib.Context,
      *args: Any,
      **kwargs: Any,
  ) -> None:
    """Initializes the stepper by invoking the model factory with spec and ctx."""
    del args, kwargs  # unused in torchax forward stepper wrapper

    model, input_args, input_kwargs = self._call_factory(spec, ctx)

    model_jittable = self._init_model_and_inputs(
        model, input_args, input_kwargs
    )
    self._runnable_step = functools.partial(
        _call_functional_model, model_jittable
    )

  def compile(
      self,
      compile_config: compile_lib.CompileConfig,
      target: Any = None,
  ):
    del target  # unused in torchax compilation
    if compile_config.scope != compile_lib.Scope.STEP:
      raise ValueError(
          "TorchAx stepper wrapper only supports Step scope compilation."
      )
    self._runnable_step = torchax.interop.jax_jit(self._runnable_step)

  def get_step_fn(self) -> step_lib.StepFn:
    assert self._runnable_step is not None

    def step_call():
      out = self._runnable_step(
          self._weights, self._buffers, self._input_args, self._input_kwargs
      )
      return out

    return step_call


class TorchaxBackwardStepper(_BaseTorchaxStepper):
  """Wraps a benchmark model and optimizer into a TorchAx / JAX backward/training step."""

  def __init__(self, compute_loss: Callable[..., torch.Tensor] | None = None):
    super().__init__()
    self._opt_state = None
    self._optax_optimizer = None
    self._compute_loss = compute_loss

  def init_with_benchmark_args(
      self,
      spec: registry_lib.BenchmarkSpec,
      ctx: context_lib.Context,
      *args: Any,
      **kwargs: Any,
  ) -> None:
    """Initializes the stepper by invoking the model factory with spec and ctx."""
    del args, kwargs  # unused in torchax backward stepper wrapper

    model, input_args, input_kwargs, optimizer = self._call_factory(spec, ctx)

    model_jittable = self._init_model_and_inputs(
        model, input_args, input_kwargs
    )
    self._optax_optimizer = optimizer_lib.get_optax_optimizer(optimizer)
    self._opt_state = torchax.interop.call_jax(
        self._optax_optimizer.init, self._weights
    )
    self._runnable_step = functools.partial(
        _train_step_fn,
        model_jittable,
        self._optax_optimizer,
        compute_loss=self._compute_loss,
    )

  def compile(
      self,
      compile_config: compile_lib.CompileConfig,
      target: Any = None,
  ):
    del target  # unused in torchax compilation
    if compile_config.scope != compile_lib.Scope.STEP:
      raise ValueError(
          "TorchAx stepper wrapper only supports Step scope compilation."
      )
    # Donate memory for weights (arg index 0) and opt_state (arg index 2)
    # to JAX for in-place buffer reuse across training iterations.
    self._runnable_step = torchax.interop.jax_jit(
        self._runnable_step, kwargs_for_jax_jit={"donate_argnums": (0, 2)}
    )

  def get_step_fn(self) -> step_lib.StepFn:
    assert self._runnable_step is not None

    def step_call():
      loss, self._weights, self._opt_state = self._runnable_step(
          self._weights,
          self._buffers,
          self._opt_state,
          self._input_args,
          self._input_kwargs,
      )
      return (loss, self._weights, self._opt_state)

    return step_call


def forward() -> step_lib.Stepper:
  """Forward only stepper for TorchAx."""
  return TorchaxForwardStepper()


def training(
    accum_steps: int = 1,
    compute_loss: Callable[..., torch.Tensor] | None = None,
) -> step_lib.Stepper:
  """Training stepper for TorchAx.

  Args:
    accum_steps: The number of micro-batches per optimizer step. Only 1 is
      supported.
    compute_loss: Optional `compute_loss(model, input_args, input_kwargs)` that
      returns a scalar loss tensor, as in the TorchTPU training steppers. If
      `None`, the loss is derived from the model output by `_default_loss_fn`.

  Returns:
    A `TorchaxBackwardStepper`.
  """
  if accum_steps != 1:
    raise ValueError(
        "TorchAx stepper wrapper does not support accumulation steps."
    )
  return TorchaxBackwardStepper(compute_loss=compute_loss)


TORCHAX_STEPPERS: dict[step_lib.StepperType, Any] = {
    step_lib.StepperType.FORWARD: forward,
    step_lib.StepperType.TRAINING: training,
}


def resolve_torchax_stepper(
    name: step_lib.StepperType,
    **kwargs: Any,
) -> step_lib.Stepper:
  """Build the TorchAx stepper with the provided kwargs.

  Args:
    name: The StepperType (e.g. StepperType.FORWARD, StepperType.TRAINING).
    **kwargs: Stepper kwargs passed to the factory.

  Returns:
    An uninitialized instance conforming to the step_lib.Stepper protocol.
  """
  return TORCHAX_STEPPERS[name](**kwargs)
