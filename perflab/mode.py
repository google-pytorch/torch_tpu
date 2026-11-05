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

"""Execution modes for each framework and target kind."""

from collections.abc import Callable, Iterator, Mapping, Sequence
import contextlib
import enum
import functools
import itertools
import math
from typing import Any

import torch
from torch._subclasses import fake_tensor
from torch.jit import _builtins as jit_builtins
from torch.utils import _pytree as pytree
from perflab import common
from perflab import flags as flags_lib
from perflab import target as target_lib

_aten = torch.ops.aten

# The most elements of a tensor whose values FakeTensorMode keeps in the
# FakeTensorMode CPU smoke test, e.g. the token IDs and attention masks of the
# benchmark inputs (at most 16 x 2048 at full scope) and the values computed
# from them, such as Gemma 4's one-hot patch pooling weights (`[1, 2520, 280]`)
# and its image token mask expanded to the input embeddings (`[1, 2048, 2560]`
# at full scope).
_MAX_CONSTANT_NUMEL = 1 << 23

# The most elements of a floating-point factory op output whose values the smoke
# test keeps, e.g. the per-layer drop-path rates of a model.
_MAX_FLOATING_POINT_CONSTANT_NUMEL = 1024

_ARANGE_FACTORIES = frozenset({
    _aten.arange.default,
    _aten.arange.start,
    _aten.arange.start_step,
})

# The factory ops whose floating-point outputs the smoke test computes on the
# CPU. The random ones are e.g. the LayerDrop draws, `torch.rand([])`, of
# BART-like models in training. The floating-point outputs of `ones`, `zeros`
# and `full`, mostly parameter initializations, stay fake.
_FLOATING_POINT_FACTORIES = _ARANGE_FACTORIES | {
    _aten.linspace.default,
    _aten.logspace.default,
    _aten.rand.default,
    _aten.rand.generator,
    _aten.randn.default,
    _aten.randn.generator,
}

# The factory ops whose integer and boolean outputs the smoke test computes on
# the CPU, e.g. the token IDs and attention masks of the benchmark inputs.
_FACTORIES = _FLOATING_POINT_FACTORIES | {
    _aten.full.default,
    _aten.ones.default,
    _aten.randint.default,
    _aten.randint.generator,
    _aten.randint.low,
    _aten.randint.low_generator,
    _aten.randperm.default,
    _aten.randperm.generator,
    _aten.zeros.default,
}

# The factory ops that make integer outputs unless given a `dtype`.
_INTEGER_FACTORIES = frozenset({
    _aten.randint.default,
    _aten.randint.generator,
    _aten.randint.low,
    _aten.randint.low_generator,
    _aten.randperm.default,
    _aten.randperm.generator,
})

# The factory ops that read only the metadata of a tensor argument (its shape,
# strides, dtype and device), e.g. the `torch.ones_like(input_ids)` attention
# mask that VITS makes. As for `_FACTORIES`, the smoke test computes their
# integer and boolean outputs on the CPU.
_LIKE_FACTORIES = frozenset({
    _aten.full_like.default,
    _aten.ones_like.default,
    _aten.zeros_like.default,
})
_TENSOR_FACTORIES = _LIKE_FACTORIES | {
    _aten.new_full.default,
    _aten.new_ones.default,
    _aten.new_zeros.default,
}

# The ops whose outputs are uninitialized memory, e.g. the `self.new_empty(...)`
# of many meta functions. FakeTensorMode would keep that memory as the values of
# their outputs if their inputs are constants.
_UNINITIALIZED_OUTPUT_OPS = frozenset({
    _aten.empty_like.default,
    _aten.new_empty.default,
    _aten.new_empty_strided.default,
})


class Framework(enum.Enum):
  """Execution framework (torch vs. torchax).

  This is not a runtime command-line flag. It is decided by which test binary
  runs and passed in as that binary's module-level constant. It appears only in
  the test execution matrix.
  """

  TORCH = "torch"
  TORCHAX = "torchax"


# TODO: b/534438865 - Move RunMode defn to harness/mode.py
def modes_for(
    framework: Framework | str,
    device_kind: target_lib.DeviceKind | str,
) -> Sequence[common.RunMode]:
  """Execution modes for the given framework and device kind.

  |          | torch                                            | torchax  |
  |----------|--------------------------------------------------|----------|
  | tpu      | eager_default, eager_optimized, compiled         | compiled |
  | cuda     | eager_default, compiled                          |    N/A   |
  | cpu      | eager_default                                    |    N/A   |

  The debug mode eager_defer_never_and_launch_blocking never runs by default.

  Args:
    framework: The execution framework (Framework enum or string representation
      such as "torch" or "torchax").
    device_kind: The hardware device kind (DeviceKind enum or string
      representation such as "tpu", "cuda", or "cpu").

  Returns:
    A list of supported RunMode instances for this framework and device kind.

  Raises:
    ValueError: If an unknown framework or device kind string is provided.
  """
  # The passed values can be strings. Explicitly coerce them into enums to
  # catch invalid values.
  framework = Framework(framework)
  device_kind = target_lib.DeviceKind(device_kind)

  if framework is Framework.TORCHAX:
    # torchax runs compiled only on TPU.
    if device_kind is not target_lib.DeviceKind.TPU:
      raise ValueError("torchax only runs on TPU")
    return [common.RunMode.COMPILED]
  if device_kind is target_lib.DeviceKind.TPU:
    return [
        common.RunMode.EAGER_DEFAULT,
        common.RunMode.EAGER_OPTIMIZED,
        common.RunMode.COMPILED,
    ]
  if device_kind is target_lib.DeviceKind.CPU:
    return [common.RunMode.EAGER_DEFAULT]
  # torch on cuda.
  return [
      common.RunMode.EAGER_DEFAULT,
      common.RunMode.COMPILED,
  ]


@contextlib.contextmanager
def run_mode_context(
    run_mode: common.RunMode | str,
    target: target_lib.Target,
) -> Iterator[None]:
  """Configures the execution environment for different run modes.

  Args:
    run_mode: The execution mode to configure the environment for.
    target: The hardware target descriptor.

  Yields:
    None.

  Raises:
    ValueError: If an unexpected or unsupported run mode is provided.
  """
  # The passed values can be strings. Explicitly coerce them into enums to
  # catch invalid values.
  run_mode = common.RunMode(run_mode)
  if target.device_kind is not target_lib.DeviceKind.TPU:
    if target.device_kind is target_lib.DeviceKind.CUDA:
      torch.set_float32_matmul_precision("high")
    yield
    return

  from torch_tpu._internal import execution_mode  # pylint: disable=g-import-not-at-top

  original_eager_mode = execution_mode.eager_mode
  new_eager_mode = None

  match run_mode:
    case common.RunMode.EAGER_DEFAULT:
      new_eager_mode = execution_mode.EagerMode.DEFER_NEVER
    case common.RunMode.EAGER_OPTIMIZED:
      new_eager_mode = execution_mode.EagerMode.DEFER_AND_FUSE
    case common.RunMode.EAGER_DEFER_NEVER_AND_LAUNCH_BLOCKING:
      new_eager_mode = execution_mode.EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING
    case common.RunMode.COMPILED:
      pass  # Explicitly do nothing.

  if new_eager_mode is not None:
    execution_mode.eager_mode = new_eager_mode

  try:
    yield
  finally:
    # Revert back to the original execution mode.
    execution_mode.eager_mode = original_eager_mode


class _SmokeFakeTensorMode(fake_tensor.FakeTensorMode):
  """FakeTensorMode that keeps the values of small tensors for the smoke test.

  Models read Python values from tensors while they are built and run: their
  per-layer drop-path rates, e.g. `[x.item() for x in torch.linspace(0,
  drop_path_rate, depth)]`, their LayerDrop draws, as in `if torch.rand([]) <
  self.layerdrop`, and values computed from their inputs, as in BART's
  `torch.unique_consecutive(input_ids.eq(eos_token_id).sum(1))`. FakeTensorMode
  reads a Python value only from a tensor whose values it knows (a constant).
  It makes the outputs of an op constants if all the op's input tensors are,
  but never the outputs of factory ops. So this computes the small outputs of
  `_FACTORIES` and `_TENSOR_FACTORIES` on the CPU: the integer and boolean ones
  with at most `_MAX_CONSTANT_NUMEL` elements, e.g. token IDs and attention
  masks, and the floating-point ones of `_FLOATING_POINT_FACTORIES` with at most
  `_MAX_FLOATING_POINT_CONSTANT_NUMEL`.

  It also changes three things that FakeTensorMode does with constants. As this
  overrides `__torch_dispatch__`, they also apply to the ops of FakeTensorMode's
  own op implementations, e.g. the `self.new_empty(...)` of the meta function of
  `aten.index.Tensor`.

  *   FakeTensorMode computes an op whose inputs are all constants on the CPU
      and only then checks whether the outputs are small enough to keep. This
      hides the constants from an op whose outputs are not, e.g. the `[batch,
      1, seq_len, seq_len]` expansion of an attention mask, so that
      FakeTensorMode does not compute it.
  *   This also hides them from `_UNINITIALIZED_OUTPUT_OPS`, whose outputs
      FakeTensorMode would keep as constants of uninitialized memory.
  *   When an op writes values that FakeTensorMode does not compute to a
      constant, FakeTensorMode clears it through a map of the constants that
      `fake_tensor_context` disables. This clears it instead. Other views of
      the same values keep their constants.
  *   When `allow_data_dependent_outputs` is enabled on the mode (opt-in via
      `allow_data_dependent_fake_outputs()` for the duration of the active
      `fake_tensor_context()` so other benchmarks still fail loudly), it returns
      deterministic placeholders for scalar reads, `nonzero`, and boolean-mask
      indexing on non-constant fake tensors.
  """

  allow_data_dependent_outputs: bool = False

  def __torch_dispatch__(
      self,
      func: Any,
      types: Any,
      args: Sequence[Any] = (),
      kwargs: Mapping[str, Any] | None = None,
  ) -> Any:
    kwargs = kwargs or {}
    if _keeps_factory_output(func, args, kwargs):
      with fake_tensor.unset_fake_temporarily():
        value = func(*args, **kwargs)
      return self.fake_tensor_converter.from_real_tensor(
          self, value, make_constant=True
      )
    if _keeps_tensor_factory_output(func, args, kwargs):
      like = args[0]
      with fake_tensor.unset_fake_temporarily():
        # The op reads only the metadata of `like`, so a real tensor with the
        # same metadata stands in for it.
        if func in _LIKE_FACTORIES:
          stand_in = torch.empty_strided(
              like.shape, like.stride(), dtype=like.dtype
          )
        else:
          stand_in = torch.empty(0, dtype=like.dtype)
        value = func(stand_in, *args[1:], **kwargs)
      return self.fake_tensor_converter.from_real_tensor(
          self, value, make_constant=True
      )
    if not isinstance(func, torch._ops.OpOverload):  # pylint: disable=protected-access
      return super().__torch_dispatch__(func, types, args, kwargs)
    if self.allow_data_dependent_outputs:
      if func is _aten._local_scalar_dense.default and _is_non_constant_fake(
          args[0]
      ):
        # Return False for bool so early-stopping checks (e.g. LightGlue) run
        # every layer, 0.0 for floats/complex, and 1 for ints so head indices
        # (e.g. ZoeDepth) and predicted lengths (e.g. VITS) are valid and
        # non-empty.
        if args[0].dtype is torch.bool:
          return False
        if args[0].is_floating_point() or args[0].is_complex():
          return 0.0
        return 1
      if func is _aten.nonzero.default and _is_non_constant_fake(args[0]):
        # Return one row so downstream keypoint gathering has a non-empty index
        # tensor of the expected rank.
        with self:
          return torch.empty(
              (1, args[0].dim()), dtype=torch.int64, device=args[0].device
          )
      if func is _aten.index.Tensor and any(
          _is_non_constant_bool_mask(i) for i in args[1]
      ):
        # Replace non-constant boolean masks with all-True masks so masked
        # pruning (e.g. LightGlue) keeps all elements.
        with self:
          indices = [
              torch.ones_like(i) if _is_non_constant_bool_mask(i) else i
              for i in args[1]
          ]
          return func(args[0], indices, *args[2:], **kwargs)
    tensors = _tensor_args(args, kwargs)
    # Keyed by id, as tensors compare by value.
    constants: dict[int, fake_tensor.FakeTensor] = {
        id(t): t
        for t in tensors
        if isinstance(t, fake_tensor.FakeTensor) and t.constant is not None
    }
    if not constants:
      return super().__torch_dispatch__(func, types, args, kwargs)
    written = {id(t): t for t in _written_fake_tensors(func, args, kwargs)}
    if len(constants) < len({id(t) for t in tensors}) or not (
        _propagates_constants(func)
    ):
      for t in written.values():
        t.constant = None
      return super().__torch_dispatch__(func, types, args, kwargs)
    if func not in _UNINITIALIZED_OUTPUT_OPS and _outputs_fit_in_constants(
        func, args, kwargs
    ):
      return super().__torch_dispatch__(func, types, args, kwargs)
    hidden = {key: t.constant for key, t in constants.items()}
    for t in constants.values():
      t.constant = None
    try:
      return super().__torch_dispatch__(func, types, args, kwargs)
    finally:
      for key, constant in hidden.items():
        if key not in written:
          constants[key].constant = constant


def _is_non_constant_fake(tensor: Any) -> bool:
  """Returns whether `tensor` is a FakeTensor without a constant value."""
  return isinstance(tensor, fake_tensor.FakeTensor) and tensor.constant is None


def _is_non_constant_bool_mask(index: Any) -> bool:
  """Returns whether an `aten.index.Tensor` index is a non-constant bool mask."""
  return _is_non_constant_fake(index) and index.dtype is torch.bool


def _bind(
    func: Any, args: Sequence[Any], kwargs: Mapping[str, Any]
) -> Mapping[str, Any]:
  """Returns the arguments of an op call by name, except omitted defaults."""
  names = (argument.name for argument in func._schema.arguments)  # pylint: disable=protected-access
  return dict(zip(names, args)) | dict(kwargs)


def _keeps_factory_output(
    func: Any, args: Sequence[Any], kwargs: Mapping[str, Any]
) -> bool:
  """Returns whether the smoke test computes a factory op call on the CPU."""
  if func not in _FACTORIES:
    return False
  bound = _bind(func, args, kwargs)
  device = bound.get("device")
  if device is not None and torch.device(device).type != "cpu":
    return False
  if bound.get("layout") not in (None, torch.strided):
    return False
  if _makes_floating_point(func, bound):
    return (
        func in _FLOATING_POINT_FACTORIES
        and _factory_numel(bound) <= _MAX_FLOATING_POINT_CONSTANT_NUMEL
    )
  return _factory_numel(bound) <= _MAX_CONSTANT_NUMEL


def _keeps_tensor_factory_output(
    func: Any, args: Sequence[Any], kwargs: Mapping[str, Any]
) -> bool:
  """Returns whether the smoke test computes a tensor factory op on the CPU."""
  if func not in _TENSOR_FACTORIES or not isinstance(args[0], torch.Tensor):
    return False
  like = args[0]
  bound = _bind(func, args, kwargs)
  dtype = bound.get("dtype") or like.dtype
  device = bound.get("device") or like.device
  numel = like.numel() if func in _LIKE_FACTORIES else _factory_numel(bound)
  return (
      not (dtype.is_floating_point or dtype.is_complex)
      and torch.device(device).type == "cpu"
      and like.layout == torch.strided
      and bound.get("layout") in (None, torch.strided)
      and numel <= _MAX_CONSTANT_NUMEL
  )


def _makes_floating_point(func: Any, bound: Mapping[str, Any]) -> bool:
  """Returns whether a factory op call makes a floating-point output."""
  dtype = bound.get("dtype")
  if dtype is not None:
    return dtype.is_floating_point or dtype.is_complex
  if func in _INTEGER_FACTORIES:
    return False
  if func is _aten.full.default:
    return isinstance(bound["fill_value"], (float, complex))
  if func in _ARANGE_FACTORIES:
    return any(
        isinstance(bound.get(name), float) for name in ("start", "end", "step")
    )
  # `ones`, `zeros`, `rand`, `randn`, `linspace` and `logspace`.
  return True


def _factory_numel(bound: Mapping[str, Any]) -> int | float:
  """Returns the number of elements of a factory op output, or infinity.

  Args:
    bound: The arguments of the factory op call by name. Symbolic sizes count as
      infinitely many elements.
  """
  if "size" in bound:
    shape = bound["size"]
  elif "n" in bound:  # randperm
    shape = [bound["n"]]
  elif "steps" in bound:  # linspace, logspace
    shape = [bound["steps"]]
  else:  # arange
    start, end, step = bound.get("start", 0), bound["end"], bound.get("step", 1)
    if not step or not all(
        isinstance(x, (int, float)) for x in (start, end, step)
    ):
      return math.inf
    shape = [max(0, math.ceil((end - start) / step))]
  if isinstance(shape, (list, tuple)) and all(
      isinstance(dim, int) for dim in shape
  ):
    return math.prod(shape)
  return math.inf


def _tensor_args(
    args: Sequence[Any], kwargs: Mapping[str, Any]
) -> Sequence[torch.Tensor]:
  """Returns the tensors among the arguments of an op call."""
  tensors = []
  for arg in itertools.chain(args, kwargs.values()):
    if isinstance(arg, torch.Tensor):
      tensors.append(arg)
    elif isinstance(arg, (list, tuple)):  # E.g. `Tensor[]` and `Tensor?[]`.
      tensors.extend(t for t in arg if isinstance(t, torch.Tensor))
  return tensors


def _written_fake_tensors(
    func: Any, args: Sequence[Any], kwargs: Mapping[str, Any]
) -> Sequence[fake_tensor.FakeTensor]:
  """Returns the fake tensors that an op call writes to."""
  bound = _bind(func, args, kwargs)
  written = []
  for argument in func._schema.arguments:  # pylint: disable=protected-access
    if argument.alias_info is None or not argument.alias_info.is_write:
      continue
    value = bound.get(argument.name)
    values = value if isinstance(value, (list, tuple)) else [value]
    written.extend(t for t in values if isinstance(t, fake_tensor.FakeTensor))
  return written


def _propagates_constants(func: Any) -> bool:
  """Returns whether FakeTensorMode computes an op whose inputs are constants.

  These are the checks of `FakeTensorMode._dispatch_impl` that apply in the
  smoke test: it computes neither random ops nor ops that change the shape of an
  input in place.

  Args:
    func: The op.
  """
  return torch.Tag.nondeterministic_seeded not in func.tags and (
      torch.Tag.inplace_view not in func.tags or func is _aten.detach_.default
  )


def _outputs_fit_in_constants(
    func: Any, args: Sequence[Any], kwargs: Mapping[str, Any]
) -> bool:
  """Returns whether no output of an op call has too many elements to keep.

  This runs the op on meta tensors. If that fails, e.g. for `nonzero`, whose
  output shape depends on the input values, it returns True and FakeTensorMode
  computes the op on the CPU.

  Args:
    func: The op.
    args: The positional arguments of the call.
    kwargs: The keyword arguments of the call.
  """
  with fake_tensor.unset_fake_temporarily():
    meta_args, meta_kwargs = pytree.tree_map_only(
        fake_tensor.FakeTensor,
        lambda t: torch.empty_strided(
            t.shape, t.stride(), dtype=t.dtype, device="meta"
        ),
        (args, kwargs),
    )
    try:
      outputs = func(*meta_args, **meta_kwargs)
    except Exception:  # pylint: disable=broad-exception-caught
      return True
  return all(
      t.numel() <= _MAX_CONSTANT_NUMEL
      for t in pytree.tree_leaves(outputs)
      if isinstance(t, torch.Tensor)
  )


def _with_constant_data(
    factory: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
  """Wraps `torch.tensor` or `torch.as_tensor` to read constants in its data.

  These read the values of the tensors in a nested sequence, e.g. the 0-d
  tensors of Qwen2.5-VL's `torch.tensor(mrope_position_deltas)`, without
  dispatching, which fails for fake tensors even if they are constants. The
  wrapper passes their constants instead.

  Args:
    factory: `torch.tensor` or `torch.as_tensor`.
  """

  @functools.wraps(factory)
  def wrapper(data: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
    return factory(_with_constants(data), *args, **kwargs)

  return wrapper


def _with_constants(data: Any) -> Any:
  """Returns `data` with its nested fake tensors replaced by their constants."""
  if not isinstance(data, (list, tuple)):
    return data
  return [
      x.constant
      if isinstance(x, fake_tensor.FakeTensor) and x.constant is not None
      else _with_constants(x)
      for x in data
  ]


@contextlib.contextmanager
def fake_tensor_context() -> Iterator[None]:
  """Enters FakeTensorMode for the FakeTensorMode CPU smoke test.

  Callers enter it only if `fake_tensor_smoke_enabled()`; `--fake_tensor_mode`
  is validated at flag-parse time to be set only with BENCHMARK_PLATFORM=cpu.

  While active, this disables FakeTensor memoization and the argument validation
  of `torch.distributions`, makes `torch.nn.init.trunc_normal_` a no-op and lets
  FakeTensorMode keep the values of small tensors process-wide (by patching
  private helpers, `torch.tensor` and `torch.as_tensor`), so it is not
  thread-safe.

  Yields:
    None.
  """
  mode = _SmokeFakeTensorMode(allow_non_fake_inputs=True)
  # `nn.Module._apply` (e.g. `model.to(dtype)`) always replaces FakeTensor
  # parameters with `torch.utils.swap_tensors`, whatever the `torch.__future__`
  # conversion flags, and `swap_tensors` refuses tensors that have weakrefs. The
  # converter keeps weakrefs to every fake it makes: in its tensor memo (a
  # WeakValueDictionary) and, for constants such as CLIP's
  # `torch.tensor(2.6592)` logit scale, in its constant storage mapping.
  # Dropping both and keeping fakes strongly instead lets the swap succeed. The
  # cost, fine for a one-step smoke: a real tensor converted twice yields two
  # unrelated fakes, converted fakes live as long as the mode, and an op that
  # writes fake values to a constant leaves the constants of its other views
  # stale (`_SmokeFakeTensorMode` clears the constant of the tensor written to).
  # These patch converter internals; `test_converts_constant_parameters` in
  # mode_test is the canary for upstream changes to them.
  mode.fake_tensor_converter.set_tensor_memo = lambda t, v: None
  mode.fake_tensor_converter.add_constant_storage_mapping = (
      lambda *args, **kwargs: None
  )
  mode.fake_tensor_converter.meta_converter.tensor_memo = {}  # pyrefly: ignore[bad-assignment]

  # `trunc_normal_` (e.g. in the HuggingFace ViT position embedding init)
  # rejection-samples with data-dependent control flow (`.item()`,
  # `mask.any()`), which FakeTensorMode cannot evaluate. The values do not
  # matter for fake tensors. This patches the private helper that
  # `trunc_normal_` calls rather than `trunc_normal_` itself, because modules
  # such as transformers v5 bind `trunc_normal_` at import time, so patching the
  # public name would miss them (e.g. v5 ViT). If upstream renames the helper,
  # this fails loudly with an AttributeError.
  orig_trunc_normal = torch.nn.init._no_grad_trunc_normal_  # pylint: disable=protected-access
  torch.nn.init._no_grad_trunc_normal_ = (  # pylint: disable=protected-access
      lambda tensor, *args, **kwargs: tensor
  )
  # FakeTensorMode keeps the values only of tensors with at most
  # `CONSTANT_NUMEL_LIMIT` elements, by default 1. Raise it so that it also
  # keeps them for e.g. the per-stage slices of the drop-path rates
  # (`torch.linspace(0, drop_path_rate, sum(depths)).split(depths)`) and the
  # values computed from the token IDs of the benchmark inputs.
  orig_constant_numel_limit = fake_tensor.CONSTANT_NUMEL_LIMIT
  fake_tensor.CONSTANT_NUMEL_LIMIT = _MAX_CONSTANT_NUMEL  # pyrefly: ignore[bad-assignment]
  # `torch.distributions` checks the parameters of a distribution, e.g. the
  # logits of the `Categorical` in the TAPAS cell selection loss, with
  # data-dependent control flow, which FakeTensorMode cannot evaluate.
  orig_validate_args = torch.distributions.Distribution._validate_args  # pylint: disable=protected-access
  torch.distributions.Distribution.set_default_validate_args(False)
  # `torch.tensor` and `torch.as_tensor` cannot read the constants of fake
  # tensors in their data (see `_with_constant_data`), so patch them. A
  # TorchFunctionMode would see every torch call, including those of
  # FakeTensorMode itself, which made the smoke test about 15% slower (and
  # Qwen2.5-VL about 8x slower). TorchScript compiles a call to a function only
  # if it knows it as a builtin, e.g. in the `@torch.jit.script` helpers of
  # DeBERTa v2, which transformers imports while it builds the model, so
  # register the wrappers as the builtins of the originals.
  orig_factories = (torch.tensor, torch.as_tensor)
  wrappers = tuple(_with_constant_data(f) for f in orig_factories)
  builtin_table = jit_builtins._get_builtin_table()  # pylint: disable=protected-access
  for orig, wrapper in zip(orig_factories, wrappers):
    builtin_table[id(wrapper)] = builtin_table[id(orig)]
  torch.tensor, torch.as_tensor = wrappers  # pyrefly: ignore[bad-assignment]
  try:
    with mode:
      yield
  finally:
    torch.tensor, torch.as_tensor = orig_factories
    for wrapper in wrappers:
      del builtin_table[id(wrapper)]
    torch.distributions.Distribution.set_default_validate_args(
        orig_validate_args
    )
    fake_tensor.CONSTANT_NUMEL_LIMIT = orig_constant_numel_limit
    torch.nn.init._no_grad_trunc_normal_ = orig_trunc_normal  # pylint: disable=protected-access


def allow_data_dependent_fake_outputs() -> None:
  """Enables placeholders for data-dependent ops in the active smoke mode.

  When called inside `fake_tensor_context()`, the active `_SmokeFakeTensorMode`
  returns deterministic placeholders for scalar reads
  (`aten._local_scalar_dense.default`), `aten.nonzero.default`, and
  boolean-mask `aten.index.Tensor` on non-constant fake tensors until the
  context exits. Outside `fake_tensor_context()`, this is a no-op.
  """
  mode = torch._guards.detect_fake_mode()  # pylint: disable=protected-access
  if isinstance(mode, _SmokeFakeTensorMode):
    mode.allow_data_dependent_outputs = True


def fake_tensor_smoke_enabled() -> bool:
  """Returns whether this run is the FakeTensorMode CPU smoke test.

  That is, whether `--fake_tensor_mode` is set; flag parsing fails unless it is
  set with BENCHMARK_PLATFORM=cpu.
  """
  return flags_lib.FAKE_TENSOR_MODE.value
