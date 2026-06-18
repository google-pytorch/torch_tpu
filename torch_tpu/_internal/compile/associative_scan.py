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

"""TPU implementation for torch.associative_scan (combine_mode="pointwise")."""

import torch
from torch._higher_order_ops.associative_scan import associative_scan_op
import torch.utils._pytree as pytree
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.export import export


def _handle_associative_scan_impl(combine_fn, xs, additional_inputs):
  """Lowers a pointwise associative_scan to the native scan emitter.

  Emits an associative chlo.ScanOp instead of the upstream
  generic_associative_scan decomposition. The native scan op computes an
  inclusive scan but folds its carry init into every output, and
  associative_scan has no identity to seed that init with. So we seed the carry
  with each input's first element, scan the remaining N-1 elements, and prepend
  the first element back, yielding out[0] = xs[0] for any combiner.

  Args:
    combine_fn: The combine subgraph (a torch.fx.GraphModule). It takes `(*lhs,
      *rhs, *additional_inputs)` and returns the combined leaves.
    xs: The scanned inputs, with the scan dimension already moved to 0.
    additional_inputs: Extra tensors the combiner closes over; must be tensors.

  Returns:
    A tuple of the stacked per-position outputs, one per scanned input.
  """
  xs = list(xs)
  num_scan_inputs = len(xs)
  # Lifted inputs become body placeholders below (via placeholder_like), which
  # only handles tensors; a non-tensor input would need to be baked into the
  # body as a constant, which is not wired up.
  if not all(isinstance(a, torch.Tensor) for a in additional_inputs):
    raise NotImplementedError(
        "associative_scan with non-tensor additional inputs is not supported"
        " on TPU"
    )
  extra_inputs = list(additional_inputs)

  # A scan over 0 or 1 positions is the identity.
  scan_len = xs[0].shape[0]
  if scan_len <= 1:
    return tuple(xs)

  firsts = [x[0] for x in xs]
  rests = [x[1:] for x in xs]

  # Rank-reduced body placeholders, in the order combine_fn expects: carries,
  # then scanned slices, then the extra closed-over inputs.
  make_slice = lambda x: tpu_torch_compile.placeholder(
      list(x.shape[1:]), x.dtype, False
  )
  lhs = [make_slice(x) for x in xs]
  rhs = [make_slice(x) for x in xs]
  extra = [tpu_torch_compile.placeholder_like(a) for a in extra_inputs]

  # Trace the combine subgraph to capture its outputs as the scan body.
  combined = export.EagerLikeFxInterpreter(combine_fn).run(*lhs, *rhs, *extra)
  combined = list(pytree.tree_leaves(combined))

  # The body returns the combined values twice: once as the next carry, once as
  # the per-position output.
  body_module = tpu_torch_compile.build_mlir(
      result_tensors=combined + combined,
      argument_tensors=[*lhs, *rhs, *extra],
  )

  # Stacked output shapes for the N-1 scanned positions.
  dummy_outputs = [
      tpu_torch_compile.placeholder([scan_len - 1, *c.shape], c.dtype, False)
      for c in combined
  ]

  results = tpu_torch_compile.create_scan_op(
      inits=firsts,
      inputs=[*rests, *extra_inputs],
      body_module=body_module,
      scan_direction=tpu_torch_compile.ScanDirection.kForward,
      dummy_ys=dummy_outputs,
      num_scan_inputs=num_scan_inputs,
      is_associative=True,
  )
  # create_scan_op returns [final_carries..., stacked_outputs...]. The outputs
  # cover positions 1..N-1, so prepend each input's first position.
  scan_outputs = results[num_scan_inputs:]
  return tuple(
      torch.cat([x[0:1], out], dim=0) for x, out in zip(xs, scan_outputs)
  )


try:
  # pylint: disable=g-import-not-at-top
  # pylint: disable=protected-access
  import torch._C

  # Register only the eager TPU execution keys. Fake-tensor shape inference and
  # proxy tracing keep using the upstream CompositeExplicitAutograd
  # decomposition, so Meta is intentionally left unregistered.
  for key in (
      torch._C.DispatchKey.XLA,
      torch._C.DispatchKey.PrivateUse1,
  ):
    associative_scan_op.py_impl(key)(_handle_associative_scan_impl)
except (AttributeError, RuntimeError):
  # AttributeError: Safe to ignore if the environment lacks XLA/TPU support.
  # RuntimeError: Safe to ignore if the implementation is already registered.
  pass
