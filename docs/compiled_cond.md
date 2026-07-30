# Compiled `torch.cond` on TPU

`torch.cond(pred, true_fn, false_fn, operands)` lowers to one deferred TPU op
that emits `stablehlo.if`. Both branch `GraphModule`s are traced through the
existing FX-to-MLIR path, then their `main` bodies are cloned into the true and
false regions. Outputs are static: both branches must return the same pytree,
number of tensor leaves, dtype, and shape.

The predicate must be a bool tensor with exactly one element. Rank-1 predicates
with shape `[1]` are reshaped to a scalar predicate before constructing
`stablehlo.if`.

## HLO Evidence

`tests/cond_test.py` captures StableHLO for a true predicate where the false
branch contains `torch.exp`. The expected structure is:

```mlir
%0 = "stablehlo.if"(%pred) ({
  %1 = stablehlo.add %x, %cst : tensor<3xf32>
  stablehlo.return %1 : tensor<3xf32>
}, {
  %1 = stablehlo.exponential %x : tensor<3xf32>
  %2 = stablehlo.add %1, %cst : tensor<3xf32>
  stablehlo.return %2 : tensor<3xf32>
}) : (tensor<i1>) -> tensor<3xf32>
```

The test asserts that the captured HLO contains `stablehlo.if` and the false
branch's `stablehlo.exponential`, and that it does not contain
`stablehlo.select`. That distinction matters because a `select`-based lowering
would compute both branch values before selecting, while `stablehlo.if` keeps
branch work inside the selected control-flow region.

## Collectives

Branch collectives are supported only when every rank takes the same branch.
The collective is inside the `stablehlo.if` region, so a uniformly untaken
branch does not enqueue that collective. A following unconditional collective
remains in program order after the conditional.

Per-rank predicates are supported only when neither branch contains
collectives. In that case every rank may compute a different local value, but
all ranks still rendezvous at the same unconditional collective schedule after
the cond.

## Current Limitations

This implementation intentionally does not add a general `while_loop` lowering.
Generic cond autograd is the MVP; application-specific custom VJP paths can be
added later if needed.

Branch outputs must be tensors. Dynamic branch output shapes, branch output
pytrees that differ across branches, and branch output dtype/shape mismatches
are rejected before constructing the deferred op.
