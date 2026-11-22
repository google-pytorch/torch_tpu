## Summary

`LayerIntegratedGradients.attribute` builds `num_outputs_cumsum` as a CPU
`int64` tensor (`torch.cumsum(torch.IntTensor(...))`) and the forward hook
installed on the layer indexes with its elements. The hook runs inside the
model's forward, so when the model is wrapped in `torch.compile`, Dynamo
lifts that tensor into the graph as an input (`l_num_outputs_cumsum_`,
shape `[2]`) and specializes on the indexing.

This replaces it with a list of Python ints (`itertools.accumulate`). The
values are known before the forward runs, so there is nothing to trace:
the compiled graph has no CPU tensor input and no data-dependent indexing.
The `int(...)` casts when slicing `attributions` become plain slices.

Motivation: on accelerator backends whose `torch.compile` backend rejects
or splits on CPU graph inputs (TorchTPU's `backend="tpu"`), this tensor was
the only thing stopping `LayerIntegratedGradients` from running on a
compiled model. On TorchTPU 0.2.0 it raised at compile time; on current
TorchTPU main it compiled but hit a wrong-result bug in the backend
(google-pytorch/torch_tpu#6106). With this change neither path is taken.

Measured with a small `Sequential` model compiled with `backend="tpu"` and a
backend wrapper that records graph inputs on CPU:

| Captum | graphs | CPU graph inputs | max abs diff vs eager LIG |
|---|---|---|---|
| 0.9.0 | 1 | `l_num_outputs_cumsum_: int64 [2]` | 0 |
| this PR | 1 | none | 0 |

## Test plan

- `tests/attr/layer/test_layer_integrated_gradients.py`: 13 passed (CPU,
  torch 2.11, Python 3.14, run against this branch's source).
- `tests/attr/test_accelerator.py -k LayerIntegratedGradients` with
  `CAPTUM_TEST_DEVICE=tpu` on a Cloud TPU v6e-1 (torch_tpu main): see
  results below.
- `ufmt` and `flake8` clean on the changed file.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
