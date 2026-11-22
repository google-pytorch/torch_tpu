### Environment

| Item | Value |
|---|---|
| Hardware | Cloud TPU v6e-1 (us-east5-b) |
| Software | Python 3.12, `torch 2.15.0`, `torch_tpu 0.2.0`; `main` lists `max_unpool2d` and `max_unpool3d` (and `.out`) in `UNREGISTERED_OPS` in `tests/native_functions_data.py` |

### Summary

`torch.nn.functional.max_unpool2d` fails on TPU tensors because `aten::max_unpool2d` is not registered.

```python
import torch, torch.nn.functional as F

x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
out, idx = F.max_pool2d(x, 2, return_indices=True)
F.max_unpool2d(out.to("tpu"), idx.to("tpu"), 2)
```

```
RuntimeError: operator 'aten::max_unpool2d' is not implemented for TPU. Please file a feature request
```

`max_pool2d(..., return_indices=True)` itself works on TPU; only the inverse op is missing. `max_unpool3d` is in the same state.

### Motivation

[Captum](https://github.com/meta-pytorch/captum)'s DeepLift family (`DeepLift`, `DeepLiftShap`, `LayerDeepLift`, `NeuronDeepLift`) implements its MaxPool attribution rule with `F.max_unpool{1,2,3}d` ([`captum/attr/_core/deep_lift.py`](https://github.com/meta-pytorch/captum/blob/master/captum/attr/_core/deep_lift.py#L235-L249)). Any CNN with a `MaxPool2d` layer therefore cannot be explained with DeepLift on TPU. Running Captum's own device-comparison configs on TPU (meta-pytorch/captum#1951), 262 of 266 pass; the 3 non-tolerance failures are all this op (`conv_layer_single_target_LayerDeepLift`, `conv_neuron_NeuronDeepLift`, `conv_neuron_slice_selector_NeuronDeepLift`).

A scatter of `input` into a zero tensor of the output size at `indices` (what `max_unpool2d` is, per `aten/src/ATen/native/MaxUnpooling.cpp`) covers forward; the backward in `derivatives.yaml` is a gather with the same indices, which is already supported.
