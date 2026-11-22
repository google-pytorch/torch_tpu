## Summary

Eight ATen ops are not registered for TPU on 0.2.0 and raise `NotImplementedError: operator 'aten::<op>' is not implemented for TPU`:

`smooth_l1_loss` (and its backward), `huber_loss` (and its backward), `logaddexp`, `logaddexp2`, `nansum`, `std_mean`, `trace`, `addbmm`.

All eight are listed in `UNREGISTERED_OPS` in `tests/native_functions_data.py`. They are common in loss functions (`F.smooth_l1_loss`, `F.huber_loss`), numerically stable log-space code (`logaddexp`), and model-introspection libraries (Captum's attribution methods reach `nansum`, `std_mean` and `trace` through user models).

## Reproduce

```python
import torch, torch_tpu
x = torch.randn(4, 4, device="tpu")
torch.trace(x)                      # NotImplementedError: aten::trace
torch.nansum(x)                     # aten::nansum
torch.std_mean(x, dim=0)            # aten::std_mean.correction
torch.logaddexp(x, x)               # aten::logaddexp.out
torch.nn.functional.smooth_l1_loss(x, x)  # aten::smooth_l1_loss.out
torch.addbmm(x, x[None], x[None])   # aten::addbmm
```

## Fix

Implemented as direct StableHLO lowerings (no composite kernels; `OpNameStack` rejects those) with 53 unit tests that pass on a v6e-1 against a wheel built from the branch:

https://github.com/joinalahmed/torch_tpu/tree/missing-ops-batch
(diff: https://github.com/google-pytorch/torch_tpu/compare/main...joinalahmed:torch_tpu:missing-ops-batch)

I cannot open the pull request yet: the repository allows users without write access 3 open PRs and mine (#6102, #6105, #6106) fill that quota. I will open it as soon as one of those is merged or closed, or a maintainer is welcome to pull the branch directly.

Still unregistered after this batch (same probe): `triu_indices`, `median`, `nanmedian`, `mode`, `kthvalue`, `frexp`, `renorm`, and the `.out` variants of `hypot`, `copysign`, `erfc`, `sinc`, `mish`, `index_select`, `heaviside`, `nextafter`, `range`, `logspace`, `hardshrink`.
