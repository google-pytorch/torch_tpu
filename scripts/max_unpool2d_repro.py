"""Minimal repro: aten::max_unpool2d is not implemented for TPU (hit by Captum DeepLift's MaxPool rule)."""
import torch, torch.nn.functional as F
x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
out, idx = F.max_pool2d(x, 2, return_indices=True)
print("cpu :", F.max_unpool2d(out, idx, 2).shape)
try:
    print("tpu :", F.max_unpool2d(out.to("tpu"), idx.to("tpu"), 2).shape)
except Exception as e:  # noqa: BLE001
    print("tpu :", type(e).__name__, str(e).splitlines()[0])
