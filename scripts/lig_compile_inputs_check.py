"""Count graphs and CPU graph inputs when Captum LIG wraps a compiled TPU model.

Run twice on the VM: with the released Captum and with the patched
layer_integrated_gradients.py copied over it.
"""
import captum
import torch
import torch._dynamo as dynamo
from captum.attr import LayerIntegratedGradients

torch.set_float32_matmul_precision("highest")
tpu = torch.device("tpu")
torch.manual_seed(0)
model = torch.nn.Sequential(
    torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1)
).to(tpu)
x = torch.randn(3, 4, device=tpu)

tpu_backend = dynamo.lookup_backend("tpu")
stats = {"graphs": 0, "cpu_inputs": []}


def diag_backend(gm, example_inputs):
  stats["graphs"] += 1
  phs = [n for n in gm.graph.nodes if n.op == "placeholder"]
  for n, ex in zip(phs, example_inputs):
    if isinstance(ex, torch.Tensor) and ex.device.type == "cpu":
      stats["cpu_inputs"].append(f"{n.name}:{ex.dtype}:{tuple(ex.shape)}")
  return tpu_backend(gm, example_inputs)


comp = torch.compile(model, backend=diag_backend)
lig = LayerIntegratedGradients(comp, model[0])
attr = lig.attribute(x, baselines=torch.zeros_like(x), n_steps=4)
ref = LayerIntegratedGradients(model, model[0]).attribute(
    x, baselines=torch.zeros_like(x), n_steps=4
)
print(
    f"captum {captum.__version__}: graphs={stats['graphs']}"
    f" cpu_inputs={stats['cpu_inputs']}"
    f" max_abs_diff_vs_eager={(attr - ref).abs().max().item():.3e}"
)
