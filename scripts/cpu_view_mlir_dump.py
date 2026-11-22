"""Dump the split and the StableHLO for compiled `x * cum[1]` (CPU view scalar)."""
import torch, torch_tpu
import torch_tpu._internal.compile.split_compiler as sc
orig_cs = sc._SubmodCompiler.compile_submod
def cs(self, input_mod, args, kwargs, submod_name=None):
    print("### compile_submod", submod_name)
    print(input_mod.graph)
    for i, a in enumerate(args):
        if isinstance(a, torch.Tensor):
            print(f"  arg{i}: {a.device} shape={tuple(a.shape)} stride={a.stride()} off={a.storage_offset()} dtype={a.dtype} type={type(a).__name__}")
    return orig_cs(self, input_mod, args, kwargs, submod_name)
sc._SubmodCompiler.compile_submod = cs
orig_fw = sc._WrapperModule.forward
def fw(self, *args):
    for i, a in enumerate(args):
        if isinstance(a, torch.Tensor):
            print(f"### runtime arg{i}: {a.device} shape={tuple(a.shape)} stride={a.stride()} off={a.storage_offset()} val={a.cpu().flatten().tolist()[:4]}")
    return orig_fw(self, *args)
sc._WrapperModule.forward = fw
tpu = torch.device("tpu")
x = torch.arange(6.0).reshape(2, 3).to(tpu)
cum = torch.tensor([0, 2])
out = torch.compile(lambda x: x * cum[1], backend="tpu")(x)
print("### result", out.cpu().flatten().tolist())
