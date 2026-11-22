"""Patch env_fix's installed split_compiler.py: compile TPU submodules against
contiguous, offset-0 metadata for CPU args (what `.to(device)` produces)."""
import re, sys, torch_tpu, pathlib
p = pathlib.Path(torch_tpu.__file__).parent / "_internal/compile/split_compiler.py"
s = p.read_text()
old = """    compiled_submod_real = self.compiler_fn(
        input_mod, args, module_name=full_submod_name
    )"""
new = """    # CPU arguments reach the executable through `_WrapperModule.forward`,
    # which copies them to the device as contiguous tensors with storage
    # offset 0. Compile against that layout rather than the CPU view's, whose
    # strides and offset would otherwise be baked into the executable.
    compile_args = [
        torch.empty_like(arg, memory_format=torch.contiguous_format)
        if i in cpu_arg_indices
        else arg
        for i, arg in enumerate(args)
    ]
    compiled_submod_real = self.compiler_fn(
        input_mod, compile_args, module_name=full_submod_name
    )"""
assert old in s, "anchor not found"
s = s.replace(old, new)
s = s.replace("arg.to(device) if i in self.cpu_arg_indices else arg",
              "arg.to(device, memory_format=torch.contiguous_format)\n          if i in self.cpu_arg_indices\n          else arg")
p.write_text(s); print("patched", p)
