"""Run a Hugging Face causal LM on a TPU via TorchTPU and time generation.

Pick the eager mode with env vars *before* launching (they are read at startup):

    python hf_tpu_bench.py                                  # strict eager (default)
    TORCH_TPU_DEFER_AND_FUSE=1 python hf_tpu_bench.py       # DeferAndFuse
    TORCH_TPU_LAUNCH_BLOCKING=1 python hf_tpu_bench.py      # debug (slow, for debugging)
    python hf_tpu_bench.py --compile                        # torch.compile(backend="tpu")

Needs: pip install torch_tpu "transformers>=5"
"""
import argparse
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def mode_name(compiled: bool) -> str:
    if compiled:
        return "torch.compile"
    if os.environ.get("TORCH_TPU_LAUNCH_BLOCKING") == "1":
        return "debug eager"
    if os.environ.get("TORCH_TPU_DEFER_AND_FUSE") == "1":
        return "defer-and-fuse eager"
    return "strict eager"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--prompt", default="Explain what a TPU is in two sentences.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="tpu"
    )
    if args.compile:
        # Backend string is from the TorchTPU announcement; confirm against the API reference.
        model = torch.compile(model, backend="tpu")

    inputs = tok(args.prompt, return_tensors="pt").to("tpu")
    gen = dict(max_new_tokens=args.max_new_tokens, do_sample=False)

    for i in range(args.runs):
        t0 = time.perf_counter()
        out = model.generate(**inputs, **gen)
        out = out.cpu()  # forces completion of async TPU work before stopping the clock
        dt = time.perf_counter() - t0
        n = out.shape[-1] - inputs["input_ids"].shape[-1]
        tag = "warmup (includes compile)" if i == 0 else "steady"
        print(f"[{mode_name(args.compile)}] run {i}: {n} tokens in {dt:.2f}s "
              f"({n / dt:.1f} tok/s) {tag}")

    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
