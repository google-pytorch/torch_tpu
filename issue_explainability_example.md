## Summary

Proposal: add an explainability example (`examples/explainability/captum_attribution.py`) and a docs case study, "Explainability as a performance lens", that uses Captum attribution as a compact stress test for the three execution modes.

Attribution methods such as Integrated Gradients run many forward and backward passes with host-side scalars in the loop, use forward hooks and `torch.autograd.grad`, and output differences of gradients. That makes them sensitive to exactly the things the TorchTPU docs want to teach: matmul precision, strict eager vs DeferAndFuse vs `torch.compile`, host-device copies in XProf, and peak HBM.

## What the case study shows (measured on a v6e-1, torch_tpu 0.2.0, Captum 0.9.0)

- Every public `captum.attr` method runs on TorchTPU unmodified. Deterministic ones match CPU to 1e-7 relative error on small models **only with** `torch.set_float32_matmul_precision("highest")`; the default gives 1e-3 (1e-2 for LayerGradCam).
- Qwen3-0.6B, `LayerIntegratedGradients`, 42-token prompt, 64 steps: strict eager 0.80 s, DeferAndFuse 0.36 s, `torch.compile(backend="tpu")` 0.16 s warm. First call 5 s / 43 s / 130 s.
- XProf: in strict eager 70% of host time is `aten::_to_copy` (522 blocking copies at 2.6 ms each); DeferAndFuse does the same copies at 0.57 ms each.
- Peak HBM is set by the `[steps, T, V]` logits tensor; `logits_to_keep=1` and `internal_batch_size` bound it.
- Always check the convergence delta: raw-logit target plus pad baseline does not converge on an LLM at 256 steps.

The compiled path needs meta-pytorch/captum#1952 and #6106.

## Branch

Example script, Marimo source and Markdown export, mkdocs nav, index and examples README entries; `mkdocs build --strict` passes:

https://github.com/joinalahmed/torch_tpu/tree/explainability-example
(diff: https://github.com/google-pytorch/torch_tpu/compare/main...joinalahmed:torch_tpu:explainability-example)

I cannot open the pull request yet: the repository allows users without write access 3 open PRs and mine (#6102, #6105, #6106) fill that quota. I will open it as soon as a slot frees.
