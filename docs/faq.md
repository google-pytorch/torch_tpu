# TorchTPU FAQ

## What is the difference between PyTorch/XLA and TorchTPU?

Pytorch/XLA historically served as a bridging framework, heavily dependent on
JAX runtime components and specialized APIs. TorchTPU offers a native Pytorch
backend for [Google Cloud TPUs](https://cloud.google.com/tpu) and zero JAX
runtime dependencies, and existing PyTorch workloads run out of the box with
trivial code changes (e.g., `.to("cuda")` → `.to("tpu")`)

## What is the difference between TorchAX and TorchTPU?

TorchAX (also known as torch_xla2) is an implementation that intercepts PyTorch
operations at the Python level and lowers them into XLA (often via JAX). While
TorchAX has been used heavily for inference benchmarking and prototyping,
TorchTPU is the new official standard for a drop-in, PyTorch-native experience
on TPUs.

## What is happening to PyTorch/XLA?

PyTorch/XLA is deprecated and replaced by
[TorchTPU](https://github.com/google-pytorch/torch_tpu), which is now public.

## Is TorchTPU replacing JAX?

No. TorchTPU is designed to complement the TPU ecosystem by providing a native
PyTorch entry point to Cloud TPUs. It allows developers with existing PyTorch
codebases to run directly on TPUs with zero or minimal rewrites, eliminating the
need to rewrite models in JAX or manage separate codebases for GPU and TPU
infrastructure.

*   **When to think of TorchTPU?** If you have existing PyTorch codebase, want
    multi-hardware portability (GPU/TPU) without refactoring, using
    PyTorch-native frameworks (e.g., TorchTitan, vLLM, SGLang, Hugging Face), or
    wants rapid GPU-to-TPU migration.

## What is the best way for me to connect with TorchTPU developers?

You can share feedback by emailing
[torchtpu-feedback@google.com](mailto:torchtpu-feedback@google.com) and we
welcome contributions directly to the repo:
[https://github.com/google-pytorch/torch_tpu](https://github.com/google-pytorch/torch_tpu).
You can also use the tag "TPU" at
[discuss.pytorch.org](https://discuss.pytorch.org/).

## Will there be any community engagement forums?

Yes, community engagement is extremely important for us, and we'll be sharing
details for such forums. Additionally, we plan to host monthly office hours.

## What support does TorchTPU provide for kernel development?

Refer to [TorchTPU Custom Kernel support](./custom_kernels.md).

## Which TPU generations are actively supported? What's the plan for future hardware support?

TorchTPU actively supports v5e, v5p, Trillium (TPU v6e) and Ironwood (TPU v7x).
Validation and optimization for TPU v8 and future chips is ongoing as part of
our roadmap.

## Where can we find TorchTPU's supported features & future roadmap?

See [Supported features](./feature_set.md) for the current feature set and
planned upcoming feature set.

## Is there an agent to make my onboarding easier?

Check out the TPU agent suite (Experimental):
[https://github.com/google-pytorch/tpu-agent-suite](https://github.com/google-pytorch/tpu-agent-suite)

The TPU Agent Suite is an open-source plugin delivering specialized sub-agents
to automate onboarding, diagnostics, and inference performance tuning for
PyTorch models on Cloud TPU v6e and v7x.

## Can I use TorchTPU with our custom serving solution?

Yes. While TorchTPU offers out-of-the-box integration with industry-standard
serving engines like vLLM and SGLang, custom or in-house serving solutions can
integrate with TorchTPU via native PyTorch execution APIs and lower-level
compiler backends. For advanced orchestration, customers can also deploy custom
serving setups on GKE with llm-d.

## Why is Hugging Face `generate()` slow on TPU, and how do I speed it up?

By default, Hugging Face Transformers' `model.generate()` uses a `DynamicCache`,
whose key/value tensors grow by one position per new token. In eager mode every
new shape needs new XLA programs the first time it is seen, and decoding runs op
by op, so it is slow even once those programs are cached.

Pass `cache_implementation="static"`. Transformers then preallocates a
fixed-size KV cache and, because its auto-compile check accepts `tpu` devices,
compiles the decode step with `torch.compile`:

```python
outputs = model.generate(
    **inputs, max_new_tokens=32, cache_implementation="static"
)
```

For example, greedy 32-token generation with `Qwen/Qwen3-0.6B` in `bfloat16` on
TPU v6e (torch 2.13.0, TorchTPU 0.2.0, libtpu 0.0.50, Transformers 5.19.0)
measured:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->

| KV cache                        | First call                         | Repeat call, same shapes |
| ------------------------------- | ---------------------------------- | ------------------------ |
| Default `DynamicCache`, eager   | 388 XLA compilations, 171 ms/token | 64 ms/token              |
| `cache_implementation="static"` | about 12 s, mostly compilation     | 6 ms/token               |

<!-- mdformat on -->

Trade-offs:

*   The first call pays the compilation cost, so warm up before timing.
*   The cache size is fixed at prompt length plus `max_new_tokens`, and
    attention runs over the whole cache. A longer prompt or a larger
    `max_new_tokens` gives a new size and a recompile. To avoid this, pad
    prompts to a few fixed lengths, or set `max_cache_len` to the largest call
    you expect.
*   Auto-compile on TPU needs Transformers v5; v4 only auto-compiles on CUDA.
*   If you pass your own `compile_config`, set `backend="tpu"`. Its default,
    `"inductor"`, is rejected for TPU inputs.

To learn how TorchTPU handles growing shapes under `torch.compile` in general,
see [Dynamic shapes](./dynamic_shapes.md).
