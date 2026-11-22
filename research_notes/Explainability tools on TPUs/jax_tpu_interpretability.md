# JAX/Flax/TF interpretability and explainability tooling on TPUs (as of Oct 2026)

Method note: only WebFetch was available (no web search), so coverage is limited to pages I could fetch directly (GitHub READMEs, HF, arXiv abstract, DeepMind blog). Several guessed URLs 404'd. Fetched pages were summarized by a small model, so details such as dates and counts are second-hand. No Google Cloud blog or arXiv full-text sources were reached.

## What JAX/Flax/TF interpretability tools exist, and which are TPU-ready?

### Takeaway
A real but thin JAX-native interpretability stack exists, mostly from Google DeepMind (Penzai, Mishax, Gemma Scope tooling, Flax NNX capture/perturb). The JAX libraries run on TPU because JAX does, but none of the pages fetched advertise TPU-specific interpretability support. The TF-era tools (Lucid, tf-explain) are stale and Lucid is archived. The most-used SAE and attribution libraries (SAELens, nnsight, Inseq) are PyTorch-only.

### Cited Findings
- Penzai is a JAX library that treats models as pytrees. It offers `pz.select` for patching or ablating components and capturing activations, Treescope for visualization, and loadable Gemma/Llama/Mistral/GPT-NeoX transformers. The README does not mention TPUs. The repo was archived (read-only) on Aug 25, 2026, with no named successor. — [Penzai GitHub](https://github.com/google-deepmind/penzai); [Penzai discussions](https://github.com/google-deepmind/penzai/discussions)
- Mishax (DeepMind) is a JAX/Flax mech-interp utility: `ast_patcher` (source-level probes without forking), `safe_greenlet` (turns a forward pass into a loop that can read and replace internals), and a Gemma example and Colab. About 158 stars and 29 commits, no releases; the page does not mention TPU. — [Mishax](https://github.com/google-deepmind/mishax)
- Gemma Scope (Gemma 2 2B/9B/27B, JumpReLU SAEs, plus transcoders and attention/MLP/residual variants) is distributed as weights on Hugging Face. The model page points to SAELens for loading, a Colab on JumpReLU SAE training in both PyTorch and JAX, and Mishax as the internal tool used to expose Gemma 2 activations. No hardware info on the page. — [HF gemma-scope](https://huggingface.co/google/gemma-scope); [arXiv 2408.05147](https://arxiv.org/abs/2408.05147) (abstract; hardware details not visible in what I fetched)
- Gemma Scope 2 (published Dec 19, 2025) covers the whole Gemma 3 family (270M to 27B), with SAEs, transcoders, skip- and cross-layer transcoders, and Matryoshka training. DeepMind says it required storing about 110 PB of data and training more than 1 trillion total parameters. The blog gives no hardware or framework details. — [DeepMind blog](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/)
- The official `gemma` JAX library runs on CPU, GPU and TPU and supports Gemma 2, 3, 3n and 4. The README does not mention interpretability or activation capture. — [google-deepmind/gemma](https://github.com/google-deepmind/gemma)
- Flax NNX has built-in introspection primitives: `Module.sow` with `nnx.Intermediate`, `nnx.capture` (including `method_outputs=` to record every method's output), and `nnx.Perturbation` / `perturb` for taking gradients with respect to intermediates. This is the closest NNX analogue to Linen's `capture_intermediates`/`intercept_methods`, and it works with `nnx.vmap`. — [Flax NNX guide](https://flax.readthedocs.io/en/latest/guides/extracting_intermediates.html)
- LIT (Learning Interpretability Tool, PAIR) is framework-agnostic (TF, PyTorch and others), covers sequence salience and prompt debugging for LLMs, and is described as under active development. No TPU mention on the README; accelerators are referenced only via prebuilt container images. — [PAIR-code/lit](https://github.com/PAIR-code/lit)
- PAIR `saliency` is framework-agnostic at its core (IG, XRAI, SmoothGrad, Grad-CAM, Occlusion, Blur IG). It has TF1 wrappers, a TF2 example and a PyTorch notebook. JAX and TPU are not mentioned. — [PAIR-code/saliency](https://github.com/PAIR-code/saliency)
- Lucid (TF1 only, "not currently supporting tensorflow 2") was archived on Apr 10, 2024. — [tensorflow/lucid](https://github.com/tensorflow/lucid)
- tf-explain (TF 2.x callbacks for Grad-CAM, etc.) is not archived. Its README pins tensorflow==2.6.0 and lists Python 3.6-3.8, which suggests it is dated. I could not confirm its last commit date. — [sicara/tf-explain](https://github.com/sicara/tf-explain)
- PyTorch-only competitors with no JAX/TPU mention on their pages: SAELens (PyTorch, TransformerLens/HF/NNsight integration, loads Gemma Scope per its docs), nnsight (any `torch.nn.Module`, remote via NDIF), Inseq (Captum-based, HF Transformers). — [SAELens](https://github.com/jbloomAus/SAELens); [nnsight](https://github.com/ndif-team/nnsight); [Inseq](https://github.com/inseq-team/inseq)

### Inferences
- "TPU-ready" here means "JAX-based, so XLA-compilable". I found no tool that documents TPU-specific testing for interpretability workloads.
- The archiving of Penzai (Aug 2026) leaves Flax NNX's native capture/perturb plus Mishax as the main maintained DeepMind-adjacent JAX introspection options. Mishax's maintenance status is unclear (no releases).
- Inseq-like token-attribution tooling and Captum-style gradient attribution have no JAX/TPU equivalent that I found. Hugging Face SAE/attribution ecosystems are PyTorch-centric. That is the clearest area where a PyTorch/TorchTPU project would fill a gap rather than duplicate.
- SAE inference/loading (Gemma Scope via SAELens) is PyTorch-based, so a TorchTPU project overlaps with that, not with the JAX tooling.

### Gaps
- No web search was available, so I did not survey other JAX SAE or mech-interp libraries (e.g. community JAX SAE trainers, TransformerLens-like JAX ports, Equinox-based tools). I could not confirm whether any exist.
- Did not verify Gemma Scope's training hardware (likely in the arXiv full text; not fetched).
- The gemma_scope and gemma_scope_2 GitHub URLs I guessed returned 404, so the official code repos were not examined.
- Last-commit dates for LIT, tf-explain, Mishax and saliency were not visible.

## Known explainability work on TPUs at scale

### Takeaway
The only large-scale open interpretability effort I found is Gemma Scope / Gemma Scope 2 from DeepMind. Its scale is documented, but the fetched pages never say it ran on TPUs. Treat TPU involvement as plausible (Google, JAX tooling like Mishax) but unconfirmed.

### Cited Findings
- Gemma Scope 2: about 110 PB stored, more than 1 trillion total SAE/transcoder parameters, across Gemma 3 270M-27B. Hardware not stated. — [DeepMind blog, Dec 2025](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/)
- The Gemma Scope abstract says the high cost of training a comprehensive SAE suite limits outside-industry research. No hardware stated in the abstract. — [arXiv 2408.05147](https://arxiv.org/abs/2408.05147)
- The Gemma Scope training tutorial uses JAX and PyTorch implementations of JumpReLU SAEs. — [HF gemma-scope](https://huggingface.co/google/gemma-scope)

### Inferences
- Because Mishax and the JAX Gemma stack are the stated activation-extraction path, large SAE training at DeepMind probably used JAX infrastructure. This is an inference; no source confirms TPUs.

### Gaps
- No Google Cloud blog posts, papers or case studies on explainability workloads (attribution, SAE training, circuit analysis) on TPUs were found. I could not search for them.
- Older material (2019-2021 TF/Lucid-era Google Brain work, e.g. on TPUs) was not examined.

## Pain points reported for attribution/interpretability workloads on TPUs

### Takeaway
I found no direct reports of interpretability-specific pain on TPUs. Only adjacent signals exist, and they are weak.

### Cited Findings
- Flax NNX's `perturb` mechanism requires the perturbation tree to be initialized with the same batch structure as the consuming step when using `vmap`. This is a usability constraint for gradient-of-intermediate attribution. — [Flax NNX guide](https://flax.readthedocs.io/en/latest/guides/extracting_intermediates.html)
- Mishax's README warns that careless use of source-level patching may hurt maintainability. — [Mishax](https://github.com/google-deepmind/mishax)
- A search of pytorch/xla issues for "captum" returned 0 results. A search for "hook" returned 102 hits, with titles that mostly concerned other topics (e.g. `scan` only capturing aten ops, #8691, Feb 2025; "TPU hangs on too large tensor", #9502, Jul 2025). None were explicitly about gradient hooks or attribution. — [pytorch/xla issues "captum"](https://github.com/pytorch/xla/issues?q=captum); [pytorch/xla issues "hook"](https://github.com/pytorch/xla/issues?q=is%3Aissue+hook)
- A JAX issue (#41791) reports silent BF16 gather corruption in a fusion involving a debug callback on libtpu 0.0.48. It is a correctness risk for activation-inspection code that uses debug callbacks on TPU, but a single report and I did not read the thread. — [jax issues search](https://github.com/google/jax/issues?q=TPU+hooks+OR+gradients+OR+attribution)

### Inferences
- Hook-based interception (PyTorch forward/backward hooks) and dynamic shapes in variable-length attribution loops (e.g. integrated gradients over many steps) are likely friction points under XLA's compile-then-run model. The absence of Captum/Inseq issues on pytorch/xla more likely reflects low usage than a smooth experience. Both are inferences, not sourced.
- JAX's functional pytree model (Penzai, Flax capture) avoids hooks entirely, which is a structural advantage there.

### Gaps
- No forum threads, blog posts or papers describing attribution/SAE-inference pain on TPUs were found. A targeted search of Reddit, Hugging Face forums, Google Cloud docs, and TorchTPU/PyTorch-XLA issue discussions is needed.
