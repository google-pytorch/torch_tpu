# Prior art: explainability/attribution (Captum etc.) on TPUs, PyTorch/XLA, TorchTPU

Research date: 2026-10-30. Method caveat: no web-search tool was available. I used the GitHub REST search API (unauthenticated; hit the rate limit partway through), the arXiv API, discuss.pytorch.org search.json, and direct page fetches. Bing and DuckDuckGo returned junk or a bot check, and Kaggle search is a JS page that returned nothing. Stack Overflow, Reddit, Medium, dev.to, Colab and Kaggle notebooks were NOT effectively searched. Negative results below are therefore strong for GitHub/arXiv/discuss.pytorch.org and weak for the rest.

## Has anyone built or attempted Captum (or other attribution) on PyTorch/XLA or TorchTPU?

### Takeaway
I found no repo, issue, forum thread or paper that runs Captum on PyTorch/XLA or TorchTPU. GitHub, discuss.pytorch.org and arXiv searches came back empty for that combination. The only TPU-plus-XAI prior art is hardware-acceleration research on TPUs, not Captum or PyTorch workflows. A "Captum on TorchTPU" project looks novel as far as these sources show, with the caveat that blogs, notebooks, Stack Overflow and Reddit were not covered.

### Cited Findings
- GitHub repo search for "captum tpu", "captum xla", "captum torch_xla", "integrated gradients tpu", "explainability torch_xla", "torch_tpu captum", "torchtpu explainability", "torchtpu integrated gradients", "torchtpu interpretability", "shap tpu", "attribution torch_xla", "integrated gradients xla" and "TorchTPUSprint" each returned 0 repositories — GitHub search API (queries run 2026-10-30), e.g. https://api.github.com/search/repositories?q=captum+tpu
- GitHub issue/PR search for `repo:meta-pytorch/captum TPU` returned 0, and `repo:pytorch/xla captum` returned 0 — https://api.github.com/search/issues?q=repo:meta-pytorch/captum+TPU&per_page=10
- A GitHub web search of the Captum repo issues for "TPU OR XLA" showed "0 results" (the repo had 35 open issues at that time). Captum now lives at meta-pytorch/captum — https://github.com/meta-pytorch/captum/issues
- discuss.pytorch.org search returned empty results for "captum tpu", "captum xla" and "captum TPU OR xla integrated gradients" — https://discuss.pytorch.org/search.json?q=captum%20tpu
- arXiv query for "integrated gradients" AND TPU returned a single paper, ApproXAI (below). Query for attribution AND "Tensor Processing" returned 0 — https://export.arxiv.org/api/query?search_query=all:%22integrated+gradients%22+AND+all:TPU
- The TPU+XAI papers that exist are hardware-acceleration work, not PyTorch tooling:
  - "Hardware Acceleration of Explainable Machine Learning using Tensor Processing Units" (arXiv 2103.11927, 2021) reports roughly 25x faster classification and 13x faster interpretation, using the convolution/Fourier relationship — https://export.arxiv.org/api/query?search_query=all:explainable+AND+all:TPU&max_results=15
  - "Hardware Acceleration of Explainable Artificial Intelligence" (Zhixin Pan, Prabhat Mishra; arXiv 2305.04887, 2023) claims to be the "first attempt" to use TPUs to accelerate XAI, with 39x interpretation-time and 69x energy gains. The abstract does not name the XAI algorithms, and I could not confirm whether it used real TPUs or a simulation. The page notes text overlap with 2103.11927 — https://arxiv.org/abs/2305.04887
  - "ApproXAI: Energy-Efficient Hardware Acceleration of Explainable AI using Approximate Computing" (arXiv 2504.17929, 2025) maps integrated gradients, model distillation and Shapley analysis to approximate matrix operations on TPU-based edge devices, about 2x energy efficiency — https://arxiv.org/abs/2504.17929
  - "Explainable AI-Guided Efficient Approximate DNN Generation for Multi-Pod Systolic Arrays" (arXiv 2503.16583) uses XAI to guide approximation, not XAI on TPU as a workload — same arXiv query as above.
- None of these papers is a drop-in PyTorch/Captum workflow. I only read abstracts, not full texts.

### Inferences
- Academic "XAI on TPU" work exists, so the idea of accelerating attribution on TPUs is not new in principle. The novelty gap is a practical, open-source, PyTorch-native path (Captum on TorchTPU).
- The empty results may partly reflect the tooling limits above, so the claim of novelty should be stated as "no prior art found in GitHub/arXiv/PyTorch forum searches".

### Gaps
- Kaggle and Colab notebooks, Medium/dev.to blogs, Stack Overflow and Reddit were not searched effectively (no working search access).
- GitHub code search (needs auth) was not run, so forks or gists that merely import captum with torch_xla would not show up.
- Google Scholar and the JAX ecosystem (e.g. attribution libraries for JAX on TPU) were not searched.

## TorchTPU community projects and #TorchTPUSprint

### Takeaway
TorchTPU is an active, public Google repo, and a visible community is forming around it. Every community repo I found is about serving, benchmarking or training, none about explainability. No #TorchTPUSprint entries were found.

### Cited Findings
- google-pytorch/torch_tpu is a public Apache 2.0 PyTorch backend for TPUs with custom ATen kernels, a compilation cache, torch.compile() integration, and data and tensor parallelism. Latest PyPI release 0.2.0, about 802 stars, 75 forks, 44 open issues, 244 open PRs, 4,757 commits. Python 3.11+. Its README does not mention Captum, explainability, autograd behavior or hooks, and does not mention a sprint — https://github.com/google-pytorch/torch_tpu
- Repos found by "torchtpu" search (all inference/serving/training/benchmark oriented): vllm-project/vllm-torchtpu, sgl-project/sglang-torchtpu ("SGLang's TorchTPU Plugin"), kowalski-4831/vllm-torchtpu, ericehanley/sony-torchtpu-demo, angelayi/torchtpu_parse, yossfmedhat/torchtpu-rag-bench (embedding/reranking benchmarks, TorchTPU vs CPU), xbill9/torchtpu-dev (Gemma 4 serving and QAT experiments), billylo1/nanochat-torchtpu, lynnl0927/verl-torchtpu-clone, rushabh-46/torchtpu-vllm-old — https://api.github.com/search/repositories?q=torchtpu
- The xbill9/torchtpu-dev README does not mention Captum, attribution or interpretability. Its RESOURCES.md collects TorchTPU announcements, docs and recipes (not read) — https://github.com/xbill9/torchtpu-dev
- A GitHub issue/PR search for "TorchTPUSprint" returned 0, and a repo search returned 0 — https://api.github.com/search/issues?q=TorchTPUSprint
- torch_tpu issues and PRs about autograd/gradients show active backward-pass work and open op-coverage gaps, which matter for gradient-based attribution. Examples: issue 6070 (`aten::linalg_cross.out` not implemented, open, 2026-10-30); issue 5961 (leaky_relu backward gives the full gradient at 0, open); issue 4141 (`avg_pool1d` silently returned a tensor outside the autograd graph, closed) — https://github.com/google-pytorch/torch_tpu/issues/6070 , https://github.com/google-pytorch/torch_tpu/issues/5961 , https://github.com/google-pytorch/torch_tpu/issues/4141
- No torch_tpu issue matched captum; the 215-match search for hook/autograd/gradient terms returned only op-level items — https://api.github.com/search/issues?q=repo:google-pytorch/torch_tpu+hook+OR+autograd+OR+gradient+OR+captum

### Inferences
- Gradient correctness and op coverage on TorchTPU are still being fixed. A Captum port would double as a useful correctness stress test, and would also be exposed to these gaps.
- Absence of sprint entries probably means they are not indexed in GitHub, not that none exist; the sprint may live on social media or a Google site not searched.

### Gaps
- Could not find the #TorchTPUSprint announcement or entries (social posts, Devpost, Google blogs not searchable here).
- Did not read the TorchTPU user guide (torchtpu.readthedocs.io) for hook, `register_forward_hook` or `torch.autograd.grad` support statements.
- Whether TorchTPU supports the Captum mechanisms (forward/backward hooks, `torch.autograd.grad` with `create_graph`, dynamic batch shapes) is untested in this research.

## Known workarounds (e.g. moving the model to CPU) and why

### Takeaway
I found no documented, sourced workaround thread for attribution on TPU specifically. The CPU fallback is plausible but is an inference, not a finding. The nearest precedent is Captum support for other non-CUDA accelerators.

### Cited Findings
- Captum issue "Error using Integrated Gradients on MPS" (#1240, closed 2024) is the closest precedent of non-CUDA accelerator friction in Captum — https://github.com/meta-pytorch/captum/issues/1240 (title and status only; I did not read the thread)
- Captum docs describe device handling only for DataParallel, with no general device-selection guidance — https://captum.ai/api/integrated_gradients.html

### Inferences
- Users on TPU likely fall back to CPU or GPU for attribution, since no TPU-specific Captum path is documented anywhere I looked. This is unverified.

### Gaps
- No Stack Overflow, Reddit or forum evidence of people explicitly moving models to CPU for attribution on TPU. Not searched effectively, or does not exist.
- Reasons (lazy-tensor recompilation from varying shapes, hooks, unsupported ops) are hypotheses, not sourced.

## Documented performance cost of attribution methods

### Takeaway
Captum's own docs and issue tracker show attribution cost scales linearly with n_steps times batch size, and users repeatedly hit memory and speed limits. I found no hard timing benchmarks in these sources.

### Cited Findings
- Integrated Gradients defaults to n_steps=50. Total evaluations are n_steps times the number of examples. With `internal_batch_size=None` all evaluations run in a single batch, and `internal_batch_size` splits them into sequential chunks. `additional_forward_args` tensors are repeated n_steps times — https://captum.ai/api/integrated_gradients.html
- Captum issue #1350: "GradientShap needs internal_batch_size argument to avoid out-of-memory errors" (open since 2024-09) — https://github.com/meta-pytorch/captum/issues/1350
- Captum issue #1430: "How much memory required for LLMGradientAttribution.attribute() on LLama2 tutorial?" (closed) — https://github.com/meta-pytorch/captum/issues/1430
- Further IG scaling/compatibility reports exist: #1271 "Integrated Gradients for Llama2", #1168 "Using Integrated Gradients with Resnet50", #794 "Issues with BERT-type model" — GitHub search of meta-pytorch/captum, 118 matches for IG with slow/memory/n_steps terms: https://api.github.com/search/issues?q=captum+%22integrated+gradients%22+slow+OR+memory+OR+%22n_steps%22+is:issue+repo:meta-pytorch/captum
- The TPU-XAI papers motivate acceleration on the grounds that many XAI methods need "many costly iterations" and are unsuitable for real-time use (2305.04887) — https://arxiv.org/abs/2305.04887
- Their speedup figures (25x/13x, 39x/69x) are authors' claims on their own frameworks and are not comparable to a Captum run.

### Inferences
- IG with 50 to several hundred steps is embarrassingly batchable and matrix-multiply dominated, so a fit with TPU throughput is plausible. LLM attribution is a likely motivating use case given the memory issues above.

### Gaps
- No concrete benchmark numbers (seconds per image or per prompt on GPU/CPU) were retrieved. A real baseline would need to be measured or sourced from the Captum LLM-attribution paper or tutorials.
- The Vertex AI Explainable AI page (https://docs.cloud.google.com/vertex-ai/docs/explainable-ai/overview) was not read; it could show how Google itself runs IG/XRAI/sampled Shapley and their cost guidance.
