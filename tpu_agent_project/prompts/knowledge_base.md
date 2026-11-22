Integrated Gradients expands the input to n_steps copies in one batched
forward/backward pass, so its peak memory scales with n_steps. For sizing
that on the TPU we have, answer with citations (do not answer from memory):

1. How much HBM does one TPU v6e (Trillium) chip have, and what is its
   HBM bandwidth?
2. How many chips are in a v6e-4 host, and how much HBM is that in total?
3. Does torch_tpu expose a memory API (allocated / reserved / peak) for the
   "tpu" device? Name it if a source states it.

Save the answer, with one footnoted URL per claim, as Markdown to
results/knowledge_base_answer.md in this workspace. Do not run anything on the
TPU VM for this request.
