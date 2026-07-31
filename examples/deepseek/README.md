# DeepSeek Inference

This folder contains a lightly adapted version of the official DeepSeek v3
inference code.

## Running aten ops tracer

blaze run //examples/deepseek:trace -- --alsologtostderr

Results:

```
I0805 03:13:24.519075 1267057 trace.py:65] Ops for aten namespace(45 ops):
  aten.arange.start_step
  aten.silu.default
  aten.resolve_neg.default
  aten.div.Tensor
  aten.full.default
  aten.arange.default
  aten.split_with_sizes.default
  aten.unsqueeze.default
  aten.topk.default
  aten.to.dtype
  aten.mul.Tensor
  aten.ones_like.default
  aten.ones.default
  aten.flatten.using_ints
  aten.select.int
  aten.softmax.int
  aten.zeros.default
  aten.polar.default
  aten.add.Tensor
  aten.add_.Tensor
  aten.eq.Scalar
  aten.index_put_.default
  aten.gather.default
  aten.pow.Scalar
  aten.bincount.default
  aten.where.default
  aten.einsum.default
  aten.view_as_complex.default
  aten.mul_.Tensor
  aten.triu_.default
  aten.resolve_conj.default
  aten.rms_norm.default
  aten.reciprocal.default
  aten.type_as.default
  aten.slice.Tensor
  aten.view.default
  aten.index.Tensor
  aten.zeros_like.default
  aten.embedding.default
  aten.linear.default
  aten.empty.memory_format
  aten.alias.default
  aten.copy_.default
  aten.view_as_real.default
  aten.squeeze.dim
```

## Files

*   `model.py`: a very lightly adapted version from the official version from
    [DeepSeek](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)
*   `trace.py`: traces aten ops in the forward pass of the DeepSeek model.

## Inference only

This code is inference only. DeepSeek has not released the source code to train
DeepSeek, though their prior work on MoE via DeepEp was likely used.

## DeepSeek V3's novel techniques

The rest of this doc describes novel techniques of the DeepSeek V3 LLM.

https://arxiv.org/html/2412.19437v1

## FP8

DeepSeek relies on FP8 to store weights. DeepSeek uses E4M3 (the 8th bit is
sign). The key to FP8 is that the value isn't represented in just those 8 bits!

A group of FP8 values is multiplied by an FP32 scaling factor. Nvidia H100s
natively support a single FP32 scaling factor per tensor, but DeepSeek instead
has more scaling factors: one per 128x128 block. Nvidia B200s supports block
scaling natively, sometimes called MXFP8.

The code includes a naive bf16 approach, a lighter fp8 approach that uses a
Triton kernel to unscale to bf16 and then use normal CUDA to matmul in bf16, and
a third kernel to do the matmul with FP8 inputs and outputs (though the internal
format accumulates in a higher format, TODO).

### TPU MXU

Note that TPU MXUs are 256x256. This mismatch may be an issue. You can force a
TPU do two 128x128 @ 128x128 matmuls on a single MXU using the mathematical
trick:

$$\begin{bmatrix}W_1 & 0\\0 & W_2\end{bmatrix}\begin{bmatrix}X_1 & 0\\
0 & X_2\end{bmatrix}=\begin{bmatrix}W_1 X_1 & 0\\0 &W_2 X_2\end{bmatrix}$$

## Multi-headed Latent Attention

Unlike a pure Multi-Headed Attention (MHA), DeepSeek uses MLA, which basically
applies LoRA to the QKV calculation.

LoRA basically approximates an MN matrix by the MK@KN. A matrix with $$M \times
N$$ elements can be approximated with only $$M \times K + N \times K$$ elements.

This creates specific einsum patterns that need to be validated.

The notation below is non-standard and can be ignored, but helped me think
through the einsums.

*   `---` means the dimension is in both operands, and retained.
*   `...` means the dimension is in both operands, and reduced.
*   `<text>` means the dimension is only in one operand.

### Naive attention.

Ignoring b and h, essentially sd @ dt.

```
bshd,bthd->bsht:

b: --- batch
s: query sequence (source)
h: --- heads
d: ... hidden dimension (external)

b: ---
t: key sequence (target)
h: ---
d: ...
```

### First half of a partial attention score calculation on q_nope.

This allows the wkv_b to be dequantized from FP8. The intermediate dimension is
c. Notice that this weight and the next weight, hdc and btc, could be combined
into bthd of the naive einsum.

```
bshd,hdc->bshc:

b: batch
s: query sequence (source)
h: --- heads
d: ... hidden dimension (external)

h: ---
d: ...
c: nope slice of hidden dimension
```

### Projects NoPE slice to partial attention score result.

Second half of the partial attention score calculation of q_nope. Reduce c,
"outer product" of s and t.

```
bshc,btc->bsht:

b: --- batch
s: query sequence (source)
h: heads
c: ... NoPE slice of hidden dimension

b: ---
t: key sequence (target)
c: ...
```

### Partial score calculation and application 1/3

```
bshr,btr->bsht:

b: --- batch
s: query sequence
h: heads
r: ... rope embedding dimension

b: ---
t: target dimension
r: ...
```

### Naive score application

```
bsht,bthd->bshd

The first tensor are scores, the second
is the value matrix.

Ignoring b and h, the simplified pattern st, td -> sd,
an obvious matmul reducing along t (as in MK @ KN = MN).

b: --- batch
s: sequence length (source)
h: --- heads
t: ... weighting per Target value

b: --- batch
t: ... actual target value
h: --- heads
d: hidden dimension
```

### Partial score application 2/3

bsht,btc->bshc

### Partial score application 3/3

bshc,hdc->bshd

## Rope positional embeddings

This is a pretty lightweight trick. Instead of trainable or cosine embeddings,
DeepSeek uses RoPE. The hidden dimension, or native depth of embeddings between
multi-headed attention (MHA) blocks, is 2048. Internally, the depth is 3072, but
only 1/3rd get transformed by the RoPE transform. The remaining 2/3rd is
untouched. These two parts are *concatenated* together, not added as in most
combinations of positional embeddings and token embeddings.

RoPE is easiest to implement using imaginary numbers, not a commonly used set of
ops in deep learning.

## Sharding

Since this model is inference only, DDP and FSDP do not apply. Instead, the
model relies on TP and EP. Note that the data is *replicated* across ranks. The
parallelism is entirely model parallel.

### TP

The model has manual collectives for TP: a simple column and row parallel, with
the hard coded pattern that column is naive (leaves a sharded output) and row
has an all_reduce collective.

### EP

Without the complexity of DP/FSDP, each batch is replicated across each rank,
and each rank simply only calculates local experts. The end of the MoE layer
includes an all-reduce to combine all experts.

**There is no all-to-all collective, which would used in training***.

## FF

The feedforward network is not a simple two-layer perceptron, but rather swiglu
from Google's PALM paper. `self.w2(F.silu(self.w1(x)) * self.w3(x))`.

https://arxiv.org/abs/2204.02311

## MoE Shared Experts

The shared experts are implemented with a single MLP, because

$$W_{\text{ab2, x}} \cdot (W_{\text{ab1, x}} \cdot x) =
W_{\text{a2, x}} \cdot (W_{\text{a1, x}} \cdot x) + W_{\text{b2,x }} \cdot (W_{\text{b1, x}}
\cdot x)$$

in inference. This may or may not be true during training, depending on whether
there was a topk / dropout operator after each shared expert to isolate their
behavior.

```python
self.shared_experts = MLP(
    args.dim, args.n_shared_experts * args.moe_inter_dim
)
```
