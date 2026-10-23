# FLUX の最適化 { #flux-optimization }

<!-- Link to colab and to generated notebook -->

[![Colab で開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/case_studies_flux.ipynb)

このケーススタディでは、[FLUX](https://github.com/black-forest-labs/flux) 拡散 Transformer の `DoubleStreamBlock` の PyTorch リファレンス実装を、シーケンス長 32,768 の条件で、Cloud TPU 上の TorchTPU を使ってトレーニング ステップを **1.48 倍高速化**した方法を説明します。

このページは、完全版チュートリアルの要約版です。短く自己完結したスニペットで、各最適化の背景にある主要な概念を紹介します。モデルの完全なコード、ベンチマーク、XProf トレース、テストについては、次をご覧ください。

*   [FLUX 最適化チュートリアル](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md): XProf トレースとすべてのベンチマークを再現する手順を含む、各マイルストーンの完全なウォークスルーです。
*   [PyTorch 用のカスタム Pallas カーネルを作成する](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md): マイルストーン 4 で使う Pallas SplashAttention カーネルの詳しいウォークスルーです。

この実装は、[TorchTitan による FLUX の実装](https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/models/flux/model/)と [black-forest-labs/flux](https://github.com/black-forest-labs/flux/blob/main/src/flux/) リポジトリに基づいています。

サンプルを初めて使う場合は、[他のドキュメント](./get_started_on_gke.md)で GKE 上でサンプルを実行する方法を確認しておくとよいでしょう。

## 結果の概要 { #results-at-a-glance }

単一の `DoubleStreamBlock` の平均トレーニング ステップ時間（フォワード パス + 損失 + バックワード パス + オプティマイザ ステップ）は次のとおりです。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 構成 | 平均ステップ時間 | ベースラインに対する高速化 |
| :--- | ---: | ---: |
| ベースラインのリファレンス（eager） | 178.61 ms | 1.00x |
| Pallas SplashAttention のみ（eager） | 149.81 ms | 1.19x |
| ゼロコピー ビューのみ（eager） | 178.30 ms | 1.00x |
| `torch.compile` のみ | 159.33 ms | 1.12x |
| ゼロコピー ビュー + 融合 RoPE + Pallas SplashAttention（eager） | 156.73 ms | 1.14x |
| ゼロコピー ビュー + `torch.compile` | 159.08 ms | 1.12x |
| ゼロコピー ビュー + 融合 RoPE + `torch.compile` | 153.60 ms | 1.16x |
| **ゼロコピー ビュー + 融合 RoPE + Pallas SplashAttention + `torch.compile`** | **122.51 ms** | **1.46x** |
<!-- mdformat on -->

**測定環境:**

-   **ハードウェア:** 単一の TPU v6e チップ（32 GiB HBM）。
-   **ソフトウェア:** Python 3.14、PyTorch 2.15（プレリリース版）、JAX 0.12、einops 0.8.2、ソースからビルドした TorchTPU。
-   **モデルのシェイプ:** `batch_size=2`、`hidden_size=1024`、`num_heads=8`（`head_dim=128`）、`img_seq_len=32256`、`txt_seq_len=512`（合計シーケンス長 32,768）。
-   **精度:** ブロックのパラメータと入力は bfloat16。
-   **SplashAttention のブロックサイズ:** `block_q`、`block_kv`、dK/dV のバックワード パスのタイルはすべて 2048。
-   **計測方法:** ウォームアップ 3 ステップの後に 10 ステップを計測します。計測対象のステップは連続してディスパッチされ、最後のステップの後に `torch.accelerator.synchronize()` を 1 回実行するため、平均ステップ時間（合計時間をステップ数で割った値）にはデバイスでの実行時間が含まれます。計測対象のステップの間、プロファイラは無効にしています。

このシーケンス長ではステップの大部分をアテンションが占めるため、SplashAttention とコンパイルの効果が最も大きくなります。シーケンス長が短い場合は、コンパイルの効果がより大きく（シーケンス長 640 で約 3 倍）、アテンション カーネルの効果は小さくなります。

ここで得られた教訓は、TPU 上のほとんどの Transformer モデルに当てはまります。

1.  **隠れたレイアウト コピーを避ける**: 新しい HBM バッファを実体化する並べ替えよりも、ゼロコピー ビュー（`.view()`、`torch.unbind()`）を優先します。
2.  **保存を減らし、計算を増やす**: 位置埋め込みを、実体化した行列ではなく、コンパクトな要素ごとの演算で表現します。
3.  **ブロック全体をコンパイルする**: `torch.compile()` を使うと、XLA が要素ごとの演算の連鎖を融合し、HBM との中間的な往復をなくせます。
4.  **必要に応じてカスタム カーネルを使う**: Pallas カーネルを使うと、長いシーケンス長でのアテンションのようなメモリバウンドな演算について、オンチップのタイリングを明示的に制御できます。

### このページの読み方 { #how-to-read-this-page }

各マイルストーンは、2 つのセクションの組で構成されています。

-   **Before（リファレンス）**: 元の実装とそのボトルネックです。
-   **After（最適化後）**: 最適化した実装と、それがリファレンスと同じ結果を出すことの確認です。

コードセルの 1 行目には、対応する `# BEFORE (reference)` または `# AFTER
(optimized)` コメントが付いています。

**前提条件:** マイルストーン 1 のリファレンス セルでは [einops](https://github.com/arogozhnikov/einops)（`pip install einops`）を使います。

```python
import torch

device = torch.device("tpu")
print(f"Running on {device}.")
```

## マイルストーン 1: ゼロコピーのテンソル ビュー { #milestone-1-zero-copy-tensor-views }

`DoubleStreamBlock` では、画像ストリームとテキスト ストリームが、シェイプ `(B, L, 3 * H * D)` の隠れ状態をクエリ、キー、バリューに射影します。この射影を Q、K、V にどう分割するかによって、トレーニング中に TPU が HBM 内でデータをコピーする必要があるかどうかが決まります。

### Before（リファレンス）: `einops.rearrange` { #before-reference-einopsrearrange }

リファレンス実装では、`3`（Q/K/V）の軸を先頭に移動します。

```python
img_q, img_k, img_v = einops.rearrange(
    img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads
)
```

フォワード パスでは、`einops.rearrange` は射影のストライド付きビューを返すため、この時点ではデータは移動しません。コストが現れるのは**バックワード パス**です。軸の並べ替えを通して勾配を戻すために、autograd は新しいバッファに勾配を実体化します。これは TPU 上で、各アテンション ブロックにおける余分な HBM 間コピーとして実行されます（XProf では `copy` 演算として表示されます）。

```python
# BEFORE (reference): einops.rearrange moves the Q/K/V axis to the front.
import einops

num_heads, head_dim = 4, 32
qkv = torch.randn(2, 16, 3 * num_heads * head_dim, device=device)

q_before, _k, _v = einops.rearrange(
    qkv, "B L (K H D) -> K B L H D", K=3, H=num_heads
)
print(f"q shape: {tuple(q_before.shape)}")
```

### After（最適化後）: `.view()` + `torch.unbind()` { #after-optimized-view-torchunbind }

最適化した実装では、`3` の軸を `(H, D)` に隣接させたまま、`torch.unbind` で分割します。こうするとバックワード パスで Q、K、V の勾配が元のレイアウトに直接スタックされるため、余分なコピーは不要になります。結果の表では、コピーはステップ全体のごく一部にすぎないため、ステップ時間への効果は小さい（eager モードで **178.61 ms から 178.30 ms**、コンパイル時に **159.33 ms から 159.08 ms**）ですが、コストをかけずに取り除けます。

> **Note:** このマイルストーンのセルが示すのは、2 つの方法で同じ Q、K、V が得られることだけです。フォワード パスではどちらも同じストライド付きビューを返すため、出力は同一に見えます。`einops.rearrange` による余分な HBM コピーはバックワード パスでのみ発生し、セルの出力ではなく、トレーニング ステップ全体の XProf トレースで確認できます。トレースについては、[FLUX 最適化チュートリアルのマイルストーン 1](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-1) をご覧ください。

```python
# AFTER (optimized): keep Q/K/V adjacent to (H, D) and unbind.
_b, _l, _ = qkv.shape
q_after, _k, _v = torch.unbind(
    qkv.view(_b, _l, 3, num_heads, head_dim), dim=2
)
print(f"q shape: {tuple(q_after.shape)}")
print(f"Matches reference: {torch.equal(q_after.cpu(), q_before.cpu())}")
```

## マイルストーン 2: 融合された代数的 RoPE { #milestone-2-fused-algebraic-rope }

FLUX は、クエリとキーに複数軸の Rotary Position Embeddings（RoPE）を適用します。以下のセルでは、単一の RoPE 軸について両方のバージョンを実装します。

### Before（リファレンス）: 2x2 回転行列 { #before-reference-2x2-rotation-matrices }

リファレンス実装では、周波数のペアごとに完全な `2x2` 回転行列 `[[cos,
-sin], [sin, cos]]` を実体化し、シェイプを変えた 6D のクエリ/キー テンソルに対して行列ベクトル積として適用します。この方法では
`cos` と `sin` を 2 回ずつ保存することになり、余分なスタックとシェイプ変更が XProf で大きなコピーブロックとして現れます。

```python
# BEFORE (reference): materialize 2x2 rotation matrices, then contract.
def rope_ref(pos, dim, theta=10000):
  """Returns rotation matrices of shape (B, S, D/2, 2, 2)."""
  scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
  omega = 1.0 / (theta**scale)
  out = torch.einsum("...n,d->...nd", pos, omega)
  out = torch.stack(
      [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)],
      dim=-1,
  )
  return out.reshape(*out.shape[:-1], 2, 2).float()

def apply_rope_ref(xq, xk, freqs_cis):
  """6D matrix-vector contraction against the rotation matrices."""
  xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
  xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
  xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
  xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
  return (
      xq_out.reshape(*xq.shape).type_as(xq),
      xk_out.reshape(*xk.shape).type_as(xk),
  )

_b, _s, _h, rope_head_dim = 2, 64, 4, 32
positions = (
    torch.arange(_s, dtype=torch.float32, device=device)
    .unsqueeze(0)
    .expand(_b, _s)
)
rope_xq = torch.randn(_b, _s, _h, rope_head_dim, device=device)
rope_xk = torch.randn(_b, _s, _h, rope_head_dim, device=device)

# Add a broadcast head dimension: (B, S, 1, D/2, 2, 2).
_freqs_ref = rope_ref(positions, rope_head_dim).unsqueeze(2)
q_rope_ref, k_rope_ref = apply_rope_ref(rope_xq, rope_xk, _freqs_ref)

print(f"Rotation matrix freqs shape: {tuple(_freqs_ref.shape)}")
```

### After（最適化後）: コンパクトな `[cos, sin]` ペア + 要素ごとの回転 { #after-optimized-compact-cos-sin-pairs-elementwise-rotation }

最適化した実装では、コンパクトな `[cos, sin]` ペアだけを保存し、要素ごとの演算で回転を直接適用します。

```python
# (x0 * cos - x1 * sin, x0 * sin + x1 * cos)
xq_out_0 = xq_0 * cos - xq_1 * sin
xq_out_1 = xq_0 * sin + xq_1 * cos
```

これにより位置埋め込みのメモリ使用量が半分になり、融合積和（FMA）命令に対応付けられ、コンパイル時には周囲の射影やアテンションの演算と融合されます。ゼロコピー ビューと `torch.compile` に加えてこれを適用すると、ステップ時間が **159.08 ms から 153.60 ms（3.4%）** に短縮されます。

どちらのバージョンも同じオペランドに対して同じ乗算と加算を行うため、同一の結果が得られます。次のセルでは、同じ入力で最適化バージョンを実行し、リファレンスと比較します。

> **Note:** これらのセルが示すのは、周波数テンソルが小さくなることと、両方のバージョンで同一の結果が得られることです。高速化が現れるのは、コンパイルされた `DoubleStreamBlock` の中で RoPE が実行され、要素ごとの回転が周囲の演算と融合される場合だけです。XProf トレースについては、[FLUX 最適化チュートリアルのマイルストーン 2](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-2) をご覧ください。

```python
# AFTER (optimized): store compact [cos, sin] pairs, rotate elementwise.
def rope(pos, dim, theta=10000):
  """Returns compact [cos, sin] pairs of shape (B, S, D/2, 2)."""
  scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
  omega = 1.0 / (theta**scale)
  out = torch.einsum("...n,d->...nd", pos, omega)
  return torch.stack([torch.cos(out).float(), torch.sin(out).float()], dim=-1)

def apply_rope(xq, xk, freqs_cis):
  """Direct 2D algebraic rotation with elementwise math."""
  cos = freqs_cis[..., 0]
  sin = freqs_cis[..., 1]
  b, s, h, d = xq.shape
  xq_ = xq.float().view(b, s, h, d // 2, 2)
  xk_ = xk.float().view(b, s, h, d // 2, 2)
  xq_0, xq_1 = xq_[..., 0], xq_[..., 1]
  xk_0, xk_1 = xk_[..., 0], xk_[..., 1]
  xq_out = torch.stack(
      [xq_0 * cos - xq_1 * sin, xq_0 * sin + xq_1 * cos], dim=-1
  ).view(b, s, h, d)
  xk_out = torch.stack(
      [xk_0 * cos - xk_1 * sin, xk_0 * sin + xk_1 * cos], dim=-1
  ).view(b, s, h, d)
  return xq_out.type_as(xq), xk_out.type_as(xk)

# Add a broadcast head dimension: (B, S, 1, D/2, 2).
freqs = rope(positions, rope_head_dim).unsqueeze(2)
_q_new, _k_new = apply_rope(rope_xq, rope_xk, freqs)

print(f"Compact freqs shape:         {tuple(freqs.shape)}")
print(
    "Max |q_ref - q_new|:        "
    f" {(q_rope_ref.cpu() - _q_new.cpu()).abs().max().item():.3e}"
)
print(
    "Max |k_ref - k_new|:        "
    f" {(k_rope_ref.cpu() - _k_new.cpu()).abs().max().item():.3e}"
)
torch.testing.assert_close(
    _q_new.cpu(), q_rope_ref.cpu()
)  # TORCH_ASSERT_CLOSE_OK=<Using standard PyTorch assert for user facing documentation only.>
torch.testing.assert_close(
    _k_new.cpu(), k_rope_ref.cpu()
)  # TORCH_ASSERT_CLOSE_OK=<Using standard PyTorch assert for user facing documentation only.>
```

## マイルストーン 3: ブロック全体の XLA コンパイル { #milestone-3-whole-block-xla-compilation }

### Before（リファレンス）: eager 実行 { #before-reference-eager-execution }

eager モードでは、PyTorch の演算ごとに独立した TPU プログラムが起動されるため、中間のアクティベーションが HBM を往復し、ホストはディスパッチで頻繁にストールします。結果の表では、ベースラインの `DoubleStreamBlock` は eager モードで 1 ステップあたり **178.61 ms** かかります。

次のセルでは、マイルストーン 2 の融合 RoPE を eager で、演算ごとに実行します。

```python
# BEFORE (reference): eager execution, one TPU program per op.
q_eager, k_eager = apply_rope(rope_xq, rope_xk, freqs)
print("Ran fused RoPE eagerly.")
```

### After（最適化後）: `torch.compile()` { #after-optimized-torchcompile }

`torch.compile()` で `DoubleStreamBlock` 全体をコンパイルすると、ブロックが単一の XLA グラフに lowering され、コンパイラが要素ごとの演算、正規化、残差加算を融合できるようになります。

```python
block = DoubleStreamBlock(
    hidden_size=1024,
    num_heads=8,
    mlp_ratio=4.0,
    config=OptimizationConfig(
        use_zero_copy_view=True,
        use_fused_rope=True,
        compile=True,
    ),
)
compiled_block = block.maybe_compile()  # Wraps the block in torch.compile().
```

コンパイルによって、ステップ時間は **178.61 ms から 159.33 ms（1.12 倍）** になります。シーケンス長 32,768 ではアテンションがステップの大部分を占めるため、効果は控えめです。シーケンス長 640 では、コンパイルにより約 3 倍高速になります。

同じパターンはどの関数にも適用できます。次のセルでは融合 RoPE をコンパイルし、eager 実行と結果が一致することを確認します。

> **Note:** これらのセルが確認するのは、コンパイルしても結果が変わらないことだけです。融合と高速化は、セルの出力ではなく、`DoubleStreamBlock` 全体の XProf とステップ時間の測定に現れます。eager とコンパイル後のトレースの比較については、[FLUX 最適化チュートリアルのマイルストーン 3](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-3) をご覧ください。

```python
# AFTER (optimized): compile into a single fused XLA graph.
compiled_apply_rope = torch.compile(apply_rope, backend="tpu")
_q_compiled, _k_compiled = compiled_apply_rope(rope_xq, rope_xk, freqs)

torch.testing.assert_close(
    _q_compiled.cpu(), q_eager.cpu()
)  # TORCH_ASSERT_CLOSE_OK=<Using standard PyTorch assert for user facing documentation only.>
torch.testing.assert_close(
    _k_compiled.cpu(), k_eager.cpu()
)  # TORCH_ASSERT_CLOSE_OK=<Using standard PyTorch assert for user facing documentation only.>
print("Compiled fused RoPE matches eager execution.")
```

## マイルストーン 4: カスタム Pallas SplashAttention カーネル { #milestone-4-custom-pallas-splashattention-kernel }

シーケンス長が長い場合（例: `S = 2048` や `S = 4096`）、完全なジョイント アテンションはメモリバウンドになります。このマイルストーンはスニペットのみで示します。完全で実行可能な実装については、[Pallas カーネルのウォークスルー](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md)をご覧ください。

### Before（リファレンス）: 標準の PyTorch SDPA { #before-reference-standard-pytorch-sdpa }

```python
attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```

タイル化されていないアテンションは、`O(S^2)` のスコア行列を HBM 内に実体化します。

### After（最適化後）: Pallas SplashAttention { #after-optimized-pallas-splashattention }

```python
attn = splash_sdpa(q, k, v, is_causal=False)  # Drop-in replacement.
```

`splash_sdpa` は、[TorchTitan による SplashAttention の実装](https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/experiments/tpu/kernels/splash_attention.py)に基づく、オンチップのタイル単位でアテンションを計算するカスタムの [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) SplashAttention カーネルです。

このカーネルは `torch.tpu.pallas.jax_op` を使って PyTorch に公開されます。フォワード パスとバックワード パスの関数には `jax.Array` 型のアノテーションが付いており、`jax_op` はそこから PyTorch カスタム演算子のスキーマを推論します。フォワード パスでは softmax の `logsumexp` 残差を保存するため、バックワード パスのカーネルでフォワード パスの計算を再実行する必要はありません。

```python
torch_fwd_fn = torch.tpu.pallas.jax_op(f"pallas::{op_fwd_name}", splash_fn)
torch_bwd_fn = torch.tpu.pallas.jax_op(f"pallas::{op_bwd_name}", splash_bwd_fn)


def setup_context(ctx, inputs, output):
  q, k, v = inputs
  out, lse = output
  ctx.save_for_backward(q, k, v, out, lse)


def backward(ctx, grad_out, grad_lse):
  del grad_lse
  q, k, v, out, lse = ctx.saved_tensors
  return torch_bwd_fn(q, k, v, out, lse, grad_out)


torch_fwd_fn.register_autograd(backward, setup_context=setup_context)
```

> **Important:** **JAX は実行時には実行されません。** Pallas は `jax` パッケージに含まれて配布されているため、カーネルは `jax` の構文で記述します。ただし、`torch.tpu.pallas.jax_op` が JAX を使うのはトレース時の 1 回だけで、抽象的な shape/dtype のプレースホルダを使ってカーネルを MLIR モジュールとしてエクスポートします。トレーニングと推論の間は、eager モードでも `torch.compile` でも、カーネルは `torch.Tensor` の TPU バッファ上で直接実行されます。

**結果**: 結果の表では、SplashAttention のみで eager のベースラインより **1.19 倍**高速です（178.61 ms から 149.81 ms）。ゼロコピー ビューと融合 RoPE を有効にした場合、eager（**182.61 ms から 156.73 ms、1.17 倍**）とコンパイル（**153.60 ms から 122.51 ms、1.25 倍**）の両方のモードで、標準の SDPA より高速になります。

これらの結果は、ワークロードに合わせてカーネルを設定することが前提です。デフォルトの 512 トークンのブロックでは、同じシーケンス長でのコンパイル時に SplashAttention は標準の SDPA より約 10% 遅く、float32 の入力ではさらに遅くなりました。カスタム カーネルを使う前に、使用するシーケンス長でいくつかのブロックサイズをベンチマークし、カーネルを bfloat16 で実行し、プロファイリングを行ってください。

**詳細:**

*   **Pallas SplashAttention のサンプル（このケーススタディで使用）**: [`splash_attention.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/splash_attention.py) にはカーネル統合の完全なコードが含まれており、[`splash_attention_test.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/splash_attention_test.py) は PyTorch SDPA とのフォワード パスと勾配の一致を検証します。[カーネルのウォークスルー](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md)では、ブロックサイズ、マスク、autograd の登録、テストを順を追って説明しています。
*   **Pallas の基本**: [Pallas を使用したカスタム カーネル](./custom_kernels.md)では、TorchTPU で Pallas カーネルを記述する方法を紹介しています。
