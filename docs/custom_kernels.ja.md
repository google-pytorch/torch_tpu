# TorchTPU のカスタム カーネル { #torchtpu-custom-kernels }

TorchTPU は標準の PyTorch フレームワークの演算子（[`torch.nn.functional.scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) など）をネイティブに高速化しますが、モデル固有の演算を高速化したり、演算を融合してメモリ帯域幅のオーバーヘッドを減らしたり、標準の PyTorch 演算子でまだカバーされていない新しいアーキテクチャを実装したりするために、カスタム カーネルの統合が必要になる場合があります。

## 統合のアプローチを選択する { #choosing-an-integration-approach }

TorchTPU では、ユースケースに応じて、TPU でカスタム カーネルを使用する 3 つの方法を提供しています。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| アプローチ | ステータス | 最適な用途 |
| :---- | :---- | :---- |
| **Tokamax** | 本番環境対応 | カーネル コードを記述せずに、一般的な LLM/MoE のビルディング ブロックをそのまま高速化します。 |
| **Pallas（`torch.tpu.pallas.jax_op`）** | 本番環境対応 | JAX/Pallas で独自の最高性能の TPU カーネルを記述し、PyTorch から呼び出します。 |
| **Helion** | 試験運用版 | PyTorch ネイティブの DSL を使用して、アクセラレータ間（GPU、TPU など）で移植可能なカーネルを記述します。 |
<!-- mdformat on -->

## 1. Tokamax のビルド済みカーネルを使用する { #1-using-pre-built-kernels-from-tokamax }

カスタム カーネルを一から記述する前に、その演算子が [Tokamax](http://openxla.org/tokamax) で利用できるかどうかを確認してください。Tokamax は、PyTorch とネイティブに相互運用できる高性能な TPU カーネルのライブラリです。

### 使用する場面 { #when-to-use }

モデルで次のような標準的な高性能ビルディング ブロックが必要な場合は、Tokamax を使用します。

*   **Mixture-of-Experts（MoE）:** グループ化された行列乗算、トークンの gather と scatter。
*   **Attention のバリエーション:** Paged Attention、Gated DeltaNet、Kimi Delta Attention、Multi-head Latent Attention（MLA）。
*   **選択:** 最適化された `TopK` 計算。

### 仕組み { #how-it-works }

Tokamax のカーネルは JAX/Pallas で記述されており、統合された自動チューニング（タイルサイズの自動探索など）、低オーバーヘッドの XProf ベンチマーク、TPU の世代をまたいだ回帰テストに支えられています。内部では、Tokamax は `torch.tpu.pallas.jax_op` を使用してフレームワーク間の相互運用性を透過的に処理し、ネイティブの PyTorch API をモデルに直接公開します。

## 2. カスタム Pallas カーネルを記述する { #2-writing-custom-pallas-kernels }

PyTorch や Tokamax でカバーされていない独自の演算が必要な場合は、[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) で TPU カーネルを記述し、`torch.tpu.pallas.jax_op` を使用して PyTorch 演算子として登録できます。

### Pallas カーネルを統合する方法 { #how-to-integrate-a-pallas-kernel }

1.  入力と出力のメモリ参照（`Ref`）を使用して **Pallas カーネルを定義します**。
2.  `pl.pallas_call` を使用して **カーネルを JAX 関数でラップし**、出力のシェイプと dtype を指定します。
3.  `torch.tpu.pallas.jax_op` を使用して **JAX 関数を TorchTPU に登録します**。

```python
import jax
from jax.experimental import pallas as pl
import torch

# 1. Define the Pallas kernel
def add_vectors_pallas_kernel(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = x + y

# 2. Wrap the Pallas kernel into a JAX function
def add_vectors_jax(x: jax.Array, y: jax.Array) -> jax.Array:
  return pl.pallas_call(
      add_vectors_pallas_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  )(x, y)

# 3. Wrap it for PyTorch using the functional interface
add_vectors_fn = torch.tpu.pallas.jax_op(
    "pallas::add_vectors",
    add_vectors_jax,
)

# 4. Call it like a standard PyTorch function
x = torch.tensor([1.0, 2.0, 3.0], device="tpu")
y = torch.tensor([4.0, 5.0, 6.0], device="tpu")
z = add_vectors_fn(x, y)
```

## 3. Helion で移植可能なカーネルを記述する（試験運用版） { #3-writing-portable-kernels-with-helion-experimental }

アクセラレータ間（TPU、GPU など）で移植可能なカーネルが必要な場合は、カーネル記述用の PyTorch ネイティブ DSL である [Helion](https://helionlang.com/) を使用できます。

> **Note:** TorchTPU での Helion のサポートは現在 **試験運用版** です。本番環境のワークロードでは、Tokamax または `torch.tpu.pallas.jax_op` を使用してください。

### Helion カーネルを記述する方法 { #how-to-write-a-helion-kernel }

関数を `@helion.kernel` でデコレートします。ホスト側のセットアップには標準の PyTorch 演算を使用し、分割されたタイルに対するデバイスの実行ループを定義するには `hl.tile` を使用します。

```python
import helion
import helion.language as hl
import torch

# 1. Define the kernel in Helion
@helion.kernel
def add_vectors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  # Standard PyTorch host-side setup
  x, y = torch.broadcast_tensors(x, y)
  out = torch.empty_like(x)

  # Device execution loop over automatically partitioned tiles
  for tile in hl.tile(out.size()):
    out[tile] = x[tile] + y[tile]

  return out

# 2. Call it like a standard PyTorch function
x = torch.tensor([1.0, 2.0, 3.0], device="tpu")
y = torch.tensor([4.0, 5.0, 6.0], device="tpu")
z = add_vectors(x, y)
```

Helion の詳細については、[Helion のドキュメント](https://helionlang.com/)をご覧ください。質問は [PyTorch デベロッパー ディスカッション フォーラム](https://dev-discuss.pytorch.org/c/helion/31)で受け付けています。問題は [GitHub で報告](https://github.com/pytorch/helion/issues)してください。
