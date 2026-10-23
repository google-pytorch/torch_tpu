# Muse Glimmer 30B: トレーニング { #muse-glimmer-30b-training }

このケーススタディでは、スタンドアロンの Muse Glimmer 30B のサンプルを見ていきます。このサンプルでは、4 個の Ironwood TPU 上で FSDP を使用して、モデル全体のシンプルなトレーニング ループを示します。

GKE（Google Kubernetes Engine）クラスタ内の TPU でこのサンプルを実行する方法については、サンプルの [README](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop) をご覧ください。ここでは代わりに、実装上の選択のいくつかについて説明します。

この例では、シンプルさと読みやすさを重視しています。4 個の Ironwood TPU を備えた単一のホストのみで実行されます。当然ながら、これは現実的な事前トレーニングの規模ではありませんが、低コストで実行、反復、学習できる、実際に動作するサンプルを提供することを目的としています。

この純粋な PyTorch（カスタム カーネルなし）のコードは、シーケンス長 5120 で、ランクあたり約 2,400～2,800 TPS（合計で 19k～22k TPS）のトレーニング速度を達成しています。今後の更新で、TPU で実行する際のさらなる最適化とパフォーマンスのベスト プラクティスについて説明する予定です。

## モデルの定義 { #model-definition }
[Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) は、比較的標準的な Transformer アーキテクチャに従っています。因果的セルフ アテンションとフィードフォワード（FFN）ブロックからなる 52 個のレイヤで構成されています。モデルの 4 層ごとに 1 層がフル アテンションを使用し、それ以外のすべてのレイヤはスライディング ウィンドウ アテンション（SWA）を使用します。

モデルのコードは [model_fsdp.py](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/model_fsdp.py)（および回転位置エンコーディング用の [rope.py](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/rope.py)）で完全に自己完結しています。

構成は [model_fsdp.py](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/model_fsdp.py) で指定されています。

```python
class Config:
  # General model config
  vocab_size: int = 202048
  num_layers: int = 52
  dmodel: int = 6656
  dff: int = 19968
  dhead: int = 128
  num_q_heads: int = 32
  num_kv_heads: int = 2

  # Details
  swa_size: int = 2048
  # ... omitted ...

  # Perf evaluation config
  batch_size: int = 8  # Local (per rank) batch size
  seq_len: int = 5120
```

モデルのバックボーンの定義は、要するに次のようになります。

```python
self.embed_tokens = nn.Embedding(config.vocab_size, config.dmodel)
self.embed_norm = nn.RMSNorm(
    (config.dmodel,), eps=config.rmsnorm_eps, elementwise_affine=False
)
self.layers = nn.ModuleList(
    DecoderBlock(config, layer_idx=i) for i in range(config.num_layers)
)
self.norm = nn.RMSNorm((config.dmodel,), eps=config.rmsnorm_eps)
```

モデルのヘッド（「LM ヘッド」）がここには含まれていないことに注意してください。トレーニングの場合、通常は損失関数と一緒に配置するほうが適しています。後述のチャンク単位の融合された損失に関するセクションをご覧ください。

## レイヤごとの torch.compile { #per-layer-torchcompile }

TorchTPU は、`torch.compile` を使用したときに最高のパフォーマンスを発揮します。これは、XLA コンパイラが演算の融合、メモリの最適化、計算と通信の自動オーバーラップなど、さまざまな最適化を実行できるためです。

この例では、`torch.compile` をモデル全体ではなく、モデルの各レイヤに個別に適用する方法を示します。どちらのアプローチも問題なく機能しますが、モデル全体のコンパイルにはかなり長い時間がかかる場合があります。また、少なくともこのモデルでは、パフォーマンスの面で単一のレイヤの範囲を超えてコンパイルするメリットはありません。

```python
class MuseGlimmer30B(nn.Module):

  def __init__(self, config: Config):
    # ... omitted ...
    self.layers = nn.ModuleList(
        DecoderBlock(config, layer_idx=i) for i in range(config.num_layers)
    )

  def apply_torch_compile(self, backend):
    self.embed_norm = torch.compile(self.embed_norm, dynamic=False, backend=backend)
    self.layers = nn.ModuleList(
        torch.compile(layer, dynamic=False, backend=backend)
        for layer in self.layers
    )
    self.norm = torch.compile(self.norm, dynamic=False, backend=backend)
```

ここで、`dynamic=False` は動的シェイプを使用しないこと（トレーニング中はすべてのシェイプが静的）を意味します。また、`backend` 変数は、`"tpu"` バックエンドと `"inductor"`（CUDA 用）バックエンドを簡単に切り替えられるようにするためのものです。

TorchTPU は内部でコンパイル済みのレイヤをキャッシュします。このモデルにはレイヤの種類が 2 つ（フルと SWA）しかないため、コンパイル処理が大幅に高速化されます。

## アクティベーション チェックポイント { #activation-checkpointing }

モデルの重みは HBM を約 60 GiB（bf16）占有し、単一の Ironwood TPU TensorCore（96 GiB）に簡単に収まります。

しかし、トレーニング中は勾配とオプティマイザの状態（AdamW のモメンタムと分散など）のためのメモリも必要です。さらに重要なのは、autograd の機能に必要な中間アクティベーションがメモリ使用量の大部分を占めやすいことです。アクティベーション チェックポイント（再実体化、rematerialization とも呼ばれます）は、後者に対処するための非常に一般的な手法です。

モデルをレイヤごとにコンパイルしているため、アクティベーション チェックポイントも同じ粒度で適用するのが自然です。これは、標準の [`torch.utils.checkpoint`](https://docs.pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint) の機能を使用して行います。

```python
class DecoderBlock(nn.Module):
  # ... omitted ...

  def forward(self, h: torch.Tensor, rope_sin_cos: tuple) -> torch.Tensor:
    return torch.utils.checkpoint.checkpoint(
        self.forward_impl,
        h,
        rope_sin_cos,
        use_reentrant=False,
    )

  def forward_impl(self, h: torch.Tensor, rope_sin_cos: tuple) -> torch.Tensor:
    h = h + self.post_attention_layernorm(
        self.self_attn(self.input_layernorm(h), rope_sin_cos)
    )
    h = h + self.post_feedforward_layernorm(
        self.mlp(self.pre_feedforward_layernorm(h))
    )
    return h
```

> **Note:** アクティベーション チェックポイント（AC）はモデルのレイヤ（`DecoderBlock` モジュール）の `forward` メソッドの内側で適用され、`torch.compile` は外側、つまりレイヤ自体に適用されます。この順序が重要です。AC をレイヤの内側に保持したまま、`torch.compile` をモデル全体に適用することもでき、これは問題なく機能します。しかし、この 2 つを入れ替えることはできません。たとえば、AC を同じ方法で適用しつつ、`torch.compile` を内側のアテンションと FFN のサブレイヤに個別に適用すると、期待どおりに動作しません。

TorchTPU では、`torch.compile` は AC と特に相性が良いです。（グローバルな）バックワード パス中、各レイヤは（再実体化のために）小規模なフォワードとバックワードの結合パスを実行します。TPU バックエンドのコンパイラ（XLA）はフォワードとバックワードを結合した計算グラフ全体を認識できるため、追加の最適化を行い、中間テンソルのメモリ使用量を削減することもできます。

## 「ナイーブな」FSDP { #naive-fsdp }

この例では、ネイティブの PyTorch の集団通信演算を直接使用して、FSDP（fully-sharded data-parallel）トレーニングを行う最もシンプルな方法を示したいと考えました。

> **Tip:** このアプローチのさらにシンプルな説明については、[手動 FSDP のサンプル](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual)をご覧ください。

FSDP は、アテンションと FFN のすべての射影の重みを手動でシャーディングすることで行います。例:

```python
# QKV projections are fully sharded:
self.q_proj = nn.Linear(dm, num_q_heads * dh // world_size, bias=False)
```

> **Note:** PyTorch の `nn.Linear` レイヤは重み行列を転置して保存するため、ここでは 2 番目の次元でシャーディングしています。重みの実際のシェイプは `[num_q_heads * dh / world_size, dm]` になります。PyTorch の all-gather / reduce-scatter の集団通信は 1 番目の次元で動作するため、これは重要です。

次に、フォワード パス中に、PyTorch の関数型集団通信を使用して、重みを使用する前に収集します。

```python
# Functional collectives
import torch.distributed._functional_collectives as func_col

# ... all-gather:
q_proj_w = func_col.all_gather_single(self.q_proj.weight, gather_dim=0, group=self.pg)

# ... use full weight directly:
q = F.linear(h, q_proj_w).view(local_bsz, seq_len, num_q_heads, dh)
```

同じ考え方を、他のすべての射影（アテンションの Q/K/V/out/gate、FFN の up/gate/down）にも適用します。

重要なのは、`func_col.all_gather_single` のような関数型集団通信が PyTorch の autograd エンジンに登録されていることです。そのため、`.backward()` パス中に対応する reduce-scatter の集団通信が自動的に得られます（つまり、自分で記述する必要はありません）。

この実装は、現在のレイヤを実行している間に次のレイヤの重みをプリフェッチしようとしないという点で、比較的ナイーブであることに気付くでしょう。これでは、TPU TensorCore が集団通信の完了を待ってストールやアイドル状態にならないでしょうか？

レイヤ全体に `torch.compile` を適用しているため、TorchTPU のコンパイラ（XLA）は、フォワード パスと、フォワードとバックワードを結合したパスの両方で、集団通信の順序を変更し、SparseCore にオフロードして非同期で実行できます。この場合、これだけで通信と計算のほぼ完全なオーバーラップを実現できることがわかっています。

## チャンク単位で融合された LM ヘッドとクロスエントロピー損失 { #chunkwise-fused-lm-head-cross-entropy-loss }

この例では、デフォルトでローカル バッチサイズ 8 とシーケンス長 5120 を使用します。クロスエントロピーのトレーニング損失を単純に計算すると、シェイプ `[8, 5120, 202048]` の中間 logits テンソルを実体化する必要があります。最後の数値はモデルの語彙サイズです。bf16 では、これだけで約 16 GiB の HBM メモリが必要になり、無駄が大きすぎます。

この問題の一般的な解決策は、語彙行列の重みの matmul とクロスエントロピー損失を融合して、損失の計算をチャンク単位で行うことです。1 つのアプローチは語彙の次元に沿ってチャンク化すること（オンライン softmax が必要で、純粋な PyTorch で表現するのは困難です）で、もう 1 つのアプローチはシーケンスの次元に沿ってチャンク化することです。ここでは後者を行います。

```python
class MuseGlimmerChunkwiseFusedLMHeadLoss(nn.Module):
  # ... omitted ...

  for i in range(0, h.shape[0], chunk_size):
    h_chunk.data.copy_(h[i : (i + chunk_size)])
    tgt_chunk.data.copy_(tgt[i : (i + chunk_size)])

    self.fused_loss_fn_fwd_bwd(
        h_chunk, tgt_chunk, self.lm_head.weight, alpha, cap, loss_scaler
    )

    h_chunk_grads.append(h_chunk.grad)
    h_chunk.grad = None

  return torch.cat(h_chunk_grads, dim=0).view(bsz, seq_len, dm)
```

`fused_loss_fn_fwd_bwd` は、トークンの単一のチャンクに対して、融合された語彙の matmul、クロスエントロピー損失、バックワード パスを計算する静的関数を `torch.compile` でコンパイルしたバージョンを指します。

```python
@staticmethod
def _chunk_fused_lmhead_loss_fwd_bwd(
    h, tgt, lm_head_w, alpha, cap, loss_scaler
):
  logits = F.tanh(F.linear(h, lm_head_w) * (alpha / cap)) * cap
  loss = F.cross_entropy(input=logits, target=tgt, reduction="sum")
  loss *= loss_scaler
  # Note: in recent versions of PyTorch, .backward is no longer a graph break!
  loss.backward()
  return
```

このアプローチのメリットを最大限に得るために、チャンクに対してバックワード パスも実行します。最近のバージョンの PyTorch では、`loss.backward()` を呼び出してもグラフ ブレークが発生しなくなったため、TorchTPU のコンパイラ（XLA）はより多くの最適化と融合を実行できます。

## この例に含まれていないもの { #whats-missing-in-this-example }

この例では基本的なトレーニング ループのみを示しており、わかりやすさのために、実用的なトレーニング環境で重要となるさまざまな詳細を省略しています。

 - データの読み込み、トークン化
 - モデルのチェックポイントの読み込みと保存
 - ジョブの健全性モニタリング、収束の診断
 - フォールト トレランス

TorchTPU を使用した現実的な本番環境レベルのトレーニング環境については、[TorchTitan](https://github.com/google-pytorch/torchtitan/) をご覧ください。
