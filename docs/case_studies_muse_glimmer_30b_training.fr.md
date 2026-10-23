# Muse Glimmer 30B : entraînement { #muse-glimmer-30b-training }

Dans cette étude de cas, nous passons en revue notre exemple autonome
Muse Glimmer 30B. Il présente une boucle d'entraînement simple pour le modèle
complet, avec FSDP sur 4 TPU Ironwood.

Pour savoir comment exécuter cet exemple sur les TPU de votre cluster GKE
(Google Kubernetes Engine), consultez le
[README](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop)
de l'exemple. Ici, nous expliquons plutôt certains choix d'implémentation.

Cet exemple privilégie la simplicité et la lisibilité. Il s'exécute sur un
seul hôte équipé de 4 TPU Ironwood. Ce n'est évidemment pas une échelle
réaliste pour un pré-entraînement, mais notre objectif est de fournir un
exemple fonctionnel que vous pouvez exécuter, modifier et étudier à faible
coût.

Ce code en PyTorch pur (sans kernels personnalisés) atteint une vitesse
d'entraînement d'environ 2 400 à 2 800 TPS par rang (soit 19k à 22k TPS au
total), avec une longueur de séquence de 5 120. Dans de prochaines mises à
jour, nous prévoyons de décrire d'autres optimisations et bonnes pratiques de
performances pour l'exécution sur TPU.

## Définition du modèle { #model-definition }
[Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) suit
une architecture transformer assez standard.
Il se compose de 52 couches de blocs d'auto-attention causale et de
feed-forward (FFN). Une couche sur quatre du modèle utilise l'attention
complète, et toutes les autres utilisent l'attention à fenêtre glissante (SWA).

Le code du modèle est entièrement autonome dans
[model_fsdp.py](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/model_fsdp.py)
(et
[rope.py](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/rope.py)
pour l'encodage positionnel rotatif).

La configuration est définie dans
[model_fsdp.py](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/model_fsdp.py) :

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

La définition du backbone du modèle se résume à :

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

Notez que la tête du modèle (aussi appelée « LM head ») n'apparaît pas ici :
pour l'entraînement, il est généralement préférable de la placer avec la
fonction de perte. Consultez la section sur la perte fusionnée par chunks
ci-dessous.

## torch.compile par couche { #per-layer-torchcompile }

TorchTPU offre les meilleures performances avec `torch.compile`, grâce à la
capacité du compilateur XLA à effectuer diverses optimisations, comme la fusion
d'opérations, l'optimisation de la mémoire et le chevauchement automatique des
calculs avec les communications.

Cet exemple montre comment appliquer `torch.compile` séparément à chaque couche
du modèle (plutôt qu'au modèle entier). Les deux approches fonctionnent bien,
mais la compilation du modèle entier peut prendre nettement plus de temps et,
au moins pour ce modèle, compiler au-delà d'une seule couche n'apporte aucun
gain de performances.

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

Ici, `dynamic=False` signifie qu'il n'y a pas de formes dynamiques (toutes les
formes sont statiques pendant l'entraînement), et la variable `backend` sert
simplement à faciliter le passage entre les backends `"tpu"` et `"inductor"`
(pour CUDA).

En interne, TorchTPU met en cache les couches compilées. Comme ce modèle ne
comporte que deux types de couches (attention complète et SWA), la compilation
est fortement accélérée.

## Activation checkpointing { #activation-checkpointing }

Les poids du modèle occupent environ 60 Gio de HBM (en bf16) et tiendraient
facilement sur un seul TensorCore de TPU Ironwood (qui dispose de 96 Gio).

Toutefois, pendant l'entraînement, il faut aussi de la mémoire pour les
gradients et les états de l'optimiseur (par exemple, le momentum et la
variance d'AdamW). Surtout, l'utilisation de la mémoire peut facilement être
dominée par les activations intermédiaires nécessaires au fonctionnement
d'autograd. L'activation checkpointing (aussi appelé rematérialisation) est
une technique très courante pour résoudre ce dernier problème.

Comme nous compilons le modèle couche par couche, il est très naturel
d'appliquer l'activation checkpointing à la même granularité. Pour cela, nous
utilisons la fonctionnalité standard
[`torch.utils.checkpoint`](https://docs.pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint) :

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

> **Note:** L'activation checkpointing (AC) est appliqué dans la méthode
> `forward` de la couche du modèle (module `DecoderBlock`), et `torch.compile`
> est appliqué à l'extérieur, sur la couche elle-même. Cet ordre est important.
> Vous pourriez garder l'AC dans la couche et appliquer `torch.compile` au
> modèle entier : cela fonctionnerait bien.
> En revanche, vous ne pouvez pas les inverser. Par exemple, si vous appliquiez
> l'AC de la même façon, mais `torch.compile` à l'intérieur, séparément sur les
> sous-couches d'attention et de FFN, cela ne fonctionnerait pas comme prévu.

`torch.compile` fonctionne particulièrement bien avec l'AC dans TorchTPU :
pendant la passe arrière (globale), chaque couche effectue une petite passe
conjointe avant + arrière (en raison de la rematérialisation). Comme le
compilateur du backend TPU (XLA) voit l'ensemble du graphe de calcul conjoint
avant + arrière, il peut effectuer des optimisations supplémentaires, et même
réduire l'utilisation de la mémoire pour les tensors intermédiaires.

## FSDP « naïf » { #naive-fsdp }

Dans cet exemple, nous voulions montrer la façon la plus simple possible de
faire un entraînement FSDP (fully-sharded data-parallel), en utilisant
directement les opérations collectives natives de PyTorch.

> **Tip:** Pour une description encore plus simple de cette approche,
> consultez notre
> [exemple de FSDP manuel](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual).

Nous mettons en œuvre FSDP en shardant manuellement tous les poids de
projection de l'attention et du FFN, par exemple :

```python
# QKV projections are fully sharded:
self.q_proj = nn.Linear(dm, num_q_heads * dh // world_size, bias=False)
```

> **Note:** La couche PyTorch `nn.Linear` stocke sa matrice de poids
> transposée. C'est pourquoi nous shardons ici selon la 2e dimension. La forme
> réelle du poids sera `[num_q_heads * dh / world_size, dm]`. Ce point est
> important, car les opérations collectives all-gather / reduce-scatter de
> PyTorch opèrent sur la 1re dimension.

Ensuite, pendant la passe avant, nous utilisons les opérations collectives
fonctionnelles de PyTorch pour rassembler les poids avant de les utiliser :

```python
# Functional collectives
import torch.distributed._functional_collectives as func_col

# ... all-gather:
q_proj_w = func_col.all_gather_single(self.q_proj.weight, gather_dim=0, group=self.pg)

# ... use full weight directly:
q = F.linear(h, q_proj_w).view(local_bsz, seq_len, num_q_heads, dh)
```

Le même principe s'applique à toutes les autres projections (Q/K/V/out/gate de
l'attention, et up/gate/down du FFN).

Point essentiel : les opérations collectives fonctionnelles comme
`func_col.all_gather_single` sont enregistrées auprès du moteur autograd de
PyTorch. Pendant la passe `.backward()`, nous obtenons donc automatiquement les
opérations collectives reduce-scatter correspondantes (nous n'avons pas besoin
de les écrire nous-mêmes).

Vous remarquerez que cette implémentation est relativement naïve : elle
n'essaie pas de précharger les poids de la couche suivante pendant l'exécution
de la couche courante. Les TensorCores du TPU ne risquent-ils pas alors de
rester bloqués, inactifs, en attendant la fin des opérations collectives ?

Comme nous appliquons `torch.compile` à la couche entière, le compilateur de
TorchTPU (XLA) peut réordonner les opérations collectives et les décharger
vers le SparseCore pour les exécuter de façon asynchrone, aussi bien dans la
passe avant que dans les passes conjointes avant + arrière. Dans ce cas, cela
suffit pour obtenir un chevauchement quasi parfait entre communications et
calcul.

## LM head et perte d'entropie croisée fusionnées par chunks { #chunkwise-fused-lm-head-cross-entropy-loss }

Par défaut, l'exemple utilise une taille de batch locale de 8 et une longueur
de séquence de 5 120. Si nous calculions naïvement la perte d'entropie
croisée de l'entraînement, nous devrions matérialiser un tensor intermédiaire de
logits de forme `[8, 5120, 202048]`, où le dernier nombre est la taille du
vocabulaire du modèle. En bf16, ce tensor seul nécessiterait déjà environ
16 Gio de mémoire HBM, ce qui serait un gaspillage excessif.

Une solution courante consiste à calculer la perte par chunks, en fusionnant la
multiplication matricielle par la matrice de poids du vocabulaire avec la perte
d'entropie croisée. Une première approche découpe les chunks selon la
dimension du vocabulaire (ce qui nécessite un softmax en ligne et se révèle
difficile à exprimer en PyTorch pur). Une autre approche découpe selon la
dimension de séquence : c'est celle que nous utilisons ici.

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

`fused_loss_fn_fwd_bwd` désigne la version compilée avec `torch.compile` de la
fonction statique qui calcule, sur un seul chunk de tokens, la multiplication
matricielle par le vocabulaire, la perte d'entropie croisée et la passe
arrière, fusionnées :

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

Pour tirer le meilleur parti de cette approche, nous effectuons même la passe
arrière sur le chunk. Dans les versions récentes de PyTorch, l'appel à
`loss.backward()` ne provoque plus de rupture de graphe, ce qui permet au
compilateur de TorchTPU (XLA) d'effectuer davantage d'optimisations et de
fusions.

## Ce qui manque dans cet exemple { #whats-missing-in-this-example }

Cet exemple illustre uniquement la boucle d'entraînement de base et, par souci
de clarté, omet plusieurs aspects importants de toute configuration
d'entraînement réelle :

 - le chargement des données et la tokenisation ;
 - le chargement et l'enregistrement des checkpoints du modèle ;
 - la surveillance de l'état des jobs et les diagnostics de convergence ;
 - la tolérance aux pannes.

Pour une configuration d'entraînement réaliste, de niveau production, avec
TorchTPU, nous vous recommandons de consulter
[TorchTitan](https://github.com/google-pytorch/torchtitan/).
