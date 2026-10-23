# Optimisation de FLUX { #flux-optimization }

<!-- Link to colab and to generated notebook -->

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/case_studies_flux.ipynb)

Cette étude de cas montre comment nous avons pris une implémentation PyTorch de
référence du `DoubleStreamBlock` du diffusion transformer
[FLUX](https://github.com/black-forest-labs/flux) et rendu une étape
d'entraînement **1,48x plus rapide** sur Cloud TPU avec TorchTPU, pour une
longueur de séquence de 32 768.

Cette page est un condensé du tutoriel complet. Elle met en avant le concept
clé de chaque optimisation à l'aide de courts extraits autonomes. Pour le code
complet du modèle, les benchmarks, les traces XProf et les tests, consultez :

*   [Tutoriel d'optimisation de FLUX](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md) :
    présentation complète de chaque jalon, avec les traces XProf et les
    instructions pour reproduire chaque benchmark.
*   [Écrire des kernels Pallas personnalisés pour PyTorch](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md) :
    présentation pas à pas du kernel Pallas SplashAttention utilisé dans
    le jalon 4.

L'implémentation s'appuie sur
l'[implémentation de FLUX dans TorchTitan](https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/models/flux/model/)
et sur le dépôt
[black-forest-labs/flux](https://github.com/black-forest-labs/flux/blob/main/src/flux/).

Si vous découvrez les exemples, familiarisez-vous d'abord avec leur exécution
sur GKE [à l'aide de nos autres documents](./get_started_on_gke.md).

## Résultats en bref { #results-at-a-glance }

Temps moyen d'une étape d'entraînement (passe avant + perte + passe arrière +
étape de l'optimiseur) pour un seul `DoubleStreamBlock` :

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Configuration | Temps moyen par étape | Accélération par rapport à la référence |
| :--- | ---: | ---: |
| Référence de base (eager) | 178,61 ms | 1,00x |
| Pallas SplashAttention seul (eager) | 149,81 ms | 1,19x |
| Vues zero-copy seules (eager) | 178,30 ms | 1,00x |
| `torch.compile` seul | 159,33 ms | 1,12x |
| Vues zero-copy + RoPE fusionné + Pallas SplashAttention (eager) | 156,73 ms | 1,14x |
| Vues zero-copy + `torch.compile` | 159,08 ms | 1,12x |
| Vues zero-copy + RoPE fusionné + `torch.compile` | 153,60 ms | 1,16x |
| **Vues zero-copy + RoPE fusionné + Pallas SplashAttention + `torch.compile`** | **122,51 ms** | **1,46x** |
<!-- mdformat on -->

**Configuration de mesure :**

-   **Matériel :** une seule puce TPU v6e (32 Gio de HBM).
-   **Logiciels :** Python 3.14, PyTorch 2.15 (préversion), JAX 0.12, einops
    0.8.2 et TorchTPU compilé à partir des sources.
-   **Forme du modèle :** `batch_size=2`, `hidden_size=1024`, `num_heads=8`
    (`head_dim=128`), `img_seq_len=32256`, `txt_seq_len=512` (longueur de
    séquence totale de 32 768).
-   **Précision :** paramètres du bloc et entrées en bfloat16.
-   **Tailles de bloc de SplashAttention :** 2048 pour `block_q`, `block_kv` et
    les tuiles dK/dV de la passe arrière.
-   **Chronométrage :** 3 étapes de préchauffage suivies de 10 étapes
    chronométrées. Les étapes chronométrées sont lancées l'une après l'autre et
    `torch.accelerator.synchronize()` s'exécute une seule fois après la
    dernière : le temps moyen par étape (temps total divisé par le nombre
    d'étapes) inclut donc l'exécution sur l'appareil. Le profileur est désactivé
    pendant les étapes chronométrées.

À cette longueur de séquence, l'attention domine l'étape : SplashAttention et
la compilation apportent donc les gains les plus importants. Pour des
séquences plus courtes, la compilation compte davantage (environ 3x pour une
longueur de séquence de 640) et le kernel d'attention compte moins.

Ces leçons s'appliquent à la plupart des modèles transformer sur TPU :

1.  **Évitez les copies de disposition cachées** : préférez les vues zero-copy
    (`.view()`, `torch.unbind()`) aux réorganisations qui matérialisent de
    nouveaux buffers en HBM.
2.  **Stockez moins, calculez plus** : exprimez les embeddings positionnels
    sous forme de calculs élément par élément compacts plutôt que de matrices
    matérialisées.
3.  **Compilez le bloc entier** : `torch.compile()` permet à XLA de fusionner
    les chaînes d'opérations élément par élément et de supprimer les
    allers-retours intermédiaires vers la HBM.
4.  **Recourez à des kernels personnalisés si nécessaire** : les kernels Pallas
    offrent un contrôle explicite du découpage en tuiles (tiling) sur la puce
    pour les opérations limitées par la mémoire, comme l'attention sur de
    longues séquences.

### Comment lire cette page { #how-to-read-this-page }

Chaque jalon est présenté sous la forme de deux sections :

-   **Avant (référence)** : l'implémentation d'origine et son goulot
    d'étranglement.
-   **Après (optimisé)** : l'implémentation optimisée, suivie d'une
    vérification qu'elle produit le même résultat que la référence.

Chaque cellule de code porte sur sa première ligne le commentaire
correspondant `# BEFORE (reference)` ou `# AFTER
(optimized)`.

**Prérequis :** la cellule de référence du jalon 1 utilise
[einops](https://github.com/arogozhnikov/einops) (`pip install einops`).

```python
import torch

device = torch.device("tpu")
print(f"Running on {device}.")
```

## Jalon 1 : vues de tensors zero-copy { #milestone-1-zero-copy-tensor-views }

Dans `DoubleStreamBlock`, les flux image et texte projettent des états cachés
de forme `(B, L, 3 * H * D)` en requêtes, clés et valeurs. La façon dont cette
projection est séparée en Q, K et V détermine si le TPU doit copier des données
en HBM pendant l'entraînement.

### Avant (référence) : `einops.rearrange` { #before-reference-einopsrearrange }

L'implémentation de référence déplace l'axe `3` (Q/K/V) en première position :

```python
img_q, img_k, img_v = einops.rearrange(
    img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads
)
```

Pendant la passe avant, `einops.rearrange` renvoie des vues à pas (strided
views) de la projection : aucune donnée n'est encore déplacée. Le coût apparaît
dans la **passe arrière** : pour faire remonter les gradients à travers la
réorganisation des axes, autograd matérialise le gradient dans un nouveau
buffer. Sur TPU, cela se traduit par une copie supplémentaire de HBM à HBM dans
chaque bloc d'attention (visible sous la forme d'une opération `copy` dans
XProf).

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

### Après (optimisé) : `.view()` + `torch.unbind()` { #after-optimized-view-torchunbind }

L'implémentation optimisée garde l'axe `3` adjacent à `(H, D)` et le sépare
avec `torch.unbind`. La passe arrière empile alors directement les gradients de
Q, K et V dans la disposition d'origine, sans copie supplémentaire. Dans le
tableau des résultats, l'effet sur le temps par étape est faible
(**178,61 ms à 178,30 ms** en mode eager, **159,33 ms à 159,08 ms** en mode
compilé), car la copie ne représente qu'une petite fraction de l'étape. Mais
sa suppression ne coûte rien.

> **Note:** Les cellules de ce jalon montrent seulement que les deux
> approches produisent les mêmes Q, K et V. Pendant la passe avant, les deux
> renvoient les mêmes vues à pas : leurs sorties semblent donc identiques. La
> copie HBM supplémentaire due à `einops.rearrange` n'a lieu que pendant la
> passe arrière. Elle est visible dans une trace XProf d'une étape
> d'entraînement complète, et non dans la sortie de la cellule. Consultez
> le [jalon 1 du tutoriel d'optimisation de FLUX](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-1)
> pour voir les traces.

```python
# AFTER (optimized): keep Q/K/V adjacent to (H, D) and unbind.
_b, _l, _ = qkv.shape
q_after, _k, _v = torch.unbind(
    qkv.view(_b, _l, 3, num_heads, head_dim), dim=2
)
print(f"q shape: {tuple(q_after.shape)}")
print(f"Matches reference: {torch.equal(q_after.cpu(), q_before.cpu())}")
```

## Jalon 2 : RoPE algébrique fusionné { #milestone-2-fused-algebraic-rope }

FLUX applique des Rotary Position Embeddings (RoPE) multi-axes aux requêtes et
aux clés. Les cellules ci-dessous implémentent les deux versions pour un seul
axe RoPE.

### Avant (référence) : matrices de rotation 2x2 { #before-reference-2x2-rotation-matrices }

L'implémentation de référence matérialise une matrice de rotation `2x2`
complète `[[cos,
-sin], [sin, cos]]` pour chaque paire de fréquences, et l'applique sous forme
de produit matrice-vecteur sur des tensors de requêtes/clés remodelés en 6D.
Cela stocke `cos` et `sin` deux fois, et les empilements et remodelages
supplémentaires apparaissent comme de gros blocs de copie dans XProf.

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

### Après (optimisé) : paires `[cos, sin]` compactes + rotation élément par élément { #after-optimized-compact-cos-sin-pairs-elementwise-rotation }

L'implémentation optimisée ne stocke que des paires `[cos, sin]` compactes et
applique directement la rotation par des calculs élément par élément :

```python
# (x0 * cos - x1 * sin, x0 * sin + x1 * cos)
xq_out_0 = xq_0 * cos - xq_1 * sin
xq_out_1 = xq_0 * sin + xq_1 * cos
```

Cela divise par deux l'empreinte mémoire de l'embedding positionnel, se traduit
en instructions de multiplication-addition fusionnée (FMA) et fusionne avec les
opérations de projection et d'attention environnantes lors de la compilation.
En plus des vues zero-copy et de `torch.compile`, cela réduit le temps par
étape de **159,08 ms à 153,60 ms (3,4 %)**.

Comme les deux versions effectuent les mêmes multiplications et additions sur
les mêmes opérandes, elles produisent des résultats identiques. La cellule
ci-dessous exécute la version optimisée sur les mêmes entrées et la compare à
la référence :

> **Note:** Ces cellules montrent le tensor de fréquences plus petit et le fait
> que les deux versions produisent des résultats identiques. L'accélération
> n'apparaît que lorsque RoPE s'exécute dans le `DoubleStreamBlock` compilé, où
> la rotation élément par élément fusionne avec les opérations environnantes.
> Consultez
> le [jalon 2 du tutoriel d'optimisation de FLUX](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-2)
> pour voir les traces XProf.

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

## Jalon 3 : compilation XLA du bloc entier { #milestone-3-whole-block-xla-compilation }

### Avant (référence) : exécution eager { #before-reference-eager-execution }

En mode eager, chaque opération PyTorch lance un programme TPU indépendant : les
activations intermédiaires font donc des allers-retours par la HBM et l'hôte
se bloque souvent sur le dispatch. Dans le tableau des résultats, le
`DoubleStreamBlock` de référence prend **178,61 ms** par étape en mode eager.

La cellule ci-dessous exécute le RoPE fusionné du jalon 2 en mode eager,
opération par opération :

```python
# BEFORE (reference): eager execution, one TPU program per op.
q_eager, k_eager = apply_rope(rope_xq, rope_xk, freqs)
print("Ran fused RoPE eagerly.")
```

### Après (optimisé) : `torch.compile()` { #after-optimized-torchcompile }

Compiler l'ensemble du `DoubleStreamBlock` avec `torch.compile()` l'abaisse en
un seul graphe XLA. Le compilateur peut alors fusionner les opérations élément
par élément, les normalisations et les additions résiduelles :

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

La compilation fait passer le temps par étape de **178,61 ms à 159,33 ms
(1,12x)**. Le gain est modeste pour une longueur de séquence de 32 768, car
l'attention domine l'étape. Pour une longueur de séquence de 640, la
compilation est environ 3x plus rapide.

Le même principe s'applique à n'importe quelle fonction. La cellule ci-dessous
compile le RoPE fusionné et vérifie qu'il correspond à l'exécution eager :

> **Note:** Ces cellules vérifient seulement que la compilation ne modifie pas
> le résultat. La fusion et l'accélération apparaissent dans XProf et dans les
> mesures du temps par étape du `DoubleStreamBlock` complet, et non dans la
> sortie de la cellule. Consultez
> le [jalon 3 du tutoriel d'optimisation de FLUX](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-3)
> pour comparer les traces eager et compilées.

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

## Jalon 4 : kernel Pallas SplashAttention personnalisé { #milestone-4-custom-pallas-splashattention-kernel }

Pour de longues séquences (par exemple, `S = 2048` ou `S = 4096`), l'attention
conjointe complète devient limitée par la mémoire. Ce jalon n'est présenté
que sous forme d'extraits. Consultez la
[présentation du kernel Pallas](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md)
pour l'implémentation complète et exécutable.

### Avant (référence) : SDPA PyTorch standard { #before-reference-standard-pytorch-sdpa }

```python
attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```

L'attention sans découpage en tuiles matérialise une matrice de scores en
`O(S^2)` dans la HBM.

### Après (optimisé) : Pallas SplashAttention { #after-optimized-pallas-splashattention }

```python
attn = splash_sdpa(q, k, v, is_causal=False)  # Drop-in replacement.
```

`splash_sdpa` est un kernel SplashAttention
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) personnalisé qui
calcule l'attention par tuiles sur la puce. Il s'appuie sur
l'[implémentation de SplashAttention dans TorchTitan](https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/experiments/tpu/kernels/splash_attention.py).

Le kernel est exposé à PyTorch avec `torch.tpu.pallas.jax_op`. Ses fonctions de
passe avant et arrière sont annotées avec des types `jax.Array`, à partir
desquels `jax_op` déduit le schéma de l'opérateur PyTorch personnalisé. La
passe avant enregistre le résidu `logsumexp` du softmax : le kernel de la passe
arrière n'a donc pas besoin de réexécuter le calcul de la passe avant :

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

> **Important:** **JAX n'est pas exécuté au runtime.** Pallas est distribué
> dans le package `jax` : le kernel est donc écrit avec la syntaxe `jax`.
> Cependant, `torch.tpu.pallas.jax_op` n'utilise JAX qu'une seule fois, au
> moment du traçage, avec des placeholders abstraits de forme et de dtype pour
> exporter le kernel sous forme de module MLIR. Pendant l'entraînement et
> l'inférence, en mode eager comme sous `torch.compile`, le kernel s'exécute
> directement sur des buffers TPU `torch.Tensor`.

**Résultats** : dans le tableau des résultats, SplashAttention seul est
**1,19x** plus rapide que la référence eager (de 178,61 ms à 149,81 ms). Avec
les vues zero-copy et le RoPE fusionné activés, il est plus rapide que la SDPA
standard en mode eager (**de 182,61 ms à 156,73 ms, 1,17x**) comme en mode
compilé (**de 153,60 ms à 122,51 ms, 1,25x**).

Ces résultats dépendent de la configuration du kernel pour la charge de
travail. Avec les blocs de 512 tokens par défaut, SplashAttention était environ
10 % plus lent que la SDPA standard avec compilation, pour la même longueur de
séquence, et les entrées float32 étaient encore plus lentes. Testez quelques
tailles de bloc pour votre longueur de séquence, exécutez le kernel en
bfloat16 et profilez avant de recourir à un kernel personnalisé.

**En savoir plus :**

*   **Exemple Pallas SplashAttention (utilisé dans cette étude de cas)** :
    [`splash_attention.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/splash_attention.py)
    contient l'intégration complète du kernel, et
    [`splash_attention_test.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/splash_attention_test.py)
    vérifie la parité de la passe avant et des gradients avec la SDPA de
    PyTorch. La
    [présentation du kernel](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md)
    explique pas à pas les tailles de bloc, les masques, l'enregistrement
    auprès d'autograd et les tests.
*   **Bases de Pallas** : [Kernels personnalisés avec Pallas](./custom_kernels.md)
    explique comment écrire des kernels Pallas avec TorchTPU.
