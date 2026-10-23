# Kernels personnalisés dans TorchTPU { #torchtpu-custom-kernels }

TorchTPU accélère nativement les opérateurs standards du framework PyTorch
(comme
[`torch.nn.functional.scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)).
Vous pouvez toutefois avoir besoin d'intégrer des kernels personnalisés pour
accélérer des opérations propres à votre modèle, fusionner des opérations afin
de réduire le surcoût en bande passante mémoire, ou implémenter des
architectures nouvelles que les opérateurs PyTorch standards ne couvrent pas
encore.

## Choisir une approche d'intégration { #choosing-an-integration-approach }

TorchTPU propose trois façons d'utiliser des kernels personnalisés sur TPU,
selon votre cas d'utilisation :

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Approche | État | Idéal pour |
| :---- | :---- | :---- |
| **Tokamax** | Prêt pour la production | Accélération prête à l'emploi des briques courantes des LLM/MoE, sans écrire de code de kernel. |
| **Pallas (`torch.tpu.pallas.jax_op`)** | Prêt pour la production | Écrire des kernels TPU sur mesure, aux performances maximales, en JAX/Pallas, et les appeler depuis PyTorch. |
| **Helion** | Expérimental | Écrire des kernels portables entre accélérateurs (par exemple GPU, TPU) avec un DSL natif PyTorch. |
<!-- mdformat on -->

## 1. Utiliser des kernels prédéfinis de Tokamax { #1-using-pre-built-kernels-from-tokamax }

Avant d'écrire un kernel personnalisé à partir de zéro, vérifiez si l'opérateur
est disponible dans [Tokamax](http://openxla.org/tokamax), une bibliothèque de
kernels TPU hautes performances avec une interopérabilité native avec PyTorch.

### Quand l'utiliser { #when-to-use }

Utilisez Tokamax lorsque votre modèle a besoin de briques standards hautes
performances, notamment :

*   **Mixture-of-Experts (MoE) :** multiplication matricielle groupée, gathers
    et scatters de tokens.
*   **Variantes de l'attention :** paged attention, Gated DeltaNet, Kimi Delta
    Attention et Multi-head Latent Attention (MLA).
*   **Sélection :** calcul `TopK` optimisé.

### Fonctionnement { #how-it-works }

Les kernels Tokamax sont écrits en JAX/Pallas et s'appuient sur un autotuning
intégré (comme la recherche automatique de la taille des tuiles), sur des
benchmarks XProf à faible surcoût et sur des tests de régression sur les
différentes générations de TPU. En interne, Tokamax utilise
`torch.tpu.pallas.jax_op` pour gérer de façon transparente l'interopérabilité
entre frameworks et expose directement des API PyTorch natives à votre modèle.

## 2. Écrire des kernels Pallas personnalisés { #2-writing-custom-pallas-kernels }

Si vous avez besoin d'une opération sur mesure que ni PyTorch ni Tokamax ne
couvrent, vous pouvez écrire un kernel TPU en
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) et l'enregistrer
comme opérateur PyTorch avec `torch.tpu.pallas.jax_op`.

### Intégrer un kernel Pallas { #how-to-integrate-a-pallas-kernel }

1.  **Définissez le kernel Pallas** à l'aide de références mémoire d'entrée et
    de sortie (`Ref`s).
2.  **Encapsulez le kernel dans une fonction JAX** avec `pl.pallas_call` pour
    indiquer les formes et les dtypes de sortie.
3.  **Enregistrez la fonction JAX auprès de TorchTPU** avec
    `torch.tpu.pallas.jax_op`.

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

## 3. Écrire des kernels portables avec Helion (expérimental) { #3-writing-portable-kernels-with-helion-experimental }

Si vous voulez que vos kernels soient portables entre accélérateurs (par
exemple TPU, GPU), vous pouvez utiliser [Helion](https://helionlang.com/), un
DSL natif PyTorch pour écrire des kernels.

> **Note:** La prise en charge de Helion dans TorchTPU est actuellement
> **expérimentale**. Pour les charges de travail de production, utilisez
> Tokamax ou `torch.tpu.pallas.jax_op`.

### Écrire un kernel Helion { #how-to-write-a-helion-kernel }

Décorez votre fonction avec `@helion.kernel`. Utilisez des opérations PyTorch
standards pour la configuration côté hôte, et `hl.tile` pour définir la boucle
d'exécution sur l'appareil, qui parcourt des tuiles partitionnées :

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

Pour en savoir plus sur Helion, consultez la
[documentation Helion](https://helionlang.com/), posez vos questions sur le
[forum de discussion des développeurs PyTorch](https://dev-discuss.pytorch.org/c/helion/31)
ou [signalez des problèmes sur GitHub](https://github.com/pytorch/helion/issues).
