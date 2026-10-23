# Gérer la précision numérique dans TorchTPU { #manage-numerical-precision-in-torchtpu }

Les TPU Google répartissent les opérations à virgule flottante entre deux
unités matérielles pour équilibrer débit et précision numérique :

*   **Unité de multiplication matricielle (MXU) :** exécute les opérations de
    contraction (comme `torch.matmul`, `torch.mm`, `torch.bmm` et les
    convolutions) avec des entrées `bfloat16` natives (significande de 8 bits)
    et une accumulation `float32` sur 32 bits.
*   **Unité d'exécution vectorielle (VPU) :** exécute les opérations sans
    contraction (comme l'arithmétique élément par élément, les activations, la
    normalisation de couche et le softmax) en `float32` IEEE-754 complet
    (significande de 24 bits).

Ce guide explique comment configurer les modes de précision de la
multiplication matricielle `float32`, vérifier la parité numérique entre CPU
et TPU, contrôler les types d'accumulateur de sortie et l'excès de précision,
et valider dans le code le comportement bit à bit de la MXU.

### Comparer la précision de PyTorch sur CUDA et sur TPU { #compare-pytorch-precision-on-cuda-and-tpu }

TorchTPU prend en charge les API de précision standards de PyTorch
(`torch.set_float32_matmul_precision` et `torch.get_float32_matmul_precision`),
ainsi que des contrôles propres aux TPU (`torch.tpu.Precision` et
`torch.tpu.precision`) :

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Dimension          | PyTorch sur CUDA                     | PyTorch sur TPU (TorchTPU)             |
| :----------------- | :----------------------------------- | :------------------------------------- |
| **Mode de précision par défaut** | `"highest"` (`float32` pur) | `"medium"` (`Precision.DEFAULT`)       |
| **Primitive matérielle de calcul rapide** | Tensor Cores (`TF32`) | MXU systoliques (`bfloat16`)             |
| **Bits en calcul rapide** | 1 bit de signe + 8 bits d'exposant + 10 bits de mantisse (`TF32` sur 19 bits) | 1 bit de signe + 8 bits d'exposant + 7 bits de mantisse (`bfloat16` sur 16 bits) |
| **Précision d'accumulation** | `float32` (`"highest"`, `"high"`), `TF32` (`"medium"`) | `float32` dans tous les modes |
| **Mécanisme de précision supérieure** | Unités matérielles `float32`/`TF32` dédiées | Décomposition en plusieurs passes (3 ou 6 passes MXU) |
| **Impact de `high`/`highest` sur les performances** | Vitesse de référence ou ralentissement modéré (~1–2x) | Coût de calcul MXU ~3x pour `"high"` ; ~6x pour `"highest"` |
| **API de portée** | `torch.set_float32_matmul_precision` global | `torch.set_float32_matmul_precision` <br> global + `torch.tpu.precision` à portée limitée |
<!-- mdformat on -->

## Définir les modes de précision { #specify-precision-modes }

Lorsque vous multipliez des tensors `float32` sur la MXU, TorchTPU fait
correspondre les niveaux de précision de PyTorch à l'enum
`torch.tpu.Precision` :

*   `"medium"` / `Precision.DEFAULT` (1 passe) : **Par défaut.** Arrondit les
    entrées `float32` en `bfloat16` (multiplication des 8 bits de poids fort de
    la significande) et accumule en `float32`. Recommandé pour presque toutes
    les charges de travail d'entraînement et d'inférence.
*   `"high"` / `Precision.HIGH` (3 passes) : divise les entrées `float32` en
    composantes `bfloat16` haute et basse et calcule trois termes croisés
    (multiplication des 16 premiers bits de la significande), pour un coût de
    calcul MXU environ 3 fois plus élevé.
*   `"highest"` / `Precision.HIGHEST` (6 passes) : divise les entrées `float32`
    en trois composantes `bfloat16` et calcule six termes croisés
    (multiplication des 24 bits de la significande) pour approcher la précision
    `float32` IEEE-754 complète, pour un coût de calcul MXU environ 6 fois plus
    élevé.

Vous pouvez configurer la précision de la multiplication matricielle de deux
façons :

*   **Configuration globale :** appelez `torch.set_float32_matmul_precision("medium"
    | "high" | "highest")` pour définir le mode par défaut du processus.
*   **Gestionnaire de contexte à portée limitée :** utilisez
    `with torch.tpu.precision(mode):` pour n'appliquer une précision supérieure
    qu'aux couches ou sous-modules sensibles sur le plan numérique.
*   **Avertissement sur les performances :** sélectionner `"high"` ou
    `"highest"` émet un avertissement unique (`TORCH_WARN_ONCE`). Conservez le
    mode `"medium"` par défaut, sauf si vous observez des problèmes de
    convergence liés à la précision.

> **Note:** Entrer dans `torch.tpu.precision` ou en sortir provoque une rupture
> de graphe `torch.compile`. Lorsque vous utilisez
> `torch.compile(..., fullgraph=True)`, définissez la précision en dehors de la
> fonction compilée avec `torch.set_float32_matmul_precision(...)` ou
> encapsulez l'appel externe dans `with
> torch.tpu.precision(...):`.

Utilisez `torch.tpu.precision` pour comparer le temps d'exécution en régime
établi des trois modes :

```python
import time
import torch

device = torch.device("tpu")
precision = torch.tpu.precision  # Context manager
Precision = torch.tpu.Precision  # Enum class

print(f"Default matmul precision: {torch.get_float32_matmul_precision()}")

size = (8, 2048, 2048)
a = torch.randn(size, dtype=torch.float32, device=device)
b = torch.randn(size, dtype=torch.float32, device=device)

def run_matmul():
  res = torch.matmul(a, b)
  torch.accelerator.synchronize()
  return res

for mode in [Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST]:
  with precision(mode):
    # Warm up to compile the graph before timing steady-state execution
    run_matmul()
    start = time.perf_counter()
    run_matmul()
    print(f"{mode} time: {time.perf_counter() - start:.4f}s")
```

### Comparer la précision des différents modes { #compare-accuracy-across-precision-modes }

Comparez les sorties du TPU à une référence CPU `float32` calculée sur les
mêmes entrées pour voir dans quelle mesure chaque mode de précision réduit
l'erreur numérique :

```python
# CPU float32 reference result
c_ref = torch.matmul(a.cpu(), b.cpu())

for mode in [Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST]:
  with precision(mode):
    res = torch.matmul(a, b)
    diff = res.cpu() - c_ref
    print(f"Mode: {mode}")
    print(f"  Max Absolute Error: {diff.abs().max():.6e}")
    print(
        "  Mean Relative Error: "
        f" {(diff.abs() / (c_ref.abs() + 1e-8)).mean():.6e}\n"
    )
```

## Vérifier la parité numérique { #verify-numerical-parity }

Lorsque vous portez un modèle sur TPU ou que vous déboguez des différences de
convergence, vérifiez la parité numérique : exécutez la même couche ou le même
sous-module sur CPU et sur TPU avec des poids synchronisés et des entrées
identiques, puis comparez les sorties avec `torch.testing.assert_close` :

```python
import torch.nn as nn

# 1. Set up identical models in bfloat16 for a fair comparison
model_cpu = nn.Linear(10, 10).cpu().to(torch.bfloat16)
model_tpu = nn.Linear(10, 10).to(device).to(torch.bfloat16)

# 2. Synchronize weights (mandatory for parity checks)
model_tpu.load_state_dict(model_cpu.state_dict())

# 3. Create identical input data for both devices
input_data = torch.randn(5, 10, dtype=torch.bfloat16)
input_tpu = input_data.to(device)

# 4. Run the forward pass
out_cpu = model_cpu(input_data)
out_tpu = model_tpu(input_tpu)

# 5. Compare results
# Note: TorchTPU's bfloat16 accumulation and fusion differences can exceed
# default tolerances, generating the `assert_close` diagnostic report below.
try:
  torch.testing.assert_close(out_tpu.cpu(), out_cpu)
  print("Parity check passed!")
except AssertionError as e:
  print("Parity check failed (expected for hardware precision drift):\n")
  print(e)
```

### Interpréter les écarts détaillés { #interpret-detailed-mismatches }

Lorsque des tensors dépassent les tolérances, `torch.testing.assert_close`
affiche un résumé qui aide à distinguer une dérive normale des calculs à
virgule flottante d'un bug logique :

```text
Tensor-likes are not close!

Mismatched elements: 1 / 50 (2.0%)
Greatest absolute difference: 0.006596 at index (3, 9) (up to 1e-05 allowed)
Greatest relative difference: 0.165375 at index (3, 9) (up to 0.016 allowed)
```

*   **Mismatched elements :** un pourcentage élevé d'éléments présentant de
    faibles différences absolues indique généralement des différences normales
    d'arrondi ou d'ordre d'accumulation dues aux fusions du compilateur, plutôt
    qu'un bug.
*   **Greatest absolute / relative difference :** indique la plus grande erreur
    et son indice dans le tensor, pour que vous puissiez examiner les valeurs
    aberrantes ou choisir des seuils `atol` et `rtol` appropriés.

### Bonnes pratiques pour la parité { #best-practices-for-parity }

*   **Utilisez les mêmes types de données sur les deux appareils :**
    comparez `bfloat16` sur TPU à `bfloat16` sur CPU, ou `float32` sur TPU
    (avec `Precision.HIGHEST`) à `float32` sur CPU. Comparer `bfloat16` à
    `float32` mesure l'erreur de quantification et non la dérive matérielle.
*   **Vérifiez les gradients en plus des activations :** après avoir appelé
    `loss.backward()`, comparez `model_tpu.weight.grad.cpu()` à
    `model_cpu.weight.grad` selon le même principe.
*   **Partez des tolérances standards :** utilisez comme base les tolérances par
    défaut indiquées dans la
    [documentation de PyTorch sur les tests](https://docs.pytorch.org/docs/stable/testing.html),
    et n'assouplissez `atol` ou `rtol` que là où c'est attendu en raison d'une
    accumulation en précision réduite.

## Définir la précision de sortie { #specify-output-precision }

Lorsque vous multipliez des tensors `bfloat16` avec `torch.mm` ou `torch.bmm`,
la MXU accumule les produits en `float32` avant d'arrondir la sortie finale en
`bfloat16`. Passez `out_dtype=torch.float32` pour renvoyer directement
l'accumulateur `float32`, sans conversion vers un type de moindre précision :

```python
a_bmm = torch.randn((2, 256, 256), dtype=torch.bfloat16, device=device)
# Using out_dtype to force float32 output accumulator precision
res_bmm = torch.bmm(a_bmm, a_bmm, out_dtype=torch.float32)
print(f"Input dtype: {a_bmm.dtype}")
print(f"Output dtype: {res_bmm.dtype}")
```

## Contrôler l'excès de précision { #control-excess-precision }

Comme la MXU du TPU accumule en `float32` les multiplications matricielles
`bfloat16`, le compilateur XLA peut optimiser les opérations fusionnées en
conservant les valeurs intermédiaires en `float32` au lieu de les arrondir en
`bfloat16` entre les étapes. Le fait de conserver ces bits intermédiaires
supplémentaires s'appelle l'*excès de précision*.

Vous pouvez configurer ce comportement de façon globale avec
`torch.tpu.allow_excess_precision` (qui définit l'option de compilation
sous-jacente `--xla_allow_excess_precision` et vaut `False` par défaut) :

*   **Désactiver l'excès de précision (`False`, par défaut) :** définissez
    `torch.tpu.allow_excess_precision = False` pour forcer le compilateur à
    conserver les conversions intermédiaires en `bfloat16`, afin que les
    opérations fusionnées reproduisent le comportement d'arrondi `bfloat16` du
    mode eager.
*   **Autoriser l'excès de précision (`True`) :** définissez `torch.tpu.allow_excess_precision =
    True` pour permettre au compilateur d'éliminer les conversions
    intermédiaires et de conserver en `float32` les résultats intermédiaires
    fusionnés.

> **Important:** Définissez `torch.tpu.allow_excess_precision` tout au début de
> votre script, avant d'exécuter ou de compiler la moindre opération TPU. Le
> runtime XLA verrouille cette option au démarrage de la première compilation ;
> toute tentative ultérieure de la modifier avec une autre valeur génère une
> `RuntimeError`.

## Comprendre la précision de la MXU du TPU { #understand-tpu-mxu-precision }

Comprendre comment la MXU du TPU traite les nombres à virgule flottante permet
d'expliquer le fonctionnement de l'excès de précision et des modes `float32`
à plusieurs passes :

*   **Plage ou précision (`bfloat16`) :**
    [les travaux de recherche de Google Cloud sur `bfloat16`](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)
    ont montré que les réseaux de neurones peuvent s'entraîner avec une
    précision de mantisse réduite sans perte de précision du modèle, à
    condition de conserver la large plage dynamique de `float32` (essentielle
    pour des opérations comme les normes et les gradients). Alors que `float32`
    dispose de 23 bits de mantisse (24 bits de précision de significande en
    comptant le bit de tête implicite), `bfloat16` conserve la même plage
    d'exposant sur 8 bits avec seulement 7 bits de mantisse (8 bits de
    précision de significande, soit un epsilon machine de `2**-7`, environ
    0,78 %).
*   **Formats d'échange et formats arithmétiques :** les systèmes à virgule
    flottante distinguent les **formats d'échange** (la façon dont les nombres
    sont stockés en mémoire) des **formats arithmétiques** (la façon dont le
    matériel calcule et arrondit les opérations intermédiaires, comme les
    3 bits supplémentaires de garde, d'arrondi et collant d'une ALU standard).
    `bfloat16` peut désigner l'un ou l'autre.
*   **Accumulation le long de la dimension de contraction :** multiplier deux
    matrices de forme `[I, J] @ [J, K]` revient à calculer une grille de
    produits scalaires de forme `[1,
    J] @ [J, 1]`, chacun additionnant `J` produits élément par élément le long
    de la **dimension de contraction** `J`. Comme `bfloat16` a un epsilon
    machine de `2**-7` (~`0.008`), accumuler `J` produits en `bfloat16`
    provoquerait d'importantes erreurs d'arrondi. La MXU accepte donc des
    entrées `bfloat16` (et `fp8` sur les générations plus récentes comme le
    TPU v7x) et accumule les sommes intermédiaires en `float32`. Cet
    accumulateur joue le même rôle que les bits de garde, d'arrondi et collant
    d'une ALU, avant l'arrondi au format de sortie.
*   **Différence entre formats d'entrée et de sortie :** l'association
    d'entrées `bfloat16` et d'un accumulateur `float32` est à l'origine à la
    fois de l'*excès de précision* (conservation de valeurs intermédiaires
    `float32` entre opérations fusionnées) et d'une *précision inférieure à
    celle attendue* (une matmul `float32` en 1 passe tronque les entrées à des
    significandes `bfloat16` de 8 bits). À l'inverse, l'unité d'exécution
    vectorielle (VPU) du TPU opère sur les 23 bits de mantisse de `float32`
    (sauf pour quelques cas limites de l'IEEE-754, comme les nombres
    sous-normaux).

### Développement polynomial en plusieurs passes pour `float32` { #multi-pass-polynomial-expansion-for-float32 }

Comme une MXU n'accepte que des entrées `bfloat16` (significande de 8 bits),
elle ne peut pas calculer une multiplication matricielle `float32` complète sur
24 bits en une seule passe. En pratique, la plupart des modèles modernes
stockent leurs poids en `bfloat16` et exécutent rarement des matmul `float32`.
Lorsqu'une précision `float32` complète est nécessaire pour une matmul, XLA
utilise l'algorithme décrit dans
[Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations](https://arxiv.org/pdf/1904.06376)
(Henry, Tang et Heinecke). Cet algorithme représente chaque significande
`float32` de 24 bits comme la somme de trois valeurs `bfloat16` de 8 bits
(`High + Med + Low`) et développe leur produit :

```text
(a + b + c)(d + e + f) = ad + ae + af + bd + be + bf + cd + ce + cf
```

Multiplier deux sommes de 3 termes donne neuf produits `bfloat16`, dont les
trois termes d'ordre le plus bas provoquent toujours un dépassement inférieur
(underflow) en `float32` :

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
|          | High        | Med                      | Low                      |
| :------- | :---------- | :----------------------- | :----------------------- |
| **High** | High x High *(1 passe)* | High x Med *(3 passes)*    | High x Low *(6 passes)*    |
| **Med**  | Med x High *(3 passes)* | Med x Med *(6 passes)*     | Med x Low *(underflow)* |
| **Low**  | Low x High *(6 passes)* | Low x Med *(underflow)* | Low x Low *(underflow)* |
<!-- mdformat on -->

### Évaluer les performances en 3 et 6 passes { #evaluate-3-pass-and-6-pass-performance }

Même si `Precision.HIGH` (3 passes) et `Precision.HIGHEST` (6 passes)
nécessitent 3 et 6 fois plus de passes MXU, l'impact de bout en bout dépend de
votre charge de travail :

*   **Dimensions de contraction importantes :** dans les modèles modernes
    entraînés avec une accumulation `float32` (comme DeepSeek-V3) et dotés de
    grandes dimensions de contraction, le bruit d'accumulation et le
    non-déterminisme sur des milliers d'éléments l'emportent sur la
    contribution des termes croisés de mantisse d'ordre inférieur. Le mode en
    1 passe (`Precision.DEFAULT`) suffit donc.
*   **Modèles limités par la bande passante mémoire :** les anciens modèles
    `float32` n'effectuent souvent pas assez de calculs par octet transféré
    depuis la mémoire à haut débit (HBM) pour saturer la MXU (ou les Tensor
    Cores des GPU). Dans ces charges de travail limitées par la mémoire, les
    passes MXU supplémentaires des modes en 3 et 6 passes utilisent des cycles
    MXU qui seraient sinon inactifs. Le ralentissement de bout en bout est
    alors inférieur à 3 ou 6 fois.

## Valider le comportement de la précision bit à bit { #validate-bitwise-precision-behavior }

Vous pouvez vérifier comment la MXU traite les bits de mantisse et l'excès de
précision en construisant des entrées à l'aide de deux règles des nombres à
virgule flottante binaires :

*   `1 + 2**-n` crée un nombre dont la significande binaire est
    `1.000...1000...`, avec un `1` fractionnaire au `n`-ième bit de mantisse.
*   `(1 + m) * (1 + n) = 1 + m + n + mn` : multiplier deux nombres dont les bits
    sont à 1 aux positions de mantisse 6 et 7 donne un résultat dont les bits
    sont à 1 aux positions 6, 7 et `6 + 7 = 13`. Comme `bfloat16` ne contient
    que 7 bits de mantisse, la position 13 dépasse la capacité de `bfloat16` et
    n'est conservée que si la valeur reste dans l'accumulateur `float32` :

```py
value = (1 + 2**-6) * (1 + 2**-7)  # 1.0000_01 * 1.0000_001 = 1.0000_0110_0000_1
```

Placer ces valeurs dans des matrices `256 x 256` déclenche l'exécution sur la
MXU et vous permet de vérifier si le 13e bit de mantisse est perdu à l'arrondi
ou conservé :

```python
# Python uses float64 by default.
bit_6 = 1 + 2**-6
bit_7 = 1 + 2**-7
bits_6_7 = 1.0 + 2**-6 + 2**-7
bits_6_7_13 = 1.0 + 2**-6 + 2**-7 + 2**-13

print("=== Python CPU values ===")
print(f"{bit_6=:.20f}")
print(f"{bit_7=:.20f}")
print(f"{bits_6_7=:.20f}")
print(f"{bits_6_7_13=:.20f}")
print(f"{(bit_6 * bit_7)=:.20f}")
assert bit_6 * bit_7 == bits_6_7_13

print("=== PyTorch TPU values ===")

eye = torch.eye(256, dtype=torch.bfloat16, device=device)
bit_6_tensor = eye.clone() * bit_6
bit_7_tensor = eye.clone() * bit_7
bits_6_7_tensor = eye.float().clone() * bits_6_7
bits_6_7_13_tensor = eye.float().clone() * bits_6_7_13

# Use a 1-element vector rather than a 0D scalar because PyTorch's type
# promotion rules ignore 0D scalar tensor dtypes when paired with 2D tensors.
one_float32 = torch.tensor([1.0], dtype=torch.float32, device=device)

print(f"{bit_6_tensor[0,0]=:.20f}")
print(f"{bit_7_tensor[0,0]=:.20f}")
print(f"{bits_6_7_tensor[0,0]=:.20f}")
print(f"{bits_6_7_13_tensor[0,0]=:.20f}")

def mul_add(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
  return a @ b * c

# In eager execution, `a @ b` rounds to bfloat16 before `* c`, dropping bit 13.
print("=== TPU values: Eager ===")
eager_res = mul_add(bit_6_tensor, bit_7_tensor, one_float32)[0, 0]
print(f"{eager_res=:.20f}")
assert eager_res == bits_6_7_tensor[0, 0]

# With allow_excess_precision = False, the compiled graph also rounds `a @ b`
# to bfloat16 before multiplying by `c`, dropping bit 13.
print("=== TPU values: compiled with allow_excess_precision = False ===")
torch.compiler.reset()
torch.tpu.allow_excess_precision = False
mul_add_compiled_false = torch.compile(mul_add, backend="tpu")
print(f"{torch.tpu.allow_excess_precision=}")
res_false = mul_add_compiled_false(bit_6_tensor, bit_7_tensor, one_float32)[
    0, 0
]
print(f"{res_false=:.20f}")

try:
  assert res_false == bits_6_7_tensor[0, 0]
  print("  [PASSED] Successfully disabled excess precision.")
except AssertionError:
  print(
      "  [WARNING] Failed to disable excess precision (XLA compiled another"
      " graph with default 'True' earlier in this session)."
  )

# Changing allow_excess_precision after XLA compilation has already started in
# this process raises a RuntimeError because compiler options are locked on
# the first compilation.
#
# When `torch.tpu.allow_excess_precision = True` is set at the start of a
# fresh Python session before any TPU operations or compilations run, XLA
# keeps `a @ b` in float32 when fusing with `* c`, preserving the 13th
# mantissa bit (res_true == bits_6_7_13_tensor[0, 0] == 1.0235595703125).
print("=== TPU values: compiled with allow_excess_precision = True ===")
torch.compiler.reset()
try:
  torch.tpu.allow_excess_precision = True
  mul_add_compiled_true = torch.compile(mul_add, backend="tpu")
  print(f"{torch.tpu.allow_excess_precision=}")
  res_true = mul_add_compiled_true(bit_6_tensor, bit_7_tensor, one_float32)[
      0, 0
  ]
  print(f"{res_true=:.20f}")
  assert res_true == bits_6_7_13_tensor[0, 0]
  print("  [PASSED] Successfully enabled excess precision.")
except (RuntimeError, AssertionError) as e:
  print(
      "  [EXPECTED] Cannot change excess precision after first compilation:"
      f" {e}"
  )
```
