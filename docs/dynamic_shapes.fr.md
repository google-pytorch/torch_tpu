# Formes dynamiques { #dynamic-shapes }

<!-- Link to colab and to generated notebook -->

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/dynamic_shapes.ipynb)

Lorsque vous compilez un modèle avec `torch.compile(model)`, TorchTPU abaisse le
calcul en StableHLO et le compile avec le compilateur XLA (voir
[Fonctionnement de TorchTPU](./how_torchtpu_works.md)). Le compilateur XLA est
traditionnellement conçu autour de **formes de tensors connues statiquement**.
Cela permet de disposer à l'avance les buffers de la mémoire à haut débit
(HBM), de fusionner les pipelines d'opérateurs et de découper les
multiplications matricielles en tuiles adaptées aux réseaux systoliques (MXU)
du TPU. Lorsque les dimensions des tensors d'entrée varient d'un appel à
l'autre dans une charge de travail `torch.compile`, compiler pour chaque forme
exacte invaliderait les gardes de forme de Dynamo et déclencherait une
recompilation XLA complète et coûteuse à chaque nouvelle forme.

Pour prendre en charge les **formes dynamiques** sans recompilations
constantes, TorchTPU utilise le **dynamisme borné** : les bornes supérieures
des dimensions dynamiques sont fixées à la compilation, tandis que XLA suit
dynamiquement les tailles réelles des dimensions à l'exécution.

## Comment TorchTPU gère les formes dynamiques { #how-dynamic-shapes-are-handled-by-torchtpu }

Lorsque vous compilez un modèle ou une fonction avec le dynamisme par défaut,
c'est-à-dire `torch.compile(model, dynamic=None)` (quand les formes d'entrée
varient d'un appel à l'autre), ou que vous marquez explicitement des dimensions
dynamiques avec `torch._dynamo.mark_dynamic`, Dynamo trace le calcul et produit
un graphe FX dans lequel les dimensions variables sont représentées par des
variables entières symboliques (`torch.SymInt`) au lieu de constantes fixes.
TorchTPU traite ces dimensions symboliques de la façon suivante :

### Compilation { #compilation }

*   TorchTPU intercepte le graphe FX tracé et détermine une plage globale
    `[lower_bound, upper_bound]` pour chaque dimension symbolique (à partir de
    `mark_dynamic(..., min=..., max=...)` ou déduite par Dynamo de la taille
    initiale `S` à l'exécution). Il divise ensuite cette plage en buckets
    contigus alignés sur des multiples de `bucket_size` (`128` par défaut, ce
    qui correspond à la dimension de tuile à 128 voies du TPU), jusqu'à
    `num_dynamic_buckets` (`1` par défaut, `8` au maximum) :
    *   Le premier bucket va de `lower_bound` jusqu'au premier multiple de
        `128` supérieur ou égal à `S` (`U_1`), le deuxième bucket va de
        `U_1 + 1` jusqu'au multiple de `128` suivant (`U_2`), et ainsi de suite
        (par exemple, `[lower_bound, 128]`, `[129, 256]`, ...).
    *   TorchTPU enregistre auprès de Dynamo une seule garde de plage qui
        couvre l'étendue complète de tous les buckets. Dynamo ne déclenche donc
        une recompilation que si la dimension dépasse la borne globale
        `upper_bound`.
*   Pour chaque bucket, TorchTPU remplace la dimension symbolique par la borne
    supérieure statique `U` du bucket, transmet la taille réelle de la
    dimension à l'exécution comme entrée, et abaisse le graphe en StableHLO avec
    des types de tensors bornés. Ces types suivent à la fois la borne supérieure
    statique `U` et la taille active à l'exécution dans toutes les opérations.
*   XLA compile le module obtenu pour chaque borne supérieure `U` de bucket
    configurée.

### Exécution { #execution }

*   À l'exécution, TorchTPU sélectionne l'exécutable compilé du bucket
    correspondant à la taille réelle de la dimension dynamique.

*   Lors de l'exécution du module compilé, XLA complète (padding) le tensor
    d'entrée dynamique jusqu'à la borne supérieure statique `U` du bucket, et
    réinitialise les valeurs de la zone de padding située au-delà de la taille
    active avec des valeurs d'identité algébrique. Les éléments de padding
    n'affectent ainsi pas le résultat.

*   Pour les tensors de sortie dynamiques, le padding reste à la fin du buffer.
    TorchTPU peut ainsi renvoyer le buffer à PyTorch avec une simple mise à jour
    des métadonnées de forme, au lieu de découper physiquement le tensor.

## Cas d'utilisation type : décodage autorégressif { #motivating-use-case-autoregressive-decoding }

Le **décodage autorégressif de LLM** avec un cache KV qui grandit
dynamiquement est l'une des principales charges de travail qui justifient le
dynamisme borné. Dans une boucle de décodage standard, chaque étape ajoute
les états de clé et de valeur du nouveau token à l'historique en cache, le long de
la dimension de séquence, via `torch.cat` :

*   **Prefill :** le prompt d'entrée a la forme `[B, S]` (taille de batch `B`,
    longueur de séquence du prompt `S`) et produit un cache KV de forme
    `[B, H_kv, S, D]` (`H_kv` têtes clé/valeur, dimension de tête `D`).
*   **Étape de décodage 1 :** le token d'entrée a la forme `[B, 1]` et est
    concaténé avec le cache pour produire la forme `[B, H_kv, S + 1, D]`.
*   **Étape de décodage `t` :** la longueur de séquence du cache KV passe à
    `S + t`.

Avec une compilation à formes statiques (`dynamic=False`), Dynamo pose une
garde sur la longueur de séquence exacte du cache KV. Chaque étape de décodage
modifie la forme du tensor, invalide la garde et déclenche une **recompilation
XLA complète pour chaque token généré**. Des étapes de décodage de quelques
millisecondes se transforment alors en compilations de plusieurs secondes, à
moins de réécrire le modèle pour préallouer des buffers statiques et effectuer
des mutations sur place.

Le **dynamisme borné** résout ce problème sans réécriture du modèle : à mesure
que le cache KV grandit au fil des étapes de décodage (`S + 1`, `S + 2`, ...),
toutes les formes comprises dans le bucket compilé (`<= 128`, `<= 256`, ...)
réutilisent le même exécutable XLA.

Pour un tutoriel pratique qui applique le dynamisme borné à une boucle de
décodage LLM complète (et le compare à la préallocation d'un cache KV
statique), consultez
[Inférence LLM de bout en bout](./case_studies_muse_glimmer_30b_inference.md).

## Exemple : boucle autorégressive avec un tensor qui grandit { #example-autoregressive-loop-with-a-growing-tensor }

Pour observer en pratique le comportement de la compilation statique
(`dynamic=False`) et du dynamisme borné avec le dynamisme par défaut
(`dynamic=None`), prenez une boucle autorégressive minimale dans laquelle
chaque étape ajoute un nouveau token le long de la dimension de séquence
(`dim=2`) avec `torch.cat` :

```python
import time
import torch
from torch._functorch._aot_autograd import autograd_cache

torch._logging.set_logs()  # pylint: disable=protected-access
device = torch.device("tpu")

def decode_step(
    kv_cache: torch.Tensor, new_token: torch.Tensor
) -> torch.Tensor:
  return torch.cat([kv_cache, new_token], dim=2)

def run_decode_loop(
    compiled_step, num_steps: int = 10, prefix: str = ""
) -> None:
  kv_cache = torch.ones((1, 8, 16, 128), dtype=torch.bfloat16, device=device)
  new_token = torch.ones((1, 8, 1, 128), dtype=torch.bfloat16, device=device)
  torch.accelerator.synchronize()

  for step in range(num_steps):
    start = time.perf_counter()
    kv_cache = compiled_step(kv_cache, new_token)
    torch.accelerator.synchronize()
    step_ms = (time.perf_counter() - start) * 1000
    print(
        f"{prefix}Step {step + 1}/{num_steps} (seq_len={kv_cache.shape[2]}):"
        f" {step_ms:.2f} ms",
        flush=True,
    )
```

### 1. Exécution avec `dynamic=False` { #1-running-with-dynamicfalse }

```python
torch._dynamo.reset()
autograd_cache.AOTAutogradCache.clear()
torch._dynamo.config.recompile_limit = 64

step_static = torch.compile(decode_step, backend="tpu", dynamic=False)
run_decode_loop(step_static, prefix="[dynamic=False] ")
```

Exemple de sortie :

```text
[dynamic=False] Step 1/10 (seq_len=17): 49.54 ms
[dynamic=False] Step 2/10 (seq_len=18): 43.51 ms
[dynamic=False] Step 3/10 (seq_len=19): 39.52 ms
[dynamic=False] Step 4/10 (seq_len=20): 38.63 ms
[dynamic=False] Step 5/10 (seq_len=21): 38.40 ms
[dynamic=False] Step 6/10 (seq_len=22): 38.23 ms
[dynamic=False] Step 7/10 (seq_len=23): 38.11 ms
[dynamic=False] Step 8/10 (seq_len=24): 38.55 ms
[dynamic=False] Step 9/10 (seq_len=25): 37.89 ms
[dynamic=False] Step 10/10 (seq_len=26): 37.75 ms
```

**Ce qui s'est passé :** avec `dynamic=False`, Dynamo pose une garde sur la
longueur de séquence exacte de `kv_cache` (`16`, `17`, `18`, ...). Comme
`torch.cat` augmente `dim=2` de `1` à chaque itération, la garde de forme
échoue à chaque appel et déclenche une recompilation XLA complète à chaque
étape.

### 2. Exécution avec `dynamic=None` (dynamisme borné) { #2-running-with-dynamicnone-bounded-dynamism }

```python
torch._dynamo.reset()
autograd_cache.AOTAutogradCache.clear()

step_dynamic = torch.compile(decode_step, backend="tpu", dynamic=None)
run_decode_loop(step_dynamic, prefix="[dynamic=None] ")
```

Exemple de sortie :

```text
[dynamic=None] Step 1/10 (seq_len=17): 49.59 ms
[dynamic=None] Step 2/10 (seq_len=18): 206.68 ms
[dynamic=None] Step 3/10 (seq_len=19): 64.29 ms
[dynamic=None] Step 4/10 (seq_len=20): 0.51 ms
[dynamic=None] Step 5/10 (seq_len=21): 0.43 ms
[dynamic=None] Step 6/10 (seq_len=22): 0.38 ms
[dynamic=None] Step 7/10 (seq_len=23): 0.40 ms
[dynamic=None] Step 8/10 (seq_len=24): 0.44 ms
[dynamic=None] Step 9/10 (seq_len=25): 0.45 ms
[dynamic=None] Step 10/10 (seq_len=26): 0.37 ms
```

**Ce qui s'est passé :** avec `dynamic=None`, seules les 3 premières
itérations subissent un surcoût de compilation. À partir de l'étape 4,
l'exécution tombe à environ 0,4 ms par étape (environ 100 fois plus rapide),
sans aucune recompilation :

*   **Étape 1 (`seq_len=16 -> 17`) :** Dynamo voit les entrées pour la première
    fois et compile un exécutable statique pour `seq_len=16`.
*   **Étape 2 (`seq_len=17 -> 18`) :** comme `dim=2` est passée de `16` à `17`,
    Dynamo marque automatiquement `dim=2` comme dynamique (`SymInt`) et
    TorchTPU compile un exécutable à dynamisme borné pour le bucket `[17, 128]`
    en utilisant la disposition initiale du tensor d'entrée.
*   **Étape 3 (`seq_len=18 -> 19`) :** le buffer de sortie à dynamisme borné
    produit par l'étape 2 utilise la disposition de sortie avec padding de XLA.
    TorchTPU compile donc une nouvelle fois le bucket `[17,
    128]` pour cette disposition XLA stable.
*   **Étapes 4 à 10 (`seq_len=19 -> 26`) :** la longueur de séquence (dans le
    bucket `<= 128`) et la disposition du buffer XLA correspondent toutes deux à
    l'exécutable en cache. Chaque étape suivante lance donc directement
    l'exécutable XLA précompilé, sans recompilation.

> **Tip:** Si vous savez à l'avance quelle dimension va varier, vous pouvez
> appeler `torch._dynamo.mark_dynamic(kv_cache, 2, min=16, max=128)` avant la
> première étape. Dynamo trace alors `dim=2` comme dynamique dès l'étape 1, ce
> qui évite la compilation statique initiale de l'étape 1 et vous permet
> d'indiquer des bornes `[min, max]` explicites.

## Limites actuelles { #current-limitations }

Le dynamisme borné de TorchTPU évolue activement et il est actuellement
optimisé pour le cas d'utilisation du **cache KV autorégressif**. Tenez compte
des limites suivantes lorsque vous l'appliquez à vos modèles :

*   **Nombre maximal de buckets et dynamisme non borné :** au plus **8 buckets**
    (`num_dynamic_buckets <= 8`) sont compilés par trace. Le dynamisme non borné
    n'est pas encore pris en charge : si une dimension dynamique dépasse la
    borne supérieure du plus grand bucket compilé, une recompilation est
    déclenchée.
*   **Contraintes de forme de Dynamo :** les bornes supérieures des buckets
    doivent respecter les contraintes de forme actives de Dynamo (comme les
    exigences de divisibilité de `reshape` ou `view`). Sinon, TorchTPU se
    rabat sur un seul bucket de borne supérieure `2 * S` (deux fois la taille
    initiale `S` à l'exécution).
*   **Surcoût de calcul et de HBM jusqu'à la borne supérieure du bucket :**
    comme l'exécution porte sur des buffers complétés jusqu'à la borne
    supérieure statique `U` du bucket, l'empreinte en HBM et le nombre de FLOP
    dépendent de `U` et non de la taille réelle à l'exécution. Si vous
    définissez une valeur `bucket_size` très grande ou
    `mark_dynamic(..., max=U)` alors que la taille réelle est bien inférieure à
    `U`, vous payez le coût en calcul et en mémoire de `U`.
*   **Changements de disposition XLA :** même lorsque la forme d'un tensor se
    situe dans un bucket déjà compilé, un changement de la disposition physique
    XLA sous-jacente d'un buffer d'entrée déclenche une recompilation du
    bucket.
*   **Pas de cache de compilation persistant (`AOTAutogradCache`) :** les
    exécutables à dynamisme borné ne sont pas encore sérialisés dans
    `AOTAutogradCache`. Ils sont donc recompilés à chaque démarrage d'un
    nouveau processus.
*   **Couverture des opérateurs :** tous les opérateurs PyTorch ne prennent pas
    encore en charge les formes dynamiques bornées.
*   **Vues comme entrées du modèle :** la prise en charge des vues de tensors
    transmises directement comme entrées du modèle compilé est limitée.
*   **Dynamisme dépendant des données :** les opérations dont les formes de
    sortie dépendent des valeurs des tensors à l'exécution (comme
    `torch.nonzero`) ne sont pas prises en charge.
*   **Kernels personnalisés :** les formes dynamiques bornées ne sont pas prises
    en charge avec les kernels TPU personnalisés (comme les kernels Pallas
    enregistrés via `torch.tpu.pallas.jax_op`).
*   **dynamic=True :** le mode torch.compile() *dynamic=True* n'est pas pris en
    charge.
