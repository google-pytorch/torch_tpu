# Fonctionnement de TorchTPU { #how-torchtpu-works }

## Les TPU en bref { #tpu-primer }

Les [TPU Google](https://cloud.google.com/tpu) sont des ASIC spécialisés,
conçus spécifiquement pour effectuer les opérations massives sur les tensors
qui sont à la base du deep learning. L'architecture des TPU
[varie](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm) d'une
génération à l'autre mais, de façon générale, une puce TPU contient :

*   Un ou plusieurs TensorCores. Chacun est un processeur monothread composé
    d'unités de multiplication matricielle (MXU), d'une unité vectorielle (VPU)
    et d'une unité scalaire.
*   Un ou plusieurs SparseCores, spécialisés dans les opérations creuses, vers
    lesquels le TensorCore peut décharger des opérations.
*   Une grande mémoire à haut débit (HBM) : 288 Go à 8,6 To/s sur le
    [TPU 8i](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive).
    Elle peut servir à stocker les données de ML et à les partager avec le CPU.
*   Une interconnexion bidirectionnelle entre puces (ICI) : 19,2 To/s sur les
    [TPU 8t et 8i](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/).
    Elle permet une communication rapide entre les puces TPU au sein de
    [tranches TPU](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices)
    configurées par l'utilisateur. Ces tranches sont des sous-groupes logiques
    d'un [pod TPU](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu-pod),
    qui comprend un nombre prédéfini de puces TPU connectées selon une
    configuration en tore 3D (pour les TPU 7 et TPU 8t) ou Boardfly (TPU 8i).
*   Un réseau de centre de données bidirectionnel (DCN) : 400 Go/s sur le
    [TPU 8t](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/).
    Il permet la communication entre les puces TPU de différentes tranches.

Les TPU se programment avec la pile du
[compilateur XLA](https://github.com/openxla/xla) via deux points d'entrée
différents : [StableHLO](https://openxla.org/stablehlo/spec) et
[Mosaic](https://docs.jax.dev/en/latest/pallas/design/design.html#lowering-pallas-to-mosaic-for-tpu).
Le jeu d'instructions (ISA) réel des TPU est propriétaire, susceptible de
changer et non public. En pratique, vous pouvez considérer StableHLO et (plus
encore) Mosaic comme l'équivalent de PTX. L'image ci-dessous en donne une vue
d'ensemble :

![Fonctionnement de TorchTPU](./images/torch_tpu_inputs_to_tpu.svg)

Un programme C++ peut accéder à un TPU local en instanciant un
[client PjRT](https://openxla.org/xla/pjrt/cpp_api_overview). Son API fournit
des méthodes pour compiler du MLIR (StableHLO ou Mosaic) en exécutable TPU,
lancer cet exécutable et copier des données vers le TPU ou depuis le TPU.
Notez que l'exécution sur TPU est asynchrone : le CPU peut attendre qu'un
résultat TPU soit prêt en attendant le PjRTBuffer associé. StableHLO suit le
[paradigme SSA](https://en.wikipedia.org/wiki/Static_single-assignment_form) et
n'autorise donc pas les mutations sur place : une opération d'écriture produit
toujours une nouvelle valeur (au moins conceptuellement). Ce point est
particulièrement important pour TorchTPU, qui fournit la sémantique familière
des « vues » de PyTorch, y compris les mutations, grâce à la décomposition et
à l'inversion des vues. Le compilateur XLA suppose que les formes des tensors
sont connues au moment de la compilation. TorchTPU assouplit cette contrainte
grâce au [dynamisme borné](./dynamic_shapes.md). Comme il s'agit d'une
limitation pour les charges de travail PyTorch, nous avons ajouté la prise en
charge du dynamisme borné : les bornes maximales des dimensions d'une forme
sont connues à la compilation, mais les dimensions réelles ne sont connues
qu'à l'exécution.

## TorchTPU : un backend hors arbre avec prise en charge du chargement automatique { #torchtpu-an-out-of-tree-backend-with-autoload-support }

Conformément aux
[bonnes pratiques actuelles](https://docs.pytorch.org/tutorials/advanced/extend_dispatcher.html),
selon lesquelles les backends supplémentaires sont construits hors de l'arbre
source et enregistrés à l'exécution, nous étendons le dispatcher PyTorch avec un
nouveau backend C++, enregistré sous le nom `PrivateUse1`. Nous avons également
utilisé le mécanisme standard `at::Tensor` pour définir une implémentation du
stockage des tensors propre à l'appareil (en utilisant le
`c10::UniqueVoidPtr` à l'intérieur de `c10::StorageImpl`).

Nous avons conçu TorchTPU en pensant à la facilité d'utilisation. Lorsque
l'utilisateur importe PyTorch sur une machine TPU où le wheel TorchTPU est
installé, Python charge automatiquement le wheel TorchTPU. Ce chargement
automatique configure le backend et remplace le nom `PrivateUse1` par `tpu`.
Dans un exemple simple, un programme TorchTPU ne diffère donc en rien d'un
programme utilisant les backends intégrés à l'arbre source, et un programme
PyTorch peut s'exécuter sur le TPU avec peu ou pas de modifications :

```python
import torch
import torch.distributed as dist

def main():
    device = torch.device("tpu")
    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    word_size = dist.get_world_size()
    ...
    model = model.to(device)
    ...
```

Comme le TPU devient l'accélérateur courant et que `tpu_dist` est le backend
par défaut de TorchTPU, il n'est pas nécessaire de les indiquer
explicitement :

```python
import torch
import torch.distributed as dist

def main():
    device = torch.accelerator.current_accelerator()
    dist.init_process_group()
    rank = dist.get_rank()
    word_size = dist.get_world_size()
    ...
    model = model.to(device)
    ...
```

## Capturer le graphe { #capturing-the-graph }

TorchTPU trace les opérations eager au niveau ATen, en s'appuyant sur les hooks
de dispatch des opérations PyTorch qui font partie de l'infrastructure des
backends. Les opérations TorchTPU sont construites sous forme de programmes
StableHLO et transmises à XLA une par une ou par groupes, ce qui permet au
compilateur de fusionner largement les opérations pour mieux utiliser le
matériel. Lorsque TorchTPU reçoit un callback du dispatcher PyTorch, le runtime :

*   Crée une instance de Deferred Op qui contient :
    *   Un objet builder StableHLO qui, lorsqu'il est appelé, renvoie une
        fonction StableHLO implémentant la sémantique de `aten::mul`, comme
        ci-dessous (notez que le broadcasting implicite et la promotion des
        dtypes sont gérés ailleurs)

```
func.func @main(%arg0: tensor<4x16xbfloat16>, %arg1: tensor<4x16xbfloat16>) -> tensor<4x16xbfloat16> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<4x16xbfloat16>
  return %0 : tensor<4x16xbfloat16>
}
```

*   Une table facultative de paramètres statiques de l'opération (par exemple,
    training et momentum pour `aten::native_batch_norm`), utilisée pour
    calculer une clé de cache ;
*   Renvoie un nouveau `at::Tensor` dont le `c10::StorageImpl` pointe (par
    indirection) vers cette instance de Deferred Op.

Lorsqu'il atteint un point de synchronisation approprié, qui dépend du mode
d'exécution, TorchTPU :

*   Analyse la région d'opérations ;
*   Détermine quelles opérations matérialiser en code exécutable ;
*   Crée une clé de cache de compilation pour le graphe ;
*   Si la clé existe déjà dans le cache, récupère l'exécutable et l'envoie au
    TPU ;
*   Si elle n'existe pas dans le cache, compile l'objet de façon asynchrone,
    l'ajoute au cache et l'envoie pour exécution ;
*   Attend éventuellement que la valeur soit prête et la recopie, si le point de
    synchronisation atteint l'exige.

Nous détaillons les différents modes d'exécution dans
[Modèle d'exécution de TorchTPU](./torchtpu_execution_model.md).

## Le cache de compilation de TorchTPU { #the-torchtpu-compilation-cache }

Pour éviter les compilations inutiles, TorchTPU utilise un cache de
compilation à plusieurs niveaux :

*   **Niveau 1 :** cache en mémoire local au processus, qui permet de partager
    les compilations au sein d'un même processus. C'est le plus rapide des trois
    niveaux. Il est toujours activé.
*   **Niveau 2 :** mémoire partagée locale à la machine, qui permet le partage
    entre processus sur un même hôte. Ce niveau est utile pour
    l'entraînement et l'inférence sur plusieurs appareils. Le niveau 2 est
    activé par défaut pour les programmes distribués. Pour les programmes sur un
    seul appareil, il n'apporte aucun avantage et il est donc désactivé.
*   **Niveau 3 :** cache persistant reposant sur un système de fichiers distant
    (par exemple, Google Cloud Storage), qui permet la réutilisation entre
    hôtes, programmes et redémarrages de jobs. Ce niveau est utile lorsque vous
    exécutez des programmes TPU dans des environnements comme Colab et les
    centres de données. Il est désactivé par défaut, mais vous pouvez l'activer
    en indiquant un emplacement de stockage.

Chaque entrée est un exécutable TPU compilé qui représente un graphe de calcul
StableHLO généré à partir d'une opération ATen ou d'une séquence d'opérations
ATen. Les graphes StableHLO sont représentés par des empreintes qui combinent
la structure du graphe, les formes des tensors d'entrée et de sortie, et les
options du compilateur, y compris les versions du matériel TPU, de libtpu et du
runtime PjRt.

## Prise en charge de `torch.compile` { #torchcompile-support }

Pour offrir une expérience utilisateur cohérente avec les autres matériels qui
prennent en charge `torch.compile()`, nous enregistrons un nouveau
[backend](https://docs.pytorch.org/docs/stable/torch.compiler_custom_backends.html)
TorchDynamo, que vous pouvez utiliser comme suit : `compiled_model = torch.compile(model,
backend="tpu")`.

Dynamo appelle le backend TorchTPU en lui transmettant un `torch.fx.GraphModule`
et une liste de tensors d'entrée d'exemple. Le runtime trace alors le graphe
fourni comme en mode eager, mais avec toutes les opérations différées et sans
aucune exécution sur le TPU. Le runtime génère et compile un seul programme
StableHLO pour l'ensemble du graphe FX et renvoie un callable Python qui
encapsule l'appel. Notez que TorchTPU n'utilise pas `TorchInductor` : il
réutilise plutôt la pile mature du compilateur XLA, développée en parallèle de
l'architecture TPU.

## Pour aller plus loin { #more-detail }

Les documents suivants donnent plus de détails sur :

*   Le [modèle de mémoire](./torchtpu_memory_model.md) pris en charge par le TPU
    et ses implications pour PyTorch ;
*   Les [modes d'exécution](./torchtpu_execution_model.md) pris en charge par
    TorchTPU ;
*   La prise en charge des [kernels personnalisés](./custom_kernels.md) ;
*   L'[exécution distribuée](./distributed_support.md) sur des clusters de TPU.
