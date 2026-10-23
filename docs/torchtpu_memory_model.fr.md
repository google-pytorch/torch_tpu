# TorchTPU et le modèle de mémoire des TPU { #torchtpu-and-the-tpu-memory-model }

PyTorch a évolué en même temps que les GPU et met efficacement leur puissance à
la disposition des utilisateurs de Python. TorchTPU étend PyTorch pour exprimer
la puissance de XLA sur les TPU.

Le matériel physique des GPU et des TPU est différent, tout comme les
paradigmes de programmation qui ciblent chacune de ces architectures
matérielles. TorchTPU existe pour combler cet écart dans les deux sens. Nous
cherchons à prendre en charge dans XLA les interfaces natives des GPU, comme
les vues de tensors et les streams, tout en exposant la puissance propre au
compilateur XLA dans des interfaces PyTorch familières.

Comprendre comment TorchTPU atteint ces objectifs peut aider les utilisateurs à
optimiser des bases de code natives TPU pour en tirer le maximum.

## Matériel : CPU, GPU, TPU { #hardware-cpus-gpus-tpus }

Un CPU à usage général (comme celui de votre ordinateur ou de votre appareil
mobile) se compose de plusieurs cœurs, chacun prenant en charge un ou plusieurs
threads. Chaque thread peut s'exécuter en parallèle, tout en accédant à une
mémoire partagée en lecture/écriture à accès aléatoire.

Un GPU ressemble à bien des égards à un CPU, mais il possède beaucoup plus de
threads. Un CPU moderne de forte puissance peut avoir quelques dizaines de
threads, alors qu'un GPU de forte puissance en a des *milliers*. En
contrepartie, chaque thread de GPU est individuellement un peu moins puissant
qu'un thread de CPU, et l'indépendance des threads est soumise à certaines
restrictions. Les GPU restent néanmoins utiles pour les charges de travail aux
opérations hautement parallélisables, comme le rendu graphique ou l'algèbre
linéaire.

Un TPU repose sur une conception fondamentalement différente. Le cœur du TPU
est le TensorCore et ses unités de multiplication matricielle (MXU). Ces MXU
fonctionnent comme un
[réseau systolique](https://en.wikipedia.org/wiki/Systolic_array), un peu à la
manière d'un registre SIMD sur un CPU. Elles utilisent des registres larges, à
deux dimensions, qui peuvent effectuer nettement plus de calcul dense par
cycle d'horloge qu'un thread de CPU ou de GPU.

Toutefois, le TensorCore est fondamentalement monothread : il n'est pas
possible de subdiviser un réseau systolique pour effectuer plusieurs calculs
simultanés. De plus, avec des registres larges, un TensorCore est très peu
efficace pour les opérations de lecture/écriture creuses à accès aléatoire,
car un grand bloc de mémoire doit être chargé en une seule fois.

Les générations de TPU les plus récentes complètent leurs TensorCores par des
SparseCores qui, comme leur nom l'indique, sont spécialisés dans les opérations
creuses, y compris les opérations de lecture/écriture plus ciblées. Ces cœurs
séparés permettent aussi un parallélisme limité : chaque SparseCore peut
effectuer des opérations en même temps que chaque TensorCore.

## Modèle de programmation : CUDA ou XLA { #programming-model-cuda-vs-xla }

Les interfaces logicielles natives des GPU et des TPU sont adaptées aux
capacités de ces plateformes.

Prenons l'exemple de CUDA, le modèle de programmation GPU le plus utilisé sous
PyTorch. CUDA est une interface de programmation de bas niveau, semblable au C
ou au C++. Comme la mémoire du CPU, les buffers CUDA sont intrinsèquement des
tableaux d'octets à une dimension. Un kernel CUDA est une procédure impérative :
une fonction void, pour reprendre la terminologie du C. Chaque thread reçoit
des pointeurs vers certains buffers et peut lire à n'importe quel décalage de
n'importe quel buffer, puis y réécrire, octet par octet.

Ce modèle de programmation ne fonctionne pas pour les TPU. La mémoire des TPU
n'est pas organisée en tableaux d'octets à une dimension, mais en tuiles à deux
dimensions alignées sur les registres. Et, comme indiqué plus haut, les
TensorCores n'ont pas de threads et sont peu efficaces pour l'accès aléatoire.

Le compilateur des TPU, XLA, utilise plutôt un modèle déclaratif et
fonctionnel. XLA accepte des programmes écrits dans un langage appelé
Multi-Level Intermediate Representation (MLIR), qui comprend plusieurs
dialectes. Le dialecte StableHLO permet de décrire la sortie logique de tensor
attendue d'un programme. XLA compile ce programme en opérations internes de plus
bas niveau, optimisées pour votre génération de TPU.

## Buffers : `torch.UntypedStorage` { #buffers-torchuntypedstorage }

Les outils fondamentaux de PyTorch sont très étroitement alignés sur le modèle
de programmation CUDA.

Un objet de stockage, de la classe Python `torch.UntypedStorage`, n'est guère
plus qu'une adresse mémoire (`torch.UntypedStorage.data_ptr()`) associée à une
taille en octets (`torch.UntypedStorage.nbytes()`). Cela correspond bien aux
interfaces du CPU comme de CUDA.

Ce n'est toutefois pas suffisant pour l'exécution sur TPU : dans XLA, un buffer
est toujours bien typé et possède une forme multidimensionnelle connue. Comme
la mémoire interne d'un TPU est alignée sur des registres 2D, il est essentiel
de savoir comment un buffer est disposé. Un `torch.UntypedStorage` seul ne
contient donc pas assez d'informations pour l'exécution sur TPU.

Pour résoudre ce problème, TorchTPU ajoute une couche supplémentaire. Sur TPU,
un `torch.UntypedStorage.data_ptr` ne pointe pas directement vers la mémoire du
TPU : il pointe vers un emplacement de la mémoire du CPU qui contient des
métadonnées supplémentaires sur le buffer TPU. **C'est la clé du
fonctionnement de TorchTPU** : ces métadonnées supplémentaires permettent une
interprétation active entre l'API existante de PyTorch, inspirée de CUDA, et la
réalité de la conception fonctionnelle de XLA et du matériel TPU.

## `torch.Tensor` { #torchtensor }
Un `torch.Tensor` peut être vu comme deux choses à la fois.

D'abord, un `torch.Tensor` est la représentation *physique* d'un motif d'accès à
un `torch.UntypedStorage` sous-jacent. Un `Tensor` contient des métadonnées
essentielles : les strides (facteurs multiplicatifs), les tailles (bornes des
indices), le décalage (emplacement de l'élément 0) et le type d'élément (qui
fixe le nombre d'octets par élément logique). Ces métadonnées permettent de
faire correspondre un tuple d'indices dans un tensor à un décalage en octets
dans le tableau linéaire 1D du buffer qui sous-tend ce `Tensor`.

Ensuite, un `torch.Tensor` est la représentation *logique* d'un objet
mathématique. Lorsque vous multipliez deux tensors matriciellement, le résultat
est un autre `Tensor`, de forme appropriée, qui représente le résultat. Le motif exact
des strides et le décalage dans le stockage importent peu ici : ce que le
`Tensor` signifie en tant que calcul compte davantage que ce qu'il est en tant
que référence mémoire.

Lors du passage de CUDA au TPU, il est essentiel de comprendre cette
distinction : le comportement **logique** des opérations PyTorch sur les
tensors peut être reproduit correctement sur XLA/TPU, même si la réalité
**physique** de l'appareil est intrinsèquement différente !

## Décomposition des vues { #view-decomposition }

Sur CUDA, le langage des strides d'un `torch.Tensor` est utilisé directement
par les kernels CUDA internes : les strides sont transmis aux procédures
internes des threads pour décrire comment calculer les décalages dans le buffer
concerné. Ces décalages sont calculés dynamiquement à l'exécution.

Sur TPU, en revanche, les strides physiques ne sont pas directement
utilisables : une description des strides à une dimension ne convient pas à
une mémoire à deux dimensions alignée sur les registres. Le comportement
logique des vues *est* toutefois pris en charge via StableHLO. La plupart des
opérations de vue sur les tensors ont un équivalent direct en StableHLO :
reshape, transposition, découpage (slicing) et broadcasting.

Lorsqu'il reçoit un motif de strides de `torch.Tensor`, TorchTPU effectue une
opération appelée *décomposition des vues*. Elle interprète un motif de strides
et un type d'élément comme une séquence d'opérations logiques qui produisent le
tensor de sortie voulu, avec chaque élément au bon endroit.

Par exemple :

```python
# Logical values: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
# sizes=(10,), strides=(1,), storage_offset=0
base = torch.arange(10, dtype=torch.float)

# Logical values: [[[1., 1.], [4., 4.]], [[3., 3.], [6., 6.]]]
physical_view = base.as_strided(
  sizes=(2, 2, 2),
  strides=(2, 3, 0),
  storage_offset=1
)

# Logical values: [[[1., 1.], [4., 4.]], [[3., 3.], [6., 6.]]]
# StableHLO: slice, reshape, slice, transpose, reshape, broadcast_in_dim
logical_view = base[1:].view(3, 3)[:2, ::2].t().unsqueeze(-1).expand(2, 2, 2)
```

La version de bas niveau qui manipule les strides avec
`torch.Tensor.as_strided` peut s'exprimer de façon équivalente par une séquence
d'opérations plus familières, qui peuvent à leur tour s'exprimer en StableHLO
et être compilées par XLA.

Notez toutefois qu'**une équivalence logique ne signifie pas des performances
équivalentes**. Une opération de reshape ou de transposition peut, dans certains
cas, nécessiter une copie complète sur TPU pour passer d'une disposition
alignée sur les registres à une autre. Sur TorchTPU, **les vues ne sont donc
pas « gratuites »** comme elles le sont souvent sur CUDA, en raison des
différences physiques entre le matériel GPU et le matériel TPU.

## Mutations sur place et don de buffer { #in-place-mutations-and-buffer-donation }

StableHLO est un langage de programmation fonctionnel. Plus précisément encore,
c'est un
[langage à assignation unique statique (SSA)](https://en.wikipedia.org/wiki/Static_single-assignment_form).
Les valeurs ne peuvent donc qu'être créées : jamais modifiées ni réassignées.
Cela aide le compilateur XLA à produire des plans d'exécution optimisés.

Ni les IR des GPU ni celles des CPU ne sont purement fonctionnelles, et par
conséquent PyTorch ne l'est pas non plus. Les opérations sur les tensors dont le
nom se termine par un trait de soulignement, ou qui utilisent l'argument
nommé « `out=` », écrasent les octets de leur `torch.UntypedStorage` de base.
Comme plusieurs `torch.Tensors` peuvent partager un même buffer de stockage,
une écriture dans n'importe quel tensor devient immédiatement visible par tous
les autres tensors.

TorchTPU gère ce cas en exploitant sa couche de métadonnées. Lorsqu'une
opération PyTorch demande une mise à jour sur place ou une affectation à un
tensor de sortie, TorchTPU :

*   Décrit la mise à jour logique à XLA, ce qui produit un nouveau buffer
*   Remplace le buffer à l'intérieur de l'objet de métadonnées pointé par le
    `torch.UntypedStorage`

Il s'agit en fait d'une fonctionnalisation automatique : toutes les mutations
sont exprimées à la place sous forme d'opérations hors place.

Pour réaliser cette fonctionnalisation automatique, il faut pouvoir exécuter
l'algorithme de décomposition des vues en sens inverse, ce que nous appelons
l'inversion des vues. Elle n'est possible que pour les vues sans
chevauchement. Écrire dans un tensor issu d'un broadcast n'est pas valide, car
plusieurs emplacements logiques correspondent à un seul emplacement physique,
ce qui rend l'instruction ambiguë. Sur le backend CUDA, l'opération est
considérée comme un comportement indéfini, mais peut échouer explicitement ou
non. Sur TorchTPU, elle échoue toujours explicitement.

Dans le cas d'une écriture partielle, l'inversion des vues utilise une
opération StableHLO appelée
[`stablehlo.dynamic_update_slice`](https://openxla.org/stablehlo/spec#dynamic_update_slice)
pour calculer un tensor de sortie fusionné, qui contient à la fois les valeurs
écrasées et non écrasées, aux bons emplacements.

Bien entendu, une opération hors place double le pic de mémoire : l'ancienne
et la nouvelle version coexistent brièvement avant que l'ancienne valeur soit
supprimée. Pour éviter cela, TorchTPU tente, lorsque c'est possible,
d'appliquer l'équivalent le plus proche des mutations sur place dans XLA : le
**don de buffer** (buffer donation). Lorsque la dernière référence à un buffer
obsolète est supprimée, TorchTPU annote le StableHLO pour « donner »
l'entrée. XLA peut alors consommer l'entrée, ce qui évite l'augmentation de
mémoire due à la fonctionnalisation automatique.

Il n'est pas toujours possible de donner un buffer en toute sécurité. Si
TorchTPU ne peut pas prouver qu'un buffer peut être donné sans risque, il se
rabat sur une mise à jour hors place. Cela peut être particulièrement coûteux
pour de petites écritures dans de grands buffers : dans ce cas, la mise à jour
hors place copie l'intégralité du tensor de stockage de base, et pas seulement
la partie écrite !

## Copie à l'écriture { #copy-on-write }

La fonctionnalisation présente toutefois des avantages. L'un d'eux est la
possibilité d'adopter un comportement de copie à l'écriture (copy-on-write).

Sur le backend CUDA, l'appel à `torch.Tensor.clone()` alloue immédiatement un
nouveau buffer de mémoire physique suffisamment grand pour contenir les données
du tensor. Ensuite, toutes les écritures sur place dans les différents
`torch.UntypedStorages` sont entièrement indépendantes, mais le coût en mémoire
est payé immédiatement.

Sur le backend TorchTPU, en revanche, l'immuabilité d'une valeur assignée
statiquement permet à une copie d'être une simple référence à l'original, sans
qu'aucune copie physique immédiate soit nécessaire.

Dès qu'une mise à jour sur place a lieu, nous devons la répercuter uniquement
sur le `torch.UntypedStorage` qui a été modifié, et pas sur l'autre. Cela
empêche le don de buffer (puisque nous avons toujours besoin de l'ancienne et
de la nouvelle valeur) : dans ce cas, la mise à jour sur place est donc
effectuée hors place, sans don. Cette mise à jour hors place alloue de la
nouvelle mémoire, mais cette mémoire n'est allouée que de façon paresseuse, au
moment où elle est nécessaire, contrairement à PyTorch/CUDA, qui paie le coût
en mémoire d'avance.

## Kernels personnalisés { #custom-kernels }

TorchTPU fait de son mieux pour appliquer des dons de buffer qui optimisent la
mémoire, et fournit une transformation logiquement équivalente des vues CUDA à
strides vers des opérations de vue StableHLO. Cela dit, il s'agit toujours
d'opérations de haut niveau, qui ne sont pas directement liées à la disposition
physique d'un buffer TPU et ne garantissent aucune sémantique de bas niveau
particulière.

Pour les utilisateurs qui veulent un contrôle total de bas niveau, TorchTPU
prend aussi en charge l'intégration de kernels personnalisés. Ces kernels
permettent de remplacer de façon chirurgicale des opérations critiques, voire
de réécrire entièrement un modèle, en spécifiant manuellement le comportement
des buffers TPU. Vous pouvez ainsi garantir un comportement optimal pour le don
de buffer et les mises à jour sur place.

## Points clés { #takeaways }
Lorsque vous travaillez avec TorchTPU, retenez les points clés suivants :

*   **Le code PyTorch produit le même résultat logique sur CUDA et sur
    TorchTPU :** vous pouvez utiliser des opérations CUDA de bas niveau
    familières (vues à stockage partagé, manipulation des strides, mises à jour
    sur place) et obtenir les mêmes résultats sur TPU, même si les performances
    peuvent différer.
*   **Les copies sont peu coûteuses :** copier un tensor sous une forme
    contiguë est souvent gratuit en calcul, grâce au comportement de copie à
    l'écriture.
*   **Les vues sont coûteuses :** lire un tensor avec une autre forme ou selon
    un motif non contigu peut forcer une redisposition sur le TPU, ce qui peut
    être coûteux. Privilégiez les tensors contigus lorsque c'est possible.
*   **Écraser un tensor contigu est généralement économe en mémoire :** un
    tensor contigu peut généralement être écrasé grâce à un don de buffer si
    toutes les anciennes valeurs sont effacées.
*   **Les écritures partielles sont parfois coûteuses en mémoire :** si le don
    de buffer n'est pas possible, le coût en mémoire d'une écriture est
    proportionnel à la taille du torch.UntypedStorage, et non à la taille des
    données écrites. Privilégiez de nombreux petits tensors plutôt que des
    portions de grands tensors.
*   **Les kernels personnalisés offrent un contrôle total :** lorsque la
    traduction de CUDA vers XLA échoue ou est insuffisante, implémenter un
    kernel personnalisé natif TPU permet d'optimiser des aspects précis des
    performances de bas niveau.
