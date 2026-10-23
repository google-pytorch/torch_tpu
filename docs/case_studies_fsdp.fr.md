# Bonnes pratiques FSDP { #fsdp-best-practices }

Fully Sharded Data Parallel (FSDP) sharde les paramètres du modèle, les
gradients et l'état de l'optimiseur entre les rangs de parallélisme de données,
de sorte que chaque cœur TPU ne stocke qu'une fraction `1 / world_size` du
modèle. Avant l'exécution d'une couche, ses poids complets sont reconstruits
avec un **all-gather** ; pendant la passe arrière, les gradients sont combinés
et de nouveau shardés avec un **reduce-scatter**.

Ce guide explique la méthode recommandée pour utiliser FSDP avec TorchTPU et
renvoie vers des exemples exécutables dans
[`examples/fsdp`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp).

Dans ce guide, vous allez découvrir :

1.  **Quelle approche FSDP choisir** avec TorchTPU.
2.  **SimpleFSDP (recommandé)** : FSDP pensé pour la compilation, avec un seul
    appel d'encapsulation.
3.  **FSDP manuel** : placer les opérations collectives directement dans la
    définition du modèle.
4.  **Les bonnes pratiques** qui s'appliquent à toutes les approches.
5.  **L'exécution des exemples** : commandes de lancement et ce que vérifie
    chaque exemple.

## 1. Choisir une approche FSDP { #1-choosing-an-fsdp-approach }

```
On TPU, performance comes from compiling the whole training step into a single
XLA program with `torch.compile(backend="tpu")`. When the XLA compiler sees all
forward, backward, and collective operations at once, it can schedule
collectives to overlap with computation and combine small collectives when
profitable.

Choose an FSDP approach based on how well it composes with whole-graph
compilation:
```

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
    | Approche | Quand l'utiliser | Exemple |
    | :------- | :------------- | :------ |
    | **SimpleFSDP** (`simple_fsdp.data_parallel`) | **Choix par défaut recommandé.** Vous voulez utiliser FSDP ou HSDP sans modifier la définition du modèle.| [`fsdp_to_simple_fsdp/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp) |
    | **FSDP manuel** (opérations collectives manuelles) | Vous êtes responsable du code du modèle et vous voulez placer le code de communication à côté du calcul pour optimiser les performances. | [`manual/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual) |
    | **PyTorch FSDPv2** (`fully_shard`) | Code existant qui n'a pas encore été migré. Il fonctionne, mais ses hooks d'exécution provoquent des ruptures de graphe et, pour obtenir des performances compétitives, il nécessite la prise en charge à venir du déchargement des streams vers le SparseCore du TPU.
.| [Guide de migration](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md) |
<!-- mdformat on -->

```
SimpleFSDP and manual FSDP both express the collectives as part of the traced
graph, so XLA can optimize them together with the rest of the step. PyTorch
FSDPv2 (`fully_shard`) relies on Python pre-forward and post-backward hooks.
These hooks split the step into many smaller graphs, even when the model is
compiled.

For a side-by-side comparison of compiled FSDPv2 and compiled SimpleFSDP, see
the
[FSDPv2 to SimpleFSDP tutorial](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md).
```

## 2. Approche recommandée : SimpleFSDP { #2-recommended-simplefsdp }

[`simple_fsdp`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/simple_fsdp.py)
est une implémentation de FSDP conçue pour fonctionner avec `torch.compile`.
Elle a été introduite dans TorchTitan, et une copie autonome qui ne dépend que
de PyTorch est fournie dans `examples/fsdp/simple_fsdp.py`.

SimpleFSDP stocke chaque paramètre sous la forme d'un `DTensor` shardé. Lorsqu'un
paramètre est lu, une paramétrisation appelle `DTensor.redistribute` pour
effectuer son all-gather. La passe arrière autograd de `redistribute` effectue
le reduce-scatter des gradients. Il n'y a aucun hook d'exécution, si bien que
toute l'étape se compile sans rupture de graphe. Le modèle entier est shardé
avec un seul appel à `simple_fsdp.data_parallel(...)`, sans encapsulation par
couche ni réglage du prefetch, et HSDP est pris en charge avec
`mode="hybrid_shard"` sur un `DeviceMesh` 2D.

Pour apprendre à utiliser SimpleFSDP, consultez :

*   [README de SimpleFSDP](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/README.md) :
    démarrage rapide et options disponibles.
*   [Tutoriel de migration de FSDPv2 vers SimpleFSDP](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md) :
    comment migrer du code `fully_shard` existant.

## 3. Autre approche : FSDP manuel { #3-alternative-manual-fsdp }

Si vous contrôlez la définition du modèle, vous pouvez écrire FSDP directement
avec des opérations collectives manuelles
(`torch.distributed._functional_collectives`) et `torch.compile`. Chaque couche
ne stocke que son shard de poids et effectue un all-gather du poids complet
pendant la passe avant. Les opérations collectives fonctionnelles disposent de
formules autograd : la passe arrière de l'all-gather est donc un
reduce-scatter, et vous n'avez pas besoin d'écrire une
`torch.autograd.Function` personnalisée.

Cette approche garde le code de communication à côté du code de calcul, ce qui
peut faciliter l'optimisation des performances. En contrepartie, vous devez
modifier la définition du modèle.

Pour apprendre à implémenter FSDP manuellement, consultez le
[guide FSDP manuel](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual/manual_fsdp.md).

## 4. Bonnes pratiques { #4-best-practices }

Ces recommandations s'appliquent à SimpleFSDP et à FSDP manuel.

1.  **Compilez toute l'étape d'entraînement avec
    `torch.compile(backend="tpu")`.** Si vous ne pouvez pas compiler toute
    l'étape, compilez au moins ensemble la passe avant, la perte et la passe
    arrière. Compiler uniquement le module du modèle ne produit pas un graphe
    unique, ce qui limite la capacité de XLA à chevaucher les opérations
    collectives avec le calcul.
2.  **Utilisez l'activation checkpointing sur chaque couche shardée.**
    Encapsulez la passe avant de chaque couche dans
    `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`. Les poids
    non shardés sont ainsi libérés après la passe avant de chaque couche au lieu
    d'être conservés jusqu'à la passe arrière, ce qui donne le comportement
    mémoire ZeRO-3 attendu de FSDP.
3.  **Faites en sorte que les dimensions shardées soient divisibles par la
    taille du maillage.** FSDP manuel l'exige. SimpleFSDP réplique les paramètres
    trop petits pour être shardés. Utilisez `shard_dim` pour choisir une
    dimension qui se divise sans reste.
4.  **Utilisez une graine aléatoire différente pour chaque rang lors de
    l'initialisation des poids shardés.** Si chaque rang initialise son propre
    shard avec la même graine, le poids complet est le même bloc aléatoire
    répété sur tous les rangs, ce qui peut nuire à la convergence. Pour obtenir
    des shards distincts et reproductibles, initialisez chaque rang avec une
    graine qui dépend du rang, par exemple `torch.manual_seed(base_seed + dist.get_rank())`.
5.  **Faites exécuter le même programme à tous les rangs.** Ne placez pas
    d'opérations qui forcent la matérialisation (comme `.item()`) dans des
    branches propres à un rang, comme `if
    rank == 0`.
6.  **Utilisez HSDP lorsque le sharding complet sur l'ensemble du maillage ne passe
    pas à l'échelle.** Utilisez `mode="hybrid_shard"` avec un maillage 2D lorsque
    votre tranche ICI est trop grande pour sharder efficacement le modèle, ou
    lorsque le sharding sur cette tranche génère trop de trafic réseau. Lorsque
    vous travaillez sur plusieurs tranches, il est presque toujours préférable de
    répliquer (parallélisme de données) sur le DCN, plus lent, et de sharder à
    l'intérieur de chaque tranche.
7.  **Vérifiez la mémoire et les valeurs numériques avant de passer à
    l'échelle.** Comparez les résultats à une référence non shardée ou FSDPv2
    avec `assert_close`, et mesurez la mémoire conservée à la frontière entre
    passe avant et passe arrière. Les exemples incluent une vérification de
    parité (`fsdp_to_simple_fsdp_worker.py`) et un test de mémoire
    (`manual_fsdp_memory_test.py`).

## 5. Exécuter les exemples { #5-running-the-examples }

Chaque exemple est fourni avec un script de lancement qui s'exécute sur
8 cœurs TPU (par exemple, `v6e-8`) :

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Exemple | Commande de lancement | Ce qui est vérifié |
:-------- | :------------- | :------------- |
| [`fsdp_to_simple_fsdp/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp) | `./run_fsdp_to_simple_fsdp.sh` | FSDPv2 compilé et SimpleFSDP <br> compilé ont des pertes et des <br> prédictions identiques, et leurs <br> temps d'étape sont indiqués. |
| [`manual/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual) | `./launch.sh` ou `python3 launch.py` | Les gradients et les prédictions de FSDP <br> manuel correspondent à une référence CPU <br> sur un seul appareil. |
| [`manual/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual) | `python3 manual_fsdp_memory_test.py` | Les poids non shardés sont libérés <br> après la passe avant. |
<!-- mdformat on -->
