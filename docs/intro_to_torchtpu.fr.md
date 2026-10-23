# Présentation de TorchTPU { #introduction-to-torchtpu }

<!-- Link to colab and to generated notebook -->

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/intro_to_torchtpu.ipynb)

TorchTPU est un backend PyTorch natif et open source pour les
[Cloud TPU de Google](https://cloud.google.com/tpu), développé en collaboration
entre Google et l'équipe PyTorch de Meta. Avec TorchTPU, les développeurs
peuvent exécuter des charges de travail PyTorch existantes sur du matériel TPU
avec un minimum de modifications du code, et faire évoluer les charges de
travail d'entraînement et d'inférence d'une seule puce jusqu'à des pods TPU
à plusieurs tranches.

## Pourquoi utiliser TorchTPU ? { #why-use-torchtpu }

Les TPU Google sont des circuits intégrés spécifiques à une application (ASIC)
conçus pour les opérations matricielles et vectorielles qui alimentent le deep
learning. Ils offrent un débit élevé, une bonne efficacité énergétique et un
bon rapport performances/prix à grande échelle. Historiquement, les TPU ont
été associés aux écosystèmes JAX et TensorFlow. TorchTPU apporte une prise en
charge native des TPU directement dans
l'[écosystème PyTorch](https://github.com/google-pytorch/torch_tpu). Il a été
conçu délibérément autour des trois principes directeurs suivants :

*   **Intégration native à PyTorch :** TorchTPU s'intègre directement à PyTorch,
    sans dépendance au runtime JAX. Il conserve la sémantique standard de
    PyTorch pour les vues de tensors et les mutations sur place, adopte le
    modèle d'exécution de PyTorch à un processus par appareil et prend en
    charge `torch.compile` via Dynamo.
*   **Facilité d'utilisation immédiate :** pour porter une charge de travail
    d'un GPU vers un TPU, commencez par remplacer `device="cuda"` par
    `device="tpu"` et `backend="nccl"` par `backend="tpu_dist"`. Votre modèle
    PyTorch non modifié s'exécute immédiatement en mode eager, avec des messages
    d'erreur clairs pour diagnostiquer les problèmes du runtime ou du
    compilateur.
*   **Performances à l'échelle de la production :** au-delà de l'exécution eager
    opération par opération, TorchTPU propose un mode eager à fusion différée,
    un pipeline de compilation de Dynamo vers XLA, un dynamisme borné qui évite
    les recompilations dues aux formes, et des API de kernels personnalisés pour
    vous aider à exploiter le matériel au maximum.

## Démarrage rapide : exécution sur TPU { #quickstart-running-on-tpu }

La cellule suivante présente un exemple simple de programme PyTorch qui
s'exécute sur TPU. Vous pouvez exécuter
[cet exemple sur Colab](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/intro_to_torchtpu.ipynb)
en activant l'environnement d'exécution TPU.

```python
import torch

device = torch.device("tpu")
x = torch.randn((4, 4), device=device)
y = torch.ones((4, 4), device=device)

result = torch.matmul(x, y).cpu()
print(f"Executed on device: {x.device}")
print(f"Result shape on CPU: {result.shape}, sum: {result.sum():.4f}")
```

## Fonctionnalités clés { #key-features }

*   **Large couverture des opérateurs et des modèles :** prend en charge
    450 opérateurs ATen de base (passes avant et arrière) et fait l'objet d'une
    validation continue sur 5 000 modèles HuggingFace populaires, qui couvrent
    des LLM (Llama, Qwen, Gemma, DeepSeek, Mistral), des modèles de vision (ViT,
    ResNet, ConvNeXt, DINOv2), des modèles audio (Whisper, Wav2Vec2) et des
    pipelines de diffusion (FLUX, Stable Diffusion).
*   **Trois modes d'exécution eager :**
    *   **Strict (par défaut) :** envoie de façon asynchrone un opérateur
        ATen à la fois au TPU, ce qui correspond au comportement eager
        standard de PyTorch sur GPU.
    *   **Debug (`TORCH_TPU_LAUNCH_BLOCKING=1`) :** exécute chaque opérateur de
        façon synchrone avec le CPU et active la vérification des bornes à
        l'exécution pour simplifier le débogage.
    *   **DeferAndFuse (`TORCH_TPU_DEFER_AND_FUSE=1`) :** diffère et fusionne
        des séquences d'opérateurs ATen en blocs de calcul optimisés jusqu'à un
        point de matérialisation (comme `.cpu()` ou `.item()`). Ce mode réduit
        nettement le trafic vers la mémoire à haut débit (HBM) et accélère
        l'exécution eager.
*   **Compilation de graphes avec `torch.compile` :** enregistre le backend
    Dynamo `"tpu"` (`torch.compile(model)`) pour abaisser des graphes FX
    PyTorch entiers en [StableHLO](https://openxla.org/stablehlo/spec) et les
    compiler avec le [compilateur OpenXLA](https://openxla.org/xla). La prise en
    charge active des formes dynamiques (`torch._dynamo.mark_dynamic`) permet
    aux graphes compilés de gérer des tailles de batch ou des longueurs de
    séquence variables, jusqu'à une borne indiquée, sans recompilation.
*   **Dynamisme borné :** limite nativement les recompilations causées par des
    tailles de batch et des longueurs de séquence variables, en compilant par
    rapport à des bornes supérieures dynamiques.
*   **Cache de compilation à plusieurs niveaux :** élimine les compilations XLA
    redondantes grâce à trois niveaux de cache :
    *   **Niveau 1 :** cache en mémoire local au processus (toujours activé).
    *   **Niveau 2 :** mémoire partagée locale à l'hôte, pour la réutilisation
        entre processus dans les jobs sur plusieurs appareils.
    *   **Niveau 3 :** cache distant persistant (comme Cloud Storage), pour la
        réutilisation entre hôtes et entre redémarrages de jobs.
*   **Primitives distribuées natives :** implémente les opérations collectives
    TPU au niveau du Process Group `torch.distributed` (`c10d::Backend`) via
    `backend="tpu_dist"`, avec prise en charge de `DistributedDataParallel`
    (DDP), `FullyShardedDataParallel` (FSDP2 et SimpleFSDP) et `DTensor`. Vous
    pouvez aussi utiliser `torch.tpu.distributed.topology_aware_mesh` pour
    aligner les dimensions de parallélisme sur les interconnexions physiques des
    TPU.
*   **Kernels TPU personnalisés :** écrivez des kernels personnalisés en
    [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) (appelables via
    `torch.tpu.pallas.jax_op`, avec prise en charge directe des opérations
    collectives RDMA entre puces), écrivez des kernels par tuiles avec le DSL
    expérimental [Helion](https://pytorch.org/blog/helion/), ou importez des
    kernels TPU prédéfinis et optimisés depuis Tokamax.
*   **Intégrations à l'écosystème :** fonctionne immédiatement avec les
    frameworks standards de serving et d'entraînement PyTorch et avec les outils
    de profilage, notamment `torch.profiler` (TensorBoard/XProf), vLLM, SGLang,
    TorchTitan, Ray et TorchRec.

## Limites et travaux futurs { #limitations-and-future-work }

Comme tout produit, TorchTPU présente des limites connues, qui seront traitées
au fil du temps. Notre première version offre une couverture solide et
immédiate des principales charges de travail, mais TorchTPU est un effort
collaboratif porté par la communauté. Consultez nos
[fonctionnalités et notre feuille de route](./feature_set.md) pour voir ce qui
se prépare, ou participez directement en
[soumettant des PR](https://github.com/google-pytorch/torch_tpu) sur GitHub pour
nous aider à lever ces limites. Il s'agit d'un projet communautaire : nous
serions ravis d'avoir de vos nouvelles et vous invitons à collaborer avec nous
via [GitHub](https://github.com/google-pytorch/torch_tpu),
[https://discuss.pytorch.org/c/tpu](https://discuss.pytorch.org/c/tpu),
[https://dev-discuss.pytorch.org/c/tpu/](https://dev-discuss.pytorch.org/c/tpu/)
et [torchtpu-feedback@google.com](mailto:torchtpu-feedback@google.com).

*   Les formes dynamiques provoquent des recompilations
    *   Sur TorchTPU, `torch.compile` définit `dynamic=False` par défaut.
        `dynamic=None` et `dynamic=True` ne sont pas encore pris en charge en
        production.
        [Voir la discussion sur les formes dynamiques](./dynamic_shapes.md)
    *   Par exemple, l'exécution de l'étape decode/generate sur des LLM Hugging
        Face entraîne des recompilations à chaque étape, sauf si le dynamisme
        borné est activé.
*   Les opérations de vue peuvent être coûteuses sur TPU (voir
    [Modèle de mémoire de TorchTPU](./torchtpu_memory_model.md)).
*   Le MPMD avec opérations collectives est pris en charge sur le plan
    fonctionnel, mais n'atteint pas encore les performances maximales. Le
    chevauchement des opérations collectives avec le calcul est soumis à
    [certaines contraintes](./distributed_support.md) :
    *   La prise en charge complète des streams avec déchargement vers le
        SparseCore arrivera prochainement.
    *   Utilisez SimpleFSDP avec `torch.compile()` plutôt que FSDPv2 en mode
        eager ou avec `torch.compile`. Consultez les
        [bonnes pratiques FSDP](./case_studies_fsdp.md)
    *   La méthode recommandée pour obtenir les meilleures performances des
        optimisations XLA sur les modules qui communiquent (c'est-à-dire pour
        permettre le chevauchement calcul/communication) consiste à utiliser le
        décorateur `spmd_safe` sur les régions dont **les utilisateurs
        garantissent** qu'elles sont exactement identiques
*   TorchTPU ne prend pas encore en charge FlexAttention de façon native
*   Les TPU de la génération actuelle ne prennent pas en charge l'accumulation
    des matmul dans une précision inférieure à fp32 (par exemple, tf32).
*   Un seul processus peut accéder à un appareil TPU, ce qui limite la
    prise en charge du RL co-localisé (notez qu'une puce TPU peut être associée
    à un ou plusieurs appareils, accessibles par des processus distincts du
    système d'exploitation).

**Autres recommandations :**

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Limite | Recommandation |
| :--- | :--- |
| Les modes eager ont un temps de préchauffage non nul | Activez le cache de compilation persistant |
| Torch.compile prend beaucoup de temps | Activez le cache de compilation persistant |
| Les CNN dans PyTorch, qui utilisent par défaut une disposition channel-first, sont très peu efficaces sur TPU | Utilisez cette [option PyTorch](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html) pour définir une disposition channel-last et obtenir de meilleures performances. |
| Le mappage du maillage d'appareils peut être sous-optimal par défaut, en raison des différences de topologie entre les TPU et les autres architectures | Utilisez `topology_aware_mesh` pour obtenir un mappage optimal du maillage d'appareils et pouvoir le personnaliser |
<!-- mdformat on -->

## Premiers pas { #getting-started }

Cette documentation est organisée en quatre sections structurées pour vous
accompagner de la première configuration jusqu'au déploiement distribué en
production :

*   **Premiers pas :** tutoriels pas à pas pour configurer votre environnement
    et déployer des charges de travail sur [une VM](./get_started_on_a_vm.md)
    ou [via GKE](./get_started_on_gke.md).
*   **Approfondissement :** présentations détaillées du
    [modèle de mémoire des TPU](./torchtpu_memory_model.md), de
    l'[exécution distribuée](./distributed_support.md) et des différents
    [modes d'exécution](./torchtpu_execution_model.md).
*   **Études de cas :** recettes de bout en bout couvrant les optimisations du
    [modèle Flux](./case_studies_flux.md), l'entraînement de
    [Muse Glimmer](./case_studies_muse_glimmer_30b_training.md),
    l'[inférence LLM de bout en bout](./case_studies_muse_glimmer_30b_inference.md)
    et l'utilisation efficace de [FSDP](./case_studies_fsdp.md).
*   **Analyser votre charge de travail :**
    [profilage](./performance_profiling_with_xprof.md), gestion de la
    [précision numérique](./numerical_precision.md)
