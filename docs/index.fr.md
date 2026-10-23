<p align="center">
 <img src="images/torchtpu_banner.png" alt="TorchTPU" style="width: 90%;">
</p>
<p align="center">
<a href="https://github.com/google-pytorch/torch_tpu"><b>GitHub</b></a> | <a href="https://discuss.pytorch.org/c/tpu/45"><b>Forum de discussion</b></a> | <a href="https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu"><b>PyTorch sur TPU</b></a>
</p>

## À propos { #about }

TorchTPU est un backend PyTorch natif et open source pour les Cloud TPU de
Google, développé en collaboration entre Google et Meta. Avec TorchTPU, vous
pouvez exécuter des modèles PyTorch existants sur du matériel TPU avec un
minimum de modifications du code, et faire évoluer les charges de travail
d'entraînement et d'inférence d'une seule puce jusqu'à des pods TPU
à plusieurs tranches.

## Que souhaitez-vous faire aujourd'hui ? { #what-are-you-trying-to-do-today }

<!-- mdformat off(reason: preserve MkDocs grid card comments and 4-space list indentation) -->
<!--mkdocs:<div class="grid cards two-columns" markdown>-->

- **<!--mkdocs::material-rocket-launch:{ .lg .middle }--> Premiers pas**

    Provisionnez des environnements, comprenez les architectures fondamentales et exécutez vos premières charges de travail.

    - [<!--mkdocs::octicons-arrow-right-24:--> Présentation de TorchTPU](./intro_to_torchtpu.md) : pourquoi TorchTPU a été créé, ce qu'il apporte et comment il peut vous aider.
    - [<!--mkdocs::octicons-arrow-right-24:--> Premiers pas avec TorchTPU sur une VM](./get_started_on_a_vm.md) : configurez TorchTPU sur une VM TPU.
    - [<!--mkdocs::octicons-arrow-right-24:--> Premiers pas avec TorchTPU sur GKE](./get_started_on_gke.md) : utilisez TorchTPU via GKE.
    - [<!--mkdocs::octicons-arrow-right-24:--> Fonctionnalités prises en charge et feuille de route](./feature_set.md) : fonctionnalités actuellement prises en charge et prévisions pour les prochaines versions.
    - [<!--mkdocs::octicons-arrow-right-24:--> Référence de l'API](./api_reference.md) : l'API TorchTPU et les variables d'environnement de la version actuelle.
    - [<!--mkdocs::octicons-arrow-right-24:--> FAQ TorchTPU](./faq.md) : questions fréquentes sur TorchTPU.

- **<!--mkdocs::material-sine-wave:{ .lg .middle }--> Architecture détaillée et concepts fondamentaux**

    Examinez en détail le fonctionnement des dispositions mémoire, des moteurs d'exécution et de l'exécution distribuée.

    - [<!--mkdocs::octicons-arrow-right-24:--> Fonctionnement de TorchTPU](./how_torchtpu_works.md) : comment TorchTPU s'intègre à PyTorch et au matériel TPU, à tous les niveaux de la pile.
    - [<!--mkdocs::octicons-arrow-right-24:--> Le modèle d'exécution de TorchTPU](./torchtpu_execution_model.md) : description détaillée de la façon dont TorchTPU exécute le code PyTorch et des différents modes d'exécution qu'il prend en charge.
    - [<!--mkdocs::octicons-arrow-right-24:--> Le modèle de mémoire de TorchTPU](./torchtpu_memory_model.md) : comment TorchTPU gère la façon dont le TPU et le compilateur XLA traitent la mémoire.
    - [<!--mkdocs::octicons-arrow-right-24:--> Prise en charge du calcul distribué](./distributed_support.md) : comment TorchTPU prend en charge les charges de travail distribuées.
    - [<!--mkdocs::octicons-arrow-right-24:--> Prise en charge des kernels personnalisés](./custom_kernels.md) : comment écrire des kernels personnalisés pour TorchTPU.
    - [<!--mkdocs::octicons-arrow-right-24:--> Formes dynamiques](./dynamic_shapes.md) : prise en charge des formes de tensors dynamiques.

- **<!--mkdocs::material-book-open-variant:{ .lg .middle }--> Études de cas**

    Découvrez des exemples concrets d'utilisation de TorchTPU pour résoudre des problèmes réels.

    - [<!--mkdocs::octicons-arrow-right-24:--> Optimisation de FLUX](./case_studies_flux.md) : exemple qui montre comment identifier et appliquer plusieurs optimisations sur FLUX.
    - [<!--mkdocs::octicons-arrow-right-24:--> Inférence Muse Glimmer 30B](./case_studies_muse_glimmer_30b_inference.md) : exemple de décodage itératif avec de vrais poids et un cache KV statique ou dynamique.
    - [<!--mkdocs::octicons-arrow-right-24:--> Entraînement Muse Glimmer 30B](./case_studies_muse_glimmer_30b_training.md) : une boucle d'entraînement sur une implémentation autonome de Muse Glimmer 30B.
    - [<!--mkdocs::octicons-arrow-right-24:--> Utiliser FSDP](./case_studies_fsdp.md) : l'approche recommandée pour utiliser FSDP avec TorchTPU.

- **<!--mkdocs::material-book-open-variant:{ .lg .middle }--> Analyser votre charge de travail**

    - [<!--mkdocs::octicons-arrow-right-24:--> Précision numérique et parité](./numerical_precision.md) : contrôler la précision dans TorchTPU.
    - [<!--mkdocs::octicons-arrow-right-24:--> Profiler les performances avec XProf](./performance_profiling_with_xprof.md) : comment utiliser XProf comme profileur pour TorchTPU.

<!--mkdocs:</div>-->
<!-- mdformat on -->
