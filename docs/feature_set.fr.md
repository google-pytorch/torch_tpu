# Fonctionnalités prises en charge par TorchTPU et travaux futurs { #torchtpu-supported-feature-set-and-future-work }

## Fonctionnalités actuellement prises en charge { #current-supported-feature-set }

Le tableau ci-dessous présente la couverture fonctionnelle actuelle à la date
de la première version open source.

Pour vous aider à évaluer TorchTPU pour vos charges de travail, nous classons
les fonctionnalités en deux niveaux :

*   **Validées :** fonctionnalités qui disposent de tests continus de bout en
    bout, d'une couverture CI/CD automatisée et d'une vérification par
    benchmarks sur des architectures de modèles standards.
*   **Expérimentales :** implémentations précoces, API émergentes ou
    fonctionnalités prototypes. Elles fonctionnent et sont disponibles pour
    recueillir des retours précoces, mais leur comportement ou leur API peuvent
    évoluer en fonction des retours de la communauté.

> **Note:** TorchTPU évolue rapidement et de manière ouverte. Pour connaître
> l'état le plus récent ou demander des fonctionnalités précises, consultez
> notre [GitHub](https://github.com/google-pytorch/torch_tpu).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
|  | **Validées** | **Expérimentales** |
| ----- | ----- | ----- |
| **Engagement auprès de la communauté PyTorch** | <ul><li>Version OSS de TorchTPU, vLLM, SGLang, TorchTitan, TorchRec</li><li>Acceptation des PR externes</li><li>Documentation / tutoriels</li><li>CI/CD</li><li>Colab</li></ul> | <ul><li>Agent TPU pour la migration ATEN</li></ul> |
| **Prise en charge des TPU** | <ul><li>v5e, v5p, v6e, v7</li></ul> |  |
| **Fonctionnalités de base** | <ul><li>Facilité de migration</li><li>Mélange eager \+ compilé</li><li>FW \+ BW</li><li>Torch.profile</li><li>Opérations dynamiques (recompilation)</li></ul> | <ul><li>SymInts</li><li>Tensors irréguliers (jagged)</li></ul> |
| **Architectures de modèles** | <ul><li>Transformer (Dense \+ MoE) : Qwen, Llama, Muse Glimmer, Gemma</li><li>Diffusion : WAN, FLUX</li><li>CNN</li></ul> | <ul><li>GLM, Kimi, DeepSeek, GPT OSS et bien d'autres (amélioration progressive)</li><li>Mamba</li><li>LSTM/GRU</li><li>DLRM</li></ul> |
| **Prise en charge des kernels** | <ul><li>Pallas</li><li>Tokamax</li></ul> | <ul><li>Helion</li></ul> |
| **Quantification** | <ul><li>FP8/FP4 sur le tensor complet</li><li>FP8 par canal</li></ul> |  |
| **Calcul distribué** | <ul><li>Maillage optimal</li><li>SimpleFSDP</li><li>Checkpointing</li><li>Déchargement des activations</li><li>Multitranche jusqu'à 8K puces</li></ul> | <ul><li>FSDPv2 (performances faibles)</li><li>MPMD (performances faibles)</li><li>TorchComm</li></ul> |
| **Intégrations à l'écosystème** | <ul><li>Hugging Face</li><li>Ray</li><li>vLLM / SGLang</li><li>TorchTitan</li><li>TorchRec</li></ul> |  |
| **Solutions prêtes pour la production** | <ul><li>Inférence (multi-hôte, désagrégation P/D)</li><li>Fine-tuning (jusqu'à \~8K puces)</li></ul> |  |
<!-- mdformat on -->

## Travaux futurs { #future-work }

Cette section décrit nos horizons de développement et nos pistes d'exploration.
Il ne s'agit pas d'un calendrier de publication ferme.

En tant que projet open source tourné vers la communauté, nos priorités de
développement s'adaptent volontairement aux retours de la communauté et à
l'adoption par les développeurs. Les fonctionnalités, les calendriers et les
approches d'architecture peuvent évoluer à mesure que nous découvrons les
charges de travail qui comptent le plus pour vous.

Nous organisons nos travaux à venir selon deux grands horizons :

*   **Priorités à court terme :** domaines de conception active, de
    prototypage en cours et de consolidation des performances que l'équipe
    principale traite actuellement en priorité.
*   **Explorations futures :** explorations d'architecture à long terme,
    intégrations potentielles à l'écosystème et éléments du backlog demandés
    par la communauté, que nous étudions activement.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
|  | **Priorités à court terme** | **Explorations futures** |
| ----- | ----- | ----- |
| **Engagement auprès de la communauté PyTorch** | <ul><li>Davantage de documentation / tutoriels / articles de blog</li></ul> | <ul><li>Agent TPU pour la migration des kernels</li></ul> |
| **Prise en charge des TPU** | <ul><li>v8i, v8t</li></ul> |  |
| **Fonctionnalités de base** | <ul><li>Streams</li><li>Capture de graphe</li><li>Mode eager avec démarrage rapide</li><li>Opérations dynamiques (sans recompilation)</li></ul> |  |
| **Architectures de modèles** | <ul><li>Prise en charge des modèles dès leur sortie</li></ul> |  |
| **Quantification** | <ul><li>FP8/FP4 par blocs avec torchao (performances maximales sur TPUv8t)</li></ul> |  |
| **Calcul distribué** | <ul><li>FSDPv2 (avec performances)</li><li>MPMD (avec performances)</li><li>Entraînement sur \> 8K puces</li><li>Checkpointing multiniveau</li><li>Goodput</li></ul> | <ul><li>Entraînement sur >> 8K puces</li></ul>  |
| **Intégrations à l'écosystème** | <ul><li>VeRL</li></ul> |  |
| **Solutions prêtes pour la production** | <ul><li>RL</li></ul> | <ul><li>Pré-entraînement</li></ul> |
<!-- mdformat on -->
