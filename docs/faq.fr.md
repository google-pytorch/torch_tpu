# FAQ TorchTPU { #torchtpu-faq }

## Quelle est la différence entre PyTorch/XLA et TorchTPU ? { #what-is-the-difference-between-pytorchxla-and-torchtpu }

Historiquement, PyTorch/XLA servait de framework passerelle, fortement dépendant
de composants du runtime JAX et d'API spécialisées. TorchTPU fournit un backend
PyTorch natif pour les [Google Cloud TPU](https://cloud.google.com/tpu), sans
aucune dépendance au runtime JAX. Vos charges de travail PyTorch existantes
fonctionnent immédiatement, avec des modifications de code minimes (par exemple,
`.to("cuda")` → `.to("tpu")`).

## Quelle est la différence entre TorchAX et TorchTPU ? { #what-is-the-difference-between-torchax-and-torchtpu }

TorchAX (également appelé torch_xla2) est une implémentation qui intercepte les
opérations PyTorch au niveau Python et les abaisse (lowering) vers XLA (souvent
via JAX). TorchAX a beaucoup servi pour les benchmarks d'inférence et le
prototypage, mais TorchTPU est désormais la nouvelle norme officielle pour une
expérience PyTorch native et prête à l'emploi sur TPU.

## Que se passe-t-il avec PyTorch/XLA ? { #what-is-happening-to-pytorchxla }

PyTorch/XLA est obsolète et remplacé par
[TorchTPU](https://github.com/google-pytorch/torch_tpu), désormais public.

## TorchTPU remplace-t-il JAX ? { #is-torchtpu-replacing-jax }

Non. TorchTPU complète l'écosystème TPU en fournissant un point d'entrée PyTorch
natif vers les Cloud TPU. Les développeurs qui disposent d'une base de code
PyTorch existante peuvent l'exécuter directement sur TPU sans réécriture, ou
presque. Ils n'ont plus besoin de réécrire leurs modèles en JAX ni de maintenir
des bases de code distinctes pour les infrastructures GPU et TPU.

*   **Quand envisager TorchTPU ?** Si vous disposez d'une base de code
    PyTorch existante, si vous voulez une portabilité multimatérielle (GPU/TPU) sans
    refactorisation, si vous utilisez des frameworks PyTorch natifs (par
    exemple, TorchTitan, vLLM, SGLang, Hugging Face) ou si vous voulez migrer
    rapidement du GPU vers le TPU.

## Quel est le meilleur moyen d'échanger avec les développeurs de TorchTPU ? { #what-is-the-best-way-for-me-to-connect-with-torchtpu-developers }

Vous pouvez envoyer vos commentaires par e-mail à
[torchtpu-feedback@google.com](mailto:torchtpu-feedback@google.com). Les
contributions directes au dépôt sont les bienvenues :
[https://github.com/google-pytorch/torch_tpu](https://github.com/google-pytorch/torch_tpu).
Vous pouvez aussi utiliser le tag « TPU » sur
[discuss.pytorch.org](https://discuss.pytorch.org/).

## Y aura-t-il des forums d'échange avec la communauté ? { #will-there-be-any-community-engagement-forums }

Oui. L'engagement de la communauté est extrêmement important pour nous, et nous
communiquerons les détails de ces forums. Nous prévoyons également d'organiser
des permanences (office hours) mensuelles.

## Quelle prise en charge TorchTPU offre-t-il pour le développement de kernels ? { #what-support-does-torchtpu-provide-for-kernel-development }

Consultez [Prise en charge des kernels personnalisés dans TorchTPU](./custom_kernels.md).

## Quelles générations de TPU sont activement prises en charge ? Qu'est-il prévu pour le matériel futur ? { #which-tpu-generations-are-actively-supported-whats-the-plan-for-future-hardware-support }

TorchTPU prend activement en charge v5e, v5p, Trillium (TPU v6e) et Ironwood
(TPU v7x). La validation et l'optimisation pour le TPU v8 et les puces futures
sont en cours dans le cadre de notre feuille de route.

## Où trouver les fonctionnalités prises en charge et la feuille de route de TorchTPU ? { #where-can-we-find-torchtpus-supported-features-future-roadmap }

Consultez [Fonctionnalités prises en charge](./feature_set.md) pour connaître
l'ensemble actuel des fonctionnalités et celles prévues prochainement.

## Existe-t-il un agent pour faciliter ma prise en main ? { #is-there-an-agent-to-make-my-onboarding-easier }

Découvrez la suite d'agents TPU (expérimentale) :
[https://github.com/google-pytorch/tpu-agent-suite](https://github.com/google-pytorch/tpu-agent-suite)

La TPU Agent Suite est un plug-in open source qui fournit des sous-agents
spécialisés pour automatiser la prise en main, le diagnostic et l'optimisation
des performances d'inférence des modèles PyTorch sur Cloud TPU v6e et v7x.

## Puis-je utiliser TorchTPU avec notre solution de serving personnalisée ? { #can-i-use-torchtpu-with-our-custom-serving-solution }

Oui. TorchTPU offre une intégration prête à l'emploi avec des moteurs de serving
standards du secteur comme vLLM et SGLang. Les solutions de serving
personnalisées ou internes peuvent aussi s'intégrer à TorchTPU via les API
d'exécution PyTorch natives et des backends de compilateur de plus bas niveau.
Pour une orchestration avancée, les clients peuvent également déployer des
configurations de serving personnalisées sur GKE avec llm-d.
