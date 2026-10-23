# Référence de l'API TorchTPU { #torchtpu-api-reference }

Ce document est le guide de référence des API Python publiques et des variables
d'environnement de TorchTPU. Il vous aide à :

*   **Migrer de CUDA vers TPU** : faire correspondre les appels `torch.cuda`
    courants aux API
    [`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html)
    de PyTorch, indépendantes de l'appareil.
*   **Utiliser les fonctionnalités propres aux TPU** : configurer les modes
    d'exécution, la précision numérique de la MXU, la topologie distribuée, les
    opérateurs Pallas/TPU personnalisés, les optimiseurs traçables et le
    profilage XProf via le module **`torch.tpu`**.
*   **Configurer le runtime** : ajuster le cache de compilation, les
    vérifications de débogage et les paramètres de lancement multi-hôte via des
    variables d'environnement.

> **Note:** Pour la liste exhaustive et générée automatiquement de toutes les
> API Python publiques et variables d'environnement, qui fait référence,
> consultez
> [API publiques de TorchTPU (`public_apis.md`)](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/public_apis.md).

## **Requêtes de base sur les appareils** { #core-device-queries }

Lorsque vous importez `torch` avec TorchTPU installé, le backend enregistre
automatiquement l'appareil `"tpu"` et le module `torch.tpu`. Lorsque PyTorch
fournit une API
[`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html)
indépendante de l'appareil, utilisez-la plutôt qu'une API propre au backend :
votre code reste ainsi portable d'un accélérateur à l'autre. Utilisez
`torch.tpu` pour les propriétés matérielles propres aux TPU qui n'ont pas
d'équivalent générique.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API CUDA | API TPU / accélérateur | Description |
| :--- | :--- | :--- |
| `torch.device("cuda")` | `torch.accelerator.current_accelerator()` | Renvoie `torch.device("tpu")` lorsque le TPU est l'accélérateur actif. |
| `torch.cuda.is_available()` | `torch.accelerator.is_available()` | Vérifie si le backend TPU est disponible. |
| `torch.cuda.is_initialized()` | `torch.tpu.is_initialized()` | Vérifie si l'état du runtime TorchTPU a été initialisé. |
| `torch.cuda.device_count()` | `torch.accelerator.device_count()` | Nombre d'appareils visibles par le processus actuel. |
| `torch.cuda.current_device()` | `torch.accelerator.current_device_index()` | Renvoie l'index de l'appareil TPU actif. |
| `torch.cuda.set_device(rank)` | `torch.accelerator.set_device_index(rank)` | Définit l'index de l'appareil TPU actif pour le processus actuel. |
| `torch.cuda.device(rank)` | `torch.accelerator.device_index(rank)` | Gestionnaire de contexte qui définit temporairement l'index de l'appareil actif. |
| `torch.cuda.get_device_name(device=None)` | `torch.tpu.get_device_properties(device=None).name` | Renvoie le nom lisible du modèle de TPU connecté (par exemple, `"TPU v6e"`). |
| `torch.cuda.get_device_properties(device=None)` | `torch.tpu.get_device_properties(device=None)` | Renvoie les propriétés statiques de l'appareil TPU (notamment `name`, `devices_per_chip`, `hbm_bytes_per_chip`, `hbm_bytes_per_device`, `bf16_flops_per_chip`, `bf16_flops_per_device`, `sparse_cores_per_device` et `sparse_core_simd_width`). |
<!-- mdformat on -->

Les API `torch.accelerator` de ce tableau nécessitent PyTorch 2.11 ou version
ultérieure. Dans les rares cas où vous utilisez une version plus ancienne de
PyTorch, utilisez plutôt l'API `torch.tpu` correspondante (par exemple
`torch.tpu.device_count()` ou `torch.tpu.current_device()`). À partir de
PyTorch 2.11, ces API `torch.tpu` redondantes sont internes à TorchTPU, et les
appeler directement depuis du code utilisateur génère une erreur.

```python
import torch

# Initialize hardware
device = torch.device("tpu")

print(f"torch.accelerator.is_available(): {torch.accelerator.is_available()}")
print(f"torch.accelerator.device_count(): {torch.accelerator.device_count()}")
print(
    "torch.accelerator.current_device_index():"
    f" {torch.accelerator.current_device_index()}"
)
print(
    "torch.tpu.get_device_properties().name:"
    f" {torch.tpu.get_device_properties().name}"
)
```

## **Génération de nombres aléatoires (RNG)** { #random-number-generation-rng }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API CUDA | API TPU / accélérateur | Description |
| :--- | :--- | :--- |
| `torch.cuda.manual_seed(seed)` | `torch.tpu.manual_seed(seed)` | Initialise la graine du générateur de nombres aléatoires de l'appareil TPU actif. |
| `torch.cuda.manual_seed_all(seed)` | `torch.tpu.manual_seed_all(seed)` | Initialise la graine du RNG sur tous les appareils TPU locaux. |
| `torch.cuda.initial_seed()` | `torch.accelerator.random.initial_seed()` | Renvoie la graine aléatoire initiale de l'appareil TPU actuel. |
| `torch.cuda.get_rng_state()` | `torch.accelerator.random.get_rng_state()` | Renvoie le tensor d'état du RNG pour l'appareil actuel. |
| `torch.cuda.set_rng_state(new_state, device=...)` | `torch.tpu.set_rng_state(new_state, device=...)` | Définit le tensor d'état du RNG pour l'appareil TPU actif du processus actuel. |
| `torch.cuda.default_generators` | `torch.tpu.default_generators` | Tuple, initialisé à la demande, d'instances `torch.Generator` par défaut, indexé par numéro d'appareil local. |
<!-- mdformat on -->

Les API `torch.accelerator.random` de ce tableau nécessitent PyTorch 2.13 ou
version ultérieure. Dans les rares cas où vous utilisez une version plus
ancienne de PyTorch, utilisez plutôt `torch.tpu.initial_seed()` ou
`torch.tpu.get_rng_state()`. À partir de PyTorch 2.13, ces API `torch.tpu` sont
internes à TorchTPU, et les appeler depuis votre code génère une erreur.

```python
# Set the seed across all cores
torch.tpu.manual_seed_all(42)
print("Seeded all TPU cores with value 42.")
```

## **Précision mixte automatique (AMP)** { #automatic-mixed-precision-amp }

TorchTPU s'intègre au framework standard
[`torch.amp.autocast`](https://docs.pytorch.org/docs/stable/amp.html) de
PyTorch avec la chaîne de type d'appareil `"tpu"`.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API CUDA | API TPU | Description |
| :--- | :--- | :--- |
| `torch.amp.autocast("cuda", ...)` | `torch.amp.autocast("tpu", ...)` | Gestionnaire de contexte / décorateur qui active la précision mixte automatique (`bfloat16` par défaut sur TPU). |
| `torch.cuda.get_amp_supported_dtype()` | `torch.tpu.get_amp_supported_dtype()` | Liste les dtypes pris en charge sur TPU (`bfloat16`, `float16`, `float32`, `float64`, `float8_*`, ainsi que les types entiers, booléens et complexes). |
<!-- mdformat on -->

```python
supported_dtypes = torch.tpu.get_amp_supported_dtype()
print(f"AMP supported dtypes: {supported_dtypes}")
```

## **Streams et événements** { #streams-and-events }

Pour la gestion des streams et la synchronisation des appareils, utilisez les
API standards
[`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html)
plutôt que des API propres au backend. N'utilisez `torch.tpu` que pour créer
directement des objets TPU `Stream` ou `Event`, ou pour configurer l'épinglage
(pinning) des cœurs TPU.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API CUDA | API TPU / accélérateur | Description |
| :--- | :--- | :--- |
| `torch.cuda.current_stream()` | `torch.accelerator.current_stream()` | Renvoie le stream actif d'un appareil. |
| `torch.cuda.set_stream(stream)` | `torch.accelerator.set_stream(stream)` | Définit le stream actif. |
| `torch.cuda.default_stream(device=None)` | `torch.tpu.default_stream(device=None)` | Renvoie le `TpuStream` par défaut de l'appareil. |
| `torch.cuda.stream(stream)` | `torch.tpu.stream(stream)` | Gestionnaire de contexte qui sélectionne un `TpuStream` donné. |
| `torch.cuda.synchronize()` | `torch.accelerator.synchronize()` | Attend la fin de tout le travail en attente sur l'appareil TPU. |
| `torch.cuda.Stream` | `torch.tpu.Stream(device=None, priority=0, *, pinned=None)` | Sous-classe de `torch.Stream`. Prend en charge `synchronize()`, `record_event()`, `wait_event()`, `wait_stream()` et l'épinglage des cœurs (`pinned="dense"` ou `"sparse"`, interrogeable via `is_dense` / `is_sparse`). |
| `torch.cuda.Event` | `torch.tpu.Event(enable_timing=False)` | Sous-classe de `torch.Event`. Prend en charge `record()`, `synchronize()`, `query()` et `elapsed_time(end_event)`. |
<!-- mdformat on -->

Les API `torch.accelerator` de ce tableau nécessitent PyTorch 2.11 ou version
ultérieure. Dans les rares cas où vous utilisez une version plus ancienne de
PyTorch, utilisez plutôt `torch.tpu.current_stream()`,
`torch.tpu.set_stream()` ou `torch.tpu.synchronize()`.

> **Note:** Les streams sont un concept CUDA sans équivalent direct sur TPU.
> Dans le modèle d'exécution différée de TorchTPU, le compilateur XLA et le
> runtime PjRt planifient et chevauchent automatiquement les opérations en
> fonction des dépendances de données. Vous n'avez donc pas besoin de gérer
> manuellement plusieurs streams.

## **API propres aux TPU** { #tpu-specific-apis }

Au-delà des équivalents CUDA standards, `torch.tpu` expose des API conçues
spécifiquement pour les modes d'exécution TPU, la précision de la MXU, la
compilation XLA, la topologie distribuée, les kernels Pallas personnalisés, les
optimiseurs traçables et le profilage.

### **Modes d'exécution et options du compilateur** { #execution-modes-and-compiler-options }

Contrôlez la façon dont XLA met en file d'attente, fusionne et compile les
opérations eager. Pour une présentation de l'architecture, consultez
[Modes d'exécution de TorchTPU](./torchtpu_execution_model.md).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Description |
| :--- | :--- |
| `torch.tpu.EagerMode` | Enum qui contrôle l'exécution eager : `DEFER_AND_FUSE`, `DEFER_NEVER` et `DEFER_NEVER_AND_LAUNCH_BLOCKING`. |
| `torch.tpu.eager_mode` | Lit ou définit globalement l'`EagerMode` actif. |
| `torch.tpu.set_eager_mode(mode)` | Gestionnaire de contexte qui définit temporairement l'`EagerMode` actif. |
| `torch.tpu.compiler.custom_compiler_options(options)` | Gestionnaire de contexte qui transmet des flags personnalisés du compilateur XLA aux opérations eager matérialisées dans le bloc. |
<!-- mdformat on -->

```python
print(f"Default eager_mode: {torch.tpu.eager_mode}")
with torch.tpu.set_eager_mode(torch.tpu.EagerMode.DEFER_AND_FUSE):
  x = torch.ones((4, 4), device="tpu") + 1
  print(
      f"Inside context manager: {torch.tpu.eager_mode}, sum={x.sum().item()}"
  )
```

### **Précision numérique** { #numerical-precision }

Contrôlez la précision de l'unité de multiplication matricielle (MXU) et
l'accumulation intermédiaire en `float32`. Pour des conseils détaillés,
consultez [Précision numérique](./numerical_precision.md).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Description |
| :--- | :--- |
| `torch.tpu.Precision` | Enum qui contrôle la précision de la MXU pour les multiplications matricielles et les convolutions en `float32` : `Precision.DEFAULT`, `Precision.HIGH` et `Precision.HIGHEST`. |
| `torch.tpu.precision(level)` | Gestionnaire de contexte qui définit le niveau de `Precision` des multiplications matricielles et des convolutions dans le bloc. |
| `torch.tpu.allow_excess_precision` | Lit ou définit globalement le flag XLA `--xla_allow_excess_precision`, qui conserve en `float32` les valeurs intermédiaires fusionnées. |
<!-- mdformat on -->

```python
a = torch.randn(64, 64, device="tpu", dtype=torch.float32)
b = torch.randn(64, 64, device="tpu", dtype=torch.float32)
with torch.tpu.precision(torch.tpu.Precision.HIGH):
  c = torch.matmul(a, b)
print(f"Computed HIGH-precision matmul with output shape: {tuple(c.shape)}")
```

### **Entraînement distribué et topologie** { #distributed-training-and-topology }

TorchTPU enregistre `"tpu_dist"` (également disponible sous l'alias `"tpu"`)
comme backend `torch.distributed` par défaut pour les appareils `"tpu"`.
Consultez [Premiers pas avec TorchTPU sur GKE](./get_started_on_gke.md) et
[Bonnes pratiques FSDP](./case_studies_fsdp.md).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Description |
| :--- | :--- |
| `torch.tpu.topology_aware_mesh(mesh_shape, ...)` | Calcule un tensor de disposition des rangs, adapté à la topologie, de forme `mesh_shape` pour `DeviceMesh("tpu", mesh=...)`. |
| `torch.tpu.distributed.set_tpu_launch_env(...)` | Renseigne les variables d'environnement de topologie TPU et de lancement multi-hôte avant de démarrer les workers. |
| `torch.tpu.distributed.spmd_safe(func)` | Décorateur qui déclare que tous les rangs exécutent des opérations TPU identiques dans `func`, ce qui active la fusion des opérations collectives en mode eager. |
| `torch.tpu.window_num_stripes` | Lit ou définit le nombre maximal de bandes (stripes) DMA simultanées pour les transferts pair à pair. |
| `torch.tpu.window_stripe_chunk_mb` | Lit ou définit la taille minimale de charge utile (en Mo) par bande DMA pour les transferts pair à pair. |
<!-- mdformat on -->

### **Opérateurs TPU personnalisés (`torch.tpu.ops`) et Pallas (`torch.tpu.pallas`)** { #tpu-custom-operators-torchtpuops-and-pallas-torchtpupallas }

TorchTPU expose des opérateurs spécialisés pour le matériel sous `torch.tpu.ops`
et un wrapper de kernels JAX/Pallas personnalisés sous `torch.tpu.pallas`.
Consultez [Ajouter des kernels personnalisés avec Pallas](./custom_kernels.md).

#### **Opérateurs généraux, d'attention, MoE et collectifs** { #general-attention-moe-and-collective-operators }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Description |
| :--- | :--- |
| `torch.tpu.ops.max_pool{1,2,3}d(...)` | Applique un max pooling 1D, 2D ou 3D sur TPU sans matérialiser de tensor d'index. |
| `torch.tpu.ops.max_pool{1,2,3}d_backward(...)` | Calcule la passe arrière de `max_pool{1,2,3}d` sans tensor d'index. |
| `torch.tpu.ops.optimization_barrier(inputs)` | Empêche XLA de déplacer, fusionner ou dédupliquer des calculs de part et d'autre de la barrière. |
| `torch.tpu.ops.splash_sdpa(q, k, v, ...)` | Attention par produit scalaire mis à l'échelle (scaled dot-product attention) reposant sur le kernel Pallas Splash Attention pour TPU. |
| `torch.tpu.ops.ragged_dot(lhs, rhs, group_sizes, ...)` | Multiplication matricielle groupée pour les charges de travail Mixture-of-Experts (MoE). |
| `torch.tpu.ops.ragged_dot_weight_grad(lhs, grad_output, group_sizes)` | Calcule le gradient des poids pour `ragged_dot`. |
| `torch.tpu.ops.align_jagged_to_multiple(nt, ...)` | Complète (padding) un buffer `NestedTensor` `torch.jagged` jusqu'à un multiple donné pour l'alignement TPU. |
| `torch.tpu.ops.strip_jagged_padding(nt, ...)` | Supprime le padding final ajouté par `align_jagged_to_multiple`. |
| `torch.tpu.ops.all_gather_dim(input, dim, group=None)` | Effectue un all-gather de `input` sur `group` et concatène le long de la dimension `dim`. |
| `torch.tpu.ops.reduce_scatter_dim(input, dim, group=None, ...)` | Réduit `input` sur `group` et répartit les morceaux le long de la dimension `dim`. |
| `torch.tpu.ops.ragged_all_to_all(...)` | Échange des portions de tensors de longueur variable entre les rangs d'un groupe de processus. |
| `torch.tpu.ops.send(tensors, dst, tag)` / `torch.tpu.ops.recv(tensors, src, tag)` | Envoi et réception point à point de tensors sur le groupe de processus TPU par défaut. |
| `torch.tpu.pallas.jax_op(name, fn=None, ...)` | Enregistre une fonction JAX/Pallas comme opérateur PyTorch personnalisé traçable. |
<!-- mdformat on -->

#### **Opérateurs d'embedding SparseCore** { #sparsecore-embedding-operators }

Ces opérateurs nécessitent un TPU doté de SparseCores. La propriété
`sparse_cores_per_device` de `torch.tpu.get_device_properties()` indique le
nombre de SparseCores par appareil (0 sur les TPU sans SparseCore), et
`sparse_core_simd_width` indique leur largeur SIMD.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Description |
| :--- | :--- |
| `torch.tpu.ops.sparse_dense_matmul(...)` | Effectue la recherche et l'agrégation d'embeddings CSR sur le SparseCore du TPU. |
| `torch.tpu.ops.sparse_dense_matmul_grad_with_sgd(...)` / `_with_adagrad(...)` / `_with_adam(...)` | Calcule les gradients de la passe arrière de `sparse_dense_matmul` et applique sur le SparseCore une mise à jour fusionnée et sur place de la table par SGD, Adagrad ou Adam. |
| `torch.tpu.ops.sparse_dense_matmul_activation_unstack(...)` | Sépare les activations d'embedding SparseCore empilées en tensors par caractéristique. |
| `torch.tpu.ops.sparse_dense_matmul_gradient_stack(...)` | Empile les gradients d'activation par caractéristique en un seul tensor pour le SparseCore. |
| `torch.tpu.ops.sparse_gather(...)` | Rassemble des lignes d'un tensor 2D à l'aide d'index CSR sur le SparseCore du TPU. |
| `torch.tpu.ops.sparse_gather_backward(...)` | Effectue un scatter-add des gradients de `sparse_gather` sur le SparseCore du TPU. |
| `torch.tpu.ops.sparse_iota(...)` | Génère, sur le SparseCore, les index de position dans chaque ligne des éléments non nuls d'une matrice CSR. |
| `torch.tpu.ops.sparse_map_row(...)` | Applique, sur le SparseCore, un calcul MLIR personnalisé ligne par ligne à des matrices CSR creuses. |
<!-- mdformat on -->

### **Étape d'entraînement fusionnée et optimiseurs (`torch.tpu.optimizers`)** { #fused-training-step-and-optimizers-torchtpuoptimizers }

`torch.tpu.optimizers` trace la passe avant, le calcul de la perte, la passe
arrière et l'étape fonctionnelle de l'optimiseur dans un seul graphe TPU
compilé.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Description |
| :--- | :--- |
| `torch.tpu.optimizers.SingleTraceTrainer(model, optimizer=None, ...)` | Gère les paramètres fonctionnels (`params`) et l'état de l'optimiseur (`param_group`), et compile l'étape d'entraînement complète via `make_compiled_train_step(...)`. |
| `torch.tpu.optimizers.Optimizer(lr=..., weight_decay=...)` | Classe de base des optimiseurs fonctionnels sans état (expose `defaults`, `init_param_group` et `step`). |
| `torch.tpu.optimizers.AdamW(...)` | Optimiseur AdamW fonctionnel et traçable (`AdamW.ParamGroup`). |
| `torch.tpu.optimizers.Muon(...)` | Optimiseur Muon fonctionnel et traçable (`Muon.ParamGroup`). |
| `torch.tpu.optimizers.Sgd(...)` | Optimiseur SGD fonctionnel et traçable (`Sgd.ParamGroup`). |
<!-- mdformat on -->

### **Profilage (`torch.tpu.profiler`)** { #profiling-torchtpuprofiler }

Configurez le traçage matériel TPU et Python pour XProf et TensorBoard.
Consultez [XProf pour TorchTPU](./performance_profiling_with_xprof.md).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Description |
| :--- | :--- |
| `torch.tpu.profiler.TpuProfilerConfig(...)` | Configuration expérimentale transmise à `torch.profiler.profile(..., experimental_config=...)` pour configurer le traçage de l'hôte, de l'appareil TPU et de Python pour XProf. |
<!-- mdformat on -->

## **Variables d'environnement** { #environment-variables }

Le comportement de TorchTPU est contrôlé par plusieurs variables
d'environnement qui configurent le compilateur XLA, le pilote TPU (`libtpu`) et
le sous-système de débogage de PyTorch.

> **CRITIQUE : initialisation globale au processus** Les variables
> d'environnement de TorchTPU sont **verrouillées** dès l'initialisation du
> backend (généralement lors de `import torch` ou de `torch.device("tpu")`).

### **Visibilité des appareils, runtime et lancement distribué** { #device-visibility-runtime-and-distributed-launch }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Variable d'environnement | Description |
| :--- | :--- |
| `TPU_VISIBLE_DEVICES` | Index des puces TPU visibles par le processus actuel, séparés par des virgules (équivalent de `CUDA_VISIBLE_DEVICES`). |
| `RANK` | Rang global du processus actuel dans un job distribué. |
| `LOCAL_RANK` | Rang local à l'hôte du processus actuel. |
| `WORLD_SIZE` | Nombre total de processus dans le job distribué. |
| `NNODES` | Nombre total de nœuds hôtes dans un job multi-hôte. |
| `NODE_RANK` | Rang du nœud hôte actuel dans un job multi-hôte. |
| `MASTER_ADDR` | Adresse du nœud coordinateur de rang 0 pour `torch.distributed`. |
| `MASTER_PORT` | Port de `MASTER_ADDR` utilisé pour le rendez-vous `torch.distributed`. |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | Définissez-la sur `"0"` pour désactiver l'enregistrement automatique du backend `torch_tpu`. |
| `TORCH_TPU_ALLOW_LEGACY_DEVICE_APIS` | Lorsqu'elle est activée, permet d'appeler les API d'appareil internes de `torch.tpu` (celles qui ont un équivalent `torch.accelerator`) en dehors du cœur de PyTorch, avec un avertissement unique au lieu de lever une `RuntimeError`. Utilisez-la uniquement comme solution temporaire : il est préférable de migrer vers `torch.accelerator.*`, et cette variable sera probablement supprimée dans une prochaine version. |
| `LIBTPU_INIT_ARGS` | Flags d'initialisation séparés par des espaces, transmis à `libtpu` et au runtime PJRT. |
| `TPU_PREMAPPED_BUFFER_SIZE` | Taille en octets du buffer mémoire hôte-vers-appareil prémappé pour le client PJRT. |
| `TMPDIR` | Répertoire temporaire utilisé par le runtime et le compilateur. |
<!-- mdformat on -->

### **Topologie matérielle et coordination multi-hôte** { #hardware-topology-and-multi-host-coordination }

Ces variables spécifient la topologie matérielle des TPU et la mise en réseau
des processus. Dans la plupart des cas, elles sont définies automatiquement par
`torch.tpu.distributed.set_tpu_launch_env()`, le lanceur TorchTPU ou
l'environnement Google Cloud.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Variable d'environnement | Description |
| :--- | :--- |
| `TORCH_TPU_TOPOLOGY` | Topologie physique de la tranche (slice) TPU (par exemple, `2x2x1`, `2x2x4`). |
| `TORCH_TPU_SLICEBUILDER_ADDRESSES` | Adresses `host:port` des workers pour le service de construction de tranches TPU (slice builder), séparées par des virgules. |
| `TORCH_TPU_DEVICES_PER_HOST` | Nombre de puces TPU par hôte, utilisé pour déduire les adresses du slice builder. |
| `TORCH_TPU_BASE_PORT` | Numéro de port de départ utilisé pour déduire les adresses du slice builder. |
| `TORCH_TPU_SPMD_HANDSHAKE_STAGE` | Étape à laquelle les rangs vérifient que les graphes d'opérations collectives concordent entre les rangs. |
| `TORCHCOMM_RANK` | Rang du processus pour `torchcomm` (prioritaire sur `RANK`). |
| `TORCHCOMM_SIZE` | Nombre total de processus (world size) pour `torchcomm` (prioritaire sur `WORLD_SIZE`). |
<!-- mdformat on -->

### **Mode d'exécution, compilation et mise en cache** { #execution-mode-compilation-and-caching }

Ces variables contrôlent la façon dont les opérations et les graphes PyTorch
sont envoyés (dispatch), abaissés (lowering), optimisés par XLA et mis en
cache entre les exécutions et les rangs.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Variable d'environnement | Description |
| :--- | :--- |
| `TORCH_TPU_DEFER_AND_FUSE` | Lorsqu'elle est activée (`"1"`/`"true"`), définit le mode eager par défaut sur `EagerMode.DEFER_AND_FUSE`. |
| `TORCH_TPU_LAUNCH_BLOCKING` | Lorsqu'elle est activée (`"1"`/`"true"`), définit le mode eager par défaut sur `EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING`. |
| `TORCH_TPU_TIER2_COMPILATION_CACHE` | Nom du répertoire, sous `/dev/shm/torch_tpu_cache/`, du cache de compilation local à l'hôte (ou `"disabled"`). |
| `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` | Chemin (par exemple, `gs://...` ou chemin local) du cache de compilation persistant de niveau 3, partagé entre les hôtes et les exécutions. |
| `XLA_FLAGS` | Flags du compilateur XLA séparés par des espaces (par exemple, `--xla_dump_hlo_as_text --xla_dump_to=...`). |
<!-- mdformat on -->

### **Débogage, sécurité, profilage et traçage** { #debugging-safety-profiling-and-tracing }

Ces variables contrôlent le contexte de débogage, les vérifications de sécurité
à l'exécution et la sortie du profileur pour TorchTPU.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Variable d'environnement | Description |
| :--- | :--- |
| `TORCH_SHOW_CPP_STACKTRACES` | Lorsqu'elle est définie sur `"1"`, enrichit les exceptions Python `RuntimeError` avec les backtraces C++. |
| `TORCH_TPU_ENABLE_DEBUG_CHECKS` | Lorsqu'elle est définie sur `"1"`, active les vérifications de débogage et de limites à l'exécution sur TPU. |
| `TORCH_TPU_PROFILER_OUTPUT_DIR` | Répertoire de sortie de secours pour les fichiers de trace XProf lorsque `TpuProfilerConfig.run_dir` n'est pas défini. |
| `TORCH_TPU_XPROF_SESSION_ID` | ID de session de profilage XProf partagé entre les workers multi-hôtes. |
| `TORCH_TRACE` | Répertoire de sortie des journaux de trace structurés du compilateur et du dispatcher (`tlparse`). |
<!-- mdformat on -->

### **Exemples de configuration des variables d'environnement** { #environment-variable-configuration-snippets }

#### **1. Stacktraces C++ (`TORCH_SHOW_CPP_STACKTRACES`)** { #1-c-stacktraces-torch_show_cpp_stacktraces }

Définir `TORCH_SHOW_CPP_STACKTRACES=1` avant d'importer `torch` facilite
grandement le diagnostic des erreurs qui surviennent au plus profond du backend
C++ ou de XLA. Au lieu d'une simple `RuntimeError` Python, vous obtenez un `C++
CapturedTraceback` détaillé qui montre les frames d'exécution exactes.

```python
import os
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
import torch

device = torch.device("tpu")
a = torch.randn(10, device=device)
b = torch.randn(20, device=device)
# Raises RuntimeError enriched with C++ Error Trace and Captured Traceback:
c = a + b
```

#### **2. Vérifications de débogage internes et validation des limites (`TORCH_TPU_ENABLE_DEBUG_CHECKS`)** { #2-internal-debug-checks-boundary-validation-torch_tpu_enable_debug_checks }

Par défaut, le TPU ignore les vérifications de limites pour gagner en vitesse.
L'activation de `TORCH_TPU_ENABLE_DEBUG_CHECKS=1` détecte des bugs subtils liés
à des index hors limites (par exemple dans `torch.take`), mais ralentit
l'exécution en raison de la synchronisation CPU/TPU supplémentaire. Les
opérations qui échoueraient sinon silencieusement ou corrompraient les données
lèvent une exception explicite comme `take(): expected indices to be
in range` :

```python
import os
os.environ["TORCH_TPU_ENABLE_DEBUG_CHECKS"] = "1"
import torch

device = torch.device("tpu")
x = torch.tensor([1.0, 2.0, 3.0], device=device)
indices = torch.tensor([5], device=device)  # Out of bounds
# Raises an explicit out-of-bounds exception when materialized:
y = torch.take(x, indices).cpu()
```

#### **3. Vidage de l'IR XLA / HLO (`XLA_FLAGS`)** { #3-dumping-xla-ir-hlo-xla_flags }

En définissant `XLA_FLAGS`, vous pouvez forcer le compilateur à vider (dump) le
texte HLO (High-Level Optimizer) dans un répertoire. Le répertoire de dump (par
exemple, `/tmp/hlo_dump`) contient :

*   **`*.after_optimizations.txt`** : le graphe HLO final, qui montre les
    opérations au niveau matériel.
*   **`*.flagfile`** : la liste complète des flags actifs du compilateur XLA
    (pour vérifier `xla_optimization_level`).
*   **`*tpu_comp_env.txt`** : les paramètres internes de l'environnement TPU
    utilisés pendant la compilation.

```python
import os

dump_dir = "/tmp/hlo_dump"
os.makedirs(dump_dir, exist_ok=True)
os.environ["XLA_FLAGS"] = f"--xla_dump_hlo_as_text --xla_dump_to={dump_dir}"

import torch
device = torch.device("tpu")
x = torch.randn(10, device=device)
y = (x + 1).to("cpu")
```

#### **4. Mise en cache de la compilation à plusieurs niveaux (`TORCH_TPU_TIER2_COMPILATION_CACHE` et `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`)** { #4-multi-tiered-compilation-caching-torch_tpu_tier2_compilation_cache-torch_tpu_tier3_compilation_cache_root }

TorchTPU utilise un cache à plusieurs niveaux pour éviter les compilations
redondantes entre les exécutions et les rangs :

*   **Niveau 2 (`TORCH_TPU_TIER2_COMPILATION_CACHE`)** : stocké localement dans
    la mémoire partagée de l'hôte. Il attend un **nom** de répertoire (par
    exemple, `"tpu_tier2_cache"`), que le backend place automatiquement dans
    `/dev/shm/torch_tpu_cache/<name>`.
*   **Niveau 3 (`TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`)** : cache de
    compilation persistant, partagé entre les hôtes et les sessions
    (généralement un URI de bucket `gs://` ou un chemin de système de fichiers
    partagé).

Les artefacts binaires compilés (exécutables `*.bin`) sont enregistrés de façon
asynchrone dans les répertoires de cache et réutilisés lors des exécutions
suivantes, ce qui réduit considérablement la latence de compilation JIT :

```python
import os

tier2_name = "tpu_tier2_cache"
tier2_path = os.path.join("/dev/shm/torch_tpu_cache", tier2_name)
tier3_path = "/tmp/tpu_tier3_cache"  # Or "gs://<bucket_name>/<path_to_cache>"
os.makedirs(tier2_path, exist_ok=True)
os.makedirs(tier3_path, exist_ok=True)

# TORCH_TPU_TIER2_COMPILATION_CACHE expects just the directory NAME inside /dev/shm/torch_tpu_cache/
os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"] = tier2_name
os.environ["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"] = tier3_path

import torch
device = torch.device("tpu")
a = torch.randn(1024, 1024, device=device)
_ = (a @ a).to("cpu")
```
