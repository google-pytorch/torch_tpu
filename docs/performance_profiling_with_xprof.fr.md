# Profilage des performances avec XProf { #performance-profiling-with-xprof }

Savoir où le temps est passé est essentiel pour entraîner de grands modèles sur
TPU. Ce guide explique comment capturer des traces d'exécution avec le profileur
TPU et les analyser dans TensorBoard.

> **Warning:** **Prérequis :** TensorBoard nécessite `setuptools<81`. La
> dernière version supprime `pkg_resources`, dont TensorBoard dépend.
> Installez-le d'abord :
>
> ```shell
> pip install "setuptools<81"
> pip install xprof tensorboard
> ```

> **Note:** **Version de PyTorch :** l'intégration TPU native de
> `torch.profiler` nécessite PyTorch 2.12 ou une version ultérieure.

## 1. Le profileur TPU { #1-the-tpu-profiler }

Le `torch.profiler` standard (intégré à `torch_tpu`) fournit une interface de
**gestionnaire de contexte** pour tracer les activités du CPU et du TPU. Vous
encapsulez le code à profiler dans un bloc `torch.profiler.profile()`, et les
traces collectées sont enregistrées sur le disque pour être consultées dans
TensorBoard.

**Composants clés :**

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Composant                                  | Rôle                            |
| ------------------------------------------ | ------------------------------- |
| `torch.profiler.profile()`                 | Gestionnaire de contexte standard qui démarre et arrête <br> la collecte des traces |
| `torch.profiler.ProfilerActivity.CPU`      | Trace les opérations côté CPU (dispatch, chargement <br> des données) |
| `torch.profiler.ProfilerActivity.TPU`      | Trace l'exécution sur le matériel TPU (calcul, <br> mémoire, interconnexion) |
| `torch.profiler.tensorboard_trace_handler` | Callback qui enregistre les traces dans un répertoire pour <br> TensorBoard       |
| `TpuProfilerConfig`                        | Utilitaire pour la configuration personnalisée du profilage TPU |
<!-- mdformat on -->

```python
import torch
from torch.tpu.profiler import TpuProfilerConfig

device = torch.device("tpu")
```

## 2. Configuration personnalisée du profilage (`TpuProfilerConfig`) { #2-custom-profiling-configuration-tpuprofilerconfig }

Les options transmises via `TpuProfilerConfig` permettent de contrôler
finement les niveaux des traceurs de l'hôte et de l'appareil, ainsi que la
structure des répertoires de sortie.

### Options de configuration prises en charge { #sanctioned-configuration-options }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Clé d'option          | Type           | Valeur par défaut | Description               |
| :-------------------- | :------------- | :------ | :------------------------ |
| `host_tracer_level`   | `int`          | `2`     | Contrôle la verbosité du traceur de l'hôte CPU <br> (0=désactivé, 1=critique, 2=tout). |
| `device_tracer_level` | `int`          | `1`     | Contrôle la verbosité du traceur matériel <br> de l'appareil TPU (0=désactivé, 1=activé). |
| `python_tracer_level` | `int`          | `0`     | Contrôle le niveau de traçage des fonctions <br> Python (0=désactivé, 1=activé). |
| `run_dir`             | `pathlib.Path` | `None`  | Répertoire de sortie où sont enregistrés les fichiers <br> de trace du matériel TPU (`.xplane.pb`). |
| `worker_rank`         | `str`          | `None`  | Rang du worker ajouté comme suffixe au nom <br> du fichier XPlane (`<hostname>_<rank>.xplane.pb`). |
<!-- mdformat on -->

```python
import pathlib
import torch
from torch.tpu.profiler import TpuProfilerConfig

config = TpuProfilerConfig(
    host_tracer_level=2,
    device_tracer_level=1,
    python_tracer_level=0,
    run_dir=pathlib.Path("./profiler_output"),
    worker_rank="0",
)

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.TPU,
    ],
    experimental_config=config,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_output"),
) as prof:
    model(inputs)
```

## 3. Capturer un profil { #3-capturing-a-profile }

La cellule ci-dessous exécute une petite boucle d'entraînement dans le contexte
du profileur. Points clés :

1.  Encapsulez votre boucle d'entraînement dans `torch.profiler.profile()`
2.  Indiquez les activités : `CPU` pour les opérations côté hôte, `TPU` pour
    l'exécution sur le matériel
3.  Définissez `on_trace_ready` pour enregistrer les traces dans un répertoire
    avec `torch.profiler.tensorboard_trace_handler`
4.  (Facultatif) Transmettez `experimental_config` avec `TpuProfilerConfig`
    pour contrôler les niveaux des traceurs

> **Important:** Le profileur ajoute un surcoût. Ne profilez qu'un
> sous-ensemble représentatif d'étapes (par exemple, 100 à 1 000), et non tout
> votre entraînement.

```python
# Build a small model and optimizer
model = (
    torch.nn.Sequential(
        torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10)
    )
    .to(device)
    .to(torch.bfloat16)
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
data = torch.randn(32, 128, device=device, dtype=torch.bfloat16)

# Where to save the profiler traces
log_dir = "/tmp/profiler_output"

# Configure TPU-specific tracer levels
config = TpuProfilerConfig(
    host_tracer_level=2,
    device_tracer_level=1,
)

# Profile 100 training steps with native PyTorch profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.TPU,
    ],
    experimental_config=config,
    on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
) as prof:
  for step in range(100):
    optimizer.zero_grad()
    loss = model(data).sum()
    loss.backward()
    optimizer.step()
    torch.accelerator.synchronize()
    prof.step()

print(f"✅ Profile captured: {100} steps saved to {log_dir}")
```

## 4. Afficher les traces dans TensorBoard { #4-viewing-traces-in-tensorboard }

Exécutez les commandes suivantes :

**Étape 1 :** installez le plug-in du profileur (nécessaire pour que le tableau
de bord s'affiche) :

```shell
pip install tensorboard-plugin-profile "setuptools<81"
```

**Étape 2 :** lancez TensorBoard :

```shell
tensorboard --logdir=/tmp/profiler_output --port=6006
```

Vous pouvez aussi exécuter XProf comme serveur autonome.

```shell
xprof --logdir=/tmp/profiler_output --port=6006
```

**Étape 3 :** ouvrez `http://localhost:6006` dans votre navigateur et
sélectionnez l'onglet **« Profile »**.

> **Note:** Vous devez exécuter TensorBoard depuis le terminal intégré de
> VS Code pour que le transfert de port fonctionne automatiquement.

**Sur GKE :** écrivez le répertoire des traces dans un bucket Cloud Storage
monté avec le
[pilote CSI Cloud Storage FUSE](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver)
pour que les traces survivent au pod, puis ouvrez-les avec `xprof
--logdir=gs://<bucket>/profiler_output`.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Vue                | Contenu affiché                                        |
| ------------------ | ------------------------------------------------------ |
| **Overview**       | Résumé général : temps par étape, utilisation de l'appareil, principales opérations |
| **Trace Viewer**   | Chronologie de chaque opération sur le CPU et le TPU : repérez les trous (matériel inactif) |
| **Op Profile**     | Répartition du temps par opération : trouvez les opérations les plus coûteuses |
| **Memory Profile** | Utilisation de la HBM au fil du temps : détectez les pics de mémoire |
<!-- mdformat on -->

## 5. Points à surveiller { #5-what-to-look-for }

### Signaux d'alerte dans les traces { #red-flags-in-traces }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Motif                   | Signification           | Correction               |
| ----------------------- | ----------------------- | ------------------------ |
| Grands trous entre les opérations TPU | Goulot d'étranglement côté hôte <br> (chargement des données, surcoût Python) | Préchargez les données, réduisez la logique Python dans la boucle |
| Rafales TPU courtes et fréquentes   | Trop de ruptures de graphe → recompilations fréquentes | Réduisez les appels à `.item()` / `.cpu()`, utilisez `torch.compile` |
| Faible utilisation de la MXU | Dimensions des tensors non alignées sur la taille des tuiles (128) | Complétez les dimensions jusqu'à des multiples de 128 |
<!-- mdformat on -->

### Le profil idéal { #the-ideal-profile }

Une charge de travail TPU bien optimisée présente :

-   **De longs blocs de calcul TPU continus** : le compilateur a fusionné de
    nombreuses opérations en une seule exécution
-   **Des trous CPU→TPU minimaux** : le chargement des données chevauche le
    calcul
-   **Une utilisation élevée de la MXU (>60 %)** : les formes des tensors sont
    alignées sur le matériel

## 6. Vérifier que les traces ont été capturées { #6-verifying-the-traces-were-captured }

La cellule ci-dessous vérifie que le répertoire de sortie du profileur contient
des fichiers de trace.

```python
import os

trace_files = os.listdir(log_dir)
print(f"Trace directory: {log_dir}")
print(f"Files found: {len(trace_files)}")
for f in sorted(trace_files)[:10]:
  size_kb = os.path.getsize(os.path.join(log_dir, f)) / 1024
  print(f"  {f} ({size_kb:.1f} KB)")

if trace_files:
  print(f"\n✅ Traces ready. View with:")
  print(f"   tensorboard --logdir={log_dir}")
else:
  print("\n❌ No trace files found. Check profiler configuration.")
```

## 7. Profilage des kernels (compteurs de performances détaillés) { #7-kernel-profiling-fine-grained-performance-counters }

Le profilage des kernels permet de collecter des compteurs de performances
matériels détaillés (par exemple, pour les kernels Pallas) afin de diagnostiquer
des goulots d'étranglement de performances qui restent sinon invisibles dans
les traces d'événements standards.

> **Note:** L'échantillonnage des compteurs de performances détaillés est une
> fonctionnalité matérielle avancée disponible sur **Ironwood (TPU v7/TPU7x)**
> et les générations de TPU suivantes.

### Activer les compteurs de performances { #enabling-performance-counters }

Vous pouvez activer les compteurs de performances en transmettant la
configuration des compteurs via `experimental_options` dans
`TpuProfilerConfig` :

```python
from torch.tpu.profiler import TpuProfilerConfig

config = TpuProfilerConfig(
    experimental_options={
        "tpu_enable_periodic_counter_sampling": True,
        "tpu_tc_perf_counter_sampling_options": (
            "interval_us:1 scaling:0 counter_size_bits:1 indices:10"
            " indices:11 indices:56 indices:57 indices:58"
        ),
    }
)
```

Pour obtenir la liste détaillée des paramètres de configuration (comme les
intervalles d'échantillonnage, la mise à l'échelle et les modes de
déclenchement) et savoir comment déterminer les indices des compteurs,
consultez le
[guide officiel de profilage des kernels OpenXLA](https://openxla.org/xprof/kernel-profiling).

## 8. Aide-mémoire { #8-quick-reference }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
Étape       | Commande / code
----------- | --------------
**Importer** | `from torch.tpu.profiler import TpuProfilerConfig`
**Profiler** | `with torch.profiler.profile(activities=[...], on_trace_ready=...):`
**Enregistrer** | `torch.profiler.tensorboard_trace_handler("/tmp/profiler_output")`
**Afficher** | `tensorboard --logdir=/tmp/profiler_output`
<!-- mdformat on -->

> **Note:** Ne profilez que 100 à 1 000 étapes pour que les fichiers de trace
> restent faciles à gérer. Le profileur ajoute un surcoût d'environ 5 à 10 %
> par étape.
