# Profilazione delle prestazioni con XProf { #performance-profiling-with-xprof }

Capire dove viene speso il tempo è fondamentale per addestrare modelli di
grandi dimensioni su TPU. Questa guida spiega come acquisire tracce di
esecuzione con il TPU Profiler e analizzarle in TensorBoard.

> **Warning:** **Prerequisito:** TensorBoard richiede `setuptools<81`.
> L'ultima versione rimuove `pkg_resources`, da cui TensorBoard dipende.
> Installalo per primo:
>
> ```shell
> pip install "setuptools<81"
> pip install xprof tensorboard
> ```

> **Note:** **Versione di PyTorch:** l'integrazione TPU nativa di
> `torch.profiler` richiede PyTorch 2.12 o versioni successive.

## 1. Il profiler TPU { #1-the-tpu-profiler }

Il `torch.profiler` standard (integrato con `torch_tpu`) offre un'interfaccia
di **context manager** per tracciare le attività sia della CPU sia della TPU.
Racchiudi il codice da profilare in un blocco `torch.profiler.profile()` e le
tracce raccolte vengono salvate su disco per essere visualizzate in
TensorBoard.

**Componenti principali:**

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Componente                                 | Ruolo                           |
| ------------------------------------------ | ------------------------------- |
| `torch.profiler.profile()`                 | Context manager standard che avvia/arresta <br> la raccolta delle tracce |
| `torch.profiler.ProfilerActivity.CPU`      | Traccia le operazioni lato CPU (dispatch, <br> caricamento dei dati) |
| `torch.profiler.ProfilerActivity.TPU`      | Traccia l'esecuzione sull'hardware TPU (calcolo, <br> memoria, interconnessione) |
| `torch.profiler.tensorboard_trace_handler` | Callback che salva le tracce in una directory per <br> TensorBoard       |
| `TpuProfilerConfig`                        | Helper per la configurazione personalizzata della profilazione TPU |
<!-- mdformat on -->

```python
import torch
from torch.tpu.profiler import TpuProfilerConfig

device = torch.device("tpu")
```

## 2. Configurazione personalizzata della profilazione (`TpuProfilerConfig`) { #2-custom-profiling-configuration-tpuprofilerconfig }

Le opzioni passate tramite `TpuProfilerConfig` consentono un controllo
granulare dei livelli dei tracer dell'host e del dispositivo, oltre che della
struttura delle directory di output.

### Opzioni di configurazione supportate { #sanctioned-configuration-options }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Chiave dell'opzione   | Tipo           | Predefinito | Descrizione           |
| :-------------------- | :------------- | :------ | :------------------------ |
| `host_tracer_level`   | `int`          | `2`     | Controlla il livello di dettaglio del tracer <br> dell'host CPU (0=disattivato, 1=critico, 2=tutto). |
| `device_tracer_level` | `int`          | `1`     | Controlla il livello di dettaglio del tracer <br> del dispositivo hardware TPU (0=disattivato, 1=attivo). |
| `python_tracer_level` | `int`          | `0`     | Controlla il livello di tracciamento delle <br> funzioni Python (0=disattivato, 1=attivo). |
| `run_dir`             | `pathlib.Path` | `None`  | Directory di output in cui vengono salvati i file <br> delle tracce hardware TPU (`.xplane.pb`). |
| `worker_rank`         | `str`          | `None`  | Rank del worker aggiunto come suffisso al <br> nome del file XPlane (`<hostname>_<rank>.xplane.pb`). |
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

## 3. Acquisire un profilo { #3-capturing-a-profile }

La cella seguente esegue un piccolo loop di addestramento all'interno del
contesto del profiler. I punti chiave:

1.  Racchiudi il loop di addestramento in `torch.profiler.profile()`
2.  Specifica le attività: `CPU` per le operazioni lato host, `TPU` per
    l'esecuzione sull'hardware
3.  Imposta `on_trace_ready` per salvare le tracce in una directory con
    `torch.profiler.tensorboard_trace_handler`
4.  (Facoltativo) Passa `experimental_config` con `TpuProfilerConfig` per
    controllare i livelli dei tracer

> **Important:** il profiler aggiunge overhead. Profila solo un sottoinsieme
> rappresentativo di step (ad esempio 100–1000), non l'intera esecuzione
> dell'addestramento.

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

## 4. Visualizzare le tracce in TensorBoard { #4-viewing-traces-in-tensorboard }

Esegui i comandi seguenti:

**Passaggio 1:** installa il plug-in del profiler (necessario perché venga
visualizzata la dashboard):

```shell
pip install tensorboard-plugin-profile "setuptools<81"
```

**Passaggio 2:** avvia TensorBoard:

```shell
tensorboard --logdir=/tmp/profiler_output --port=6006
```

Puoi anche eseguire XProf come server autonomo.

```shell
xprof --logdir=/tmp/profiler_output --port=6006
```

**Passaggio 3:** apri `http://localhost:6006` nel browser e seleziona la
scheda **"Profile"**.

> **Note:** devi eseguire TensorBoard dal terminale integrato di VS Code,
> in modo che il port forwarding funzioni automaticamente.

**Su GKE:** scrivi la directory delle tracce in un bucket Cloud Storage
montato con il
[driver CSI di Cloud Storage FUSE](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver),
in modo che le tracce sopravvivano al pod, poi aprile con `xprof
--logdir=gs://<bucket>/profiler_output`.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Vista              | Cosa mostra                                            |
| ------------------ | ------------------------------------------------------ |
| **Overview**       | Riepilogo di alto livello: tempo per step, utilizzo del dispositivo, operazioni principali |
| **Trace Viewer**   | Timeline di ogni operazione su CPU e TPU: cerca i vuoti (hardware inattivo) |
| **Op Profile**     | Suddivisione del tempo per operazione: individua le operazioni più costose |
| **Memory Profile** | Utilizzo della HBM nel tempo: rileva i picchi di memoria |
<!-- mdformat on -->

## 5. Cosa cercare { #5-what-to-look-for }

### Segnali di allarme nelle tracce { #red-flags-in-traces }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Pattern                 | Significato             | Soluzione                |
| ----------------------- | ----------------------- | ------------------------ |
| Ampi vuoti tra le operazioni TPU | Collo di bottiglia lato host <br> (caricamento dei dati, overhead di Python) | Esegui il prefetch dei dati, riduci la logica Python nel loop |
| Brevi raffiche frequenti sulla TPU   | Troppi graph break → ricompilazioni frequenti | Riduci le chiamate `.item()` / `.cpu()`, usa `torch.compile` |
| Basso utilizzo delle MXU | Dimensioni dei tensori non allineate alla dimensione dei tile (128) | Aggiungi padding alle dimensioni fino a multipli di 128 |
<!-- mdformat on -->

### Il profilo ideale { #the-ideal-profile }

Un carico di lavoro TPU ben ottimizzato mostra:

-   **Blocchi di calcolo TPU lunghi e continui**: il compilatore ha fuso molte
    operazioni in un'unica esecuzione
-   **Vuoti CPU→TPU minimi**: il caricamento dei dati è sovrapposto al calcolo
-   **Elevato utilizzo delle MXU (>60%)**: le forme dei tensori sono allineate
    all'hardware

## 6. Verificare che le tracce siano state acquisite { #6-verifying-the-traces-were-captured }

La cella seguente verifica che la directory di output del profiler contenga
file di traccia.

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

## 7. Profilazione dei kernel (contatori di prestazioni granulari) { #7-kernel-profiling-fine-grained-performance-counters }

La profilazione dei kernel consente di raccogliere contatori di prestazioni
hardware granulari (ad esempio per i kernel Pallas) per diagnosticare colli di
bottiglia delle prestazioni che altrimenti non sarebbero visibili nelle tracce
di eventi standard.

> **Note:** il campionamento granulare dei contatori di prestazioni è una
> funzionalità hardware avanzata disponibile per **Ironwood (TPU v7/TPU7x)** e
> per le generazioni di TPU successive.

### Attivare i contatori di prestazioni { #enabling-performance-counters }

Puoi attivare i contatori di prestazioni passando la configurazione dei
contatori tramite `experimental_options` in `TpuProfilerConfig`:

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

Per l'elenco dettagliato dei parametri di configurazione (come gli intervalli
di campionamento, lo scaling e le modalità di trigger) e le istruzioni su come
determinare gli indici dei contatori, consulta la
[guida ufficiale alla profilazione dei kernel di OpenXLA](https://openxla.org/xprof/kernel-profiling).

## 8. Riferimento rapido { #8-quick-reference }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
Passaggio   | Comando / codice
----------- | --------------
**Importazione** | `from torch.tpu.profiler import TpuProfilerConfig`
**Profilazione** | `with torch.profiler.profile(activities=[...], on_trace_ready=...):`
**Salvataggio**  | `torch.profiler.tensorboard_trace_handler("/tmp/profiler_output")`
**Visualizzazione** | `tensorboard --logdir=/tmp/profiler_output`
<!-- mdformat on -->

> **Note:** profila solo 100–1000 step per mantenere gestibili i file di
> traccia. Il profiler aggiunge un overhead di circa il 5–10% per step.
