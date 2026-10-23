# Performance-Profiling mit XProf { #performance-profiling-with-xprof }

Beim Training großer Modelle auf der TPU ist es entscheidend zu verstehen, wo
die Zeit verbraucht wird. In diesem Leitfaden erfahren Sie, wie Sie
Ausführungs-Traces mit dem TPU-Profiler erfassen und in TensorBoard
analysieren.

> **Warning:** **Voraussetzung:** TensorBoard benötigt `setuptools<81`. Die
> neueste Version entfernt `pkg_resources`, von dem TensorBoard abhängt.
> Installieren Sie es zuerst:
>
> ```shell
> pip install "setuptools<81"
> pip install xprof tensorboard
> ```

> **Note:** **PyTorch-Version:** Die native TPU-Integration von
> `torch.profiler` erfordert PyTorch 2.12 oder neuer.

## 1. Der TPU-Profiler { #1-the-tpu-profiler }

Der Standard-`torch.profiler` (integriert in `torch_tpu`) bietet eine
**Kontextmanager**-Schnittstelle, mit der Sie Aktivitäten auf CPU und TPU
tracen. Sie umschließen den Code, den Sie profilieren möchten, mit einem
`torch.profiler.profile()`-Block. Die erfassten Traces werden auf der
Festplatte gespeichert, damit Sie sie in TensorBoard ansehen können.

**Wichtige Komponenten:**

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Komponente                                  | Aufgabe                            |
| ------------------------------------------ | ------------------------------- |
| `torch.profiler.profile()`                 | Standard-Kontextmanager, der die Trace-Erfassung <br> startet und stoppt |
| `torch.profiler.ProfilerActivity.CPU`      | Erfasst Traces der Operationen auf der CPU-Seite (Dispatch, <br> Laden der Daten) |
| `torch.profiler.ProfilerActivity.TPU`      | Erfasst Traces der Ausführung auf der TPU-Hardware (Berechnung, <br> Speicher, Interconnect) |
| `torch.profiler.tensorboard_trace_handler` | Callback, der Traces für TensorBoard <br> in einem Verzeichnis speichert       |
| `TpuProfilerConfig`                        | Hilfsklasse für eine eigene Konfiguration des TPU-Profilings                   |
<!-- mdformat on -->

```python
import torch
from torch.tpu.profiler import TpuProfilerConfig

device = torch.device("tpu")
```

## 2. Eigene Profiling-Konfiguration (`TpuProfilerConfig`) { #2-custom-profiling-configuration-tpuprofilerconfig }

Mit den Optionen, die Sie über `TpuProfilerConfig` übergeben, steuern Sie die
Tracer-Stufen für Host und Gerät sowie die Struktur der Ausgabeverzeichnisse
im Detail.

### Unterstützte Konfigurationsoptionen { #sanctioned-configuration-options }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Optionsschlüssel            | Typ           | Standard | Beschreibung               |
| :-------------------- | :------------- | :------ | :------------------------ |
| `host_tracer_level`   | `int`          | `2`     | Steuert den Detailgrad des CPU-Host-Tracers <br> (0=aus, 1=kritisch, 2=alle). |
| `device_tracer_level` | `int`          | `1`     | Steuert den Detailgrad des Tracers für das <br> TPU-Hardwaregerät (0=aus, 1=an). |
| `python_tracer_level` | `int`          | `0`     | Steuert die Stufe für das Tracing von <br> Python-Funktionen (0=aus, 1=an). |
| `run_dir`             | `pathlib.Path` | `None`  | Ausgabeverzeichnis, in dem die Trace-Dateien der <br> TPU-Hardware (`.xplane.pb`) gespeichert werden. |
| `worker_rank`         | `str`          | `None`  | Worker-Rank, der als Suffix an den <br> XPlane-Dateinamen angehängt wird (`<hostname>_<rank>.xplane.pb`). |
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

## 3. Profil erfassen { #3-capturing-a-profile }

Die folgende Zelle führt eine kleine Trainingsschleife im Profiler-Kontext
aus. Die wichtigsten Punkte:

1.  Umschließen Sie Ihre Trainingsschleife mit `torch.profiler.profile()`
2.  Geben Sie die Aktivitäten an: `CPU` für Ops auf dem Host, `TPU` für die
    Ausführung auf der Hardware
3.  Legen Sie `on_trace_ready` fest, um Traces mit
    `torch.profiler.tensorboard_trace_handler` in einem Verzeichnis zu speichern
4.  (Optional) Übergeben Sie `experimental_config` mit `TpuProfilerConfig`, um
    die Tracer-Stufen zu steuern

> **Important:** Der Profiler verursacht Overhead. Profilieren Sie nur eine
> repräsentative Teilmenge der Schritte (z. B. 100–1000), nicht den gesamten
> Trainingslauf.

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

## 4. Traces in TensorBoard ansehen { #4-viewing-traces-in-tensorboard }

Führen Sie die folgenden Befehle aus:

**Schritt 1:** Installieren Sie das Profiler-Plug-in (erforderlich, damit das
Dashboard angezeigt wird):

```shell
pip install tensorboard-plugin-profile "setuptools<81"
```

**Schritt 2:** Starten Sie TensorBoard:

```shell
tensorboard --logdir=/tmp/profiler_output --port=6006
```

XProf kann auch als eigenständiger Server ausgeführt werden.

```shell
xprof --logdir=/tmp/profiler_output --port=6006
```

**Schritt 3:** Öffnen Sie `http://localhost:6006` in Ihrem Browser und wählen
Sie den Tab **„Profile“** aus.

> **Note:** Sie müssen TensorBoard im integrierten Terminal von VS Code
> ausführen, damit die Portweiterleitung automatisch funktioniert.

**Auf GKE:** Schreiben Sie das Trace-Verzeichnis in einen Cloud Storage-Bucket,
der mit dem
[Cloud Storage FUSE-CSI-Treiber](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver)
eingebunden ist, damit die Traces den Pod überdauern. Öffnen Sie sie dann mit
`xprof
--logdir=gs://<bucket>/profiler_output`.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Ansicht               | Was sie zeigt                                          |
| ------------------ | ------------------------------------------------------ |
| **Overview**       | Übersicht: Schrittdauer, Geräteauslastung, Top-Ops |
| **Trace Viewer**   | Zeitachse aller Ops auf CPU und TPU. Achten Sie auf Lücken (ungenutzte Hardware) |
| **Op Profile**     | Aufschlüsselung der Zeit pro Operation. Finden Sie die teuersten Ops |
| **Memory Profile** | HBM-Nutzung im Zeitverlauf. Erkennen Sie Speicherspitzen |
<!-- mdformat on -->

## 5. Worauf Sie achten sollten { #5-what-to-look-for }

### Warnsignale in Traces { #red-flags-in-traces }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Muster                 | Bedeutung                 | Lösung                      |
| ----------------------- | ----------------------- | ------------------------ |
| Große Lücken zwischen TPU-Ops | Engpass auf dem Host <br> (Laden der Daten, Python-Overhead) | Daten vorab laden, Python-Logik in der Schleife reduzieren |
| Häufige kurze TPU-Aktivitätsschübe   | Zu viele Graph-Breaks → häufige Neukompilierung | Aufrufe von `.item()` / `.cpu()` reduzieren, `torch.compile` verwenden |
| Geringe MXU-Auslastung | Tensor-Dimensionen sind nicht an der Tile-Größe (128) ausgerichtet | Dimensionen auf Vielfache von 128 auffüllen |
<!-- mdformat on -->

### Das ideale Profil { #the-ideal-profile }

Eine gut optimierte TPU-Arbeitslast zeigt Folgendes:

-   **Lange, zusammenhängende Rechenblöcke auf der TPU**: Der Compiler hat
    viele Ops zu einer Ausführung fusioniert
-   **Minimale Lücken zwischen CPU und TPU**: Das Laden der Daten überlappt
    mit der Berechnung
-   **Hohe MXU-Auslastung (>60 %)**: Die Tensor-Shapes sind an der Hardware
    ausgerichtet

## 6. Prüfen, ob die Traces erfasst wurden { #6-verifying-the-traces-were-captured }

Die folgende Zelle prüft, ob das Ausgabeverzeichnis des Profilers
Trace-Dateien enthält.

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

## 7. Kernel-Profiling (detaillierte Leistungszähler) { #7-kernel-profiling-fine-grained-performance-counters }

Mit Kernel-Profiling können Sie detaillierte Leistungszähler der Hardware
erfassen (z. B. für Pallas-Kernels). So diagnostizieren Sie
Performance-Engpässe, die in Standard-Event-Traces sonst nicht sichtbar sind.

> **Note:** Das Sampling detaillierter Leistungszähler ist eine fortgeschrittene
> Hardwarefunktion, die für **Ironwood (TPU v7/TPU7x)** und spätere
> TPU-Generationen verfügbar ist.

### Leistungszähler aktivieren { #enabling-performance-counters }

Sie können Leistungszähler aktivieren, indem Sie die Zählerkonfiguration über
`experimental_options` in `TpuProfilerConfig` übergeben:

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

Die ausführliche Liste der Konfigurationsparameter (etwa Sampling-Intervalle,
Skalierung und Trigger-Modi) und eine Anleitung zum Ermitteln der Zählerindizes
finden Sie im offiziellen
[OpenXLA-Leitfaden zum Kernel-Profiling](https://openxla.org/xprof/kernel-profiling).

## 8. Kurzreferenz { #8-quick-reference }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
Schritt        | Befehl / Code
----------- | --------------
**Importieren**  | `from torch.tpu.profiler import TpuProfilerConfig`
**Profilieren** | `with torch.profiler.profile(activities=[...], on_trace_ready=...):`
**Speichern**    | `torch.profiler.tensorboard_trace_handler("/tmp/profiler_output")`
**Ansehen**    | `tensorboard --logdir=/tmp/profiler_output`
<!-- mdformat on -->

> **Note:** Profilieren Sie nur 100–1000 Schritte, damit die Trace-Dateien
> handhabbar bleiben. Der Profiler verursacht pro Schritt etwa 5–10 % Overhead.
