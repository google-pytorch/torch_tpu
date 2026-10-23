# TorchTPU-API-Referenz { #torchtpu-api-reference }

Dieses Dokument ist die Referenz für die öffentlichen Python-APIs und
Umgebungsvariablen von TorchTPU. Es hilft Ihnen dabei:

*   **Von CUDA zu TPU migrieren**: Gängige Aufrufe von `torch.cuda` den
    geräteunabhängigen APIs
    [`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html)
    von PyTorch zuordnen.
*   **TPU-spezifische Funktionen nutzen**: Über das Modul **`torch.tpu`**
    Ausführungsmodi, numerische MXU-Genauigkeit, verteilte Topologie,
    eigene Pallas-/TPU-Operatoren, tracebare Optimizer und
    XProf-Profiling konfigurieren.
*   **Laufzeit konfigurieren**: Über Umgebungsvariablen Kompilierungscaching,
    Debugging-Prüfungen und Starteinstellungen für mehrere Hosts anpassen.

> **Note:** Die vollständige, automatisch generierte und maßgebliche Liste
> aller öffentlichen Python-APIs und Umgebungsvariablen finden Sie unter
> [Öffentliche APIs von TorchTPU (`public_apis.md`)](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/public_apis.md).

## **Grundlegende Geräteabfragen** { #core-device-queries }

Wenn `torch` bei installiertem TorchTPU importiert wird, registriert das
Backend automatisch das Gerät `"tpu"` und das Modul `torch.tpu`. Wo PyTorch
eine geräteunabhängige API
[`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html)
bietet, verwenden Sie diese statt einer backend-spezifischen API. So bleibt Ihr
Code über Beschleuniger hinweg portabel. Verwenden Sie `torch.tpu` für
TPU-spezifische Hardwareeigenschaften ohne generisches Äquivalent.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| CUDA-API | TPU-/Accelerator-API | Beschreibung |
| :--- | :--- | :--- |
| `torch.device("cuda")` | `torch.accelerator.current_accelerator()` | Gibt `torch.device("tpu")` zurück, wenn TPU der aktive Beschleuniger ist. |
| `torch.cuda.is_available()` | `torch.accelerator.is_available()` | Prüft, ob das TPU-Backend verfügbar ist. |
| `torch.cuda.is_initialized()` | `torch.tpu.is_initialized()` | Prüft, ob der Laufzeitstatus von TorchTPU initialisiert wurde. |
| `torch.cuda.device_count()` | `torch.accelerator.device_count()` | Anzahl der Geräte, die für den aktuellen Prozess sichtbar sind. |
| `torch.cuda.current_device()` | `torch.accelerator.current_device_index()` | Gibt den Index des aktiven TPU-Geräts zurück. |
| `torch.cuda.set_device(rank)` | `torch.accelerator.set_device_index(rank)` | Legt den Index des aktiven TPU-Geräts für den aktuellen Prozess fest. |
| `torch.cuda.device(rank)` | `torch.accelerator.device_index(rank)` | Kontextmanager, der den Index des aktiven Geräts vorübergehend festlegt. |
| `torch.cuda.get_device_name(device=None)` | `torch.tpu.get_device_properties(device=None).name` | Gibt den lesbaren Modellnamen der angeschlossenen TPU zurück (z. B. `"TPU v6e"`). |
| `torch.cuda.get_device_properties(device=None)` | `torch.tpu.get_device_properties(device=None)` | Gibt statische Eigenschaften des TPU-Geräts zurück (darunter `name`, `devices_per_chip`, `hbm_bytes_per_chip`, `hbm_bytes_per_device`, `bf16_flops_per_chip`, `bf16_flops_per_device`, `sparse_cores_per_device` und `sparse_core_simd_width`). |
<!-- mdformat on -->

Die APIs von `torch.accelerator` in dieser Tabelle erfordern PyTorch 2.11 oder
höher. Falls Sie in seltenen Fällen eine ältere PyTorch-Version verwenden,
nutzen Sie stattdessen die entsprechende API von `torch.tpu` (zum Beispiel
`torch.tpu.device_count()` oder `torch.tpu.current_device()`). Ab PyTorch 2.11
sind diese redundanten APIs von `torch.tpu` intern in TorchTPU. Wenn Sie sie
direkt aus Nutzercode aufrufen, wird ein Fehler ausgelöst.

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

## **Zufallszahlengenerierung (RNG)** { #random-number-generation-rng }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| CUDA-API | TPU-/Accelerator-API | Beschreibung |
| :--- | :--- | :--- |
| `torch.cuda.manual_seed(seed)` | `torch.tpu.manual_seed(seed)` | Setzt den Seed für den Zufallszahlengenerator des aktiven TPU-Geräts. |
| `torch.cuda.manual_seed_all(seed)` | `torch.tpu.manual_seed_all(seed)` | Setzt den Seed für den RNG auf allen lokalen TPU-Geräten. |
| `torch.cuda.initial_seed()` | `torch.accelerator.random.initial_seed()` | Gibt den anfänglichen Zufalls-Seed des aktuellen TPU-Geräts zurück. |
| `torch.cuda.get_rng_state()` | `torch.accelerator.random.get_rng_state()` | Gibt den Tensor mit dem RNG-Status für das aktuelle Gerät zurück. |
| `torch.cuda.set_rng_state(new_state, device=...)` | `torch.tpu.set_rng_state(new_state, device=...)` | Legt den Tensor mit dem RNG-Status für das aktive TPU-Gerät des aktuellen Prozesses fest. |
| `torch.cuda.default_generators` | `torch.tpu.default_generators` | Verzögert initialisiertes Tupel von Standardinstanzen von `torch.Generator`, indiziert nach der lokalen Geräteordnungszahl. |
<!-- mdformat on -->

Die APIs von `torch.accelerator.random` in dieser Tabelle erfordern PyTorch
2.13 oder höher. Falls Sie in seltenen Fällen eine ältere PyTorch-Version
verwenden, nutzen Sie stattdessen `torch.tpu.initial_seed()` oder
`torch.tpu.get_rng_state()`. Ab PyTorch 2.13 sind diese APIs von `torch.tpu`
intern in TorchTPU. Wenn Sie sie aus Ihrem Code aufrufen, wird ein Fehler
ausgelöst.

```python
# Set the seed across all cores
torch.tpu.manual_seed_all(42)
print("Seeded all TPU cores with value 42.")
```

## **Automatic Mixed Precision (AMP)** { #automatic-mixed-precision-amp }

TorchTPU ist in das Standard-Framework
[`torch.amp.autocast`](https://docs.pytorch.org/docs/stable/amp.html) von
PyTorch integriert und verwendet dafür den Gerätetyp-String `"tpu"`.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| CUDA-API | TPU-API | Beschreibung |
| :--- | :--- | :--- |
| `torch.amp.autocast("cuda", ...)` | `torch.amp.autocast("tpu", ...)` | Kontextmanager / Decorator, der Automatic Mixed Precision aktiviert (auf TPU standardmäßig `bfloat16`). |
| `torch.cuda.get_amp_supported_dtype()` | `torch.tpu.get_amp_supported_dtype()` | Listet die auf TPU unterstützten dtypes auf (`bfloat16`, `float16`, `float32`, `float64`, `float8_*`, Ganzzahl-, boolesche und komplexe Typen). |
<!-- mdformat on -->

```python
supported_dtypes = torch.tpu.get_amp_supported_dtype()
print(f"AMP supported dtypes: {supported_dtypes}")
```

## **Streams und Events** { #streams-and-events }

Verwenden Sie für die Stream-Verwaltung und die Gerätesynchronisierung die
Standard-APIs von
[`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html)
statt backend-spezifischer APIs. Verwenden Sie `torch.tpu` nur, wenn Sie
TPU-Objekte vom Typ `Stream` oder `Event` direkt erstellen oder das Pinning von
TPU-Kernen konfigurieren.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| CUDA-API | TPU-/Accelerator-API | Beschreibung |
| :--- | :--- | :--- |
| `torch.cuda.current_stream()` | `torch.accelerator.current_stream()` | Gibt den aktiven Stream für ein Gerät zurück. |
| `torch.cuda.set_stream(stream)` | `torch.accelerator.set_stream(stream)` | Legt den aktiven Stream fest. |
| `torch.cuda.default_stream(device=None)` | `torch.tpu.default_stream(device=None)` | Gibt den Standard-`TpuStream` für das Gerät zurück. |
| `torch.cuda.stream(stream)` | `torch.tpu.stream(stream)` | Kontextmanager, der einen bestimmten `TpuStream` auswählt. |
| `torch.cuda.synchronize()` | `torch.accelerator.synchronize()` | Wartet, bis alle ausstehenden Arbeiten auf dem TPU-Gerät abgeschlossen sind. |
| `torch.cuda.Stream` | `torch.tpu.Stream(device=None, priority=0, *, pinned=None)` | Unterklasse von `torch.Stream`. Unterstützt `synchronize()`, `record_event()`, `wait_event()`, `wait_stream()` und Kern-Pinning (`pinned="dense"` oder `"sparse"`, abfragbar über `is_dense` / `is_sparse`). |
| `torch.cuda.Event` | `torch.tpu.Event(enable_timing=False)` | Unterklasse von `torch.Event`. Unterstützt `record()`, `synchronize()`, `query()` und `elapsed_time(end_event)`. |
<!-- mdformat on -->

Die APIs von `torch.accelerator` in dieser Tabelle erfordern PyTorch 2.11 oder
höher. Falls Sie in seltenen Fällen eine ältere PyTorch-Version verwenden,
nutzen Sie stattdessen `torch.tpu.current_stream()`, `torch.tpu.set_stream()`
oder `torch.tpu.synchronize()`.

> **Note:** Streams sind ein CUDA-Konzept ohne direkte Entsprechung auf TPU. Im
> verzögerten Ausführungsmodell von TorchTPU planen und überlappen der
> XLA-Compiler und die PjRt-Laufzeit Operationen automatisch anhand von
> Datenabhängigkeiten. Eine manuelle Verwaltung mehrerer Streams ist daher
> nicht nötig.

## **TPU-spezifische APIs** { #tpu-specific-apis }

Über die Standardäquivalente zu CUDA hinaus stellt `torch.tpu` APIs bereit, die
speziell für TPU-Ausführungsmodi, MXU-Genauigkeit, XLA-Kompilierung, verteilte
Topologie, eigene Pallas-Kernels, tracebare Optimizer und Profiling
entwickelt wurden.

### **Ausführungsmodi und Compiler-Optionen** { #execution-modes-and-compiler-options }

Steuern Sie, wie Eager-Operationen in die Warteschlange gestellt, fusioniert
und von XLA kompiliert werden. Eine Erläuterung der Architektur finden Sie
unter [Ausführungsmodi von TorchTPU](./torchtpu_execution_model.md).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Beschreibung |
| :--- | :--- |
| `torch.tpu.EagerMode` | Enum zur Steuerung der Eager-Ausführung: `DEFER_AND_FUSE`, `DEFER_NEVER` und `DEFER_NEVER_AND_LAUNCH_BLOCKING`. |
| `torch.tpu.eager_mode` | Ruft den aktiven `EagerMode` global ab oder legt ihn fest. |
| `torch.tpu.set_eager_mode(mode)` | Kontextmanager, der den aktiven `EagerMode` vorübergehend festlegt. |
| `torch.tpu.compiler.custom_compiler_options(options)` | Kontextmanager, der benutzerdefinierte XLA-Compiler-Flags an Eager-Operationen übergibt, die innerhalb des Blocks materialisiert werden. |
<!-- mdformat on -->

```python
print(f"Default eager_mode: {torch.tpu.eager_mode}")
with torch.tpu.set_eager_mode(torch.tpu.EagerMode.DEFER_AND_FUSE):
  x = torch.ones((4, 4), device="tpu") + 1
  print(
      f"Inside context manager: {torch.tpu.eager_mode}, sum={x.sum().item()}"
  )
```

### **Numerische Genauigkeit** { #numerical-precision }

Steuern Sie die Genauigkeit der Matrix Multiply Unit (MXU) und die
Zwischenakkumulation in `float32`. Ausführliche Hinweise finden Sie unter
[Numerische Genauigkeit](./numerical_precision.md).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Beschreibung |
| :--- | :--- |
| `torch.tpu.Precision` | Enum zur Steuerung der MXU-Genauigkeit für Matmuls und Faltungen in `float32`: `Precision.DEFAULT`, `Precision.HIGH` und `Precision.HIGHEST`. |
| `torch.tpu.precision(level)` | Kontextmanager, der die `Precision`-Stufe für Matmul- und Faltungsoperationen innerhalb des Blocks festlegt. |
| `torch.tpu.allow_excess_precision` | Ruft das XLA-Flag `--xla_allow_excess_precision` global ab oder legt es fest, um fusionierte Zwischenwerte in `float32` zu halten. |
<!-- mdformat on -->

```python
a = torch.randn(64, 64, device="tpu", dtype=torch.float32)
b = torch.randn(64, 64, device="tpu", dtype=torch.float32)
with torch.tpu.precision(torch.tpu.Precision.HIGH):
  c = torch.matmul(a, b)
print(f"Computed HIGH-precision matmul with output shape: {tuple(c.shape)}")
```

### **Verteiltes Training und Topologie** { #distributed-training-and-topology }

TorchTPU registriert `"tpu_dist"` (auch mit dem Alias `"tpu"`) als
Standard-Backend von `torch.distributed` für Geräte vom Typ `"tpu"`. Siehe
[Erste Schritte mit TorchTPU auf GKE](./get_started_on_gke.md) und
[Best Practices für FSDP](./case_studies_fsdp.md).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Beschreibung |
| :--- | :--- |
| `torch.tpu.topology_aware_mesh(mesh_shape, ...)` | Berechnet einen topologiebewussten Tensor für das Rank-Layout mit der Shape `mesh_shape` für `DeviceMesh("tpu", mesh=...)`. |
| `torch.tpu.distributed.set_tpu_launch_env(...)` | Befüllt die Umgebungsvariablen für die TPU-Topologie und den Start auf mehreren Hosts, bevor Worker gestartet werden. |
| `torch.tpu.distributed.spmd_safe(func)` | Decorator, der deklariert, dass alle Ranks in `func` identische TPU-Operationen ausführen. Dadurch wird die Fusion kollektiver Operationen im Eager-Modus möglich. |
| `torch.tpu.window_num_stripes` | Ruft die maximale Anzahl gleichzeitiger DMA-Stripes für Peer-to-Peer-Übertragungen ab oder legt sie fest. |
| `torch.tpu.window_stripe_chunk_mb` | Ruft die minimale Nutzlastgröße (in MB) pro DMA-Stripe für Peer-to-Peer-Übertragungen ab oder legt sie fest. |
<!-- mdformat on -->

### **Eigene TPU-Operatoren (`torch.tpu.ops`) und Pallas (`torch.tpu.pallas`)** { #tpu-custom-operators-torchtpuops-and-pallas-torchtpupallas }

TorchTPU stellt hardwarespezifische Operatoren unter `torch.tpu.ops` und einen
Wrapper für eigene JAX/Pallas-Kernels unter `torch.tpu.pallas` bereit. Siehe
[Eigene Kernels mit Pallas hinzufügen](./custom_kernels.md).

#### **Allgemeine, Attention-, MoE- und kollektive Operatoren** { #general-attention-moe-and-collective-operators }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Beschreibung |
| :--- | :--- |
| `torch.tpu.ops.max_pool{1,2,3}d(...)` | Wendet 1D-, 2D- oder 3D-Max-Pooling auf TPU an, ohne einen Index-Tensor zu materialisieren. |
| `torch.tpu.ops.max_pool{1,2,3}d_backward(...)` | Berechnet den Backward-Pass für `max_pool{1,2,3}d` ohne Index-Tensor. |
| `torch.tpu.ops.optimization_barrier(inputs)` | Verhindert, dass XLA Berechnungen über die Barriere hinweg verschiebt, fusioniert oder dedupliziert. |
| `torch.tpu.ops.splash_sdpa(q, k, v, ...)` | Scaled Dot-Product Attention auf Basis des Pallas-Kernels Splash Attention für TPU. |
| `torch.tpu.ops.ragged_dot(lhs, rhs, group_sizes, ...)` | Gruppierte Matrixmultiplikation für Mixture-of-Experts-Arbeitslasten (MoE). |
| `torch.tpu.ops.ragged_dot_weight_grad(lhs, grad_output, group_sizes)` | Berechnet den Gewichtsgradienten für `ragged_dot`. |
| `torch.tpu.ops.align_jagged_to_multiple(nt, ...)` | Füllt den Puffer eines `NestedTensor` mit dem Layout `torch.jagged` zur TPU-Ausrichtung auf ein angegebenes Vielfaches auf. |
| `torch.tpu.ops.strip_jagged_padding(nt, ...)` | Entfernt das nachgestellte Padding, das `align_jagged_to_multiple` hinzugefügt hat. |
| `torch.tpu.ops.all_gather_dim(input, dim, group=None)` | Führt ein All-Gather von `input` über `group` aus und verkettet entlang der Dimension `dim`. |
| `torch.tpu.ops.reduce_scatter_dim(input, dim, group=None, ...)` | Reduziert `input` über `group` und verteilt Teilstücke entlang der Dimension `dim`. |
| `torch.tpu.ops.ragged_all_to_all(...)` | Tauscht Tensor-Slices variabler Länge zwischen den Ranks einer Prozessgruppe aus. |
| `torch.tpu.ops.send(tensors, dst, tag)` / `torch.tpu.ops.recv(tensors, src, tag)` | Punkt-zu-Punkt-Senden und -Empfangen von Tensoren in der Standard-TPU-Prozessgruppe. |
| `torch.tpu.pallas.jax_op(name, fn=None, ...)` | Registriert eine JAX/Pallas-Funktion als tracebaren eigenen PyTorch-Operator. |
<!-- mdformat on -->

#### **SparseCore-Embedding-Operatoren** { #sparsecore-embedding-operators }

Diese Operatoren erfordern eine TPU mit SparseCores. Die Eigenschaft
`sparse_cores_per_device` von `torch.tpu.get_device_properties()` gibt die
Anzahl der SparseCores pro Gerät an (0 bei TPUs ohne SparseCores), und
`sparse_core_simd_width` ist deren SIMD-Breite.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Beschreibung |
| :--- | :--- |
| `torch.tpu.ops.sparse_dense_matmul(...)` | Führt CSR-Embedding-Lookup und -Aggregation auf TPU SparseCore aus. |
| `torch.tpu.ops.sparse_dense_matmul_grad_with_sgd(...)` / `_with_adagrad(...)` / `_with_adam(...)` | Berechnet die Backward-Gradienten für `sparse_dense_matmul` und wendet auf SparseCore ein fusioniertes In-Place-Update der Tabelle mit SGD, Adagrad oder Adam an. |
| `torch.tpu.ops.sparse_dense_matmul_activation_unstack(...)` | Teilt gestapelte SparseCore-Embedding-Aktivierungen in Tensoren pro Feature auf. |
| `torch.tpu.ops.sparse_dense_matmul_gradient_stack(...)` | Stapelt Aktivierungsgradienten pro Feature in einen einzelnen Tensor für SparseCore. |
| `torch.tpu.ops.sparse_gather(...)` | Sammelt Zeilen aus einem 2D-Tensor anhand von CSR-Indizes auf TPU SparseCore. |
| `torch.tpu.ops.sparse_gather_backward(...)` | Führt ein Scatter-Add der Gradienten für `sparse_gather` auf TPU SparseCore aus. |
| `torch.tpu.ops.sparse_iota(...)` | Erzeugt Positionsindizes innerhalb einer Zeile für Elemente ungleich null in einer CSR-Matrix auf SparseCore. |
| `torch.tpu.ops.sparse_map_row(...)` | Wendet eine eigene zeilenweise MLIR-Berechnung auf dünnbesetzte CSR-Matrizen auf SparseCore an. |
<!-- mdformat on -->

### **Fusionierter Trainingsschritt und Optimizer (`torch.tpu.optimizers`)** { #fused-training-step-and-optimizers-torchtpuoptimizers }

`torch.tpu.optimizers` erfasst Forward-Pass, Loss-Berechnung, Backward-Pass und
den funktionalen Optimizer-Schritt per Tracing in einem einzigen kompilierten
TPU-Graphen.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Beschreibung |
| :--- | :--- |
| `torch.tpu.optimizers.SingleTraceTrainer(model, optimizer=None, ...)` | Verwaltet funktionale Parameter (`params`) und den Optimizer-Zustand (`param_group`) und kompiliert den vollständigen Trainingsschritt über `make_compiled_train_step(...)`. |
| `torch.tpu.optimizers.Optimizer(lr=..., weight_decay=...)` | Basisklasse für zustandslose funktionale Optimizer (stellt `defaults`, `init_param_group` und `step` bereit). |
| `torch.tpu.optimizers.AdamW(...)` | Tracebarer funktionaler AdamW-Optimizer (`AdamW.ParamGroup`). |
| `torch.tpu.optimizers.Muon(...)` | Tracebarer funktionaler Muon-Optimizer (`Muon.ParamGroup`). |
| `torch.tpu.optimizers.Sgd(...)` | Tracebarer funktionaler SGD-Optimizer (`Sgd.ParamGroup`). |
<!-- mdformat on -->

### **Profiling (`torch.tpu.profiler`)** { #profiling-torchtpuprofiler }

Konfigurieren Sie das Tracing von TPU-Hardware und Python für XProf und
TensorBoard. Siehe [XProf für TorchTPU](./performance_profiling_with_xprof.md).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Beschreibung |
| :--- | :--- |
| `torch.tpu.profiler.TpuProfilerConfig(...)` | Experimentelle Konfiguration, die an `torch.profiler.profile(..., experimental_config=...)` übergeben wird, um das Tracing von Host, TPU-Gerät und Python für XProf zu konfigurieren. |
<!-- mdformat on -->

## **Umgebungsvariablen** { #environment-variables }

Das Verhalten von TorchTPU wird über mehrere Umgebungsvariablen gesteuert. Sie
konfigurieren den XLA-Compiler, den TPU-Treiber (`libtpu`) und das eigene
Debug-Subsystem von PyTorch.

> **KRITISCH: Prozessglobale Initialisierung** Die Umgebungsvariablen von
> TorchTPU werden in dem Moment **gesperrt**, in dem das Backend initialisiert
> wird (normalerweise bei `import torch` oder `torch.device("tpu")`).

### **Gerätesichtbarkeit, Laufzeit und verteilter Start** { #device-visibility-runtime-and-distributed-launch }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Umgebungsvariable | Beschreibung |
| :--- | :--- |
| `TPU_VISIBLE_DEVICES` | Kommagetrennte Indizes der TPU-Chips, die für den aktuellen Prozess sichtbar sind (analog zu `CUDA_VISIBLE_DEVICES`). |
| `RANK` | Globaler Rank des aktuellen Prozesses in einem verteilten Job. |
| `LOCAL_RANK` | Hostlokaler Rank des aktuellen Prozesses. |
| `WORLD_SIZE` | Gesamtzahl der Prozesse im verteilten Job. |
| `NNODES` | Gesamtzahl der Hostknoten in einem Job mit mehreren Hosts. |
| `NODE_RANK` | Rank des aktuellen Hostknotens in einem Job mit mehreren Hosts. |
| `MASTER_ADDR` | Adresse des Koordinatorknotens mit Rank 0 für `torch.distributed`. |
| `MASTER_PORT` | Port auf `MASTER_ADDR`, der für das Rendezvous von `torch.distributed` verwendet wird. |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | Auf `"0"` setzen, um die automatische Registrierung des Backends `torch_tpu` zu deaktivieren. |
| `TORCH_TPU_ALLOW_LEGACY_DEVICE_APIS` | Wenn aktiviert, dürfen interne Geräte-APIs von `torch.tpu` (solche mit einem Äquivalent in `torch.accelerator`) von außerhalb des PyTorch-Kerns aufgerufen werden. Statt eines `RuntimeError` erscheint dann einmalig eine Warnung. Verwenden Sie das nur als vorübergehende Lösung. Die Migration zu `torch.accelerator.*` ist vorzuziehen, und diese Variable wird voraussichtlich in einem künftigen Release entfernt. |
| `LIBTPU_INIT_ARGS` | Durch Leerzeichen getrennte Initialisierungs-Flags, die an `libtpu` und die PJRT-Laufzeit übergeben werden. |
| `TPU_PREMAPPED_BUFFER_SIZE` | Größe des vorab gemappten Speicherpuffers vom Host zum Gerät in Byte für den PJRT-Client. |
| `TMPDIR` | Temporäres Verzeichnis, das von Laufzeit und Compiler verwendet wird. |
<!-- mdformat on -->

### **Hardwaretopologie und Koordination mehrerer Hosts** { #hardware-topology-and-multi-host-coordination }

Diese Variablen geben die TPU-Hardwaretopologie und die Vernetzung der Prozesse
an. In den meisten Fällen werden sie automatisch von
`torch.tpu.distributed.set_tpu_launch_env()`, dem TorchTPU-Launcher oder der
Google Cloud-Umgebung festgelegt.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Umgebungsvariable | Beschreibung |
| :--- | :--- |
| `TORCH_TPU_TOPOLOGY` | Physische Topologie des TPU-Slices (z. B. `2x2x1`, `2x2x4`). |
| `TORCH_TPU_SLICEBUILDER_ADDRESSES` | Kommagetrennte Worker-Adressen im Format `host:port` für den Slice-Builder-Dienst der TPU. |
| `TORCH_TPU_DEVICES_PER_HOST` | Anzahl der TPU-Chips pro Host beim Ableiten der Slice-Builder-Adressen. |
| `TORCH_TPU_BASE_PORT` | Startportnummer beim Ableiten der Slice-Builder-Adressen. |
| `TORCH_TPU_SPMD_HANDSHAKE_STAGE` | Phase, in der die Ranks prüfen, ob die Graphen mit kollektiven Operationen auf allen Ranks übereinstimmen. |
| `TORCHCOMM_RANK` | Prozess-Rank für `torchcomm` (hat Vorrang vor `RANK`). |
| `TORCHCOMM_SIZE` | World Size für `torchcomm` (hat Vorrang vor `WORLD_SIZE`). |
<!-- mdformat on -->

### **Ausführungsmodus, Kompilierung und Caching** { #execution-mode-compilation-and-caching }

Diese Variablen steuern, wie PyTorch-Operationen und -Graphen dispatcht,
gelowert, von XLA optimiert und über Ausführungen und Ranks hinweg gecacht
werden.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Umgebungsvariable | Beschreibung |
| :--- | :--- |
| `TORCH_TPU_DEFER_AND_FUSE` | Wenn aktiviert (`"1"`/`"true"`), wird der Standard-Eager-Modus auf `EagerMode.DEFER_AND_FUSE` gesetzt. |
| `TORCH_TPU_LAUNCH_BLOCKING` | Wenn aktiviert (`"1"`/`"true"`), wird der Standard-Eager-Modus auf `EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING` gesetzt. |
| `TORCH_TPU_TIER2_COMPILATION_CACHE` | Verzeichnisname unter `/dev/shm/torch_tpu_cache/` für den hostlokalen Kompilierungscache (oder `"disabled"`). |
| `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` | Pfad (z. B. `gs://...` oder ein lokaler Pfad) zum persistenten Kompilierungscache der Stufe 3, der von Hosts und Ausführungen gemeinsam genutzt wird. |
| `XLA_FLAGS` | Durch Leerzeichen getrennte XLA-Compiler-Flags (z. B. `--xla_dump_hlo_as_text --xla_dump_to=...`). |
<!-- mdformat on -->

### **Debugging, Sicherheit, Profiling und Tracing** { #debugging-safety-profiling-and-tracing }

Diese Variablen steuern den Debugging-Kontext, Sicherheitsprüfungen zur
Laufzeit und die Profiler-Ausgabe von TorchTPU.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Umgebungsvariable | Beschreibung |
| :--- | :--- |
| `TORCH_SHOW_CPP_STACKTRACES` | Bei `"1"` werden Python-Ausnahmen vom Typ `RuntimeError` um C++-Backtraces ergänzt. |
| `TORCH_TPU_ENABLE_DEBUG_CHECKS` | Bei `"1"` werden Debug- und Bereichsprüfungen zur Laufzeit auf TPU aktiviert. |
| `TORCH_TPU_PROFILER_OUTPUT_DIR` | Fallback-Ausgabeverzeichnis für XProf-Trace-Dateien, wenn `TpuProfilerConfig.run_dir` nicht gesetzt ist. |
| `TORCH_TPU_XPROF_SESSION_ID` | Gemeinsame ID der XProf-Profiling-Sitzung für Worker auf mehreren Hosts. |
| `TORCH_TRACE` | Ausgabeverzeichnis für strukturierte Trace-Logs von Compiler und Dispatcher (`tlparse`). |
<!-- mdformat on -->

### **Konfigurationsbeispiele für Umgebungsvariablen** { #environment-variable-configuration-snippets }

#### **1. C++-Stacktraces (`TORCH_SHOW_CPP_STACKTRACES`)** { #1-c-stacktraces-torch_show_cpp_stacktraces }

Wenn Sie `TORCH_SHOW_CPP_STACKTRACES=1` vor dem Import von `torch` setzen,
lassen sich Fehler tief im C++-Backend oder in XLA viel leichter
diagnostizieren. Statt eines einfachen Python-`RuntimeError` erhalten Sie
einen detaillierten `C++
CapturedTraceback`, der die genauen Ausführungsframes zeigt.

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

#### **2. Interne Debug-Prüfungen und Bereichsvalidierung (`TORCH_TPU_ENABLE_DEBUG_CHECKS`)** { #2-internal-debug-checks-boundary-validation-torch_tpu_enable_debug_checks }

Standardmäßig überspringt die TPU Bereichsprüfungen zugunsten der
Geschwindigkeit. Mit `TORCH_TPU_ENABLE_DEBUG_CHECKS=1` finden Sie subtile Bugs
im Zusammenhang mit Indizes außerhalb des gültigen Bereichs (etwa in
`torch.take`). Durch die zusätzliche Synchronisierung zwischen CPU und TPU
wird die Ausführung jedoch langsamer. Operationen, die sonst unbemerkt
fehlschlagen oder Daten beschädigen würden, lösen dann eine explizite
Ausnahme aus, etwa `take(): expected indices to be
in range`:

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

#### **3. XLA-IR / HLO ausgeben (`XLA_FLAGS`)** { #3-dumping-xla-ir-hlo-xla_flags }

Mit `XLA_FLAGS` können Sie den Compiler zwingen, den Text des High-Level
Optimizer (HLO) in ein Verzeichnis auszugeben. Das Dump-Verzeichnis (z. B.
`/tmp/hlo_dump`) enthält dann:

*   **`*.after_optimizations.txt`**: Der finale HLO-Graph mit den Operationen
    auf Hardwareebene.
*   **`*.flagfile`**: Eine vollständige Liste der aktiven XLA-Compiler-Flags
    (zur Prüfung von `xla_optimization_level`).
*   **`*tpu_comp_env.txt`**: Interne TPU-Umgebungseinstellungen, die bei der
    Kompilierung verwendet wurden.

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

#### **4. Mehrstufiges Kompilierungscaching (`TORCH_TPU_TIER2_COMPILATION_CACHE` und `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`)** { #4-multi-tiered-compilation-caching-torch_tpu_tier2_compilation_cache-torch_tpu_tier3_compilation_cache_root }

TorchTPU verwendet mehrstufiges Caching, um redundante Kompilierungen über
Ausführungen und Ranks hinweg zu vermeiden:

*   **Stufe 2 (`TORCH_TPU_TIER2_COMPILATION_CACHE`)** wird lokal im gemeinsamen
    Speicher des Hosts abgelegt. Erwartet wird ein Verzeichnis-**Name** (z. B.
    `"tpu_tier2_cache"`), den das Backend automatisch unter
    `/dev/shm/torch_tpu_cache/<name>` anlegt.
*   **Stufe 3 (`TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`)** ist ein persistenter
    Kompilierungscache, der von Hosts und Sitzungen gemeinsam genutzt wird
    (typischerweise ein Bucket-URI mit `gs://` oder ein Pfad in einem
    gemeinsamen Dateisystem).

Kompilierte binäre Artefakte (ausführbare Dateien `*.bin`) werden asynchron in
den Cache-Verzeichnissen gespeichert und in nachfolgenden Ausführungen
wiederverwendet. Das verringert die Latenz der JIT-Kompilierung erheblich:

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
