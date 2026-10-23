# Funktionsweise von TorchTPU { #how-torchtpu-works }

## Grundlagen zu TPUs { #tpu-primer }

[Google TPUs](https://cloud.google.com/tpu) sind spezialisierte ASICs, die
gezielt für die massiven Tensor-Operationen entwickelt wurden, auf denen Deep
Learning beruht. Die
[Architektur von TPUs unterscheidet sich](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)
zwischen den Generationen. Grob gesagt enthält ein TPU-Chip Folgendes:

*   Einen oder mehrere TensorCores. Jeder davon ist ein Single-Thread-Prozessor
    mit Matrixmultiplikationseinheiten (MXUs), einer Vektoreinheit (VPU) und
    einer Skalareinheit.
*   Einen oder mehrere SparseCores. Sie sind auf Sparse-Operationen
    spezialisiert, und der TensorCore kann Operationen an sie auslagern.
*   Einen großen High Bandwidth Memory (HBM), 288 GB bei 8,6 TB/s in
    [TPU 8i](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive).
    Er kann ML-Daten speichern und mit der CPU teilen.
*   Bidirektionalen Inter-Chip Interconnect (ICI), 19,2 TBps in
    [TPU 8t und 8i](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/).
    Er ermöglicht schnelle Kommunikation zwischen TPU-Chips innerhalb von
    benutzerkonfigurierten
    [TPU-Slices](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices).
    Das sind logische Untergruppen eines
    [TPU-Pods](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu-pod),
    der aus einer vorkonfigurierten Anzahl von TPU-Chips besteht, die in einer
    3D-Torus-Konfiguration (bei TPU 7, TPU 8t) oder einer Boardfly-Konfiguration
    (TPU 8i) verbunden sind.
*   Bidirektionales Data-Center-Network (DCN), 400 GBps in
    [TPU 8t](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/).
    Es ermöglicht die Kommunikation zwischen TPU-Chips über Slices hinweg.

TPUs werden mit dem [XLA-Compiler](https://github.com/openxla/xla)-Stack über
zwei verschiedene Einstiegspunkte programmiert:
[StableHLO](https://openxla.org/stablehlo/spec) und
[Mosaic](https://docs.jax.dev/en/latest/pallas/design/design.html#lowering-pallas-to-mosaic-for-tpu).
Die eigentliche TPU-ISA ist proprietär, kann sich ändern und ist nicht
öffentlich verfügbar. Im Grunde können Sie sich StableHLO und (noch mehr)
Mosaic als Entsprechung zu PTX vorstellen. Die folgende Abbildung zeigt eine
Übersicht:

![Funktionsweise von TorchTPU](./images/torch_tpu_inputs_to_tpu.svg)

Ein C++-Programm kann auf eine lokale TPU zugreifen, indem es einen
[PjRT-Client](https://openxla.org/xla/pjrt/cpp_api_overview) instanziiert.
Dessen API bietet Methoden, um MLIR (StableHLO oder Mosaic) in eine
TPU-Executable zu kompilieren, diese Executable zu starten und Daten zur TPU
und von der TPU zu kopieren. Beachten Sie, dass die TPU-Ausführung asynchron
ist. Die CPU kann auf ein TPU-Ergebnis warten, indem sie auf den zugehörigen
PjRTBuffer wartet. StableHLO folgt dem
[SSA-Paradigma](https://en.wikipedia.org/wiki/Static_single-assignment_form)
und erlaubt daher keine In-place-Mutationen. Eine Schreiboperation erzeugt also
immer einen neuen Wert (zumindest konzeptionell). Das ist besonders wichtig für
TorchTPU, das die vertraute „View“-Semantik von PyTorch einschließlich
Mutationen durch Zerlegung und Umkehrung von Views bereitstellt. Der
XLA-Compiler geht davon aus, dass Tensor-Shapes zur Kompilierzeit bekannt sind.
In TorchTPU wird diese Einschränkung durch
[Bounded Dynamism](./dynamic_shapes.md) gelockert. Da wir dies als
Einschränkung für PyTorch-Arbeitslasten erkannt haben, fügen wir Unterstützung für
Bounded Dynamism hinzu. Dabei sind die Maximalwerte der Dimensionen einer
Shape zur Kompilierzeit bekannt, die tatsächlichen Dimensionen aber erst zur
Laufzeit.

## TorchTPU: ein Out-of-Tree-Backend mit Autoload-Unterstützung { #torchtpu-an-out-of-tree-backend-with-autoload-support }

Gemäß der
[aktuellen Best Practice](https://docs.pytorch.org/tutorials/advanced/extend_dispatcher.html),
nach der zusätzliche Backends außerhalb des Quellbaums (Out-of-Tree) erstellt
und zur Laufzeit registriert werden, erweitern wir den PyTorch-Dispatcher um ein
neues C++-Backend, das als `PrivateUse1` registriert ist. Außerdem haben wir
den Standardmechanismus `at::Tensor` genutzt, um eine gerätespezifische
Implementierung des Tensor-Speichers zu definieren (mit dem
`c10::UniqueVoidPtr` in `c10::StorageImpl`).

Bei der Entwicklung von TorchTPU stand die Benutzerfreundlichkeit im
Vordergrund. Wenn Sie PyTorch auf einem TPU-Rechner importieren, auf dem das
TorchTPU-Wheel installiert ist, lädt Python das TorchTPU-Wheel automatisch.
Dabei wird das Backend eingerichtet und der Name `PrivateUse1` durch `tpu`
ersetzt. Ein TorchTPU-Programm unterscheidet sich dadurch in einfachen
Beispielen nicht von den In-Tree-Backends, und ein PyTorch-Programm kann mit
wenigen oder gar keinen Änderungen auf der TPU laufen:

```python
import torch
import torch.distributed as dist

def main():
    device = torch.device("tpu")
    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    word_size = dist.get_world_size()
    ...
    model = model.to(device)
    ...
```

Da die TPU zum aktuellen Beschleuniger wird und `tpu_dist` das
Standard-Backend für TorchTPU ist, müssen Sie beides nicht explizit angeben:

```python
import torch
import torch.distributed as dist

def main():
    device = torch.accelerator.current_accelerator()
    dist.init_process_group()
    rank = dist.get_rank()
    word_size = dist.get_world_size()
    ...
    model = model.to(device)
    ...
```

## Graph erfassen { #capturing-the-graph }

TorchTPU tract Eager-Ops auf ATen-Ebene und nutzt dafür die
Op-Dispatch-Hooks von PyTorch, die Teil der Backend-Infrastruktur sind.
TorchTPU-Ops werden als StableHLO-Programme erstellt und einzeln oder in
Gruppen an XLA übergeben. So kann der Compiler Operationen umfassend
fusionieren und die Hardware effektiver nutzen. Wenn TorchTPU einen
Callback-Aufruf vom PyTorch-Dispatcher erhält, geht die Laufzeit so vor:

*   Sie erstellt eine Deferred-Op-Instanz, die Folgendes enthält:
    *   Ein StableHLO-Builder-Objekt, das beim Aufruf eine StableHLO-Funktion
        zurückgibt, die die Semantik von `aten::mul` implementiert, wie unten
        gezeigt (implizites Broadcasting und dtype-Promotion werden an anderer
        Stelle behandelt)

```
func.func @main(%arg0: tensor<4x16xbfloat16>, %arg1: tensor<4x16xbfloat16>) -> tensor<4x16xbfloat16> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<4x16xbfloat16>
  return %0 : tensor<4x16xbfloat16>
}
```

*   Eine optionale Zuordnung statischer Op-Parameter (z. B. training und
    momentum für `aten::native_batch_norm`), die zur Berechnung eines
    Cache-Schlüssels verwendet wird;
*   Sie gibt einen neuen `at::Tensor` zurück, dessen `c10::StorageImpl` (über
    eine Indirektion) auf diese Deferred-Op-Instanz zeigt.

Sobald ein geeigneter Synchronisationspunkt erreicht ist, der vom
Ausführungsmodus abhängt, geht TorchTPU so vor:

*   Es analysiert den Bereich der Ops;
*   Es bestimmt, welche Ops in ausführbaren Code materialisiert werden;
*   Es erstellt einen Schlüssel für den Kompilierungscache des Graphen;
*   Wenn der Eintrag bereits im Cache vorhanden ist, ruft es die Executable ab
    und startet sie auf der TPU;
*   Wenn der Eintrag nicht im Cache vorhanden ist, kompiliert es das Objekt
    asynchron, fügt es dem Cache hinzu und startet die Ausführung;
*   Gegebenenfalls wartet es, bis der Wert bereit ist, und kopiert ihn zurück,
    falls das für den erreichten Synchronisationspunkt angemessen ist.

Weitere Details zu den verschiedenen Ausführungsmodi finden Sie unter
[Ausführungsmodell von TorchTPU](./torchtpu_execution_model.md).

## Der Kompilierungscache von TorchTPU { #the-torchtpu-compilation-cache }

Um unnötige Kompilierungen zu vermeiden, verwendet TorchTPU einen
mehrstufigen Kompilierungscache:

*   **Stufe 1:** Prozesslokaler In-Memory-Cache, mit dem Kompilierungen
    innerhalb desselben Prozesses geteilt werden. Das ist die schnellste der
    drei Stufen. Sie ist immer aktiviert.
*   **Stufe 2:** Rechnerlokaler gemeinsamer Speicher, der das Teilen zwischen
    Prozessen auf demselben Host ermöglicht. Das ist nützlich für Training und
    Inferenz auf mehreren Geräten. Stufe 2 ist bei verteilten Programmen
    standardmäßig aktiviert. Bei Programmen auf einem einzelnen Gerät bietet
    sie keinen Vorteil und ist daher deaktiviert.
*   **Stufe 3:** Persistenter Cache auf Basis eines Remote-Dateisystems (z. B.
    Google Cloud Storage), der die Wiederverwendung über Hosts, Programme und
    Job-Neustarts hinweg ermöglicht. Das ist nützlich, wenn Sie TPU-Programme in
    Ausführungsumgebungen wie Colab und Rechenzentren ausführen. Diese Stufe ist
    standardmäßig deaktiviert, kann aber durch Angabe eines Speicherorts
    aktiviert werden.

Jeder Eintrag ist eine kompilierte TPU-Executable, die einen
StableHLO-Berechnungsgraphen darstellt, der aus einer ATen-Op oder einer Folge
von ATen-Ops erzeugt wurde. Die StableHLO-Graphen werden durch Fingerprints
dargestellt. Diese kombinieren die Struktur des Graphen, die Tensor-Shapes der
Ein- und Ausgaben sowie die Compiler-Optionen einschließlich der Versionen der
TPU-Hardware, von libtpu und der PjRt-Laufzeit.

## Unterstützung für `torch.compile` { #torchcompile-support }

Damit die Nutzung konsistent mit anderer Hardware bleibt, die
`torch.compile()` unterstützt, registrieren wir ein neues TorchDynamo-
[Backend](https://docs.pytorch.org/docs/stable/torch.compiler_custom_backends.html),
das Sie so verwenden können: `compiled_model = torch.compile(model,
backend="tpu")`.

Dynamo ruft das TorchTPU-Backend auf und übergibt dabei ein
`torch.fx.GraphModule` sowie eine Liste von Beispiel-Eingabe-Tensoren. Die
Laufzeit tract den übergebenen Graphen dann so, als befände sie sich im
Eager-Modus, wobei aber alle Ops verzögert werden und nichts auf der TPU
ausgeführt wird. Die Laufzeit erzeugt und kompiliert ein einziges
StableHLO-Programm für den gesamten FX-Graphen und gibt ein aufrufbares
Python-Objekt zurück, das den Aufruf kapselt. Beachten Sie, dass TorchTPU
`TorchInductor` nicht verwendet. Stattdessen nutzt es den ausgereiften
XLA-Compiler-Stack, der zusammen mit der TPU-Architektur entwickelt wurde.

## Weitere Details { #more-detail }

In den folgenden Dokumenten gehen wir näher auf diese Themen ein:

*   Das [Speichermodell](./torchtpu_memory_model.md), das die TPU unterstützt,
    und seine Auswirkungen auf PyTorch;
*   Die [Ausführungsmodi](./torchtpu_execution_model.md), die TorchTPU
    unterstützt;
*   Die Unterstützung für [eigene Kernels](./custom_kernels.md);
*   [Verteilte Ausführung](./distributed_support.md) auf TPU-Clustern.
