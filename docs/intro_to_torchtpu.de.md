# Einführung in TorchTPU { #introduction-to-torchtpu }

<!-- Link to colab and to generated notebook -->

[![In Colab öffnen](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/intro_to_torchtpu.ipynb)

TorchTPU ist ein natives Open-Source-Backend für PyTorch auf
[Google Cloud TPUs](https://cloud.google.com/tpu), das Google und das
PyTorch-Team bei Meta gemeinsam entwickelt haben. Mit TorchTPU können
Entwickler vorhandene PyTorch-Arbeitslasten mit minimalen Codeänderungen auf
TPU-Hardware ausführen und Trainings- und Inferenz-Arbeitslasten von einem
einzelnen Chip bis zu TPU-Pods mit mehreren Slices skalieren.

## Warum TorchTPU? { #why-use-torchtpu }

Google TPUs sind anwendungsspezifische integrierte Schaltungen (ASICs), die
eigens für die Matrix- und Vektoroperationen entwickelt wurden, auf denen Deep
Learning beruht. Sie bieten hohen Durchsatz, Energieeffizienz und eine gute
Performance pro Dollar bei großem Maßstab. TPUs wurden bisher vor allem mit den
Ökosystemen von JAX und TensorFlow eingesetzt. TorchTPU bringt native
TPU-Unterstützung direkt in das
[PyTorch-Ökosystem](https://github.com/google-pytorch/torch_tpu) und wurde
bewusst nach den folgenden drei Leitprinzipien entworfen:

*   **Vollwertiger Teil von PyTorch:** TorchTPU ist direkt in PyTorch
    integriert und hat keine Abhängigkeiten von der JAX-Laufzeit. Es behält die
    Standardsemantik von PyTorch für Tensor-Views und In-place-Mutationen bei,
    übernimmt das Ausführungsmodell von PyTorch mit einem Prozess pro Gerät und
    unterstützt `torch.compile` über Dynamo.
*   **Sofort einsatzbereit:** Um eine Arbeitslast von einer GPU auf eine TPU zu
    portieren, ändern Sie zunächst `device="cuda"` in `device="tpu"` und
    `backend="nccl"` in `backend="tpu_dist"`. Ihr unverändertes PyTorch-Modell
    läuft sofort im Eager-Modus, und klare Fehlermeldungen helfen bei der
    Diagnose von Laufzeit- oder Compiler-Problemen.
*   **Performance im Produktionsmaßstab:** Über die Eager-Ausführung Op für Op
    hinaus bietet TorchTPU einen Eager-Modus mit verzögerter Fusion, eine
    Compiler-Pipeline von Dynamo zu XLA, Bounded Dynamism zur Vermeidung von
    Neukompilierungen durch geänderte Shapes sowie APIs für eigene Kernels.
    So erreichen Sie eine maximale Auslastung der Hardware.

## Schnellstart: Ausführung auf der TPU { #quickstart-running-on-tpu }

In der folgenden Zelle sehen Sie ein einfaches Beispiel für ein
PyTorch-Programm, das auf einer TPU läuft. Sie können
[dieses Beispiel in Colab](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/intro_to_torchtpu.ipynb)
ausführen, wenn Sie die TPU-Laufzeit aktivieren.

```python
import torch

device = torch.device("tpu")
x = torch.randn((4, 4), device=device)
y = torch.ones((4, 4), device=device)

result = torch.matmul(x, y).cpu()
print(f"Executed on device: {x.device}")
print(f"Result shape on CPU: {result.shape}, sum: {result.sum():.4f}")
```

## Wichtige Features { #key-features }

*   **Umfassende Abdeckung von Operatoren und Modellen:** Unterstützt 450
    zentrale ATen-Operatoren (sowohl Forward- als auch Backward-Pass) und wird
    kontinuierlich mit 5.000 beliebten HuggingFace-Modellen validiert. Dazu
    gehören LLMs (Llama, Qwen, Gemma, DeepSeek, Mistral), Vision-Modelle (ViT,
    ResNet, ConvNeXt, DINOv2), Audiomodelle (Whisper, Wav2Vec2) und
    Diffusion-Pipelines (FLUX, Stable Diffusion).
*   **Drei Eager-Ausführungsmodi:**
    *   **Strict (Standard):** Sendet jeweils einen ATen-Operator asynchron an
        die TPU. Das entspricht dem Standardverhalten von PyTorch im
        Eager-Modus auf GPUs.
    *   **Debug (`TORCH_TPU_LAUNCH_BLOCKING=1`):** Führt jeden Operator synchron
        mit der CPU aus und aktiviert Bereichsprüfungen zur Laufzeit, um das
        Debugging zu vereinfachen.
    *   **DeferAndFuse (`TORCH_TPU_DEFER_AND_FUSE=1`):** Verzögert Folgen von
        ATen-Operatoren bis zu einem Materialisierungspunkt (etwa `.cpu()` oder
        `.item()`) und fusioniert sie zu optimierten Berechnungsblöcken. Das
        reduziert den Datenverkehr im High Bandwidth Memory (HBM) erheblich und
        beschleunigt die Eager-Ausführung.
*   **Graph-Kompilierung mit `torch.compile`:** Registriert das Dynamo-Backend
    `"tpu"` (`torch.compile(model)`), um ganze PyTorch-FX-Graphen nach
    [StableHLO](https://openxla.org/stablehlo/spec) zu lowern und mit dem
    [OpenXLA-Compiler](https://openxla.org/xla) zu kompilieren. Dank aktiver
    Unterstützung für dynamische Shapes (`torch._dynamo.mark_dynamic`) können
    kompilierte Graphen unterschiedliche Batchgrößen oder Sequenzlängen bis zu
    einer festgelegten Grenze verarbeiten, ohne neu zu kompilieren.
*   **Bounded Dynamism:** Verringert nativ Neukompilierungen durch variable
    Batchgrößen und Sequenzlängen, indem gegen dynamische Obergrenzen kompiliert
    wird.
*   **Mehrstufiger Kompilierungscache:** Vermeidet redundante
    XLA-Kompilierungen mit drei Cache-Stufen:
    *   **Stufe 1:** Prozesslokaler In-Memory-Cache (immer aktiviert).
    *   **Stufe 2:** Hostlokaler gemeinsamer Speicher für die
        prozessübergreifende Wiederverwendung in Jobs mit mehreren Geräten.
    *   **Stufe 3:** Persistenter Remote-Cache (etwa Cloud Storage) für die
        Wiederverwendung über Hosts und Job-Neustarts hinweg.
*   **Native verteilte Primitive:** Implementiert TPU-Collectives auf Ebene der
    `torch.distributed`-Prozessgruppe (`c10d::Backend`) über
    `backend="tpu_dist"` und unterstützt `DistributedDataParallel` (DDP),
    `FullyShardedDataParallel` (FSDP2 und SimpleFSDP) sowie `DTensor`. Mit
    `torch.tpu.distributed.topology_aware_mesh` können Sie die
    Parallelitätsdimensionen außerdem an die physischen TPU-Interconnects
    anpassen.
*   **Eigene TPU-Kernels:** Schreiben Sie eigene Kernels in
    [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) (aufrufbar über
    `torch.tpu.pallas.jax_op` mit direkter Unterstützung für RDMA-Collectives
    zwischen Chips), schreiben Sie Tile-basierte Kernels mit der
    experimentellen DSL [Helion](https://pytorch.org/blog/helion/) oder
    importieren Sie optimierte, vorgefertigte TPU-Kernels aus Tokamax.
*   **Integrationen ins Ökosystem:** Funktioniert sofort mit gängigen
    PyTorch-Frameworks für Serving und Training sowie mit Profiling-Suites,
    darunter `torch.profiler` (TensorBoard/XProf), vLLM, SGLang, TorchTitan,
    Ray und TorchRec.

## Einschränkungen und künftige Arbeiten { #limitations-and-future-work }

Wie jedes Produkt hat auch TorchTPU bekannte Einschränkungen, die im Laufe der
Zeit behoben werden. Unser erstes Release bietet zwar eine robuste, sofort
nutzbare Abdeckung für wichtige Arbeitslasten, TorchTPU ist aber ein gemeinsames,
von der Community getragenes Projekt. Unter
[Features und Roadmap](./feature_set.md) sehen Sie, was geplant ist. Sie können
auch direkt einsteigen und uns helfen, diese Einschränkungen zu beheben, indem
Sie auf GitHub [PRs einreichen](https://github.com/google-pytorch/torch_tpu).
Dies ist ein Community-Projekt. Wir freuen uns, von Ihnen zu hören, und laden
Sie ein, mit uns zusammenzuarbeiten, über
[GitHub](https://github.com/google-pytorch/torch_tpu),
[https://discuss.pytorch.org/c/tpu](https://discuss.pytorch.org/c/tpu),
[https://dev-discuss.pytorch.org/c/tpu/](https://dev-discuss.pytorch.org/c/tpu/)
und [torchtpu-feedback@google.com](mailto:torchtpu-feedback@google.com).

*   Dynamische Shapes verursachen Neukompilierungen
    *   `torch.compile` setzt in TorchTPU standardmäßig `dynamic=False`.
        `dynamic=None` oder `dynamic=True` werden für den Produktionseinsatz
        noch nicht unterstützt.
        [Weitere Informationen zu dynamischen Shapes](./dynamic_shapes.md)
    *   Wenn Sie zum Beispiel Decode- oder Generate-Schritte mit LLMs von
        Hugging Face ausführen, kommt es bei jedem Schritt zu
        Neukompilierungen, sofern Bounded Dynamism nicht aktiviert ist.
*   View-Ops können auf der TPU teuer sein (siehe
    [Speichermodell von TorchTPU](./torchtpu_memory_model.md)).
*   MPMD mit Collectives wird funktional unterstützt, erreicht aber noch nicht
    die maximale Performance. Für die Überlappung von Collectives mit
    Berechnungen gelten
    [einige Einschränkungen](./distributed_support.md):
    *   Volle Stream-Unterstützung mit SparseCore-Offload folgt in naher
        Zukunft.
    *   Verwenden Sie SimpleFSDP mit `torch.compile()` statt FSDPv2 mit Eager
        oder `torch.compile`, siehe [Best Practices für FSDP](./case_studies_fsdp.md)
    *   Der empfohlene Weg, bei kommunizierenden Modulen die maximale
        Performance aus den XLA-Optimierungen herauszuholen (also Berechnung
        und Kommunikation zu überlappen), ist der Decorator `spmd_safe` für
        Bereiche, bei denen **die Nutzer garantieren**, dass sie exakt gleich
        sind
*   TorchTPU unterstützt FlexAttention derzeit nicht nativ
*   TPUs der aktuellen Generation unterstützen keine Matmul-Akkumulation mit
    geringerer Genauigkeit als fp32 (z. B. tf32).
*   Nur ein Prozess kann auf ein TPU-Gerät zugreifen. Das schränkt die
    Unterstützung für co-lokalisiertes RL ein (ein TPU-Chip kann allerdings ein
    oder mehrere zugeordnete Geräte haben, auf die separate
    Betriebssystemprozesse zugreifen können).

**Weitere Empfehlungen:**

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Einschränkung | Empfehlung |
| :--- | :--- |
| Eager-Modi haben eine Warm-up-Zeit größer als null | Aktivieren Sie den persistenten Kompilierungscache |
| Torch.compile dauert lange | Aktivieren Sie den persistenten Kompilierungscache |
| CNNs in PyTorch verwenden standardmäßig ein Channel-First-Layout, das auf der TPU sehr ineffizient ist | Legen Sie mit dieser [PyTorch-Option](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html) ein Channel-Last-Layout fest, um die Performance zu verbessern. |
| Die Zuordnung des Device Mesh ist standardmäßig eventuell nicht optimal, weil sich die Topologie von TPUs von anderen Architekturen unterscheidet | Verwenden Sie `topology_aware_mesh` für eine optimale Zuordnung des Device Mesh und um diese Zuordnung anpassen zu können |
<!-- mdformat on -->

## Erste Schritte { #getting-started }

Diese Dokumentation ist in vier Abschnitte gegliedert, die Sie von der ersten
Einrichtung bis zur verteilten Bereitstellung in der Produktion begleiten:

*   **Erste Schritte:** Schritt-für-Schritt-Anleitungen, mit denen Sie Ihre
    Umgebung konfigurieren und Arbeitslasten auf
    [einer VM](./get_started_on_a_vm.md) oder [über GKE](./get_started_on_gke.md)
    bereitstellen.
*   **Details:** Ausführliche Erläuterungen zum
    [Speichermodell der TPU](./torchtpu_memory_model.md), zur
    [verteilten Ausführung](./distributed_support.md) und zu den verschiedenen
    [Ausführungsmodi](./torchtpu_execution_model.md).
*   **Fallstudien:** Durchgängige Anleitungen zu Optimierungen für das
    [Flux-Modell](./case_studies_flux.md), zum Training von
    [Muse Glimmer](./case_studies_muse_glimmer_30b_training.md), zur
    [LLM-Inferenz von Anfang bis Ende](./case_studies_muse_glimmer_30b_inference.md)
    und zum effektiven Einsatz von [FSDP](./case_studies_fsdp.md).
*   **Arbeitslast analysieren:**
    [Profiling](./performance_profiling_with_xprof.md) und Umgang mit der
    [numerischen Genauigkeit](./numerical_precision.md)
