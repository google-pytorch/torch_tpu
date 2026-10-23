<p align="center">
 <img src="images/torchtpu_banner.png" alt="TorchTPU" style="width: 90%;">
</p>
<p align="center">
<a href="https://github.com/google-pytorch/torch_tpu"><b>GitHub</b></a> | <a href="https://discuss.pytorch.org/c/tpu/45"><b>Diskussionsforum</b></a> | <a href="https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu"><b>PyTorch auf TPU</b></a>
</p>

## Über TorchTPU { #about }

TorchTPU ist ein natives Open-Source-Backend für PyTorch auf Google Cloud TPUs,
das Google und Meta gemeinsam entwickelt haben. Mit TorchTPU können Sie
vorhandene PyTorch-Modelle mit minimalen Codeänderungen auf TPU-Hardware
ausführen und Trainings- und Inferenz-Arbeitslasten von einem einzelnen Chip bis zu
TPU-Pods mit mehreren Slices skalieren.

## Was möchten Sie heute tun? { #what-are-you-trying-to-do-today }

<!-- mdformat off(reason: preserve MkDocs grid card comments and 4-space list indentation) -->
<!--mkdocs:<div class="grid cards two-columns" markdown>-->

- **<!--mkdocs::material-rocket-launch:{ .lg .middle }--> Erste Schritte**

    Stellen Sie Umgebungen bereit, lernen Sie die grundlegenden Architekturen kennen und führen Sie Ihre ersten Arbeitslasten aus.

    - [<!--mkdocs::octicons-arrow-right-24:--> Einführung in TorchTPU](./intro_to_torchtpu.md): Warum TorchTPU entwickelt wurde, was es bietet und wie es Ihnen helfen kann.
    - [<!--mkdocs::octicons-arrow-right-24:--> Erste Schritte mit TorchTPU auf einer VM](./get_started_on_a_vm.md): TorchTPU auf einer TPU-VM einrichten.
    - [<!--mkdocs::octicons-arrow-right-24:--> Erste Schritte mit TorchTPU auf GKE](./get_started_on_gke.md): TorchTPU über GKE verwenden.
    - [<!--mkdocs::octicons-arrow-right-24:--> Unterstützte Features und Roadmap](./feature_set.md): Derzeit unterstützte Features und Pläne für künftige Releases.
    - [<!--mkdocs::octicons-arrow-right-24:--> API-Referenz](./api_reference.md): Die TorchTPU-API und die Umgebungsvariablen für das aktuelle Release.
    - [<!--mkdocs::octicons-arrow-right-24:--> TorchTPU-FAQ](./faq.md): Häufig gestellte Fragen zu TorchTPU.

- **<!--mkdocs::material-sine-wave:{ .lg .middle }--> Architektur und Kernkonzepte im Detail**

    Erfahren Sie im Detail, wie Speicherlayouts, Ausführungs-Engines und verteilte Ausführung funktionieren.

    - [<!--mkdocs::octicons-arrow-right-24:--> Funktionsweise von TorchTPU](./how_torchtpu_works.md): Wie TorchTPU von oben bis unten mit PyTorch und der TPU-Hardware zusammenarbeitet.
    - [<!--mkdocs::octicons-arrow-right-24:--> Das Ausführungsmodell von TorchTPU](./torchtpu_execution_model.md): Ausführliche Details dazu, wie TorchTPU PyTorch-Code ausführt, und zu den unterstützten Ausführungsmodi.
    - [<!--mkdocs::octicons-arrow-right-24:--> Das Speichermodell von TorchTPU](./torchtpu_memory_model.md): Wie TorchTPU mit der Art umgeht, wie die TPU und der XLA-Compiler Speicher verwalten.
    - [<!--mkdocs::octicons-arrow-right-24:--> Unterstützung für verteiltes Rechnen](./distributed_support.md): Wie TorchTPU verteilte Arbeitslasten unterstützt.
    - [<!--mkdocs::octicons-arrow-right-24:--> Unterstützung für eigene Kernels](./custom_kernels.md): Wie Sie eigene Kernels für TorchTPU schreiben.
    - [<!--mkdocs::octicons-arrow-right-24:--> Dynamische Shapes](./dynamic_shapes.md): Unterstützung für dynamische Tensor-Shapes.

- **<!--mkdocs::material-book-open-variant:{ .lg .middle }--> Fallstudien**

    Sehen Sie sich praktische Beispiele dafür an, wie Sie mit TorchTPU reale Probleme lösen.

    - [<!--mkdocs::octicons-arrow-right-24:--> FLUX-Optimierung](./case_studies_flux.md): Ein Beispiel, das zeigt, wie Sie mehrere Optimierungen für FLUX finden und anwenden.
    - [<!--mkdocs::octicons-arrow-right-24:--> Muse Glimmer 30B: Inferenz](./case_studies_muse_glimmer_30b_inference.md): Ein Beispiel für iteratives Decoding mit echten Gewichten und statischem oder dynamischem KV-Cache.
    - [<!--mkdocs::octicons-arrow-right-24:--> Muse Glimmer 30B: Training](./case_studies_muse_glimmer_30b_training.md): Eine Trainingsschleife für eine eigenständige Implementierung von Muse Glimmer 30B.
    - [<!--mkdocs::octicons-arrow-right-24:--> FSDP verwenden](./case_studies_fsdp.md): Der empfohlene Ansatz für FSDP mit TorchTPU.

- **<!--mkdocs::material-book-open-variant:{ .lg .middle }--> Arbeitslast analysieren**

    - [<!--mkdocs::octicons-arrow-right-24:--> Numerische Genauigkeit und Parität](./numerical_precision.md): Genauigkeit in TorchTPU steuern.
    - [<!--mkdocs::octicons-arrow-right-24:--> Performance mit XProf profilieren](./performance_profiling_with_xprof.md): So verwenden Sie XProf als Profiler für TorchTPU.

<!--mkdocs:</div>-->
<!-- mdformat on -->
