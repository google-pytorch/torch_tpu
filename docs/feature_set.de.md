# Unterstützte Features von TorchTPU und künftige Arbeiten { #torchtpu-supported-feature-set-and-future-work }

## Derzeit unterstützte Features { #current-supported-feature-set }

Die folgende Tabelle zeigt unsere aktuelle Feature-Abdeckung zum Zeitpunkt des
ersten Open-Source-Releases.

Damit Sie TorchTPU für Ihre Arbeitslasten besser bewerten können, teilen wir die
Features in zwei Stufen ein:

*   **Validiert:** Funktionen mit durchgängigen, kontinuierlichen Tests,
    automatisierter CI/CD-Abdeckung und Benchmark-Prüfung für gängige
    Modellarchitekturen.
*   **Experimentell:** Implementierungen in einem frühen Stadium, neue APIs oder
    Prototyp-Features. Sie sind funktionsfähig und für frühes Feedback
    verfügbar, ihr Verhalten oder ihre APIs können sich aber aufgrund von
    Rückmeldungen aus der Community noch ändern.

> **Note:** TorchTPU entwickelt sich offen und schnell weiter. Den aktuellen
> Stand finden Sie auf unserem
> [GitHub](https://github.com/google-pytorch/torch_tpu). Dort können Sie auch
> bestimmte Features anfragen.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
|  | **Validiert** | **Experimentell** |
| ----- | ----- | ----- |
| **Zusammenarbeit mit der PyTorch-Community** | <ul><li>OSS-Release für TorchTPU, vLLM, SGLang, TorchTitan, TorchRec</li><li>Externe PRs werden angenommen</li><li>Dokumentation / Tutorials</li><li>CI/CD</li><li>Colab</li></ul> | <ul><li>TPU-Agent für die ATEN-Migration</li></ul> |
| **TPU-Unterstützung** | <ul><li>v5e, v5p, v6e, v7</li></ul> |  |
| **Grundfunktionen** | <ul><li>Einfache Migration</li><li>Gemischt Eager \+ kompiliert</li><li>FW \+ BW</li><li>Torch.profile</li><li>Dynamische Ops (mit Neukompilierung)</li></ul> | <ul><li>SymInts</li><li>Jagged Tensors</li></ul> |
| **Modellarchitekturen** | <ul><li>Transformer (Dense \+ MoE): Qwen, Llama, Muse Glimmer, Gemma</li><li>Diffusion: WAN, FLUX</li><li>CNN</li></ul> | <ul><li>GLM, Kimi, DeepSeek, GPT OSS und viele mehr (schrittweise Optimierung)</li><li>Mamba</li><li>LSTM/GRU</li><li>DLRM</li></ul> |
| **Kernel-Unterstützung** | <ul><li>Pallas</li><li>Tokamax</li></ul> | <ul><li>Helion</li></ul> |
| **Quantisierung** | <ul><li>FP8/FP4 für den gesamten Tensor</li><li>FP8 pro Kanal</li></ul> |  |
| **Verteilt** | <ul><li>Optimales Mesh</li><li>SimpleFSDP</li><li>Checkpointing</li><li>Activation Offloading</li><li>Multi-Slice bis 8K Chips</li></ul> | <ul><li>FSDPv2 (geringe Performance)</li><li>MPMD (geringe Performance)</li><li>TorchComm</li></ul> |
| **Integrationen ins Ökosystem** | <ul><li>Hugging Face</li><li>Ray</li><li>vLLM / SGLang</li><li>TorchTitan</li><li>TorchRec</li></ul> |  |
| **Produktionsreife Lösungen** | <ul><li>Inferenz (Multi-Host, P/D-Disaggregation)</li><li>Fine-Tuning (bis \~8K Chips)</li></ul> |  |
<!-- mdformat on -->

## Künftige Arbeiten { #future-work }

Dieser Abschnitt beschreibt unsere künftigen Entwicklungsziele und
Erkundungen. Er ist kein verbindlicher Release-Zeitplan.

Als Open-Source-Projekt, bei dem die Community an erster Stelle steht, richten
wir unsere Entwicklungsprioritäten bewusst nach dem Feedback der Community und
der Akzeptanz bei Entwicklern aus. Features, Zeitpläne und Architekturansätze
können sich ändern, wenn wir erfahren, welche Arbeitslasten für Sie am
wichtigsten sind.

Wir gliedern unsere anstehenden Arbeiten in zwei grobe Zeithorizonte:

*   **Kurzfristiger Schwerpunkt:** Bereiche mit aktivem Design, laufendem
    Prototyping und Performance-Härtung, die das Kernteam derzeit priorisiert.
*   **Künftige Erkundung:** Langfristige Architekturerkundungen, mögliche
    Integrationen ins Ökosystem und von der Community gewünschte Backlog-Punkte,
    die wir aktiv untersuchen.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
|  | **Kurzfristiger Schwerpunkt** | **Künftige Erkundung** |
| ----- | ----- | ----- |
| **Zusammenarbeit mit der PyTorch-Community** | <ul><li>Mehr Dokumentation / Tutorials / Blogs</li></ul> | <ul><li>TPU-Agent für die Kernel-Migration</li></ul> |
| **TPU-Unterstützung** | <ul><li>v8i, v8t</li></ul> |  |
| **Grundfunktionen** | <ul><li>Streams</li><li>Graph Capture</li><li>Eager-Modus mit schneller Startzeit</li><li>Dynamische Ops (ohne Neukompilierung)</li></ul> |  |
| **Modellarchitekturen** | <ul><li>Modellunterstützung ab Tag 1</li></ul> |  |
| **Quantisierung** | <ul><li>Blockweises FP8/FP4 mit torchao (Spitzen-Performance auf TPUv8t)</li></ul> |  |
| **Verteilt** | <ul><li>FSDPv2 (mit Performance)</li><li>MPMD (mit Performance)</li><li>Training \> 8K Chips</li><li>Mehrstufiges Checkpointing</li><li>Goodput</li></ul> | <ul><li>Training >> 8K Chips</li></ul>  |
| **Integrationen ins Ökosystem** | <ul><li>VeRL</li></ul> |  |
| **Produktionsreife Lösungen** | <ul><li>RL</li></ul> | <ul><li>Pre-Training</li></ul> |
<!-- mdformat on -->
