# TorchTPU-FAQ { #torchtpu-faq }

## Was ist der Unterschied zwischen PyTorch/XLA und TorchTPU? { #what-is-the-difference-between-pytorchxla-and-torchtpu }

PyTorch/XLA diente bisher als Brücken-Framework, das stark von
JAX-Laufzeitkomponenten und speziellen APIs abhing. TorchTPU bietet ein natives
PyTorch-Backend für [Google Cloud TPUs](https://cloud.google.com/tpu) ohne
Abhängigkeiten von der JAX-Laufzeit. Vorhandene PyTorch-Arbeitslasten laufen sofort
mit trivialen Codeänderungen (z. B. `.to("cuda")` → `.to("tpu")`).

## Was ist der Unterschied zwischen TorchAX und TorchTPU? { #what-is-the-difference-between-torchax-and-torchtpu }

TorchAX (auch als torch_xla2 bekannt) ist eine Implementierung, die
PyTorch-Operationen auf Python-Ebene abfängt und nach XLA lowert (oft über
JAX). TorchAX wurde zwar viel für Inferenz-Benchmarks und Prototyping
verwendet, aber TorchTPU ist der neue offizielle Standard für eine direkt
einsetzbare, PyTorch-native Erfahrung auf TPUs.

## Was passiert mit PyTorch/XLA? { #what-is-happening-to-pytorchxla }

PyTorch/XLA ist veraltet und wird durch
[TorchTPU](https://github.com/google-pytorch/torch_tpu) ersetzt, das jetzt
öffentlich verfügbar ist.

## Ersetzt TorchTPU JAX? { #is-torchtpu-replacing-jax }

Nein. TorchTPU ergänzt das TPU-Ökosystem um einen nativen PyTorch-Einstieg in
Cloud TPUs. Entwickler mit vorhandenen PyTorch-Codebasen können damit ohne oder
mit nur minimalen Änderungen direkt auf TPUs arbeiten. Sie müssen Modelle weder
in JAX neu schreiben noch getrennte Codebasen für GPU- und TPU-Infrastruktur
pflegen.

*   **Wann kommt TorchTPU infrage?** Wenn Sie eine vorhandene PyTorch-Codebasis
    haben, Portabilität über mehrere Hardwaretypen (GPU/TPU) ohne Refactoring
    möchten, PyTorch-native Frameworks verwenden (z. B. TorchTitan, vLLM,
    SGLang, Hugging Face) oder schnell von GPU zu TPU migrieren möchten.

## Wie kann ich am besten mit den TorchTPU-Entwicklern in Kontakt treten? { #what-is-the-best-way-for-me-to-connect-with-torchtpu-developers }

Sie können Feedback per E-Mail an
[torchtpu-feedback@google.com](mailto:torchtpu-feedback@google.com) senden.
Beiträge direkt im Repository sind willkommen:
[https://github.com/google-pytorch/torch_tpu](https://github.com/google-pytorch/torch_tpu).
Sie können auch das Tag „TPU“ auf
[discuss.pytorch.org](https://discuss.pytorch.org/) verwenden.

## Wird es Community-Foren geben? { #will-there-be-any-community-engagement-forums }

Ja, der Austausch mit der Community ist uns sehr wichtig. Wir werden Details zu
solchen Foren bekannt geben. Außerdem planen wir monatliche Sprechstunden.

## Welche Unterstützung bietet TorchTPU für die Kernel-Entwicklung? { #what-support-does-torchtpu-provide-for-kernel-development }

Siehe [Unterstützung für eigene Kernels in TorchTPU](./custom_kernels.md).

## Welche TPU-Generationen werden aktiv unterstützt? Wie sieht der Plan für künftige Hardware aus? { #which-tpu-generations-are-actively-supported-whats-the-plan-for-future-hardware-support }

TorchTPU unterstützt aktiv v5e, v5p, Trillium (TPU v6e) und Ironwood (TPU v7x).
Die Validierung und Optimierung für TPU v8 und künftige Chips läuft im Rahmen
unserer Roadmap.

## Wo finde ich die unterstützten Features und die künftige Roadmap von TorchTPU? { #where-can-we-find-torchtpus-supported-features-future-roadmap }

Unter [Unterstützte Features](./feature_set.md) finden Sie den aktuellen und
den geplanten Feature-Umfang.

## Gibt es einen Agenten, der mir das Onboarding erleichtert? { #is-there-an-agent-to-make-my-onboarding-easier }

Probieren Sie die TPU Agent Suite aus (experimentell):
[https://github.com/google-pytorch/tpu-agent-suite](https://github.com/google-pytorch/tpu-agent-suite)

Die TPU Agent Suite ist ein Open-Source-Plug-in mit spezialisierten Sub-Agenten,
die Onboarding, Diagnose und die Optimierung der Inferenzleistung für
PyTorch-Modelle auf Cloud TPU v6e und v7x automatisieren.

## Kann ich TorchTPU mit unserer eigenen Serving-Lösung verwenden? { #can-i-use-torchtpu-with-our-custom-serving-solution }

Ja. TorchTPU bietet sofort einsatzbereite Integrationen mit branchenüblichen
Serving-Engines wie vLLM und SGLang. Eigene oder interne Serving-Lösungen können
TorchTPU aber auch über native PyTorch-Ausführungs-APIs und Compiler-Backends
auf niedrigerer Ebene einbinden. Für erweiterte Orchestrierung können Kunden
außerdem eigene Serving-Setups mit llm-d auf GKE bereitstellen.
