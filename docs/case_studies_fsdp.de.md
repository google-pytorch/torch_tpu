# Best Practices für FSDP { #fsdp-best-practices }

Fully Sharded Data Parallel (FSDP) verteilt Modellparameter, Gradienten und den
Optimizer-Zustand per Sharding auf die datenparallelen Ranks, sodass jeder
TPU-Kern nur einen `1 / world_size`-Anteil des Modells speichert. Bevor eine
Schicht ausgeführt wird, werden ihre vollständigen Gewichte mit einem
**All-Gather** rekonstruiert. Im Backward-Pass werden die Gradienten mit einem
**Reduce-Scatter** zusammengeführt und erneut geshardet.

In diesem Leitfaden erfahren Sie, wie Sie FSDP auf TorchTPU am besten einsetzen.
Außerdem verweist er auf ausführbare Beispiele in
[`examples/fsdp`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp).

In diesem Leitfaden lernen Sie Folgendes:

1.  **Welcher FSDP-Ansatz** auf TorchTPU der richtige ist.
2.  **SimpleFSDP (empfohlen)**: FSDP, das auf Kompilierung ausgelegt ist und
    mit einem einzigen Wrapping-Aufruf auskommt.
3.  **Manuelles FSDP**: Collectives direkt in der Modelldefinition.
4.  **Best Practices**, die für jeden Ansatz gelten.
5.  **Beispiele ausführen**: Startbefehle und was jedes Beispiel prüft.

## 1. FSDP-Ansatz auswählen { #1-choosing-an-fsdp-approach }

```
On TPU, performance comes from compiling the whole training step into a single
XLA program with `torch.compile(backend="tpu")`. When the XLA compiler sees all
forward, backward, and collective operations at once, it can schedule
collectives to overlap with computation and combine small collectives when
profitable.

Choose an FSDP approach based on how well it composes with whole-graph
compilation:
```

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
    | Ansatz | Einsatzbereich | Beispiel |
    | :------- | :------------- | :------ |
    | **SimpleFSDP** (`simple_fsdp.data_parallel`) | **Empfohlener Standard.** Sie möchten FSDP oder HSDP verwenden, ohne die Modelldefinition zu ändern.| [`fsdp_to_simple_fsdp/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp) |
    | **Manuelles FSDP** (manuelle Collectives) | Der Modellcode gehört Ihnen und Sie möchten den Kommunikationscode für das Performance-Tuning direkt neben der Berechnung haben. | [`manual/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual) |
    | **PyTorch FSDPv2** (`fully_shard`) | Bestehender Code, der noch nicht migriert wurde. Er funktioniert, aber seine Laufzeit-Hooks verursachen Graph-Breaks, und für eine konkurrenzfähige Performance ist die geplante Unterstützung für Stream-Offload auf den SparseCore der TPU erforderlich.
.| [Migrationsleitfaden](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md) |
<!-- mdformat on -->

```
SimpleFSDP and manual FSDP both express the collectives as part of the traced
graph, so XLA can optimize them together with the rest of the step. PyTorch
FSDPv2 (`fully_shard`) relies on Python pre-forward and post-backward hooks.
These hooks split the step into many smaller graphs, even when the model is
compiled.

For a side-by-side comparison of compiled FSDPv2 and compiled SimpleFSDP, see
the
[FSDPv2 to SimpleFSDP tutorial](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md).
```

## 2. Empfohlen: SimpleFSDP { #2-recommended-simplefsdp }

[`simple_fsdp`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/simple_fsdp.py)
ist eine FSDP-Implementierung, die für `torch.compile` entwickelt wurde. Sie
wurde in TorchTitan eingeführt. Eine eigenständige Kopie, die nur von PyTorch
abhängt, finden Sie in `examples/fsdp/simple_fsdp.py`.

SimpleFSDP speichert jeden Parameter als geshardeten `DTensor`. Wenn auf einen
Parameter zugegriffen wird, ruft eine Parametrisierung `DTensor.redistribute`
auf, um ihn per All-Gather zusammenzuführen. Der Autograd-Backward-Pass von
`redistribute` führt das Reduce-Scatter der Gradienten aus. Es gibt keine
Laufzeit-Hooks, daher wird der gesamte Schritt ohne Graph-Breaks kompiliert. Das
gesamte Modell wird mit einem einzigen Aufruf von `simple_fsdp.data_parallel(...)`
geshardet, ohne Wrapping pro Schicht und ohne Prefetch-Tuning. HSDP wird mit
`mode="hybrid_shard"` auf einem 2D-`DeviceMesh` unterstützt.

Weitere Informationen zur Verwendung von SimpleFSDP finden Sie hier:

*   [SimpleFSDP-README](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/README.md):
    Schnellstart und verfügbare Optionen.
*   [Tutorial: Von FSDPv2 zu SimpleFSDP](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md):
    So migrieren Sie bestehenden `fully_shard`-Code.

## 3. Alternative: Manuelles FSDP { #3-alternative-manual-fsdp }

Wenn Sie die Modelldefinition selbst kontrollieren, können Sie FSDP direkt mit
manuellen Collectives (`torch.distributed._functional_collectives`) und
`torch.compile` schreiben. Jede Schicht speichert nur ihren Gewichts-Shard und
führt im Forward-Pass ein All-Gather für das vollständige Gewicht aus.
Funktionale Collectives haben Autograd-Formeln. Daher ist der Backward-Pass des
All-Gather ein Reduce-Scatter, und Sie müssen keine eigene
`torch.autograd.Function` schreiben.

So bleibt der Kommunikationscode neben dem Berechnungscode, was das
Performance-Tuning erleichtern kann. Der Nachteil: Sie müssen die
Modelldefinition ändern.

Wie Sie manuelles FSDP implementieren, erfahren Sie im
[Leitfaden zu manuellem FSDP](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual/manual_fsdp.md).

## 4. Best Practices { #4-best-practices }

Diese Empfehlungen gelten für SimpleFSDP und manuelles FSDP.

1.  **Kompilieren Sie den gesamten Trainingsschritt mit
    `torch.compile(backend="tpu")`.** Wenn Sie nicht den gesamten Schritt
    kompilieren können, kompilieren Sie mindestens Forward-Pass, Loss und
    Backward-Pass zusammen. Wenn Sie nur das Modellmodul kompilieren, entsteht
    kein einzelner Graph. Das schränkt ein, wie gut XLA die Collectives mit der
    Berechnung überlappen kann.
2.  **Verwenden Sie Activation Checkpointing für jede geshardete Schicht.**
    Umschließen Sie den Forward-Pass jeder Schicht mit
    `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`. Dadurch
    werden die nicht geshardeten Gewichte nach dem Forward-Pass jeder Schicht
    freigegeben, statt bis zum Backward-Pass im Speicher zu bleiben. So erhalten
    Sie das ZeRO-3-Speicherverhalten, das Sie von FSDP erwarten.
3.  **Sorgen Sie dafür, dass geshardete Dimensionen durch die Mesh-Größe
    teilbar sind.** Manuelles FSDP setzt das voraus. SimpleFSDP repliziert
    Parameter, die zu klein zum Sharden sind. Wählen Sie mit `shard_dim` eine
    Dimension, die sich gleichmäßig teilen lässt.
4.  **Verwenden Sie für jeden Rank einen anderen Seed, wenn Sie geshardete
    Gewichte initialisieren.** Wenn jeder Rank seinen eigenen Shard mit
    demselben Seed initialisiert, besteht das vollständige Gewicht aus demselben
    Zufallsblock, der sich über die Ranks wiederholt. Das kann die Konvergenz
    beeinträchtigen. Für reproduzierbare, unterschiedliche Shards initialisieren
    Sie jeden Rank mit einem Rank-abhängigen Seed, etwa
    `torch.manual_seed(base_seed + dist.get_rank())`.
5.  **Lassen Sie alle Ranks dasselbe Programm ausführen.** Platzieren Sie keine
    Operationen, die eine Materialisierung erzwingen (etwa `.item()`), in
    Rank-spezifischen Verzweigungen wie `if
    rank == 0`.
6.  **Verwenden Sie HSDP, wenn vollständiges Sharding über das gesamte Mesh
    nicht skaliert.** Verwenden Sie `mode="hybrid_shard"` mit einem 2D-Mesh,
    wenn Ihr ICI-Slice zu groß ist, um das Modell effektiv zu sharden, oder wenn
    das Sharding darüber zu viel Netzwerkverkehr erzeugt. Wenn Sie über mehrere
    Slices hinweg arbeiten, ist es fast immer am besten, über das langsamere DCN
    zu replizieren (datenparallel) und innerhalb jedes Slices zu sharden.
7.  **Prüfen Sie Speicher und Numerik, bevor Sie hochskalieren.** Vergleichen
    Sie mit `assert_close` gegen eine nicht geshardete oder FSDPv2-Referenz und
    messen Sie den Speicher, der über die Grenze zwischen Forward- und
    Backward-Pass hinweg belegt bleibt. Die Beispiele enthalten eine
    Paritätsprüfung (`fsdp_to_simple_fsdp_worker.py`) und einen Speichertest
    (`manual_fsdp_memory_test.py`).

## 5. Beispiele ausführen { #5-running-the-examples }

Zu jedem Beispiel gibt es ein Launcher-Skript, das auf 8 TPU-Kernen läuft (zum
Beispiel `v6e-8`):

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Beispiel | Startbefehl | Was geprüft wird |
:-------- | :------------- | :------------- |
| [`fsdp_to_simple_fsdp/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp) | `./run_fsdp_to_simple_fsdp.sh` | Kompiliertes FSDPv2 und kompiliertes <br> SimpleFSDP liefern übereinstimmende Losses <br> und Vorhersagen. Ihre Schrittdauern <br> werden ausgegeben. |
| [`manual/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual) | `./launch.sh` oder `python3 launch.py` | Gradienten und Vorhersagen von manuellem FSDP <br> stimmen mit einer CPU-Referenz auf einem einzelnen Gerät überein. |
| [`manual/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual) | `python3 manual_fsdp_memory_test.py` | Nicht geshardete Gewichte werden nach dem <br> Forward-Pass freigegeben. |
<!-- mdformat on -->
