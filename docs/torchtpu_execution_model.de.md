# Ausführungsmodell von TorchTPU { #torchtpu-execution-model }

Wenn Sie ein PyTorch-Programm mit TorchTPU ausführen, passiert im Hintergrund
vieles, um die Ausführung zu optimieren. Wenn Sie dieses Modell verstehen,
wissen Sie besser, wie Sie Ihre Nutzung von TorchTPU abstimmen und einige
besondere Stärken von TorchTPU nutzen können.

## Ausführungspipeline { #execution-pipeline }

### Deferred Op erstellen { #create-deferred-op }

Wenn eine PyTorch-Operation (kurz „Op“) im TorchTPU-Backend ausgeführt wird,
findet zunächst keine Arbeit auf der TPU selbst statt.

Stattdessen wird das logische Ergebnis der Berechnung in eine Deferred Op
(verzögerte Op) verpackt. Das ist ein Speicherobjekt auf der CPU-Seite, das
alle Informationen enthält, die zur Ausführung dieser Operation nötig sind,
unter anderem ihre Eingaben, die auszuführenden logischen Schritte sowie Shape
und Typ der Ausgaben.

Der Rest der Pipeline, der diese symbolische Deferred Op durch echte Daten
ersetzt, heißt Materialisierung.

### Op in einen Stream einreihen { #enqueue-op-on-a-stream }

TorchTPU unterstützt die Schnittstelle `torch.Stream`. Immer wenn eine
DeferredOp erstellt wird, wird sie in den aktuell aktiven Stream eingereiht,
damit sie später ausgeführt wird.

An diesem Punkt ist der aufrufende Thread fertig. Alles Weitere geschieht
asynchron im Hintergrund (außer im Launch-Blocking-Modus, mehr dazu später).

### Stream analysieren und Ausführungsplan erstellen { #analyze-stream-and-prepare-execution-plan }

Sobald es in einem Stream Arbeit gibt, übernimmt irgendwann ein
Hintergrund-Thread die Arbeit aus diesem Stream. Dieser Hintergrund-Thread
analysiert dann die Arbeit im Stream als Ganzes und berücksichtigt dabei viele
verschiedene Faktoren, darunter:

*   In wie viele einzelne Ausführungen muss die Arbeit aufgeteilt werden?
*   Was sind die Ein- und Ausgaben jedes Ausführungsblocks?
*   Welche Optimierungsstufe soll bei der Kompilierung jeweils angewendet
    werden?
*   Ist jede Berechnungseinheit schon bekannt, oder ist sie neu?
*   Auf welchem Kern (TensorCore oder SparseCore) soll diese Arbeit ausgeführt
    werden?

Weil der Hintergrund-Thread die anstehende Arbeit gemeinsam betrachtet, kann er
bessere Entscheidungen zur Ausführung treffen.

Sobald der Hintergrund-Thread die endgültigen Ausgaben jedes Blocks ermittelt
hat, markiert er diese Ausgaben als ausstehend. Damit zeigt er an, dass sie
künftig ausgeführt werden. Dann fährt er mit dem nächsten Schritt fort.

### Executables kompilieren (oder abrufen) { #compile-or-fetch-executables }

Für jede Ausführungsphase prüft der Hintergrund-Thread in einem
Kompilierungscache, ob bereits eine Executable mit den richtigen Einstellungen
kompiliert wurde. Standardmäßig ist das ein Cache im lokalen Arbeitsspeicher
(Stufe 1). Optional können Sie einen persistenten gerätelokalen Cache (Stufe 2)
oder Remote-Caches (Stufe 3) aktivieren, um die Kosten für Neukompilierungen zu
minimieren.

Bei einem Cache-Miss wird der Kompilierungsgraph der Einheit in MLIR
umgewandelt (hauptsächlich StableHLO, außer bei eigenen Kernels) und an einen
Threadpool für die Kompilierung übergeben. Dieser Threadpool kompiliert viele Kompilierungseinheiten asynchron und parallel und
gibt sie an den Hintergrund-Worker-Thread zurück, sobald sie bereit sind.

### Ausführung starten { #start-execution }

Sobald die Eingaben gesammelt sind und die Executable vorbereitet ist, wird
eine Ausführungsaufgabe in die Laufzeit von XLA,
[PjRt](https://openxla.org/xla/pjrt), eingereiht.

PjRt gibt Promises für diese künftige Ausgabe zurück. Diese werden mit den
Ausgaben verknüpft, um anzuzeigen, dass sie gerade materialisiert werden.

### (Optional) Auf den Abschluss warten { #optional-wait-for-completion }

Um festzustellen, wann die Arbeit abgeschlossen ist, können Sie ein
`torch.Event` im Stream aufzeichnen und es mit der Methode `torch.Event.query`
prüfen.

Alternativ oder zusätzlich können Sie den Stream mit einer beliebigen
„synchronize“-Anweisung blockieren (`torch.Event.synchronize`,
`torch.Stream.synchronize` oder `torch.accelerator.synchronize` für das gesamte
Gerät). Diese hält Ihren Python-Code an, bis die TPU den Abschluss meldet.

## Ausführungsmodi { #execution-modes }

Die Programmierpipeline von TorchTPU ist flexibel und unterstützt mehrere
verschiedene Ausführungsarten.

### Strikter Eager-Modus { #strict-eager-mode }

Dieser Modus, auch „Default Eager“ oder „Defer Never“ genannt, ist das
vertraute PyTorch-Verhalten, wie Sie es ohne weitere Konfiguration kennen. In
diesem Modus markiert jede Op, die an die Warteschlange angehängt wird, den
Stream sofort zur Ausführung. Dadurch verarbeitet der Hintergrund-Worker für
die Materialisierung die Ops einzeln nacheinander.

Diese Ausführungsgraphen mit nur einer Op bieten wenig Optimierungspotenzial.
Sie werden daher mit der niedrigeren Kompilierungsstufe `-O1` kompiliert, um
die Kompiliergeschwindigkeit zu maximieren. Da es viele einzelne Kompilierungen
gibt, lassen sie sich sehr effektiv über den Threadpool für die Kompilierung
parallelisieren, und einzelne Operationen werden sehr häufig wiederverwendet.
Das führt zu minimalem Kompilierungsaufwand beim „Warm-up“ des
Kompilierungscaches.

Beachten Sie: Dieser Modus führt Ops zwar sofort und kontinuierlich aus, aber
dennoch asynchron. Wenn Sie die Ergebnisse sehen möchten, ist weiterhin eine
Synchronisation nötig, um auf den Abschluss der TPU zu warten. Laufzeitfehler
können außerdem viel später auftreten als die Op, die sie verursacht hat.

### Debug-Eager-Modus { #debug-eager-mode }

Dieser Modus, auch „Launch Blocking“ genannt, wird durch Setzen der
Umgebungsvariablen `TORCH_TPU_LAUNCH_BLOCKING=1` aktiviert. Er ist nur für
interaktives Debugging gedacht, nicht für den Produktionseinsatz.

Dieser Modus entspricht dem strikten Eager-Modus (Ausführung Op für Op), nur
dass der Host bei jeder Operation blockiert und synchronisiert. Das verlangsamt
die Ausführung künstlich, sorgt aber dafür, dass Fehler gemeldet werden, sobald
sie auftreten. So lassen sich Fehler leichter lokalisieren.

### `torch.compile` { #torchcompile }

Für maximale Performance bietet `torch.compile` sowohl mehr
Optimierungspotenzial als auch weniger Laufzeit-Overhead.

Wenn Sie eine bestimmte Funktion oder ein Modul mit `torch.compile`
kompilieren, erhält das Backend von TorchTPU den vollständigen Umfang des
Kompilierungsziels als
[FX-Graph](https://docs.pytorch.org/docs/stable/fx.html). Das liefert die
größtmögliche Menge an Informationen über die beabsichtigte Ausführung,
zusammen mit der Garantie, dass sie sich nicht ändert. Dadurch können wir
aggressive Optimierungen vornehmen, die im Eager-Modus zu riskant wären.

Als Backend für den kompilierten Modus führt TorchTPU zunächst einen Trace aus
(mit „Platzhalter“-Tensoren, ähnlich wie FakeTensor, aber mit TPU-spezifischen
Metadaten). Dabei entsteht ein vollständig verzögerter Berechnungsgraph, der den
FX-Graphen nachbildet. Sobald dieser erstellt ist, übergeben wir dieses
StableHLO-Programm an XLA, das es mit der höheren Optimierungsstufe `-O2`
kompiliert. Dieses Programm wird dann in ein aufrufbares Python-Objekt verpackt
und zur späteren Ausführung an den Nutzer zurückgegeben.

Wenn dieses aufrufbare Objekt aufgerufen wird, überspringen wir sofort die
meisten Schritte der Eager-Ausführung. Es werden keine neuen DeferredOps
erstellt, keine weitere Graph-Analyse ist nötig, und die Executable liegt
bereits vor, sodass kein Caching erforderlich ist. Stattdessen wird die
Executable sofort zur Ausführung an PjRt übergeben. Das minimiert die Kosten auf
dem Host bei einem Berechnungsgraphen, den der XLA-Compiler vollständig
optimiert hat.

### Optimierter Eager-Modus { #optimized-eager-mode }

Dieser Modus, auch „Defer and Fuse“ genannt, wird durch Setzen von
`TORCH_TPU_DEFER_AND_FUSE=1` aktiviert. Es ist ein neuer Hybridmodus, der viele
Vorteile von `torch.compile` in den Eager-Modus bringt.

In diesem Modus beginnt die Ausführung nicht sofort, wenn eine neue DeferredOp
erstellt wird. Die Ops bleiben zwischen Python-Aufrufen verzögert und warten in
der Stream-Warteschlange.

Die Materialisierung beginnt, wenn eine von zwei Bedingungen erfüllt ist:

*   Der Nutzer führt eine Aktion aus, die die Ausführung erzwingt. Die
    häufigsten sind der Start einer Übertragung vom Gerät zum Host oder ein
    expliziter Synchronize-Aufruf.
*   Wir erkennen eine Wiederholung in den dispatchten Ops. Das zeigt an, dass wir
    uns in einer Schleife befinden und eine Iteration dieser Schleife zur
    einfachen Wiederverwendung kompilieren sollten.

Das löst eine Materialisierung im Hintergrund aus, ähnlich wie im
Standard-Eager-Modus, nur dass der kompilierte Graph größer als eine einzelne
Operation ist. Tensoren, die beim Auslösen des Materialisierungssignals bereits
verworfen oder überschrieben wurden, werden übersprungen. Meist werden sie dann
in nachgelagerte Operationen fusioniert oder ganz eliminiert, wenn die Ausgabe
verworfen wird, ohne gelesen zu werden. (Hinweis: Bei verteilten kollektiven
Ops wird die Dead Code Elimination übersprungen, um Deadlocks zwischen TPUs zu
vermeiden.)

Da dabei größere Kompilierungsgraphen entstehen, verwenden wir die
Optimierungsstufe 2 (`-O2`), um die Ausführungszeit zu verbessern. Da wir
weniger, aber größere Graphen kompilieren, profitiert die Kompilierung weniger
von der Parallelisierung im Threadpool. Rechnen Sie daher mit längeren
Kompilierzeiten als im strikten Eager-Modus.

Der optimierte Eager-Modus ist zwar nicht so stark optimiert wie
`torch.compile`, bringt aber ohne Änderungen am Modellcode eine bis zu 50 %
höhere Performance als der Standard-Eager-Modus.

## Fallstricke { #pitfalls }

Diese Ausführungspipeline ist zwar sowohl auf Latenz als auch auf Durchsatz
optimiert, es gibt aber Fälle, in denen die Performance leiden kann.

### Alle Modi: Kontrollfluss in Python und datenabhängige Shapes { #all-modes-python-side-control-flow-and-data-dependent-shapes }

In PyTorch kann man leicht versehentlich und unbemerkt eine Übertragung vom
Gerät zum Host erzwingen. Eine Operation wie „`if my_tensor.all()`“ erstellt
unbemerkt einen Tensor mit booleschem Wert, synchronisiert den Stream und
kopiert diesen booleschen Wert auf die CPU, bevor der Kontrollfluss fortgesetzt
wird.

Im Standard-Eager-Modus erzwingt das keine neuen Graph-Breaks, aber die gesamte
Ausführungspipeline des Streams muss leerlaufen, bevor es weitergeht. Noch
schlimmer: Im optimierten Eager-Modus oder im kompilierten Modus erzwingt dieser
Synchronisationspunkt einen Graph-Break, entweder im per Eager-Modus fusionierten Stream
oder im FX-Graphen. Dadurch entsteht eine Optimierungsbarriere.

Ein ähnliches Verhalten kann auch bei PyTorch-Operationen mit datenabhängigen
Ausgabe-Shapes auftreten, etwa `torch.masked_select`. Da die Größe des
Ausgabe-Tensors erst bekannt ist, wenn die Masken-Eingabe ausgewertet wurde,
erzwingt auch das eine vollständig blockierende Synchronisation des Streams und
eine Übertragung der Shape-Metadaten vom Gerät zum Host, bevor es weitergeht.

### Strikter Eager-Modus: Winzige Executables verursachen Mikroblasen { #strict-eager-mode-tiny-executables-cause-microbubbles }

Das Einreihen einer Executable in PjRt dauert eine kurze Zeit, meist 50
Mikrosekunden oder weniger. In den meisten Fällen dauert jede Executable länger
(auch wenn es nur 1 oder 2 Millisekunden sind) und verbirgt diese Latenz
vollständig. So entsteht ein stabiler Zustand, in dem die TPU kontinuierlich mit
maximaler Effizienz arbeitet.

Wenn jedoch zu viele Executables eingereiht werden, die unter diesem
Schwellenwert liegen, etwa im strikten Eager-Modus mit kleinen Tensoren, kann
die TPU „verhungern“. Wenn die TPU schneller läuft, als die CPU neue Arbeit
einreihen kann, wird die TPU-Ausführung von winzigen, aber häufigen
„Mikroblasen“ (Microbubbles) unterbrochen, weil die Dispatch-Geschwindigkeit zum
Engpass wird.

Um das zu vermeiden, sorgen Sie dafür, dass Ihre Modelle bei jeder Operation
genug Arbeit verrichten, um diesen kleinen Start-Overhead auszugleichen. Oder
verwenden Sie den optimierten Eager-Modus, um Ausführungen automatisch und
laufend zu Kompilierungsblöcken zu fusionieren.

### Kompilierter Modus: Views als Argumente oder Ausgaben { #compiled-mode-views-as-arguments-or-outputs }

Beim Aufruf einer kompilierten Executable verlangt PjRt, dass Eingabewerte
zusammenhängend (contiguous) sind, keinen Offset haben und einen bestimmten
Datentyp aufweisen. Wenn ein Tensor-Argument eine nicht triviale View ist oder
per Bitcast aus dem ursprünglichen Datentyp umgewandelt wurde, erstellt
TorchTPU im Eager-Modus (strikt oder optimiert) eine temporäre Kopie des
Tensors, um die kompilierte Executable korrekt aufzurufen.

Um diese temporäre Kopie zu vermeiden, sorgen Sie dafür, dass alle Eingaben
einer kompilierten Funktion zusammenhängend sind. Wenn Sie mehrere kompilierte
Funktionen nacheinander aufrufen möchten, sorgen Sie dafür, dass die Ausgaben
der vorherigen Funktionen zusammenhängend sind. So vermeiden Sie Reibung durch
den Eager-Modus zwischen den optimierten Funktionen.

### Optimierter Eager-Modus: Beibehalten überflüssiger Tensoren { #optimized-eager-mode-retaining-redundant-tensors }

In allen Eager-Modi werden Operationen in der Reihenfolge ausgeführt, in der
sie erstellt werden. Immer wenn ein Materialisierungssignal ausgelöst wird,
werden alle zu diesem Zeitpunkt aktiven Tensoren materialisiert.

Wenn darunter unwichtige Zwischen-Tensoren sind, kann der Defer-and-Fuse-
Mechanismus sie fälschlicherweise als notwendige Ausgaben einstufen und
materialisieren. Das kann die Effizienz des Kompilierungsgraphen verringern.

Um die Fusion zu verbessern und den Speicherverbrauch zu minimieren, geben Sie
Tensoren frei, sobald Sie sie nicht mehr brauchen, oder verwenden Sie
In-place-Operationen, um veraltete Tensoren durch aktuelle Werte zu ersetzen.

## Zusammenfassung { #takeaways }

Das verzögerte Ausführungsmodell von TorchTPU bietet effizientes Pipelining mit
verschiedenen Abwägungen zwischen Kompilierzeit und Laufzeit.

*   **Für minimale Kompilierzeit:** Verwenden Sie den strikten Eager-Modus (den
    Standard). Dieser Modus beginnt sofort mit der Ausführung und
    parallelisiert die Kompilierung, um das erste Ergebnis so schnell wie
    möglich zu liefern.
*   **Für maximalen Durchsatz nach dem Warm-up und minimale Latenz:**
    Verwenden Sie `torch.compile`. Damit entfällt fast der gesamte Overhead des
    Eager-Modus, und das Optimierungspotenzial des Compilers wird maximal
    ausgeschöpft.
*   **Wenn eine schnellere Laufzeit gewünscht ist, `torch.compile` aber keine
    Option ist:** Ziehen Sie den optimierten Eager-Modus in Betracht.
*   **Für das Debugging:** Verwenden Sie den Launch-Blocking-Modus, damit CPU
    und TPU im Gleichschritt arbeiten.
*   **Vermeiden Sie erzwungene Synchronisationen:** Halten Sie nur eine
    minimale Menge zusammenhängender Tensoren vor und verwenden Sie
    rechenintensive Operationen, damit die TPU stets mit produktiver Arbeit
    versorgt ist.
