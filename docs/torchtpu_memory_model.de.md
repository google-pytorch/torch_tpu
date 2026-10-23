# TorchTPU und das Speichermodell der TPU { #torchtpu-and-the-tpu-memory-model }

PyTorch hat sich zusammen mit GPUs entwickelt und macht deren Leistung für
Python-Nutzer effektiv zugänglich. TorchTPU erweitert PyTorch, um die Leistung
von XLA auf TPUs nutzbar zu machen.

Die physische Hardware von GPUs und TPUs unterscheidet sich, ebenso die
Programmierparadigmen für die jeweilige Hardwarearchitektur. TorchTPU
überbrückt diese Lücke in beide Richtungen. Unser Ziel ist es, GPU-native
Schnittstellen wie Tensor-Views und Streams in XLA zu unterstützen und
gleichzeitig die besonderen Stärken des XLA-Compilers über vertraute
PyTorch-Schnittstellen bereitzustellen.

Wenn Sie verstehen, wie TorchTPU diese Ziele erreicht, können Sie TPU-native
Codebasen gezielt optimieren und den größten Nutzen daraus ziehen.

## Hardware: CPUs, GPUs, TPUs { #hardware-cpus-gpus-tpus }

Eine Universal-CPU (wie die in Ihrem Computer oder Mobilgerät) besteht aus
mehreren Kernen, die jeweils einen oder mehrere Threads unterstützen. Jeder
Thread kann parallel ausgeführt werden und dabei auf gemeinsamen Speicher mit
wahlfreiem Lese- und Schreibzugriff zugreifen.

Eine GPU ähnelt einer CPU in vielerlei Hinsicht, hat aber viel mehr Threads als
eine CPU. Eine moderne leistungsstarke CPU hat vielleicht ein paar Dutzend
Threads, eine leistungsstarke GPU dagegen *Tausende*. Der Nachteil: Jeder
einzelne GPU-Thread ist etwas schwächer als ein CPU-Thread, und es gibt einige
Einschränkungen bei der Unabhängigkeit der Threads. Trotzdem eignen sich GPUs
dadurch gut für Arbeitslasten mit stark parallelisierbaren Operationen, etwa
Grafik-Rendering oder lineare Algebra.

Eine TPU ist grundlegend anders aufgebaut. Das Herzstück der TPU ist der
TensorCore mit seinen Matrixmultiplikationseinheiten (MXUs). Diese MXUs
arbeiten als
[systolisches Array](https://en.wikipedia.org/wiki/Systolic_array), was in
gewisser Weise einem SIMD-Register auf einer CPU ähnelt. Sie verwenden breite,
zweidimensionale Register, die pro Taktzyklus deutlich mehr dichte
Rechenarbeit leisten können als ein CPU- oder GPU-Thread.

Der TensorCore ist jedoch grundsätzlich Single-Threaded. Ein systolisches Array
lässt sich nicht unterteilen, um mehrere Berechnungen gleichzeitig
auszuführen. Wegen der breiten Register ist ein TensorCore außerdem sehr
schlecht bei Sparse-Lese- und -Schreiboperationen mit wahlfreiem Zugriff, da
ein großer Speicherblock auf einmal geladen werden muss.

Die neuesten TPU-Generationen ergänzen ihre TensorCores um SparseCores, die
(wie der Name schon sagt) auf Sparse-Operationen spezialisiert sind,
einschließlich präziserer Lese- und Schreiboperationen. Die separaten Kerne
ermöglichen außerdem ein begrenztes Maß an Parallelität, da jeder SparseCore
Operationen gleichzeitig mit jedem TensorCore ausführen kann.

## Programmiermodell: CUDA und XLA im Vergleich { #programming-model-cuda-vs-xla }

Die nativen Softwareschnittstellen für GPUs und TPUs sind auf die Fähigkeiten
der jeweiligen Plattform zugeschnitten.

Nehmen wir CUDA als Beispiel, das am häufigsten verwendete
GPU-Programmiermodell unter PyTorch. CUDA ist eine Low-Level-
Programmierschnittstelle, ähnlich wie C oder C++. CUDA-Puffer sind wie
CPU-Speicher grundsätzlich eindimensionale Byte-Arrays. Ein CUDA-Kernel ist
eine imperative Prozedur, in der Terminologie von C eine void-Funktion. Jeder
Thread erhält Pointer auf einige Puffer und kann an jedem Offset in jedem Puffer
lesen und Byte für Byte zurückschreiben.

Dieses Programmiermodell funktioniert für TPUs nicht. Der Speicher auf TPUs ist
nicht als eindimensionale Byte-Arrays ausgerichtet, sondern als
zweidimensionale, an Registern ausgerichtete Tiles. Und wie bereits erwähnt,
haben TensorCores keine Threads und sind schlecht bei wahlfreiem Zugriff.

Stattdessen verwendet der TPU-Compiler XLA ein deklaratives, funktionales
Modell. XLA nimmt Programme in einer Sprache namens Multi-Level Intermediate
Representation (MLIR) entgegen, die mehrere Dialekte enthält. Mit dem
StableHLO-Dialekt lässt sich die gewünschte logische Tensor-Ausgabe eines
Programms beschreiben. XLA kompiliert dieses Programm in interne Operationen auf
niedrigerer Ebene, die für Ihre TPU-Generation optimiert sind.

## Puffer: `torch.UntypedStorage` { #buffers-torchuntypedstorage }

Die zentralen Werkzeuge von PyTorch sind sehr eng am CUDA-Programmiermodell
ausgerichtet.

Ein Speicherobjekt der Python-Klasse `torch.UntypedStorage` ist kaum mehr als
eine Speicheradresse (`torch.UntypedStorage.data_ptr()`) und eine Größe in Bytes
(`torch.UntypedStorage.nbytes()`). Das passt gut zu den Schnittstellen von CPU
und CUDA.

Für die Ausführung auf der TPU reicht das jedoch nicht. Ein Puffer in XLA hat
immer einen festen Typ und eine bekannte mehrdimensionale Shape. Da der
interne Speicher einer TPU an 2D-Registern ausgerichtet ist, muss unbedingt
bekannt sein, wie ein Puffer angeordnet ist. Ein `torch.UntypedStorage` allein
enthält daher nicht genug Informationen für die Ausführung auf der TPU.

Um dieses Problem zu lösen, fügt TorchTPU eine zusätzliche Ebene hinzu. Auf der
TPU zeigt ein `torch.UntypedStorage.data_ptr` nicht direkt auf TPU-Speicher,
sondern auf eine Stelle im CPU-Speicher, die zusätzliche Metadaten über den
TPU-Puffer enthält. **Das ist der Schlüssel zur Funktionsweise von
TorchTPU.** Diese zusätzlichen Metadaten ermöglichen eine aktive Übersetzung
zwischen der bestehenden CUDA-ähnlichen API von PyTorch und den Gegebenheiten
des funktionalen Designs von XLA und der TPU-Hardware.

## `torch.Tensor` { #torchtensor }
Ein `torch.Tensor` lässt sich als zwei Dinge zugleich betrachten.

Erstens ist ein `torch.Tensor` eine *physische* Darstellung eines
Zugriffsmusters auf ein zugrunde liegendes `torch.UntypedStorage`. Ein `Tensor`
enthält wesentliche Metadaten für die Strides (Multiplikationsfaktoren), Sizes
(Indexgrenzen), den Offset (Position des 0-Elements) und den Elementtyp (legt
die Bytes pro logischem Element fest). Mit diesen Metadaten lässt sich ein
Index-Tupel in einen Tensor auf einen Byte-Offset in dem linearen 1D-Puffer
abbilden, der diesem `Tensor` zugrunde liegt.

Zweitens ist ein `torch.Tensor` eine *logische* Darstellung eines
mathematischen Objekts. Wenn Sie zwei Tensoren per Matrixmultiplikation
verknüpfen, ist das Ergebnis ein weiterer `Tensor`, mit einer passenden Shape,
der das Ergebnis darstellt. Das genaue Muster der Strides und der Storage-Offset
spielen hier kaum eine Rolle. Was der `Tensor` als Berechnung bedeutet, ist
wichtiger als das, was der Tensor als Speicherreferenz ist.

Beim Wechsel von CUDA zur TPU ist es entscheidend, diesen Unterschied zu
verstehen: Das **logische** Verhalten von PyTorch-Operationen auf Tensoren kann
auf XLA/TPU korrekt umgesetzt werden, auch wenn die **physische** Realität des
Geräts grundlegend anders ist!

## View-Zerlegung { #view-decomposition }

Bei der Ausführung auf CUDA verwenden die internen CUDA-Kernels die
Stride-Beschreibung eines `torch.Tensor` direkt. Die Strides werden an die
internen Thread-Prozeduren übergeben, um zu beschreiben, mit welchem Offset in
den jeweiligen Puffer zugegriffen wird. Diese Offsets werden dynamisch zur
Laufzeit berechnet.

Auf der TPU lassen sich die physischen Strides dagegen nicht direkt verwenden.
Eine eindimensionale Stride-Beschreibung ist für zweidimensionalen, an
Registern ausgerichteten Speicher unbrauchbar. Das logische Verhalten von Views
*wird* aber über StableHLO unterstützt. Für die meisten View-Operationen auf
Tensoren gibt es eine direkte Entsprechung in StableHLO: Reshape, Transpose,
Slicing und Broadcasting.

Wenn TorchTPU ein Stride-Muster eines `torch.Tensor` erhält, führt es eine
Operation namens *View-Zerlegung* (View Decomposition) aus. Dabei werden ein
Stride-Muster und ein Elementtyp als Folge logischer Operationen interpretiert,
die den gewünschten Ausgabe-Tensor mit jedem Element an der richtigen Stelle
erzeugen.

Ein Beispiel:

```python
# Logical values: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
# sizes=(10,), strides=(1,), storage_offset=0
base = torch.arange(10, dtype=torch.float)

# Logical values: [[[1., 1.], [4., 4.]], [[3., 3.], [6., 6.]]]
physical_view = base.as_strided(
  sizes=(2, 2, 2),
  strides=(2, 3, 0),
  storage_offset=1
)

# Logical values: [[[1., 1.], [4., 4.]], [[3., 3.], [6., 6.]]]
# StableHLO: slice, reshape, slice, transpose, reshape, broadcast_in_dim
logical_view = base[1:].view(3, 3)[:2, ::2].t().unsqueeze(-1).expand(2, 2, 2)
```

Die Low-Level-Version mit Stride-Manipulation über `torch.Tensor.as_strided`
lässt sich gleichwertig als Folge vertrauterer Operationen ausdrücken, die dann
in StableHLO dargestellt und von XLA kompiliert werden können.

Beachten Sie jedoch: **Logische Gleichwertigkeit bedeutet nicht gleiche
Performance**. Eine Reshape- oder Transpose-Operation kann auf der TPU in
manchen Fällen eine vollständige Kopie erfordern, um von einem an Registern
ausgerichteten Layout in ein anderes zu wechseln. Auf TorchTPU sind **Views
daher nicht „kostenlos“**, wie sie es auf CUDA oft sind, weil sich die
GPU- und TPU-Hardware physisch unterscheiden.

## In-place-Mutationen und Buffer Donation { #in-place-mutations-and-buffer-donation }

StableHLO ist eine funktionale Programmiersprache. Genauer gesagt ist es eine
[Sprache in Static Single Assignment Form (SSA)](https://en.wikipedia.org/wiki/Static_single-assignment_form).
Werte können also nur neu erstellt, aber nie geändert oder neu zugewiesen
werden. Das hilft dem XLA-Compiler, optimierte Ausführungspläne zu erstellen.

Weder GPU- noch CPU-IRs sind rein funktional, und folglich ist es PyTorch auch
nicht. Tensor-Operationen, die mit einem Unterstrich enden oder das
Schlüsselwortargument „`out=`“ verwenden, überschreiben die Bytes in ihrem
zugrunde liegenden `torch.UntypedStorage`. Da sich mehrere `torch.Tensors`
einen gemeinsamen Speicherpuffer teilen können, wird ein Schreibvorgang in
einen Tensor sofort für alle anderen Tensoren sichtbar.

TorchTPU löst das mithilfe seiner Metadatenebene. Wenn eine PyTorch-Operation
eine In-place-Aktualisierung oder eine Zuweisung an einen Out-Tensor anfordert,
geht TorchTPU so vor:

*   Es beschreibt die logische Aktualisierung gegenüber XLA, wodurch ein neuer
    Puffer entsteht
*   Es tauscht den Puffer im Metadatenobjekt aus, auf das das
    `torch.UntypedStorage` zeigt

Das ist im Wesentlichen eine automatische Funktionalisierung: Alle Mutationen
werden stattdessen als Out-of-place-Operationen ausgedrückt.

Für die automatische Funktionalisierung muss der Algorithmus zur
View-Zerlegung auch umgekehrt ausgeführt werden können. Wir nennen das
View-Umkehrung (View Inversion). Sie ist nur bei nicht überlappenden Views
möglich. Das Schreiben in einen Tensor, der durch Broadcasting entstanden ist,
ist ungültig, weil mehrere logische Positionen derselben physischen Position
entsprechen und die Anweisung dadurch mehrdeutig wird. Im CUDA-Backend gilt
eine solche Operation als undefiniertes Verhalten und schlägt nicht unbedingt
explizit fehl. In TorchTPU schlägt sie immer explizit fehl.

Bei einem teilweisen Schreibvorgang verwendet die View-Umkehrung eine
StableHLO-Operation namens
[`stablehlo.dynamic_update_slice`](https://openxla.org/stablehlo/spec#dynamic_update_slice),
um einen zusammengeführten Ausgabe-Tensor zu berechnen, der sowohl die
überschriebenen als auch die nicht überschriebenen Werte an den richtigen
Stellen enthält.

Eine Out-of-place-Operation verdoppelt natürlich den Spitzenspeicherbedarf:
Alte und neue Version existieren kurz nebeneinander, bevor der alte Wert
verworfen wird. Um das zu vermeiden, versucht TorchTPU nach Möglichkeit, die
nächstliegende Entsprechung von XLA zu In-place-Mutationen anzuwenden:
**Buffer Donation**. Wenn die letzte Referenz auf einen veralteten Puffer
entfernt wird, annotiert TorchTPU das StableHLO so, dass die Eingabe
„gespendet“ wird. XLA kann die Eingabe dann verbrauchen, und der zusätzliche
Speicherbedarf durch die automatische Funktionalisierung entfällt.

Buffer Donation lässt sich nicht immer sicher durchführen. Wenn TorchTPU nicht
nachweisen kann, dass ein Puffer sicher gespendet werden kann, greift es auf
eine Out-of-place-Aktualisierung zurück. Das kann bei kleinen Schreibvorgängen
in große Puffer besonders teuer sein. In diesem Fall kopiert die
Out-of-place-Aktualisierung den gesamten zugrunde liegenden Storage-Tensor,
nicht nur den Teil, in den geschrieben wird!

## Copy-on-Write { #copy-on-write }

Die Funktionalisierung hat aber auch Vorteile. Einer davon ist, dass
Copy-on-Write-Verhalten möglich wird.

Im CUDA-Backend alloziert ein Aufruf von `torch.Tensor.clone()` sofort einen
neuen physischen Speicherpuffer, der groß genug für die Daten des Tensors ist.
Danach sind alle In-place-Schreibvorgänge in die verschiedenen
`torch.UntypedStorages` vollständig unabhängig, aber die Speicherkosten fallen
sofort an.

Im TorchTPU-Backend dagegen bedeutet die Unveränderlichkeit eines statisch
zugewiesenen Werts, dass eine Kopie einfach eine Referenz auf das Original sein
kann. Eine sofortige physische Kopie ist nicht nötig.

Sobald es eine In-place-Aktualisierung gibt, müssen wir diese Aktualisierung nur
im geänderten `torch.UntypedStorage` abbilden und nicht im anderen. Das
verhindert Buffer Donation (da wir weiterhin den alten und den neuen Wert
brauchen). In diesem Fall wird die In-place-Aktualisierung daher ohne Donation
out-of-place ausgeführt. Diese Out-of-place-Aktualisierung alloziert neuen
Speicher, aber erst verzögert und nur bei Bedarf, anders als bei PyTorch/CUDA,
wo die Speicherkosten im Voraus anfallen.

## Eigene Kernels { #custom-kernels }

TorchTPU versucht nach bestem Bemühen, speicheroptimierende Buffer Donations
anzuwenden, und bietet eine logisch gleichwertige Transformation von
CUDA-Views mit Strides in View-Operationen von StableHLO. Trotzdem sind das
weiterhin High-Level-Operationen, die nicht direkt an das physische Layout eines
TPU-Puffers gebunden sind und keine bestimmte Low-Level-Semantik garantieren.

Für Nutzer, die volle Low-Level-Kontrolle wünschen, unterstützt TorchTPU auch
die Integration eigener Kernels. Damit lassen sich kritische Operationen gezielt
ersetzen oder sogar ganze Modelle neu schreiben, wobei das Verhalten der
TPU-Puffer manuell festgelegt wird. So können Sie optimale Donations und ein
optimales Verhalten bei In-place-Aktualisierungen sicherstellen.

## Zusammenfassung { #takeaways }
Wenn Sie mit TorchTPU arbeiten, sollten Sie die folgenden wichtigen Fakten
kennen:

*   **PyTorch-Code liefert auf CUDA und TorchTPU dasselbe logische Ergebnis:**
    Sie können vertraute Low-Level-Operationen aus CUDA verwenden (Views mit
    gemeinsamem Storage, Stride-Manipulation, In-place-Aktualisierungen) und auf
    der TPU dieselben Ergebnisse erzielen, auch wenn die Performance nicht
    unbedingt dieselbe ist.
*   **Kopien sind günstig:** Einen Tensor in eine zusammenhängende Form zu
    kopieren, kostet dank Copy-on-Write oft keine Rechenzeit.
*   **Views sind teuer:** Wenn Sie einen Tensor mit einer anderen Shape oder
    einem nicht zusammenhängenden Muster lesen, kann das auf der TPU ein
    Relayout erzwingen, was teuer sein kann. Verwenden Sie nach Möglichkeit
    zusammenhängende Tensoren.
*   **Das Überschreiben eines zusammenhängenden Tensors ist meist
    speichereffizient:** Ein zusammenhängender Tensor lässt sich meist mit
    Buffer Donation überschreiben, wenn alle alten Werte gelöscht werden.
*   **Teilweise Schreibvorgänge sind manchmal speicherintensiv:** Wenn Buffer
    Donation nicht möglich ist, sind die Speicherkosten eines Schreibvorgangs
    proportional zur Größe des torch.UntypedStorage, nicht zur Größe der
    geschriebenen Daten. Verwenden Sie lieber viele kleine Tensoren statt
    Slices großer Tensoren.
*   **Eigene Kernels bieten volle Kontrolle:** Wenn die Übersetzung von CUDA
    nach XLA fehlschlägt oder nicht ausreicht, können Sie mit einem eigenen,
    TPU-nativen Kernel gezielt auf eine bestimmte Low-Level-Performance
    optimieren.
