# TorchTPU y el modelo de memoria de la TPU { #torchtpu-and-the-tpu-memory-model }

PyTorch evolucionó junto con las GPU y pone su potencia al alcance de los
usuarios de Python de forma eficaz. TorchTPU extiende PyTorch para expresar la
potencia de XLA en las TPU.

El hardware físico de las GPU y las TPU es diferente, al igual que los
paradigmas de programación de cada arquitectura de hardware. TorchTPU existe
para cerrar esta brecha en ambas direcciones. Nuestro objetivo es admitir en XLA
interfaces nativas de GPU, como las vistas de tensores y los streams, y, al
mismo tiempo, exponer la potencia única del compilador XLA en interfaces de
PyTorch conocidas.

Comprender cómo TorchTPU logra estos objetivos puede ayudar a los usuarios a
optimizar bases de código nativas de TPU para obtener el máximo beneficio.

## Hardware: CPU, GPU y TPU { #hardware-cpus-gpus-tpus }

Una CPU de uso general (como la que hace funcionar tu computadora o dispositivo
móvil) consta de varios núcleos, y cada uno admite uno o más subprocesos. Cada
subproceso puede ejecutarse en paralelo y, al mismo tiempo, acceder a memoria
compartida con capacidades de lectura y escritura de acceso aleatorio.

Una GPU se parece en muchos aspectos a una CPU; sin embargo, una GPU tiene
muchos más subprocesos que una CPU. Una CPU moderna de alta potencia puede tener
algunas decenas de subprocesos, pero una GPU de alta potencia tiene *miles*. La
contrapartida es que cada subproceso de GPU es, de forma individual, algo menos
potente que un subproceso de CPU, y hay algunas restricciones en la
independencia de los subprocesos. Aun así, esto hace que las GPU sean útiles
para cargas de trabajo con operaciones muy paralelizables, como la renderización
de gráficos o el álgebra lineal.

Una TPU tiene un diseño fundamentalmente diferente. El elemento central de la
TPU es el TensorCore y sus unidades de multiplicación de matrices (MXU). Estas MXU
funcionan como un
[arreglo sistólico](https://en.wikipedia.org/wiki/Systolic_array), que es algo
similar a un registro SIMD en una CPU. Usan registros anchos y bidimensionales
que pueden realizar mucho más trabajo de cálculo denso por ciclo de reloj que un
subproceso de CPU o GPU.

Sin embargo, el TensorCore es fundamentalmente de un solo subproceso; no es
posible subdividir un arreglo sistólico para hacer varios cálculos simultáneos.
Además, tener registros anchos significa que un TensorCore es muy poco eficiente
para realizar operaciones dispersas de lectura y escritura de acceso aleatorio,
ya que se debe cargar un bloque de memoria grande de una sola vez.

Las generaciones más recientes de TPU complementan sus TensorCores con
SparseCores, que (como su nombre lo indica) están especializados en operaciones
dispersas, incluidas operaciones de lectura y escritura más precisas. Los
núcleos separados también permiten una cantidad limitada de paralelismo, ya que
cada SparseCore puede realizar operaciones al mismo tiempo que cada TensorCore.

## Modelo de programación: CUDA frente a XLA { #programming-model-cuda-vs-xla }

Las interfaces de software nativas de las GPU y las TPU están adaptadas a las
capacidades de esas plataformas.

Toma CUDA como ejemplo, ya que es el modelo de programación de GPU más usado
como base de PyTorch. CUDA es una interfaz de programación de bajo nivel,
similar a C o C++. Los búferes de CUDA son, al igual que la memoria de la CPU,
arreglos de bytes intrínsecamente unidimensionales. Un kernel de CUDA es un
procedimiento imperativo (una función void, para usar la terminología de C). Cada
subproceso recibe punteros a algunos búferes y puede leer desde cualquier
desplazamiento en cualquier búfer y volver a escribir en ellos, byte por byte.

Este modelo de programación no funciona para las TPU. La memoria de las TPU no
se alinea como arreglos de bytes unidimensionales, sino como tiles
bidimensionales alineados con los registros. Y, como se mencionó, los
TensorCores no tienen subprocesos y son poco eficientes con el acceso aleatorio.

En cambio, el compilador de TPU, XLA, usa un modelo declarativo y funcional. XLA
acepta programas escritos en un lenguaje llamado Multi-Level Intermediate
Representation (MLIR), que contiene varios dialectos. Con el dialecto
StableHLO, es posible describir la salida lógica de tensores que se espera de un
programa. XLA compila ese programa en operaciones internas de nivel más bajo que
están optimizadas para tu generación específica de TPU.

## Búferes: `torch.UntypedStorage` { #buffers-torchuntypedstorage }

Las herramientas principales de PyTorch están muy alineadas con el modelo de
programación de CUDA.

Un objeto de almacenamiento, que usa la clase de Python
`torch.UntypedStorage`, es poco más que una dirección de memoria
(`torch.UntypedStorage.data_ptr()`) y un tamaño en bytes
(`torch.UntypedStorage.nbytes()`). Esto se corresponde bien con las interfaces
de CPU y de CUDA.

Sin embargo, esto no es suficiente para la ejecución en TPU; en XLA, un búfer
siempre tiene un tipo bien definido y una forma multidimensional conocida. Como
la memoria interna de una TPU está alineada con registros 2D, es fundamental
saber cómo se distribuye un búfer. Esto significa que un `torch.UntypedStorage`
por sí solo no contiene suficiente información para la ejecución en TPU.

Para resolver este problema, TorchTPU agrega una capa adicional. En TPU, un
`torch.UntypedStorage.data_ptr` no apunta directamente a la memoria de la TPU,
sino a una ubicación en la memoria de la CPU que contiene metadatos adicionales
sobre el búfer de TPU. **Esta es la clave del funcionamiento de TorchTPU**: estos
metadatos adicionales permiten una interpretación activa entre la API existente
de PyTorch, similar a CUDA, y la realidad del diseño funcional de XLA y del
hardware de TPU.

## `torch.Tensor` { #torchtensor }
Puedes pensar en un `torch.Tensor` como dos cosas a la vez.

Primero, un `torch.Tensor` es una representación *física* de un patrón de acceso
sobre un `torch.UntypedStorage` subyacente. Un `Tensor` contiene metadatos
esenciales para los strides (factores multiplicativos), los tamaños (límites de
los índices), el desplazamiento (ubicación del elemento 0) y el tipo de
elemento (establece los bytes por elemento lógico). Estos metadatos permiten
asignar una tupla de índices de un tensor a un desplazamiento de bytes en el
arreglo de búfer lineal 1D que respalda ese `Tensor`.

Segundo, un `torch.Tensor` es una representación *lógica* de un objeto
matemático. Cuando multiplicas dos tensores como matrices, el resultado es otro
`Tensor`, con una forma adecuada, que representa el resultado. El patrón exacto de
strides y el desplazamiento de almacenamiento tienen una importancia mínima
aquí; lo que el `Tensor` significa como cálculo es más relevante que lo que el
tensor es como referencia de memoria.

Cuando pasas de CUDA a TPU, comprender esta distinción es fundamental: el
comportamiento **lógico** de las operaciones de PyTorch con tensores se puede
realizar correctamente en XLA/TPU, aunque la realidad **física** del dispositivo
sea intrínsecamente diferente.

## Descomposición de vistas { #view-decomposition }

Cuando se ejecuta en CUDA, los kernels internos de CUDA usan directamente el
lenguaje de strides de un `torch.Tensor`; los strides se proporcionan a los
procedimientos internos de los subprocesos para describir cómo desplazarse en el
búfer correspondiente. Estos desplazamientos se calculan de forma dinámica en el
entorno de ejecución.

Pero en TPU, los strides físicos no se pueden usar directamente; una descripción
unidimensional de strides no sirve para una memoria bidimensional alineada con
registros. Sin embargo, el comportamiento lógico de las vistas *sí* es
compatible a través de StableHLO. La mayoría de las operaciones de vista de
tensores tienen un equivalente directo en StableHLO: cambio de forma,
transposición, corte y broadcasting.

Cuando recibe un patrón de strides de `torch.Tensor`, TorchTPU realiza una
operación llamada *descomposición de vistas* (view decomposition). Esta
operación interpreta un patrón de strides y un tipo de elemento como una
secuencia de operaciones lógicas que producen el tensor de salida deseado con
cada elemento en el lugar correcto.

Por ejemplo:

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

La versión de bajo nivel con manipulación de strides que usa
`torch.Tensor.as_strided` se puede expresar de forma equivalente con una
secuencia de operaciones más conocidas, que luego se pueden expresar en
StableHLO y compilar con XLA.

Sin embargo, es importante tener en cuenta que **la equivalencia lógica no
implica un rendimiento equivalente**. En algunos casos, una operación de cambio
de forma o de transposición puede requerir una copia completa en la TPU para
pasar de un diseño alineado con registros a otro. Esto significa que, en
TorchTPU, **las vistas no son "gratuitas"** como suelen serlo en CUDA, debido a
las diferencias físicas entre el hardware de GPU y el de TPU.

## Mutaciones in situ y donación de búferes { #in-place-mutations-and-buffer-donation }

StableHLO es un lenguaje de programación funcional. Más específicamente, es un
[lenguaje de asignación única estática (SSA)](https://en.wikipedia.org/wiki/Static_single-assignment_form).
Esto significa que los valores solo se pueden crear desde cero; nunca se
cambian ni se reasignan. Esto ayuda al compilador XLA a producir planes de
ejecución optimizados.

Ni las IR de GPU ni las de CPU son puramente funcionales y, en consecuencia,
PyTorch tampoco lo es. Las operaciones con tensores que terminan con un guion
bajo, o que usan el argumento de palabra clave "`out=`", sobrescriben los bytes
de su `torch.UntypedStorage` base. Como varios `torch.Tensors` pueden compartir
un búfer de almacenamiento común, una escritura en cualquier tensor se vuelve
visible de inmediato para todos los demás tensores.

TorchTPU maneja esto aprovechando nuestra capa de metadatos. Cuando una
operación de PyTorch solicita una actualización in situ o una asignación a un
tensor de salida, TorchTPU hace lo siguiente:

*   Describe la actualización lógica a XLA, lo que produce un búfer nuevo
*   Intercambia el búfer dentro del objeto de metadatos al que apunta el
    `torch.UntypedStorage`

En esencia, esto es una funcionalización automática: todas las mutaciones se
expresan, en cambio, como operaciones fuera de lugar (out-of-place).

Para lograr la funcionalización automática, es necesario poder ejecutar el
algoritmo de descomposición de vistas a la inversa, lo que llamamos inversión de
vistas. Esto solo es posible con vistas que no se superponen; escribir en un
tensor que resultó de un broadcast no es válido, ya que varias ubicaciones
lógicas corresponden a una sola ubicación física, lo que hace que la
instrucción sea ambigua. En el backend de CUDA, la operación se consideraría un
comportamiento indefinido, pero puede fallar de forma explícita o no; en
TorchTPU, siempre falla de forma explícita.

En el caso de una escritura parcial, la inversión de vistas usa una operación de
StableHLO llamada
[`stablehlo.dynamic_update_slice`](https://openxla.org/stablehlo/spec#dynamic_update_slice)
para calcular un tensor de salida combinado, que contiene tanto los valores
sobrescritos como los no sobrescritos en las ubicaciones correctas.

Por supuesto, una operación fuera de lugar duplica el pico de memoria: las
versiones anterior y nueva coexisten brevemente antes de que se descarte el
valor anterior. Para evitarlo, TorchTPU intenta aplicar, cuando es posible, el
equivalente más cercano de XLA a las mutaciones in situ: la **donación de
búferes**. Cuando se quita la última referencia a un búfer obsoleto, TorchTPU
anota el StableHLO para "donar" la entrada, lo que permite que XLA consuma la
entrada y evita el aumento de memoria de la funcionalización automática.

No siempre es posible donar búferes de forma segura. Si TorchTPU no puede
demostrar que es seguro donar un búfer, recurre a una actualización fuera de
lugar. Esto puede ser especialmente costoso para escrituras pequeñas en búferes
grandes: en este caso, la actualización fuera de lugar copia todo el tensor de
almacenamiento base, no solo la parte en la que se escribe.

## Copia en escritura { #copy-on-write }

Sin embargo, la funcionalización tiene ventajas. Una de ellas es la posibilidad
de adoptar el comportamiento de copia en escritura (copy-on-write).

En el backend de CUDA, llamar a `torch.Tensor.clone()` asigna de inmediato un
nuevo búfer de memoria física lo suficientemente grande para contener los datos
del tensor. Después de esto, cualquier escritura in situ en los distintos
`torch.UntypedStorages` es totalmente independiente, pero el costo de memoria se
paga de inmediato.

En cambio, en el backend de TorchTPU, la inmutabilidad de un valor asignado de
forma estática significa que una copia puede ser simplemente una referencia al
original, sin que se requiera una copia física inmediata.

En cuanto se produce una actualización in situ, debemos reflejarla solo en el
`torch.UntypedStorage` que se modificó y no en el otro. Esto impide la donación
de búferes (ya que todavía necesitamos los valores anterior y nuevo), por lo que,
en este caso, la actualización in situ se realiza fuera de lugar y sin donación.
Esta actualización fuera de lugar asigna memoria nueva, pero esa memoria solo se
asigna de forma diferida según sea necesario, a diferencia de PyTorch/CUDA, que
paga el costo de memoria por adelantado.

## Kernels personalizados { #custom-kernels }

TorchTPU hace todo lo posible por aplicar donaciones de búferes que optimizan la
memoria y proporciona una transformación lógicamente equivalente de las vistas
con strides de CUDA a operaciones de vista de StableHLO. Dicho esto, siguen
siendo operaciones de alto nivel, que no están vinculadas directamente al diseño
físico de un búfer de TPU y que no garantizan ninguna semántica de bajo nivel en
particular.

Para los usuarios que quieren el máximo control de bajo nivel, TorchTPU también
admite la integración de kernels personalizados. Estos permiten reemplazar de
forma precisa operaciones críticas, o incluso reescribir modelos completos, con
la especificación manual del comportamiento de los búferes de TPU. Puedes usar
esto para garantizar un comportamiento óptimo de donación y de actualización in
situ.

## Conclusiones { #takeaways }
Cuando trabajas con TorchTPU, debes conocer los siguientes datos clave:

*   **El código de PyTorch produce el mismo resultado lógico en CUDA y en
    TorchTPU:** Puedes usar operaciones conocidas de bajo nivel de CUDA (vistas
    con almacenamiento compartido, manipulación de strides, actualizaciones in
    situ) y obtener los mismos resultados en TPU, aunque el rendimiento puede no
    ser el mismo.
*   **Las copias son económicas:** Copiar un tensor en una forma contigua suele
    no tener costo de cálculo, gracias al comportamiento de copia en escritura.
*   **Las vistas son costosas:** Leer un tensor con una forma diferente o con un
    patrón no contiguo puede forzar un cambio de diseño (relayout) en la TPU,
    lo que puede ser costoso. Siempre que sea posible, prefiere tensores
    contiguos.
*   **Sobrescribir un tensor contiguo suele ser eficiente en memoria:** Por lo
    general, un tensor contiguo se puede sobrescribir con una donación de
    búferes si se borran todos los valores anteriores.
*   **Las escrituras parciales a veces son costosas en memoria:** Si la donación
    de búferes no es posible, el costo de memoria de una escritura es
    proporcional al tamaño del torch.UntypedStorage, no al tamaño de los datos
    escritos que se modifican. Prefiere usar muchos tensores pequeños en lugar
    de partes de tensores grandes.
*   **Los kernels personalizados te dan control total:** Cuando la traducción de
    CUDA a XLA falla o no es suficiente, implementar un kernel personalizado
    nativo de TPU permite optimizar para un rendimiento específico de bajo
    nivel.
