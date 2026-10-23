# Cómo funciona TorchTPU { #how-torchtpu-works }

## Introducción a las TPU { #tpu-primer }

Las [TPU de Google](https://cloud.google.com/tpu) son ASIC especializados,
diseñados específicamente para realizar las operaciones masivas con tensores que
son la base del aprendizaje profundo. La
[arquitectura de las TPU varía](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)
entre generaciones, pero, en términos generales, un chip TPU contiene lo
siguiente:

*   Uno o más TensorCores. Cada uno es un procesador de un solo subproceso que
    incluye unidades de multiplicación de matrices (MXU), una unidad vectorial
    (VPU) y una unidad escalar.
*   Uno o más SparseCores, que están especializados en operaciones dispersas y a
    los que el TensorCore puede descargar operaciones.
*   Una memoria de alto ancho de banda (HBM) grande (288 GB a 8.6 TB/s en
    [TPU 8i](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive)),
    que se puede usar para almacenar datos de AA y compartirlos con la CPU.
*   Interconexión entre chips (ICI) bidireccional (19.2 TBps en
    [TPU 8t y 8i](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/)),
    que permite una comunicación rápida entre chips TPU dentro de
    [porciones de TPU](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices)
    (slices) configuradas por el usuario. Las porciones son subgrupos lógicos de
    un [pod de TPU](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu-pod),
    que se compone de una cantidad preconfigurada de chips TPU conectados en una
    configuración de toro 3D (en TPU 7 y TPU 8t) o Boardfly (en TPU 8i).
*   Red de centro de datos (DCN) bidireccional (400 GBps en
    [TPU 8t](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/)),
    que permite la comunicación entre chips TPU de distintas porciones.

Las TPU se programan con la pila del
[compilador XLA](https://github.com/openxla/xla) a través de dos puntos de
entrada diferentes: [StableHLO](https://openxla.org/stablehlo/spec) y
[Mosaic](https://docs.jax.dev/en/latest/pallas/design/design.html#lowering-pallas-to-mosaic-for-tpu).
La ISA real de la TPU es propietaria, puede cambiar y no está disponible
públicamente. En la práctica, puedes pensar en StableHLO y (todavía más) en
Mosaic como equivalentes a PTX. Consulta la siguiente imagen para ver un
esquema general:

![Cómo funciona TorchTPU](./images/torch_tpu_inputs_to_tpu.svg)

Un programa de C++ puede acceder a una TPU local si crea una instancia de un
[cliente de PjRT](https://openxla.org/xla/pjrt/cpp_api_overview), cuya API
proporciona métodos para compilar MLIR (StableHLO o Mosaic) en un ejecutable de
TPU, iniciar ese ejecutable y copiar datos hacia y desde la TPU. Ten en cuenta
que la ejecución en TPU es asíncrona y que la CPU puede esperar a que un
resultado de la TPU esté listo si espera el PjRTBuffer asociado. StableHLO sigue
el [paradigma SSA](https://en.wikipedia.org/wiki/Static_single-assignment_form)
y, por lo tanto, no permite mutaciones in situ; es decir, una operación de
escritura siempre produce un valor nuevo (al menos de forma conceptual). Esto es
especialmente importante para TorchTPU, que ofrece la conocida semántica de
"vistas" de PyTorch, incluidas las mutaciones, mediante la descomposición y la
inversión de vistas. El compilador XLA supone que las formas de los tensores se
conocen en el momento de la compilación. En TorchTPU, esta restricción se
flexibiliza con el [dinamismo acotado](./dynamic_shapes.md). Como reconocemos
que esto es una limitación para las cargas de trabajo de PyTorch, estamos
agregando compatibilidad con el dinamismo acotado, es decir, cuando los límites máximos
de las dimensiones de una forma se conocen en el momento de la compilación,
pero las dimensiones reales solo se conocen en el entorno de ejecución.

## TorchTPU: Un backend fuera del árbol con compatibilidad con la carga automática { #torchtpu-an-out-of-tree-backend-with-autoload-support }

Siguiendo la
[práctica recomendada actual](https://docs.pytorch.org/tutorials/advanced/extend_dispatcher.html),
en la que los backends adicionales se construyen fuera del árbol (out of tree)
y se registran en el entorno de ejecución, extendemos el despachador de PyTorch
con un nuevo backend de C++, registrado como `PrivateUse1`. Además, usamos el
mecanismo estándar de `at::Tensor` para definir una implementación del
almacenamiento de tensores específica del dispositivo (con el
`c10::UniqueVoidPtr` dentro de `c10::StorageImpl`).

Diseñamos TorchTPU pensando en la facilidad de uso. Cuando el usuario importa
PyTorch en una máquina con TPU en la que está instalado el wheel de TorchTPU,
Python carga automáticamente el wheel de TorchTPU. La carga automática del wheel
configura el backend y reemplaza el nombre `PrivateUse1` por `tpu`. Así, en un
ejemplo simple, un programa de TorchTPU no se diferencia de los backends
incluidos en el árbol, y un programa de PyTorch puede ejecutarse en la TPU con
pocos cambios o ninguno:

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

Como la TPU se convierte en el acelerador actual y `tpu_dist` es el backend
predeterminado de TorchTPU, no es necesario especificarlos de forma explícita:

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

## Captura del grafo { #capturing-the-graph }

TorchTPU traza las operaciones eager a nivel de ATen y se basa en los hooks de
despacho de operaciones de PyTorch que forman parte de la infraestructura del
backend. Las operaciones de TorchTPU se construyen como programas de StableHLO
y se pasan a XLA de forma individual o en grupos, lo que permite que el
compilador realice una fusión significativa de operaciones para usar el
hardware de forma más eficaz. Cuando TorchTPU recibe una invocación de
devolución de llamada del despachador de PyTorch, el entorno de ejecución hace
lo siguiente:

*   Crea una instancia de operación diferida (Deferred Op) que contiene lo
    siguiente:
    *   Un objeto constructor (builder) de StableHLO que, cuando se invoca,
        devuelve una función de StableHLO que implementa la semántica de
        `aten::mul`, como se muestra a continuación (ten en cuenta que cualquier
        broadcasting implícito y promoción de dtype se manejan en otro lugar)

```
func.func @main(%arg0: tensor<4x16xbfloat16>, %arg1: tensor<4x16xbfloat16>) -> tensor<4x16xbfloat16> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<4x16xbfloat16>
  return %0 : tensor<4x16xbfloat16>
}
```

*   Un mapa opcional de parámetros estáticos de la operación (por ejemplo,
    training y momentum para `aten::native_batch_norm`) que se usa para
    calcular una clave de caché.
*   Devuelve un nuevo `at::Tensor` con un `c10::StorageImpl` que apunta
    (mediante indirección) a esa instancia de operación diferida.

Cuando llega a un punto de sincronización adecuado, que depende del modo de
ejecución, TorchTPU hace lo siguiente:

*   Analiza la región de operaciones.
*   Determina qué operaciones materializar en código ejecutable.
*   Crea una clave de caché de compilación para el grafo.
*   Si ya existe en la caché, obtiene el ejecutable y lo despacha en la TPU.
*   Si no existe en la caché, compila el objeto de forma asíncrona, lo agrega a
    la caché y lo despacha para su ejecución.
*   Si corresponde para el punto de sincronización alcanzado, espera a que el
    valor esté listo y lo copia de vuelta.

En [El modelo de ejecución de TorchTPU](./torchtpu_execution_model.md),
analizamos con más detalle los distintos modos de ejecución.

## La caché de compilación de TorchTPU { #the-torchtpu-compilation-cache }

Para evitar compilaciones innecesarias, TorchTPU usa una caché de compilación de
varios niveles:

*   **Nivel 1:** Caché en memoria local del proceso para compartir compilaciones
    dentro del mismo proceso. Es el más rápido de los tres niveles. Siempre está
    habilitado.
*   **Nivel 2:** Memoria compartida local de la máquina que permite compartir
    entre procesos en el mismo host. Es útil para el entrenamiento o la
    inferencia en varios dispositivos. El nivel 2 está habilitado de forma
    predeterminada para los programas distribuidos, pero en los programas de un
    solo dispositivo no ofrece ningún beneficio, por lo que está inhabilitado.
*   **Nivel 3:** Caché persistente respaldada por un sistema de archivos remoto
    (por ejemplo, Google Cloud Storage), que permite reutilizar entre hosts,
    programas y reinicios de trabajos. Es útil cuando ejecutas programas de TPU
    en entornos de ejecución como Colab y centros de datos. Está inhabilitado de
    forma predeterminada, pero puedes habilitarlo si especificas una ubicación
    de almacenamiento.

Cada entrada es un ejecutable de TPU compilado que representa un grafo de
cálculo de StableHLO generado a partir de una operación de ATen o de una
secuencia de operaciones de ATen. Los grafos de StableHLO se representan con
huellas digitales que combinan la estructura del grafo, las formas de los
tensores de entrada y salida, y las opciones del compilador, incluidas las
versiones del hardware de TPU, libtpu y el entorno de ejecución de PjRt.

## Compatibilidad con `torch.compile` { #torchcompile-support }

Para mantener una experiencia del usuario coherente con otro hardware que
admite `torch.compile()`, registramos un nuevo
[backend](https://docs.pytorch.org/docs/stable/torch.compiler_custom_backends.html)
de TorchDynamo, que se puede usar de la siguiente manera:
`compiled_model = torch.compile(model,
backend="tpu")`.

Dynamo invoca el backend de TorchTPU y le pasa un `torch.fx.GraphModule` junto
con una lista de tensores de entrada de ejemplo. En ese momento, el entorno de
ejecución traza el grafo proporcionado como si estuviéramos en modo eager, pero
con todas las operaciones diferidas y sin ejecución en la TPU. El entorno de
ejecución genera y compila un único programa de StableHLO para todo el grafo FX
y devuelve un objeto invocable de Python que encapsula la invocación. Ten en
cuenta que TorchTPU no usa `TorchInductor`, sino que reutiliza la pila madura
del compilador XLA, que se desarrolló junto con la arquitectura de TPU.

## Más detalles { #more-detail }

En los siguientes documentos, profundizaremos en estos temas:

*   El [modelo de memoria](./torchtpu_memory_model.md) que admite la TPU y sus
    implicaciones para PyTorch.
*   Los [modos de ejecución](./torchtpu_execution_model.md) que admite
    TorchTPU.
*   La compatibilidad con [kernels personalizados](./custom_kernels.md).
*   La [ejecución distribuida](./distributed_support.md) en clústeres de TPU.
