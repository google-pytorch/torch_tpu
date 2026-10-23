# Prácticas recomendadas para FSDP { #fsdp-best-practices }

Fully Sharded Data Parallel (FSDP) fragmenta (shards) los parámetros del
modelo, los gradientes y el estado del optimizador entre los rangos de
paralelismo de datos, de modo que cada núcleo de TPU almacena solo una fracción
de `1 / world_size` del modelo. Antes de que se ejecute una capa, sus pesos
completos se reconstruyen con un **all-gather**; durante el pase hacia atrás,
los gradientes se combinan y se vuelven a fragmentar con un
**reduce-scatter**.

En esta guía, se explica la forma recomendada de usar FSDP en TorchTPU y se
indican ejemplos ejecutables en
[`examples/fsdp`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp).

En esta guía, aprenderás lo siguiente:

1.  **Qué enfoque de FSDP elegir** en TorchTPU.
2.  **SimpleFSDP (recomendado)**: FSDP orientado a la compilación con una sola
    llamada de encapsulamiento.
3.  **FSDP manual**: colocar las operaciones colectivas directamente en la
    definición del modelo.
4.  **Prácticas recomendadas** que se aplican a todos los enfoques.
5.  **Ejecución de los ejemplos**: comandos de inicio y lo que verifica cada
    ejemplo.

## 1. Elige un enfoque de FSDP { #1-choosing-an-fsdp-approach }

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
    | Enfoque | Cuándo usarlo | Ejemplo |
    | :------- | :------------- | :------ |
    | **SimpleFSDP** (`simple_fsdp.data_parallel`) | **Opción predeterminada recomendada.** Quieres FSDP o HSDP sin cambiar la definición del modelo.| [`fsdp_to_simple_fsdp/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp) |
    | **FSDP manual** (operaciones colectivas manuales) | Eres responsable del código del modelo y quieres que el código de comunicación esté junto al de cálculo para ajustar el rendimiento. | [`manual/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual) |
    | **PyTorch FSDPv2** (`fully_shard`) | Código existente que aún no se migró. Funciona, pero sus hooks del entorno de ejecución causan interrupciones de grafo y, para lograr un rendimiento competitivo, requiere la próxima compatibilidad con la descarga de streams al SparseCore de la TPU.
.| [Guía de migración](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md) |
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

## 2. Recomendado: SimpleFSDP { #2-recommended-simplefsdp }

[`simple_fsdp`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/simple_fsdp.py)
es una implementación de FSDP diseñada para funcionar con `torch.compile`. Se
introdujo en TorchTitan, y en `examples/fsdp/simple_fsdp.py` se proporciona una
copia autónoma que solo depende de PyTorch.

SimpleFSDP almacena cada parámetro como un `DTensor` fragmentado. Cuando se
accede a un parámetro, una parametrización llama a `DTensor.redistribute` para
hacer un all-gather. El pase hacia atrás de autograd de `redistribute` realiza
el reduce-scatter de los gradientes. No hay hooks en el entorno de ejecución,
por lo que todo el paso se compila sin interrupciones de grafo. Todo el modelo
se fragmenta con una sola llamada a `simple_fsdp.data_parallel(...)`, sin
encapsulamiento por capa ni ajuste de la precarga, y HSDP es compatible con
`mode="hybrid_shard"` en un `DeviceMesh` 2D.

Para aprender a usar SimpleFSDP, consulta lo siguiente:

*   [README de SimpleFSDP](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/README.md):
    inicio rápido y opciones disponibles.
*   [Instructivo de FSDPv2 a SimpleFSDP](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md):
    cómo migrar código existente de `fully_shard`.

## 3. Alternativa: FSDP manual { #3-alternative-manual-fsdp }

Si controlas la definición del modelo, puedes escribir FSDP directamente con
operaciones colectivas manuales (`torch.distributed._functional_collectives`) y
`torch.compile`. Cada capa almacena solo su fragmento de pesos y hace un
all-gather del peso completo en el pase hacia adelante. Las operaciones
colectivas funcionales tienen fórmulas de autograd, por lo que el pase hacia
atrás del all-gather es un reduce-scatter, y no necesitas escribir una
`torch.autograd.Function` personalizada.

Así, el código de comunicación se mantiene junto al código de cálculo, lo que
puede facilitar el ajuste del rendimiento. La desventaja es que debes cambiar la
definición del modelo.

Para aprender a implementar FSDP manual, consulta la
[guía de FSDP manual](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual/manual_fsdp.md).

## 4. Prácticas recomendadas { #4-best-practices }

Estas recomendaciones se aplican a SimpleFSDP y a FSDP manual.

1.  **Compila todo el paso de entrenamiento con
    `torch.compile(backend="tpu")`.** Si no puedes compilar todo el paso,
    compila al menos el pase hacia adelante, la pérdida y el pase hacia atrás
    juntos. Compilar solo el módulo del modelo no produce un único grafo, lo que
    limita la capacidad de XLA para superponer las operaciones colectivas con el
    cálculo.
2.  **Usa checkpointing de activaciones en cada capa fragmentada.** Encapsula
    el pase hacia adelante de cada capa en
    `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`. Esto libera
    los pesos no fragmentados después del pase hacia adelante de cada capa, en
    lugar de mantenerlos hasta el pase hacia atrás, lo que da el comportamiento
    de memoria de ZeRO-3 que se espera de FSDP.
3.  **Haz que las dimensiones fragmentadas sean divisibles por el tamaño de la
    malla.** FSDP manual lo requiere. SimpleFSDP replica los parámetros que son
    demasiado pequeños para fragmentarse. Usa `shard_dim` para elegir una
    dimensión que se divida de forma exacta.
4.  **Usa una semilla distinta en cada rango cuando inicialices pesos
    fragmentados.** Si todos los rangos inicializan su propio fragmento con la
    misma semilla, el peso completo es el mismo bloque aleatorio repetido en
    todos los rangos, lo que puede perjudicar la convergencia. Para obtener
    fragmentos reproducibles y distintos, usa en cada rango una semilla que
    dependa del rango, como `torch.manual_seed(base_seed + dist.get_rank())`.
5.  **Haz que todos los rangos ejecuten el mismo programa.** No coloques
    operaciones que fuercen la materialización (como `.item()`) dentro de ramas
    específicas de un rango, como `if
    rank == 0`.
6.  **Usa HSDP cuando la fragmentación completa en toda la malla no escale.**
    Usa `mode="hybrid_shard"` con una malla 2D cuando tu porción de ICI sea
    demasiado grande para fragmentar el modelo de forma eficaz, o cuando
    fragmentar en ella genere demasiado tráfico de red. Cuando ejecutas en
    varias porciones, casi siempre es mejor replicar (paralelismo de datos) en
    la DCN, que es más lenta, y fragmentar dentro de cada porción.
7.  **Verifica la memoria y los resultados numéricos antes de escalar.**
    Compara con una referencia sin fragmentar o de FSDPv2 mediante
    `assert_close` y mide la memoria retenida en el límite entre el pase hacia
    adelante y el pase hacia atrás. Los ejemplos incluyen una verificación de
    paridad (`fsdp_to_simple_fsdp_worker.py`) y una prueba de memoria
    (`manual_fsdp_memory_test.py`).

## 5. Ejecuta los ejemplos { #5-running-the-examples }

Cada ejemplo incluye una secuencia de comandos de inicio que se ejecuta en 8
núcleos de TPU (por ejemplo, `v6e-8`):

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Ejemplo | Comando de inicio | Qué verifica |
:-------- | :------------- | :------------- |
| [`fsdp_to_simple_fsdp/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp) | `./run_fsdp_to_simple_fsdp.sh` | FSDPv2 compilado y SimpleFSDP <br> compilado tienen pérdidas y <br> predicciones coincidentes, y se <br> informan sus tiempos por paso. |
| [`manual/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual) | `./launch.sh` o `python3 launch.py` | Los gradientes y las predicciones de FSDP manual <br> coinciden con una referencia de CPU de un solo dispositivo. |
| [`manual/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual) | `python3 manual_fsdp_memory_test.py` | Los pesos no fragmentados se liberan después del <br> pase hacia adelante. |
<!-- mdformat on -->
