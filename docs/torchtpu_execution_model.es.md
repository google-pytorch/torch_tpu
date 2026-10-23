# El modelo de ejecución de TorchTPU { #torchtpu-execution-model }

Cuando ejecutas un programa de PyTorch con TorchTPU, ocurren muchas cosas
internamente para optimizar la ejecución. Comprender este modelo te ayuda a
saber cómo ajustar tu experiencia y aprovechar algunas de las ventajas únicas de
TorchTPU.

## Canalización de ejecución { #execution-pipeline }

### Crea una operación diferida { #create-deferred-op }

Cuando se ejecuta una operación de PyTorch (o simplemente "op") en el backend de
TorchTPU, al principio no se realiza ningún trabajo en la TPU.

En cambio, el resultado lógico del cálculo se empaqueta en una operación
diferida (deferred op). Es un objeto en la memoria del lado de la CPU que
contiene toda la información necesaria para realizar esta operación: sus
entradas, los pasos lógicos que se deben seguir y la forma y el tipo de las
salidas, entre otros datos.

El resto de la canalización, que reemplaza esta operación diferida simbólica por
datos reales, se denomina materialización.

### Pon en cola la operación en un stream { #enqueue-op-on-a-stream }

TorchTPU admite la interfaz `torch.Stream`. Cada vez que se crea una
DeferredOp, se envía al stream activo actual para ponerla en cola y ejecutarla
más adelante.

En este punto, el subproceso que hizo la llamada termina; todo lo que sigue
ocurre de forma asíncrona en segundo plano (excepto en el modo launch-blocking;
encontrarás más información sobre esto más adelante).

### Analiza el stream y prepara el plan de ejecución { #analyze-stream-and-prepare-execution-plan }

Cuando hay trabajo pendiente en un stream, en algún momento un subproceso en
segundo plano toma el trabajo de ese stream. Luego, este subproceso en segundo
plano analiza en conjunto el trabajo del stream y considera muchos factores
diferentes, como los siguientes:

*   ¿En cuántas ejecuciones individuales se debe dividir?
*   ¿Cuáles son las entradas y salidas de cada bloque de ejecución?
*   ¿Qué nivel de optimización de compilación se debe aplicar a cada uno?
*   ¿Ya vimos cada unidad de cálculo antes o es nueva?
*   ¿En qué núcleo (TensorCore o SparseCore) se debe ejecutar este trabajo?

Al considerar el trabajo pendiente en conjunto, el subproceso en segundo plano
puede tomar mejores decisiones de ejecución.

Una vez que el subproceso en segundo plano identifica las salidas finales de
cada bloque, marca estas salidas como pendientes para indicar que se ejecutarán
en el futuro y pasa al siguiente paso.

### Compila (u obtén) ejecutables { #compile-or-fetch-executables }

Para cada etapa de ejecución, el subproceso en segundo plano consulta una caché
de compilación para ver si ya se compiló un ejecutable con la configuración
correcta. De forma predeterminada, es una caché en memoria local (nivel 1), con
opciones para habilitar una caché persistente local del dispositivo (nivel 2) o
cachés remotas (nivel 3) para minimizar los costos de recompilación.

Si se produce un error de caché, el grafo de compilación de la unidad se
convierte a MLIR (principalmente StableHLO, excepto en el caso de los kernels
personalizados definidos por el usuario) y se entrega a un grupo de subprocesos
de compilación. Este grupo de subprocesos compila de forma asíncrona muchas
unidades de compilación en paralelo y las devuelve al subproceso de trabajo en
segundo plano cuando están listas.

### Inicia la ejecución { #start-execution }

Una vez que se recopilan las entradas y se prepara el ejecutable, se pone en
cola una tarea de ejecución en el entorno de ejecución de XLA,
[PjRt](https://openxla.org/xla/pjrt).

PjRt devuelve promesas para esta salida futura, que luego se conectan a las
salidas para indicar que se están materializando.

### Espera a que finalice (opcional) { #optional-wait-for-completion }

Para saber cuándo finaliza el trabajo, puedes registrar un `torch.Event` en el
stream y usar el método `torch.Event.query` para inspeccionarlo.

Como alternativa o como complemento, puedes bloquear el stream con cualquier
instrucción de "sincronización" (`torch.Event.synchronize`,
`torch.Stream.synchronize` o `torch.accelerator.synchronize` para todo el
dispositivo), que pausa tu código de Python hasta que la TPU indique que
terminó.

## Modos de ejecución { #execution-modes }

La canalización de programación de TorchTPU es flexible y admite varias
modalidades de ejecución diferentes.

### Modo eager estricto { #strict-eager-mode }

También conocido como "eager predeterminado" o "defer never", este modo es la
experiencia conocida de PyTorch lista para usar. En este modo, cada operación
que se agrega a la cola marca de inmediato ese stream para su ejecución. Como
resultado, el proceso de trabajo de materialización en segundo plano procesa
las operaciones de a una.

Estos grafos de ejecución de una sola operación tienen un potencial de
optimización mínimo, por lo que se compilan con el nivel de compilación más bajo
`-O1` para maximizar la velocidad de compilación. Como hay muchas compilaciones
individuales, se paralelizan de forma muy eficaz en el grupo de subprocesos de
compilación, y el nivel de reutilización de las operaciones individuales es muy
alto. Como resultado, la sobrecarga de compilación es mínima mientras se
"precalienta" la caché de compilación.

Ten en cuenta que, aunque este modo ejecuta las operaciones de inmediato y de
forma continua, lo hace de forma asíncrona. Aún se necesita una sincronización para esperar a que la TPU
termine si quieres observar los resultados, y cualquier error del entorno de
ejecución puede aparecer mucho después de la operación que lo causó.

### Modo eager de depuración { #debug-eager-mode }

También conocido como "launch blocking", se habilita si estableces la variable
de entorno `TORCH_TPU_LAUNCH_BLOCKING=1`. Está pensado solo para la depuración
interactiva, no para el uso en producción.

Este modo es igual al eager estricto (ejecución operación por operación),
excepto que el host se bloquea y se sincroniza en cada operación. Esto ralentiza
la ejecución de forma artificial, pero garantiza que las fallas se informen en
cuanto ocurren, lo que ayuda a identificar los errores con precisión.

### `torch.compile` { #torchcompile }

Para obtener el máximo rendimiento, `torch.compile` ofrece un mayor potencial de
optimización y una menor sobrecarga en el entorno de ejecución.

Cuando compilas una función o un módulo específicos con `torch.compile`, el
backend de TorchTPU recibe el alcance completo del objetivo de compilación como
un [grafo FX](https://docs.pytorch.org/docs/stable/fx.html). Esto proporciona la
máxima cantidad de información sobre la ejecución prevista, junto con la
garantía de que no cambiará, lo que nos permite hacer optimizaciones agresivas
que serían demasiado riesgosas en el modo eager.

Como backend de modo compilado, TorchTPU realiza una traza inicial (con tensores
"marcadores de posición", similares a FakeTensor, pero que conservan metadatos
específicos de la TPU) y produce un grafo de cálculo completamente diferido que
replica el grafo FX. Una vez creado, le proporcionamos este programa de
StableHLO a XLA para que lo compile con el nivel de optimización más alto
`-O2`. Luego, este programa se empaqueta en un objeto invocable de Python y se
devuelve al usuario para su ejecución posterior.

Cuando se invoca este objeto, omitimos de inmediato la mayoría de los pasos de
la ejecución eager. No se crean nuevas DeferredOps, no se necesita ningún otro
análisis del grafo y el ejecutable ya está disponible, por lo que no se requiere
almacenamiento en caché. En cambio, el ejecutable se envía de inmediato a PjRt
para su ejecución, lo que minimiza el costo del lado del host con un grafo de
cálculo completamente optimizado por el compilador XLA.

### Modo eager optimizado { #optimized-eager-mode }

También conocido como modo "Defer and Fuse", se habilita si estableces
`TORCH_TPU_DEFER_AND_FUSE=1`. Es un modo híbrido nuevo que lleva muchas de las
ventajas de `torch.compile` al modo eager.

En este modo, crear una nueva DeferredOp no inicia la ejecución de inmediato;
permitimos que las operaciones permanezcan diferidas entre llamadas de Python,
en espera en la cola del stream.

Comenzamos la materialización cuando se cumple una de dos condiciones:

*   El usuario realiza alguna acción que fuerza la ejecución. Las más comunes
    son iniciar una transferencia del dispositivo al host o una llamada
    explícita de sincronización.
*   Detectamos una repetición en las operaciones despachadas, lo que indica que
    estamos en un bucle y que debemos compilar una iteración de ese bucle para
    reutilizarla con facilidad.

Esto activa una materialización en segundo plano, similar al modo eager
predeterminado, excepto que el grafo que se compila es más grande que una sola
operación. Se omite cualquier tensor que ya se haya descartado o sobrescrito
cuando se activó la señal de materialización. Por lo general, se fusionan en
operaciones posteriores o se eliminan por completo si la salida se descarta sin
leerse. (Nota: La eliminación de código muerto se omite para las operaciones
colectivas distribuidas, a fin de evitar interbloqueos entre TPU).

Como esto genera grafos de compilación más grandes, usamos el nivel de
optimización 2 (`-O2`) para mejorar el tiempo de ejecución. Como compilamos una
menor cantidad de grafos más grandes, el beneficio de paralelización del grupo
de subprocesos de compilación es menor, por lo que los tiempos de compilación
serán más largos que en el modo eager estricto.

Aunque no está tan optimizado como el modo `torch.compile`, el modo eager
optimizado muestra una mejora de rendimiento de hasta un 50% respecto del modo
eager predeterminado, sin cambios en el código del modelo.

## Errores comunes { #pitfalls }

Aunque esta canalización de ejecución está optimizada tanto para la latencia
como para la capacidad de procesamiento, hay casos en los que el rendimiento
puede verse afectado.

### Todos los modos: Flujo de control del lado de Python y formas dependientes de los datos { #all-modes-python-side-control-flow-and-data-dependent-shapes }

En PyTorch, es fácil forzar de forma accidental y silenciosa una transferencia
del dispositivo al host sin querer. Una operación como "`if my_tensor.all()`"
construye de forma silenciosa un tensor de valor booleano, sincroniza el stream
y copia ese booleano a la CPU antes de continuar con el flujo de control.

En el modo eager predeterminado, esto no fuerza nuevas interrupciones de grafo,
pero sí obliga a que se vacíe toda la canalización de ejecución del stream antes
de continuar. Peor aún, en el modo eager optimizado o en el modo compilado, este
punto de sincronización forzada provoca una interrupción de grafo, ya sea en el
stream fusionado de forma eager o en el grafo FX, lo que crea una barrera de
optimización.

Un comportamiento similar también puede ocurrir con operaciones de PyTorch que
tienen formas de salida dependientes de los datos, como `torch.masked_select`.
Como el tamaño del tensor de salida se desconoce hasta que se evalúa la máscara
de entrada, esto también fuerza una sincronización del stream totalmente
bloqueante y una transferencia de los metadatos de forma del dispositivo al host
antes de continuar.

### Modo eager estricto: Los ejecutables muy pequeños causan microburbujas { #strict-eager-mode-tiny-executables-cause-microbubbles }

Poner en cola un ejecutable con PjRt lleva poco tiempo: por lo general, 50
microsegundos o menos. En la mayoría de los casos, cada ejecutable tarda más que
esto (aunque sea solo 1 o 2 milisegundos) y oculta por completo esta latencia,
lo que lleva a un estado estable en el que la TPU funciona de forma continua con
la máxima eficiencia.

Sin embargo, si se ponen en cola demasiados ejecutables por debajo de este
umbral (como cuando usas el modo eager estricto con tamaños de tensor pequeños),
la TPU puede quedarse sin trabajo. Cuando la TPU se ejecuta más rápido de lo que
la CPU puede poner en cola más trabajo, la ejecución de la TPU se interrumpe por
"microburbujas" pequeñas pero frecuentes, ya que la velocidad de despacho se
convierte en el cuello de botella.

Para evitarlo, asegúrate de que tus modelos realicen suficiente trabajo en cada
operación para compensar esta pequeña sobrecarga de inicio, o usa el modo eager
optimizado para fusionar automáticamente las ejecuciones en bloques de
compilación sobre la marcha.

### Modo compilado: Vistas como argumentos o salidas { #compiled-mode-views-as-arguments-or-outputs }

Cuando llamas a un ejecutable compilado, PjRt requiere que los valores de
entrada sean contiguos, tengan desplazamiento cero y sean de un tipo de datos
específico. Si un argumento Tensor es una vista no trivial o se convirtió
mediante bitcast desde el tipo de datos original, TorchTPU hace una copia
temporal del tensor en modo eager (estricto u optimizado) para llamar
correctamente al ejecutable compilado.

Para evitar esta copia temporal, asegúrate de que todas las entradas de una
función compilada sean contiguas. Si planeas llamar a varias funciones
compiladas en secuencia, asegúrate de que la salida de las funciones anteriores
sea contigua para evitar la fricción del modo eager entre las funciones
optimizadas.

### Modo eager optimizado: Retención de tensores redundantes { #optimized-eager-mode-retaining-redundant-tensors }

En todos los modos eager, las operaciones se ejecutan en el orden en que se
crean y, cada vez que se activa una señal de materialización, se materializan
todos los tensores activos en ese momento.

Si alguno de estos tensores es un tensor intermedio sin importancia, el
mecanismo de defer-and-fuse puede identificarlo erróneamente como una salida
necesaria y materializarlo, lo que podría reducir la eficiencia del grafo de
compilación.

Para mejorar la fusión y minimizar el uso de memoria, asegúrate de liberar los
tensores en cuanto ya no los necesites, o usa operaciones in situ para
reemplazar tensores obsoletos por valores actuales.

## Conclusiones { #takeaways }

El modelo de ejecución diferida de TorchTPU ofrece una canalización eficiente
con distintos equilibrios entre el tiempo de compilación y el tiempo de
ejecución.

*   **Para minimizar el tiempo de compilación:** Usa el modo eager estricto (el
    predeterminado). Este modo inicia la ejecución de inmediato y paraleliza la
    compilación para lograr el menor tiempo posible hasta el primer resultado.
*   **Para obtener la máxima capacidad de procesamiento una vez precalentado y
    una latencia mínima:** Usa `torch.compile`. Esto elimina casi toda la
    sobrecarga del modo eager y maximiza el potencial de optimización del
    compilador.
*   **Para los casos en que se prefiere un tiempo de ejecución más rápido, pero
    `torch.compile` no es una opción:** Considera usar el modo eager
    optimizado.
*   **Para depurar:** Usa el modo launch-blocking para forzar que la CPU y la TPU
    avancen de forma sincronizada.
*   **Evita las sincronizaciones forzadas:** Mantén un conjunto mínimo de
    tensores contiguos y usa operaciones con alta densidad de cálculo para que
    la TPU siempre tenga trabajo productivo que hacer.
