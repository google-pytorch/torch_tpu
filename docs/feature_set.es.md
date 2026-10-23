# Conjunto de funciones compatibles con TorchTPU y trabajo futuro { #torchtpu-supported-feature-set-and-future-work }

## Conjunto de funciones compatibles actualmente { #current-supported-feature-set }

En la siguiente tabla, se muestra la cobertura de funciones actual a partir de
la versión inicial de código abierto.

Para ayudarte a evaluar TorchTPU para tus cargas de trabajo, clasificamos las
funciones en dos niveles:

*   **Validadas:** Capacidades que tienen pruebas continuas de extremo a
    extremo, cobertura automatizada de CI/CD y verificación con comparativas de
    rendimiento (benchmarks) en arquitecturas de modelos estándar.
*   **Experimentales:** Implementaciones en etapa inicial, APIs emergentes o
    funciones prototipo. Son funcionales y están disponibles para recibir
    comentarios tempranos, pero su comportamiento o su API pueden cambiar
    según los aportes de la comunidad.

> **Note:** TorchTPU evoluciona rápidamente de forma abierta. Para conocer el
> estado más reciente o solicitar funciones específicas, visita nuestro
> [GitHub](https://github.com/google-pytorch/torch_tpu).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
|  | **Validadas** | **Experimentales** |
| ----- | ----- | ----- |
| **Participación de la comunidad de PyTorch** | <ul><li>Versión de código abierto de TorchTPU, vLLM, SGLang, TorchTitan, TorchRec</li><li>Aceptación de PR externas</li><li>Documentación e instructivos</li><li>CI/CD</li><li>Colab</li></ul> | <ul><li>Agente de TPU para la migración de ATEN</li></ul> |
| **Compatibilidad con TPU** | <ul><li>v5e, v5p, v6e, v7</li></ul> |  |
| **Funcionalidad básica** | <ul><li>Facilidad de migración</li><li>Modo eager y compilado combinados</li><li>FW \+ BW</li><li>Torch.profile</li><li>Operaciones dinámicas (recompilación)</li></ul> | <ul><li>SymInts</li><li>Tensores irregulares (jagged)</li></ul> |
| **Arquitecturas de modelos** | <ul><li>Transformer (denso \+ MoE): Qwen, Llama, Muse Glimmer, Gemma</li><li>Difusión: WAN, FLUX</li><li>CNN</li></ul> | <ul><li>GLM, Kimi, DeepSeek, GPT OSS y muchos más (mejora progresiva)</li><li>Mamba</li><li>LSTM/GRU</li><li>DLRM</li></ul> |
| **Compatibilidad con kernels** | <ul><li>Pallas</li><li>Tokamax</li></ul> | <ul><li>Helion</li></ul> |
| **Cuantización** | <ul><li>FP8/FP4 de tensor completo</li><li>FP8 por canal</li></ul> |  |
| **Distribuido** | <ul><li>Malla óptima</li><li>SimpleFSDP</li><li>Checkpointing</li><li>Descarga de activaciones</li><li>Varias porciones (multi-slice) de hasta 8K chips</li></ul> | <ul><li>FSDPv2 (bajo rendimiento)</li><li>MPMD (bajo rendimiento)</li><li>TorchComm</li></ul> |
| **Integraciones con el ecosistema** | <ul><li>Hugging Face</li><li>Ray</li><li>vLLM / SGLang</li><li>TorchTitan</li><li>TorchRec</li></ul> |  |
| **Soluciones listas para producción** | <ul><li>Inferencia (varios hosts, desagregación P/D)</li><li>Ajuste fino (hasta \~8K chips)</li></ul> |  |
<!-- mdformat on -->

## Trabajo futuro { #future-work }

En esta sección, se describen nuestros horizontes de desarrollo y exploraciones
futuras, no un calendario de lanzamientos comprometido.

Como proyecto de código abierto que prioriza a la comunidad, nuestras
prioridades de desarrollo responden de forma intencional a los comentarios de
la comunidad y a la adopción por parte de los desarrolladores. Las funciones,
los plazos y los enfoques de arquitectura pueden cambiar a medida que
aprendemos qué cargas de trabajo te importan más.

Organizamos el trabajo próximo en dos horizontes amplios:

*   **Enfoque a corto plazo:** Áreas de diseño activo, prototipado en curso y
    refuerzo del rendimiento que el equipo principal prioriza actualmente.
*   **Exploración futura:** Exploraciones de arquitectura a largo plazo,
    posibles integraciones con el ecosistema y elementos pendientes
    solicitados por la comunidad que investigamos activamente.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
|  | **Enfoque a corto plazo** | **Exploración futura** |
| ----- | ----- | ----- |
| **Participación de la comunidad de PyTorch** | <ul><li>Más documentación, instructivos y blogs</li></ul> | <ul><li>Agente de TPU para la migración de kernels</li></ul> |
| **Compatibilidad con TPU** | <ul><li>v8i, v8t</li></ul> |  |
| **Funcionalidad básica** | <ul><li>Streams</li><li>Captura de grafos</li><li>Modo eager con tiempo de inicio rápido</li><li>Operaciones dinámicas (sin recompilación)</li></ul> |  |
| **Arquitecturas de modelos** | <ul><li>Compatibilidad con modelos desde el día 1</li></ul> |  |
| **Cuantización** | <ul><li>FP8/FP4 por bloques con torchao (rendimiento máximo en TPUv8t)</li></ul> |  |
| **Distribuido** | <ul><li>FSDPv2 (con rendimiento)</li><li>MPMD (con rendimiento)</li><li>Entrenamiento con \> 8K chips</li><li>Checkpointing de varios niveles</li><li>Goodput</li></ul> | <ul><li>Entrenamiento con >> 8K chips</li></ul>  |
| **Integraciones con el ecosistema** | <ul><li>VeRL</li></ul> |  |
| **Soluciones listas para producción** | <ul><li>RL</li></ul> | <ul><li>Preentrenamiento</li></ul> |
<!-- mdformat on -->
