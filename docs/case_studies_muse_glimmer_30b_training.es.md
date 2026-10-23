# Muse Glimmer 30B: entrenamiento { #muse-glimmer-30b-training }

En este caso práctico, revisaremos nuestro ejemplo independiente de Muse Glimmer
30B, que muestra un bucle de entrenamiento simple para el modelo completo con
FSDP en 4 TPU Ironwood.

Para obtener información sobre cómo ejecutar este ejemplo en TPU de tu clúster
de GKE (Google Kubernetes Engine), consulta el
[README](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop)
del ejemplo. Aquí, en cambio, explicamos algunas de las decisiones de
implementación.

Este ejemplo se enfoca en la simplicidad y la legibilidad. Se ejecuta en un
solo host con 4 TPU Ironwood. Claramente, esta no es una escala realista de
preentrenamiento, pero nuestro objetivo es ofrecerte un ejemplo funcional
que puedas ejecutar, iterar y estudiar a bajo costo.

Este código de PyTorch puro (sin kernels personalizados) alcanza una velocidad
de entrenamiento de entre 2,400 y 2,800 TPS por rango (es decir, entre 19,000 y
22,000 TPS en total), con una longitud de secuencia de 5120. En futuras
actualizaciones, planeamos describir más optimizaciones y prácticas recomendadas
de rendimiento para la ejecución en TPU.

## Definición del modelo { #model-definition }
[Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) sigue
una arquitectura de transformer bastante estándar.
Consta de 52 capas de bloques de autoatención causal y de redes feed-forward
(FFN). Una de cada 4 capas del modelo usa atención completa, y todas las demás
capas usan atención de ventana deslizante (SWA).

El código del modelo es completamente autónomo y está en
[model_fsdp.py](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/model_fsdp.py)
(y en
[rope.py](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/rope.py)
para la codificación posicional rotatoria).

La configuración se especifica en
[model_fsdp.py](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/muse_glimmer_30b/pure_pytorch_fsdp_trainloop/model_fsdp.py):

```python
class Config:
  # General model config
  vocab_size: int = 202048
  num_layers: int = 52
  dmodel: int = 6656
  dff: int = 19968
  dhead: int = 128
  num_q_heads: int = 32
  num_kv_heads: int = 2

  # Details
  swa_size: int = 2048
  # ... omitted ...

  # Perf evaluation config
  batch_size: int = 8  # Local (per rank) batch size
  seq_len: int = 5120
```

La definición de la columna vertebral (backbone) del modelo se reduce a lo
siguiente:

```python
self.embed_tokens = nn.Embedding(config.vocab_size, config.dmodel)
self.embed_norm = nn.RMSNorm(
    (config.dmodel,), eps=config.rmsnorm_eps, elementwise_affine=False
)
self.layers = nn.ModuleList(
    DecoderBlock(config, layer_idx=i) for i in range(config.num_layers)
)
self.norm = nn.RMSNorm((config.dmodel,), eps=config.rmsnorm_eps)
```

Observa que el cabezal del modelo (también llamado "LM head") no está aquí: en
el caso del entrenamiento, suele ser mejor ubicarlo junto con la función de
pérdida. Consulta la sección sobre la pérdida fusionada por segmentos más
adelante.

## torch.compile por capa { #per-layer-torchcompile }

TorchTPU tiene su mejor rendimiento con `torch.compile`, gracias a la capacidad
del compilador XLA de realizar diversas optimizaciones, como fusiones de
operaciones, optimización de memoria y superposición automática de los cálculos
con las comunicaciones.

En este ejemplo, se muestra cómo puedes aplicar `torch.compile` a cada capa del
modelo por separado (en lugar de a todo el modelo). Ambos enfoques funcionan
bien, pero la compilación de todo el modelo puede tardar bastante más y, en
términos de rendimiento, al menos para este modelo, compilar más allá del
alcance de una sola capa no aporta ningún beneficio.

```python
class MuseGlimmer30B(nn.Module):

  def __init__(self, config: Config):
    # ... omitted ...
    self.layers = nn.ModuleList(
        DecoderBlock(config, layer_idx=i) for i in range(config.num_layers)
    )

  def apply_torch_compile(self, backend):
    self.embed_norm = torch.compile(self.embed_norm, dynamic=False, backend=backend)
    self.layers = nn.ModuleList(
        torch.compile(layer, dynamic=False, backend=backend)
        for layer in self.layers
    )
    self.norm = torch.compile(self.norm, dynamic=False, backend=backend)
```

Aquí, `dynamic=False` significa que no hay formas dinámicas (todas las formas
son estáticas durante nuestro entrenamiento), y la variable `backend` solo sirve
para facilitar el cambio entre los backends `"tpu"` e `"inductor"` (para CUDA).

Internamente, TorchTPU almacena en caché las capas compiladas y, como en este
modelo solo hay dos tipos de capas (completa y SWA), esto acelera mucho el
proceso de compilación.

## Checkpointing de activaciones { #activation-checkpointing }

Los pesos del modelo ocupan alrededor de 60 GiB (en bf16) de HBM y cabrían
fácilmente en un solo TensorCore de TPU Ironwood (que tiene 96 GiB).

Sin embargo, durante el entrenamiento también necesitamos memoria para los
gradientes y los estados del optimizador (por ejemplo, el momentum y la varianza
de AdamW). Y, lo que es más importante, el uso de memoria puede quedar dominado
fácilmente por las activaciones intermedias que necesita la funcionalidad de
autograd. El checkpointing de activaciones (también llamado rematerialización)
es una técnica muy común para abordar esto último.

Como compilamos el modelo por capa, es muy natural aplicar el checkpointing de
activaciones con la misma granularidad. Para ello, usamos la funcionalidad
estándar de
[`torch.utils.checkpoint`](https://docs.pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint):

```python
class DecoderBlock(nn.Module):
  # ... omitted ...

  def forward(self, h: torch.Tensor, rope_sin_cos: tuple) -> torch.Tensor:
    return torch.utils.checkpoint.checkpoint(
        self.forward_impl,
        h,
        rope_sin_cos,
        use_reentrant=False,
    )

  def forward_impl(self, h: torch.Tensor, rope_sin_cos: tuple) -> torch.Tensor:
    h = h + self.post_attention_layernorm(
        self.self_attn(self.input_layernorm(h), rope_sin_cos)
    )
    h = h + self.post_feedforward_layernorm(
        self.mlp(self.pre_feedforward_layernorm(h))
    )
    return h
```

> **Note:** El checkpointing de activaciones (AC) se aplica dentro del método
> `forward` de la capa del modelo (módulo `DecoderBlock`), y `torch.compile` se
> aplica fuera, sobre la propia capa. Este orden es importante.
> Podrías mantener el AC dentro de la capa y aplicar `torch.compile` a todo el
> modelo, y funcionaría bien.
> Pero no puedes intercambiarlos. Por ejemplo, si aplicaras el AC de la misma
> forma, pero aplicaras `torch.compile` dentro, por separado, a las subcapas de
> atención y FFN, no funcionaría como se espera.

`torch.compile` funciona especialmente bien con AC en TorchTPU: durante el pase
hacia atrás (global), cada capa realiza un pequeño pase conjunto hacia adelante
y hacia atrás (debido a la rematerialización) y, como el compilador del backend
de TPU (XLA) ve todo el grafo de cálculo conjunto hacia adelante y hacia atrás,
puede hacer optimizaciones adicionales e incluso reducir el uso de memoria de
los tensores intermedios.

## FSDP "ingenuo" { #naive-fsdp }

En este ejemplo, quisimos mostrar la forma más simple posible de hacer un
entrenamiento con FSDP (paralelismo de datos totalmente fragmentado) usando
directamente las operaciones colectivas nativas de PyTorch.

> **Tip:** Para ver una descripción aún más simple de este enfoque, consulta
> nuestro
> [ejemplo de FSDP manual](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual).

Hacemos FSDP fragmentando manualmente todos los pesos de proyección de atención
y FFN, por ejemplo:

```python
# QKV projections are fully sharded:
self.q_proj = nn.Linear(dm, num_q_heads * dh // world_size, bias=False)
```

> **Note:** La capa `nn.Linear` de PyTorch almacena su matriz de pesos
> transpuesta, por lo que aquí fragmentamos en la 2.ª dimensión. La forma real
> del peso será `[num_q_heads * dh / world_size, dm]`. Esto es importante
> porque las operaciones colectivas all-gather / reduce-scatter de PyTorch
> funcionan en la 1.ª dimensión.

Luego, durante el pase hacia adelante, usamos las operaciones colectivas
funcionales de PyTorch para reunir los pesos antes de usarlos:

```python
# Functional collectives
import torch.distributed._functional_collectives as func_col

# ... all-gather:
q_proj_w = func_col.all_gather_single(self.q_proj.weight, gather_dim=0, group=self.pg)

# ... use full weight directly:
q = F.linear(h, q_proj_w).view(local_bsz, seq_len, num_q_heads, dh)
```

La misma idea se aplica a todas las demás proyecciones (Q/K/V/out/gate de
atención y up/gate/down de FFN).

Lo fundamental es que las operaciones colectivas funcionales, como
`func_col.all_gather_single`, están registradas en el motor de autograd de
PyTorch, por lo que durante el pase `.backward()` obtenemos automáticamente las
operaciones colectivas reduce-scatter correspondientes (es decir, no necesitamos
escribirlas nosotros).

Notarás que esta implementación es relativamente ingenua, ya que no intenta
precargar los pesos de la siguiente capa mientras se ejecuta la capa actual.
¿Esto no haría que los TensorCores de la TPU se detengan o queden inactivos
mientras esperan que finalicen las operaciones colectivas?

Como aplicamos `torch.compile` a toda la capa, el compilador de TorchTPU (XLA)
puede reordenar las operaciones colectivas y descargarlas para ejecutarlas de
forma asíncrona en el SparseCore, tanto en el pase hacia adelante como en los
pases conjuntos hacia adelante y hacia atrás. En este caso, esto resulta
suficiente para lograr una superposición casi perfecta entre las comunicaciones
y el cálculo.

## LM head y pérdida de entropía cruzada fusionados por segmentos { #chunkwise-fused-lm-head-cross-entropy-loss }

De forma predeterminada, el ejemplo usa un tamaño de batch local de 8 y una
longitud de secuencia de 5120. Si calculáramos la pérdida de entrenamiento de
entropía cruzada de forma ingenua, tendríamos que materializar un tensor
intermedio de logits con la forma `[8, 5120, 202048]`, donde el último número es
el tamaño del vocabulario del modelo. En bf16, esto por sí solo ya requeriría
unos 16 GiB de memoria HBM, lo cual es un desperdicio excesivo.

Una solución común a este problema es calcular la pérdida por segmentos (chunks)
y fusionar la matmul de los pesos de la matriz de vocabulario con la pérdida de
entropía cruzada. Un enfoque es segmentar a lo largo de la dimensión del
vocabulario (lo que requiere online-softmax y es difícil de expresar en PyTorch
puro), y otro enfoque es segmentar a lo largo de la dimensión de secuencia.
Aquí hacemos lo segundo:

```python
class MuseGlimmerChunkwiseFusedLMHeadLoss(nn.Module):
  # ... omitted ...

  for i in range(0, h.shape[0], chunk_size):
    h_chunk.data.copy_(h[i : (i + chunk_size)])
    tgt_chunk.data.copy_(tgt[i : (i + chunk_size)])

    self.fused_loss_fn_fwd_bwd(
        h_chunk, tgt_chunk, self.lm_head.weight, alpha, cap, loss_scaler
    )

    h_chunk_grads.append(h_chunk.grad)
    h_chunk.grad = None

  return torch.cat(h_chunk_grads, dim=0).view(bsz, seq_len, dm)
```

`fused_loss_fn_fwd_bwd` hace referencia a la versión compilada con
`torch.compile` de la función estática que calcula, en un solo segmento de
tokens, la matmul de vocabulario fusionada + la pérdida de entropía cruzada + el
pase hacia atrás:

```python
@staticmethod
def _chunk_fused_lmhead_loss_fwd_bwd(
    h, tgt, lm_head_w, alpha, cap, loss_scaler
):
  logits = F.tanh(F.linear(h, lm_head_w) * (alpha / cap)) * cap
  loss = F.cross_entropy(input=logits, target=tgt, reduction="sum")
  loss *= loss_scaler
  # Note: in recent versions of PyTorch, .backward is no longer a graph break!
  loss.backward()
  return
```

Para aprovechar al máximo este enfoque, incluso realizamos el pase hacia atrás
en el segmento. En las versiones más recientes de PyTorch, invocar
`loss.backward()` ya no causa una interrupción de grafo, lo que permite que el
compilador de TorchTPU (XLA) realice más optimizaciones y fusiones.

## Qué falta en este ejemplo { #whats-missing-in-this-example }

Este ejemplo ilustra solo el bucle de entrenamiento básico y, para mayor
claridad, omite varios detalles importantes para cualquier configuración de
entrenamiento práctica:

 - carga de datos y tokenización
 - carga y guardado de checkpoints del modelo
 - supervisión del estado de los trabajos y diagnóstico de convergencia
 - tolerancia a fallas

Para una configuración de entrenamiento realista y de nivel de producción con
TorchTPU, te recomendamos consultar
[TorchTitan](https://github.com/google-pytorch/torchtitan/).
