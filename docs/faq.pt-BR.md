# Perguntas frequentes sobre o TorchTPU { #torchtpu-faq }

## Qual é a diferença entre PyTorch/XLA e TorchTPU? { #what-is-the-difference-between-pytorchxla-and-torchtpu }

Historicamente, o PyTorch/XLA serviu como um framework de ponte, fortemente
dependente de componentes do runtime do JAX e de APIs especializadas. O TorchTPU
oferece um backend nativo do PyTorch para
[Google Cloud TPUs](https://cloud.google.com/tpu), sem nenhuma dependência do
runtime do JAX, e as cargas de trabalho PyTorch existentes funcionam de imediato
com mudanças triviais no código (por exemplo, `.to("cuda")` → `.to("tpu")`)

## Qual é a diferença entre TorchAX e TorchTPU? { #what-is-the-difference-between-torchax-and-torchtpu }

O TorchAX (também conhecido como torch_xla2) é uma implementação que intercepta
operações do PyTorch no nível do Python e as converte (lowering) para XLA
(geralmente via JAX). Embora o TorchAX tenha sido muito usado em benchmarks de
inferência e prototipagem, o TorchTPU é o novo padrão oficial para uma
experiência drop-in e nativa do PyTorch em TPUs.

## O que está acontecendo com o PyTorch/XLA? { #what-is-happening-to-pytorchxla }

O PyTorch/XLA foi descontinuado e substituído pelo
[TorchTPU](https://github.com/google-pytorch/torch_tpu), que agora é público.

## O TorchTPU vai substituir o JAX? { #is-torchtpu-replacing-jax }

Não. O TorchTPU foi projetado para complementar o ecossistema de TPU, oferecendo
um ponto de entrada nativo do PyTorch para Cloud TPUs. Ele permite que
desenvolvedores com bases de código PyTorch existentes executem diretamente em
TPUs sem reescrita ou com reescrita mínima, eliminando a necessidade de
reescrever modelos em JAX ou de manter bases de código separadas para
infraestruturas de GPU e TPU.

*   **Quando pensar no TorchTPU?** Se você tem uma base de código PyTorch
    existente, quer portabilidade entre vários hardwares (GPU/TPU) sem
    refatoração, usa frameworks nativos do PyTorch (por exemplo, TorchTitan,
    vLLM, SGLang, Hugging Face) ou quer migrar rapidamente de GPU para TPU.

## Qual é a melhor forma de entrar em contato com os desenvolvedores do TorchTPU? { #what-is-the-best-way-for-me-to-connect-with-torchtpu-developers }

Você pode enviar feedback por e-mail para
[torchtpu-feedback@google.com](mailto:torchtpu-feedback@google.com), e
contribuições diretas no repositório são bem-vindas:
[https://github.com/google-pytorch/torch_tpu](https://github.com/google-pytorch/torch_tpu).
Você também pode usar a tag "TPU" em
[discuss.pytorch.org](https://discuss.pytorch.org/).

## Haverá fóruns de participação da comunidade? { #will-there-be-any-community-engagement-forums }

Sim. A participação da comunidade é extremamente importante para nós, e vamos
compartilhar detalhes sobre esses fóruns. Além disso, planejamos realizar
plantões de dúvidas (office hours) mensais.

## Que suporte o TorchTPU oferece para desenvolvimento de kernels? { #what-support-does-torchtpu-provide-for-kernel-development }

Consulte [Suporte a kernels personalizados no TorchTPU](./custom_kernels.md).

## Quais gerações de TPU têm suporte ativo? Qual é o plano de suporte a hardware futuro? { #which-tpu-generations-are-actively-supported-whats-the-plan-for-future-hardware-support }

O TorchTPU oferece suporte ativo a v5e, v5p, Trillium (TPU v6e) e Ironwood
(TPU v7x). A validação e a otimização para TPU v8 e chips futuros estão em
andamento como parte do nosso roadmap.

## Onde encontrar os recursos compatíveis e o roadmap futuro do TorchTPU? { #where-can-we-find-torchtpus-supported-features-future-roadmap }

Consulte [Recursos compatíveis](./feature_set.md) para ver o conjunto atual de
recursos e o conjunto planejado para os próximos lançamentos.

## Existe um agente para facilitar minha integração inicial? { #is-there-an-agent-to-make-my-onboarding-easier }

Confira a TPU Agent Suite (experimental):
[https://github.com/google-pytorch/tpu-agent-suite](https://github.com/google-pytorch/tpu-agent-suite)

A TPU Agent Suite é um plugin de código aberto que fornece subagentes
especializados para automatizar a integração inicial (onboarding), o
diagnóstico e o ajuste de desempenho de inferência para modelos PyTorch em
Cloud TPU v6e e v7x.

## Posso usar o TorchTPU com nossa solução de serving personalizada? { #can-i-use-torchtpu-with-our-custom-serving-solution }

Sim. Embora o TorchTPU ofereça integração pronta para uso com mecanismos de
serving padrão do setor, como vLLM e SGLang, soluções de serving personalizadas
ou internas podem se integrar ao TorchTPU por meio de APIs de execução nativas
do PyTorch e de backends de compilador de nível mais baixo. Para orquestração
avançada, os clientes também podem implantar configurações de serving
personalizadas no GKE com llm-d.
