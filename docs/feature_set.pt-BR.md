# Recursos compatíveis com o TorchTPU e trabalhos futuros { #torchtpu-supported-feature-set-and-future-work }

## Recursos compatíveis no momento { #current-supported-feature-set }

A tabela abaixo mostra a cobertura atual de recursos na primeira versão de
código aberto.

Para ajudar você a avaliar o TorchTPU para suas cargas de trabalho, dividimos os
recursos em dois níveis:

*   **Validado:** recursos com testes contínuos de ponta a ponta, cobertura
    automatizada de CI/CD e verificação por benchmarks em arquiteturas de modelo
    padrão.
*   **Experimental:** implementações em estágio inicial, APIs emergentes ou
    recursos em protótipo. Eles funcionam e estão disponíveis para feedback
    inicial, mas podem passar por ajustes de comportamento ou mudanças de API
    com base nas contribuições da comunidade.

> **Note:** o TorchTPU evolui rapidamente e de forma aberta. Para ver o status
> mais recente ou pedir recursos específicos, acesse nosso
> [GitHub](https://github.com/google-pytorch/torch_tpu).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
|  | **Validado** | **Experimental** |
| ----- | ----- | ----- |
| **Engajamento com a comunidade PyTorch** | <ul><li>Versão OSS do TorchTPU, vLLM, SGLang, TorchTitan, TorchRec</li><li>Aceitação de PRs externos</li><li>Documentação / tutoriais</li><li>CI/CD</li><li>Colab</li></ul> | <ul><li>Agente de TPU para migração de ATEN</li></ul> |
| **Suporte a TPU** | <ul><li>v5e, v5p, v6e, v7</li></ul> |  |
| **Funcionalidade básica** | <ul><li>Facilidade de migração</li><li>Modo eager e compilado combinados</li><li>FW \+ BW</li><li>Torch.profile</li><li>Operações dinâmicas (com recompilação)</li></ul> | <ul><li>SymInts</li><li>Tensores irregulares (jagged)</li></ul> |
| **Arquiteturas de modelo** | <ul><li>Transformer (denso \+ MoE): Qwen, Llama, Muse Glimmer, Gemma</li><li>Difusão: WAN, FLUX</li><li>CNN</li></ul> | <ul><li>GLM, Kimi, DeepSeek, GPT OSS e muitos outros (hill climb)</li><li>Mamba</li><li>LSTM/GRU</li><li>DLRM</li></ul> |
| **Suporte a kernels** | <ul><li>Pallas</li><li>Tokamax</li></ul> | <ul><li>Helion</li></ul> |
| **Quantização** | <ul><li>FP8/FP4 por tensor inteiro</li><li>FP8 por canal</li></ul> |  |
| **Execução distribuída** | <ul><li>Malha (mesh) ideal</li><li>SimpleFSDP</li><li>Checkpointing</li><li>Offload de ativações</li><li>Multi-slice com até 8 mil chips</li></ul> | <ul><li>FSDPv2 (baixo desempenho)</li><li>MPMD (baixo desempenho)</li><li>TorchComm</li></ul> |
| **Integrações com o ecossistema** | <ul><li>Hugging Face</li><li>Ray</li><li>vLLM / SGLang</li><li>TorchTitan</li><li>TorchRec</li></ul> |  |
| **Soluções prontas para produção** | <ul><li>Inferência (vários hosts, P/D desagregado)</li><li>Ajuste fino (até \~8 mil chips)</li></ul> |  |
<!-- mdformat on -->

## Trabalhos futuros { #future-work }

Esta seção descreve nossos horizontes de desenvolvimento e explorações futuras,
e não um cronograma de lançamentos com compromisso firmado.

Como um projeto de código aberto voltado para a comunidade, nossas prioridades
de desenvolvimento respondem intencionalmente ao feedback da comunidade e à adoção
pelos desenvolvedores. Recursos, prazos e abordagens de arquitetura podem mudar
conforme entendemos quais cargas de trabalho são mais importantes para você.

Organizamos os próximos trabalhos em dois horizontes amplos:

*   **Foco de curto prazo:** áreas de design ativo, prototipagem em andamento e
    reforço de desempenho que a equipe principal está priorizando no momento.
*   **Exploração futura:** explorações de arquitetura de longo prazo, possíveis
    integrações com o ecossistema e itens de backlog pedidos pela comunidade que
    estamos investigando ativamente.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
|  | **Foco de curto prazo** | **Exploração futura** |
| ----- | ----- | ----- |
| **Engajamento com a comunidade PyTorch** | <ul><li>Mais documentação / tutoriais / posts de blog</li></ul> | <ul><li>Agente de TPU para migração de kernels</li></ul> |
| **Suporte a TPU** | <ul><li>v8i, v8t</li></ul> |  |
| **Funcionalidade básica** | <ul><li>Streams</li><li>Captura de grafos</li><li>Modo eager com inicialização rápida</li><li>Operações dinâmicas (sem recompilação)</li></ul> |  |
| **Arquiteturas de modelo** | <ul><li>Suporte a modelos desde o primeiro dia</li></ul> |  |
| **Quantização** | <ul><li>FP8/FP4 por blocos com torchao (desempenho máximo na TPUv8t)</li></ul> |  |
| **Execução distribuída** | <ul><li>FSDPv2 (com desempenho)</li><li>MPMD (com desempenho)</li><li>Treinamento com \> 8 mil chips</li><li>Checkpointing em vários níveis</li><li>Goodput</li></ul> | <ul><li>Treinamento com >> 8 mil chips</li></ul>  |
| **Integrações com o ecossistema** | <ul><li>VeRL</li></ul> |  |
| **Soluções prontas para produção** | <ul><li>RL</li></ul> | <ul><li>Pré-treinamento</li></ul> |
<!-- mdformat on -->
