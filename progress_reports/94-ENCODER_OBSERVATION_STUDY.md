# Observação completa no encoder

**Etapa consolidada dos registros 99, 100.** Ganho pequeno; gate de extensão reprovado, sem promoção.

Protocolos, resultados e auditorias foram reunidos sem alterar resultados, budgets ou gates. As partes históricas seguem em ordem: estados de lançamento/execução descrevem aquele momento, não tarefas ainda pendentes. Consulte as últimas partes para o encerramento.

## Registros desta etapa

- [Registro 99](#report-99) — Observação completa no encoder: contraste controlado sobre H2
- [Registro 100](#report-100) — Observação completa no encoder: ganho pequeno, extensão não autorizada pelo gate

<a id="report-99"></a>

## Registro histórico 99 — Observação completa no encoder: contraste controlado sobre H2

Fonte original preservada no arquivo: `99-ENCODER_OBSERVATION_CONTROLLED_PROTOCOL.md`.

<a id="report-99-observação-completa-no-encoder-contraste-controlado-sobre-h2"></a>

**15/11/2026 UTC. Registrado antes de smoke/piloto.**
Continuação autônoma do [resultado 98](93-NEURAL_ANCHOR_INNOVATION_STUDY.md#report-98).

<a id="report-99-1-pergunta-causal"></a>

#### 1. Pergunta causal

H2 melhorou ~7,48% em erro e ~2,99% em bytes sobre seu controle, mas ficou
~28,03% pior em erro que o VQ completo. Uma diferença concreta é a observação
antes da análise: o VQ completo vê canais adjacentes adquiridos; o encoder
de âncoras H2 vê os pares, e o de inovação vê os ímpares menos a previsão.

Testar **a observação original da outra fase apenas no encoder**, sem oferecer
contexto não transmitido ao decoder. Não mudar coder, quantizadores, grades,
padding, normalização, loss, dados ou scheduler simultaneamente.

<a id="report-99-2-construção-com-função-inicial-preservada"></a>

#### 2. Construção com função inicial preservada

Os dois braços partem **do mesmo selecionado H2, update 16.384**, não de seus
endpoints e não da origem VQ ainda congelada:
`joint_anchor_vq_fold0_20261115/arms/closed-loop/best.pt`.

Adicionar a cada primeiro stem um ramo convolucional da outra fase, sem bias,
com pesos inicialmente **zero**:

```text
feature_anchor = existing_stem(original_even) + extra_stem(original_odd)
feature_innovation = existing_stem(innovation_odd) + extra_stem(original_even)
```

- **Local-analysis (controle):** a entrada extra é sempre zero.
- **Full-observation:** a entrada extra contém a fase original adjacente,
  alinhada na mesma janela. A última âncora sem ímpar recebe zero somente no
  contexto ausente, não no alvo.
- Mesmos 1.152 parâmetros adicionais nas duas arquiteturas. No controle, o ramo
  extra precisa ter gradiente zero; no candidato, finito e não nulo no smoke.
- A via principal convolucional original é preservada, não fundida arbitrariamente
  em outro kernel que altere a função antes do ajuste.
- Exigir igualdade bit a bit do waveform normalizado e dos dois streams de
  índices com a fonte H2 no maior recorte de ajuste, antes de treinar.

Preditor, decoders, codebooks e demais encoders recebem adaptação conjunta nos
dois braços. O preditor começa com os **mesmos pesos já treinados** em ambos.
Não é um contraste preditor-zero versus preditor-neural.

<a id="report-99-3-por-que-isso-não-dá-informação-gratuita-ao-receptor"></a>

#### 3. Por que isso não dá informação gratuita ao receptor

Os dados completos já existem no lado encoder. A observação adicional altera
somente a seleção dos **índices transmitidos e pagos**. O receptor continua usando
os dois livros, decoder, preditor e registros, sem amostras originais adicionais.

Os contextos temporários são limpos ao sair do forward, não entram no
`state_dict`, não são buffers persistidos e nunca são lidos por `decode_indices`.
Há teste de receptor sem esses contextos. O formato de síntese FJH1/FAK1 continua
o mesmo; o loader reconhece explicitamente a nova arquitetura
`joint-encoder-observation-vq-v1`, sem inferir versão pelo shape dos pesos.

Paridade, cobertura física, máscaras e side FNE1 global permanecem.
Nenhum canal/banda é removido; não há denoising ou reamostragem do waveform.

<a id="report-99-4-budget-dados-e-comparação"></a>

#### 4. Budget, dados e comparação

- Mesmas 5.600 janelas de 44 arquivos; 43 arquivos P1 excluídos dos batches.
- Seed **2116**, mesma sequência entre os dois braços, batch de uma janela.
- Herança declarada: fonte densa OOI/FEBUS e selecionado H2 de 16.384 updates.
  A fonte é idêntica entre braços; não alegar treino do zero ou checkpoint de 32k.
- **AdamW novo nos dois braços**, LR `1e-4`, wd `1e-5`, clip 1, FP32 determinístico;
  momentos antigos não são resetados seletivamente.
- Mesma loss MSE/L1 + média VQ, seleção prospectiva e calibração de 44 recortes.
- **8.192 updates novos por braço**, extensão condicional até **32.768 totais
  novos por braço**, não além. Teto científico 65.536, oito updates técnicos.
- Mesmo gate de investimento do protocolo 97: ≥1% sobre inicialização,
  ≥0,2% no último bloco, erro até 3% acima do controle local selecionado,
  bytes até 105%, ambas as perplexidades ≥16 e cobertura/replay válidos.
  Os nomes legados do gate referentes a “independent” significam aqui o
  controle explicitado em `control_name=local-analysis`.
- Smoke contínuo/resume valida parâmetros, momentos e RNG; só depois lançar piloto.
- Após o budget, avaliação ampliada automática de selecionados/endpoints nos
  129 checks/43 arquivos e da referência VQ original com o mesmo FAK1.

O alvo de ~0,8 bpc continua uma referência da seleção, não taxa garantida ou
tolerância física. O número de posições/framing de ramos é mantido: a auditoria
98 encontrou 8,33% mais posições que o VQ completo nesse layout; não esconder
esse custo ou atribuí-lo todo ao contexto.

Sem leitura de validação/teste/EDAS reservado para este piloto. Checks são
desenvolvimento consumido; confirmação P3 continua condicionada a superar a
fronteira prática e às guardas físicas, não só a melhorar o controle local.

<a id="report-99-5-código-preservação-e-lançamento"></a>

#### 5. Código, preservação e lançamento

`finetuning/encoder_observation_vq.py` implementa o ramo de observação e loader.
Runner, loader genérico e supervisor ganharam opções explícitas para esse
contraste, mantendo a leitura dos checkpoints anteriores. Os fontes exatos da
campanha 97 foram arquivados/verificados em `SOURCE_SNAPSHOT/` **antes** dessas
extensões; seus hashes/resultados históricos não são substituídos.

Pipeline:

```text
smoke → identidade contínuo/resume → 8.192 updates
      → gate registrado → eventual continuação a 32.768
      → avaliação ampliada, sem updates
```

Destino: `results/experiments/joint_encoder_observation_fold0_20261115/`.
GPU 0 controle local; GPU 1 observação completa. O pipeline não precisa de
uma pausa/manualização a cada transição já prevista e não promove modelos.

**Estado inicial:** protocolo e implementação registrados, ainda sem treinamento
deste contraste ou resultado científico.

<a id="report-99-6-integração-confirmada-e-piloto-iniciado"></a>

#### 6. Integração confirmada e piloto iniciado

**425 testes locais passaram.** Nas duas GPUs, os smokes confirmaram a mesma
função inicial da fonte H2: bits float32 do waveform normalizado e ambos os
streams de índices iguais, conforme `INITIAL_FUNCTION_PARITY.json`.
Também passaram isolamento de gradientes do ramo extra e receptor independente
sem acesso às observações originais.

`RESUME_INTEGRATION.json` confirmou parâmetros, momentos AdamW e RNG
exatamente iguais entre execução contínua e retomada, nos dois braços.
O supervisor iniciou então o piloto científico:

| Braço | GPU física | PID inicial | Budget inicial |
|---|---:|---:|---:|
| Local-analysis | 0 | 817585 | 8.192 updates |
| Full-observation | 1 | 817586 | 8.192 updates |

Comandos em `LAUNCH_pilot_*.json` na raiz registrada. O gate e a avaliação ampliada
seguem automaticamente dentro do teto, sem promoção ou acesso aos conjuntos
reservados. Ainda não há resultado científico deste contraste nesta atualização.

<a id="report-99-resultado"></a>

##### Resultado

O piloto terminou, não passou o critério de melhoria de 1% e não foi estendido.
A avaliação ampliada também terminou. Resultados e limites estão no
[relatório 100](94-ENCODER_OBSERVATION_STUDY.md#report-100).

---

<a id="report-100"></a>

## Registro histórico 100 — Observação completa no encoder: ganho pequeno, extensão não autorizada pelo gate

Fonte original preservada no arquivo: `100-ENCODER_OBSERVATION_RESULTS.md`.

<a id="report-100-observação-completa-no-encoder-ganho-pequeno-extensão-não-autorizada-pelo-gate"></a>

**15/11/2026 UTC.** Fechamento do [protocolo 99](94-ENCODER_OBSERVATION_STUDY.md#report-99).

<a id="report-100-1-execução-e-decisão-registrada"></a>

#### 1. Execução e decisão registrada

Smokes, identidade da função inicial, isolamento de gradientes e retomada
determinística passaram. Os dois braços completaram **8.192 updates novos cada**.
Não houve extensão: `learning_one_percent: false`; os outros critérios passaram.
Os livros permaneceram saudáveis, com médias no último bloco do candidato
~628 / ~500 para âncoras/inovação.

O selecionado do candidato nos dez checks de treino foi update 2.048:
NRMSE* 0,613946 contra 0,619782 inicial, melhora de **~0,94%**, abaixo do 1%
registrado. Não reduzir retrospectivamente esse limiar para prolongar o treino.
Ambos os selecionados são do update 2.048. Endpoints ficam publicados, não
substituem silenciosamente a seleção.

<a id="report-100-2-avaliação-ampliada-sem-reseleção"></a>

#### 2. Avaliação ampliada, sem reseleção

Mesmos 129 checks/43 arquivos/dez grupos/quatro famílias e 12.762.000 células:

| Modelo | Update novo | NRMSE* macro ↓ | Stream bpc ↓ |
|---|---:|---:|---:|
| Controle local selecionado | 2.048 | 0,629368 | 0,740660 |
| Observação completa selecionada | 2.048 | 0,625462 | 0,740221 |
| Controle local endpoint | 8.192 | 0,627789 | 0,744055 |
| Observação completa endpoint | 8.192 | 0,621573 | 0,742450 |
| Fonte H2 anterior selecionada | 16.384 herdados | 0,628995 | 0,738904 |
| VQ maduro, mesmo FAK1 | — | 0,491288 | 0,676730 |

Entre selecionados, ganho de erro de apenas **~0,62%**, com bytes quase iguais.
Há regressão em single-mode 11 e 7. A auditoria pareada registrou oito dos
43 arquivos com NRMSE* maior, 16 com p90 maior e 13 pares grupo×banda com erro
espectral maior. Fase tem suporte/cobertura variável e continua descritiva.
Não há ganho generalista demonstrado nem deslocamento da referência prática.

<a id="report-100-3-o-que-o-negativo-significa"></a>

#### 3. O que o negativo significa

O ramo de observação original foi efetivamente conectado e recebeu gradientes;
o controle teve gradiente zero nessa entrada. Portanto, não é um resultado de
um ramo desligado por falha de integração.

Esse ajuste curto, a partir do H2 já treinado, **não resolveu o déficit**
simplesmente oferecendo a outra fase ao encoder. Não demonstra que informação
de entrada seja irrelevante em qualquer arquitetura ou que uma inicialização
diferente convergiria ao mesmo resultado.

Encerrar esta continuação no budget do piloto; **sem P3, sem promoção e sem
outra continuação igual**. A evidência positiva do H2 contra seu controle em 98
permanece; este resultado testa outra pergunta.

Artefatos: `results/experiments/joint_encoder_observation_fold0_20261115/`,
incluindo gate, estados, frames, checks ampliados, auditoria física e `COMPLETE`.
Os resultados científicos estão localmente e no remoto; os pesos técnicos
duplicados permanecem preservados no remoto.

<a id="report-100-4-próxima-etapa-diagnóstico-fora-das-configurações-de-treino"></a>

#### 4. Próxima etapa: diagnóstico fora das configurações de treino

Antes de escolher outra variante neural, comparar os checkpoints **já congelados**
nas configurações de validação de desenvolvimento. A triagem usada até aqui
inclui arquivos de treino historicamente vistos pela referência madura; isso
não responde sozinho sobre transferência entre configurações.

Essa avaliação não reabre o gate, não muda checkpoints selecionados e **não é
confirmação P3 nem teste final**. A validação já pertence ao desenvolvimento
histórico do projeto. O [protocolo 101](95-FROZEN_DEVELOPMENT_VALIDATION.md#report-101)
registra escopo, escalas, custos e proteção do teste/EDAS antes dos sinais.
