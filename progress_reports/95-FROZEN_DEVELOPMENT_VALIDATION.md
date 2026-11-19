# Validação de desenvolvimento congelada

**Etapa consolidada dos registros 101, 102.** Cobertura integral dos cinco modelos; sem reseleção ou confirmação independente.

Protocolos, resultados e auditorias foram reunidos sem alterar resultados, budgets ou gates. As partes históricas seguem em ordem: estados de lançamento/execução descrevem aquele momento, não tarefas ainda pendentes. Consulte as últimas partes para o encerramento.

## Registros desta etapa

- [Registro 101](#report-101) — Diagnóstico em validação de desenvolvimento com checkpoints congelados
- [Registro 102](#report-102) — Validação congelada: H2 transfere seu ganho, mas não supera o VQ maduro

<a id="report-101"></a>

## Registro histórico 101 — Diagnóstico em validação de desenvolvimento com checkpoints congelados

Fonte original preservada no arquivo: `101-FROZEN_DEVELOPMENT_VALIDATION_PROTOCOL.md`.

<a id="report-101-diagnóstico-em-validação-de-desenvolvimento-com-checkpoints-congelados"></a>

**15/11/2026 UTC. Registrado antes da leitura dos waveforms desta etapa.**
Continuação autônoma após 98 e 100, sem novo treino.

<a id="report-101-1-motivação-e-limites"></a>

#### 1. Motivação e limites

Os contrastes anteriores usaram checks de **arquivos de treino**, historicamente
vistos pela origem VQ madura. H2 melhorou seu controle, mas não a referência;
a observação completa do encoder acrescentou pouco. Antes de outro treinamento,
medir transferência para configurações diferentes.

Esta é uma **avaliação diagnóstica na validação de desenvolvimento já consumida
pelo histórico do projeto**, não uma amostra independente nova. A referência VQ
também foi selecionada historicamente com métricas desse desenvolvimento.
Não chamar isso de confirmação P3, não reinterpretar os gates anteriores nem
promover por uma média favorável.

O teste fold 0 e EDAS Polyimide/Acrylate continuam sem acesso a waveforms.
Nenhuma estatística dos sinais de validação ajusta pesos, passos, coder ou
escalas globais. Resultados poderão orientar pesquisa posterior como desenvolvimento,
não servir simultaneamente de confirmação independente.

<a id="report-101-2-população-e-congelamento"></a>

#### 2. População e congelamento

Manifesto `spatial_band_v1_fold0_20261024`, apenas `split=val`:

- **7.220 janelas**, **51 arquivos**, **seis configurações**, quatro famílias;
- **635.820.000 células** físicas retidas float32;
- cobertura integral desse split preparado, incluindo tiles pequenos, caudas e
  constantes; não só os maiores tiles de P1;
- mesmos mapas e alvos em todos os modelos; sem denoising, clipping,
  reamostragem ou mudança da política espacial.

Checagens preliminares foram apenas de metadados/existência: os 7.220 NPZ existem
localmente e no remoto. Conferir SHA-256 antes de inferência; não refazer os
recortes. Avaliação falha explicitamente em caso de dados/codec não suportados,
sem descartar janelas para melhorar métricas.

Checkpoints definidos antes dos sinais, sem escolher endpoints por essa avaliação:

1. VQ denso maduro, origem fold 0 época selecionada 23;
2. independente do protocolo 97, selecionado update 30.720;
3. H2 do protocolo 97, selecionado update 16.384;
4. controle local do protocolo 99, selecionado update novo 2.048;
5. observação completa do protocolo 99, selecionado update novo 2.048.

Nada será re-selecionado entre os checkpoints/epochs com base nesta avaliação.
Os respectivos protocolos de seleção e hashes acompanham a exportação.

<a id="report-101-3-métricas-e-taxa-prospectivas"></a>

#### 3. Métricas e taxa prospectivas

Reutilizar P0 v2: pooling SSE/contagem/variância por canal físico/arquivo,
macro canais → arquivos → configurações → famílias.
Configurações não vistas usam **as escalas por família já congeladas no treino**,
definidas em `reference_scales.json` de P0; não calibrar novos pisos nos sinais
de validação. Publicar as escalas efetivamente aplicadas e suas proveniências.

Publicar RMSE*, erro absoluto, viés/máximo, p90 dos canais por arquivo,
correlação com cobertura, bandas, amplitude/fase e cobertura espectral.
Não confundir média de p90 por arquivo com p90 pooled de toda a população.
Suportes de fase podem diferir; não declarar equivalência física sem tolerância
e suporte comparáveis.

Coder comum FAK1, com contagens ajustadas **apenas nos mesmos 44 recortes de
arquivos de treino**, nunca na validação. Ambos os livros de H2 são pagos.
Registros, side, framing, mapas e todas as janelas/caudas entram no stream;
pesos completos compartilhados são discriminados, sem alegar receptor mínimo.
Constantes bit a bit usam o mesmo modo literal em todos os modelos.

O transporte avalia o corpus de janelas armazenadas, não uma aquisição contínua
fictícia de 1 s: offsets/lacunas permanecem explícitos, sem concatenar NPZ.
Arquivos científicos/otimizadores não são confundidos com payload.

<a id="report-101-4-recursos-e-decisões"></a>

#### 4. Recursos e decisões

Duas GPUs remotas, tarefas sequenciais por GPU para não sobrepor processos:
GPU 0 referência + independente + controle local; GPU 1 H2 + observação completa.
Zero updates. Acumulação de métricas streaming evita reter todo o corpus e
reconstruções na memória; todos os frames/custos permanecem auditáveis.

Antes de abrir sinais, gravar hashes de fontes, pesos, manifesto, escalas e
lista dos caminhos autorizados. Leitor deve rejeitar qualquer split diferente
de `val`. Calibração tem leitor train-only separado.

Depois:

- preservar todos os resultados, inclusive cobertura/falhas e configurações piores;
- distinguir efeito de contexto/inovação contra controles da comparação prática;
- não iniciar automaticamente P3 ou outra seed por completar esta etapa;
- decidir a próxima hipótese com a evidência de generalização, não apenas
  repetindo variantes no mesmo pequeno conjunto de treino.

**Estado inicial:** protocolo registrado; nenhum waveform de validação desta
etapa foi lido e não há novo treinamento autorizado implicitamente aqui.

<a id="report-101-resultado-posterior"></a>

#### Resultado posterior

Os cinco modelos concluíram a cobertura integral sem updates ou reseleção.
O acompanhamento local foi interrompido por reinício, mas os processos remotos
terminaram normalmente; não houve repetição da avaliação. Resultados e decisão
estão no [relatório 102](95-FROZEN_DEVELOPMENT_VALIDATION.md#report-102).

---

<a id="report-102"></a>

## Registro histórico 102 — Validação congelada: H2 transfere seu ganho, mas não supera o VQ maduro

Fonte original preservada no arquivo: `102-FROZEN_DEVELOPMENT_VALIDATION_RESULTS.md`.

<a id="report-102-validação-congelada-h2-transfere-seu-ganho-mas-não-supera-o-vq-maduro"></a>

**15/11/2026.** Resultado do [protocolo 101](95-FROZEN_DEVELOPMENT_VALIDATION.md#report-101).

<a id="report-102-1-execução-recuperada-após-reinício-local"></a>

#### 1. Execução recuperada após reinício local

O reinício do servidor de acompanhamento cancelou a chamada SSH local, **não
a avaliação remota**. Os cinco processos terminaram com código zero e `COMPLETE`.
Não relançamos inferência nem sobrescrevemos resultados. Protocolos, registros,
manifestos, logs e métricas foram recuperados; a cópia dos frames foi retomada
separadamente, preservando os originais remotos.

A recuperação dos **36.100 frames / 307.913.994 bytes** foi concluída.
Posteriormente, os inventários SHA-256 ordenados local/remoto foram comparados
e todos os CRCs de frames e índices decodificados conferidos, sem inferência
neural ou reabertura de waveforms. Evidência: `RECOVERED_FRAME_INTEGRITY.json`.

Cada modelo cobriu **7.220 janelas, 51 arquivos, seis configurações, quatro
famílias e 635.820.000 células**. Mesmos canais físicos, contagens e escalas
congeladas entre candidatos. Nenhuma janela omitida; **zero updates e nenhuma
reseleção**.

Classes em cada avaliação: **635.779.258 células neurais ordinárias e 40.742
escapes exatos**; nenhuma janela literal. As classes somam a cobertura integral.
O receptor verificou os bits dos escapes. Todos os custos de frames, registros
e transporte estão incluídos; pesos completos compartilhados ficam separados.

As seis configurações desconhecidas usam exclusivamente os pisos familiares
congelados no treino: doped 3.448,538356; multicore 5.319,683738; multimode
73,581139; single-mode 1.109,066051. Nenhum piso foi reajustado na validação.

Artefatos:
`results/experiments/frozen_development_validation_fold0_20261115/`.
`POSTFLIGHT_AUDIT.json` valida população, contabilidade e comparações sem
reabrir waveforms. A auditoria não constitui confirmação independente de todas
as propriedades do decoder.

<a id="report-102-2-resultado-global"></a>

#### 2. Resultado global

| Checkpoint previamente selecionado | NRMSE* macro famílias ↓ | Stream bpc ↓ |
|---|---:|---:|
| VQ maduro | **0,518680** | **0,704511** |
| Independente 97 | 0,709116 | 0,801547 |
| H2 conjunto 97 | 0,648971 | 0,789539 |
| Controle local 99 | 0,650905 | 0,791404 |
| Observação completa 99 | 0,645562 | 0,790700 |

H2 reduz **8,48% do erro e 1,50% dos bytes** contra o independente, com melhora
nas seis configurações. O mecanismo transfere para estas configurações de
desenvolvimento, mas **não desloca a fronteira prática**: frente ao VQ maduro,
H2 tem **25,12% mais erro e 12,07% mais bytes**.

Observação completa reduz somente **0,82% do erro e 0,089% dos bytes** contra o
controle local, com regressão em single-mode 10. Isso não reabre o gate
historicamente reprovado de 99. Também não escolhemos esse checkpoint como
“vencedor” para confirmação por apresentar a menor média entre os H2.

Exports completos: ~111,9 MB na referência, ~223,8 MB nos dois ramos H2,
incluindo encoders. Não são receptores mínimos.

<a id="report-102-3-configurações-e-guardas-físicas"></a>

#### 3. Configurações e guardas físicas

| Configuração | VQ maduro | Independente | H2 conjunto |
|---|---:|---:|---:|
| Doped / 5m LSF | 0,621582 | 0,844993 | 0,792070 |
| Multicore / C1 and C3 | 0,555136 | 0,677798 | 0,637463 |
| Multimode / 6 | 0,528948 | 0,767930 | 0,684000 |
| Single-mode / 10 | 0,327005 | 0,490998 | 0,425180 |
| Single-mode / 2 | 0,470476 | 0,681709 | 0,604067 |
| Single-mode / 6 | 0,309685 | 0,464519 | 0,417800 |

Contra o independente, H2 ainda piora em **dois arquivos** e no p90 de canais
de **cinco arquivos**. Entre 18 pares configuração×banda, há duas regressões de
erro espectral e sete de amplitude RMS. A macro favorável não aprova todas as
guardas.

Contra a referência, H2 piora em **49/51 arquivos** e no p90 de **50/51**.
Observação completa contra seu controle piora em sete arquivos, no p90 de 20
e no erro de nove pares configuração×banda. As comparações de fase são
descritivas: os suportes definidos podem diferir; coberturas constam nos JSON.

<a id="report-102-4-decisão-e-continuação"></a>

#### 4. Decisão e continuação

**Não promover H2, não iniciar P3, não prolongar 97/99 e não gastar outra seed
para confirmar uma vitória prática que não apareceu.** O negativo não depende
apenas dos pequenos checks históricos de treino. A evidência de desenvolvimento
mantém a separação: mecanismo causal útil versus codec prático inferior.

A próxima investigação deve voltar à representação **do tile completo** e
isolar onde a referência perde waveform por bit, antes de acrescentar outro
ramo ou mais símbolos. Priorizar um diagnóstico train-only de erro por
canal/banda e custo de índices/padding do VQ maduro, com receptor efetivo.
Qualquer alternativa de representação deve ter orçamento de bits comparável,
controle full-tile e protocolo anterior ao treino; não presumir que RVQ,
decoder condicionado ou novo contexto superarão a referência por seus nomes.

Esta validação continua **desenvolvimento historicamente consumido**, não teste
independente. Não reutilizá-la como confirmação após orientar novas hipóteses.
Teste fold 0 e EDAS reservados permanecem sem acesso a waveforms nesta sequência.
