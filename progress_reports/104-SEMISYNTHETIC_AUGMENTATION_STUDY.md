# Dados — composição semissintética e diversidade de regimes

**Protocolo registrado em 15/11/2026, antes de gerar os novos exemplos ou treinar.**
Este documento reúne protocolo e resultados da etapa de dados da revisão 90.

## Hipótese e limites físicos

Aumentar a variedade de campos espaciais/temporais vistos no treino pode ajudar
a representação a aproveitar informação além dos tons recorrentes. Comparar
FEBUS-only com FEBUS + campos sintéticos, mantendo arquitetura, inicialização,
grade, loss, número de updates e sequência dos arquivos.

Os portadores medidos são os sinais completos dos arquivos **fit/train**, não
gravações certificadas de fundo puro. Não removê-los, filtrá-los ou atribuir todo
seu conteúdo a ruído. O alvo aumentado é a soma completa; o alvo dos checks
continua sendo exclusivamente o FEBUS medido original.

O gerador aplica uma resposta sintética de diferença espacial sobre gauge
e derivada temporal centrada a campos de deslocamento analíticos: mistura de
ondas/chirps e transientes com envelope. Usa dt/dx armazenados e sorteia
GL/dx entre 5, 6,25 e 25 para representar incerteza, **não** para alegar
calibração instrumental ou unidades absolutas. Frequências de 0,5%–35% do
sample rate e comprimentos de onda de 2,5–128 dx; direções e fases variáveis.
Normalizar a energia do campo composto e sortear amplitude RMS entre
0,25 e 1 vezes `max(std(portador), escala_fit_do_grupo)`.

Não há novo dado independente nem certificação de resposta FEBUS. Este é um
teste de aumento de diversidade controlada, mais limitado que adquirir sessões.
O corpus não permite certificar independência de sessões de fundo: usar arquivos
fit/check disjuntos e declarar essa limitação, sem vender o resultado como P3.

## Desenho congelado

- VQ denso maduro, K2048×128 e mesma grade full-tile. Sem nova arquitetura,
  normalização, scheduler, quantizador, prior ou loss.
- Dois braços: **real** e **mixed**. 8.192 updates cada, fresh AdamW
  LR=1e-4, weight_decay=1e-5, clip=1, FP32, seed de amostragem/modelo 2117.
- Mesma sequência de recortes e mesmas células apresentadas nos dois braços.
  No mixed, updates pares recebem composição; ímpares são originais.
  RNG exclusivo da composição, seed 104, não altera a ordem dos recortes.
- Somente arquivos de treino fora dos 43 arquivos check históricos.
  Os 44 recortes fit calibram os coders em **FEBUS original**, nunca sintético.
- Avaliar fonte inicial e endpoint nas mesmas 136 janelas check. Sem seleção
  intermediária, novo sweep ou extensão acima do budget.
- Toda composição deve ser finita e preservar shape/dtype; overflow interrompe
  a execução, sem clipping para salvar exemplos.
- Testes e smoke de dois updates (um original/um composto) precedem a execução
  científica; smoke não conta no orçamento e não escolhe hiperparâmetros.
- Pesos, frames, transporte, side e registros dos dois modelos cobrados como
  nas etapas anteriores; expor bytes compartilhados separadamente.

## Decisão

Comparar mixed contra real igualmente adaptado **e** contra fonte madura.
Ganho em loss sintética não é resultado em FEBUS. Publicar macros prospectivas,
caudas, bandas e contagens reais/compostas. Investimento exige ≥5% menos erro
a bytes completos não maiores, sem regressão de macro por configuração,
e inspeção das guardas físicas; não promover pela média ou pelo braço mais
barato com mais erro. Gates reprovados encerram este desenho, não toda simulação.
Nenhum acesso a waveforms val/test/EDAS ou aquisição de fontes externas.

Destino: `results/experiments/semisynthetic_augmentation_20261115/`.

## Engenharia e lançamento

485 testes locais passaram, incluindo determinismo da composição, preservação
do portador, tiles pequenos, geometria inválida e rejeição de diferenças na
sequência/células/frames iniciais do par. O supervisor só permite a fase
científica após smoke pareado e conferência da função inicial via frames pagos.
Implementações: `finetuning/run_semisynthetic_study.py`,
`finetuning/semisynthetic_fields.py` e `tools/run_semisynthetic_pair.py`.

## Resultado concluído e recuperação após reinício

O reinício interrompeu somente o acompanhamento local. Smoke e fase científica
terminaram no remoto com código zero nos dois braços; **nenhum treino foi
repetido**. Recuperados 665 arquivos com SHA-256 conferido; quatro checkpoints
permanecem no remoto com tamanho/hash inventariados em `RECOVERY_AUDIT.json`.
Os artefatos remotos totalizam 1.365.314.337 bytes.

Cada braço completou **8.192 updates / 655.595.000 apresentações de células**.
O mixed fez 4.096 composições, envolvendo 333.389.000 apresentações de células;
as demais usaram os portadores originais. A sequência de recortes e o total de
células foram idênticos, assim como os frames iniciais nas duas GPUs.
Exposições repetidas não são novas amostras independentes.

Avaliação em **136 janelas / 43 arquivos / dez configurações / quatro famílias /
12.799.000 células FEBUS originais**:

| Modelo | NRMSE* macro famílias ↓ | Stream bpc ↓ |
|---|---:|---:|
| Fonte madura congelada | 0,490336 | 0,679244 |
| Continuação somente real | 0,491848 | 0,678543 |
| Continuação mixed | 0,497602 | 0,678552 |

Mixed teve **1,16968% mais erro / 0,001382% mais bytes** que o controle real,
e **1,48165% mais erro / 0,10187% menos bytes** que a fonte.
O controle real também piorou o erro da fonte. Não escolher o braço
menos ruim para promover esta continuação.

Contra o controle real, mixed piorou:

- macros de **oito das dez configurações**;
- NRMSE* de **38/43 arquivos** e p90 de canais de **30/43 arquivos**;
- erro espectral em **19/30 pares configuração×banda**;
- distância de amplitude RMS a1 em **16/30**.

Contra a fonte madura, piorou todas as dez macros por configuração,
NRMSE* em 38 arquivos, p90 em 35 e erro espectral em 21/30 pares.
Fase permanece descritiva, com suportes reportados; não inferir uma guarda
certificada de fase de contagens com cobertura diferente.

## Decisão e limites

**Gate reprovado. Sem extensão, sweep de amplitudes/gerador, promoção ou P3.**
O resultado rejeita esta composição e dose sobre a fonte madura, não toda
simulação nem o benefício de novas sessões. Não é um teste de pré-treino
semissintético do zero nem uma curva de diversidade de aquisições.
Portadores não eram fundo puro certificado, e o gerador não é um simulador
FEBUS calibrado. Não alegar que aumentar diversidade física real seria inútil.

Fontes/snapshots, histórico, bytes de frames e cobertura foram auditados sem
reabrir waveforms. Evidências: `POSTFLIGHT_AUDIT.json`, `PHYSICAL_VS_REAL.json`
e `PHYSICAL_VS_SOURCE.json` no diretório da etapa. Teste e EDAS permanecem
reservados; pesos e dados originais não foram modificados.
Também foram decodificados localmente os símbolos dos **552 frames** de
smoke/fonte/endpoint: 3.608.000 posições de índices e oito frames literais,
com CRCs de frames/símbolos conferidos, sem inferência ou leitura de waveforms.

## Segundo contraste: curva de diversidade, protocolo anterior aos resultados

Não é extensão do gerador reprovado. A revisão 90 também propôs variar a
diversidade dos grupos de treino. Usar a fonte madura para isso seria inadequado:
ela já foi adaptada a todos os grupos. O preflight localizou os **pesos OOI
originais**, SHA-256
`7d2da0d1cbd1d3dae579d7fd900850afc9ef15f645ea844168c0f78f18b0a27c`,
em `models/febus_finetune_base/best.pt` no remoto, com o hash histórico conferido.
Nenhum waveform reservado foi aberto nessa inspeção.

### Desenho

- Três braços aninhados de **4/6/10 configurações (40/60/100%)**.
  A proposta 25/50/100% não é literalmente aplicável a dez grupos mantendo as
  quatro famílias em todos os braços; não arredondar 2,5 grupos silenciosamente.
- Quatro grupos: primeiro grupo lexicográfico de cada família. Seis: acrescentar
  o próximo grupo multicore e o próximo single-mode. Dez: todos os demais.
  Seleção somente pelo manifesto, sem consultar sinais/resultados.
- Mesmos pesos OOI originais nos três braços, mesma geometria densa
  `spatial4-temporal4`, K2048×128, loss e normalização. A alteração de strides
  é comum aos três; não comparar o efeito dela como ganho de diversidade.
- **8.192 updates por braço**, AdamW fresco LR=1e-4, wd=1e-5, clip1, FP32.
  Seed 2117. Família em ciclo de quatro, com seleção uniforme de configuração,
  arquivo e recorte elegível dentro da família.
- Recortes de treino com **8 canais contíguos × 1.000 amostras**, para cobrar
  exatamente 65.536.000 apresentações de células por braço. Mesma sequência de
  famílias; configurações/arquivos necessariamente variam conforme o braço.
  Tiles menores não são apagados: apenas não integram o sampler deste treino,
  e continuam na avaliação comum integral de 136 checks.
- Treinar somente nos arquivos fora dos 43 checks. Calibrar o coder com recortes
  fit dos **quatro grupos comuns**, igualmente nos três braços. Nenhum sinal
  dos grupos ausentes é usado para ajustar pesos ou probabilidades do respectivo
  braço. As escalas de métricas prospectivas preexistentes não são entradas da rede.
- Smoke de dois updates por braço, antes dos científicos; fonte inicial e
  endpoint avaliados nos mesmos 136 checks, sem seleção intermediária.
- Duas GPUs: braços 4/6 em paralelo; braço10 após os dois, sem aumentar budget.
  Publicar custo de encoder/decoder, stream completo e pesos compartilhados.

### Interpretação e encerramento

Esse desenho mede **amplitude de regimes/configurações** a dose total e por
família iguais. Não mede quantidade de sessões independentes nem convergência:
em cada família, mais configurações reduzem repetição por configuração.
A curva pode piorar, melhorar ou não ser monotônica; publicar todos os pontos,
incluindo vistos/ausentes no treino. Não escolher subconjuntos após resultados.
Não usar validação/teste/EDAS nem chamar checks historicamente consumidos de
confirmação nova. Preservar fonte madura como referência prática, mas não como
controle causal desta curva de inicialização distinta.

Sem extensão de budget/seed/subconjunto para salvar tendência. O resultado
informará a prioridade de dados; não autoriza P3 por si só. Condicionamento
físico de H3 permanece separado e limitado pela proveniência documentada em 98.

Destino: `results/experiments/regime_diversity_curve_20261115/`.

Preflight: todos os dez grupos possuem recortes elegíveis de oito canais nos
arquivos de treino não check. 487 testes locais passaram; os testes novos
conferem aninhamento, cobertura das quatro famílias e crop sem mutar o portador.
O supervisor exige smoke completo nos três braços e igualdade dos frames da
fonte antes de aceitar a fase científica. Nenhuma tendência foi medida ainda.

O supervisor remoto da curva foi lançado após o fechamento do contraste
semissintético. Seus processos/logs/checkpoints são independentes; não retomar
ou repetir o treino da composição concluída. O custo científico máximo novo
da curva é 24.576 updates e 196.608.000 apresentações de células, separado
dos 16.384 updates e 1.311.190.000 apresentações da composição.

## Resultado da curva de diversidade

Os três braços terminaram com código zero, **8.192 updates e 65.536.000
apresentações de células cada**, exatamente 2.048 updates por família.
Smokes ficaram fora desses totais. A fonte inicial e seus frames pagos foram
idênticos; calibração do coder restrita aos quatro grupos comuns foi conferida.

Mesmos 136 checks / 43 arquivos / dez configurações / quatro famílias /
12.799.000 células, sem seleção de endpoint:

| Configurações no treino | NRMSE* macro famílias ↓ | Stream bpc ↓ |
|---|---:|---:|
| Fonte OOI comum, antes da adaptação | 1,130954 | 0,392898 |
| 4 grupos | 1,009123 | 0,596059 |
| 6 grupos | **0,983039** | 0,605377 |
| 10 grupos | 1,083851 | **0,587398** |

- **4 → 6:** 2,58476% menos erro, mas 1,56320% mais bytes. Oito arquivos
  tiveram NRMSE* maior, 20 tiveram p90 maior e 10/30 pares grupo×banda
  tiveram erro espectral maior.
- **6 → 10:** 10,25514% mais erro e 2,96977% menos bytes. NRMSE* pior em
  39/43 arquivos, p90 em 41/43 e erro espectral em 18/30 pares.
- **4 → 10:** 7,40532% mais erro e 1,45299% menos bytes. Não há ganho
  monotônico de fidelidade com a quantidade nominal de configurações.

### Resultados por configuração, sem misturar conjuntos vistos e ausentes

As letras indicam presença no ajuste de pesos: V = visto, A = ausente.
Todos os números são NRMSE* do mesmo grupo físico nos checks reais.

| Configuração | 4 grupos | 6 grupos | 10 grupos |
|---|---:|---:|---:|
| doped / 5m SMF | 0,975445 V | 0,981061 V | 1,039511 V |
| multicore / C1 and C4 | 0,876401 V | 0,864188 V | 0,895147 V |
| multicore / C2 and C3 | 0,881827 A | 0,866900 V | 0,902093 V |
| multimode / Configuration 7 | 1,020327 V | 0,962705 V | 1,189785 V |
| single-mode / Configuration 11 | 1,769889 V | 1,667543 V | 1,770933 V |
| single-mode / Configuration 12 | 1,162082 A | 1,108680 V | 1,197491 V |
| single-mode / Configuration 3 | 0,966602 A | 1,022459 A | 1,071578 V |
| single-mode / Configuration 4 | 1,197895 A | 1,140312 A | 1,211166 V |
| single-mode / Configuration 7 | 1,128196 A | 1,121968 A | 1,201683 V |
| single-mode / Configuration 8 | 0,744958 A | 0,676117 A | 0,792082 V |

Ausente do ajuste de pesos não significa nova aquisição independente:
proveniência de sessão continua não certificada, e estes checks já eram
desenvolvimento consumido. Não comparar médias de “vistos” compostas por
grupos diferentes como se fossem a mesma população.

### Interpretação e decisão

O ponto de seis grupos melhora a média de quatro, **comprando bytes e com
regressões físicas**; dez grupos não melhora a fidelidade. Não há deslocamento
de fronteira demonstrado. A referência prática madura continua disponível
em NRMSE* ~0,490336 / ~0,679244 bpc nessa coorte, mas tem inicialização e
histórico de adaptação diferentes: não é controle causal da diversidade.

O experimento mede adaptação curta em crops de oito canais e desempenho em
tiles completos, não uma lei de escala nem convergência. Uma única seed,
diluição da repetição entre configurações e amostras efetivamente diferentes
entre braços limitam a atribuição universal. Mesma dose de células/família
não garante a mesma dificuldade nem uma trajetória de aprendizagem equivalente.
Não concluir que novas sessões físicas sejam inúteis.

**Encerrar a curva sem extensão, escolha de outra seed/subconjunto ou P3.**
O resultado responde à hipótese operacional registrada, sem transformar o
negativo da composição em uma busca indefinida de aumentos de dados.
Nenhum candidato foi promovido e nenhum waveform val/test/EDAS foi aberto.

Recuperação: **989 arquivos conferidos por SHA-256**, seis checkpoints
preservados no remoto com hashes; 2.035.956.875 bytes remotos inventariados.
Auditoria local decodificou **828 frames / 5.238.000 posições de índices /
12 literais**, conferindo CRCs, contagens, disjunção de arquivos e snapshots,
sem inferência ou leitura de waveforms. Evidências: `RECOVERY_AUDIT.json`,
`POSTFLIGHT_AUDIT.json` e `PHYSICAL_{6_VS_4,10_VS_6,10_VS_4}.json`
no diretório da curva.

Fechamento de engenharia: **487 testes passaram**, `git diff --check` sem
erros e 284 links locais conferidos. Numeração 1–104 sequencial; relatórios
1–89 intactos. A situação das hipóteses restantes foi explicitada na seção 12
da revisão [90](90-FIDELITY_PER_BIT_NEXT_RESEARCH_REVIEW.md).
