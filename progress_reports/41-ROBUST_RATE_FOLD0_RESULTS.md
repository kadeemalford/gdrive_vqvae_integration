# Curva de taxas no fold 0: confirmação e limites por configuração

## Evidência completa

`results/experiments/high_rate_fold0_robust_20261017/`: três braços com 24 épocas,
98.304 updates por braço, `COMPLETE` e `full_validation/COMPLETE` presentes.
Scores abaixo vêm de `full_validation/summary.csv` e `*.groups.csv`, nas 5.920
janelas, não da amostra de seleção. Todos usam a mesma referência robusta.

| Candidato | Época selecionada | RMSE macro | Correlação | Compressão nominal pooled |
|---|---:|---:|---:|---:|
| Source congelado/reencodado | 0 | 0,752794 | 0,637139 | 161,931x |
| 1x | 23 | 0,719250 | 0,673098 | 161,931x |
| 2x | 24 | 0,592612 | 0,791020 | 84,228x |
| 4x | 24 | **0,479352** | **0,869126** | **42,811x** |

4x reduz RMSE em **33,36%** frente a 1x, e **19,11%** frente a 2x. O custo de
transmitir mais índices é explícito: 120.138.480 / 238.370.000 / 476.740.000 bits
de índices, mais **1.002.888 bytes de side** em cada braço, para 648.540.000 células
float32. Razão pooled = bits de entrada / (índices fixos + side).
Não é tamanho de codec implantado: não inclui container, distribuição do modelo
nem codificação entrópica. Não confundir pooled com média macro de razões ou com
as taxas da amostra de seleção do `REPORT.md` automático.

## Configurações e canais

| Família/configuração | RMSE 1x | RMSE 2x | RMSE 4x | Correlação 4x | p90 por canal 4x |
|---|---:|---:|---:|---:|---:|
| Doped / 5m LSF | 0,694523 | 0,537380 | 0,433689 | 0,903497 | 0,797073 |
| Multicore / C1 and C3 | 0,802314 | 0,671921 | 0,526922 | 0,848245 | 0,645484 |
| Multimode / Configuration 6 | 0,824348 | 0,695120 | 0,566143 | 0,823841 | 0,672368 |
| Single-mode / Configuration 10 | 0,404666 | 0,282286 | 0,215361 | 0,973263 | 0,688124 |
| Single-mode / Configuration 2 | 0,740555 | 0,688288 | **0,604817** | **0,794035** | **0,844405** |
| Single-mode / Configuration 6 | 0,522225 | 0,427504 | 0,351781 | 0,935467 | 0,727530 |

RMSE, correlação e p90 por canal melhoram monotonicamente nas seis configurações.
Isso não prova melhora em cada janela/canal. Multimode ainda é a pior **família**
(0,566143), mas single-mode Configuration 2 é a pior **configuração** (0,604817).
Não usar a média da família single-mode para ocultar essa limitação.

O p90 macro por canal cai de 1,012713 para 0,937458 e 0,717070; o RMSE médio por
canal **não fraco** cai de 0,831605 para 0,735512 e 0,557868. A média relativa sem
filtro chega a milhares, dominada por canais quase constantes, e não deve ser
interpretada isoladamente como erro físico generalizado nem removida dos registros.

## Amplitude, fase e frequência

| Diagnóstico macro | 1x | 2x | 4x |
|---|---:|---:|---:|
| Razão RMS reconstrução/alvo | 0,705007 | 0,820055 | 0,901455 |
| Alinhamento de fase FFT | 0,796005 | 0,875982 | 0,929120 |
| Razão de potência na banda alta | 0,379641 | 0,611710 | 0,778262 |

No 4x, RMS ainda é 0,848052 em multicore e 0,835075 em multimode. Melhora ampla
não equivale a amplitude/fase perfeitas, nem a tolerâncias de aplicação atendidas.
Escapes médios macro: 0,00007793 das células; são preservados e cobrados. A
proteção não explica as limitações remanescentes nas janelas ordinárias.

## Decisão

A direção da curva taxa/fidelidade foi confirmada em outro fold, não a identidade
dos scores absolutos: grupos, source, seed e referências diferem entre estudos.
Não há nova aquisição independente, teste intocado nem promoção de modelo final.

Os braços 2x e 4x foram selecionados na última época e ainda melhoravam na amostra:
entre épocas 17 e 24, RMSE 2x 0,600172→0,589426 e 4x 0,484252→0,473985.
Uma continuação **limitada de ambos**, à mesma taxa de cada braço e com igual
orçamento adicional de updates, separa melhor saturação de treino de limites de
bitrate. Não aumentar taxa/complexidade nem trocar loss neste passo.

Protocolo: [relatório 42](42-ROBUST_RATE_CONTINUATION_PROTOCOL.md).
