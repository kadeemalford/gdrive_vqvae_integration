# Alocação espaço-temporal: resultado completo e necessidade de confirmação

## 1. Resultado

Três braços completaram 24 épocas/98.304 updates adicionais cada, seguidos de
avaliação completa. Fonte: `results/experiments/rate_allocation_pilot_20261012/`,
histories, `REPORT.md`, `full_validation/summary.csv` e tabelas por grupo/janela.
Os três selecionaram epoch **23**, não o último checkpoint.

| Candidato | RMSE macro | Correlação macro | Pior família RMSE | p90 canal macro | Compressão nominal pooled |
|---|---:|---:|---:|---:|---:|
| Anchor congelado | 0,719900 | 0,682337 | 0,817242 | 1,228475 | 165,914x |
| Temporal | 0,709433 | 0,697489 | 0,826075 | 1,226249 | 165,914x |
| Balanced grid | 0,694050 | 0,706272 | **0,800028** | 1,167163 | **165,914x** |
| Spatial | **0,685863** | **0,709805** | 0,807557 | **1,108900** | 164,624x |

Mesmas 5.820 janelas, quatro famílias/seis configurações. Balanced redistribui
índices à mesma taxa; Spatial usa 0,8% mais índices em T=1.000. Razões são pooled
índices+side, não taxa de codec completo. Nenhum teste/auditoria externa consultado.

## 2. Configurações

| Família/configuração | Temporal RMSE | Balanced grid | Spatial |
|---|---:|---:|---:|
| Doped / 5m SMF | 0,673068 | 0,660262 | **0,660170** |
| Multicore / C1 and C4 | **0,777407** | 0,786285 | 0,782411 |
| Multimode / Configuration 7 | 0,826075 | **0,800028** | 0,807557 |
| Single-mode / Configuration 11 | **0,375986** | 0,407964 | 0,434388 |
| Single-mode / Configuration 3 | 0,738774 | 0,622977 | **0,518182** |
| Single-mode / Configuration 7 | 0,568790 | 0,557930 | **0,527371** |

Balanced e Spatial melhoram RMSE e correlação nas **seis configurações frente
ao anchor congelado**. Não dominam o controle temporal: multicore/Configuration 11
ficam melhores no Temporal; multimode é melhor em Balanced. Não selecionar uma
grade por janela pelo erro conhecido contra o original.

O ganho Spatial frente ao Temporal é ~3,32% relativo no RMSE macro, e ~4,72%
frente ao anchor. Erro médio dos canais não fracos: Temporal **0,930422**,
Balanced **0,889297**, Spatial **0,859517**. Mais densidade espacial melhorou
cauda/estatísticas por canal além da média global, embora haja trade-offs de grupos.
Quantis continuam agregações macro de estatísticas por janela, não quantis pooled.

## 3. Decisão

A alocação espaço-temporal é uma evidência mais promissora que prolongar
specialists ou reativar códigos. Manter **Balanced e Spatial como candidatos
de pesquisa**, com Temporal como controle; nenhuma geometria única é promovida.

Antes de novos ajustes no fold 1, repetir em outro conjunto de configurações
e outra ordem de treino. Comparar contrastes **dentro do fold**, sem confundir
qualidade diferente de warm-start com ganho arquitetural. Protocolo de confirmação
no [relatório 31](31-RATE_ALLOCATION_FOLD_CONFIRMATION_PROTOCOL.md).

Os resultados ainda são distantes de fidelidade alta; novos grupos independentes
e requisitos físicos continuam necessários para aceitação. A sequência de pesquisa
informada pela validação não transforma essa validação em holdout independente.
