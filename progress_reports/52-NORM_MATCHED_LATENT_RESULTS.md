# Energia latente igualada não resolve a regressão multimode

## Avaliação completa

Fonte: `results/experiments/norm_matched_quantization_20261019/`, `COMPLETE`,
summaries/groups e CSVs de janelas com checksum. Duas avaliações completas,
5.920 janelas por source, zero updates. Serviço terminou com exit 0 em ~13 min.

| Source | RMSE VQ | Contínuo bruto | Contínuo com norma igualada |
|---|---:|---:|---:|
| Midpoint 2x | **0,583174** | 0,603755 | 0,587279 |
| Epoch 24 4x | **0,479352** | 0,542679 | 0,529118 |

Igualar energia melhora RMSE frente ao bypass bruto, mas ainda piora ~0,70% e
10,38% frente ao VQ. Macro contínuo norm-matched: correlação 0,793696/0,837163,
p90 por canal 0,900327/0,712054 e RMS 0,911570/0,988527. Uma melhora de RMS ou p90
macro não pode ocultar piora de forma de onda em uma configuração.

O controle foi efetivo em energia: maior diferença absoluta de RMS latente
transmitido versus quantizado foi **4,27e-8 / 4,74e-8** nos dois corpus completos.
Escalas observadas min/mediana/max: 2x **0,597095/0,922781/0,999478**;
4x **0,705488/0,943780/1,001441**. Nenhum coeficiente foi ajustado para melhorar score.

## Multimode: regressão ainda em todas as janelas

Pareamento: `results/experiments/norm_matched_quantization_paired_20261019/`,
gerado por `finetuning/summarize_quantization_probe.py`: 11.840 linhas e 26
grupos/strata. O VQ anterior é a referência; não houve nova inferência desses
controles nem troca de fontes/coordenadas.

| Taxa / subconjunto multimode | Janelas | RMSE VQ | RMSE norm-matched |
|---|---:|---:|---:|
| 2x / completo | 600 | 0,680903 | **0,821785** |
| 2x / sem escapes | 560 | 0,679339 | 0,817094 |
| 2x / com escapes | 40 | 0,702800 | 0,887461 |
| 4x / completo | 600 | 0,566143 | **0,850464** |
| 4x / sem escapes | 560 | 0,566021 | 0,846543 |
| 4x / com escapes | 40 | 0,567848 | 0,905366 |

**100% das 600 janelas pioram em ambas as taxas**, inclusive as 560 sem escapes.
Norma igualada reduz a penalidade multimode do bypass bruto, mas não a elimina.
Correlação multimode cai de 0,732466→0,628608 no 2x e 0,823841→0,642075 no 4x,
ligeiramente pior até que o bypass bruto. Não é somente um viés de amplitude.

As outras cinco configurações melhoram RMSE frente ao VQ, mas os limites por
canal persistem: p90 2x single-mode Configuration 2 1,089240→1,099524; p90 4x
single-mode Configuration 6 0,727530→0,734093. Não promover pela contagem de ganhos.

## Taxa e interpretação

Razões pooled permanecem **0,233791x / 0,116901x**: controle float32 que expande
dados ~4,28x/8,55x, não codec concorrente na taxa VQ. Ganho já está incorporado
nos componentes latentes transmitidos; não há escalar extra de side.

A energia total explica **parte**, mas não toda a regressão observada no bypass.
Direção, covariance, organização espaço-temporal e distribuição dos latentes
continuam diferentes. O controle não identifica uma causa única nem prova que o
quantizador seja ótimo/inócuo. O decoder é coadaptado à representação quantizada.

**Encerrar esses controles de bypass.** Não buscar mais normas/misturas/alphas
na mesma validação e não aumentar bitrate/codebook por essa evidência isolada.
Fontes VQ permanecem referências; nenhum controle contínuo é promovido.

## Próximo experimento isolado

Comparar adaptação **full versus decoder-only** no 4x VQ de epoch 24, com LR
menor e orçamento limitado. Não introduzir loss auxiliar ou latentes contínuos
no treino: testar somente o escopo de parâmetros adaptáveis, com controles de
seleção prospectivos e validação completa. Isso permite verificar se adaptação
do decoder preservando encoder/codebook produz outro compromisso de fidelidade,
sem alegar que o diagnóstico anterior já prevê o resultado.

Protocolo no [relatório 53](53-GUARDED_DECODER_PILOT_PROTOCOL.md). Ainda é pesquisa
no corpus de desenvolvimento; não substitui novas aquisições ou tolerâncias físicas.
