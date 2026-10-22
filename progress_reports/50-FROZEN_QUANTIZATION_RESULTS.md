# Latentes contínuos: melhora heterogênea e regressão multimode generalizada

## 1. Avaliação completa

Fonte: `results/experiments/frozen_quantization_probe_20261019/`, `COMPLETE`,
summaries/groups e CSVs de janelas referenciados por caminho/hash em cada JSON.
Quatro avaliações completas, **5.920 janelas por candidato**, zero updates.
Serviço terminou com exit 0 em ~26 min. Controles VQ reproduziram exatamente
os scores dos sources sem calibração de ganho.

| Taxa/modo | RMSE macro | Correlação | p90 macro por canal | RMS macro | Fase FFT | Razão payload+side |
|---|---:|---:|---:|---:|---:|---:|
| 2x VQ | 0,583174 | 0,798344 | 0,929726 | 0,827074 | 0,881931 | 84,228x |
| 2x contínuo | **0,603755** | 0,795107 | 0,907179 | 0,971798 | 0,888258 | **0,233791x** |
| 4x VQ | 0,479352 | 0,869126 | 0,717070 | 0,901455 | 0,929120 | 42,811x |
| 4x contínuo | **0,542679** | 0,838397 | 0,718388 | 1,030009 | 0,915605 | **0,116901x** |

RMSE macro piora ~3,53% / 13,21% no bypass. A melhora de RMS global, principalmente
no 2x, novamente não assegura fidelidade. Contínuo expande os dados de entrada
~4,28x / 8,55x no corpus, cobrados todos os componentes float32, não apenas índices.
Não é uma comparação de codecs com bitrate equivalente.

## 2. Configurações

| Família/configuração | RMSE 2x VQ → contínuo | RMSE 4x VQ → contínuo |
|---|---:|---:|
| Doped / 5m LSF | 0,529478 → 0,493984 | 0,433689 → 0,424700 |
| Multicore / C1 and C3 | 0,664687 → 0,626989 | 0,526922 → 0,492475 |
| Multimode / Configuration 6 | **0,680903 → 0,867336** | **0,566143 → 0,884215** |
| Single-mode / Configuration 10 | 0,266640 → 0,207689 | 0,215361 → 0,183182 |
| Single-mode / Configuration 2 | 0,681993 → 0,675132 | 0,604817 → 0,591440 |
| Single-mode / Configuration 6 | 0,424251 → 0,397310 | 0,351781 → 0,333358 |

Cinco de seis configurações melhoram RMSE/correlação, mas isso não remove as
limitações por canal: no contínuo 2x, p90 de single-mode Configuration 2 piora
de 1,089240 para 1,102055. No 4x, p90 de single-mode Configuration 6 piora
de 0,727530 para 0,742354. Não promover por contagem de grupos favorecidos.

## 3. Auditoria pareada e domínio numérico

Runner: `finetuning/summarize_quantization_probe.py`.
Evidência: `results/experiments/frozen_quantization_paired_20261019/`:
**11.840 linhas pareadas**, 26 grupos/strata de escape, hashes e `COMPLETE`.
Valida coordenadas, fontes/referência, side/células e estatísticas de quantização.
Não mistura tentativas: usa o CSV completo e seu checksum registrados no JSON.

**Todas as 600 janelas multimode pioram RMSE em ambas as taxas**, nos seis
arquivos. A perda não está restrita às 40 janelas com escapes:

| Taxa / subconjunto multimode | Janelas | RMSE VQ | RMSE contínuo |
|---|---:|---:|---:|
| 2x, com escapes | 40 | 0,702800 | 0,930589 |
| 2x, sem escapes | 560 | 0,679339 | 0,862818 |
| 4x, com escapes | 40 | 0,567848 | 0,929011 |
| 4x, sem escapes | 560 | 0,566021 | 0,881015 |

Os piores deltas incluem janelas sem escapes. Não atribuir a regressão inteira
à patologia numérica já protegida, nem sugerir novo retraining por normalização.

Erro relativo encoder→quantizado médio por janela no corpus: **0,371088 / 0,306983**
para 2x/4x; em multimode, **0,535340 / 0,473860**. Esses valores são no espaço
latente e não equivalem a erro físico, causalidade ou bits necessários para corrigir
o sinal. Uso de códigos e RMS dos latentes permanecem registrados por janela.

## 4. Interpretação e próximo controle

O resultado é compatível com mudança de distribuição no decoder, especialmente
em multimode; não demonstra que a quantização seja inócua, ótima ou a única causa
do limite. A entrada contínua muda energia **e** direção/estrutura dos latentes.
O decoder foi otimizado para códigos, não para todos os z do encoder.

Não aumentar codebook/bitrate ou retreinar automaticamente por esse bypass.
Antes, isolar uma parte do desvio: controle contínuo com **mesma energia total
do quantizado**, regra determinística e única, sem ajustar coeficiente no corpus.
Isso não iguala a distribuição ou torna o braço rate-matched. Protocolo no
[relatório 51](51-NORM_MATCHED_LATENT_CONTROL_PROTOCOL.md).

Nenhuma aquisição nova ou avaliação de teste: a análise continua exploratória.
Os controles contínuos não são candidatos para promoção como codec generalista.
