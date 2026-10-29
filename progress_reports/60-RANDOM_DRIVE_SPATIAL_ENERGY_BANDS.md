# Faixas de alta energia: verificação em amostra aleatória do Drive

## Resultado

**O padrão não é exclusivo dos dois HDF5 single-mode anteriormente visualizados,
mas também não aparece de maneira uniforme em todas as famílias/configurações.**

Foram baixados **24 HDF5**, aproximadamente **1,74 GiB**, de **12 configurações**:
três configurações por família e dois arquivos por configuração, sorteados com
seed `20261022`. Os dois exemplos anteriores foram excluídos do sorteio. Todos
os downloads foram conferidos pelo MD5 do Drive e SHA-256 local. **Sem inferência
neural, treinamento, filtragem ou alteração dos sinais brutos.**

Evidência: [relatório completo](../results/experiments/spatial_band_audit_20261022/REPORT.md),
[resumo visual](../results/experiments/spatial_band_audit_20261022/family_summary.png),
[índice de figuras](../results/experiments/spatial_band_audit_20261022/FIGURES.md),
`files.csv`, `blocks.csv`, `families.csv`, `selection.csv`, `downloads.json` e
`protocol.json` no mesmo diretório. HDF5 em `data/spatial_band_audit_20261022/`.
Runner: `acquisition/audit_spatial_energy_bands.py`.

## Métrica e comparação

Energia = soma dos desvios temporais quadráticos, **centrados por canal e bloco**.
Foram analisados os mesmos intervalos centrais de um segundo `[500:20500]` de
**todos os 127 blocos**, mantendo toda a extensão espacial original. Células não
finitas não participam das somas, mas suas posições não são comprimidas nem
reordenadas. “Top 10%” usa arredondamento para cima do número de canais finitos.

| Família | Configurações sorteadas | Energia nos top 10% canais: mínimo / mediana / máximo | Arquivos com top 10% ≥ 50% |
|---|---|---:|---:|
| Doped | 5m LSF, 5m SMF, 5m i29 | 59,6% / 71,1% / 78,6% | 6/6 |
| Multicore | C1 and C2, C1 and C3, C2 and C4 | 16,8% / 25,7% / 40,0% | 0/6 |
| Multimode | Configurations 5, 6, 7 | 93,3% / 93,8% / ~100% | 6/6 |
| Single-mode | Configurations 3, 7, 11 | 29,2% / 41,5% / 83,0% | 2/6 |

O limiar de 50% é **descritivo**, fixado no protocolo, não critério de invalidade
física. Os blocos são correlacionados: 127 blocos não são 127 aquisições independentes.

## Onde o padrão aparece

- **Doped:** canais próximos ao início da faixa espacial dominam a energia.
  A concentração aparece nos 43 blocos avaliados, nas três configurações.
- **Multimode:** os mapas mostram faixas fortes nas extremidades, frequentemente
  em **ambas as pontas**, com interior muito menos energético. Uma faixa contígua
  precisa abranger quase toda a fibra para conter ambas; isso não contradiz o
  fato de poucos canais concentrarem quase toda a energia.
- **Multicore:** energia espacial mais distribuída nesta amostra; nenhum dos
  35 blocos ultrapassou o indicador top10 ≥ 50%. Isso não prova ausência de
  faixas em todos os arquivos multicore do corpus.
- **Single-mode:** depende do arquivo. Em Configuration 7, 10.000 Hz concentrou
  **83,0%** nos top 10%, enquanto 1.000 Hz concentrou **29,2%**. Configuration 11
  ficou em 44,4%/52,3%, e Configuration 3 em 35,8%/38,6%. Não há justificativa
  para uma máscara fixa única por família a partir desses dados.

Um arquivo multimode Configuration 6 contém **44.287 células finitas acima de
1e20**, máximo ~**1,34e36**. É o tipo de patologia numérica já investigado nos
relatórios 7 e 32–35; deve ser separado das faixas de alta amplitude sem essa
patologia. Mesmo excluindo esse arquivo da interpretação, os outros cinco
multimode concentram **93,3–97,7%** da energia nos top 10% canais. O diagnóstico
auxiliar de excluir apenas abs>1e20 não elimina necessariamente toda a faixa
dinâmica residual, como já mostrou o histórico.

## Relação com o treinamento e limites

Esta auditoria analisa a **extensão espacial bruta completa**. O relatório 7
usou amostras preparadas e, em alguns arquivos, recortes centrais de no máximo
512 canais. Por exemplo, Configuration 5 multimode tem 1.012 canais brutos;
seu recorte central deixa de fora as extremidades mais energéticas. Além disso,
as definições de energia e os arquivos/janelas amostrados diferem. **Não tratar
as porcentagens atuais e históricas como o mesmo estimador nem concluir que
todas essas faixas entram no treino tal como aparecem no bruto.**

A repetição sustenta investigar faixa dinâmica espacial, ponderação de perdas
e normalização/regiões, mas **não prova ganho causal de remover ou filtrar
canais**. Também não estabelece se uma faixa é resposta física válida, ruído,
efeito de borda ou falha instrumental. Qualquer exclusão permanente exige
justificativa física e declaração explícita de mudança de alvo/cobertura.

Os arquivos foram sorteados do corpus de 207 HDF5 já inventariado: foram novos
downloads para esta auditoria, **não novas aquisições independentes**. A conclusão
é exploratória; não foi alterado o roadmap nem iniciado experimento neural.
