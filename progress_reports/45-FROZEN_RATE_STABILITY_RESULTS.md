# Estabilidade congelada: opção conservadora 2x, limite persistente 4x

## Avaliação concluída

Fonte: `results/experiments/frozen_rate_stability_20261018/`, `COMPLETE` e
`full_validation/COMPLETE`, summaries e grupos/janelas dos oito candidatos.
Serviço terminou com exit 0 em ~52 min. **Zero updates de treino**.
Mesmas 5.920 janelas de validação, referência robusta, custos de índices/side
e parâmetros em cada taxa. Nenhum ensemble na inferência.

| Taxa / candidato | RMSE macro | Correlação | p90 macro por canal | RMS macro |
|---|---:|---:|---:|---:|
| 2x / epoch 24 | 0,592612 | 0,791020 | 0,937458 | 0,820055 |
| 2x / epoch 40 | **0,577962** | **0,802679** | **0,925811** | 0,827215 |
| 2x / loss-selected epoch 38 | 0,579587 | 0,800896 | 0,930830 | 0,821601 |
| 2x / midpoint 24–40 | 0,583174 | 0,798344 | 0,929726 | 0,827074 |
| 4x / epoch 24 | 0,479352 | 0,869126 | 0,717070 | 0,901455 |
| 4x / epoch 40 | **0,466790** | **0,875729** | **0,706400** | 0,898395 |
| 4x / loss-selected epoch 38 | 0,468368 | 0,874871 | 0,708213 | 0,901988 |
| 4x / midpoint 24–40 | 0,471579 | 0,873223 | 0,708164 | 0,904139 |

Taxas pooled permanecem **84,228x** no 2x e **42,811x** no 4x, com as limitações
de contabilização nominal anteriores. Midpoint tem epoch 0 de construção, não
época treinada; os pais são 24/40. Source/checkpoints permanecem imutáveis.

## 2x: melhorar macro não é o único critério

Frente a epoch 24, loss-selected 38 melhora RMSE e correlação nas seis
configurações, mas piora p90 por canal em single-mode Configuration 2
(delta +0,042107) e Configuration 6 (+0,022471). Seleção por loss não resolve
automaticamente a fidelidade generalista.

O **midpoint 2x melhora RMSE, correlação e p90 por canal nas seis configurações**
frente a epoch 24, embora tenha RMSE macro pior que epoch 40. RMSE macro cai
~1,59%; representa compromisso exploratório mais conservador nesses três
critérios de grupo, não dominância em cada janela/canal ou em toda métrica física.

| Configuração single-mode | RMSE 24 | RMSE loss-selected | RMSE midpoint |
|---|---:|---:|---:|
| Configuration 2 | 0,688288 | 0,681160 | 0,681993 |
| Configuration 6 | 0,427504 | 0,424009 | 0,424251 |

O checkpoint midpoint deve ser preservado como candidato de pesquisa, sem
substituir automaticamente os endpoints ou ser promovido como modelo final.

## 4x: regressão reduzida, não eliminada

Single-mode Configuration 2 continua sendo a pior configuração:

| Candidato 4x | RMSE Configuration 2 | Delta frente a epoch 24 |
|---|---:|---:|
| Epoch 24 | 0,604817 | — |
| Epoch 40 | 0,612373 | +0,007557 |
| Loss-selected 38 | 0,610411 | +0,005595 |
| Midpoint | 0,606715 | +0,001899 |

Midpoint também piora correlação (-0,000187) e p90 por canal (+0,003713) desse
grupo frente a epoch 24. Nas demais configurações melhora esses três critérios,
mas isso não autoriza ignorar a exceção. Loss-selected ainda piora p90 multimode
em +0,001458. Nenhum candidato novo 4x supera epoch 24 em todos os grupos nesses
critérios; o menor RMSE macro de epoch 40 continua sendo um trade-off.

## Decisão e próximos critérios

- Encerrar este piloto fixo; **não ampliar busca de alpha sobre a mesma validação**.
- Não iniciar mais épocas apenas porque a curva macro ainda desce. A regressão
  documentada requer critérios por configuração/canal, não só mais compute.
- Preservar checkpoints 24/38/40 e midpoints; não promover nenhum como final.
- Midpoint 2x é candidato para confirmação futura, não uma política padrão de
  averaging. O resultado 4x impede generalizar a técnica a todas as taxas.
- Antes de novo treino de confirmação, fixar critérios de retenção por pior
  configuração e cauda por canal, além de seleção mais representativa que duas
  janelas/arquivo. Critérios devem ser versionados antes da execução, preservando
  o seletor histórico macro como controle; não reinterpretar retrospectivamente
  o teste atual como comparação confirmatória de seletores.
- Para decisão final, faltam tolerâncias físicas/bitrate da aplicação e aquisições
  independentes. Todas as comparações acima reutilizam configurações de
  desenvolvimento; selecionar candidatos a partir delas aumenta viés exploratório.

Nenhum novo treino longo foi lançado após este piloto. Isso evita prolongar uma
trajetória com ganho marginal e perdas localizadas sem hipótese adicional isolada.
