# Densidade temporal: validação completa e continuação longa

> A continuação foi concluída; resultados e decisões subsequentes no
> [relatório 18](18-TEMPORAL_DENSITY_LONG_RESULTS_AND_DECISION_GATES.md).

## 1. Resultado e decisão

A validação completa confirmou os ganhos do piloto: **2x e 4x melhoram RMSE e
correlação nas seis configurações reservadas**, comparadas ao standard selecionado.
4x é melhor nas quatro médias de família; seu RMSE macro cai de **0,867647 para
0,840263** (~3,16% relativo) e correlação sobe de **0,450308 para 0,502038**.

Esse avanço é mais consistente que as adaptações anteriores, mas **não demonstra
fidelidade suficiente**: multimode permanece fraco e a taxa é maior. Não promover
o candidato como modelo final nem tratar mais bits como ganho gratuito.

Os braços densos ainda melhoravam no último epoch do piloto. Decisão: comparar
continuação longa de **2x e 4x**, usando corpus inteiro e validação completa para
medir saturação e trade-off de taxa, sem repetir o standard praticamente estagnado.

## 2. Integridade e população de avaliação

Piloto: `results/experiments/temporal_density_pilot_20261008/`.
Validação ampliada: `results/experiments/temporal_density_full_validation_20261008/`.

- Três braços com seis epochs de 2.048 updates, mesma inicialização de pesos/ordem
  de treino, normalização global escape-v1 e objetivo temporal+VQ.
- Mudança de stride temporal altera representação/receptive field e número de
  índices, sem mudar parâmetro/stride espacial ou alvo. Não é comparação à mesma taxa.
- Candidatos selecionados no piloto de 102 janelas: standard epoch 5, 2x/4x epoch 6.
- Avaliação ampliada **sem updates** nas mesmas 5.820 janelas, 51 arquivos, quatro
  famílias e seis configurações do fold 1; hashes dos checkpoints/seleção registrados.
- Loader stride-aware; previsão de posições conferida com índice real do forward
  nos braços densos. Não interpretar flags de geometria só pelo shape dos pesos.
- Nenhuma consulta ao teste. A validação também informa decisões: não é auditoria
  externa independente, apesar de usar todos os recortes reservados.

## 3. Fidelidade e taxa na validação completa

| Braço | RMSE macro | Correlação macro | Pior família RMSE | Razão nominal pooled índices+side |
|---|---:|---:|---:|---:|
| Standard | 0,867647 | 0,450308 | 0,966653 | 620,23× |
| Temporal-2x | 0,856491 | 0,471902 | 0,961211 | 325,09× |
| Temporal-4x | **0,840263** | **0,502038** | **0,951289** | **165,91×** |

| Família/configuração | Standard | 2x | 4x |
|---|---:|---:|---:|
| Doped / 5m SMF | 0,824581 | 0,808211 | 0,791866 |
| Multicore / C1 and C4 | 0,920902 | 0,909930 | 0,890618 |
| Multimode / Configuration 7 | 0,966653 | 0,961211 | 0,951289 |
| Single-mode / Configuration 11 | 0,708377 | 0,707206 | 0,703355 |
| Single-mode / Configuration 3 | 0,792619 | 0,786004 | 0,736593 |
| Single-mode / Configuration 7 | 0,774358 | 0,746619 | 0,741888 |

Razões pooled usam soma dos bits para 465.920.000 células de entrada, não médias
de razões. Bits de índices: **22.176.000 / 44.000.000 / 88.000.000**; side stream
232.800 bytes em cada braço. Esses valores não incluem container, mapas físicos,
model IDs/arquitetura e codificação entrópica; não são codecs completos implantáveis.

### Forma de onda e canais

| Métrica macro | Standard | 2x | 4x |
|---|---:|---:|---:|
| RMS ratio | 0,454018 | 0,465033 | 0,499811 |
| Erro magnitude FFT | 0,819905 | 0,810326 | 0,786750 |
| Alinhamento fase FFT | 0,793429 | 0,812256 | 0,820360 |
| Mediana de erro relativo por canal | 0,917906 | 0,907219 | 0,902200 |
| p90 de erro relativo por canal | 1,112015 | 1,095191 | 1,116217 |

Amplitude/fase melhoram no macro, mas p90 não é monotônico. A média de erros
relativos por canal continua sensível aos canais quase constantes. Ganho em todas
as configurações não implica ganho em cada canal; os requisitos físicos da aplicação
continuam necessários para decidir qualidade/custo aceitáveis.

## 4. Leitura da curva e relação com o controle clássico

No piloto, 2x e 4x escolheram a última época e continuaram reduzindo RMSE macro.
O standard selecionou epoch 5 e teve regressão de single-mode frente à própria
inicialização. Isso justifica testar mais exposição dos candidatos densos, não
assumir que qualquer continuação longa será benéfica.

O DCT 4x do relatório 15 teve RMSE macro ~0,831461 na **amostra** de 102 janelas,
contra ~0,839339 do piloto neural 4x na mesma amostra. Seu orçamento/bitstream é
diferente e ligeiramente maior, então não declarar um vencedor exato por bitrate.
Também há trade-offs por família: rede melhor em doped, DCT forte em single-mode.
Não comparar diretamente os números da amostra com médias da validação completa.

Uma seed/fold e poucas configurações por família limitam a robustez. A continuação
vem antes de decisões de implantação, não substitui outros folds/seeds/holdout externo.

## 5. Continuação longa iniciada em background

Runner: `finetuning/run_temporal_density_long.py`.
Saída: `results/experiments/temporal_density_long_20261008/`.
Log: `results/experiments/temporal_density_long_20261008.log`.
Resultados longos pendentes nesta atualização; nenhuma promoção presumida.

| Item | Protocolo |
|---|---|
| Braços | temporal-2x / temporal-4x |
| Inicialização | cada braço parte do próprio best_macro epoch 6 do piloto |
| Otimizador | AdamW **novo**, LR **3e-6**, weight decay 1e-5 |
| Dados | todas as **12.140 janelas de treino**, uma vez por epoch, seed 2026 |
| Orçamento máximo | **8 epochs por braço**, até 194.240 updates adicionais totais |
| Validação | **5.820 janelas completas**, quatro famílias, por epoch |
| Seleção/parada | RMSE macro restaurado, patience 3, min_delta 1e-4 |
| Candidatos | inicialização epoch 0 preservada, best_loss/best_macro/last separados |
| Normalização/geometria | escape-v1, física e strides herdados/preservados |
| Teste | fora da seleção/avaliação |

Não é simplesmente `--resume` para baixar LR: um optimizer retomado restauraria
sua LR antiga. Os novos braços começam com weights do piloto e optimizer fresco.
Dentro da continuação, resume restaura o optimizer **3e-6** e o RNG salvo.

Orçamento máximo e ordem de dados são pareados; early stopping pode realizar
orçamentos diferentes. Fonte/checksums/protocolo são imutáveis no resume. Recuperação
de summary após interrupção no último save usa somente estado já gravado, sem
inventar métricas. Dataset completo de treino/validação é verificado antes do lançamento.

## 6. Operação e análise posterior

Estimativa preliminar: **algumas horas (~4–8 h)** para os dois braços, dependente
de I/O, formas, validação completa e parada. Atualizar ETA com epochs medidos.
Não foram lançados treinos concorrentes na GPU.

```bash
# Recuperar a sequência caso o serviço/terminal seja interrompido:
.venv/bin/python finetuning/run_temporal_density_long.py \
  --output-dir results/experiments/temporal_density_long_20261008 --resume

# Ao concluir, analisar inicial/selecionado/final e custos:
.venv/bin/python finetuning/summarize_numeric_adaptation.py \
  --experiment-dir results/experiments/temporal_density_long_20261008
```

Background depende do serviço OpenCode; um terminal independente/tmux permite
retomada sem essa dependência. Checkpoints são por epoch completo; trabalho do
epoch interrompido será repetido. Saves atômicos e lock impedem sobreposição do runner.

**108 testes passaram** antes de iniciar, incluindo controles de orçamento restante,
early stopping retomável e rejeição de geometria divergente. Os checkpoints fonte
e dados usados na continuação permanecem intactos.

Depois: comparar saturação, pior família/configuração, canais, amplitude/fase e
fronteira de taxa; preservar a inicialização se a continuação regredir. Se houver
ganho sustentado, confirmar em seeds/configurações adicionais; se houver plateau,
investigar arquitetura/inicialização, representação ou sistema híbrido, sem
converter automaticamente o checkpoint de maior bitrate em modelo final.
