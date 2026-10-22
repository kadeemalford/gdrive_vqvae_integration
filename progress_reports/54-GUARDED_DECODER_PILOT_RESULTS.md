# Full versus decoder-only: ganho pequeno e nenhum admitido pela guarda

## Conclusão completa

Fonte: `results/experiments/guarded_decoder_pilot_20261019/`, `COMPLETE`,
`full_validation/COMPLETE`, `selection.json`, históricos e snapshots das oito
épocas de cada braço. Ambos concluíram **32.768 updates** com LR 3e-6, optimizer
novo por braço e validação completa a cada época. Todos os 16 snapshots foram
preservados. Avaliação final dos selecionados: mesmas 5.920 janelas.

| Candidato | Época do piloto | RMSE macro | Correlação | p90 macro por canal | RMS macro |
|---|---:|---:|---:|---:|---:|
| Baseline = source epoch 24 | 0 | 0,479352 | 0,869126 | 0,717070 | 0,901455 |
| Full, selecionado macro | 8 | **0,476036** | **0,870449** | 0,715337 | 0,896813 |
| Decoder, selecionado macro | 8 | 0,477693 | 0,869518 | **0,714645** | 0,894861 |
| Full, guarda completa | **0** | 0,479352 | 0,869126 | 0,717070 | 0,901455 |
| Decoder, guarda completa | **0** | 0,479352 | 0,869126 | 0,717070 | 0,901455 |

Ganhos relativos de RMSE do selecionado macro: **~0,69% full / ~0,35% decoder**.
Compressão pooled permanece **42,811x**, parâmetros/índices/side sem aumento.
Ambos atenuam mais a amplitude macro. Fase FFT: baseline 0,929120, full 0,930000,
decoder 0,929482. Não houve melhoria universal de todos os diagnósticos.

Época 0 da guarda é um **modelo pré-treinado**, o source original de epoch 24,
não uma rede não treinada. Os selecionados pela guarda reproduziram exatamente
seus scores. Nenhum candidato foi promovido como final.

## Configurações no selecionado macro

| Família/configuração | RMSE baseline | Full 8 | Decoder 8 |
|---|---:|---:|---:|
| Doped / 5m LSF | 0,433689 | 0,430538 | 0,433218 |
| Multicore / C1 and C3 | 0,526922 | 0,524505 | 0,525538 |
| Multimode / Configuration 6 | 0,566143 | 0,563810 | 0,565748 |
| Single-mode / Configuration 10 | 0,215361 | 0,204972 | 0,207184 |
| Single-mode / Configuration 2 | 0,604817 | 0,602171 | 0,602473 |
| Single-mode / Configuration 6 | 0,351781 | 0,348726 | 0,349143 |

As seis configurações melhoram RMSE nos dois selecionados; full também supera
decoder em RMSE e correlação nas seis. Mas decoder tem vantagens de p90 em alguns
grupos. Congelar encoder/codebook não trouxe uma solução global de preservação.
Compute e parâmetros otimizados diferem, apesar de iguais budgets de updates.

## Por que a guarda rejeitou todos os 16 snapshots

`selection.json` registra cada violação por época. Nenhum passou simultaneamente
RMSE/correlação/p90 por configuração contra a fase 0. Exemplos finais:

- **Full 8:** p90 por canal pior em doped (+0,000820), multimode (+0,002578) e
  single-mode Configuration 2 (+0,002516), apesar de RMSE melhor nos seis grupos.
- **Decoder 8:** doped tem correlação menor em **0,000718** e p90 pior em
  **0,0000404**. Esses são os únicos vetos da época final desse braço.
- Full teve regressão de p90 multimode e single-mode Configuration 2 nas oito
  épocas. Decoder teve perda de correlação doped nas oito épocas.

Não exagerar a relevância física desses deltas, especialmente o último p90:
**epsilon numérico não é tolerância de aplicação**. A guarda conservadora
preservou baseline conforme pré-especificado; não provou que os candidatos
sejam inaceitáveis em qualquer uso. Também não ajustar tolerância após observar
esses resultados apenas para admitir um candidato.

Não foi um erro da amostra curta: cada época usou o conjunto completo. O limite
agora é o compromisso entre critérios e o objetivo de treino, não somente
representatividade do seletor. Isso justifica testar uma hipótese de objetivo,
em vez de repetir mais épocas do mesmo piloto.

## Próximo passo limitado

Uma penalização train-only, leve e **com pesos limitados**, de MSE por canal,
para verificar se preserva melhor caudas sem mudar representação, taxa ou
arquitetura. Não usar inverso de variância sem limite, porque canais quase
constantes já geraram diagnósticos relativos enormes nos estudos anteriores.

Reutilizar **o full completo deste piloto como controle**, sem repeti-lo.
Novo braço full compartilha source/LR/seed/budget/guarda e muda apenas o termo
auxiliar de treino. Protocolo no [relatório 55](55-BOUNDED_CHANNEL_OBJECTIVE_PROTOCOL.md).

Não promover decoder-only, não estender automaticamente este piloto e não
confundir seleção na validação completa com confirmação externa. Continuam
faltando tolerâncias físicas/bitrate e aquisições independentes para decisão final.
