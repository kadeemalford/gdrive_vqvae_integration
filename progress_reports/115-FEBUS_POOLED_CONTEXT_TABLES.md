# Compatibilidade de tabela contextual: reduzir a instalação sem relaxar a precisão

**Protocolo prospectivo.** Continuação dos [resultados de transferência](113-FEBUS_FROZEN_INSTALLATION_AMORTIZATION.md#resultados-e-transferência)
e da [auditoria](114-FEBUS_DEPLOYMENT_DECISION_AUDIT.md).
Hipótese: compartilhar linhas entre contextos que diferem apenas pelo sinal
do vizinho espacial pode reduzir a instalação e estabilizar contagens escassas,
sem sacrificar a maior parte da economia de transporte. Pode também perder
dependência de sinal relevante; isso é que será medido.

## Contraste fixo

- Usar as mesmas janelas **fit de 114** e **check de transferência de 116**.
  Checks já consumidos: diagnóstico de desenvolvimento, não confirmação nova.
  Nenhum arquivo, janela, grupo ou precisão será trocado pelo resultado.
- Reusar ε, K, reset, geometria e ordem causal. Manter todos os 18 estados
  temporais e, para o vizinho espacial, usar magnitude sem sinal, zero e
  indisponível: **10 estados espaciais**, até **180 linhas contextuais**
  em vez de 324. Somar as contagens no fit antes de aplicar a mesma
  suavização e a mesma regra de backoff. Manter mapa de 324 entradas
  apontando às linhas compartilhadas, sem novo decoder.
- Comparar a reconstrução por hashes e roundtrip com os coders do 118;
  nenhum ganho de bytes pode resultar de diferença nos índices ou no
  waveform quantizado. Comparar ao **contextual completo** para isolar o
  efeito desta intervenção e ao **marginal** para situar a economia total.
- Medir instalação por configuração, transporte e total nos quatro
  horizontes já definidos (1, 5, 10 e 20 janelas por arquivo), em ambas as
  precisões. Sem varrer número de estados, outros buckets ou modelos.

## Decisão prospectiva

O ponto primário é **ε/σ_cal=0,03 a 20 janelas**. Considerar promissora para
confirmação a candidata que reduzir **≥3%** dos bytes completos do contextual
integral, sem mais de **1%** de regressão em nenhuma configuração nesse ponto.
Resultados menores, positivos e bit-exatos podem ser registrados como ganho
incremental ou redução de instalação; não haverá ajuste pelos checks.
O ponto 0,1 e os horizontes menores informam robustez e trade-offs, não
substituem o primário. Se não houver ganho primário, encerrar a hipótese de
pooling adotada, sem repetir com variantes. O critério de 116 permanece
inalterado; esta triagem **não libera F4-A** ou confirma generalização.

Somente CPU e NPZ de treino; sem HDF5, reservas, GPU ou sessões novas.

## Resultado

No primário, **52.001 B** a menos na instalação não compensaram **346.339 B**
a mais no transporte: **+2,62%** no total contra o contextual integral.
Seis configurações regrediram mais de 1%. Foram confirmadas 760
reconstruções idênticas e zero violações; 49 testes passaram.
**Hipótese encerrada, sem variantes nem promoção.**
[Resultados por horizonte](../results/experiments/feasibility_pooled_context_20261118/retry_01/results.json).
