# FEBUS: compartilhar linhas reduz a instalação, mas aumenta o transporte

Resultado do [protocolo 120](120-FEBUS_POOLED_CONTEXT_TABLES.md). O painel
de transferência foi reutilizado; nenhum resultado constitui confirmação nova.
As mesmas duas precisões, índices quantizados e reconstruções foram preservados.

| ε/σ_cal | Janelas/arquivo | Bytes totais com compartilhamento | Variação contra contextual integral | Economia contra marginal |
|---|---:|---:|---:|---:|
| 0,03 | 1 | 833.669 | −4,06% | −24,76% |
| 0,03 | 5 | 3.083.995 | +1,12% | −2,12% |
| 0,03 | 10 | 5.894.833 | +2,12% | +1,07% |
| 0,03 | 20 | 11.532.542 | **+2,62%** | +2,72% |
| 0,1 | 1 | 501.337 | +0,60% | −3,97% |
| 0,1 | 5 | 1.961.366 | +5,01% | +6,49% |
| 0,1 | 10 | 3.782.764 | +5,78% | +8,01% |
| 0,1 | 20 | 7.441.695 | +6,14% | +8,83% |

No primário (0,03 e 20 janelas), **52.001 bytes poupados de instalação**
não compensaram **346.339 bytes adicionais de transporte**. Seis
configurações tiveram regressão superior a 1%. O ganho a uma janela foi
somente contra o contextual integral; o marginal custou menos.

**Decisão:** encerrar este compartilhamento de linhas, sem ajuste no check
ou novas variantes. A evidência é compatível com a utilidade de manter o
sinal espacial explícito nesse desenho, mas não isola informação de sinal
de mudanças nas contagens/pooling. A economia do contextual integral segue
válida; não promover esta candidata nem iniciar F4-A.

Foram conferidos 760 casos com hashes de reconstrução coincidentes com 116,
zero violações de ε, 64 transportes no ledger e 49 testes. Execução CPU em
100,66 s; [artefatos](../results/experiments/feasibility_pooled_context_20261118/retry_01/).
A tentativa interrompida foi preservada; corrigiu-se a validação de geometria
do fit, sem mudar o contraste ou selecionar novos dados. Sem GPU ou reservas.
