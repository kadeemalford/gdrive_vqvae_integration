# Adaptação longa: resultados de ajuste completo versus decoder-only

## Conclusão

O ajuste completo foi superior ao decoder-only, mas o avanço é pequeno:
RMSE macro **0,871100 → 0,868975** (~0,24% de redução relativa).
As quatro médias de família melhoraram no melhor checkpoint completo; **single-mode
Configuration 3 regrediu**, mostrando que a média por família também pode esconder
regressões. Multimode continua difícil, com RMSE **0,968188** e correlação **0,248242**.

Não prolongar automaticamente a mesma adaptação nem promover o modelo como solução
generalista. Usar o melhor completo como referência de pesquisa e investigar
representação/normalização, mantendo baselines, dados e checkpoints anteriores.

## 1. Protocolo e integridade

Experimento: `results/experiments/preservation_long_20261007/`.

- Fold 1, uniform `best.pt` como fonte comum, fresh AdamW LR 1e-6, weight decay 1e-5.
- Full: todos os parâmetros; decoder: apenas decoder, encoder/quantizador congelados.
- 8 epochs completos por braço, **12.140 updates por epoch**, 97.120 por braço.
- Validação completa: **5.820 janelas de 51 arquivos**, quatro famílias, seis configurações.
- Mesmo hash de validação nos braços, inicialização incluída como candidata, escape-v1.
- Parada/seleção macro por configuração dentro da família e entre famílias; não teste.
- Full foi retomado por resume; CSVs têm epochs 1–8 sem lacunas/duplicatas.
- Tempo das épocas completas: **2,33 h full / 1,57 h decoder**; não inclui todos os
  custos de inicialização, saves e trabalho interrompido/refeito.

Verificação nos pesos reais: **55 tensores de encoder/quantizador do decoder-only
bit-exatos à origem**, 62 tensores do decoder alterados. Resultado em `integrity.json`.
VQ loss do decoder-only permaneceu constante na validação, coerente com congelamento.

## 2. Seleção, evolução e fidelidade

| Braço | Estado | Epoch | RMSE macro | Correlação macro | Pior família RMSE |
|---|---|---:|---:|---:|---:|
| Ambos | Inicialização | 0 | 0,871100 | 0,441494 | 0,970075 |
| Full | Melhor macro | 6 | 0,868975 | 0,446423 | 0,968188 |
| Full | Final | 8 | 0,869154 | 0,446558 | 0,968293 |
| Decoder | Melhor macro | 6 | 0,869765 | 0,444763 | 0,968497 |
| Decoder | Final | 8 | 0,869925 | 0,444858 | 0,968365 |

Melhor val_loss temporal+VQ: full epoch 2, decoder epoch 7; melhor macro epoch 6 nos
dois. Seletores respondem a coisas diferentes. Epoch 8 não supera epoch 6 no RMSE
macro, embora algumas correlações e configurações continuem melhorando.

| Família | RMSE inicial | Full selecionado | Decoder selecionado |
|---|---:|---:|---:|
| Doped | 0,831078 | 0,829792 | 0,831491 |
| Multicore | 0,922680 | 0,921596 | 0,922030 |
| Multimode | 0,970075 | 0,968188 | 0,968497 |
| Single-mode | 0,760569 | 0,756322 | 0,757042 |

Full melhora RMSE em **5/6 configurações**. A exceção:
single-mode Configuration 3, **0,790683 → 0,796269**, correlação
**0,573771 → 0,567662**. No decoder ela também piora, além da regressão em doped.

Amplitude permanece um gargalo. No full, RMS ratio macro **0,446869 → 0,442542**;
em multimode **0,265191 → 0,254669**. O erro de magnitude FFT macro melhora pouco,
**0,824422 → 0,822982**. Pequena melhora de RMSE/correlação não equivale a recuperar
amplitude e detalhes; isso desaconselha atribuir a baixa fidelidade somente a
overfitting agressivo ou à falta de mais epochs conservadores.

## 3. Limites e decisão seguinte

- Um fold/seed, poucas configurações por família e seleção na mesma validação usada
  para medir ganhos: resultados exploratórios, não auditoria externa independente.
- Congelar encoder/codebook não foi melhor nesse orçamento; não prova que todas as
  formas de fine-tuning parcial/replay sejam inúteis.
- Não comparar automaticamente estes resultados com o fold 2 do piloto numérico:
  populações de validação e cobertura de famílias são diferentes.
- O checkpoint full epoch 6 é referência para ablação posterior, não produção final.

Próximo estudo: representação da escala dos canais, reversibilidade e overhead,
descrito no [relatório 11](11-CHANNEL_REPRESENTATION_AUDIT_AND_PILOT.md).
O roadmap permanece atemporal: essa transição liga adaptação generalista à etapa
de representação, sem fixar uma arquitetura ou especialista como solução.

Artefatos de análise: `REPORT.md`, `comparison.json`, históricos completos por epoch,
família/configuração e verificação de congelamento. Nenhum dado ou checkpoint apagado.
