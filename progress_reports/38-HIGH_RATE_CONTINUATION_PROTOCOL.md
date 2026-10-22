# Continuação do braço 4x à mesma taxa

> Continuação e avaliação concluídas; resultados no
> [relatório 39](39-HIGH_RATE_CONTINUATION_RESULTS.md).

## 1. Protocolo

Partir do **last.pt epoch 24** do braço rate_4x, que também é o best_macro, com
**optimizer AdamW/RNG preservados**. Continuar até epoch **40**: **16 x 4.096 =
65.536 updates adicionais**, LR 3e-5, mesma loss, geometry spatial4-temporal4,
normalização escape-v1 e sampling window. Refresh desligado; nenhum novo dataset.

É continuação verdadeira, não pesos com optimizer reiniciado. A taxa não aumenta:
nominal atual ~42,133x pooled. Patience 41 não interrompe o orçamento de 40 epochs.
Os outros braços 1x/2x e o 4x anterior permanecem congelados como referências;
o maior compute adicional do 4x será declarado.

Source e métricas originais permanecem intactos. Cópias atômicas verificadas por
hash preservam best/best_macro/last e históricos; summary de término anterior
não é copiado como se a continuação já tivesse encerrado. Epoch **24** é baseline
da análise da continuação; epoch 0 permanece no histórico completo.

## 2. Seleção e avaliação

- Mesmo piloto de 102 janelas/validação, incluindo candidatos anteriores.
- Após treino, avaliar selecionado nas **5.820 janelas completas**, juntamente
  com rate_1x, rate_2x e rate_4x anterior congelados.
- Confrontar pior família/configuração, canais robustos, amplitude/fase e custo.
- Se regredir, conservar o best_macro epoch 24; não escolher automaticamente last.
- Nenhum score de teste ou promoção automática. Maior taxa já é custo assumido,
  e a continuação não estabelece generalização entre configurações novas.

## 3. Operação

Runner: `finetuning/run_high_rate_continuation.py`.
Saída: `results/experiments/high_rate_continuation_20261015/`.
Log: `results/experiments/high_rate_continuation_20261015.pipeline.log`.
Serviço **`febus-high-rate-continuation-20261015`**, sequencial na GPU.
**Resultados pendentes nesta atualização**.

Últimas épocas do source mediram ~**11,04 min/epoch**: estimativa **3h15–4h**,
incluindo avaliação final, sujeita a carga/I/O. Mesmo perfil de memória já testado
no sweep, ~5,37 GB máximo alocado no smoke de 512x1.000.

```bash
systemctl --user status febus-high-rate-continuation-20261015

# Somente sem cópia ativa:
.venv/bin/python finetuning/run_high_rate_continuation.py \
  --output-dir results/experiments/high_rate_continuation_20261015 --resume
```

Source/históricos, dataset e código fixados por hashes; lock e save atômico.
Resume restaura optimizer/RNG por epoch completo e não amplia orçamento restante.
Serviço independe da sessão interativa, não garante sobrevivência a reboot/logout.
**151 testes passaram**, incluindo orçamento restante 24→40 e conclusão sem
reiniciar/estender epochs completos, além dos testes prévios de fork/metadata.
