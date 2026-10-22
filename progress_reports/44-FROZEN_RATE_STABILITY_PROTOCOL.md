# Piloto congelado de estabilidade de checkpoints 2x/4x

> Avaliação concluída; resultados e decisão no
> [relatório 45](45-FROZEN_RATE_STABILITY_RESULTS.md).

## Hipótese limitada

Testar se um ponto intermediário fixo na trajetória preserva ganhos adicionais
sem a regressão observada em Configuration 2. Não é pedido de nova arquitetura,
novo treinamento, ensemble por janela ou especialistas por configuração.

Para **cada taxa**, avaliar quatro candidatos nas mesmas 5.920 janelas:

1. Epoch 24, selecionado do estudo inicial.
2. Epoch 40, selecionado macro da continuação.
3. Epoch 38, `best.pt` já retido por menor loss (controle de seletor alternativo).
4. **Midpoint 50/50 dos pesos 24/40**, coeficiente fixo, sem busca de alpha.

O midpoint inclui encoder, decoder e embeddings VQ. A continuação preserva os
índices dos códigos, mas a operação é não linear: embeddings intermediários podem
piorar quantização. Não supor alinhamento funcional perfeito nem equivalência à
média de reconstruções. É uma única rede/um único codebook na inferência, sem
novos parâmetros, índices adicionais ou metadados por janela.

## Integridade e métricas

Fontes mantidas imutáveis e referenciadas por SHA; midpoint exige config, shape,
dtype, fold e normalização coerentes, pesos finitos e igualdade de buffers não
flutuantes. Média computada em float64 e convertida ao dtype original.
Checkpoint midpoint tem epoch 0 de construção e metadata explícita de pais
24/40, **sem optimizer/RNG para treino**; não apresentá-lo como epoch treinada.

Comparar RMSE/correlação por configuração, pior configuração/família, caudas por
canal, RMS/fase/banda alta e bits pooled. Qualquer melhora macro acompanhada de
regressão por configuração permanece trade-off, não promoção automática.
Não definir tolerância de regressão sem requisitos físicos da aplicação.

Validação reutilizada orientou a hipótese: **exploração, não teste independente**.
Não usar o melhor entre estes candidatos para alegar resultado confirmatório.
Se midpoint piorar, registrar a negativa sem expandir automaticamente uma busca
de coeficientes sobre as mesmas configurações.

## Validação e execução

**162 testes passaram**, incluindo auditoria pareada/cluster bootstrap e média de
pesos com rejeição de estados incompatíveis. Smoke GPU dos dois midpoints em
uma janela multimode com escapes (512x1.000) e uma de single-mode Configuration 2
(111x1.000): saídas/VQ finitos, shapes e contagens de índices corretos; restauração
física executada. Nenhum update ou peso do smoke salvo/reutilizado.
Evidência: `results/experiments/frozen_rate_stability_smoke_20261018/`.

Runner: `finetuning/run_frozen_rate_stability.py`.
Saída: `results/experiments/frozen_rate_stability_20261018/`.
Serviço: `febus-frozen-rate-stability-20261018`.
Log: `results/experiments/frozen_rate_stability_20261018.pipeline.log`.
Oito avaliações completas, **zero updates de treino**; resultados pendentes.

```bash
systemctl --user status febus-frozen-rate-stability-20261018
# Apenas sem processo ativo:
.venv/bin/python finetuning/run_frozen_rate_stability.py \
  --output-dir results/experiments/frozen_rate_stability_20261018 --resume
```

Lock, hashes e avaliação retomável preservam tentativas parciais. Inibição de idle
temporária, sem alteração permanente; não garante sobrevivência à suspensão
manual/reboot. Aviso no chat não é garantido após reinício do OpenCode.
