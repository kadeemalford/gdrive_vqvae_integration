# Continuação limitada 2x/4x no fold 0

> Estudo concluído. Resultados e auditoria pareada no
> [relatório 43](43-ROBUST_CONTINUATION_RESULTS_AND_PAIRED_AUDIT.md).

## Hipótese e controle

Os dois candidatos ainda melhoram ao final do orçamento inicial (relatório 41).
Medir ganhos adicionais com **mesma taxa por braço**, preservando optimizer/RNG,
dados, geometria, referência robusta, sampling e objetivo. Nada de novos dados,
novas losses, refresh de codebook ou roteamento por família.

Source imutável: `results/experiments/high_rate_fold0_robust_20261017/`, checkpoints
`rate_2x/last.pt` e `rate_4x/last.pt`, epoch 24. Históricos e selecionados são
copiados atomicamente/verificados por SHA; originais não são sobrescritos.

## Orçamento

- Épocas 25–40: **16 × 4.096 = 65.536 updates adicionais por braço**.
- AdamW e RNG restaurados; LR 3e-5, weight decay 1e-5, seed 2027.
- Geometrias spatial4-temporal2 / spatial4-temporal4, K=2.048 e mesma arquitetura.
- `robust-escape-v1` e validação física, mesma máscara/escala de referência.
- Patience 41 mantém o orçamento limitado de 40 épocas; selecionado anterior
  continua candidato, e last não é automaticamente o melhor.
- Seleção em duas janelas por arquivo; avaliação final dos selecionados nas
  **5.920 janelas** de validação do fold 0.

Comparar frozen 1x, initial 2x/4x e continued 2x/4x. O frozen 1x possui **menos
compute total**: não alegar comparação compute-matched entre todas as taxas.
O contraste principal é antes/depois **dentro de cada taxa**. Reportar regressões
por configuração, caudas por canal, amplitude/fase e bits de índices+side pooled.

## Decisão posterior

Ganhos apenas macro ou pioras em configurações não autorizam promoção. Se ganhos
forem pequenos, priorizar análise dos limites e critérios físicos da aplicação,
não continuação indefinida. Aquisições independentes e tolerâncias de aceitação
continuam necessárias para uma decisão final de generalização/deploy.

## Execução e recuperação

Runner: `finetuning/run_robust_rate_continuation.py`.
Saída: `results/experiments/robust_rate_continuation_20261018/`.
Serviço: `febus-robust-rate-continuation-20261018`.
Log pipeline: `results/experiments/robust_rate_continuation_20261018.pipeline.log`.
Resultados pendentes; estimativa aproximada **4–7 horas**, sujeita a avaliação/IO.

Guardas de source/fold/optimizer/RNG, hashes de código/dataset/referências, locks
e espaço livre mínimo sem exclusões automáticas. Testes incluem rejeição de
mudanças de fold, normalização, geometria, objetivo e estado de continuação.
Validação antes do lançamento: **157 testes passaram**, `git diff --check` sem
erros, protocolo dry-run aprovado e operação CUDA finita em processo novo.
Serviço iniciado em 2026-10-18, com referências anteriores preservadas.

```bash
systemctl --user status febus-robust-rate-continuation-20261018
# Retomar apenas se não houver cópia ativa:
.venv/bin/python finetuning/run_robust_rate_continuation.py \
  --output-dir results/experiments/robust_rate_continuation_20261018 --resume
```

Inibição de idle é temporária, sem mudança permanente de configuração. Suspensão
manual/reboot pode interromper GPU/serviço. O aviso no chat depende de a sessão
de acompanhamento continuar ativa; **não é garantido após reinício do OpenCode**.
