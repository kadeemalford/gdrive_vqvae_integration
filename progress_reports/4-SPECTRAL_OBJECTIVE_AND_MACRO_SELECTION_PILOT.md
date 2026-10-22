# Piloto de objetivo espectral e seleção macro por família

Data: 2026-10-07. Artefatos: `results/experiments/spectral_pilot_20261007/`.

## Decisão

**Não iniciar comparação longa com esta perda espectral/peso.** Magnitude FFT
melhorou ligeiramente nas quatro famílias, mas RMSE macro melhorou em só 1/4 folds
de validação e alinhamento de fase piorou nas quatro famílias. Recuperar energia
não se traduziu em reconstrução generalista melhor.

Instrumentação macro e seleção separada foram implementadas para próximos testes.
Manter perda temporal como referência; não promover checkpoints do piloto nem
alterar pesos originais/baselines. O resultado limita esta fórmula, peso,
inicialização e orçamento, não descarta toda perda espectral nem prova escassez de dados.

## 1. Implementação

Com `--validation-details`, o treinador registra por época:

- `validation_metrics.csv`: agregação por janela, macro e pior família;
- `validation_groups.csv`: resultados por família/configuração;
- perda temporal separada de VQ, perda espectral, RMSE, correlação, RMS ratio,
  magnitude FFT, fase e potência em bandas normalizadas à Nyquist;
- `best.pt`: menor val_loss temporal+VQ por janela, como no protocolo histórico;
- `best_macro.pt`: menor RMSE relativo macro (configurações uniformes dentro da
  família, famílias uniformes entre si), sem VQ na métrica;
- `last.pt`: pesos, AdamW, época, RNG e estado de ambos os seletores.

`--val-samples-per-file 0` usa validação completa; valor positivo é cap estratificado
de piloto. Manifesto rejeita janelas de teste no seletor. Hash da seleção impede
retomar com outra seleção. Defaults preservam perda/CSV históricos.

### Objetivo opcional

`L = 0,5*MSE + 0,5*L1 + VQ + lambda*L_STFT`.

`L_STFT`: média, para FFT sizes 128, 256 e 512, de
`0,5*|||S(recon)| - |S(original)|| / ||S(original)||`
mais `0,5*mean(abs(log1p(|S(recon)|) - log1p(|S(original)|)))`.

STFT por canal, Hann, hop N/4, sem padding central; magnitudes normalizadas pela
norma L2 da janela. Só escalas que cabem na entrada são usadas. Sem resampling,
conversão Strain/Strain Rate ou mudança geométrica. Perda temporal permanece na
fórmula: magnitude não controla sozinha fase/forma de onda. Piloto usou lambda 0,1;
padrão permanece 0.

### Inicialização como candidata

Após analisar o piloto, adicionado `--evaluate-initial` com `--validation-details`:
avalia a inicialização como época 0 e a mantém como candidata em best/best_macro,
com estado retomável em last. Evita escolher obrigatoriamente uma época posterior
pior. Época 0 aparece nos CSVs de validação, não como treino em `metrics.csv`.

Proteção adicionada **depois** do piloto: seus checkpoints selecionam épocas 1–3.
A origem foi comparada separadamente, com hash das mesmas janelas, sem alterar
retrospectivamente os checkpoints.

## 2. Protocolo executado

- Quatro folds, braços pareados `temporal` (lambda 0) e `spectral` (lambda 0,1).
- Mesma inicialização em cada par: best uniforme do mesmo fold no experimento completo.
  AdamW **novo** em ambos, LR `3e-6`, seed 2026. Ramificação de pesos, não continuação
  do estado AdamW anterior.
- Sampler uniforme, três épocas × 256 atualizações: 768 por braço/fold, 6.144 totais.
  Mesma seleção/ordem de treino; apenas termo espectral muda.
- Duas janelas por arquivo de validação: 102/102/72/138 nos folds 0/1/2/3, total 414,
  207 arquivos, 23 configurações. Mesma seleção do diagnóstico anterior, hashes verificados.
- Comparação de seletores na mesma trajetória; early stopping histórico preservado.
- Nenhuma avaliação nova nos testes. Seleção/avaliação usam validação, portanto os
  resultados são exploratórios, não estimativa independente de generalização.
- Oito braços concluíram três épocas; soma dos tempos das épocas ~8,54 minutos,
  excluindo carga/salvamento de checkpoints e relatório.

## 3. Resultados na validação

Checkpoints selecionados por RMSE macro em ambos os braços; menor é melhor.

| Fold | Temporal | Espectral | Delta espectral − temporal |
|---|---:|---:|---:|
| 0 | 0,88315 | 0,88401 | +0,00086 |
| 1 | 0,87098 | 0,87078 | −0,00020 |
| 2 | 0,86757 | 0,86767 | +0,00010 |
| 3 | 0,94183 | 0,94264 | +0,00081 |

Diferenças pequenas, uma semente e orçamento curto: sem declaração de significância
estatística. Não há sinal suficiente para escalar essa configuração.

### Famílias: critério macro em ambos os braços

| Família | RMSE temporal / espectral | Corr. temporal / espectral | Erro FFT temporal / espectral | Fase temporal / espectral |
|---|---:|---:|---:|---:|
| Doped | 0,87821 / 0,87901 | 0,4841 / 0,4842 | 0,7612 / 0,7578 | 0,7362 / 0,7330 |
| Multicore | 0,92629 / 0,92632 | 0,3714 / 0,3714 | 0,8836 / 0,8813 | 0,7788 / 0,7721 |
| Multimode | 0,98511 / 0,98624 | 0,1909 / 0,1905 | 0,9130 / 0,9114 | 0,4658 / 0,4631 |
| Single-mode | 0,80682 / 0,80684 | 0,5230 / 0,5238 | 0,7251 / 0,7228 | 0,7281 / 0,7274 |

Magnitude FFT caiu em todas as famílias, mas RMSE piorou em todas nessa agregação.
RMS ratio temporal → espectral: doped 0,5420 → 0,5522, multicore 0,3731 → 0,3789,
multimode 0,2794 → 0,2862, single-mode 0,5705 → 0,5758. Recuperar energia sem
alinhamento da forma de onda não é suficiente.

### Comparação com a inicialização

Média das quatro famílias, uniformizando configurações dentro de cada família:

| Modelo | RMSE macro | Corr. macro | Erro FFT macro | Fase macro |
|---|---:|---:|---:|---:|
| Best uniforme de origem | 0,89895 | 0,39260 | 0,82046 | 0,67689 |
| Continuação temporal, macro pós-treino | 0,89910 | 0,39236 | 0,82071 | 0,67721 |
| Continuação espectral, macro pós-treino | 0,89960 | 0,39248 | 0,81832 | 0,67390 |

Nenhuma ramificação melhorou RMSE macro agregado frente à origem. Não promover
checkpoint apenas por ser novo; manter opção da época 0 nas próximas continuações.

### Comparação dos seletores

Fold 1 temporal: perda seleciona época 2 e macro época 3, RMSE 0,87154 → 0,87098.
Fold 1 espectral: perda seleciona época 1 e macro época 3, RMSE 0,87110 → 0,87078.
Nos demais folds, seletores coincidem na época 1. Valida a implementação dos dois
seletores, não prova que macro sozinha resolve reconstrução. Macro ainda pode ocultar
regressão de uma família; acompanhar pior família e usar validação completa em treinos
longos. Perdas totais com lambdas diferentes não são diretamente comparáveis.

## 4. Próximo passo recomendado

1. Manter instrumentação macro, inicialização candidata e best histórico separados.
2. Não escalar essa variante espectral isolada. Se revisitar, considerar fase/complexo
   e bandas sem inventar resolução física. FFT complexa com MSE pode ser redundante
   com perda temporal por Parseval; exigir ablação, não assumir benefício.
3. **Priorizar teste pequeno de capacidade/overfit e rate-distortion**, em janelas
   exclusivamente de treino de famílias fortes/fracas. A arquitetura consegue
   reconstruir até exemplos memorizados? Se necessário, comparar com autoencoder
   contínuo realmente treinado: latente contínuo do diagnóstico anterior não é esse
   controle. Medir antes de atribuir falha à massa de dados.
4. Se houver limitação de capacidade/taxa, comparar stride menor/multiescala com
   orçamento pequeno, reportando custo/razão de compressão. Se treino for bem ajustado
   e validação continuar ruim, priorizar cobertura/normalização/geometria e dados FEBUS-like.
5. MoS/ensemble, corpus ampliado e treino do zero continuam condicionais. Modelo final adiado.

## 5. Reprodução e operação

Comando executado (concluído):

```bash
.venv/bin/python finetuning/run_spectral_experiment.py --pilot \
  --output-dir results/experiments/spectral_pilot_20261007 --quiet
```

Outro piloto, incluindo a proteção da inicialização:

```bash
.venv/bin/python finetuning/run_spectral_experiment.py --pilot --evaluate-initial \
  --output-dir results/experiments/spectral_pilot_new
```

Sem `--pilot`: até dez épocas completas/validação completa, **não recomendado
automaticamente para esta variante**. Diretório novo; `--resume` preserva flags,
checkpoints/logs. `--dry-run` verifica plano. Progresso no terminal por padrão;
`--quiet` mantém detalhes só nos logs. Relatório/JSON gerados ao término.

Novas opções Make: `FINETUNE_SPECTRAL_WEIGHT`, `FINETUNE_VALIDATION_DETAILS=1`,
`FINETUNE_VAL_SAMPLES_PER_FILE`, `FINETUNE_EVALUATE_INITIAL=1`. Última requer details.
Mudança de objetivo/seleção de validação exige novo experimento, não resume diferente.

**31 testes passaram**, incluindo gradientes finitos, silêncio, estratificação, macro,
época 0, retomada e rejeição de mudança de objetivo. Piloto terminou; **nenhum
processo em background ou treinamento longo foi iniciado**.
