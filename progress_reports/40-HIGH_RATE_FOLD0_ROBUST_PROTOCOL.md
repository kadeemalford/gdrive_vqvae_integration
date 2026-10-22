# Confirmação da curva neural no fold 0 com referência robusta

> Estudo e avaliação completos. Resultados e decisão no
> [relatório 41](41-ROBUST_RATE_FOLD0_RESULTS.md).

## 1. Objetivo e source

Comparar orçamento atual, ~2x e ~4x índices em novas configurações, sem presumir
que o ganho do fold 1 se repete. Mesmo source Spatial fold 0 epoch 23, **não pesos
do fold 1**, e seed de adaptação 2027 nos três braços.

Os 40 casos de faixa dinâmica residual da validação recebem a política experimental
**robust-escape-v1**, idêntica nos três candidatos. O treino desse fold foi auditado
como inalterado pela política. Não é um ensaio de retreinar para corrigir normalização:
é confirmação de taxas com avaliação numérica adequada.

## 2. Integração experimental da política

- Loader de dados e máscara de loss usam a mesma política de normalização explícita.
- Restauração física, referência, média/escala e região ordinária são robustas
  para **todos** os candidatos desta comparação, inclusive source congelado.
- Escapes bit-exatos com índices/valores transmitidos; side é cobrado na taxa.
- Eval rejeita mistura robust/legacy: denominadores diferentes não podem aparecer
  como ranking comparável. Para comparação de políticas, usar evaluator congelado
  com máscara comum do relatório 34.
- Defaults legacy/escape-v1/channel permanecem iguais. A política robusta não
  virou automaticamente default de produção ou foi aplicada aos estudos anteriores.
- Resume rejeita mudança de normalização e usa a representação do protocolo.

`source_robust.pt` contém **exatamente as mesmas weights/config** do source original,
verificadas tensor a tensor, sem optimizer herdado ou updates. Somente metadata
de pré-processamento/reencoding é explicitamente identificada, com source SHA/epoch.
O source original nunca é sobrescrito; wrapper congelado tem epoch 0 de reencoding.

## 3. Orçamento e comparação

| Braço | Geometria | Índices em 512x1.000 |
|---|---|---:|
| Rate 1x | spatial-4x | 8.064 |
| Rate 2x | spatial4-temporal2 | 16.000 |
| Rate 4x | spatial4-temporal4 | 32.000 |

Mesmo K=2.048, parâmetros/arquitetura, alvo bruto, sampling window e optimizer
**AdamW novo**, LR 3e-5/weight decay 1e-5/grad clip 1. **24 x 4.096 = 98.304 updates
por braço**, 294.912 totais; patience 25 evita parada antes do orçamento fixo.

Seleção em duas janelas/arquivo, incluindo inicialização como candidata. Avaliação
dos selecionados e source robusto congelado nas **5.920 janelas completas do fold 0**.
Medir erro/correlação por grupo, pior família, cauda por canal, amplitude/fase,
fração de escapes e bitrate pooled real de índices+side.

Não comparar diretamente números absolutos com scores legados do fold 0: máscara
e escala de referência mudaram nos casos extremos. Comparar **contrastes dentro
desta confirmação**, depois a direção da curva entre folds. Warm-starts e grupos
diferem; não é CV aninhada, teste intocado ou auditoria externa.

## 4. Testes e memória

**155 testes passaram**, incluindo loader/máscara/referência robusta coerentes,
roundtrip de treino/resume, normalização de continuação e rejeição de referência
mista. Guards anteriores de source/fold continuam em uso.

Smoke com **uma das 40 janelas afetadas reais**, 512x1.000: três updates descartados,
gradientes/loss finitos e restauração exata dos escapes antes de cada update.
Picos ~**4,00 / 4,31 / 5,37 GB**; índices reais 8.064/16.000/32.000 conferidos.
Nenhum peso de smoke salvo/reutilizado. Evidência:
`results/experiments/high_rate_fold0_robust_smoke_20261017/`.

## 5. Execução e recuperação

Runner: `finetuning/run_high_rate_fold_confirmation.py`.
Saída: `results/experiments/high_rate_fold0_robust_20261017/`.
Log: `results/experiments/high_rate_fold0_robust_20261017.pipeline.log`.
Serviço: **`febus-high-rate-fold0-robust-20261017`**. **Resultados pendentes**.
Estimativa inicial **9–14 h**; atualizar com épocas medidas.

Recuperação em 2026-10-17: o processo permanecia vivo, mas sem avanço de log por
~57 min na época 6, após evento de saída de suspensão registrado no kernel.
Um processo CUDA novo executou corretamente; o serviço travado foi encerrado e
retomado do checkpoint completo epoch 5, sem alterações de source/protocolo/código.
Não atribuir atividade do PID a progresso efetivo. Serviço da retomada:
`febus-high-rate-fold0-resume-20261017`, com inibição temporária de idle enquanto
ativo. Isso não altera configurações permanentes nem garante recuperação de
CUDA após suspensão manual/reboot.

Recuperação em 2026-10-18: após reinício do servidor, nenhum processo do estudo
estava ativo. Braços 1x e 2x concluíram epoch 24; 4x tinha checkpoint até epoch 16.
Protocolo/hash e operação CUDA verificados antes da retomada, sem mudanças de código
ou repetição dos braços completos. Serviço: `febus-high-rate-fold0-resume-20261018`,
com a mesma inibição temporária de idle; avaliação completa permanece na sequência.

```bash
systemctl --user status febus-high-rate-fold0-robust-20261017
# Serviço da retomada:
systemctl --user status febus-high-rate-fold0-resume-20261017
# Retomada mais recente:
systemctl --user status febus-high-rate-fold0-resume-20261018

# Somente sem cópia ativa:
.venv/bin/python finetuning/run_high_rate_fold_confirmation.py \
  --output-dir results/experiments/high_rate_fold0_robust_20261017 --resume
```

Source/código/manifest/smoke/auditoria fixados por hashes; locks, save atômico
e resume por época completa. Serviço independe do OpenCode interativo, mas não
garante sobrevivência a reboot/logout. Nenhum score de teste ou promoção automática.
