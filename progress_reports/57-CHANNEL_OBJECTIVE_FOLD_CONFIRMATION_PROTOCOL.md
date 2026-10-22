# Confirmação do objetivo por canal em outras configurações e seed

> Estudo e avaliação concluídos. Resultados e decisão no
> [relatório 58](58-CHANNEL_OBJECTIVE_FOLD_CONFIRMATION_RESULTS.md).

## Hipótese fixa e natureza da confirmação

Verificar se o compromisso de fidelidade/admissão observado no fold 0 se mantém
com outras configurações e seed. **Peso 0,05, cap 4, fórmula e guarda inalterados**.
O código da loss deve ter exatamente o hash registrado no estudo de origem.
Não ampliar busca de pesos/normalizações se o resultado for negativo.

Fold 1 é desenvolvimento reutilizado em estudos anteriores, não aquisição externa
ou teste intocado. Fold, source e seed mudam entre os estudos; não atribuir
eventual diferença a um único desses fatores. O contraste causal exploratório
relevante é **auxiliar versus controle dentro do mesmo fold**.

## Source e dados separados

- Source exclusivo do fold 1:
  `neural_rate_sweep_20261014/rate_4x/best_macro.pt`, epoch 24.
- **Nenhum checkpoint/peso do fold 0 é transferido.** Só a regra de treino/guarda
  já fixada é reutilizada.
- Loader/score **escape-v1**, iguais aos do source e controles históricos do fold 1.
  Não reencodar nem misturar em um ranking com denominadores robustos do fold 0.
- **12.140 janelas de treino**, **5.820 de validação**, dados existentes, sem teste.
- Paridade da fase 0 conferida contra `high_rate_continuation_20261015/
  full_validation/initial_rate4.groups.csv`, mesmo source/hash e coordenadas.

Comparar direção e magnitude dos contrastes dentro de cada fold, não fazer
ranking absoluto ou pooling ingênuo de scores/razões entre folds diferentes.

## Braços e orçamento pré-fixados

| Braço | Escopo | Auxiliar por canal |
|---|---|---:|
| Full controle | Rede inteira | 0 |
| Channel-tail | Rede inteira | **0,05**, cap **4** |

Mesmos pesos iniciais, **AdamW novo por braço**, LR 3e-6, weight decay 1e-5,
seed **2030**, sampling window e objetivo histórico no controle.
**8 × 4.096 = 32.768 updates por braço**, **65.536 no total**. Mesmo número de
parâmetros otimizados, 27.958.081; custo extra do auxiliar é explícito, não zero.
K=2.048, geometria spatial4-temporal4 e payload nominal por janela inalterados.

São épocas 0–8 de novo piloto, não extensão com optimizer histórico. Dentro de
cada braço, resume preserva optimizer/RNG. Não repetir estudos completos do fold 0
ou a continuação histórica do fold 1 como se fossem controles deste orçamento.

## Seleção, retenção e medição

Validação completa **a cada época**, incluindo inicialização. Reter todos os
16 snapshots weights-only, last com optimizer/RNG e controle macro/loss.
Mesma guarda do relatório 53: RMSE, correlação e p90 por configuração contra
fase 0 fixa; menor RMSE macro dentre elegíveis; baseline se nenhum passar.
Epsilon numérico 1e-9 não é tolerância física; não relaxá-lo após o resultado.

Reavaliar selecionados macro/guardados por janela. Reportar resultados contra
baseline **e** controle, perdas locais, margens e fração de épocas elegíveis.
Melhora macro não garante melhora por janela/canal. Um resultado positivo só
avança evidência exploratória; não autoriza promoção final.

## Código e verificação

Modo opt-in novo: `finetuning/run_guarded_decoder_pilot.py --fold1-channel-confirmation`.
Flag incompatível com `--channel-tail-arm-only`. Normalização, dataset, source,
seed e pesos auxiliares são declarados por protocolo. Validação de source rejeita
fold/representação/epoch/geometria diferentes. Trainer e fórmula não foram
retunados para esta confirmação.

**187 testes passaram**, incluindo rejeição de pesos de outro fold e representação.
Smoke GPU pareado, source fold 1 e seed 2030 em janela de treino 512x1.000:
mesma loss base inicial nos dois braços, gradientes finitos e dois updates
descartados. Pico ~**5,37 GB**; nenhum peso de smoke salvo/reutilizado.
Evidência: `results/experiments/channel_fold1_smoke_20261019/`.

Saída: `results/experiments/channel_tail_fold1_20261019/`.
Serviço: `febus-channel-tail-fold1-20261019`.
Log: `results/experiments/channel_tail_fold1_20261019.pipeline.log`.
Estimativa **3–6 horas**, incluindo validação por época e final. Resultados pendentes.
Espaço livre conferido (~15 GiB antes do lançamento); não houve exclusões.

Retomada em 2026-10-20 após reinício do servidor: nenhum processo do estudo ativo.
Controle full com last.pt de época 7, optimizer/RNG e sete snapshots preservados;
channel-tail ainda não iniciado. Protocolo/hash e operação CUDA verificados antes
da retomada, sem repetir épocas completas ou estudos anteriores. Serviço atual:
`febus-channel-tail-fold1-resume-20261020`.

```bash
systemctl --user status febus-channel-tail-fold1-20261019
# Serviço da retomada:
systemctl --user status febus-channel-tail-fold1-resume-20261020
# Apenas sem cópia ativa:
.venv/bin/python finetuning/run_guarded_decoder_pilot.py \
  --output-dir results/experiments/channel_tail_fold1_20261019 \
  --fold1-channel-confirmation --resume
```

Hashes de origem/seleção/fórmula/source/código/dataset/referência/smoke, lock e
snapshots preservados. Treino retoma somente de last.pt. Inibição de idle é
temporária; suspensão manual/reboot pode interromper GPU/serviço. Aviso no chat
não é garantido após reinício do OpenCode. Sem downloads, teste ou promoção automática.
