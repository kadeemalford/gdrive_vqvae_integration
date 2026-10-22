# Piloto de objetivo por canal com pesos limitados e controle já completo

> Estudo concluído. Resultados e decisão no
> [relatório 56](56-BOUNDED_CHANNEL_OBJECTIVE_RESULTS.md).

## Hipótese única

Os selecionados full/decoder do relatório 54 melhoram RMSE nas seis configurações,
mas falham na preservação de correlação/caudas. Testar se **uma pequena ênfase
adicional no erro por canal** altera esse compromisso. É um surrogate de treino,
não otimização direta do p90 nem garantia de melhoria por grupo.

Não alterar normalização, geometria, arquitetura, quantização, bitrate, sampling
ou guarda. Não ampliar busca de pesos após observar a validação deste ensaio.

## Definição fixada antes do braço novo

Para cada janela, na região ordinária normalizada, calcular a variância temporal
por canal `v_c` e a variância global `v_w`. Pesos não treináveis:

`w_c = clip(v_w / max(v_c, v_w/4), 1/4, 4)`.

Se `v_w <= 1e-12`, usar pesos unitários. Canais sem células ordinárias são
excluídos; escapes não influenciam estatística, valor ou gradiente desse termo.

`L_channel = mean_c(w_c * mean_ordinary_time((prediction-target)^2))`.

Não dividir novamente pela soma dos pesos: cada peso fica explicitamente em
**[0,25; 4]**, e o termo é limitado por 4x a média de MSE por canal. Não há inverso
de std/variância sem limite nem mudança da representação de entrada/alvo.

Objetivo novo: `0,5*MSE + 0,5*L1 + VQ + 0,05*L_channel`.
Peso auxiliar **0,05** e cap **4** fixos; sem sweep, rótulos de configuração ou
roteamento. Estatísticas usam apenas o alvo da janela de **treino**, sem estimar
parâmetros na validação. Não é a normalização por canal dos estudos anteriores.

## Controle e orçamento

- Source: mesmo 4x epoch 24 dos relatórios 53–54.
- Novo braço **full**, AdamW novo, LR **3e-6**, weight decay 1e-5, seed **2029**.
- **8 × 4.096 = 32.768 updates**, mesma política window e corpus do fold 0.
- Controle: `guarded_decoder_pilot_20261019/full`, **já completo**. Não treinar
  outra cópia. Fixar hashes do protocolo, histórico e summary do controle.
- Mesmo número de parâmetros otimizados (27.958.081), mesma fonte, taxa e
  exposição nominal; o termo adicional tem custo computacional pequeno, não zero.
- Ambos começam com optimizer novo; não mudar objetivo dentro de um resume.
- Mesmo conjunto completo de 5.920 janelas por época, snapshots e regra de guarda
  prospectiva do relatório 53 contra fase 0. Baseline sempre é opção elegível.
- Selecionados macro e guardado serão avaliados também por janela, sem chamar
  essa reavaliação das mesmas janelas de teste independente.

O controle histórico não tinha o termo. A implementação nova preserva o default
**peso zero**, pulando inteiramente o auxiliar; teste verifica updates idênticos
com default/zero explícito. O novo braço tem nova origem de código registrada,
sem reescrever o protocolo/checkpoints do controle concluído.

## Diagnósticos e decisão

`train_channel_mse` registra o termo bruto; `train_loss` inclui 0,05 vezes o termo.
`train_temporal_loss`/VQ mantêm o significado histórico. **Loss/objetivo de validação
permanecem os históricos**, excluindo o auxiliar, para não mudar o controle de
medição; esse fato é explícito no summary. Seleção principal usa métricas físicas,
não compara train_loss entre objetivos diferentes como evidência de fidelidade.

Comparar novo braço versus controle full e baseline: RMSE/correlação/p90 por
configuração, amplitude/fase e caudas. Não relaxar a guarda se o auxiliar falhar.
Não promover apenas por melhoria macro ou train_loss. Resultado negativo deve
encerrar este peso fixo, não disparar busca adaptativa na mesma validação.
O corpus e a motivação são de desenvolvimento reutilizado, não confirmação
independente. Tolerâncias da aplicação continuam indefinidas.

## Código, testes e execução

Helper isolado: `finetuning/channel_objective.py`.
Trainer ganhou **`--channel-mse-weight` opt-in**, default 0, com peso/cap salvos
no checkpoint e mudança de objetivo rejeitada no resume. Sem mudança de defaults
ou normalização, nenhum modelo extra/side por canal ou janela no codec.

Runner: `finetuning/run_guarded_decoder_pilot.py --channel-tail-arm-only`.
Exige o controle completo e sua política compatível antes de lançar o braço.
Saída: `results/experiments/channel_tail_pilot_20261019/`.
Serviço: `febus-channel-tail-pilot-20261019`.
Log: `results/experiments/channel_tail_pilot_20261019.pipeline.log`.
Estimativa **2–4 horas**, incluindo validação completa por época e dos selecionados.
Resultados pendentes. Não repetir braços full/decoder concluídos.

**186 testes passaram**, incluindo máscara/gradiente em escapes, canais vazios,
variância zero, limite de peso, default zero, treino/resume e rejeição de mudança
de objetivo. Smoke GPU 512x1.000: loss/gradientes finitos e update descartado,
pico **5,37 GB**. Nenhum peso de smoke salvo/reutilizado.
Evidência: `results/experiments/channel_tail_smoke_20261019/`.

```bash
systemctl --user status febus-channel-tail-pilot-20261019
# Apenas sem processo ativo:
.venv/bin/python finetuning/run_guarded_decoder_pilot.py \
  --output-dir results/experiments/channel_tail_pilot_20261019 \
  --channel-tail-arm-only --resume
```

Hashes/lock, snapshots e retomada do protocolo anterior são preservados.
Inibição temporária de idle não garante sobrevivência a suspensão manual/reboot.
Aviso no chat não é garantido após reinício do OpenCode. Nenhuma promoção automática.
