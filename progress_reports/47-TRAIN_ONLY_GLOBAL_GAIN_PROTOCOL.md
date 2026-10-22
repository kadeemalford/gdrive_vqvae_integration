# Calibração global de amplitude com rede congelada

> Estudo concluído; resultados e decisão no
> [relatório 48](48-TRAIN_ONLY_GAIN_RESULTS.md).

## Hipótese isolada

Separar um viés simples de escala da limitação de reconstrução da forma de onda.
Os modelos ainda subestimam RMS em várias configurações, mas **RMS menor que 1
não implica que amplificar a saída reduza MSE**: parte da atenuação pode ser a
estimativa ótima de um sinal com conteúdo não reconstruído.

Testar **um ganho global positivo por modelo**, ajustado somente no treino.
Não impor ganho >1, casamento de RMS, bias, ganho por canal/janela/família ou
roteamento por metadados. Um ganho positivo não deve corrigir correlação/fase
por si só; seu objetivo limitado é medir erro de escala/amplitude.

## Fontes e dados

- 2x: `frozen_rate_stability_20261018/midpoint_2x.pt`, candidato conservador do
  relatório 45, não promovido como final.
- 4x: `high_rate_fold0_robust_20261017/rate_4x/best_macro.pt`, epoch 24,
  preservado frente às regressões da continuação.
- **9.720 janelas completas de treino do fold 0** para ajustar cada escalar.
- **5.920 janelas de validação do fold 0** para comparar identidade/calibração.
- Normalização e referência robustas em ambos os lados. Escapes são excluídos
  do ajuste de ganho e restaurados bit-exatamente na avaliação física.
- Não abrir dados de teste, nem ajustar o ganho na validação.

A hipótese foi motivada por validação reutilizada: mesmo com ajuste train-only,
o estudo continua exploratório, não uma prova de generalização independente.

## Ajuste e implementação

Para cada janela j, região ordinária Mj e saída congelada yj, acumular em float64:

`xy = Σj mean_Mj(xj*yj)` e `yy = Σj mean_Mj(yj*yj)`.

Ganho `g = float32(xy/yy)` minimiza a **média de MSE normalizado por janela**.
Cada janela tem o mesmo peso, não cada célula ou família. Recusar denominador
degenerado, valores não finitos ou ganho não positivo. Registrar MSE train antes
e depois, sem usar o resultado de validação para reajustar g.

Nenhum SGD/backprop ou optimizer; encoder, codebook e decoder ficam congelados
durante a coleta. Depois incorporar g apenas no peso/bias da **última Conv2d
linear** do decoder. Isso implementa a multiplicação da saída, salvo arredondamento,
sem novos parâmetros da rede. Não alterar encoder, índices ou embeddings VQ.
Checkpoint derivado identifica source/gain/fit-split, com epoch 0 de construção
e sem optimizer/RNG para continuação de treino.

Fit em ordem fixa com cursor/momentos salvos atomicamente a cada 50 janelas.
Retomada pode repetir as últimas até 49 inferências não persistidas, sem alterar
o objetivo ou usar validação para preenchê-las. Fontes nunca são sobrescritas.

## Taxa e critérios

Cada ganho é constante compartilhada pelo modelo, não side por janela. Embora
incorporado nos pesos distribuídos, contabilizar conservadoramente **32 bits
globais adicionais** por modelo em `charged_rate.json`. Não somar 32 bits em cada
janela ou alegar tamanho de codec implantado: container/modelo/entropia continuam
fora da taxa nominal de índices+side.

Publicar os quatro candidatos completos: identidade/calibrado por taxa, RMSE,
correlação, p90 por canal por configuração, RMS e fase, pior configuração/família,
custos pooled. Verificar que correlação/fase não tenham mudanças materiais
inesperadas e que índices/escapes continuem coerentes.

Ganho macro sem preservação dos grupos/caudas é trade-off, não promoção. Se o
ganho global não resolver os limites, não ampliar automaticamente para ganhos
por configuração/janela sobre o mesmo corpus. Resultado negativo também separa
limite de escala simples de perda de informação/forma de onda.

## Testes e execução

**169 testes passaram**, incluindo máscara ordinária, pesos iguais por janela,
ganhos <1, rejeição de ajuste degenerado e equivalência de folding de Conv2d.
Smoke GPU com ganho **fixo 1,125, não ajustado na validação** nos dois modelos:
janela de treino, multimode com escapes e single-mode Configuration 2.
Índices exatamente iguais, folding numericamente equivalente à escala de saída,
saídas finitas e restauração física com escapes bit-exatos verificados.
Evidência: `results/experiments/frozen_gain_smoke_20261018/`.

Runner: `finetuning/run_frozen_gain_calibration.py`.
Helper isolado: `finetuning/frozen_gain.py`. Defaults do trainer não foram mudados.
Saída: `results/experiments/frozen_gain_calibration_20261018/`.
Serviço: `febus-frozen-gain-calibration-20261018`.
Log: `results/experiments/frozen_gain_calibration_20261018.pipeline.log`.
Estimativa inicial **45–90 min**; resultados pendentes. Zero updates da rede,
dois escalares estimados no total e quatro avaliações físicas completas.

Retomada em 2026-10-19: após reinício, os treinos anteriores permaneciam completos,
mas não havia processo ativo da calibração. Cursor persistido do 2x: 1.950/9.720
janelas; 4x ainda não iniciado. Hashes/protocolo e operação CUDA verificados antes
de retomar, sem repetir estudos concluídos ou alterar o código. Serviço atual:
`febus-frozen-gain-resume-20261019`.

```bash
systemctl --user status febus-frozen-gain-calibration-20261018
# Serviço da retomada:
systemctl --user status febus-frozen-gain-resume-20261019
# Apenas sem processo ativo:
.venv/bin/python finetuning/run_frozen_gain_calibration.py \
  --output-dir results/experiments/frozen_gain_calibration_20261018 --resume
```

Hashes de fontes/código/manifest/smoke, lock e validação de cursor impedem misturar
protocolos. Inibição temporária de idle não altera configuração permanente nem
garante sobrevivência à suspensão manual/reboot. Aviso no chat não é garantido
após reinício do OpenCode. Nenhuma promoção automática ou novo treino longo.
