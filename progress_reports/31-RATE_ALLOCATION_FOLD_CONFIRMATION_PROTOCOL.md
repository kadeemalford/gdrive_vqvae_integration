# Confirmação de alocação em fold/configurações e seed adicionais

> Confirmação concluída. Resultados e diagnóstico de faixa dinâmica no
> [relatório 32](32-FOLD_CONFIRMATION_AND_RESIDUAL_DYNAMIC_RANGE.md).

## 1. Objetivo e ausência de vazamento de inicialização

Confirmar Temporal/Balanced grid/Spatial no **fold 0**, seed de treino/seleção
**2027**, em vez de somente continuar os candidatos do fold 1. Não usar seus pesos:
grupos de teste do fold 0 estiveram no treino do fold 1. A sobreposição é com
teste, não validação; correção de proveniência detalhada no relatório 39.

Fonte comum aos três braços:
`results/experiments/sampling_full_20261005/fold_0/window/best.pt`, epoch 9,
treinado no fold 0. Guard de lineage confere `finetune.samples_dir` e rejeita
source fold 1 ou sem proveniência. O source tem geometria padrão e histórico
diferente do anchor do fold 1: confirmar **contrastes dentro do fold**, não comparar
scores absolutos como se as duas inicializações fossem equivalentes.

Esta é replicação exploratória em corpus já investigado, **não CV aninhada,
auditoria externa intocada ou seed independente de todo o pré-treino**.

## 2. População e integridade

Fold 0: **9.720 janelas de treino / 5.920 de validação**. Configurações val:
doped 5m LSF; multicore C1 and C3; multimode Configuration 6;
single-mode Configurations 2, 6 e 10. Quatro famílias representadas.

O fold preparado precisa ser restaurado a partir dos HDF5/manifests preservados,
com verificação de CRC/tamanho dos registros NPZ pela ferramenta existente.
O runner restaura a pasta completa para cumprir seus guards de integridade,
inclusive arrays de teste arquivados; **nenhum array de teste é usado para updates,
seleção ou score de modelo**. Leitura de dados para restauração/integridade não
é avaliação de modelo. Fonte/checkpoints atuais/raw não são sobrescritos.

Configuration 6 contém os extremos finitos anteriormente auditados. Escape-v1
preserva-os bit-exatos e pontua a região ordinária separadamente; contabilizar
side bytes e fração de escapes. A proteção não torna os dados uma auditoria
independente nem autoriza clipping/exclusão física silenciosa.

## 3. Treino e avaliação

- Três geometrias anteriores: temporal-4x, balanced-2x, spatial-4x.
- Mesmo source/parametrização, objetivo temporal+VQ e sampling window nos braços.
- AdamW **novo**, LR 3e-5, weight decay 1e-5, grad clip 1, seed 2027.
- **24 x 4.096 = 98.304 updates por braço**, 294.912 totais, sem parada precoce
  antes do orçamento (patience 25).
- Normalização global escape-v1, refresh desligado, alvo original intacto.
- Seleção por RMSE macro com duas janelas/arquivo, incluindo epoch 0 como candidata.
- Avaliação dos selecionados em **todas as 5.920 janelas de validação do fold 0**.
- Rate accounting real, pior família/configuração, canais robustos, amplitude/fase.

Eval agora aceita `--fold`, default 1 preservado. Runner sempre especifica fold 0
na confirmação; protocol/checkpoint hashes fixam a população e as fontes.

## 4. Critérios de leitura

1. Direção e magnitude do contraste das grades frente ao Temporal, por configuração.
2. Verificar cauda/canais não fracos e pior família, não só correlação macro.
3. Comparar taxa igual Balanced versus Temporal; diferença Spatial ~0,8% de índices
   para T=1.000 explícita, não ganho gratuito.
4. Resultados contraditórios: tratar efeito como dependente de domínio/inicialização,
   sem promover por média conjunta de folds de composição diferente.
5. Confirmação consistente: considerar arquitetura multiescala/rate-distortion e
   auditoria em novas aquisições. Nenhuma promoção sem requisitos físicos.

## 5. Execução e recuperação

Runner: `finetuning/run_rate_allocation_confirmation.py`.
Saída: `results/experiments/rate_allocation_fold0_seed2027/`.
Log: `results/experiments/rate_allocation_fold0_seed2027.pipeline.log`.
Serviço de usuário **`febus-allocation-confirmation-fold0`**.
Sequência: restauração verificada → três treinos → avaliação completa → comparação.
**Resultados pendentes nesta atualização**.

Retomada em 2026-10-14: dataset restaurado, Temporal e Balanced grid completos
(24 épocas cada), Spatial interrompido após epoch 6. GPU/source/protocolo/código
verificados; continuação lançada no serviço
`febus-allocation-confirmation-resume-20261014`, sem repetir braços completos.
Pelo Spatial já medido (~6,08 min/epoch), restam ~1h49 de treino mais avaliação
completa; estimativa da retomada **2–2h20**.

Estimativa preliminar **9–13 h**, incluindo restauração; revisar após épocas medidas.
~35 GiB livres antes do lançamento, com guard de 18 GiB para restauração/treino;
não há deleção automática. Fontes/manifest/código fixados por hashes e lock.

```bash
systemctl --user status febus-allocation-confirmation-fold0
# Serviço da retomada:
systemctl --user status febus-allocation-confirmation-resume-20261014

# Somente sem cópia ativa:
.venv/bin/python finetuning/run_rate_allocation_confirmation.py \
  --output-dir results/experiments/rate_allocation_fold0_seed2027 --resume
```

**144 testes passaram**, incluindo rejeição de source treinado em fold errado ou
sem lineage. Topologia/GPU máxima das três geometrias já foram verificadas nos
smokes anteriores; seus updates foram descartados. Serviço independe da sessão
interativa, não garante sobrevivência a logout/reboot; resume repete fração do
epoch interrompido, sem ampliar o orçamento.
