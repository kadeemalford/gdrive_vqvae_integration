# Treino do zero: continuação controlada de orçamento e quantização

> Continuação concluída. Resultados e diagnóstico posterior no
> [relatório 24](24-SCRATCH_CONTINUATION_AND_QUANTIZATION_DIAGNOSIS.md).

## 1. Hipótese

Os braços scratch do relatório 22 selecionaram a última época e ainda melhoravam.
A utilização do codebook é restrita: perplexidade ~119/149 e 177/223 códigos
usados na última época, contra ~512 e 1.227 códigos no transferido. Primeiro testar
se **mais exposição, sem intervenção no quantizador**, melhora esses indicadores
e a reconstrução. Utilizar códigos não é objetivo final nem prova de fidelidade.

## 2. Protocolo iniciado

- Braços: `scratch_4x` e `scratch_temporal`.
- Fonte: **last.pt epoch 16** da triagem, com AdamW e RNG salvos, não apenas pesos.
- Retomar epochs **17–32**, 4.096 updates/epoch: **65.536 adicionais por braço**,
  total de 131.072 updates desde o início de cada treino scratch.
- Mesmo corpus, arquitetura, geometria 4x, codebook 2.048, objetivo temporal+VQ,
  sampling window, LR **1e-4**, weight decay 1e-5 e grad clip 1.
- Nenhuma reinicialização/reativação de código, mudança de loss ou dataset.
- Patience 33 evita parada precoce dentro do orçamento total de 32 epochs.
- Seleção nas mesmas 102 janelas, mantendo os melhores checkpoints anteriores.
- Avaliação dos selecionados em todas as **5.820 janelas de validação**, juntamente
  com **transferido epoch 13 congelado**, sem updates nessa referência.
- Taxa/parametrização permanecem iguais; compute histórico do transferido é diferente.
- Nenhum acesso ao teste, promoção automática ou seleção por holdout independente.

Fonte preservada: `results/experiments/architecture_initialization_pilot_20261010/`.
Saída: `results/experiments/scratch_budget_continuation_20261011/`.
Runner: `finetuning/run_scratch_budget_continuation.py`.

## 3. Integridade e análise

Checkpoint last/best/best_macro e históricos são copiados atomicamente para uma
saída nova, conferidos por hashes. Não copiar summary de término da triagem como
se a continuação já tivesse terminado. Marcadores de fork distinguem cópia parcial
de checkpoint modificado pelo treinamento, impedindo sobrescrita em recuperação.

Resume restaura o optimizer/RNG do último epoch completo; não repete a inicialização
do codebook nem recalibra na validação. Manifest, código e arquivos fonte são
fixados no protocolo. Checkpoints e métricas históricos não são apagados/alterados.

A comparação da continuação usa **epoch 16 como estado inicial**, não o epoch 0
aleatório. Histórico completo permanece disponível; initial/selecionado/final
permitem detectar ganho e regressão sem confundir fases do experimento.

Analisar:

1. Curvas últimas épocas, custo adicional e seleção anterior versus posterior.
2. Perda temporal/VQ separadas, perplexidade por janela e utilização por época.
3. RMSE/correlação por configuração, pior família, amplitude/fase e cauda por canal.
4. Transferido congelado como referência de fidelidade, não controle de compute igual.

Se houver plateau com quantização ainda restrita, a intervenção no codebook deve
ser **uma ablação separada com controle pareado**, sem mudar LR/arquitetura/dataset
simultaneamente. Não impor uma meta de uso de códigos no lugar de fidelidade.
Se scratch continuar melhorando, definir novo orçamento antes de classificá-lo
como inviável; se regredir, preservar o selecionado da triagem.

## 4. Execução e recuperação

Execução lançada sequencialmente na GPU em serviço de usuário
`febus-scratch-budget-20261011`. **Resultados pendentes nesta atualização**.
Pelos tempos medidos da triagem (~6–7 min/epoch), estimativa **3h40–4h**, incluindo
a avaliação final. É estimativa sujeita a I/O/carga, não garantia de horário.

Log: `results/experiments/scratch_budget_continuation_20261011.pipeline.log`;
logs detalhados/históricos por braço dentro da saída.

```bash
systemctl --user status febus-scratch-budget-20261011

# Somente sem cópia ativa:
.venv/bin/python finetuning/run_scratch_budget_continuation.py \
  --output-dir results/experiments/scratch_budget_continuation_20261011 --resume
```

O serviço é independente da sessão interativa, mas não garante sobrevivência a
reboot/logout. Resume repete a fração interrompida do epoch, sem ampliar o orçamento.
**129 testes passaram**, incluindo cópia verificada sem sobrescrita de fontes e
seleção correta do baseline epoch 16 para a análise da continuação.
