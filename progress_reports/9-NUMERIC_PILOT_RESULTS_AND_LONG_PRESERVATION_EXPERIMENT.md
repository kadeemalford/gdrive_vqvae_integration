# Piloto numérico concluído e adaptação longa com preservação

## Estado da execução

Piloto `results/experiments/numeric_adaptation_pilot_20261007/` concluído.
Próxima comparação **iniciada em background**, com autorização do usuário:
`results/experiments/preservation_long_20261007/`.
Resultados do treino longo ainda pendentes; nenhuma melhoria presumida ou modelo promovido.

## 1. Resultado do piloto

Mesma inicialização uniform fold 2, fresh AdamW, LR 1e-6, seed 2026,
3×2.048 updates por braço, mesma sequência de janelas. Comparação legacy/escape-v1.
Seleção macro inclui inicialização; validação determinística de 72 janelas de 36 arquivos.
**Essa validação contém somente multicore e single-mode**, não as quatro famílias.
Portanto não pode estabelecer ganho generalista, mesmo se a média melhorar.

| Braço | RMSE macro inicial | Melhor macro (epoch 1) | Final (epoch 3) | Pior família, inicial → selecionado |
|---|---:|---:|---:|---:|
| Legacy | 0,868262 | 0,868181 | 0,869049 | 0,918842 → 0,918708 |
| Escape-v1 | 0,868262 | 0,868126 | 0,868872 | 0,918842 → 0,918700 |

Ganhos iniciais muito pequenos (~0,000081/~0,000136 absoluto no RMSE macro).
Não há regressão de RMSE das duas famílias no selecionado, mas o final piora
single-mode. A seleção por val_loss temporal+VQ mantém a **inicialização** nos dois
braços, enquanto a seleção macro escolhe epoch 1. Não promover essa pequena
diferença como avanço de fidelidade: validação limitada, uma seed/fold e seleção
feita sobre os mesmos exemplos usados para medir o ganho.

### Exposição aos recortes afetados

Replay determinístico do schedule, validado contra o estado Python RNG do `last.pt`
dos dois braços, confirmou:

- 6.144 apresentações por braço; 5.308 coordenadas distintas.
- Apenas **18 das 40 janelas afetadas** foram apresentadas, uma vez cada.
- Apresentações afetadas por epoch: **3, 8, 7**, ou ~0,29% dos updates totais.

O piloto não oferece teste forte do impacto de treinamento da correção numérica.
A proteção permanece necessária pela perda de informação demonstrada no relatório
de integridade; sua utilidade não depende de ganhar a média de validação deste piloto.
Não aumentar artificialmente a exposição aos extremos sem mudar/documentar o protocolo.

Tempo somado das seis épocas: ~15 minutos; exclui avaliações iniciais e parte do
salvamento. Artefatos adicionais: `REPORT.md`, `comparison.json`, `exposure.json`.

## 2. Decisão seguinte

Não prolongar cegamente o fold 2, cuja validação cobre duas famílias. Investigar a
hipótese de **preservação da representação durante adaptação**, usando corpus de
treino amplo em vez de repetir o pool minúsculo dos testes de capacidade.

Comparação longa, fold **1**:

| Item | Protocolo |
|---|---|
| Inicialização | mesmo uniform fold 1 best.pt do experimento de sampling completo |
| Braços | full: encoder+quantizador+decoder; decoder: somente decoder |
| Dados por epoch | **12.140 janelas de treino**, todas uma vez, mesma ordem por braço |
| Validação | **5.820 janelas**, todas; quatro famílias, seis configurações |
| Normalização | escape-v1 em ambos; sem perda espectral |
| Otimizador | AdamW novo, LR 1e-6, weight decay 1e-5, seed 2026 |
| Orçamento máximo | **8 epochs por braço**, até 194.240 updates totais |
| Parada | RMSE macro de validação, patience 3, min_delta 1e-4 |
| Seletores | inicialização candidata; best temporal+VQ e best_macro separados |
| Teste | **não consultado** |

No fold 1, os arquivos com janelas preparadas extremas pertencem ao teste, não ao
treino/validação. A comparação **não é outra ablação de escape**: testa preservação
versus atualização da representação sob o mesmo preprocessing.
O encoder/codebook pode já estar inadequado para FEBUS; congelá-lo pode limitar
melhorias. Decoder-only é hipótese a comparar, não solução presumida nem MoS.

Early stopping agora pode usar macro RMSE sem a contribuição VQ: no decoder-only,
VQ permanece constante e não recebe gradiente. Comparar somente perda total entre
braços poderia misturar progresso de quantização com fidelidade. Os defaults antigos
do treinador continuam full/val-loss, sem alterar protocolos históricos.

Encoder e quantizador ficam em eval no braço decoder; seus parâmetros não entram
no optimizer. A arquitetura fonte usa GroupNorm e quantizador sem EMA: não há
atualização oculta dessas partes por estatísticas de BatchNorm ou EMA.
Testes sintéticos verificam congelamento bit-exato e atualização apenas do decoder.

## 3. Operação e recursos

Runner: `finetuning/run_preservation_experiment.py`; dry-run aprovado antes de iniciar.
Log: `results/experiments/preservation_long_20261007.log`.

Estimativa preliminar: **algumas horas**, possivelmente ~3–6 h para os dois braços,
variando com validação completa, I/O, tamanhos de janela e parada antecipada.
Não é ETA medida do experimento longo; atualizar com épocas completas.

Havia aproximadamente 12 GB livres. Runner exige pelo menos 5 GiB para checkpoints/
headroom e não remove arquivos anteriores. Há lock do runner, checkpoints atômicos
por arquivo e resume por epoch completo. Background depende do serviço OpenCode;
reinício pode interromper e exigir retomada:

```bash
.venv/bin/python finetuning/run_preservation_experiment.py \
  --output-dir results/experiments/preservation_long_20261007 --resume
```

Após concluir:

```bash
.venv/bin/python finetuning/summarize_numeric_adaptation.py \
  --experiment-dir results/experiments/preservation_long_20261007
```

O comparador também aceita os braços full/decoder; mostra inicial/selecionado/final,
macro, pior família e regressões por família. Seleção e análise usam a mesma
validação exploratória, não holdout independente. Não comparar diretamente os
RMSE do fold 2 com fold 1 como se fossem a mesma distribuição.

## 4. Critério de análise posterior

Verificar cada família/configuração, amplitude/fase/espectro e erros ordinários,
não somente média macro. Se o decoder reduz regressões mas não reconstrói bem,
comparar fine-tuning parcial mais amplo/replay ou interferência entre famílias;
se ambos regredirem, não escolher o último por ter treinado mais. MoS, novos dados
FEBUS-like e arquitetura/taxa permanecem alternativas condicionais.

O objetivo continua fidelidade em todas as famílias. Este experimento não é modelo
final treinado em todos os dados; aquisições/configurações novas ainda são necessárias
para auditoria externa. Não foram iniciados downloads, excluídos dados ou promovidos
checkpoints do piloto.

**66 testes passaram** antes de iniciar o treino longo, incluindo novos controles de
congelamento, política de resume e early stopping macro. Dados originais preservados.
