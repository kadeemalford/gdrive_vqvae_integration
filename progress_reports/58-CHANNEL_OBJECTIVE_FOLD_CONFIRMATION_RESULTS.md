# Confirmação no fold 1: cauda média melhora, preservação generalista não confirma

## Estudo e avaliação concluídos

Fonte: `results/experiments/channel_tail_fold1_20261019/`, `COMPLETE`,
`full_validation/COMPLETE`, históricos, `selection.json` e 16 snapshots preservados.
Dois braços completos, oito épocas e 32.768 updates cada, seed 2030, source
exclusivo do fold 1. Peso/cap **0,05/4 inalterados**, sem transferência de pesos
do fold 0. Cada época usou as 5.820 janelas completas de validação.

| Candidato | RMSE macro | Correlação | p90 macro por canal | RMS macro | Guarda |
|---|---:|---:|---:|---:|---|
| Source epoch 24 / fase 0 | 0,490010 | 0,865562 | 0,844228 | 0,917563 | Referência |
| Full controle, fase 8 | **0,487224** | **0,866805** | 0,829564 | 0,924614 | Rejeitado |
| Auxiliar 0,05, fase 8 | 0,487251 | 0,866750 | **0,826060** | 0,924428 | Rejeitado |
| Guardado, ambos os braços | 0,490010 | 0,865562 | 0,844228 | 0,917563 | **Baseline** |

Macro e final coincidem na fase 8 em ambos. **Nenhum dos 16 snapshots foi elegível**.
Taxa pooled permanece **42,133x**, sem novo side/parâmetro de codec.
Fase FFT: baseline 0,936199, controle 0,937208, auxiliar 0,937145.

Auxiliar versus controle reduz p90 macro em ~0,003505 (~0,42%), com RMSE quase
igual (+0,0000266). Contra baseline, RMSE melhora ~0,56% e p90 ~2,15%, mas esses
agregados não atendem a guarda de preservação por configuração.

## Vetos persistentes

| Grupo / critério | Baseline | Auxiliar fase 8 | Delta de perda |
|---|---:|---:|---:|
| Multimode Configuration 7 / RMSE | 0,658187 | 0,659777 | +0,001590 |
| Multimode Configuration 7 / correlação | 0,772880 | 0,771776 | -0,001105 |
| Single-mode Configuration 7 / RMSE | 0,389536 | 0,391547 | +0,002011 |
| Single-mode Configuration 7 / correlação | 0,918366 | 0,917665 | -0,000700 |
| Single-mode Configuration 11 / p90 por canal | 1,216647 | 1,257789 | +0,041142 |

Nos dois braços, correlação multimode e single-mode Configuration 7, além de p90
de Configuration 11, pioraram nas **oito épocas**. Não é uma perda oculta pela
amostra curta: a avaliação era completa, e não faltam pesos intermediários.

Contra controle, auxiliar melhora p90 **nas seis configurações**, como no fold 0.
Mas RMSE versus controle melhora em três e piora em três grupos. O termo parece
alterar consistentemente o compromisso de cauda nesse corpus, não garantir
preservação de todos os critérios contra o source.

## Pareamento por janela/arquivo

Auditorias sem novas inferências, `finetuning/audit_physical_pair.py`:

- `results/experiments/channel_fold1_vs_baseline_20261020/`.
- `results/experiments/channel_fold1_vs_control_20261020/`.

Cada uma contém **5.820 janelas pareadas**, scores e bootstrap descritivo de
arquivos, com protocolos/referências compatíveis verificados. Contra baseline:

- Multimode Configuration 7: **72,0% de 560 janelas** pioram RMSE.
- Single-mode Configuration 7: **44,0% de 500 janelas** pioram RMSE.
- Single-mode Configuration 11: **64,3% de 440 janelas** pioram p90 por canal.

Há seis arquivos por grupo; janelas/arquivos podem ser correlacionados ou cópias.
Não transformar frequências ou intervalos de arquivos em prova estatística de
generalização independente. Não inventar tolerâncias físicas para ignorar vetos.

## Conclusão e decisão

A admissão positiva do fold 0 **não foi confirmada como preservação generalista**
no fold 1. Fold/source/seed diferem entre estudos, então não atribuir o resultado
a um fator isolado. O contraste auxiliar/controle é pareado dentro de cada fold.

**Não promover a loss por canal a default**, não relaxar a guarda e não retunar
peso/cap após observar essa validação. Preservar todos os resultados, inclusive
o efeito de redução de p90 contra controle nos seis grupos de cada fold.
Não iniciar outra continuação longa desse termo sem nova hipótese ou critérios.

Próximo passo executado: consolidar os modelos já disponíveis, sem novas
inferências/treinos. A continuação histórica epoch 39 do fold 1 também foi
reavaliada **retrospectivamente na tabela de métricas**, não retreinada nem
inserida retroativamente no seletor deste piloto. Ela satisfaz os três critérios
contra seu source, mas tem outro budget/optimizer. Registro e limites no
[relatório 59](59-GENERALIST_RESEARCH_REFERENCE_REGISTRY.md).

Nenhum teste externo, aquisição independente ou tolerância física foi definido.
Os guardados deste estudo permanecem fase 0; seus arquivos/decisões não foram
reescritos para substituir pelo candidato histórico.
