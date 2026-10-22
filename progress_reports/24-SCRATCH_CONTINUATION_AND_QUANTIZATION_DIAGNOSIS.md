# Continuação scratch: resultados e diagnóstico da quantização

## 1. Resultado

A continuação dos dois braços scratch foi concluída, incluindo validação completa.
O cancelamento da sessão de espera não invalida os resultados: diretórios possuem
marcadores `COMPLETE`, summaries, histórico até epoch 32 e avaliação dos selecionados.

Fonte: `results/experiments/scratch_budget_continuation_20261011/`.
Ambos selecionaram **epoch 31**, não o último checkpoint epoch 32. Foram 65.536
updates adicionais por braço, mesmo optimizer/RNG e políticas do relatório 23.

| Candidato selecionado | Correlação macro completa | RMSE relativo completo | Pior família RMSE |
|---|---:|---:|---:|
| Transferido congelado | **0,682337** | **0,719900** | **0,817242** |
| Scratch nativo | 0,631440 | 0,771015 | 0,844333 |
| Scratch temporal | 0,673097 | 0,729713 | 0,831743 |

Taxa nominal pooled permanece **165,914x** em todos; 5.820 janelas, quatro
famílias/seis configurações. Não é taxa de codec completo nem teste independente.

Frente aos candidatos da triagem, scratch nativo passa de RMSE completo 0,778228
para 0,771015; temporal de 0,743567 para 0,729713. Mais orçamento ajudou, porém
não houve melhoria uniforme: temporal multimode **0,820682 → 0,831743** e
single-mode Configuration 3 **0,692637 → 0,698003** regrediram.

## 2. Configurações do temporal selecionado

| Família/configuração | RMSE relativo | Correlação |
|---|---:|---:|
| Doped / 5m SMF | 0,695047 | 0,719824 |
| Multicore / C1 and C4 | 0,813047 | 0,581835 |
| Multimode / Configuration 7 | 0,831743 | 0,588197 |
| Single-mode / Configuration 11 | **0,444085** | **0,887413** |
| Single-mode / Configuration 3 | 0,698003 | 0,711578 |
| Single-mode / Configuration 7 | 0,594957 | 0,808607 |

Temporal agora supera a referência transferida em Configuration 11/3, mas não
no macro; a vantagem em Configuration 11 concentra parte importante do ganho.
Seu p90 macro por canal é 1,238167 e erro não fraco 0,952813, contra
1,197300/0,940893 na triagem. Não escolher apenas pelo RMSE favorável.

## 3. Dinâmica de codebook

Na última época, scratch nativo usou **274 códigos**, perplexidade média **186,8**;
temporal **315 códigos**, perplexidade **208,7**. Houve expansão frente à época 16,
mas utilização ainda concentrada. A quantidade de códigos não é critério de sucesso
e não estabelece causalmente a origem do erro.

Epoch 32 regrediu no RMSE do piloto frente a 31, especialmente em multimode;
portanto a próxima ablação parte de **best_macro epoch 31**, preservando last 32
para rastreabilidade. O próximo experimento não é um resume silencioso de LR nova.

## 4. Diagnóstico quantizado versus contínuo

Runner: `finetuning/diagnose_quantization_gap.py`.
Resultados concluídos: `results/experiments/quantization_gap_20261011/`.
Comparar transferido selecionado e temporal selecionado em **32 janelas de treino**
(oito por família) e as mesmas **102 janelas de validação** do piloto.
Sem updates, teste ou alteração de checkpoints.

| Candidato/população | RMSE VQ | RMSE contínuo | Diferença VQ − contínuo |
|---|---:|---:|---:|
| Temporal / treino | 0,655322 | 0,623988 | 0,031334 |
| Temporal / validação | 0,728105 | 0,707678 | **0,020427** |
| Transferido / treino | 0,663863 | 0,643193 | 0,020670 |
| Transferido / validação | 0,717803 | 0,716027 | **0,001777** |

As médias de treino e validação têm composições/configurações distintas; não
interpretar sua diferença como estimativa pareada de gap de generalização.
O decoder contínuo reutiliza encoder/decoder treinados com VQ e sai da distribuição
de entrada quantizada. **Não é codec equivalente em bitrate, lower bound ou prova
de que reativar códigos recuperará a diferença**. É uma motivação de pesquisa,
com maior diferença observada no temporal do que no transferido na mesma validação.

## 5. Decisão

Manter transferido como referência congelada e temporal como linha de arquitetura
mais promissora entre os braços do zero. O nativo scratch não recebe outra extensão
inalterada neste passo: ganho menor e custo de oportunidade favorecem uma ablação
do temporal com controle explícito de quantização.

Comparar controle versus reativação limitada de códigos no mesmo source temporal,
com LR/optimizer iguais entre braços e sem alterar corpus/arquitetura/loss/taxa.
Protocolo no [relatório 25](25-PAIRED_CODEBOOK_REFRESH_PROTOCOL.md).
Nova seed, requisitos físicos e holdout independente continuam necessários.
