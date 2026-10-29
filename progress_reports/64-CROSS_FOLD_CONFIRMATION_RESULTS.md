# Confirmação concluída: transferência lidera, aceitação pendente

## Resultado e decisão

`spatial_abc_fold1_20261026/COMPLETE` e `STATUS.json` confirmam conclusão em
**28/10/2026 às 18:18 UTC**, incluindo seis avaliações completas dos selecionados
de fases 16/24. A/B/C completaram **24 épocas e 98.304 updates por braço**;
os selecionados finais são da época 24. Sem inferência no teste ou promoção final.

| Braço | RMSE macro ↓ | Correlação ↑ | p90 canal macro ↓ | Razão nominal |
|---|---:|---:|---:|---:|
| **A — transferência offshore** | **0,416764** | **0,904839** | **0,732120** | 40,367× |
| B — nativa do zero | 0,441654 | 0,893037 | 0,790221 | 40,367× |
| C — anisotrópica do zero | 0,456840 | 0,885001 | 0,795851 | 40,367× |
| Histórico 4× compatível, reavaliado | 0,457670 | 0,889594 | — | 40,367× |

Mesmas 6.940 janelas de validação, seis configurações, filtro fixo e replay
discreto exato. A reduz RMSE em **5,64% contra B**, **8,77% contra C** e
**8,94% contra o histórico** nesta rotação. Repete a direção do primeiro fold
(A/B/C: 0,445187 / 0,482705 / 0,495379). Não combinar folds como replicações
independentes nem atribuir diferenças entre eles a ganho do modelo: configurações,
seed e composição mudaram.

A lidera também no diagnóstico de canais não fracos: RMSE **0,5321 / 0,5812 /
0,5855** para A/B/C. Códigos usados na última época: **1.182 / 357 / 249** de
2.048. Tempos somados das épocas: **5,01 / 5,04 / 6,65 h**, sem avaliações ou
trabalho parcial perdido. Pré-treino offshore não está pareado em compute total.
Melhora ao teto não demonstra convergência definitiva; não estender o orçamento
retroativamente para buscar outro vencedor.

## Guardas: por que A ainda não é aprovado

Nenhum candidato passou todas as guardas. A final tem **um veto** contra sua
fase 16: p90 doped piorou 0,004769. Contra o histórico há **sete vetos**:

| Configuração | Δ RMSE | Δ correlação | Δ p90 canal |
|---|---:|---:|---:|
| doped · 5m SMF | −0,001877 | +0,000774 | **+0,010691** |
| multicore · C1 and C4 | **+0,006732** | **−0,004288** | **+0,014088** |
| multimode · Configuration 7 | −0,140295 | +0,053289 | −0,200276 |
| single-mode · Configuration 11 | **+0,059907** | **−0,017371** | **+0,208081** |
| single-mode · Configuration 3 | −0,035223 | +0,006915 | −0,141750 |
| single-mode · Configuration 7 | −0,109236 | +0,044077 | −0,275576 |

Δ = A − histórico; RMSE/p90 negativos e correlação positiva favorecem A.
O padrão doped/multicore reaparece no primeiro fold. Single-mode 11 merece
atenção: p90 de A **1,37775**, apesar de correlação agregada 0,95690.
Não substituir erro físico, amplitude/fase e caudas por correlação alta.

## Próximo passo executado: diagnóstico pareado, sem novo treino

Executei `finetuning/summarize_spatial_confirmation.py` sobre as avaliações
existentes. Ele verifica cobertura pareada por janela/arquivo e células iguais;
resume erro, correlação, amplitude RMS, fase/magnitude espectral, erro bruto e
escapes por arquivo. Artefatos:
`results/experiments/spatial_confirmation_diagnostics_20261028/`.

- Primeiro fold: **19/51 arquivos** têm RMSE maior e 18/51 correlação menor que
  sua referência histórica compatível.
- Confirmação: **24/51 arquivos** têm RMSE maior e 26/51 correlação menor.

São contagens descritivas, não teste de significância ou novo critério de seleção.
Métricas de janela são ponderadas pelas células dentro do arquivo; média dos
p90 de janelas **não é p90 pooled de canais**. Ambos os conjuntos de validação
pertencem ao histórico de desenvolvimento.

## Direção seguinte e limites

1. **Priorizar A como direção de pesquisa, não promover seus pesos.** A liderança
   repetiu; não iniciar atenção, outra loss ou refresh do codebook sem hipótese
   controlada sustentada pelos diagnósticos.
2. Revisar fisicamente as regressões pareadas, sobretudo multicore e single-mode
   11, preservando também exemplos favoráveis, antes de definir uma intervenção.
3. Definir com a aplicação tolerâncias de erro físico, amplitude/fase, cauda por
   canal e orçamento total de bits. Sem isso não declarar utilidade operacional
   nem relaxar guardas retrospectivamente.
4. Obter aquisições independentes e mapas físicos/qualidade óptica quando
   disponíveis; confirmação cruzada não cria validação externa.

**Nenhum novo treinamento foi enfileirado.** A próxima etapa sem novas premissas
foi concluída: confirmação e diagnóstico pareado. O filtro permanece inalterado;
HDF5 brutos preservados. Razões nominais usam o sinal retido e não incluem todos
os custos de mapa/container/modelo; exclusão espacial não é compressão neural.
