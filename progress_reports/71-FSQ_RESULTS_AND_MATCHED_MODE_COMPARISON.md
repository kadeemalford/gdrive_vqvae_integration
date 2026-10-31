# FSQ residual: ganho real e comparação de modos a orçamento comum

## Resultado do piloto FSQ

`residual_refinement_fsq_fold0_20261030` terminou em **30/10/2026 às 15:13 UTC**.
Passou a regra de extensão e completou **24.576 updates**; selecionado na época
**22**. Avaliações completas e streams verificadas nas fases 16/24, sem teste
reservado ou promoção final.

| Sistema | RMSE macro | Correlação | Razão real com metadados | Células refinadas |
|---|---:|---:|---:|---:|
| Base | 0,667586 | 0,727658 | 171,936× | 0% |
| FSQ em todos os tiles | **0,643206** | **0,748366** | 133,999× | 100% |
| FSQ com guarda | 0,647110 | 0,745537 | 136,456× | 92,6% |
| FSQ com guarda, teto 1,25× | 0,648796 | 0,744136 | 139,695× | 85,3% |

O ganho relativo de RMSE é **3,65% sem guarda**, **3,07% com guarda** e **2,81%**
no teto menor. A composição sem guarda apresenta uma regressão na avaliação por
configuração/métrica; as guardadas preservam os três critérios por construção
no encoder, não por validação física independente.

O refiner usou **2.048 estados** na última época, contra três códigos no VQ
residual anterior. Isso acompanhou um ganho de fidelidade; não concluir que toda
VQ seja inferior, que uso uniforme seja ideal ou que a rede esteja convergida.
Há uma seed e uma coorte de desenvolvimento reiteradamente consultada.

**Mesma cardinalidade/contagem nominal não implica mesma taxa real:** com VQ
quase constante, Huffman precisava de poucos bits. FSQ transmite mais informação
e paga mais payload. Portanto, não atribuir todo ganho a uma superioridade a
bytes iguais nem ignorar a economia lossless anterior.

## Próximo passo: modos concorrentes no mesmo teto de bytes

Runner `finetuning/compare_refinement_modes.py`, diretório
`results/experiments/matched_refinement_modes_fold0_20261030/`.
É análise retrospectiva de reconstruções congeladas, sem novo treino/inferência.

Comparar **denso somente**, **residual somente** e **híbrido**, sempre sobre a
mesma base compacta, metadados, 7.220 janelas e regra de admissão. Pesos e fontes
originais não mudam. Erros básicos dos dois caches devem coincidir dentro da
precisão numérica, antes de combinar métricas.

- Tetos **1,1× / 1,25× / 1,5× / 2×** por arquivo, referidos aos bytes dos frames
  compactos. Tetos comuns, não igualdade exata do gasto; reportar taxa efetiva.
- Todos usam a mesma prioridade: redução positiva de RMSE² relativo, ponderada
  por células, por byte extra. Não comparar prioridades distintas como se o efeito
  fosse exclusivamente do refiner.
- Só aceitar mudanças que reduzam RMSE sem piorar correlação/p90 contra o modo
  atual. Orçamentos maiores mantêm escolhas anteriores ou fazem upgrades que
  respeitem os mesmos critérios; transições residual↔denso não podem retirar
  uma melhora de cauda previamente admitida.
- Serializar novamente todos os casos em bundles reais; verificar byte a byte
  frames, ordem/composição, mapas físicos e metadados após leitura.
- Modo residual transmite base+resíduo; modo denso transmite só seu frame.
  Nenhuma features contínua do original chega ao decoder.
- Híbrido pode exigir **três modelos**; o conjunto de parâmetros usado deve ser
  cobrado separadamente uma vez. Ganho pequeno de erro pode não justificar
  implantação/latência de múltiplos encoders/decoders.

**Decisão seguinte:** confirmar por outra rotação/configuração somente se o ganho
do residual/híbrido justificar bytes, modelos e custo frente ao denso sob orçamento
comum. Caso seja dominado, não repetir treinos apenas porque FSQ utiliza mais
estados; preservar o resultado e mudar a hipótese.

## Comparação concluída: complementaridade pequena, não dominância

| Sistema, teto 1,25× | RMSE | Correlação | Razão real | Pesos usados FP32 |
|---|---:|---:|---:|---:|
| Denso selecionado | 0,646267 | 0,743512 | 138,498× | ~223,7 MB |
| Residual selecionado | 0,648712 | 0,744269 | 139,722× | ~140,2 MB |
| Híbrido | 0,643682 | 0,746568 | 137,837× | ~252,0 MB |

O híbrido consumiu ~0,48% mais bytes efetivos que o denso nesse teto. Portanto,
**não são pontos com taxa exatamente igual**, nem demonstram dominância estrita
por RMSE. Há ganho complementar pequeno com registry maior. O residual tem menor
custo de parâmetros e alguma vantagem de correlação, sem dominar o erro global.

As caudas mantêm um trade-off relevante: p90 macro do denso é ~4,172 milhões,
contra ~4,987 milhões do híbrido em 1,25×. No teto 2× são ~2,606 milhões contra
~4,987 milhões. Todas as estratégias preservam os critérios contra a base, mas
**o híbrido não é o vencedor das caudas**. Não ocultar esse resultado com guardas
que usam só o compacto como referência.

O residual guardado melhorou RMSE nas seis configurações contra a base, além de
fase e erro médio não fraco; isso justifica **uma confirmação limitada**, não
integração. A confirmação mantém candidatos denso/residual/híbrido e todos os
critérios, sem retunar a prioridade na mesma validação para fabricar vencedor.
Protocolo no [relatório 72](72-RESIDUAL_CROSS_FOLD_CONFIRMATION.md).
