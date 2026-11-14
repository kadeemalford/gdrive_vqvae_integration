# Scheduler pareado: melhora pequena de RMSE, sem avanço admissível de taxa–fidelidade

**Consolidado em 10/11/2026.** Resultado da primeira proposta do
[relatório 80](80-LEARNING_RATE_AND_EXTERNAL_PRETRAINING_REVIEW.md), após a
repetição que corrigiu o isolamento do estado do AdamW. Estudo válido:
`results/experiments/lr_schedule_fold0_v2_20261109/`, com treino, avaliação completa
e auditoria concluídos. Nenhum modelo promovido ou teste reservado consultado.

## 1. Conclusão e decisão

**Cosine sem restart trouxe ganho pequeno, mas não passou o critério de avanço.**
Na validação completa, reduziu RMSE em **0,3239% contra LR constante alto** e
**0,1590% contra LR constante médio**, ambos com o mesmo orçamento de updates.
Os streams ficaram **0,2965% / 0,1320% maiores**, respectivamente, e houve
regressões por configuração. Não confirmar em outra coorte, ampliar budgets
ou transformar o scheduler em default com base neste contraste.

Isso não é um resultado de ganho zero: cosine liderou RMSE/correlação macro.
Entretanto, **o benefício isolado da trajetória de LR foi muito menor** que
o ganho de 1,185% contra a origem, pois continuar treinando a LR constante
também melhorou o modelo. Nenhum resultado estabelece saturação/convergência
ótima ou refuta todos os schedules.

## 2. Contraste efetivamente executado

- Fonte: A transferido compacto da primeira comparação espacial, endpoint
  época 24, `spatial_abc_20261024_v2/compact_A/best_macro.pt`. É estado completo,
  não apenas pesos: todos os 117 estados AdamW iniciam em **98.304** updates.
- Três braços, **16.384 updates adicionais cada**, oito blocos de 2.048. Mesma
  sequência de janelas, pesos iniciais, momentos AdamW e RNG; sem reset seletivo.
- LR constante alto `1e-4`; cosine `1e-4 → 1e-5`; constante médio `5,5e-5`,
  igual à média aritmética do cosine. Não equivale a igualar a dinâmica do AdamW.
- CNN nativa ~27,96 M parâmetros, K2048, grade `spatial-4x`, objetivo temporal
  `0,5 MSE + 0,5 L1 + VQ`, normalização/escape/seleção espacial preservados.
- GPU física 0: **uma RTX 4090**, FP32, TF32 desligado; braços sequenciais.
  ~727–728 s de updates por braço, ~36,4 min de updates somados, além de triagem,
  serialização e avaliação. São tempos desta execução, não benchmark pareado
  de hardware ou custo total do pré-treino offshore.
- Seleção na mesma triagem histórica de **130 janelas**, com a origem elegível.
  Todos selecionaram o último bloco; **selecionado e endpoint coincidiram**.
- Validação completa: **7.220 janelas, seis configurações e quatro famílias**,
  iguais entre os candidatos. Desenvolvimento reutilizado, não teste externo.

## 3. Resultado completo com bytes medidos

| Candidato | RMSE relativo macro ↓ | Correlação macro ↑ | Stream completo (bytes) | Razão completa |
|---|---:|---:|---:|---:|
| Origem congelada | 0,667586 | 0,727657 | 15.658.668 | 162,420× |
| Constante alto | 0,661818 | 0,731712 | **15.646.620** | **162,545×** |
| **Cosine** | **0,659674** | **0,733635** | 15.693.007 | 162,065× |
| Constante médio | 0,660725 | 0,732666 | 15.672.320 | 162,278× |

Ganho de RMSE contra a origem: **0,8641% / 1,1851% / 1,0277%**, na ordem dos
três braços. A comparação que isola o scheduler, porém, é cosine versus os
**controles de exposição iguais**, não cosine versus a origem sem esses updates.

FAK1 codifica índices discretos com fallback fixo/marginal; FSB1 cobra framing,
comprimentos, mapas, manifesto, tabelas e registro. As probabilidades de cada
modelo foram ajustadas em **32 janelas somente do treino**, mesmas coordenadas
e mesma regra entre candidatos. Índices/side foram decodificados antes da
reconstrução neural; o reader independente verificou todos os containers.

Cada modelo compartilhado de implantação ocupa **111.869.967 bytes**; o estado
AdamW não é custo do decoder e não foi cobrado como peso de implantação.
Incluindo os pesos uma vez nesta coorte curta, as razões ficam em **~19,94×**.
O ganho lossless histórico de ~172× usava outro empacotamento/contrato: não
compará-lo diretamente com esses ~162× como regressão de representação.

## 4. Distribuição do benefício e vetos físicos

| Família / configuração | Origem | Constante alto | Cosine | Constante médio |
|---|---:|---:|---:|---:|
| Doped / 5m LSF | 0,644387 | 0,640457 | **0,638126** | 0,639212 |
| Multicore / C1 and C3 | 0,776540 | 0,772868 | **0,771506** | 0,772427 |
| Multimode / Configuration 6 | 0,760750 | 0,755811 | 0,753037 | **0,752909** |
| Single-mode / Configuration 10 | 0,394264 | 0,374160 | **0,365867** | 0,372183 |
| Single-mode / Configuration 2 | 0,614146 | **0,609162** | 0,610720 | 0,611405 |
| Single-mode / Configuration 6 | 0,457592 | **0,451083** | 0,451499 | 0,451467 |

Valores são RMSE relativos por configuração, não RMSE pooled da fibra inteira.
Cosine melhora todas as configurações contra a origem em RMSE, mas não domina
os controles. Por exemplo, single-mode 2/6 pioram contra constante alto.

As guardas predefinidas examinam RMSE, correlação, p90 por canal, magnitude FFT,
alinhamento de fase e distância da razão RMS a 1, por configuração. Cosine
apresentou **sete vetos contra constante alto / oito contra constante médio**.
Entre os exemplos:

- Single-mode 2 versus constante alto: RMSE `0,609162 → 0,610720`, correlação
  `0,792852 → 0,791965`, p90 `1,093732 → 1,101337`, e fase ligeiramente pior.
- Single-mode 6: RMSE `0,451083 → 0,451499` e fase pior contra constante alto.
- Doped: fase ligeiramente pior contra constante médio.
- Multimode: p90 pior; contra a origem, RMS ratio `0,733660 → 0,728462`,
  afastando-se de 1, apesar do RMSE global melhor. Os campos before/after do
  veto RMS nos JSONs representam **distância a 1**, não a razão bruta.

O p90 de multimode nesta coorte é patológico em escala relativa: ~20–25 milhões
na agregação por janela, dominando sua macro. É sensível aos denominadores de
canais de pequena variação e **não deve ser interpretado como tolerância física
operacional ou medida direta da fidelidade típica**. Não o substituímos depois
dos resultados nem retiramos canais para salvar a guarda. Mesmo sem usar esse
veto, o ganho abaixo de 1%, o aumento dos bytes e as regressões de RMSE/fase
em outras configurações já impedem avanço.

## 5. Auditoria do isolamento e alcance da repetição

A execução anterior `lr_schedule_fold0_20261109/` foi preservada e marcada com
`AUDIT_REJECTED.json`: `load_state_dict` mantinha alias do tensor CPU de `step`
do AdamW, fazendo os contadores das ramificações posteriores iniciarem maiores.
Os modelos recebiam 16.384 batches cada, mas o estado inicial não era estritamente
igual. O arquivo fonte nunca foi modificado.

Na repetição válida, o estado é copiado profundamente, os contadores da fonte
em memória são conferidos por bloco e todos os braços terminaram em **114.688**, partindo de
**98.304**. O teste de regressão verifica também que passos e momentos da fonte
não são alterados ao atualizar ramificações independentes.

**Os hashes dos sete artefatos de pesos de implantação e seus resultados foram
iguais aos da execução anterior.** Assim, neste regime avançado do AdamW, o defeito
dos contadores não teve efeito numérico observado nos modelos retidos. Isso foi
verificado pela repetição, não presumido para dispensar a correção. A comparação
anterior não foi retroativamente promovida a execução válida.

## 6. Próxima etapa e limites

O ramo cosine testado fica encerrado, sem expansão automática. **A perda atual
não era explicada principalmente por uma política de LR constante inadequada.**
Essa é uma conclusão limitada ao checkpoint, corpus e budget deste contraste,
não demonstração de que otimização tenha sido resolvida.

Foi iniciado o pipeline da segunda hipótese, conforme seção 9 do relatório 80:
**OOI → FEBUS** versus **OOI → fase QuantX → FEBUS**, com 24.576 updates novos por
braço e a mesma sequência FEBUS final. Não usa os pesos adaptados desta campanha
como nova inicialização, não adota cosine e não altera guardas. A integração GPU
é pré-requisito do piloto completo; resultados da ponte serão registrados após
sua conclusão. Somente uma RTX 4090 remota pode ser usada por esta pesquisa.

Tolerâncias finais de aplicação e independência física de aquisições seguem
pendentes; não são substituídas por esses controles de desenvolvimento.
