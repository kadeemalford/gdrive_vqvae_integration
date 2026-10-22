# Seleção generalista: representatividade e guarda por configuração

## 1. Pergunta e escopo

Separar dois problemas revelados pela continuação: uma amostra que não detecta
regressões e um seletor macro que aceita perdas já visíveis por configuração.
Somente pós-processamento das avaliações congeladas do relatório 45: **zero
updates, zero novas inferências, nenhum acesso a teste**.

Runner: `finetuning/audit_checkpoint_selection.py`.
Evidência: `results/experiments/checkpoint_selection_audit_20261018/`, protocolo,
hashes, `summary.csv`, `trials.csv`, `detection.csv` e `COMPLETE`.

## 2. Desenho fixo da auditoria

- Candidatos por taxa: initial 24, continued 40, loss-selected 38 e midpoint fixo.
- Duas taxas, duas amostragens, quatro orçamentos e 50 sementes (2027–2076).
- Amostragem random reproduz o sampler atual por arquivo, incluindo ordenação
  e RNG. Alternativa: uma janela aleatória em cada faixa de ordem temporal
  (`block_index`, `time_start`) por arquivo, sem reposição.
- Orçamentos 2/4/8/16 janelas por arquivo: **102/204/408/816 janelas**.
- Seletores: mínimo RMSE macro histórico versus macro sujeito a não regressão
  em **RMSE, correlação e p90 relativo por canal em cada configuração**.
- Baseline da própria taxa sempre elegível. Comparação numérica epsilon=1e-9,
  não uma tolerância física aceitável definida pela aplicação.
- Macro mantém pesos iguais por família e iguais por configuração dentro da
  família; não troca para média de todas as janelas ou de arquivos.
- Escolha feita somente na amostra; diagnóstico na parcela complementar e
  também no conjunto completo. Guardas de coordenadas, referência/rate e manifesto.

Total: **1.600 escolhas de seletor** e **4.800 linhas de detecção por configuração**.
O complemento não foi usado na escolha de cada ensaio, mas o corpus e os
candidatos já orientaram a hipótese: **não é holdout independente**. Sementes
reutilizam janelas/arquivos; suas frequências não são probabilidades de sucesso
em novas aquisições ou ensaios estatisticamente independentes.

## 3. Resultado: amostragem não corrige objetivo inadequado

O seletor macro escolheu epoch 40 em **todos** os desenhos e sementes de ambas
as taxas. As escolhas apresentaram alguma regressão nos três critérios por
configuração no complemento em **100%** desses ensaios. Não equivale a dizer
que todas as métricas/configurações regrediram; basta uma para violar a guarda.

No 4x, todas as amostras detectaram a piora de RMSE de single-mode Configuration 2
do continuado: **50/50 sementes em cada orçamento e desenho**. Portanto, nesse
caso, aumentar a amostra não soluciona o problema principal do seletor macro.
Com a guarda por configuração, **todos os ensaios 4x retiveram o baseline**.
Não há ganho novo de fidelidade nessa decisão: evita regressões apenas porque
nenhum candidato novo foi admitido. Não apresentar isso como modelo melhorado.

## 4. Resultado: guarda na amostra ainda pode falhar

Frequência de escolha com regressão no complemento para o seletor com guarda 2x:

| Janelas/arquivo | Random | Distribuída na ordem temporal |
|---|---:|---:|
| 2 | 18% | 20% |
| 4 | 14% | 6% |
| 8 | 6% | 6% |
| 16 | 6% | **4%** |

Esses casos eram aceitos como sem regressão na amostra. Mesmo o desenho 16
temporal falhou em **2/50 sementes**: midpoint escolhido em 96% dos ensaios,
continued em 4%. O resultado não autoriza fixar "4% de risco" para outro corpus.

Na configuração de regressão pequena do continuado 2x, single-mode Configuration 6,
a piora de RMSE foi detectada apenas em 48% das sementes random com duas janelas,
ou 46% na alternativa temporal. Com 16, as frequências foram 66%/72%. A magnitude
pequena e a heterogeneidade tornam a detecção instável, embora a guarda considere
também correlação/p90. Não exigir que uma única amostra curta identifique tudo.

Mais janelas aumentam representatividade, mas a alternativa temporal não vence
em todo orçamento. O custo de 816 janelas é 8x o de 102, antes de diferenças
de shapes/IO. Não trocar o default global com base nesta auditoria retrospectiva.

## 5. Decisão para o próximo protocolo

1. **Manter controle macro histórico** para comparação, sem reescrever estudos.
2. Reter baseline e checkpoints candidatos antes de descartar pesos; incluir
   candidatos por critérios de pior configuração/canal além do mínimo macro.
3. Usar amostragem ampliada/distribuída apenas como **triagem**. Comparar sua
   utilidade/custo prospectivamente, não como garantia de preservação.
4. Exigir **avaliação completa pareada dos finalistas** para admissão experimental
   contra o baseline da própria taxa, publicando todas as configurações e
   caudas. Neste corpus, isso favorece investigar midpoint 2x e manter baseline
   4x sob os três critérios considerados; não é promoção ou aceitação física final.
5. Se nenhum candidato passar, reter baseline em vez de tolerar automaticamente
   uma perda em uma configuração para obter ganho macro. Uma tolerância não nula
   requer justificativa física da aplicação; epsilon numérico não a substitui.
6. Não iniciar outra continuação longa com o mesmo seletor. Fixar previamente
   retenção/triagem/admissão e uma hipótese nova que justifique o compute.

A implementação deste relatório é **isolada e diagnóstica**: não mudou defaults,
loss, sampler ou resume do trainer. Critérios de admissão no corpus exploratório
não certificam generalização. Continuam necessários tolerâncias de aplicação,
bitrate aceitável e aquisições independentes para decisão final.

## 6. Reprodução e testes

```bash
.venv/bin/python finetuning/audit_checkpoint_selection.py \
  --experiment-dir results/experiments/frozen_rate_stability_20261018 \
  --output-dir results/experiments/checkpoint_selection_audit_NEW
```

Saída nova obrigatória, sem sobrescrever evidências. Testes cobrem sentido da
correlação, rejeição por cauda, fallback baseline, amostragem temporal reproduzível,
equivalência exata ao sampler random atual e ponderação por configuração.
Suite completa: **166 testes passaram**; `git diff --check` sem erros.
