# Comparação A/B/C concluída e confirmação em outro fold

**Confirmação concluída em 28/10/2026 às 18:18 UTC:** A/B/C terminaram 24 épocas
e todas as avaliações previstas, sem promoção final. Resultados, vetos e próxima
direção estão no [relatório 64](64-CROSS_FOLD_CONFIRMATION_RESULTS.md).
Os registros de retomada abaixo são históricos, não indicação de execução ativa.

## 1. Conclusão do estudo inicial

O estudo `results/experiments/spatial_abc_20261024_v2/` concluiu os seis braços,
suas extensões predefinidas e todas as avaliações completas. `COMPLETE` e
`STATUS.json` confirmam término em **2026-10-26 23:12 UTC**, sem promoção final.

**A transferência offshore foi a melhor direção nesta coorte**, tanto em taxa
compacta quanto densa. A arquitetura anisotrópica C não superou esse controle.
Isso é evidência sobre estas implementações, corpus, seed e orçamento, não prova
de que novas arquiteturas ou treino do zero sejam intrinsecamente inferiores.

## 2. Resultado completo dos selecionados após as extensões

Figuras derivadas das avaliações completas, sem novas inferências:
[taxa × fidelidade](../results/experiments/spatial_abc_20261024_v2/rate_fidelity_phase24.png)
e [variação por configuração na taxa alta](../results/experiments/spatial_abc_20261024_v2/high_rate_configuration_deltas.png).

Mesmas **7.220 janelas de validação**, quatro famílias/seis configurações,
`persistent-amplitude-v1`, normalização e loss comuns. Seleção por RMSE macro na
triagem fixa; avaliação completa posterior, sem novo ajuste dos pesos.

| Taxa | Braço | Época selecionada | RMSE macro ↓ | Correlação macro ↑ | Compressão nominal pooled |
|---|---|---:|---:|---:|---:|
| Densa | **A — offshore + adaptação** | 23 | **0,445187** | **0,888093** | 41,577× |
| Densa | B — nativa do zero | 23 | 0,482705 | 0,867561 | 41,577× |
| Densa | C — anisotrópica do zero | 22 | 0,495379 | 0,859700 | 41,577× |
| Compacta | **A — offshore + adaptação** | 24 | **0,667586** | **0,727658** | 160,211× |
| Compacta | B — nativa do zero | 24 | 0,702018 | 0,693756 | 160,211× |
| Compacta | C — anisotrópica do zero | 23 | 0,700077 | 0,697539 | 160,211× |

Todos atenderam à regra de extensão e completaram **24 épocas/98.304 updates**:
**589.824 updates** no total. Portanto, nesta realização, a exposição adicional
final também é igual entre braços; o compute histórico offshore continua não
pareado. O tempo somado das épocas completas foi ~**20,45 h**, sem incluir
avaliações completas, inicializações, interrupções ou trabalho parcial perdido.

C tem 26,3 M parâmetros versus 28,0 M na nativa, mas consumiu mais tempo:
~5,12 h versus ~3,77–3,79 h no ponto denso; ~3,10 h versus ~2,33–2,35 h no compacto.
São tempos desta máquina/protocolo, não benchmarks universais. Sua pequena
vantagem compacta sobre B não compensa uma superioridade inexistente sobre A.

Taxas nominais incluem índices e side de normalização/escapes no sinal retido,
não todos os custos do codec. Não contar exclusão de cobertura como compressão
neural. Não comparar diretamente estes macro scores com médias históricas de
outra seleção espacial/ponderação.

## 3. Guardas e limitações

**Nenhum dos seis selecionados finais passou todas as guardas aplicáveis.**
Evidência: `guards.json`, `full_phase24_*/<braço>.groups.csv` e
`baselines/historical_4x.groups.csv` no diretório do estudo.

- A denso preservou os três critérios em todas as configurações contra seu
  próprio selecionado de fase 16, mas teve **seis vetos** contra o 4× histórico
  reavaliado no mesmo corpus: p90 doped; RMSE/correlação/p90 multicore C1 and C3;
  RMSE/p90 single-mode Configuration 10.
- A compacto teve **dois vetos** contra A compacto de fase 16, apesar do ganho
  macro. B/C também apresentaram regressões por configuração/canal contra A.
- A cauda relativa macro em multimode permanece muito sensível a um pequeno
  número de canais/regiões de escala extremamente desproporcional. O detector
  conservador não eliminou toda patologia numérica; o mecanismo de escape continua
  necessário. Não interpretar p90 macro na ordem de milhões como erro uniforme
  de todos os canais, nem simplesmente excluir esses casos após ver o resultado.
- Os resultados são de uma seed e uma rotação de desenvolvimento já consultada.
  Não houve inferência no teste reservado, novas aquisições ou auditoria externa.

### Treino do zero e overfitting

A liderança de A não depende apenas da cauda numericamente problemática: o erro
médio relativo dos canais fora do grupo fraco diagnóstico foi, no ponto denso,
**0,5279 / 0,5729 / 0,5894** para A/B/C e, no compacto,
**0,7907 / 0,8347 / 0,8169**. Essa estratificação é diagnóstico complementar,
não autorização para remover canais da métrica principal ou garantia física.

As curvas não mostram overfitting como explicação dominante: a reconstrução de
treino e validação melhorou em conjunto durante boa parte do orçamento. Há gap
moderado nos compactos ao final, e a validação de época é uma amostra fixa, não
uma medição pareada de toda a generalização. Mais dados independentes continuam
potencialmente importantes; quantidade de janelas correlacionadas não basta.

No fim, A utilizava ~1.176/1.229 códigos nos pontos denso/compacto; B, 353/293;
C, 234/212, de 2.048 disponíveis. Esse contraste sugere uma diferença de
otimização/uso do gargalo, **não prova causal de falta de capacidade ou de
memorização**. Não iniciar refresh de códigos ou trocar a loss sem controle
apenas por esse diagnóstico; hipóteses semelhantes já tiveram resultados mistos.

## 4. Próximo passo: confirmação A/B/C no ponto denso

Executar uma **confirmação em outra rotação de configurações e seed**, antes de
adicionar atenção, rever a arquitetura C ou concluir que o scratch não compensa.
Priorizar somente a taxa densa por ser o objetivo de maior fidelidade e por A
já liderar também o ponto compacto. Não repetir seis braços sem uma pergunta nova.

- **Test fold 1, validação fold 2, treino folds 0 e 3**, mantendo agrupamento por
  família/configuração, sem misturar janelas de uma aquisição entre papéis.
- Mesmos 207 HDF5 e filtro/tiling; nenhuma nova fonte de dados ou ajuste de limiar.
  Preparar em `data/finetune/spatial_band_v1_fold1_20261026/` e auditar os arrays
  antes de qualquer inferência ou treinamento dessa confirmação.
- **A/B/C, somente `spatial4-temporal4`**, K=2.048, dimensão 128, objetivo,
  normalização, avaliação ponderada por células e replay exato inalterados.
- Seed **2032**, compartilhada entre braços; reservatório somente do treino.
  Troca de seed e configuração é teste de robustez da conclusão, não ablação
  causal que isole uma dessas mudanças.
- Mesmo orçamento: **16 × 4.096 updates por braço**, extensão única de oito épocas
  se a melhora entre os dois últimos blocos de quatro épocas for ≥0,2%; teto 24.
  Não descartar scratch por não ganhar na primeira triagem.
- Controle histórico compatível com essa rotação:
  `results/experiments/high_rate_continuation_20261015/rate_4x/best_macro.pt`,
  época 39, treinado no `fold_1` legado. **Não usar o histórico fold 0** como
  referência congelada desta validação, pois ele já treinou em configurações que
  agora pertencem à validação fold 2.
- Teste fold 1 continua sem inferência e fora de seleção. Esta é confirmação
  de desenvolvimento em CV, **não independência de novas aquisições**.

O runner foi generalizado com `--test-fold`, `--validation-fold`, `--rates` e
`--seed`, preservando os controles e o orçamento. Os 17 arquivos-fonte fixados
pelo estudo inicial foram arquivados e verificados em `code_snapshot/` desse
estudo antes da evolução do runner; resultados históricos não foram reescritos.

Os resultados da confirmação e o estado de execução serão registrados em seus
próprios artefatos. Caso a liderança de A se repita, priorizar sua validação
robusta e critérios físicos, em vez de iniciar outra sequência aberta de mudanças
arquiteturais sem evidência. Nenhum resultado desta rodada promove um codec final.

## 5. Confirmação preparada, auditada e enfileirada

### Retomada e conferência efetiva do filtro — 27/10

**Atualização — 28/10:** após novo reinício, não havia processo ativo nem unidade
disponível. C concluiu 24 épocas, A chegou à 18 e B à 16. Checkpoints e ambos os
CSVs de cada braço estavam coerentes; os hashes de código e metadados permaneceram
iguais aos do protocolo. A unidade foi recriada com os mesmos argumentos e
`--resume`, para continuar A da época 18 e as etapas restantes, sem alterar o
orçamento ou repetir trabalho concluído. Ainda não havia marcador `COMPLETE`.

Em uma checagem posterior ao reinício do servidor, C e A já tinham concluído
as **16 épocas iniciais**, com checkpoints e CSVs coerentes. B tinha checkpoint
na época 0, antes de completar a primeira época. Não havia processo ativo nem
unidade disponível; código e metadados continuavam com os hashes do protocolo.
A mesma unidade foi recriada com `--resume`, mantendo fold, seed, taxa e limites
originais, sem repetir as épocas concluídas. Ainda não havia `COMPLETE`; extensões
e síntese final permaneciam pendentes.

A confirmação ainda não tinha `COMPLETE`; não havia processo ativo nem unidade
transitória disponível. O checkpoint de `high_C` e ambos os CSVs confirmaram a
**época 10**. Código e metadados conservaram os hashes do protocolo. Foi retomada
a mesma fila com `--resume`, fold 1/validação 2/taxa alta/seed 2032, preservando
optimizer, RNG, próximos braços e teto original de épocas.

A aplicação do filtro foi conferida **nos dados efetivamente usados**, não apenas
no default da CLI:

- `high_C/last.pt` aponta para
  `data/finetune/spatial_band_v1_fold1_20261026` e registra
  `preprocessing.spatial_band_policy = persistent-amplitude-v1`, com parâmetros
  iguais aos do relatório 61.
- Todas as **14.620 linhas do manifesto de treino** registram essa política.
- Cada mapa de treino foi cruzado com a decisão do HDF5: IDs de filtro coincidem,
  nenhum índice selecionado para remoção aparece nos canais retidos e as regiões
  entregues ao modelo são contíguas.
- `preprocessing.json` registra as decisões para os 207 HDF5, com exclusão por
  amplitude em 40 arquivos no corpus completo, não necessariamente em todos os
  arquivos ou em todas as janelas.

O detector é executado **na preparação**, antes do recorte temporal/normalização.
O trainer carrega os NPZ já selecionados; não recalcula a máscara em cada época.
Isso mantém a decisão estável entre braços. Não implica eliminação de toda faixa
fisicamente inadequada: a regra permanece heurística/conservadora.

A nova preparação foi concluída e **todas as 28.780 janelas conferidas novamente
contra os brutos**, em `results/experiments/spatial_corpus_audit_fold1_20261026/`,
com marcador `COMPLETE`. As decisões e os parâmetros do filtro são **idênticos
para os 207 arquivos** nas duas preparações; mudaram apenas os papéis dos splits:

| Papel na confirmação | Janelas |
|---|---:|
| Treino, folds 0 e 3 | 14.620 |
| Validação, fold 2 | 6.940 |
| Teste reservado, fold 1 | 7.220 |

A triagem fixa tem **126 janelas**, até duas por arquivo/região, seed 2032.
As validações completas usam todas as 6.940 janelas. O smoke de memória/replay
continua válido para a mesma arquitetura, shapes, taxas e política numérica;
a suíte completa passou **215 testes** após generalizar o runner. O runner
também verifica que a referência histórica não treinou em configurações da
validação atual, em vez de confiar apenas no nome do diretório.

Execução: `results/experiments/spatial_abc_fold1_20261026/`.
Unidade: **`febus-spatial-abc-fold1-20261026.service`**, independente do chat,
com os mesmos limites operacionais de 48 h/22 GiB. Ela inicializa os três braços,
avalia os controles congelados e executa `high_C → high_A → high_B`, seguidos
das extensões previstas e auditorias. Teto total de **294.912 updates** para
esta confirmação; nenhum treino compacto adicional foi enfileirado.

`STATUS.json`, `protocol.json`, checkpoints, `full_phase*`, `guards.json`,
`REPORT.md` e `COMPLETE` da nova execução são as fontes de estado e resultados.
Uma notificação de espera cancelada não prova término ou falha da unidade.

```bash
systemctl --user status febus-spatial-abc-fold1-20261026.service --no-pager
cat results/experiments/spatial_abc_fold1_20261026/STATUS.json
```

Para retomar após verificar que não existe processo ativo, manter todos os
argumentos da confirmação, inclusive fold, taxa e seed:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -u \
  finetuning/run_spatial_abc_study.py \
  --output-dir results/experiments/spatial_abc_fold1_20261026 \
  --samples-dir data/finetune/spatial_band_v1_fold1_20261026 \
  --audit-dir results/experiments/spatial_corpus_audit_fold1_20261026 \
  --smoke-dir results/experiments/anisotropic_rate_smoke_checkpointed_20261024 \
  --test-fold 1 --validation-fold 2 --rates high --seed 2032 --resume
```
