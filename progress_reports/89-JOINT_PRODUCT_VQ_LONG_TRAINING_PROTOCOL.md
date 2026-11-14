# Produto VQ treinado conjuntamente: triagem e campanha longa condicionada

**Protocolo registrado em 12/11/2026, antes dos resultados.** O usuário autorizou
prosseguir até um treinamento mais longo, usando no máximo uma GPU remota, e
pediu aviso quando esse treinamento começar. Não prolongar um modelo colapsado
apenas para satisfazer duração.

## 1. Nova hipótese de representação

Encerrar a sequência de modificações pequenas de contexto/LR/loss. O próximo
contraste altera a estrutura do dicionário, com encoder e decoder inteiramente
treináveis: **flat K2048 versus produto 32×64**, ambos com 2.048 estados compostos
e **11 bits nominais por posição** na mesma grade compacta.

Os centroides do produto representam duas metades de um espaço latente rotacionado.
Cada índice composto corresponde a `left_id * 64 + right_id`. Só esse ID inteiro
é transmitido; o receptor reconstrói o vetor com os dois centroides, matriz e
centro compartilhados. Não transmite features contínuas, resíduos gratuitos ou
parâmetros por sinal ocultos.

**Isso não aumenta a capacidade combinatória:** o flat arbitrário contém como
caso particular qualquer arranjo cartesiano. A hipótese é uma melhor geometria
de otimização/compartilhamento entre combinações e coadaptação das transformadas,
não capacidade maior ou informação adicional gratuita. São 6.144 parâmetros de
centroides contra 262.144 do flat; a rotação/centro acrescentam 66.048 bytes de
buffers float32, efetivamente cobrados no modelo.

### Evidência que justifica a triagem

No relatório 79, PQ com PCA teve menor erro latente que VQ em uma condição,
mas reconstrução pior. Todos os testes de PQ mantiveram CNN e rotação congeladas;
o treino posterior alterou somente centroides por 256 updates. **Nunca foi
isolada a coadaptação completa encoder–PQ–decoder.** O presente protocolo não
reclassifica aqueles negativos como positivos nem presume que PCA por si resolve
o problema. A PCA desta campanha será ajustada de novo somente no treino do
fold 0; não reutilizar bases do fold 1 que viram grupos reservados aqui.

## 2. Dados, inicialização e controles

- Fonte comum: A compacto transferido, época 24, CNN nativa ~27,96 M parâmetros,
  corpus `spatial_band_v1_fold0_20261024`, `robust-escape-v1`, mesma geometria.
- Arquivos de checks internos separados por IDs de arquivo dos batches novos;
  uma janela por arquivo/região, seed 2110. Somente **treino** nesta triagem.
  A fonte histórica já viu esses checks: não são aquisições independentes.
- PCA ortogonal **com todas as 128 dimensões**, sem whitening/truncamento,
  componentes alternados entre metades, até 256 vetores por recorte de ajuste.
  Lloyd determinístico, 12 iterações, K32 e K64. Rotação/centro ficam fixos;
  centroides e toda a CNN treinam. Nenhuma banda/canal original é removida.
- Flat conserva os centroides adaptados originais; produto recebe os centroides
  ajustados nos arquivos de treino. Não fingir inicialização de quantizadores
  idêntica ou capacidade paramétrica igual. Backbone e corpus são comuns.
- **AdamW novo em ambos**, LR `1e-4`, weight decay `1e-5`, clip global 1, FP32,
  TF32 desligado, mesmos batches/seeds. Os momentos antigos não são herdados
  seletivamente. Todos os contadores novos começam em zero.
- Loss histórica `0,5 MSE + 0,5 L1 + VQ`, mesma máscara ordinária. Sem refresh,
  novos dados, filtros, mudança simultânea de arquitetura CNN ou objetivo de taxa.

Os 98.304 updates FEBUS anteriores e o pré-treino OOI são exposição compartilhada
da fonte. O novo treino não é do zero; produto e flat recebem o mesmo número
de updates adicionais, não necessariamente FLOPs/tempo iguais.

## 3. Caminho explícito para o treino longo

1. **Smoke:** dois updates por braço, incluindo maior tile, checagem de gradientes,
   contadores e replay discreto real. Sem validade científica de seus scores.
2. **Piloto:** **8.192 updates por braço**, blocos de 2.048, com seleção pelo menor
   RMSE físico macro nos checks de treino; inicialização elegível. Não ler sinais
   de validação/teste. Calibrar probabilidades nas mesmas 32 janelas de treino de
   ajuste para ambos e cobrar frames/FAK1/FSB1 reais dos checks.
3. **Decisão anterior à campanha longa:** produto precisa melhorar ≥1% de RMSE
   contra sua inicialização, ≥0,2% no último bloco do piloto, ficar a até **3% do
   RMSE do endpoint flat**, transmitir frames selecionados a até **105% dos bytes
   do flat selecionado** e terminar com perplexidade média de treino ≥16.
4. Se passar, auditar os artefatos e **lançar separadamente a continuação até
   65.536 updates novos totais por braço**. Preservar optimizer/RNG/ordem de
   batches e a seleção acumulada; não resetar para fazer parecer outro começo.
   Cada braço terá 57.344 updates restantes após o piloto.
5. **Avisar o usuário quando a campanha longa for lançada**, indicando GPU,
   diretório e orçamento. Sem launch automático escondido dentro do piloto.

As margens de 3%/5% acima são filtros exploratórios para decidir se vale dar
tempo à nova representação, **não relaxamento das guardas finais de codec**.
Não forçar a continuação se houver colapso, ausência de aprendizado ou custo
incompatível. Nenhuma extensão além de 65.536 está autorizada por este protocolo.

## 4. Avaliação e aceitação

Somente após ambos os braços longos terminarem, avaliar selecionados e endpoints
nas 7.220 janelas de validação. Sem seleção por esse conjunto durante treino,
sem leitura do teste. É desenvolvimento reutilizado, não confirmação externa.

Usar IDs compostos exatos, loader versionado e síntese em receptor carregado
separadamente; probabilidades ajustadas somente nas 32 janelas de treino comuns.
Cobrar counts, descriptor de semântica dos símbolos, registro, mapas, side,
framing e bundle, com pesos completos discriminados. A base/centro do produto
não são gratuitos. Mesma cardinalidade não garante o mesmo custo entrópico.

O investimento-alvo final permanece **≥5% menos RMSE físico macro que o flat
pareado, sem mais bytes completos e sem regressões por configuração** em RMSE,
correlação, p90, magnitude FFT, fase e distância RMS a 1. Um eventual positivo
precisa de confronto com a fronteira existente e confirmação em outra coorte.
Não promover por aprendizagem no piloto, perplexidade ou fatoração em si.

## 5. Implementação e recursos

`finetuning/factorized_vq.py`, `finetuning/run_factorized_long_study.py`,
`finetuning/evaluate_factorized_long_study.py`,
`finetuning/codec_wire_evaluation.py` e `tests/test_factorized_vq.py`.
Não alterar factories/experimentos históricos: a arquitetura exige loader
explícito e rejeita interpretação silenciosa como CNN flat.

GPU física 0 de `aumo-gpu-001`, uma RTX 4090 exposta por vez, braços sequenciais;
nenhum treino local. PCA/Lloyd de inicialização são ajustados na CPU remota,
sem tocar a segunda GPU. Diretórios planejados: `factorized_vq_smoke_fold0_20261112`
e `factorized_vq_long_fold0_20261112`. Código/relatórios sem commit ou push.

**Lançamento:** após os 378 testes locais, foi lançado o pipeline remoto de smoke
e piloto, com `CUDA_VISIBLE_DEVICES=0`. A fase longa não faz parte desse pipeline:
ela permanece dependente da auditoria de `LONG_GATE.json` e de lançamento
separado, com aviso ao usuário. Ainda não há resultado científico nesta atualização.

**Integração:** a primeira tentativa de smoke parou antes de qualquer update ao
chamar o leitor estrito de metadados de validação sobre checks de treino. A guarda
histórica rejeitou corretamente esses caminhos e **não foi alterada**. O novo
chamador agora constrói metadados explicitamente train-only; há teste de regressão
que rejeita validação/teste nesse caminho. A inicialização ajustada e o protocolo
interrompido ficam preservados com `INTEGRATION_BLOCKED.json`.

O smoke foi relançado em `factorized_vq_smoke_v2_fold0_20261112`, precedendo o
mesmo piloto científico ainda não iniciado pela tentativa anterior. Correção de
integração não conta como ganho científico nem muda critérios/budgets.

## 6. Piloto executado; campanha longa não iniciada

**Consolidação e auditoria: 14/11/2026.** O smoke versão 2 passou e o piloto
completou **8.192 updates por braço / 16.384 no total**, além de quatro updates
de integração. A tentativa inicial interrompida não executou updates neurais.
Foi usada somente a GPU física 0 remota, com braços sequenciais.

A preparação usou **93 recortes de ajuste / 23.740 vetores**, preservando todas
as 128 dimensões da PCA. Houve **18 checks de treino, dez configurações e quatro
famílias**, separados dos batches novos por IDs de arquivo, e 32 janelas de
calibração comuns. A fonte histórica já viu os arquivos de checks; não são
validação externa ou aquisições independentes.

`LONG_GATE.json` terminou com **`eligible: false`**. Os endpoints informados ao
gate foram:

| Evidência original do gate | Valor |
|---|---:|
| Produto, inicial | 10.583.060,916 |
| Produto, bloco anterior | 19.966.747,998 |
| Produto, endpoint | 5.386.979,445 |
| Flat, endpoint | 3.818.026,446 |
| Frames do produto selecionado | 29.223 B |
| Frames do flat selecionado | 31.628 B |
| Perplexidade média do último bloco do produto | 376,757 |

O produto ficou **41,09% acima do flat** no indicador de endpoint, reprovando
a margem de proximidade de 3%. As aparentes melhoras de 49,10% contra a origem
e 73,02% no último bloco **não devem ser interpretadas como progresso físico**:
o denominador patológico descrito abaixo domina esse indicador. Os números e
o gate originais foram preservados, sem substituição retrospectiva de métrica.

**Os 65.536 updates por braço não foram iniciados.** O nome do diretório
`factorized_vq_long_fold0_20261112` é o destino planejado da campanha, não prova
de que a fase longa ocorreu. Existem `PILOT_COMPLETE` e os históricos de 8.192;
não existem `LONG_TRAINING_COMPLETE` ou `COMPLETE`.

## 7. Auditoria do denominador e da seleção

Um check é **exatamente constante**, não apenas de pequena amplitude:

```text
train/1596Hw5G7k6ayrAh4S0iRDa3bPVNXpy3K_b01_r002_w19.npz
single_mode / Configuration 8 / região 2
10 canais × 1.000 amostras; todos os 10.000 valores brutos são zero.
```

O alvo normalizado também é zero. A proteção numérica registra média zero e
escala `1e-6`. A métrica histórica `relative_rmse` divide RMSE normalizado por
`RMS(alvo_normalizado) + 1e-12`. Assim, um erro normalizado pequeno vira uma
razão de bilhões quando o alvo inteiro é zero:

| Modelo selecionado | RMSE bruto nesse check | RMSE relativo normalizado nesse check | Participação no macro informado |
|---|---:|---:|---:|
| Flat | 4,99398e-9 | 4,99398e9 | 99,99998493% |
| Produto | 5,11703e-9 | 5,11703e9 | 99,99998484% |

Esses erros brutos são não nulos; **não** afirmar preservação exata da constante
ou aceitação operacional sem tolerâncias da aplicação. Porém, os milhões no
macro não significam erro físico de milhões nem demonstram colapso da rede.
São consequência verificável do denominador e da agregação, mesmo com ponderação
por células dentro da configuração.

O seletor original escolheu **flat no update 8.192 e produto no update 2.048**.
O menor valor da loss temporal normalizada registrado no histórico ocorreu em
**4.096 para ambos**: 0,340764 / 0,350805, flat/produto. Isso é somente uma
observação dos históricos existentes, **não uma nova seleção de checkpoints**.
No endpoint, a loss temporal foi 0,343154 / 0,352947; a do produto caiu em
relação à sua inicialização de 0,359555, sem provar convergência.

Portanto, a integridade da execução pareada permanece verificada, mas a
**interpretação do ranking e das curvas do seletor como aprendizagem típica
fica comprometida**. Não classificar este caso como o bug de alias de optimizer
do relatório 81: aqui o problema é o domínio da métrica de seleção. Não remover
o check, trocar epsilon, renomear o gate como aprovado ou reavaliar novos modelos
para salvar o resultado.

## 8. Bytes reais dos selecionados e limites da comparação

| Candidato selecionado | Update | Frames | Bundle completo | Arquivo de pesos completo |
|---|---:|---:|---:|---:|
| Flat | 8.192 | 31.628 B | 936.429 B | 111.869.687 B |
| Produto | 2.048 | 29.223 B | 934.018 B | 110.912.643 B |

Produto economizou **7,604% nos frames**, mas só **0,2575% no bundle** destes
18 checks. O transporte cobra o manifesto/mapas completos da preparação, CDFs,
descriptor e registro: ~904,7 kB de metadados comprimidos dominam essa coorte
pequena de 1.457.000 células. As razões completas são ~6,22× / 6,24×,
**não comparáveis diretamente** às razões históricas de coortes completas.
Não projetar amortização ou economia na validação como se tivessem sido medidas.

O produto usa **957.044 bytes a menos no arquivo de pesos**. Todos os buffers
de PCA/centro estão efetivamente presentes e cobrados; o custo em tensores
compartilhados é 111.832.324 / 110.874.372 bytes. Os arquivos cobram o codec
inteiro, não uma exportação mínima do receptor.

Outros diagnósticos já arquivados, sem alterar o critério de decisão:

- Correlação macro dos selecionados: **0,797282 / 0,785622**, flat/produto.
- P90 ordinário por canal: **0,945328 / 0,983729**, flat/produto.
- A métrica física auxiliar com outro piso de denominador retornou
  **0,575328 / 0,593236**; não substituir a métrica principal por ela para promover
  um candidato. Os pisos/unidades não são intercambiáveis no check constante.

A economia de bytes não demonstrou melhora conjunta de fidelidade e taxa.
Não há confirmação nas 7.220 janelas de validação, muito menos no teste.

## 9. Integridade auditada e encerramento

A auditoria adicional rodou **somente na CPU remota**, com GPU não exposta e
sem inferência neural ou updates. Verificou:

- **10.141 hashes de inputs**, código, vendors, fonte original e assets ajustados;
- checkpoints, linhagem e estados AdamW: **117 estados no flat / 118 no produto**,
  todos em 8.192 updates, coerentes com optimizer novo em ambos;
- igualdade de todos os tensores entre checkpoint selecionado e arquivo de
  implantação usado para os scores;
- descritores dos 2.048 IDs e bases 32×64, tabelas/registro, 36 frames standalone
  idênticos aos frames dos bundles, CRCs, limites dos índices, geometria e
  side original, com reader independente;
- tempos de updates de **~382,9 s / ~383,0 s**, além de preparação, triagem,
  serialização e smoke. Isso não é benchmark universal;
- reprodução do gate reprovado e ausência de marcadores de treino longo.

Registro reproduzível:
`results/experiments/factorized_vq_long_fold0_20261112/AUDIT_20261114.json`,
com hashes de artefatos, contribuição do check constante e script
`audit_existing_20261114.py`. Os artefatos pequenos e a auditoria foram copiados
para o mesmo caminho local. Não havia processo desta campanha ativo ao conferir.

**Decisão:** não iniciar a fase longa, promover modelos ou avançar outra etapa.
A limitação do seletor deve ser resolvida por protocolo próprio antes de
qualquer retomada, sem reescrever este negativo como sucesso. Conforme a
orientação mais recente do usuário, esta continuidade termina na auditoria e
documentação: **nenhum novo experimento foi lançado em 14/11**. Código de treino,
critérios e resultados originais permaneceram inalterados; sem commit/push.
