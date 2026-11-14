# CNN/contexto com análise, codebook e síntese treinados conjuntamente

**Protocolo registrado em 12/11/2026, antes de qualquer resultado novo.**
Autorizado pelo usuário após o fechamento do relatório 87.

## 1. Hipótese e diferença dos experimentos anteriores

Os probes dos relatórios 84/87 congelaram a CNN e o codebook. Podiam testar
correções de features/tokens, mas não se o sistema inteiro aprende a distribuir
informação de outra forma quando análise e síntese se coadaptam ao contexto.

Este contraste treina **todos os pesos**: encoder original, codebook, decoder
original e blocos novos antes/depois do bottleneck. Não é prolongar os pequenos
adaptadores congelados, tampouco uma arquitetura Transformer inteiramente nova
treinada do zero. É uma continuação CNN–contexto com transferência controlada.

## 2. Braços e referências

| Braço | Contexto na análise e síntese | Pesos novos | Pesos antigos |
|---|---|---:|---|
| Pointwise | MLP local com forma/posição conhecidas | 67.768 | Todos treináveis |
| Axial | Atenção espacial e temporal, com forma/posição conhecidas | 67.648 | Todos treináveis |
| Referência nativa | CNN sem contexto novo, controle constante do relatório 81 | 0 | Continuação histórica completa |

O controle pointwise recebe **a mesma informação de forma/posição** e tem
capacidade nova diferente em menos de 1%. Ambos iniciam as projeções residuais
em zero, reproduzindo a fonte. A média/variância global de GroupNorm já existe
na CNN: a proposta é contexto de conteúdo mais rico, não a primeira dependência
global do sistema.

Fonte comum: **A compacto transferido, época 24, AdamW em 98.304 updates**.
Não usar os pesos selecionados dos pequenos probes ou da ablação MSE.

## 3. Orçamento e otimização pareados

- **16.384 updates novos por braço**, oito blocos de 2.048, sem extensão automática.
- Exatamente os mesmos batches e triagem histórica de 130 janelas do controle
  constante corrigido `lr_schedule_fold0_v2_20261109`. Checar que batches são de treino.
- Backbone e codebook herdam momentos AdamW independentes; os parâmetros novos
  começam sem momentos. Não presumir que todos os contadores tenham a mesma idade:
  fim esperado **114.688 na base / 16.384 nos contextos**.
- Contextos herdam os mesmos hiperparâmetros de grupo do optimizer da base,
  exceto seus estados inicialmente vazios. LR constante `1e-4`, clip global 1.
- Loss histórica `0,5 MSE + 0,5 L1 + VQ`, normalização `robust-escape-v1`, mesma
  máscara ordinária, todas as amostras/bandas retidas. Sem refresh de códigos,
  filtros ou nova fonte de pré-treino.
- FP32, TF32 desligado, algoritmos determinísticos e atenção matemática sem
  dropout/fastpath. Os dois braços novos usam o mesmo contexto numérico.

O controle nativo existente será reutilizado após auditoria de código, fonte,
dados, sequências, contadores, pesos e streams. É uma **referência prática** com
menos parâmetros e sem essas opções específicas de atenção, não um terceiro
braço recém-executado com compute/FLOPs iguais. O contraste causal principal
da informação contextual é **axial versus pointwise**.

Antes do treino completo, a triagem das inicializações identidade deve reproduzir
o RMSE inicial histórico. Cada bloco verifica contadores da base/contexto e
imutabilidade dos contadores da fonte. Retomada só por blocos completos, com
estado/RNG e hashes congelados.

## 4. Codec e avaliação

- Grade compacta `spatial-4x`, K2048 e mesma quantidade nominal de IDs. Não
  presumir igualdade de bytes quando as frequências de símbolos mudarem.
- Encoder contextual recebe features contínuas **somente no compressor**.
- Receptor recebe índices discretos, forma original/latente e side reversível.
  Recupera os vetores exatos, aplica contexto de síntese e decoder. Não recebe
  features de análise, residual, passado original ou cache de atenção.
- Avaliar selecionado e endpoint dos dois braços nas **7.220 janelas de validação**,
  somente depois dos treinos, nunca no teste reservado. Inicialização elegível.
- Mesmo FAK1/FSB1, calibração somente nas mesmas 32 janelas de treino. Reader
  independente de container; síntese usa um modelo carregado separadamente dos
  pesos compartilhados, sem executar o encoder do receptor.
- Discriminar transporte, pesos completos e tensores necessários ao receptor.
  A comparação de arquivo compartilhado cobra conservadoramente **o codec inteiro**,
  incluindo encoder, assim como a referência histórica. Não chamar esse arquivo
  de exportação mínima de decoder. Reportar contextos de encoder/decoder separados.

## 5. Critérios antes dos resultados

Selecionar menor RMSE relativo macro na triagem histórica, com endpoint guardado
separadamente. A validação é desenvolvimento reutilizado, não confirmação externa.

Só considerar confirmação do axial se obtiver **≥5% menos RMSE físico macro
contra ambos pointwise e nativo**, sem aumentar bytes completos e sem regressão
por configuração nas guardas de RMSE, correlação, p90, magnitude FFT, fase e
distância da razão RMS a 1. Não afrouxar a guarda para salvar uma média.

Um ganho de pointwise isolado ou outra troca taxa–distorção deve ser documentado,
não rebatizado de sucesso da atenção. Qualquer positivo ainda precisará de
confronto com a fronteira preservada e confirmação em outra coorte; não há
promoção automática ou tolerâncias operacionais novas.

## 6. Execução

Somente GPU física 0, RTX 4090 remota, braços sequenciais. Primeiro integração
com dois updates por braço incluindo maior tile e replay de stream, depois a
campanha completa. Nenhum treino local. Saídas `joint_context_smoke_fold0_20261112`
e `joint_context_fold0_20261112`, em `results/experiments/`.

Implementação versionada em `finetuning/joint_context_vq.py`,
`finetuning/run_joint_context_study.py`, `finetuning/evaluate_joint_context_study.py`
e `tests/test_joint_context_vq.py`. A factory histórica não é alterada; a nova
arquitetura precisa do loader explícito, para impedir inferência silenciosa como
CNN nativa. Código e documentos permanecem sem commit/push.

## 7. Resultado concluído e auditado

Campanha `joint_context_fold0_20261112` concluída, com integração GPU/replay
prévia e validação completa. Ambos os braços fizeram **16.384 updates novos**,
selecionaram o bloco 8 e tiveram selecionado/endpoint coincidentes em métricas.
Nenhum treino foi estendido além do protocolo.

| Candidato | RMSE relativo macro ↓ | Correlação ↑ | Stream completo (bytes) | Arquivo de pesos completo (bytes) |
|---|---:|---:|---:|---:|
| Nativo, continuação histórica pareada | 0,661818 | 0,731712 | 15.646.620 | 111.869.967 |
| **Pointwise conjunto** | **0,660920** | **0,732743** | 15.664.741 | **112.144.239** |
| Axial conjunto | 0,661661 | 0,732497 | **15.593.407** | 112.159.535 |

**Axial não superou pointwise em fidelidade:** teve **0,1121% mais RMSE**,
economizando **71.334 bytes / 0,4554%** no transporte. Contra nativo, ganhou
somente **0,0237% de RMSE** e economizou **53.213 bytes / 0,3401%**. Ambos estão
muito abaixo do gate de investimento de 5%; tampouco constituem economia de 10%
a fidelidade equivalente. Houve **20 vetos contra pointwise e 13 contra nativo**.

Pointwise ganhou só **0,1356% de RMSE** contra nativo, com **0,1158% mais bytes**.
Isso também não é um avanço de representação que justifique confirmação. O
arquivo axial tem menos parâmetros novos, mas mais bytes serializados devido
à estrutura/quantidade de tensores do state dict; não substituir arquivo real
por contagem de parâmetros para favorecer um braço.

Os contextos acrescentam 135.536 bytes de tensor em cada ponta no pointwise e
135.296 em cada ponta no axial. O custo completo de pesos da tabela é cobrado
conservadoramente, incluindo encoder, como na referência; os JSONs discriminam
os tensores necessários ao receptor, sem fingir que o arquivo completo seja
uma exportação mínima.

### Distribuição por configuração

| Família / configuração | Pointwise: RMSE | Axial: RMSE |
|---|---:|---:|
| Doped / 5m LSF | 0,640861 | **0,638557** |
| Multicore / C1 and C3 | 0,772667 | **0,771577** |
| Multimode / Configuration 6 | **0,753721** | 0,755725 |
| Single-mode / Configuration 10 | 0,378425 | **0,376043** |
| Single-mode / Configuration 2 | **0,604612** | 0,614205 |
| Single-mode / Configuration 6 | **0,446254** | 0,452107 |

Axial melhora três configurações e piora três frente ao pointwise. As médias
de magnitude FFT foram **0,500764 / 0,499730**, fase **0,842412 / 0,841251** e
RMS ratio **0,771750 / 0,770872**, na ordem pointwise/axial. Nenhuma métrica
macro isolada substitui as guardas por configuração.

### Auditoria de coadaptação e compute

- Verificados no remoto: **114.688** nos estados AdamW da base e **16.384** nos
  estados dos contextos, em ambos os braços. Linhagem dos checkpoints e hashes
  de código/vendors/fonte conferidos; fonte original permanece em 98.304.
- Todos os parâmetros estavam treináveis e os pesos da base efetivamente
  mudaram. Portanto, este resultado não vem de congelamento acidental como nos
  probes anteriores. Os momentos herdados não foram compartilhados entre braços.
- Updates de treino: ~**776,6 s / 841,6 s**, pointwise/axial, além de triagem,
  checkpoints e avaliação completa. Axial custou ~8,4% mais tempo de updates
  nesta execução, não um benchmark universal de hardware.
- Picos registrados de memória alocada CUDA: ~**2,64 / 2,90 GiB**, sem significar
  memória total reservada, custo do driver ou requisitos de outro tamanho/batch.
- Uma única RTX 4090 remota, braços sequenciais. Reader independente e síntese
  com pesos carregados separadamente; teste reservado não consultado.
- Protocolos, curvas, summaries e CSVs de janelas/grupos foram copiados para
  `results/experiments/joint_context_fold0_20261112/` local.

## 8. Decisão e alcance do negativo

**Não promover, confirmar ou prolongar automaticamente esta continuação.** A
atenção axial pequena na grade latente, mesmo com coadaptação da CNN e do VQ,
não mudou o patamar frente ao controle local ou à CNN histórica. Encerrar esta
sequência de contextos sobre o mesmo bottleneck, sem retunar a guarda na validação.

Isso não prova que atenção seja inútil ou estabelece um limite informacional:
o teste usa uma referência madura, contexto estreito na grade latente, uma seed,
16.384 updates adicionais e uma coorte de desenvolvimento reutilizada. Não
equivale a testar todas as posições/profundidades de atenção ou treinamento de
um Transformer completo do zero.

O conjunto recente de negativos reduz a justificativa de outra intervenção
marginal nesta família. Uma próxima campanha precisa de **mecanismo novo de
representação/informação transmitida**, além de mais LR, pré-treino, contexto ou
largura, e deve começar por evidência de treino e orçamento real. Não inventar
um novo treino apenas porque este terminou; preservar as referências e o
compromisso físico de reconstruir também o ruído, sem redefinir sucesso.
