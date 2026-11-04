# Pilotos remotos de representação: diagnóstico, busca de índices e taxa–distorção

**Data: 31/10/2026.** Campanha concluída, autorizada pelo usuário após a revisão do
[relatório 77](77-RESEARCH_DIRECTION_REVIEW.md): diagnósticos E0/E1, seis braços
de treinamento E2, referências VQ completas e comparação em bytes comuns. Este
documento distingue resultados, análises posteriores e recomendações ainda não
executadas. Nenhum modelo foi promovido a solução generalista final.

Foram executadas **131.072 atualizações de parâmetros**, somando os seis braços.
Ao fechar a campanha, não havia processo de compute nas GPUs. Nenhum treinamento
adicional foi deixado rodando sem orçamento ou supervisão.

## 1. Ambiente, sincronização e integridade

- Máquina: `murilo@aumo-gpu-001`, repositório em
  `/home/murilo/source/das_febus/gdrive_vqvae_integration`.
- Autenticação SSH exclusivamente com a variável `SSH_PASSWORD`; seu conteúdo
  não foi gravado em código, comandos, relatórios ou artefatos do projeto.
- Duas RTX 4090 de 24 GB; PyTorch `2.11.0+cu128`. Instalado `constriction==0.5.0`
  a partir do arquivo de requisitos entrópicos existente.
- O remoto passou de `a50ae69` a `aac0b8d` por `git pull --ff-only`. Arquivos
  não rastreados da campanha de capacidade foram preservados em
  `results/remote_setup/pre_sync_20261031T145108Z/`; a alteração de `.gitignore`
  foi preservada por backup/stash. O stash não foi descartado.
- README, roadmap e relatório 77 estavam **sem commit na máquina local**.
  Foram transferidos separadamente, assim como as novas implementações, sem
  fabricar um commit ou sobrescrever trabalho remoto desconhecido.
- Verificados por SHA-256 os **21.560 NPZ de treino/validação** do fold 1 contra
  o manifesto de transferência anterior. Nenhum array de teste foi aberto.
  Auditoria: `results/remote_setup/representation_integrity_audit_20261031.json`.
- Para o diagnóstico do fold 0, transferidos somente 176 NPZ selecionados por
  metadados: 111 de treino e 65 de validação. Manifesto original, subconjunto e
  hashes estão em `data/diagnostics/e1_fold0_20261031/` no remoto. Esse subconjunto
  **não** é um novo corpus de treinamento.

Os folds identificam as preparações históricas: fold 0 usa validação 1; fold 1,
validação 2. São avaliações de desenvolvimento por configuração, não um holdout
externo novo. Não se deve combinar os dois modelos como uma única trajetória.

## 2. Reavaliação da direção

A recomendação do relatório 77 foi mantida, mas agora com evidência experimental:

1. A escolha dos índices deixa ganho recuperável mesmo com pesos congelados.
2. Ganho no objetivo temporal não garante preservação das caudas por canal.
3. O codec escalar conjunto é implementável com stream real, mas deve demonstrar
   vantagem na curva antes de receber maior capacidade ou hyperprior.
4. Contexto maior continua condicional. Os diagnósticos atuais de 50 ms não
   demonstram dependência entre blocos contíguos de 0,25–1 s.

## 3. E0 — diagnóstico de ponderação, sinal e gradientes

Na preparação **completa do fold 1**:

| Partição | Janelas | Tiles de até 3 canais | Fração das janelas | Fração das células |
|---|---:|---:|---:|---:|
| Treino | 14.620 | 1.120 | 7,66% | 0,139% |
| Validação | 6.940 | 1.120 | 16,14% | 0,484% |

Isso confirma o desalinhamento entre atualização por janela e ponderação por
células na avaliação, mas não prova que uma troca de sampler melhorará o modelo.

Em oito exemplos de treino estratificados por família e tamanho, sem qualquer
atualização de parâmetros, as normas de gradiente variaram substancialmente.
Por exemplo, no multimode, um tile de dois canais apresentou norma L2 de decoder
**0,578**, contra **0,113** no exemplo de 512 canais. São observações descritivas,
não a contribuição média desses estratos ao treinamento inteiro.

Nos recortes de treino de 50 ms, as medianas descritivas não ponderadas foram:
similaridade temporal no lag 1 **0,096**, no lag 10 **−0,00063**, similaridade
espacial adjacente **0,621**, e fração de potência em 5–10 kHz **0,441**. As
medianas das frações por banda não precisam somar um. Esses números não estabelecem
independência, gaussianidade ou um limite informacional e não autorizam descartar
altas frequências.

Artefatos: `representation_e0_e1_fold1_20261031/signal_diagnostics.json` e
`tile_gradient_audit_fold1_20261031/results.json`, em `results/experiments/`.
Ainda faltam diagnóstico contínuo em HDF5, quantis pooled por arquivo e comparação
causal de ponderação; o sampler histórico foi preservado nos primeiros treinos E2.

## 4. E1 — busca de códigos com pesos congelados

### Método

- Um recorte por arquivo/região retida, definido por metadados, seed 2078.
- Busca com Adam em vetores contínuos auxiliares, **48 passos**. A cada quatro
  passos, projeção ao codebook e avaliação da reconstrução de embeddings exatos.
- Duas taxas de busca, `0.01` e `0.05`, comparadas somente em exemplos de treino;
  o protocolo registra os exemplos, inclusive eventuais duplicações de extremo
  mínimo/máximo no mesmo estrato. A validação não escolhe a taxa de busca.
- Encoder, decoder e codebook não recebem atualizações. O candidato inicial é
  sempre elegível. Nenhum latente contínuo é enviado ao decoder.
- Primeiro controle: mesmo número de índices a 11 bits. Segundo controle:
  teto **por janela em bytes efetivos**, com marginal/range congelado e fallback
  Huffman/fixo, tabelas ajustadas exclusivamente no treino. Uma alternativa de
  projeção prioriza trocas de símbolos com menor custo marginal.
- As tabelas e os mapas são os mesmos nos dois lados da comparação. Não se afirma
  que esse coder diagnóstico seja o melhor stream histórico dos estudos 75–76.

### Resultados concluídos

| Modelo/coorte | Janelas | RMSE inicial → busca | Redução relativa | Correlação inicial → busca | Bytes dos frames inicial → busca |
|---|---:|---:|---:|---:|---:|
| Scratch 28M, fold 1, 11 bits fixos | 63 | 0,68230 → 0,64037 | 6,14% | 0,71900 → 0,75224 | Mesmo número de índices; não é controle entrópico |
| Transferido compacto, fold 0, teto real | 65 | 0,66550 → 0,63181 | 5,06% | 0,72948 → 0,75620 | 139.403 → 139.192 |
| Transferido compacto, fold 1, teto real | 63 | 0,64699 → 0,61451 | 5,02% | 0,75076 → 0,77422 | 96.763 → 96.468 |

O teto de bytes foi respeitado em **cada janela**, não apenas na soma. Os streams
recuperaram exatamente os índices, side e shapes; a reconstrução foi verificada
por replay dos embeddings discretos. Dicionários compartilhados são custo
adicional: no piloto do fold 1 somam 115.612 bytes, iguais para inicial e busca.
Não converter os frames isolados em uma alegação de razão completa de compressão.

**Limitação importante:** o p90 por canal piorou em 19/65 janelas do fold 0 e
13/63 do fold 1. No fold 1, a média ponderada desse p90 em single-mode 11 passou
de **1,3544 para 1,3786**, apesar do ganho de RMSE. No fold 0, multimode contém
denominadores de variância muito pequenos, produzindo p90 da ordem de milhões;
eles foram mantidos e não devem ser interpretados sem o erro físico absoluto.
Não houve piora de RMSE ou correlação por janela nesses dois pilotos, mas isso
é observação do experimento, não garantia geral da busca.

### Análise posterior com guarda explícita

Uma análise **post-hoc**, não confirmação independente, conserva o frame original
quando a busca piora RMSE, correlação ou p90 por canal:

| Coorte | Janelas que usam busca | Redução de RMSE macro | Frames selecionados / originais |
|---|---:|---:|---:|
| Fold 0 | 46/65 | 3,39% | 139.261 / 139.403 bytes |
| Fold 1 | 50/63 | 3,23% | 96.533 / 96.763 bytes |

A decisão é implementável no encoder, que possui o original; o decoder usa apenas
o frame selecionado. O resultado com guarda **não** atinge o alvo exploratório de
5% de ganho, embora mostre oportunidade consistente nas duas coortes. Não promover
a variante sem guarda por ter ultrapassado 5% na média.

O tempo acumulado de busca e avaliação das janelas foi aproximadamente 56,6 s no
fold 0 e 34,5 s no fold 1. Houve concorrência com outros trabalhos na mesma GPU;
isso não é benchmark isolado de latência, nem demonstra viabilidade em tempo real.

### Verificação adicional com guarda predefinida

Depois da análise anterior, o fold 1 foi repetido com seed de validação **2079**,
exclusão explícita de todas as 63 janelas anteriores e a guarda declarada no
protocolo **antes** da inferência. Calibração e tabelas continuam ajustadas só
no treino. Nas 63 novas janelas, 14 usaram fallback:

- RMSE macro: **0,64547 → 0,62095**, redução de **3,80%**;
- correlação: **0,75185 → 0,77040**;
- p90 macro por canal: **1,02065 → 0,98487**;
- frames: **97.031 → 96.801 bytes**, sem ultrapassar nenhum teto por janela;
- nenhuma regressão por janela nas três métricas guardadas.

Isso confirma o comportamento da política em recortes adicionais, mas não em
novas aquisições: são outros instantes dos mesmos arquivos/configurações de
validação. Continua abaixo do alvo exploratório de 5% com guarda e não justifica
promover automaticamente outro encoder ou uma campanha de destilação longa.
Artefatos: `representation_e1_guarded_fresh_fold1_20261031/`.

Artefatos: `representation_e0_e1_fold1_20261031/` e
`representation_e1_wire_transfer_fold{0,1}_20261031/`. Estes últimos preservam
pares de frames, tabelas, registros, calibração e `guarded_results.json`.

## 5. E2 — codec escalar com aprendizado conjunto

### Implementação e contrato

- Transformadas residuais nativas, largura-base 32, aproximadamente 6,8M
  parâmetros, geometria `spatial-4x`; sem skips contínuos gratuitos.
- Quantização escalar, prior logístico fatorizado de escala aprendida por canal.
  Treino com ruído uniforme `[-0.5, 0.5]`; avaliação com arredondamento exato.
- `L = 0.5*MSE + 0.5*L1 + lambda*R`, com erro nas células ordinárias e `R` em
  bits estimados por célula retida. O gradiente de taxa até o encoder foi medido
  no smoke de GPU e coberto por teste unitário.
- `robust-escape-v1`, preservação bit-exata do side, todas as amostras/bandas
  ordinárias, mesmas partições e sampler por janela da referência.
- Range coder com tabelas transmitidas, CRC de frame e símbolos. Símbolos fora
  de `[-255,255]` usam escape int32 explícito, **sem clipping**. Todos os
  coeficientes necessários à síntese são efetivamente codificados.
- Bytes completos incluem frames, prior, mapas e índice de transporte. Pesos
  compartilhados são declarados separadamente. O índice JSON é deliberadamente
  explícito e tem overhead maior que transportes históricos mais compactos.
- Seleção por menor RMSE macro no piloto fixo de 63 janelas; avaliação completa
  das 6.940 janelas de validação ao final, sem inferência de teste.
- Checkpoints atômicos, estados de otimizador/RNG, orçamento delimitado e uma
  única extensão condicionada a melhoria relativa de pelo menos 0,5% no último
  bloco do piloto. Nenhuma extensão é acionada só por perder para o VQ.

### Primeira triagem concluída

16 canais latentes, AdamW a `1e-4` para transformadas e prior, seed 2078,
4.096 passos iniciais e extensão predefinida até **8.192** em ambos os braços:

| Lambda | Passo selecionado | RMSE macro | Correlação | Bits/célula: estimados / frames / completos | Razão completa |
|---|---:|---:|---:|---:|---:|
| 0,05 | 7.168 | 0,48117 | 0,87191 | 0,78488 / 0,83777 / 0,90610 | 35,32× |
| 0,5 | 8.192 | 0,75559 | 0,64188 | 0,38069 / 0,43362 / 0,50195 | 63,75× |

Cada avaliação completa contém **462.380.000 células**. Cada braço acrescenta
3.949.324 bytes de tabelas/mapas/transporte aos frames. Os tensores compartilhados
do codec ocupam 27.227.652 bytes, antes do overhead do arquivo de checkpoint.
Não atribuir a diferença entre bits estimados e pacote completo exclusivamente
à calibração probabilística: ela inclui side, framing e transporte.

**Decisão:** a primeira triagem confirma implementação/gradiente/stream, não
superioridade à mesma taxa. O primeiro ponto usa mais bytes que o denso histórico;
o segundo ainda perde fidelidade para referências compactas mais baratas. A
exposição é muito menor que nos estudos VQ anteriores, e o contraste também muda
inicialização e porte da rede; não é um teste causal puro da quantização.

### Segunda triagem concluída

Para investigar sobreposição de taxas e convergência, sem ampliar a CNN:

- **C4 / lambda 0,05:** quatro canais latentes.
- **C16 / lambda 0,2:** 16 canais latentes, penalidade intermediária.
- Ambos: transformadas a `1e-4`, prior a `1e-3`; 16.384 passos iniciais, extensão
  única até 32.768, blocos de 4.096 e o mesmo critério de 0,5%.
- São escolhas de triagem motivadas pela primeira curva, não uma comparação causal
  isolada entre elas: dimensão, lambda e otimização não devem ser confundidos.

| Braço | Passos concluídos / selecionado | RMSE macro | Correlação | Bits/célula completos | Razão completa |
|---|---:|---:|---:|---:|---:|
| C4 / lambda 0,05 | 32.768 / 28.672 | 0,66329 | 0,73778 | 0,39365 | 81,29× |
| C16 / lambda 0,2 | 16.384 / 8.192 | 1,00000 | 0,000065 | 0,12160 | 263,16× |
| VQ compacto transferido, transporte comum | Pesos congelados | 0,64819 | 0,74932 | 0,26084 | 122,68× |

O C4 obteve um ponto menos caro que o primeiro C16, mas ainda é dominado pela
referência VQ compacta nessa contabilidade: mais bytes e pior RMSE/correlação.
O C16 com prior rápido **colapsou**, com taxa estimada de apenas 0,00087 bit/célula
e correlação praticamente nula. O critério encerrou esse braço em 16.384 passos,
sem extensão. Sua razão de 263× não representa uma solução de alta fidelidade.
O custo de framing/mapas/transporte permanece mesmo quando os símbolos colapsam.

A referência VQ compacta usa exatamente as mesmas 462.380.000 células e esquema
explícito de mapas/índice JSON. Seus frames ocupam 11.043.470 bytes, acrescidos de
4.032.168 bytes de tabelas/mapas/transporte. O aumento de overhead frente aos
streams históricos explica por que os 122,68× daqui não substituem os ~166× do
melhor empacotamento anterior. A referência VQ densa, também concluída, obteve
**0,41676 de RMSE / 0,90484 de correlação**, a **0,78996 bit/célula / 40,51×**,
com 45.657.965 bytes completos. Ambas reproduzem as métricas históricas dos
respectivos checkpoints, sem mudar pesos ou símbolos para favorecer o controle.

### Continuação controlada a partir de representação funcional

O colapso motivou um teste de otimização, não outro aumento da CNN: dois braços
partem do **mesmo checkpoint C16 funcional**, selecionado no passo 7.168 da
primeira triagem, com fresh optimizer/RNG e budgets iguais:

- lambda `0.05` como controle de exposição;
- lambda `0.2` para a negociação mais forte de taxa;
- prior a `1e-3`, transformadas a `1e-4`, 16.384 passos iniciais, uma extensão
  até 32.768 pelo mesmo critério, blocos de 4.096.

O pré-treino de 7.168 passos é compartilhado e deve ser somado à exposição de
cada descendente. Os pesos não vêm da referência VQ. Ambos acionaram a extensão
e concluíram **32.768 passos novos**, portanto a exposição final é pareada:

| Continuação | Passo novo selecionado | RMSE macro | Correlação | Bits/célula completos | Razão completa |
|---|---:|---:|---:|---:|---:|
| Lambda 0,05, controle | 28.672 | 0,42460 | 0,90257 | 0,93280 | 34,31× |
| Lambda 0,2 | 24.576 | 0,47846 | 0,87558 | 0,61146 | 52,33× |

A inicialização funcional evitou o colapso nesta execução. A penalidade maior
produziu uma negociação efetiva: cerca de **34,45% menos bytes** que a continuação
0,05, pagando **12,68% mais RMSE**. Isso confirma o mecanismo de taxa–distorção,
mas não confirma vantagem sobre o VQ. A continuação 0,05 continua mais cara e
ligeiramente pior em RMSE que o VQ denso.

### Comparação de pontos intermediários a teto de bytes comuns

Para não comparar o novo ponto intermediário apenas com o compacto ou o denso,
foi construído um controle de seleção de modo por tile/janela. Ele usa os frames
VQ completos já verificados, seus dois dicionários e o mesmo mapa/transporte;
pesos de waveform compartilhados ficam fora da soma, declarando que o controle
usa dois modelos. O ganho de RMSE macro por byte adicional define uma seleção
greedy fixa; uma variante exige também correlação e p90 não piores por janela.
O encoder pode medir a distorção contra o original; o decoder identifica o modelo
no próprio frame, sem precisar do alvo.

O orçamento é da **coorte completa**, não um teto por arquivo nem uma política
online. A seleção não é afirmada como ótima global. Este é um controle de
desenvolvimento explícito, não uma nova validação externa.

| Ponto escalar | Teto em bytes | RMSE escalar | RMSE VQ alocado | RMSE VQ alocado com guarda |
|---|---:|---:|---:|---:|
| C16 inicial, lambda 0,5 | 29.011.462 | 0,75559 | 0,50585 | 0,51008 |
| C4, lambda 0,05 | 22.751.881 | 0,66329 | 0,55554 | 0,55952 |
| C16 funcional, lambda 0,2 | 35.340.883 | 0,47846 | 0,46538 | 0,46989 |

No último ponto, o VQ com guarda utiliza **35.340.741 bytes** e tem p90 macro
**0,78369**, contra **0,86006** do escalar. O escalar, por outro lado, tem correlação
macro maior: **0,87558**, contra **0,86028**. Há um trade-off de métricas, não uma
dominância universal. Não relaxar retrospectivamente RMSE/caudas para chamá-lo
de vencedor. Nos dois pontos escalares acima do custo denso, o próprio VQ denso
já tem RMSE melhor e usa menos bytes.

Artefatos: `representation_compact_reference_fold1_20261031/`,
`representation_dense_reference_fold1_20261031/` e
`representation_rate_comparison_fold1_20261031/`. A lista de modos e os bytes
selecionados são preservados; não se trata de interpolar scores entre modelos.

**Decisão ao fechar a campanha:** nenhum braço E2 atingiu o alvo de investimento
predefinido frente aos controles reais. Não lançar hyperprior, ampliar a rede ou
prolongar esses braços automaticamente. O resultado é negativo para a promoção
destes pilotos, não para todo aprendizado conjunto: backbone, inicialização e
exposição histórica ainda diferem do VQ. Um próximo protocolo deve isolar uma
hipótese, como ponderação/convergência ou VQ com gradiente de taxa, antes de outra
campanha longa. A condição favorável de correlação do escalar intermediário pode
motivar análise por canal/configuração, sem dispensar as guardas concorrentes.

## 6. Verificações, reprodução e próximos critérios

Implementações novas:

- `finetuning/representation_study.py`: I/O sem acesso a teste e métricas comuns.
- `finetuning/run_index_search.py`: E0/E1 e teto de bytes.
- `finetuning/audit_tile_gradients.py`: normas de gradiente sem updates.
- `finetuning/summarize_index_search.py`: fallback explícito, marcado post-hoc.
- `finetuning/scalar_rd.py` e `compression/scalar_stream.py`: modelo/stream E2.
- `finetuning/run_scalar_rd_study.py`: piloto, extensão e avaliação completa.
- `finetuning/evaluate_representation_reference.py`: VQ com contabilidade comum.
- `finetuning/compare_representation_rates.py`: modos VQ sob teto real por coorte.
- `tools/decode_scalar_bundle.py`: decodificação sem NPZ/HDF5 original.

O codec escalar é experimental e usa seus próprios runner/decoder; não substitui
silenciosamente o VQ nos alvos de inferência existentes do `Makefile`.

A suíte local passou em **256 testes**; no remoto, 255 passaram e um foi ignorado
pela condição de ambiente do teste. O decoder independente foi executado em
CPU para três frames do primeiro braço escalar e três de cada continuação
funcional, usando apenas checkpoint, stream, prior, mapas e transporte. As avaliações em GPU verificaram replay exato de todos
os coeficientes e da reconstrução; igualdade bit a bit entre CPU e GPU não é
presumida. Corrupção, escapes e overflow de inteiro têm testes próprios.
Conferidos também 28 hashes dos arquivos de snapshots E1/E2 contra os protocolos.

Exemplo de reprodução no remoto, com novo diretório de saída:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python finetuning/run_scalar_rd_study.py \
  --samples-dir data/finetune/spatial_band_v1_fold1_20261026 \
  --repo-root vendor/smartgrids \
  --reference-checkpoint results/remote_setup/compact_transfer_fold1_weights.pt \
  --rate-weight 0.05 --latent-channels 4 --prior-learning-rate 0.001 \
  --initial-steps 16384 --maximum-steps 32768 --block-steps 4096 \
  --output-dir results/experiments/NEW_SCALAR_RUN
```

Os `protocol.json` e snapshots de cada execução são as referências da versão
efetivamente executada, não a suposição de que o arquivo de código atual é idêntico
ao da primeira triagem. A supervisão SSH inicial teve timeout de 180 s durante a
validação; os processos remotos continuaram e concluíram os artefatos. O helper
foi corrigido para não abortar por esse limite; novas séries preservam logs via
`tee`. Não houve reinício oculto ou updates adicionais nos dois primeiros braços.

Checkpoints e streams escalares completos permanecem no remoto. Relatórios,
protocolos, métricas e pares de streams E1 concluídos são copiados para os mesmos
caminhos locais em `results/experiments/`; dados/pesos continuam fora do Git.

**Próximas decisões:** manter os controles VQ, preservar E1 como oportunidade de
codificação e predefinir uma próxima ablação E2 isolada. Exigir avanço real com
guardas por configuração antes de hyperprior, maior largura ou campanhas longas.
Para E1, ampliar a verificação de caudas e avaliar
amortização/destilação do encoder somente com validação independente. A repetição
com guarda predefinida já cobre novos recortes, mas não novas aquisições.
Nenhum resultado aqui demonstra fidelidade suficiente
a 700× ou justifica usar o conjunto de teste para escolher candidatos.
