# Prior neural lossless para índices congelados

**Estudo concluído:** economia líquida insuficiente; confirmação de fold 1 e
extensão não foram executadas. Resultado no
[relatório 75](75-NEURAL_INDEX_PRIOR_NEGATIVE_RESULT.md).

## Hipótese e escopo

Modelar a redundância espacial/temporal dos **índices**, em vez de treinar outro
reconstrutor. Nenhum peso do codec DAS, codebook, máscara, normalização, índice
ou alvo é alterado. A reconstrução deve permanecer idêntica no nível de códigos
e side; o ganho pretendido é exclusivamente no número real de bytes.

Essa linha converge com modelos de contexto/entropia da literatura, mas não
extrapola seus percentuais de imagens/áudio para FEBUS. Não é denoising nem um
produto clássico alternativo; é uma camada lossless de um codec neural.

## Prior e causalidade de decodificação

- `CheckerboardPrior`: embedding 2.049×16 (um token reservado de máscara), duas
  convoluções 3×3 de 48 features e saída 1×1 para 2.048 símbolos; **160.880
  parâmetros**, não outra rede de 100M. Sem dropout, GAN ou loss perceptual.
- Duas passagens: anchors em paridade espacial/temporal par, transmitidos por
  marginal; depois os demais índices condicionados somente aos anchors.
- Não-anchors são substituídos pelo token de máscara **antes** de qualquer
  embedding/convolução. Teste altera todos os não-anchors e exige predição idêntica.
  O decoder dispõe de toda a informação condicionante após decodificar anchors.
- Rede começa na marginal de treino (head zero); aprendizado por cross-entropy
  dos não-anchors. Marginal ajustada em **todo o treino**, com uma pseudocontagem
  por símbolo. Não usar validação para frequências/CDF ou contexto.

## Dados e orçamento limitado

Primeiro, base compacta transferida do primeiro estudo espacial, **fold 0**, mesmos
11.760 exemplos de treino e 7.220 de validação, seed **2037**. Extrair índices de
treino do modelo congelado na RTX 4060; validar por índices dos frames existentes.
Não acessar teste. Registrar SHA do NPZ de origem, do checkpoint, das tabelas e
dos metadados; guardar só grids discretos pequenos, sem duplicar HDF5.

- Treino do prior na **RTX 4060 local**, batch de uma janela, FP32, AdamW LR 1e-3,
  weight decay 1e-5, clipping 1. Sem treino remoto ou adaptação do codec DAS.
- **8×512 updates**; extensão única de oito épocas se melhora recente entre os
  últimos blocos de duas épocas for ≥0,5% e o proxy de economia total for ≥5%.
  Teto 8.192 updates. O proxy assume que só metade dos símbolos usa contexto;
  não é tamanho real e não autoriza promoção.
- Seleção: menor cross-entropy por símbolo em triagem fixa de até duas janelas
  por arquivo/região. Não escolher checkpoint pela melhor reconstrução do alvo.
- Depois, codificação completa de todos os frames da validação.
- **Confirmar fold 1/seed 2038 somente se a economia medida do stream completo
  for ≥2% contra o melhor controle não neural da camada de entropia**, já incluindo
  o prior. Modelos específicos de cada checkpoint; não reutilizar um prior entre
  codebooks diferentes sem validação.

## Streams e controles

Backend opcional: **constriction 0.5.0**, fixado em `requirements-entropy.txt`.
API consultada:
[RangeEncoder/RangeDecoder](https://bamler-lab.github.io/constriction/apidoc/python/stream/queue.html),
[Categorical](https://bamler-lab.github.io/constriction/apidoc/python/stream/model.html).
Usar `perfect=False` explicitamente, palavras little-endian. O backend garante
massa não nula a todos os 2.048 símbolos de seu domínio.

Para cada janela comparar bytes reais de:
1. Payload Huffman/fixo/contexto simples existente, sem alteração.
2. Range coding somente com marginal completa do treino.
3. Range coding de anchors marginais e não-anchors pelo prior neural.

Selecionar o menor payload, com identificador explícito. Não atribuir ao prior
o benefício de atualizar a marginal ou trocar Huffman por range coding. Há um
controle que escolhe o melhor método **sem** usar a rede contextual.

Frames `FNP1` carregam geometria, método, ID do modelo, side original, CRC de
header/payload e CRC dos índices nos métodos range. A predição é computada em
**CPU FP32 com uma thread nos dois lados**, a partir somente dos anchors
decodificados. Nenhuma tabela por símbolo prevista pelo original é enviada
gratuitamente ou consultada no decoder.

Todas as sequências e side streams devem ser recuperados exatamente. Bundles
incluem mapas/metadados, tabelas, marginal, descritor e **pesos do prior uma vez**
quando usados. Pesos do reconstrutor permanecem compartilhados e seu custo é
mostrado separadamente. Esses custos não podem ser omitidos do resultado final.

**Limitação de implantação:** a reproducibilidade das probabilidades entre
softwares/CPUs distintos ainda não é garantida. CRC dos índices faz o decoder
falhar se houver divergência, em vez de retornar DAS silenciosamente diferente.
Portabilidade/integerização são trabalho posterior obrigatório; não declarar
codec de produção pelo roundtrip nesta máquina.

## Execução e decisão

Runner: `finetuning/run_neural_index_prior.py`.
Diretório: `results/experiments/neural_index_prior_20261030/`.
Unidade: `febus-neural-index-prior-20261030.service`.

Protocolo e snapshot fixam fontes/software. Cache de índices permite reaproveitar
extração; checkpoints de treino registram optimizer/RNG e melhor estado. O runner
recusa mudanças de protocolo ao retomar. Eventual artefato parcial de transporte
precisa ser verificado antes de sobrescrever; não fazer limpeza ampla.

Se a economia líquida não atingir a regra, encerrar a hipótese sem expandir
automaticamente rede/dados/epochs. Mesmo economia positiva preserva a fidelidade
**existente**, não resolve seus limites físicos nem promove um modelo final.

### Início e verificação do receptor

Experimento iniciado na RTX 4060 local. O primeiro prior já passou ao roundtrip
da validação completa; a confirmação do segundo fold depende exclusivamente da
regra de economia líquida, sem intervenção para forçar avanço. Suite:
**245 testes aprovados**, incluindo os três métodos, símbolo raro, corrupção de
header, não acesso a não-anchors e controle que decodifica sem carregar a rede.

`tools/decode_neural_index_bundle.py` implementa receptor independente: recebe
bundle e checkpoint de forma de onda, recupera tabelas/prior/índices/metadados e
produz DAS sem entrada do NPZ original. Um smoke com frame neural forçado em
70×1.000 passou em CPU, com SHA dos índices idêntico e mapa físico recuperado.
Outro teste instalou um audit hook que **proíbe abrir arquivos sob `data/`**;
o receptor continuou funcionando. Não é resultado de bitrate do prior aprendido:
o smoke usa uma rede aleatória só para exercitar a via neural e a independência
do alvo.
