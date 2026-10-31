# Codificação de índices e ROI por tiles — protocolo congelado local

**Estudo concluído:** 7.220 janelas codificadas e verificadas. Resultados e nova
ablação separada estão no [relatório 68](68-FROZEN_CODING_RESULTS_AND_TAIL_ROI.md).

## Decisão após a expansão de capacidade

A rede nativa de 96,5M reduziu o RMSE macro em 1,51% à mesma taxa compacta,
mas manteve regressões por configuração/canal. Não justifica ampliar novamente
a largura ou prolongar o orçamento automaticamente. As próximas perguntas são:

1. Quanto dos **11 bits fixos por índice** pode ser removido por codificação
   sem alterar nenhum índice ou sinal reconstruído?
2. Sob um teto de bytes por arquivo, é melhor selecionar tiles difíceis para
   o modo neural denso do que distribuí-lo sem olhar a dificuldade?

As duas etapas são executadas automaticamente pela mesma fila; nenhuma envolve
treino, PINN, denoising ou remoção adicional de canais. Se houver treino posterior,
ele utilizará a **RTX 4060 local**, conforme orientação do usuário.

## Referências, corpus e independência

Usar os checkpoints **transferidos** `compact_A/best_macro.pt` (época 24) e
`high_A/best_macro.pt` (época 23) de `spatial_abc_20261024_v2`, presentes localmente.
Ambos usam o mesmo corpus filtrado do **fold 0, validação fold 1**. É uma escolha
explícita de controles com a mesma coorte, não uma comparação direta com as redes
scratch de capacidade do fold 1. Não misturar checkpoints treinados nas
configurações agora em validação.

Corpus: `data/finetune/spatial_band_v1_fold0_20261024/`, 11.760 janelas de treino
e **7.220 de validação**, com `persistent-amplitude-v1` e representação
`robust-escape-v1` inalteradas. Shapes, canais e valores retidos não mudam.
Teste reservado não é inferido. A validação já foi usada no desenvolvimento;
não constitui teste externo ou nova evidência independente.

## Etapa 1 — Stream real e roundtrip

Implementação: `compression/symbol_stream.py`.

- Tabelas ajustadas somente em até **duas janelas por arquivo/região do treino**,
  seed 2034; nenhuma consulta à validação para frequências ou hiperparâmetros.
- Um código Huffman marginal e 17 tabelas condicionadas ao bucket do índice
  anterior (`previous_index // 128`, início 16), reiniciando em cada janela.
- Suavização aditiva de uma contagem garante suporte aos 2.048 símbolos.
- Selecionar o menor payload real entre 11 bits fixos, marginal e condicional;
  identificador do método é transmitido. Não confundir essa seleção lossless
  com mudança de quantização ou escolha de pesos pelo teste.
- Frame com versão, shape original e latente, SHA-256 da tabela/modelo,
  side stream completo de normalização/escapes e CRC. Pesos permanecem compartilhados.
- **Todas as janelas de validação são realmente codificadas e decodificadas**;
  índices e side stream devem ser idênticos. O decoder recebe apenas os índices
  recuperados e metadados codificados, sem bypass contínuo.
- Shapes máximos passaram em smoke de reconstrução exata a partir do stream:
  **8.064 índices compactos / 32.000 densos** em 512×1.000, usando GPU local.
- Reportar bytes de payload, headers, escapes, tabelas, mapas espaciais e metadados
  de aquisição. Mostrar separadamente custo dos pesos FP32 compartilhados uma vez.
  O stream não substitui a necessidade de implantar software/modelos compatíveis.

Essa é a camada lossless de um **codec neural**, não uma alternativa clássica
para reconstruir DAS. Uma eventual economia preserva exatamente os códigos,
não demonstra melhora de fidelidade ou adequação operacional do sinal existente.

## Etapa 2 — ROI por seleção de modo neural, não refinamento residual

Primeiro protótipo: decidir **compacto ou denso para cada tile existente**, sem
descartar regiões. Não é ainda a arquitetura progressiva base+resíduo da etapa
8B do roadmap; testa sua motivação de alocação antes de treinar outra rede.

- Base compacta para todo o arquivo; tetos de **1,5× e 2× seus bytes de frames**.
  Tabelas/mapas são cobrados no stream total. Orçamentos são tetos comuns, não
  promessa de igualdade exata de bytes usados. Comparar também as taxas efetivas.
- Controle `uniform`: ordem pseudoaleatória fixa por hash do ID da janela,
  independente do sinal/erro, preenchendo o orçamento com modo denso.
- ROI `rmse`: ordenar por redução positiva do RMSE normalizado ao quadrado,
  ponderada por células e dividida pelo custo adicional real de bytes. Alocação
  gulosa, sem alegar ótimo de knapsack.
- ROI `guarded`: mesma regra, mas admitir modo denso somente se correlação e p90
  ordinário por canal da janela não piorarem. Regra fixada antes de avaliar;
  sem relaxamento/tuning após observar grupos.
- O encoder possui o original e pode reconstruir os dois modos para medir erro;
  essa computação é executável em produção, embora custosa. O decoder não recebe
  o original, tabelas de erro ou seleção oracle de checkpoints: a identificação
  do modo já está no frame. Pesos de dois decoders e custo de codificar ambos
  devem ser explicitados.
- A proteção da métrica de cada janela implica proteção de sua agregação quando
  a cobertura/pesos são iguais; isso é **propriedade da regra**, não descoberta de
  generalização. Avaliar também amplitude/fase, canais não fracos e espectro,
  que não são garantidos pela admissão RMSE/correlação/p90.
- Mostrar configuração, arquivo, ROI/não ROI, fração de células densas e orçamento
  não utilizado. Uma regra conservadora pode não conseguir consumir todo o teto.
  Não forçar refinamentos prejudiciais para preencher bytes.

Esse protótipo não pressupõe que maior amplitude significa maior interesse, nem
depende de um detector de eventos. A granularidade inicial é a janela/tiling já
existente; bordas e continuidade entre modos continuam como requisito para um
codec operacional, não demonstradas só pelas médias de janelas.

## Execução e entregáveis

Runner: `finetuning/run_frozen_entropy_roi.py`.
Diretório: `results/experiments/frozen_entropy_roi_fold0_20261030/`.
Unidade local: `febus-frozen-entropy-roi-20261030.service`.

Protocolo fixa pesos, fontes, metadados, seed e regras. Fontes são arquivados em
`code_snapshot/`. Frame/cache por janela permitem retomar com hashes iguais.
Não alterar o código durante a execução; disk headroom mínimo de 8 GiB, sem
limpeza automática. Nenhum arquivo bruto é alterado.

Artefatos esperados: tabelas aprendidas, streams `.fsc`, cache de métricas,
`entropy.json`, resultados e CSVs por configuração, manifestos de modos,
`guards.json`, `REPORT.md` e `COMPLETE`. O catálogo de modos é redundante ao ID
incluído nos frames; mapas físicos e metadados necessários são cobrados.

Uma conclusão positiva deve separar: economia lossless, benefício de gastar
mais bits e eficiência da alocação ROI. Não promover um codec pelas guardas de
três métricas ou por erro relativo patológico de canais fracos.

### Decoder independente do alvo

`tools/decode_symbol_frame.py` recebe frame, tabela e checkpoint e reconstrói o
DAS sem receber o NPZ original. Confere hash do checkpoint, geometria, CRC e
restaura normalização/escapes; mapas físicos são side separado. Um frame real
70×1.000 foi decodificado em CPU por esse CLI, com shape e escapes verificados.
Este teste funcional não exige usar a GPU em paralelo à avaliação nem substitui
as verificações por janela do runner.

```bash
.venv/bin/python tools/decode_symbol_frame.py \
  --frame <janela.fsc> --table <modo.tables.json> \
  --checkpoint <checkpoint_do_modo.pt> \
  --repo-root /home/murilo-barroso/source/smartgrids \
  --device cpu --output <reconstrucao_nova.npz>
```

Coordenadas não são inventadas a partir do shape: devem acompanhar o frame no
catálogo de aquisições e nos mapas espaciais cobrados pelo estudo.
