# Preparação prospectiva e oportunidade espacial com contexto decodificado

**Campanha de 14/11/2026; operações remotas em 15/11 UTC.**
Implementação das etapas P0 e diagnóstico inicial P1 do [relatório 90](90-FIDELITY_PER_BIT_NEXT_RESEARCH_REVIEW.md).
Nenhum resultado histórico foi reclassificado. Não houve acesso a waveforms
de validação/teste ou EDAS reservado.

## 1. P0: contrato novo, separado dos seletores históricos

Implementações:

- `finetuning/prospective_metrics.py`: estatísticas agregáveis por canal físico,
  SSE/energia/contagem, variância/covariância centradas estáveis, RMSE absoluto,
  NRMSE*, viés, máximo absoluto e cobertura de correlação.
- `finetuning/audit_prospective_preparation.py`: inventário completo do treino,
  identidade de escapes, erro do roundtrip, calibração e atributos físicos
  dos HDF5, sem refazer os recortes ou modificar os brutos.

O denominador prospectivo é
`max(std pooled do alvo, escala de referência do grupo)`.
A escala é **1% da mediana dos desvios positivos por janela de treino**, congelada
por configuração. É uma escala numérica de desenvolvimento, **não tolerância
física validada**. Conversão coerente de unidades conserva NRMSE* e correlação.
Correlação é indefinida quando a variância de alvo ou reconstrução não supera
o quadrado dessa escala; a cobertura fica explícita.
Para configurações ainda não vistas, a mediana das escalas de grupos de treino
da mesma família é congelada como fallback, **antes de abrir sinais de avaliação**.
Não se ajusta um novo piso com base no desempenho na validação.

A agregação prevista é: estatísticas primeiro por canal físico/arquivo,
macro entre canais, arquivos, configurações e finalmente quatro famílias.
Mapas originais identificam os canais; não concatenamos regiões desconexas.
Os resultados cobrem os recortes armazenados, não toda a aquisição contínua.

### Auditoria completa do treino

| Quantidade | Resultado |
|---|---:|
| Janelas | 11.760 |
| Células float32 | 915.440.000 |
| Células escapadas nesta preparação | 0 |
| Janelas constantes bit a bit | 460 |
| Ocorrências de canais constantes | 4.600 |
| Side FNE1 total | 470.400 B |
| RMSE absoluto do roundtrip sem rede | 0,00984163 |
| Maior erro absoluto de roundtrip | 2 |
| Máximo, por janela não constante, de `erro máximo/std` | ~6,99e-6 |

As unidades são as armazenadas em cada produto; os números absolutos agregados
**não são uma tolerância universal**. O erro máximo 2 ocorre em sinais de grande
amplitude e deve ser lido com a escala física correspondente. Escapes são
bit-exatos por contrato/teste; não houve ocorrência deles neste corpus de treino.
Zero real continua sendo dado, não máscara de ausência.

Artefatos: `results/experiments/prospective_p0_fold0_20261114/`.
Contêm protocolo, inventário por janela com SHA-256, inventário por arquivo,
escalas e `physical_manifest.jsonl` com atributos de cada ancestral HDF5.
Unidades e parâmetros são preservados literalmente; não inferimos unidades
ausentes a partir de nomes ou da frequência do interrogador.
A versão `prospective_p0_v2_fold0_20261114/` acrescenta hashes dos mapas/métricas,
verificação explícita de shapes/contiguidade e o fallback de escala por família,
sem alterar os recortes nem as escalas por configuração. A primeira auditoria
permanece preservada.

Mapas/cobertura continuam os da preparação auditada do relatório 62. As máscaras
ordinária/escape são derivadas de FNE1; cobertura vem dos mapas; não há padding
nos NPZ. O novo lifting trata bordas e comprimentos ímpares explicitamente.
Manifestos e hashes, não arrays duplicados, versionam essa reutilização.

## 2. P1: previsão implementável, não residual original gratuito

Seleção determinística dentro do **treino**:

- 44 arquivos/132 janelas para ajustar um coeficiente por configuração;
- 43 outros arquivos/129 janelas para checagem;
- dez configurações e quatro famílias;
- três posições por arquivo no maior tile disponível: viés para tiles maiores;
- arquivos separados, mas configurações compartilhadas: não é validação externa.

O preditor usa a média dos canais-âncora vizinhos, multiplicada por um coeficiente
de mínimos quadrados ajustado apenas nos arquivos de ajuste. O coeficiente é
transmitido em cada frame. O caminho real quantiza e decodifica os canais pares
**antes** de calcular a inovação dos ímpares. A inovação também é transmitida.
O receptor não recebe vizinhos originais.

Controle: quantização escalar independente de todos os canais. Ambos usam
FNE1 robusto, mesma quantização, planos int32 comprimidos com zlib, shapes,
descritor e CRC. O custo inclui frames e setup comprimido dos mapas/offsets
somente dos arquivos transmitidos. Não se cobra o manifesto inteiro de pesquisa
como se fosse necessário para transmitir um subconjunto.

Implementações: `compression/research_coefficient_stream.py`,
`finetuning/spatial_opportunity.py` e `finetuning/run_spatial_opportunity.py`.

### Curva inicial: custo real e NRMSE* macro entre famílias

| Passo normalizado | Independente: bit/célula | Âncoras+inovação: bit/célula | Independente: NRMSE* | Âncoras+inovação: NRMSE* |
|---|---:|---:|---:|---:|
| 0,5 | 4,8050 | 4,4358 | 0,3272 | 0,3138 |
| 1 | 3,4923 | 3,0938 | 0,5653 | 0,5452 |
| 2 | 1,9402 | 1,6815 | 0,7856 | 0,7734 |
| 4 | 0,6701 | 0,5988 | 0,9315 | 0,9294 |
| 8 | 0,1895 | 0,1780 | 0,9788 | 0,9854 |

Mesma quantização **não significa mesma fidelidade** em malha fechada.
As reduções de bytes de ~7,7–13,3% nos quatro primeiros pontos são oportunidades
observadas nesse baseline, não economias certificadas a erro idêntico nem
superioridade sobre VQ maduro. No ponto ultracompacto há piora do erro.
O coder zlib é deliberadamente simples; não representa entropia ótima.

A referência otimista com âncoras originais teve razão energética da inovação
entre ~0,021 e ~0,851 conforme a configuração. Não é ganho de compressão nem
limite informacional. A curva de frames mostra que parte da dependência continua
explorável após pagar pelo contexto, mas não produz alta fidelidade a 0,2 bpc.

Artefatos:

- `results/experiments/spatial_opportunity_p1_fold0_20261114/`: execução inicial;
- `results/experiments/spatial_opportunity_p1_v2_fold0_20261114/`: reexecução com
  variância/covariância estáveis e hashes dos fontes; preserva a execução inicial.

Cada execução contém protocolo, coeficientes, setup, resultados por canal/arquivo
e frames reais para o passo 2. Os demais pontos foram codificados e decodificados
em memória, com seus tamanhos registrados.

## 3. Interpretação e próximo passo

O diagnóstico inicial sustenta executar **H1 fixa versus aprendida**, não promover
H2 como vencedora. Ainda falta o terceiro contexto do diagnóstico ampliado
(reconstrução VQ madura), estratificação espectral/física completa e confronto
com a fronteira prática. Esses itens não estão concluídos por esta tabela.

O [relatório 92](92-LEARNED_LIFTING_STUDY.md#report-92) registra o contraste
neural, o orçamento e os critérios antes dos resultados científicos.
P3 depende de deslocamento real da fronteira; P4 não implica obter novos dados
sem disponibilidade/licença, nem fabricar novas sessões independentes.

## 4. Reprodução

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m finetuning.audit_prospective_preparation \
  --corpus data/finetune/spatial_band_v1_fold0_20261024 \
  --output results/experiments/NEW_P0_DIRECTORY

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m finetuning.run_spatial_opportunity \
  --corpus data/finetune/spatial_band_v1_fold0_20261024 \
  --calibration results/experiments/prospective_p0_fold0_20261114/reference_scales.json \
  --output results/experiments/NEW_P1_DIRECTORY
```
