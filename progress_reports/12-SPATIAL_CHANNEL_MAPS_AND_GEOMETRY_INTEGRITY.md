# Rastreabilidade espacial: mapas de canais e integridade da preparação

## Conclusão e consequência para a pesquisa

Os **1.194 blocos preparados** do corpus têm linhas espaciais contíguas no recorte
final: **nenhum gap interno entre os canais realmente usados no modelo**.
A auditoria anterior identificou 138 instâncias de canais internos removidos nos
HDF5 de seis arquivos single-mode Configuration 8, mas essas lacunas estão fora
do recorte central de 512 canais. Não atribuir a baixa fidelidade dos experimentos
com esses recortes à concatenação de canais separados por essas lacunas.

Isso refina o diagnóstico do relatório 7, sem invalidar suas contagens no sinal
bruto. Geometria física diferente entre configurações, corte espacial e contexto
temporal curto continuam hipóteses; o gap interno dos recortes existentes não.

Foi recuperado um mapa rastreável para cada bloco, com ligação às 23.880 coordenadas
de janelas. Dados/modelos não foram alterados. A preparação futura também registra
os mapas sem mudar os valores ou a seleção dos canais.

## 1. Por que executar esta etapa

Enquanto o piloto de normalização por canal utiliza a GPU, tratar um requisito
independente da etapa 2 do roadmap: rastreabilidade da seleção espacial.
Essa atividade usa CPU e HDF5 em leitura; não lança outro treino concorrente,
não modifica loaders/objetivos do treino em execução e não migra/recria datasets.

Uma máscara por finitude e um recorte central podem alterar o índice inicial
de cada matriz ou ocultar lacunas. Contagem de linhas e `dx` não bastam para
recolocar uma reconstrução no array de origem ou validar a abertura física.

## 2. Método e resultados

Ferramenta: `finetuning/audit_channel_maps.py`.
Saída: `results/experiments/channel_maps_audit_20261008/`.

- Leitura dos 207 datasets brutos em chunks, sem carregar o corpus inteiro em RAM.
- Reprodução da máscara: canal finito em **todo o bloco** após conversão float32,
  consistente com a preparação original.
- Reprodução do crop central de até 512 canais sobre a sequência de canais finitos.
- Registro dos índices locais originais, índices derivados de Extent, posições
  relativas e coordenadas derivadas de metadados quando disponíveis.
- Verificação contra NPZ antigo: uma janela de menor início temporal por bloco,
  cópia canônica fold 0, **1.194 arrays bit-exatos** à leitura raw com o mapa recuperado.
- Nenhum dos 1.194 mapas tem diferença entre índices consecutivos maior que 1.
- Todos têm coordenadas deriváveis pelos atributos disponíveis. Isso não significa
  calibração externa do interrogador ou confirmação independente da unidade espacial.

Nos seis arquivos com gaps do sinal bruto: 551 canais, 545 finitos por bloco,
seleção final de índices locais **16–527**, contíguos. Esse recorte não inclui
os canais removidos responsáveis pelos gaps nos extremos da sequência finita.

A verificação não compara todas as 23.880 janelas nem todas as quatro cópias por
fold. O mapa é por bloco, porque a seleção usa a finitude do bloco inteiro;
as ligações de coordenadas não constituem avaliações de modelo/teste. Não usar
essa auditoria como evidência de generalização.

## 3. Formato e uso dos mapas

`channel_maps.jsonl`: uma entrada por bloco, com schema_version e identificador
SHA-256 do conteúdo normalizado. Contém:

- Drive ID, dataset e block_index;
- número original de canais e lista ordenada dos índices selecionados;
- contiguidade, maior passo e contagem de linhas ausentes no intervalo selecionado;
- `dx`, índices de Extent, offsets relativos e posições derivadas dos metadados.

`window_maps.csv` liga cada coordenada canônica à entrada correspondente, permitindo
consultar mapas das amostras existentes **sem regravar seus metadata_json**.
`blocks.csv` resume verificações e contiguidade; `summary.json` registra o escopo.
Esses artefatos somam aproximadamente **8,7 MB**, sem duplicar os sinais.

Coordenadas derivadas seguem a hipótese explícita da pipeline:
`Origin[0] + (Extent[0] + índice local)*Spacing[0]`, assumindo unidade espacial metro.
Somente são fornecidas quando os atributos são finitos e Extent corresponde à
dimensão espacial. Se a hipótese não puder ser validada pelos metadados, preservam-se
os índices locais e offsets relativos, sem inventar uma origem absoluta.

O mapa fornece rastreabilidade, não preenche canais removidos nem reconstrói as
regiões espaciais cortadas. O custo de mapas/container deve integrar a avaliação
futura de bitrate; o sidecar JSON não é formato otimizado de codec.

## 4. Correção na preparação de novos dados

`finetuning/channel_geometry.py` centraliza seleção e construção do mapa.
`prepare_finetune_samples.py` mantém a máscara/crop anteriores e acrescenta:

- `channel_map_id`, versão e contiguidade nos metadata_json e manifestos;
- sidecar por bloco `channel_maps.jsonl`, compartilhável entre splits/folds;
- validação de parâmetros positivos;
- recusa de diretório de saída existente e não vazio, para evitar sobrescrita de
  arquivos locais. Novas preparações exigem destino novo/vazio.

Nenhuma preparação completa nova foi executada; a integração foi verificada com
HDF5 sintético pequeno. Não substituir os diretórios usados pelo treino atual.
Para aproveitar os mapas antigos, consultar os sidecars recuperados por ID/dataset/
bloco; não exigir o novo campo em NPZ antigos que não o possuem.

## 5. Validação e decisão de rota

**86 testes passaram**, incluindo equivalência com a seleção antiga, lacunas
sintéticas, crop de 1.012 para 512 canais, parâmetros espaciais ausentes/inconsistentes,
map IDs estáveis, verificação HDF5→NPZ e recusa de sobrescrita.

Decisão: manter o piloto de normalização pareado como próxima evidência de decisão.
Não iniciar outro treino na GPU antes de interpretá-lo. Se a mudança de normalização
não justificar o ganho raw e overhead, investigar duração/contexto físico, cobertura
espacial ou rate-distortion, em vez de corrigir uma falsa hipótese de gap no recorte.
Novas aquisições e modelos especializados continuam alternativas condicionais.

Esta etapa fecha parte da lacuna de rastreabilidade, mas não implementa store canônico,
codec completo, tiling espacial ou auditoria externa. Dados brutos/preparados,
checkpoints e o processo de treino foram preservados.
