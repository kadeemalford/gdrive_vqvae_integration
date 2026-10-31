# Prior neural compacto: ganho de payload, mas perda líquida com pesos

`neural_index_prior_20261030` concluiu em **30/10/2026 às 20:01 UTC**, somente
no fold 0. Nenhum peso de reconstrução foi atualizado; os códigos, side streams,
metadados e 7.220 janelas passaram nos roundtrips.

| Camada de entropia | Bytes totais | Razão real, waveform model compartilhado |
|---|---:|---:|
| Huffman/contexto/fixo anterior | 14.787.049 | 171,994× |
| Melhor controle sem rede contextual | **14.711.822** | **172,873×** |
| Prior neural, incluindo pesos uma vez | 14.822.764 | 171,579× |

O prior fica **0,75% maior que o controle** e 0,24% maior que o stream anterior.
Logo, não é candidato admitido sob o contrato de custo definido. Não promover
pela redução isolada da cross-entropy ou do payload.

## Aprendizado e limite da conclusão

- Rede de 160.880 parâmetros, 4.096 updates/8 épocas; selecionada na época 8.
- Os bits condicionais de triagem chegaram a 8,974 por não-anchor. Houve melhora
  recente ≥0,5%, mas o proxy total foi **3,17%**, abaixo dos 5% exigidos para
  extensão; portanto o treino não foi prolongado.
- Método neural selecionado em 7.131 janelas, fallback em 89. Não basta vencer
  por janela quando há um novo prior global a transmitir.
- Os frames neurais somam **14.146.473 bytes**, contra **14.495.818** no controle:
  economia de 349.345 bytes, ~2,41% dos frames.
- O archive de metadados aumenta de 187.108 para 647.395 bytes: **460.287 bytes**
  extras, majoritariamente os pesos do prior. O arquivo de inferência tem 655.117
  bytes antes desse empacotamento.

Se uma operação maior mantivesse exatamente essa distribuição/taxa, uma conta
de amortização colocaria o break-even próximo de **9,5 mil janelas**, ou ~1,32×
o volume validado. Isso é **extrapolação contábil**, não experimento maior ou
garantia de generalização. Pré-instalar o prior junto ao reconstrutor também muda
o contrato de implantação. Não reinterpretar isso como aprovação do protocolo.

A regra de confirmação exigia ≥2% de economia líquida contra o controle e não
foi satisfeita. **Não houve treino no segundo fold**, aumento de rede ou extensão
retroativa. Portabilidade de probabilidades CPU entre plataformas também segue
pendente, embora CRC dos índices impeça divergência silenciosa.

## Direção seguinte

Encerrar essa variante neural de contexto. Antes de novos priors maiores, testar
um modelo estrutural pequeno: flags de repetição + símbolos literais nos eixos
temporal/espacial. O diagnóstico do **treino compacto**, sem usar validação para
ajuste, encontrou repetição em **4,26% das transições temporais** e **9,55% das
espaciais**. Isso motiva medir codificação literal com parâmetros minúsculos,
não promete redução líquida.

O próximo teste continua sendo apenas a camada lossless do codec **neural de
forma de onda**; não substitui o VQ-VAE por um compressor clássico de DAS. São
os mesmos checkpoints/dados, preservando fidelidade e mostrando custos completos.
Protocolo no [relatório 76](76-STATIC_INDEX_TRANSITION_PROTOCOL.md).
