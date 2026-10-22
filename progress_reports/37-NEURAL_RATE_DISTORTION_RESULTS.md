# Curva neural taxa-fidelidade: ganho forte com maior orçamento

## 1. Conclusão e população

O sweep foi concluído, inclusive avaliação completa. Fonte:
`results/experiments/neural_rate_sweep_20261014/`, protocolos, histories e
`full_validation/summary.csv`/tabelas por grupo/janela, `COMPLETE`.
Mesmas 5.820 janelas de validação do fold 1, quatro famílias/seis configurações.

**Mais índices melhoraram fortemente a fidelidade nas seis configurações**, com
melhora também na pior família e nos canais. Esse ganho é comprado com bitrate
maior e receptive field diferente, não superioridade à mesma taxa. Nenhum modelo
final promovido, nenhum score de teste ou auditoria externa independente.

## 2. Resultado completo

| Candidato | Epoch | RMSE macro | Correlação macro | Pior família RMSE | Compressão nominal pooled |
|---|---:|---:|---:|---:|---:|
| Spatial congelado | 23 anterior | 0,685863 | 0,709805 | 0,807557 | 164,624x |
| Rate 1x | 20 | 0,680527 | 0,717340 | 0,808439 | 164,624x |
| Rate 2x | 22 | 0,590393 | 0,796161 | 0,735150 | 83,826x |
| Rate 4x | 24 | **0,490010** | **0,865562** | **0,658187** | **42,133x** |

Ratio 2x/4x é aproximado: índices por posição 1 / 1,984 / 3,968 para T=1.000.
Normalização/side está incluída no nominal; container/modelo/mapas/entropy coding
não estão. Todos têm os mesmos 27.958.081 parâmetros e K=2.048.

| Configuração | RMSE 1x | RMSE 2x | RMSE 4x | Correlação 4x |
|---|---:|---:|---:|---:|
| Doped / 5m SMF | 0,645285 | 0,555410 | 0,439445 | 0,898225 |
| Multicore / C1 and C4 | 0,773093 | 0,654459 | 0,511434 | 0,857918 |
| Multimode / Configuration 7 | 0,808439 | 0,735150 | **0,658187** | **0,772880** |
| Single-mode / Configuration 11 | 0,410781 | 0,290951 | **0,247186** | **0,967869** |
| Single-mode / Configuration 3 | 0,546615 | 0,485150 | 0,416201 | 0,913436 |
| Single-mode / Configuration 7 | 0,528471 | 0,473552 | 0,389536 | 0,918366 |

Multimode continua limitante; alta correlação numa configuração single-mode não
estabelece fidelidade suficiente das demais. Para 4x, RMS ratio macro **0,917563**,
fase FFT **0,936199**, p90 relativo de canal **0,844228**, erro não fraco **0,611449**.
Multimode p90 ainda **1,000820**; Configuration 11 p90 **1,216647** apesar do bom
score global. Não converter correlação alta em tolerância física por canal.

## 3. Controle DCT pareado concluído

Fonte: `results/experiments/neural_rate_matched_dct_20261015/`.
102 janelas idênticas, duas por arquivo, quatro famílias/seis configurações;
por janela, o DCT recebe teto de bytes calculado dos índices+side do candidato.
Stream DCT real inclui header, coeficientes/localização e normalização/escapes.

| Par na amostra | Neural RMSE | DCT RMSE | Bytes neural nominais / DCT reais |
|---|---:|---:|---:|
| Rate 1x | 0,678919 | 0,835143 | 197.454 / 197.418 |
| Rate 2x | 0,587584 | 0,792094 | 387.732 / 387.588 |
| Rate 4x | **0,486022** | **0,727395** | 771.330 / 771.228 |

Neural tem menor RMSE nas **seis configurações em cada par de taxa**. Isso atualiza
o achado histórico de DCT forte em single-mode: modelos/curvas posteriores aprenderam
representação melhor no protocolo atual. Não apagar resultados negativos anteriores
nem comparar DCT da amostra com média neural da validação completa.

Comparação é teto pareado e stream DCT versus taxa neural nominal, não codecs
implantados de taxa efetiva idêntica, nem limite de informação. Seed/fold únicos
e validação já consultada limitam robustez; benchmark clássico não é otimizado.

## 4. Próxima decisão

4x ainda melhorava no fim: RMSE do piloto epochs 20–24 = 0,489497 / 0,489279 /
0,488774 / 0,486366 / **0,486022**. 1x/2x não selecionaram a última época.

Priorizar continuação **somente do 4x**, preservando optimizer/RNG e orçamento
extra definido, para medir aproximação de plateau à mesma taxa. Não repetir todos
os braços inalterados. Protocolo no [relatório 38](38-HIGH_RATE_CONTINUATION_PROTOCOL.md).

Após saturação, confirmação em configurações/fold adicionais com source correto
e proteção numérica onde necessária, ou ensaio de taxa maior/contexto se justificado.
Limiares físicos e bitrate aceitável seguem necessários para escolher um codec final.
