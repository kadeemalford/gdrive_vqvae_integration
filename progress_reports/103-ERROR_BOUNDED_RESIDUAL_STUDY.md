# H4 — correção residual com erro ponto a ponto controlado

**Protocolo registrado em 15/11/2026, antes dos novos resultados.**
Este documento reúne protocolo, execução, resultados e continuações da etapa.

## Hipótese e primeiro contraste

H4 do relatório 90 não equivale a acrescentar tokens fixos: transmitir a
correção necessária para satisfazer uma precisão declarada e medir seu custo.
Começar separando o valor da base neural do valor da codificação do residual.
Não prometer manter a taxa ultracompacta quando se exige precisão estrita.

Três precisões fixadas: erro absoluto máximo por célula ordinária
`epsilon = alpha × escala_de_referência_do_grupo`, com
`alpha ∈ {0,1; 0,03; 0,01}`. Escalas são as prospectivas já congeladas no treino.
São **precisões experimentais**, não tolerâncias validadas por uma aplicação.
O epsilon float64 efetivamente transmitido define o contrato.

Comparar, para cada precisão:

1. base zero + coeficientes de correção;
2. reconstrução do VQ denso maduro congelado + coeficientes de correção.

Ambos usam o mesmo quantizador `round((x-base)/(2 epsilon))`, coder inteiro
FRC1/FRC2/FRC3 com custos completos e escapes exatos. A reconstrução final é
float32. Toda célula que não satisfizer o limite após esse roundtrip usa escape
com posição e bits originais cobrados; não finitos e zeros com sinal também
são preservados exatamente. Verificar a garantia novamente no receptor.
O frame de correção vincula shape, epsilon e hash da base reconstruída.

A base neural não é gratuita: cobrar todos os frames originais, registro,
mapas/transporte e framing adicional. Publicar pesos compartilhados separados,
além do custo quando instalados uma vez. O controle zero não paga registro VQ
nem pesos VQ. Mesmo limite ponto a ponto não implica RMSE exatamente igual:
publicar os erros efetivamente obtidos, sem chamar isso de dominância RD.

## População, orçamento e decisões

Mesmos **136 checks train-only / 43 arquivos / dez configurações** do
diagnóstico anterior. Reutilizar seus frames baseline, nunca os índices
otimizados. Fonte e registro precisam conferir por SHA-256; nenhuma reseleção.
Zero updates nesta primeira fase; até 816 codificações de correção, 136
decodificações neurais e replay completo de cada frame.

Publicar pooling físico, bandas/caudas, bytes e contagem de escapes por ponto.
Sem excluir constantes ou tiles pequenos. Sem sinais val/test/EDAS.
Falha de garantia ou integridade interrompe a execução; não relaxar epsilon.

Se a base neural economizar ≥5% de bytes completos em pelo menos duas precisões
e não aumentar bytes em nenhuma família nesses pontos, investigar um modelo
neural condicional de residual com controle sem contexto, em protocolo
complementar registrado **neste documento antes de treiná-lo**. Caso contrário,
o preflight não encerra toda H4: avaliar separadamente a oportunidade de
predição causal no residual antes de decidir sobre treino. Não estender
automaticamente arquitetura/precisões para salvar um resultado negativo.

Destino: `results/experiments/error_bounded_residual_20261115/`.

### Verificação de engenharia e lançamento

476 testes locais passaram, incluindo extremos float32, payloads de NaN,
zeros com sinal, overflow dos coeficientes, arredondamento físico e rejeição
de base diferente/corrupção. GPUs remotas livres no preflight; usar GPU0
somente para decodificar os frames congelados. GPU1 não recebe treino
especulativo enquanto a oportunidade não foi medida.

## Outras propostas ainda abertas

- Condicionamento físico: não confundir treino por rótulo de configuração com
  generalização física. As restrições documentadas em 109 permanecem.
- Multicore: não repetir lag/preditor sobre o gate negativo de 113.
- Semissintético e diversidade: contraste ainda não executado; precisam de
  protocolo próprio, origem de fundos e partições sem vazamento.
- P3 permanece condicionado a um candidato elegível, não ao fim de uma etapa.

## Resultado da primeira fase

136 janelas / 43 arquivos / 12.799.000 células; 816 correções codificadas e
reproduzidas. Nenhuma violação de garantia. Zero updates de modelo.

| Alpha | Base zero, bpc | Base VQ + correção, bpc | Economia com base VQ |
|---|---:|---:|---:|
| 0,1 | 9,186761 | 9,203094 | −0,1778% |
| 0,03 | 11,385163 | 11,352505 | +0,2868% |
| 0,01 | 13,454644 | 13,584681 | −0,9665% |

Todos os pontos tiveram aumento de bytes em multimode e single-mode. O gate
de oportunidade da base não passou. Os pesos VQ (~111,9 MB) ainda são custo
compartilhado separado nessa tabela: instalá-los piora a comparação curta.
NRMSE* observado é próximo de 0,00165 / 0,000494 / 0,000165; alta precisão
custa aqui ~9–14 bpc, não as taxas ultracompactas históricas.
Mesmo bound não significa distorção realizada idêntica.

## Segunda fase — protocolo de residual neural causal

**Registrado após a primeira fase, antes de treinar os modelos desta fase.**
O fracasso da base com coder genérico não avalia o componente neural condicional
de H4. Testar diretamente uma representação lossless dos mesmos coeficientes
de correção. Não alterar epsilon nem o waveform reconstruído.

- Fixar **alpha=0,03**, ponto intermediário e melhor compromisso da primeira
  fase. A escolha é informada pelos resultados já vistos, não confirmação.
- Mesma fonte VQ e registro. Ajuste somente nos **44 recortes fit** históricos,
  separados por arquivo dos 136 checks. Cobrar os pesos novos e metadados.
- Dois MLPs `8→32→32→1`, ReLU, seed 103, última camada inicialmente zero.
  Controle recebe somente features da base VQ/posição; candidato recebe também
  três coeficientes da linha espacial anterior em tempos −1/0/+1.
  Todos os coeficientes dessa linha já foram decodificados.
- Features da base usam desvio da própria base; coeficientes usam RMS fit por
  grupo, congelado e transmitido. Nenhuma estatística dos checks calibra o modelo.
- 4.096 updates por braço, batch 4.096 células, Adam LR=0,001, MSE dos
  coeficientes normalizados. Seleção uniforme de grupo, arquivo fit e células.
  Sem escolha de checkpoint: somente endpoint. Sem sweep, extensão ou val/test.
- GPUs 0/1, um braço por GPU; exportação em float32. No codec, ambos os lados
  usam a mesma inferência NumPy CPU e hash do modelo, não previsões teacher-forced
  gratuitas do alvo atual.
- Codificar inteiro `q - round(predição)` e reconstruir q linha a linha.
  Se houver overflow int32, usar o frame original. Escolher por bytes completos
  entre frame original e candidato; cobrar o modelo mesmo quando há fallback.
- Receptor separado precisa reproduzir **todos os coeficientes e todos os bits
  da reconstrução física da primeira fase**. Isso dá comparação de bytes com
  distorção exatamente igual ao respectivo baseline, não só mesmo bound.

Publicar custo de ambos os braços, inferência causal e instalação do modelo.
Investimento exige ≥5% de economia total contra o residual original com os novos
pesos instalados, ganho contra controle e nenhum aumento de frames por família.
Isso permite outro estudo, não promoção; H4 ainda precisa vencer a alternativa
sem base neural e atender ao objetivo físico da aplicação.

## Resultado da segunda fase e decisão de H4

Ambos os braços completaram 4.096 updates: **8.192 updates novos** e
33.554.432 apresentações de células de coeficientes. São exposições repetidas,
não novas aquisições nem observações independentes.

| Correção sobre a mesma base VQ | Bytes completos | bpc | Economia contra coder original |
|---|---:|---:|---:|
| Coder original | 18.162.589 | 11,352505 | — |
| MLP somente base | 18.169.100 | 11,356575 | −0,03585% |
| MLP causal | 18.093.897 | 11,309569 | **+0,37821%** |

Cada MLP instalado custa 7.199 bytes, mais 498 bytes do registro de escalas.
O causal usou fallback em 80/136 janelas; controle em 129/136. Todos os frames
passaram replay de coeficientes e de bits físicos no receptor separado, sem
violações da garantia. O custo medido do codec de checks foi ~49 s por braço,
incluindo geração de candidatos, coder, replay e auditoria; não é throughput
isolado de produção.

A distorção física é **exatamente a mesma** do baseline VQ+correção dessa
precisão. O ganho causal é real neste desenvolvimento, mas insuficiente para
o gate de 5%. Mesmo combinando as duas fases, a economia contra base zero
fica abaixo de 1%, antes de instalar os ~111,9 MB do VQ compartilhado.

**Encerrar este desenho sem extensão, novo alpha/MLP/seed ou promoção.**
H4 agora teve um contraste direto de garantia física e um modelo neural de
residual causal efetivamente treinado/codificado. Isso não esgota modelos de
entropia neurais, outras bases ou contratos de precisão; não refuta toda H4.
Não diluir o resultado negativo com uma expansão adaptativa.

Artefatos da segunda fase:
`results/experiments/neural_bounded_residual_20261115/`.
Segue a hipótese de diversidade por composição em
[104](104-SEMISYNTHETIC_AUGMENTATION_STUDY.md), sem reabrir reservas.

### Auditoria de engenharia

A primeira fase preservou 816 frames e seus custos/limites foram auditados.
O snapshot do runner, inicialmente ausente por resolução de caminho absoluto,
foi recuperado da cópia local **com SHA-256 idêntico ao protocolo**, antes de
qualquer ajuste de engenharia. A segunda fase já usa caminhos relativos no
snapshot. Nenhum experimento foi repetido ou resultado alterado por esse ajuste.
O postflight de ambos os braços conferiu frames, custos completos, disjunção
fit/check por arquivo e hashes dos snapshots. Evidência em
`POSTFLIGHT_AUDIT.json` nas duas pastas de experimento.
