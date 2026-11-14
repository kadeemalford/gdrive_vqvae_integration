# Síntese com IDs fixos: lookup isolado insuficiente; contexto em avaliação

**Atualização: 12/11/2026.** Continuidade do
[relatório 83](83-POROTOMO_PRETRAINING_BRIDGE_RESULTS.md), sem promoção de modelo.

## 1. Lookup de síntese independente: resultado encerrado

Campanha `results/experiments/untied_synthesis_train_probe_fold0_20261110/`.
Foi usado o checkpoint transferido A compacto da época 24, não o offshore
sem adaptação. Encoder, codebook de análise, decoder e IDs originais foram
congelados. Somente uma nova tabela de síntese 2048×128 foi treinada.

Dois braços receberam os mesmos 256 exemplos de treino: otimização por erro
latente e otimização por waveform através do decoder congelado. Seleção em
arquivos de treino separados, por loss normalizada ponderada por células dentro
da configuração, configurações e famílias com pesos iguais. Esses arquivos
não são um holdout de aquisições independente do treinamento histórico.

| Síntese | Melhor D de triagem ↓ | Update selecionado |
|---|---:|---:|
| Original congelada | 0,371929769 | 0 |
| Controle: erro latente | 0,371927424 | 128 |
| Waveform | 0,371483377 | 256 |

O braço waveform melhorou **0,1200% contra a origem e 0,1194% contra o controle**.
Não passou a regra de extensão predefinida; ambos encerraram em 256 updates.
Nenhuma janela de validação foi aberta, nenhum codec novo foi confirmado e
nenhum modelo foi promovido. A tabela extra custaria 1.048.576 bytes de tensor
compartilhado, além da serialização; o pequeno ganho de D não demonstra ganho
de RMSE físico nem compensação desse custo.

O arquivo fonte e os parâmetros congelados foram auditados como imutáveis.
O resultado difere do PQ decoder-aware anterior, mas não justifica insistir
automaticamente neste lookup isolado ou aumentar seu budget após observar o ganho.

## 2. Probe de contexto: protocolo antes dos resultados

Pipeline lançado em 12/11, após confirmar ausência de treino ativo na GPU 0:

- `latent_context_smoke_fold0_20261112`: integração de dois updates por braço;
- `latent_context_train_probe_fold0_20261112`: 512 updates por braço, sequenciais;
- uma única extensão de **todos os braços** até 2.048 é possível conforme a
  regra congelada em `protocol.json`, não por inspeção oportunista dos resultados.

O objetivo é comparar quatro adaptadores residuais de capacidade semelhante
(diferença de parâmetros abaixo de 1%), todos iniciados como identidade:

1. **Local:** MLP pointwise dos vetores quantizados originais.
2. **Forma:** MLP com coordenadas e frações de ocupação derivadas das dimensões
   já conhecidas do recorte e da grade latente.
3. **Momentos:** a forma mais média e desvio por feature dos próprios tokens
   decodificados, sem estatísticas do sinal original.
4. **Axial:** atenção entre tokens nas duas direções, com posição/forma conhecidas.

Encoder, codebook, IDs e decoder base continuam congelados. Não entram resíduos,
features contínuas do encoder, identificação de família/arquivo ou sinal original
como informação auxiliar do adaptador. O sinal alvo é usado somente para a loss
durante treino e para os scores da triagem de treino.

**O decoder base já tem GroupNorm**, portanto já possui dependências globais
através de momentos. A hipótese é de condicionamento mais rico por forma e
conteúdo, não de introduzir a primeira dependência global na rede.

A regra inicial exige ganho ≥0,5% de D contra origem e melhora recente ≥0,2%;
um vencedor contextual deve também superar o local em ≥0,25%, ou o local deve
melhorar ≥1% contra origem. Só então todos recebem a extensão pareada. A abertura
de uma confirmação em desenvolvimento exige ganho final ≥1% de D contra origem;
alegar vantagem de contexto exige ainda ≥0,5% contra local.

**Esses percentuais são de loss normalizada, não de RMSE físico.** Uma futura
confirmação precisa cobrar bytes reais completos e pesos compartilhados, realizar
replay discreto, examinar todas as configurações e manter as guardas. O critério
estrutural de investimento continua ≥5% menos RMSE físico sem mais bytes e sem
regressões, não o limiar menor usado apenas para triagem.

## 3. Resultado do contexto: ganho pequeno, sem abertura de validação

A campanha completou a extensão predefinida para **2.048 updates em cada um dos
quatro braços**, sequencialmente. Aos 512 updates, o axial satisfez os critérios
iniciais; isso autorizou a extensão pareada, não uma promoção.

| Adaptador | Parâmetros | Melhor D de triagem ↓ | Update selecionado | Ganho de D contra origem |
|---|---:|---:|---:|---:|
| Original congelado | — | 0,371929769 | 0 | — |
| Local | 34.052 | 0,370827939 | 2.048 | 0,2962% |
| Forma | 33.884 | 0,370780484 | 2.048 | 0,3090% |
| Momentos | 34.042 | 0,370116408 | 1.664 | 0,4876% |
| Axial | 33.824 | **0,369443047** | 1.920 | **0,6686%** |

Axial superou o controle local em **0,3735%**, abaixo do limiar de 0,5% para
alegar vantagem contextual pelo protocolo. O ganho contra origem também ficou
abaixo do limiar de 1% para abrir confirmação em desenvolvimento. Portanto,
**encerrar este probe sem novo aumento de orçamento, sem abrir validação e sem
promover modelo**. O endpoint axial foi pior que seu selecionado; o score da
tabela não é o score do último update.

Os quatro optimizers foram auditados no remoto: contadores finais iguais a
2.048 e hashes de linhagem corretos. Código congelado e arquivo fonte mantiveram
os hashes registrados. O runner verificou imutabilidade dos parâmetros base;
os IDs originais permaneceram fixos. Nenhuma janela de sinal de validação foi
lida. Artefatos pequenos, históricos e adaptadores selecionados foram copiados
para o mesmo diretório de resultados local.

O resultado sugere benefício pequeno na triagem, não ausência absoluta de
informação contextual útil. Não foi medida fidelidade física completa nem taxa
de um codec com esses adaptadores. A triagem reutiliza desenvolvimento de treino,
sem múltiplas seeds ou independência de aquisição estabelecida; não apresentar
essas diferenças como significância estatística ou generalização externa.

Foi usado somente o remoto e apenas uma RTX 4090; nenhum commit/push foi feito.
