# Arquitetura e inicialização: resultados da triagem e validação completa

## 1. Conclusão

**A referência transferida 4x continua melhor no macro**, agora com correlação
**0,682337** e RMSE relativo **0,719900**, comparada a ~0,560/~0,810 antes desta
triagem. A variante temporal do zero supera o nativo do zero nas seis configurações,
à mesma taxa nominal e número de parâmetros.

Isso apoia investigar contexto temporal, mas **não prova inferioridade de treino
do zero**: ambos os braços scratch ainda melhoram ao final do orçamento e têm
utilização de codebook muito menor. A referência transferida também teve mais
compute/exposição histórica. Não promover modelo final nem trocar o dataset com
base apenas nessa triagem.

## 2. Evidência e protocolo

Experimento: `results/experiments/architecture_initialization_pilot_20261010/`.
Fontes: `protocol.json`, `initializations.json`, históricos por braço, `REPORT.md`,
`comparison.json` e `full_validation/summary.csv`, tabelas de grupos/janelas.
Treino e avaliação completos, marcador `COMPLETE` presente.

- 16 épocas x 4.096 updates = **65.536 updates adicionais por braço**.
- Mesmo corpus FEBUS, sampling/seed, objetivo temporal+VQ, LR 1e-4/AdamW novo.
- 27.958.081 parâmetros, 2.048 códigos, geometria temporal-4x nos três braços.
- Scratch sem pesos/optimizer pré-treinados; codebook inicializado com reservoir
  de 32 janelas somente do treino, oito por família, sem updates de encoder.
- Seleção em 102 janelas; avaliação sem novos updates nas **5.820 janelas**.
- Epoch selecionado: transferido **13**; scratch nativo **16**; scratch temporal **16**.
- A avaliação interrompida foi retomada preservando candidato completo e tentativas
  parciais. A retomada final terminou com sucesso em **7min49s**.
- Nenhum teste consultado. Validação exploratória já utilizada em decisões anteriores,
  não uma auditoria externa independente.

## 3. Validação completa: fidelidade e taxa

| Candidato | Correlação macro | RMSE relativo macro | Pior família RMSE | RMS ratio | Fase FFT |
|---|---:|---:|---:|---:|---:|
| Transferido 4x | **0,682337** | **0,719900** | **0,817242** | **0,717126** | **0,825599** |
| Scratch nativo 4x | 0,619895 | 0,778228 | 0,849809 | 0,689852 | 0,806898 |
| Scratch temporal | 0,658403 | 0,743567 | 0,821364 | 0,702224 | 0,814517 |

Compressão nominal pooled **165,914x em todos**: 465.920.000 células, 88.000.000
bits de índices e 232.800 bytes de normalização/side. Não inclui container,
mapas físicos, identificação de modelo ou codificação entrópica; não é bitrate
de um codec neural completo implantado.

RMS ratio ~0,717 no vencedor ainda indica atenuação. RMSE relativo ~0,720 é
erro importante comparado à variação do original; não significa 72% de fidelidade.

### Por configuração: RMSE / correlação

| Família/configuração | Transferido | Scratch nativo | Scratch temporal |
|---|---:|---:|---:|
| Doped / 5m SMF | **0,673827 / 0,739589** | 0,715648 / 0,698430 | 0,713274 / 0,701385 |
| Multicore / C1 and C4 | **0,793277 / 0,608740** | 0,841880 / 0,539785 | 0,821364 / 0,570043 |
| Multimode / Configuration 7 | **0,817242 / 0,587084** | 0,849809 / 0,553030 | 0,820682 / 0,585188 |
| Single-mode / Configuration 11 | **0,505409 / 0,856224** | 0,698537 / 0,656401 | 0,553870 / 0,817424 |
| Single-mode / Configuration 3 | 0,705322 / 0,707745 | 0,770001 / 0,632967 | **0,692637 / 0,715841** |
| Single-mode / Configuration 7 | **0,575028 / 0,817830** | 0,648188 / 0,775631 | 0,610338 / 0,797728 |

Scratch temporal ganha em todas as configurações sobre scratch nativo e em
Configuration 3 sobre transferido. Transferido é melhor nos outros cinco grupos.
Não é dominância universal da referência transferida.

## 4. Canais: média global não resolve a cauda

| Diagnóstico macro por janela/canal | Transferido | Scratch nativo | Scratch temporal |
|---|---:|---:|---:|
| Mediana do erro relativo | **0,872811** | 0,909629 | 0,890626 |
| p90 do erro relativo | 1,228475 | **1,152914** | 1,197300 |
| Erro médio de canais não fracos | **0,925070** | 0,946968 | 0,940893 |

Quantis são estatísticas por janela agregadas macro, não quantis pooled de todos
os canais. O vencedor global não vence no p90 macro. Em single-mode Configuration
11, o transferido tem p90 **1,878002**, contra **1,086138** do scratch nativo,
apesar do RMSE global muito menor. Canais de baixa amplitude podem ter grande
erro relativo sem dominar RMSE global; precisam de tolerâncias físicas próprias.

A média de erro relativo por canal continua sensível a canais quase constantes
(dezenas de milhares); não a usar isoladamente para classificar modelos.

## 5. Curvas e quantização: não encerrar scratch prematuramente

RMSE do piloto nos últimos cinco epochs:

| Braço | 12 | 13 | 14 | 15 | 16 |
|---|---:|---:|---:|---:|---:|
| Transferido | 0,725181 | **0,717803** | 0,724168 | 0,718613 | 0,718342 |
| Scratch nativo | 0,784451 | 0,783616 | 0,781647 | 0,779417 | **0,775702** |
| Scratch temporal | 0,757008 | 0,747472 | 0,744380 | 0,749372 | **0,743295** |

Estes são scores das 102 janelas, não da avaliação completa. Scratch nativo
melhora gradualmente e scratch temporal melhora com oscilação; ambos selecionam
a última época. Não há demonstração de convergência suficiente para descartá-los.

| Braço | Perplexidade média época 1 → 16 | Códigos usados na época 1 → 16 |
|---|---:|---:|
| Transferido | 359,0 → 512,4 | 1.153 → 1.227 |
| Scratch nativo | **2,0 → 119,0** | 745 → 177 |
| Scratch temporal | **11,2 → 149,0** | 866 → 223 |

A inicialização por latentes de treino não evitou uso muito concentrado no
primeiro epoch. A contagem agregada desse epoch inclui códigos usados antes da
concentração; não contradiz baixa perplexidade média. Ao fim há recuperação de
perplexidade, mas só ~8,6%/~10,9% dos códigos são usados nos braços scratch.
Isso justifica investigar dinâmica de quantização e orçamento, **não prova que
codes inativos causem toda a falha nem que reativá-los garantirá ganho**.

## 6. Decisão e próximo protocolo

1. Preservar transferido epoch 13 como nova referência de pesquisa, junto à
   anterior. Não promover a produção; guardar scratch temporal como candidato
   competitivo em Configuration 3.
2. Priorizar teste de **orçamento/convergência dos dois braços scratch**. Retomar
   optimizer/RNG, ampliar orçamento previamente definido e medir melhora/plateau,
   sem comparar o novo custo histórico como se fosse igual ao transferido.
3. Se a quantização continuar restrita, comparar controle sem mudança de quantizador
   com **uma intervenção isolada** (por exemplo, política train-only de reativação),
   mantendo arquitetura, corpus, código/taxa e orçamento; não misturar mudança de
   arquitetura, LR e dataset nessa ablação.
4. Uma alternativa temporal inicializada por transferência pode ser um próximo
   controle, mas converter kernels demanda regra explícita/testada, não carregar
   state_dict incompatível ou supor equivalência de parâmetros.
5. Confirmar candidatos em outra seed/grupos antes de escolher arquitetura final.
   Novo dataset permanece estudo separado, motivado por lacunas demonstradas.

Essa seção define o próximo protocolo; **não foi iniciado outro treino nesta
atualização**. A comparação encerrada não autoriza inferir arquitetura ótima,
scratch inviável ou necessidade automática de mais dados. Limiares físicos e
holdout externo continuam pendentes para uma decisão final.
