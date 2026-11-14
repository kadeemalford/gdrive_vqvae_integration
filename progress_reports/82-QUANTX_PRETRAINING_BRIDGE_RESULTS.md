# Ponte QuantX: pior fidelidade que investir o mesmo orçamento diretamente em FEBUS

**Data: 10/11/2026.** Resultado do protocolo delimitado na seção 9 do
[relatório 80](80-LEARNING_RATE_AND_EXTERNAL_PRETRAINING_REVIEW.md). Campanha
`results/experiments/pretraining_bridge_fold0_20261110/` concluída com avaliação
completa de streams. **Nenhum modelo promovido; nenhuma extensão automática.**

## 1. Resultado principal

Com o mesmo total de **24.576 atualizações adicionais por braço**, usar 16.384
delas em fase QuantX antes de 8.192 em FEBUS produziu **11,47% mais RMSE** que
usar todas em FEBUS. A correlação também foi menor nas seis configurações.

| Candidato | Updates novos em QuantX / FEBUS | RMSE macro ↓ | Correlação ↑ | Stream completo | Razão completa |
|---|---:|---:|---:|---:|---:|
| Offshore compacto, sem updates novos | 0 / 0 | 1,000009 | 0,091459 | 10.852.079 B | 234,359× |
| **F — FEBUS direto** | 0 / 24.576 | **0,726543** | **0,669522** | 15.155.168 B | 167,816× |
| Q — ponte de fase QuantX | 16.384 / 8.192 | 0,809862 | 0,563068 | **14.524.463 B** | **175,103×** |

Q economizou **630.705 bytes / 4,162%**, mas perdeu fidelidade. Isso é uma troca
de taxa por distorção, **não superioridade a bytes iguais nem dominância Pareto
de F sobre Q**. Falhou a meta predefinida de obter ≥5% menos RMSE que F, sem
mais bytes e preservando as guardas. A razão alta do offshore sem adaptação
também não é sucesso: sua reconstrução é muito ruim.

Cada modelo de implantação ocupa **111.869.967 bytes**, compartilhados e
discriminados do transporte. Incluindo-os uma vez nesta coorte curta, F/Q
ficam em ~**20,022× / 20,122×**. Não interpretar a economia de Q como melhoria
de fidelidade/compressão conjunta ou confundi-la com os outros empacotamentos
lossless históricos.

## 2. O que foi controlado

- Mesmos pesos offshore **originais**, com geometria compacta `spatial-4x`;
  não os pesos FEBUS adaptados do experimento de LR.
- CNN nativa, K2048 × 128, loss `0,5 MSE + 0,5 L1 + VQ`, modelo inteiro
  treinável, LR constante `1e-4`, AdamW novo no início e na fronteira de etapa
  **em ambos os braços**. Não houve mudança de scheduler/arquitetura simultânea.
- Etapa 1: F recebeu 16.384 updates FEBUS; Q recebeu 16.384 updates QuantX.
  Etapa 2: a mesma sequência de **8.192 batches FEBUS**, com mesmas seeds.
- Nos 12 HDF5 QuantX auditados, foram preparadas **4.579 janelas de 1.000
  amostras**, todos os loci, com normalização global float64 antes de float32.
  Sem conversão para strain-rate, diferenciação, denoising, remoção de drift
  ou máscara espacial inventada. Os brutos foram preservados.
- Cada janela QuantX tem **100 ms a 10 kHz e 70–100 loci**; cada janela FEBUS
  tem **50 ms a 20 kHz**, com tiles retidos nativos. Igualar 1.000 pixels não
  tornou os domínios fisicamente equivalentes.
- Q balanceou quatro grupos fonte e aquisições dentro do grupo. Isso não é
  cobertura equivalente das quatro famílias FEBUS, em particular multicore.
- Updates adicionais pareados, **não FLOPs, duração física ou exposição FEBUS**.
  O contraste responde onde investir esse orçamento, não qual fonte venceria
  com pré-treinos convergidos e fine-tuning longo de dose alvo igual.

O endpoint da etapa final foi selecionado em ambos, com a inicialização elegível.
Validação: **7.220 janelas / seis configurações / quatro famílias**, sem teste.
Trata-se de desenvolvimento reutilizado. FAK1/FSB1 e tabelas ajustadas somente
em 32 janelas de treino foram comuns; o decoder reconstruiu a partir dos índices
transmitidos, e o reader verificou os containers.

## 3. Distribuição da perda

| Família / configuração | F: RMSE | Q: RMSE | F: correlação | Q: correlação |
|---|---:|---:|---:|---:|
| Doped / 5m LSF | 0,704751 | 0,807334 | 0,709052 | 0,589885 |
| Multicore / C1 and C3 | 0,814367 | 0,871843 | 0,579400 | 0,488183 |
| Multimode / Configuration 6 | 0,822881 | 0,895739 | 0,570708 | 0,443256 |
| Single-mode / Configuration 10 | 0,552099 | 0,667805 | 0,823441 | 0,712614 |
| Single-mode / Configuration 2 | 0,645029 | 0,719655 | 0,767819 | 0,694801 |
| Single-mode / Configuration 6 | 0,495396 | 0,606137 | 0,865524 | 0,785436 |

Magnitude FFT macro: **0,568175 → 0,695385**, pior; alinhamento de fase:
**0,800175 → 0,765467**, pior; RMS ratio: **0,685741 → 0,564135**, maior atenuação.
Foram **34 vetos de configuração/métrica** na comparação Q versus F.

O p90 relativo macro de Q foi numericamente menor, mas essa macro é dominada
por escalas patológicas de canais de pequena variação em multimode, como no
relatório 81. Não a usar como evidência de preservação geral diante das perdas
de RMSE/correlação/amplitude/espectro nas configurações. Não retiramos canais,
relaxamos guardas ou selecionamos retroativamente outra preparação para salvar Q.

## 4. Dinâmica de adaptação e limites causais

Após o pré-treino externo, o RMSE **da triagem FEBUS** de Q era **1,168488**.
Durante seus quatro blocos FEBUS posteriores, caiu para
`0,900976 / 0,872183 / 0,837033 / 0,809246`. F terminou a triagem em **0,728980**.
Esses números não são os da validação completa da tabela principal.

Na etapa final, com batches iguais, a perplexidade média de treino Q passou de
~73 para ~252; F passou de ~347 para ~367. Isso é compatível com readaptação
da representação, mas não demonstra que aumentar códigos usados por si só
resolveria fidelidade. A perplexidade ~26 no fim do treino em fase QuantX também
reflete a distribuição diferente da fonte; não é prova isolada de colapso.

Ambos ainda melhoravam ao teto. O negativo não estabelece convergência nem
inutilidade de QuantX em geral: são 12 aquisições selecionadas, fase bruta,
mapa útil não confirmado e apenas 8.192 updates finais no alvo. A representação
de drift/ruído de todos os loci pode ser inadequada para transferência, mas o
experimento não isolou cada causa.

**Distinção importante:** F teve mais exposição a FEBUS. Logo, o resultado
principal não basta para atribuir toda a diferença a uma inicialização pior.
Foi executado um **controle diagnóstico G**, offshore → somente os mesmos 8.192
updates FEBUS finais, para separar efeito da fonte de exposição ao alvo. Ele
tem menos compute total e **não substitui F como comparador de promoção**.
Seu protocolo/critério foram fixados antes dos resultados; o critério principal
e a decisão de não prolongar Q permanecem intactos.

| Mesma dose FEBUS final | Updates externos novos | RMSE completo | Stream completo |
|---|---:|---:|---:|
| **G — sem ponte, somente 8.192 FEBUS** | 0 | **0,802886** | 14.810.782 B |
| Q — 16.384 QuantX + os mesmos 8.192 FEBUS | 16.384 | 0,809862 | **14.524.463 B** |

Mesmo com essa dose alvo igual, Q tem **0,869% mais RMSE**, com 1,933% menos
bytes. A diferença grande contra F também envolve mais exposição FEBUS, mas
o diagnóstico não encontrou benefício macro da ponte bruta para a inicialização sob
os 8.192 updates finais. Há benefícios localizados contra G: ~2,21% de RMSE em
single-mode 10, ~0,32% em single-mode 6 e ~0,12% em multimode; doped, multicore
e single-mode 2 pioram. Portanto, a conclusão não é de prejuízo universal em
cada configuração. Não confundir a diferença pequena G/Q com o prejuízo
prático de 11,47% contra F: são perguntas distintas e não devem ser somadas.

G selecionou seu endpoint, com AdamW final em 8.192; sua comparação completa
usa o mesmo replay/codec, as mesmas 7.220 janelas e calibração de treino.
Artefatos em `results/experiments/target_dose_control_fold0_20261110/`.
É resultado de desenvolvimento e de orçamento curto, não uma conclusão sobre
todo interrogador QuantX, todo seu corpus ou o limite de fine-tuning longo.

## 5. Auditoria e próximos passos

Os contadores do AdamW após a etapa final são **8.192 em ambos**, coerentes com
o reset comum. O problema de alias da primeira campanha LR não se repete aqui:
os optimizers das etapas são novos e a retomada usa cópia independente. A suíte
remota de integração passou antes do treino. Foram usados ~729,7 + 365,4 s de
updates em F e ~795,3 + 365,5 s em Q, além de validação/serialização, numa única
RTX 4090. Nenhum processo paralelo desta pesquisa usou a segunda GPU.

**Não ampliar a ponte QuantX bruta nem testar do zero nela automaticamente.**
O próximo candidato é PoroTomo, cujo nativo declara amplitude proporcional a
strain-rate e cuja identidade HDF5/SEG-Y foi verificada (relatório 80, seção 10).
Ele ainda difere em instrumento, banda e contexto físico; coerência na fonte
não garante melhoria no FEBUS.

A preparação pública inicial parou por um contêiner HDF5 de **6.544 bytes sem
nenhum dataset**, não por um waveform duplicado. A seleção e os 29 arquivos
preparados foram preservados. A versão 2 inspeciona estruturalmente os objetos
pequenos antes dos quantis, mantendo arquivos pequenos se contiverem waveform;
não seleciona por erro neural, amplitude ou SNR. Preparação incompleta não será
usada em treino.

O controle F foi auditado para possível reutilização: pesos, código, vendors,
sequências, dados, contadores e streams serão congelados/checados. Não repetir
seus 24.576 updates apenas para comparar outra fonte sob as mesmas condições.
Após a checagem estrutural, a preparação pública versão 2 concluiu com **60
HDF5 / 1.798 janelas**, sem janelas constantes; 2.792.312.320 bytes de brutos e
2.514.920.403 bytes preparados. O manifesto SHA-256 é
`0529aee32750a412b8300b33f21efb4f890a29bf6617cf2445e57159c201ed6d`.
Foi lançado o pipeline `porotomo_bridge_fold0_20261110`, precedido por smoke,
somente na GPU 0, reutilizando F após auditoria. A igualdade é de updates e
batches FEBUS finais, **não de FLOPs nem de quantidade de células por update**:
os exemplos públicos têm 384 canais e 1 s, distintos dos tiles FEBUS.
Não há resultado de reconstrução da ponte pública nesta atualização.
Continua vedada qualquer promoção operacional sem requisitos físicos e
confirmação independente das aquisições relevantes.
