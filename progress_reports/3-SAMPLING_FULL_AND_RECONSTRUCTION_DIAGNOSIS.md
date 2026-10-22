# Balanceamento completo e diagnóstico de reconstrução FEBUS

Data: 2026-10-07. Experimento: `results/experiments/sampling_full_20261005/`.

## Conclusão e decisão

**O balanceamento hierárquico não deve substituir o baseline uniforme como padrão.**
Melhora doped e parte de multimode, mas piora multicore e single-mode. O ganho macro
global por família é pequeno e depende da agregação; não é uma melhoria generalista
consistente. Multimode continua sendo a família mais fraca, com configurações quase
não reconstruídas. Não treinar agora o modelo final em todos os dados, nem atribuir
automaticamente a falha à quantidade de dados disponível.

O passo seguinte do roadmap — diagnóstico — foi **executado**, sem treinamento:
auditoria dos testes completos, estratificação por arquivo/configuração e diagnóstico
de espectro/quantização na validação. O principal achado é forte atenuação de amplitude
e perda de detalhes, sem recuperação relevante ao remover apenas a quantização.
Priorizar objetivo de reconstrução, normalização/geometria e análise rate-distortion
antes de outra comparação longa de amostragem ou aumento isolado do codebook.

## 1. Integridade e protocolo

- Marcador global `COMPLETE` e marcadores das quatro avaliações presentes.
- 23.880 janelas de teste avaliadas pelos dois braços (47.760 linhas).
- 23 grupos família/configuração: 12 single-mode, 5 multicore, 3 doped, 3 multimode.
- Auditoria confirmou que todas as linhas de avaliação pertencem ao teste correto.
  Configurações completas permanecem separadas entre treino, validação e teste.
- Ambos os braços partiram do checkpoint OOI local; seed 2026, AdamW, LR `1e-5`,
  perda `0,5*MSE + 0,5*L1 + VQ`, seleção por menor val_loss ponderada por janela.
- `window`: todas as janelas uma vez por época. `family-group`: sorteio com reposição
  família → configuração → janela, com o mesmo orçamento de atualizações por época.
- Retomadas por `last.pt`; CSVs sem épocas duplicadas ou faltantes. Trabalho parcial
  de uma época interrompida é descartado antes de refazê-la.
- Early stopping tornou diferentes os orçamentos realizados no fold 3. A comparação
  é dos protocolos com parada antecipada, não de atualizações exatamente iguais.

| Fold | Braço | Épocas completas | Época de best.pt | Melhor val_loss |
|---|---|---:|---:|---:|
| 0 | Uniforme | 10 | 9 | 0,75512 |
| 0 | Balanceado | 10 | 9 | 0,76045 |
| 1 | Uniforme | 10 | 10 | 0,74621 |
| 1 | Balanceado | 10 | 10 | 0,75804 |
| 2 | Uniforme | 10 | 7 | 0,79180 |
| 2 | Balanceado | 10 | 8 | 0,79909 |
| 3 | Uniforme | 8 | 7 | 0,84182 |
| 3 | Balanceado | 5 | 4 | 0,84639 |

A soma dos tempos das épocas completas foi ~23,93 h; exclui épocas interrompidas,
avaliações e períodos sem execução. A melhor val_loss global foi inferior no uniforme
em todos os folds, mas contém VQ e pesos por janela: não equivale à fidelidade macro.

## 2. Teste completo: ganhos e regressões

### Macro por fold

Média por configuração dentro da família, seguida de média das famílias presentes.
Fold 3 testa apenas single-mode e multicore, não doped nem multimode.

| Fold | Corr. uniforme | Balanceado | Δ corr. | RMSE relativo uniforme | Balanceado |
|---|---:|---:|---:|---:|---:|
| 0 | 0,2625 | 0,2638 | +0,0012 | 0,9679 | 0,9569 |
| 1 | 0,4120 | 0,4220 | +0,0100 | 0,8830 | 0,8794 |
| 2 | 0,4934 | 0,4956 | +0,0022 | 0,8544 | 0,8526 |
| 3 | 0,4435 | 0,4113 | −0,0322 | 0,8660 | 0,8832 |

Ganhar em 3/4 folds não basta: a regressão no fold 3 supera a soma dos pequenos
ganhos de correlação dos demais. Isso não contradiz a média por família abaixo:
os folds têm composições diferentes.

### Macro por família, reunindo suas configurações de teste

| Família | Corr. uniforme | Balanceado | RMSE uniforme | Balanceado | SNR uniforme → balanceado (dB) |
|---|---:|---:|---:|---:|---:|
| Doped | 0,4477 | 0,4744 | 0,8949 | 0,8705 | 1,016 → 1,235 |
| Multicore | 0,3714 | 0,3566 | 0,9262 | 0,9320 | 0,669 → 0,615 |
| Multimode | 0,2246 | 0,2384 | 0,9632 | 0,9562 | 0,339 → 0,406 |
| Single-mode | 0,5088 | 0,4923 | 0,8206 | 0,8298 | 1,916 → 1,822 |

| Agregação global | Corr. uniforme → balanceado | RMSE uniforme → balanceado |
|---|---:|---:|
| Quatro famílias uniformes; configurações uniformes dentro de cada família | 0,3881 → 0,3905 | 0,9013 → 0,8971 |
| Janelas de teste uniformes | 0,4043 → 0,3982 | 0,9003 → 0,9009 |

Acompanhar ambas e a pior família. O ganho macro de +0,0023 em correlação não
compensa automaticamente regressões em duas famílias. Não há limiar de fidelidade
suficiente acordado. RMSE relativo perto de 1 indica erro perto do RMS do sinal
normalizado; reconstruir zero nesse espaço tem RMSE relativo 1. SNR aqui é
sinal/erro de reconstrução, não qualidade de denoising de eventos.

### Configurações que explicam o trade-off

Balanceamento melhorou correlação em **8/23** configurações e RMSE em **10/23**.
Todas as cinco configurações multicore perderam correlação e pioraram RMSE.

| Família/configuração | Corr. uniforme → balanceado | RMSE uniforme → balanceado |
|---|---:|---:|
| Doped / 5m i29 | 0,2249 → 0,2890 | 1,0289 → 0,9672 |
| Doped / 5m LSF | 0,5444 → 0,5605 | 0,8383 → 0,8271 |
| Multimode / Configuration 5 | 0,0837 → 0,0678 | 1,0000 → 0,9994 |
| Multimode / Configuration 6 | 0,1521 → 0,1814 | 0,9907 → 0,9852 |
| Multimode / Configuration 7 | 0,4380 → 0,4661 | 0,8989 → 0,8840 |
| Single-mode / Configuration 8 | 0,4058 → 0,3449 | 0,8954 → 0,9200 |

Mais exposição não resolveu Configuration 5 multimode. Tabelas completas por
configuração e arquivo/frequência nominal estão na auditoria local. Títulos sem
frequência nominal recebem `unknown`; não inferir frequência experimental de sample rate.

## 3. Próximo passo executado: diagnóstico na validação

Artefatos: `results/experiments/reconstruction_diagnosis_20261007/`.

Seed 2026, duas janelas por arquivo de **validação** de cada fold: 414 janelas de
207 arquivos, cobrindo 23 grupos. Cinco checkpoints por fold (base, best e last de
cada braço), 2.070 avaliações. Diagnóstico estratificado curto, não substitui métricas
na validação completa. O modelo de cada janela é aquele para o qual seu grupo é validação.

### Amplitude e espectro: best.pt

Médias por configuração/família. Erro FFT:
`||abs(FFT(recon)) - abs(FFT(original))|| / ||abs(FFT(original))||`.
RMS ratio ideal é 1; valores menores mostram atenuação da amplitude global.

| Família | RMS ratio uniforme / balanceado | Erro magnitude FFT uniforme / balanceado | RMSE uniforme / balanceado |
|---|---:|---:|---:|
| Doped | 0,545 / 0,547 | 0,760 / 0,749 | 0,877 / 0,869 |
| Multicore | 0,373 / 0,357 | 0,883 / 0,888 | 0,926 / 0,933 |
| Multimode | 0,283 / 0,367 | 0,912 / 0,883 | 0,985 / 0,987 |
| Single-mode | 0,570 / 0,573 | 0,726 / 0,730 | 0,808 / 0,812 |

Multimode balanceado recupera mais amplitude/magnitude, mas **não necessariamente
forma de onda**: RMSE piorou ligeiramente nesse subconjunto e alinhamento de fase
ponderado caiu de ~0,464 para ~0,414. Uma perda apenas de magnitude pode estimular
conteúdo sem fase correta; manter ancoragem temporal e monitorar fase/eventos.

Todas as janelas diagnosticadas têm `dt_s=5e-5` (20 kHz), duração 0,05 s e resolução
FFT 20 Hz. Bandas como 0–1 Hz não são resolvidas. Usadas bandas normalizadas à
Nyquist (10 kHz): baixa 0–10%, média 10–50%, alta 50–100%. Razão média de potência
reconstruída/original na banda alta: ~0,24/0,27 doped, 0,08/0,08 multicore,
0,06/0,12 multimode e 0,06/0,08 single-mode (uniforme/balanceado). São médias de
razões, não frações da energia total do corpus. FFT curta sem taper tem limitações
de leakage; este diagnóstico não certifica fidelidade física em todas as bandas.

### Quantização e codebook

Modelo: 2.048 códigos, quatro níveis de downsampling (fator 16 por eixo). Decodificado
também o latente contínuo, mantendo encoder/decoder e o mesmo padding/crop.

| Família | RMSE quantizado → contínuo, uniforme best | Balanceado best |
|---|---:|---:|
| Doped | 0,877 → 0,874 | 0,869 → 0,879 |
| Multicore | 0,926 → 0,920 | 0,933 → 0,930 |
| Multimode | 0,985 → 1,011 | 0,987 → 1,019 |
| Single-mode | 0,808 → 0,808 | 0,812 → 0,822 |

Não há salto que identifique o quantizador isolado como principal gargalo. **Não
prova que quantização/capacidade são irrelevantes:** decoder foi treinado em códigos;
latentes contínuos são uma entrada alterada, não um autoencoder contínuo treinado
nem um codec comparável.

Uso agregado não mostrou colapso global para poucos códigos. Médias entre folds no
best uniforme: ~43% doped, 42% multicore, 31% multimode e 39% single-mode.
Perplexidade média por janela multimode é baixa (~39 uniforme, ~44 balanceado),
mas depende também da estrutura do sinal/amostra; não justifica aumentar codebook
cegamente. Histogramas de modelos diferentes não foram fundidos como se seus
índices tivessem o mesmo significado.

### Inspeção visual e seleção de checkpoint

Geradas 46 figuras (primeira janela selecionada por configuração, por braço).
Inspecionadas diretamente: multimode Configurations 5 e 6, single-mode Configuration 4.
Nos exemplos multimode, original e erro preservam detalhes ausentes da reconstrução,
que é atenuada/suavizada. Single-mode recupera a estrutura oscilatória dominante,
mas perde detalhes em outros canais. Exemplos não representam todas as janelas.

Best versus last não revelou solução universal. No fold 2 balanceado, RMSE macro
desse subconjunto de validação caiu de 0,8761 (best, época 8) para 0,8684 (last,
época 10); no fold 3 balanceado piorou de 0,9454 para 0,9468. Justifica instrumentar
**seleção macro na validação completa**, não trocar checkpoints pelo resultado de teste.
Só best/last sobreviveram: não se pode escolher retrospectivamente a melhor época
macro dentre todas as épocas anteriores.

## 4. Sequência revisada e próximos experimentos

1. **Concluído:** amostradores e diagnóstico. Manter `window` como baseline;
   `family-group` continua candidato, sem promover modelo final.
2. **Próxima implementação:** métricas por época de RMSE/correlação macro por família
   e configuração, perda temporal separada de VQ e espectro na validação. Salvar
   também `best_macro.pt`, sem substituir o critério histórico; comparar seleção
   sobre o mesmo treino, sem mudar amostrador/perda simultaneamente.
3. **Prioridade de treino:** perda atual versus temporal + objetivo espectral
   multirresolução, preservando MSE/L1 e monitorando amplitude/fase. Primeiro piloto
   na validação, depois protocolo completo; baseline uniforme e coeficientes definidos
   só na validação. Não adotar por magnitude FFT ou correlação isoladamente. Auditar
   normalização global versus heterogeneidade dos canais sem remover partes do alvo.
4. **Geometria e rate-distortion sobem de prioridade**, antes de novos treinos longos
   de mistura parcial de sampler ou especialistas. Quatro níveis reduzem posições
   latentes ~256 vezes antes do padding; códigos de 11 bits sugerem compressão
   teórica muito agressiva de float32 (~745× no cálculo idealizado, sem metadados,
   padding ou modelo). Taxa efetiva depende da geometria/codec. Verificar se o orçamento
   de informação admite reconstruir detalhes/ruído; comparar curvas fidelidade × taxa.
   Se necessário, antecipar stride menor/multiescala antes de insistir só em pesos OOI.
5. **MoS/ensemble mantidos**, motivados pelos trade-offs, mas especialistas/roteamento
   não podem ser escolhidos pelo teste. Estimar custo e overfit com poucos grupos,
   metadados e fallback generalista. Não resolvem sozinhos taxa/capacidade/objetivo.
6. **Cobertura/datasets e treino do zero condicionais.** Não atribuir falha à quantidade
   de dados; curvas por grupos e teste de capacidade precedem grandes downloads.
   Diversidade FEBUS-like e arquitetura podem precisar mudar, por ablações rastreáveis.
7. **Modelo final adiado** até critérios explícitos por família/espectro e auditoria
   externa. Há reconstrução fraca em recortes importantes.

Uma semente, poucos grupos por família e testes já consultados limitam inferência.
Resultados futuros nesses testes são exploratórios; reservar nova aquisição/configuração
para auditoria final. Não tratar milhares de janelas como réplicas independentes.

## 5. Reprodução e artefatos

```bash
.venv/bin/python finetuning/analyze_sampling_results.py \
  --experiment-dir results/experiments/sampling_full_20261005 \
  --output-dir results/experiments/sampling_audit_new

.venv/bin/python finetuning/diagnose_reconstruction.py \
  --experiment-dir results/experiments/sampling_full_20261005 \
  --output-dir results/experiments/reconstruction_diagnosis_new \
  --samples-per-file 2

.venv/bin/python -m unittest discover -s tests -v
```

Diretórios de saída devem ser novos. Auditoria realizada em
`results/experiments/sampling_audit_20261007/`: `configurations.csv`, `files.csv`,
`paired_deltas.csv`, `training.csv`, `summary.json`. Diagnóstico realizado em
`results/experiments/reconstruction_diagnosis_20261007/`: seleção, métricas por
janela, codebook, checkpoints, resumo, figuras e marcador `COMPLETE`.

**21 testes passaram. Nenhum novo treino ou processo em background foi iniciado
nesta etapa; inferências diagnósticas terminaram e checkpoints foram preservados.**
