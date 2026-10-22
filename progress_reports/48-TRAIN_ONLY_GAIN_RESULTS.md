# Calibração global train-only: ganho quase unitário e benefício insuficiente

## 1. Estudo concluído

Fonte: `results/experiments/frozen_gain_calibration_20261018/`, marcadores
`COMPLETE` e `full_validation/COMPLETE`, fit cursors, summaries/groups/windows e
`charged_rate.json`. Ajuste em **9.720 janelas de treino por modelo**; comparação
em **5.920 janelas de validação**, sem SGD ou updates da rede. O serviço retomado
terminou com exit 0 em ~39 min, após coleta parcial da execução anterior.

| Candidato | Ganho ajustado no treino | MSE train identidade → calibrado |
|---|---:|---:|
| Midpoint 2x | 1,006704926 | 0,312568581 → 0,312538085 |
| Epoch 24 4x | 0,999005556 | 0,200707202 → 0,200706410 |

O ótimo do ajuste global é praticamente identidade. No 4x, o ajuste escolhe
**reduzir**, não amplificar, a saída. A hipótese de que RMS subestimado exige
amplificação global não se confirmou.

## 2. Validação completa

| Taxa | RMSE macro identidade → calibrado | p90 macro por canal | RMS macro |
|---|---:|---:|---:|
| 2x | 0,583174 → **0,583473** | 0,929726 → **0,930832** | 0,827074 → 0,832621 |
| 4x | 0,479352 → 0,479285 | 0,717070 → 0,716937 | 0,901455 → 0,900559 |

2x aproxima RMS de 1, mas piora RMSE e cauda. Isso demonstra que casar amplitude
global não equivale a reconstruir o sinal perdido. 4x tem ganho absoluto de RMSE
de apenas ~0,0000664 (~0,014% relativo), com amplitude ligeiramente mais atenuada.

Correlação macro permaneceu 0,798344 no 2x e 0,869126 no 4x; fase FFT 0,881931 e
0,929120, respectivamente, com diferenças numéricas diminutas, como esperado de
ganho positivo. O escalar não recupera forma de onda, correlação ou fase.

## 3. Limites por configuração/canal

- **2x:** RMSE pior em cinco de seis configurações; p90 por canal pior nas seis.
  A maior perda de RMSE é single-mode Configuration 2: +0,001142; p90 +0,003179.
  Multicore melhora RMSE apenas ~0,000034, sem preservação de cauda.
- **4x:** RMSE melhora em cinco configurações, mas multicore piora ~0,00000165.
  P90 single-mode Configuration 10 piora ~0,00000069. São deltas muito pequenos:
  não exagerar relevância física sem tolerâncias de aplicação. O ponto principal
  é que o ganho macro também é diminuto e não elimina os limites já observados.

Um único ganho global não explica o erro residual destes candidatos. Isso não
prova que toda calibração local seja inútil; tampouco autoriza ampliar busca por
canal/configuração usando a mesma validação.

## 4. Taxa e decisão

Somados conservadoramente 32 bits globais por modelo, razões pooled:
2x **84,228320x**, 4x **42,811176x**, praticamente as razões originais.
Não houve ganho adaptativo por janela nem mudança de índices/codebook; apenas
o último head linear foi escalado no artefato derivado. A taxa não inclui todo
o container/modelo de um codec implantado.

**Não promover a calibração global** nem torná-la default. Preservar resultados
e checkpoints derivados como evidência negativa, mantendo os sources intactos.
Não iniciar novos treinos para corrigir apenas esse escalar.

Próximo diagnóstico isolado: quantização versus latentes contínuos no decoder
congelado, com custo do controle explicitamente cobrado e ressalva de mudança
de distribuição. Protocolo no [relatório 49](49-FROZEN_QUANTIZATION_PROBE_PROTOCOL.md).
Ainda são dados de desenvolvimento reutilizados, não confirmação em aquisições
independentes ou aceitação física final.
