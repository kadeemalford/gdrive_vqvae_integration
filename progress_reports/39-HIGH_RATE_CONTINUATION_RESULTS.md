# Continuação 4x: ganho adicional menor e confirmação entre configurações

## 1. Resultado completo

Fonte: `results/experiments/high_rate_continuation_20261015/`, histories,
`REPORT.md`, `full_validation/summary.csv`/groups/windows e `COMPLETE`.
Epochs 25–40 concluídos, 65.536 updates adicionais com optimizer/RNG preservados.
O selecionado foi **epoch 39**, não last 40.

| Candidato | RMSE macro completo | Correlação | Pior família RMSE | Compressão nominal |
|---|---:|---:|---:|---:|
| Rate 4x anterior, epoch 24 | 0,490010 | 0,865562 | 0,658187 | 42,133x |
| Rate 4x continuado, epoch 39 | **0,477692** | **0,871184** | **0,650614** | **42,133x** |

Mesmas 5.820 janelas do fold 1 e mesma máscara/normalização/taxa: ganho de
~2,51% relativo no RMSE, sem novos bits. O ganho é menor que o obtido ao aumentar
o orçamento de índices. Não há prova de saturação definitiva, mas é preferível
confirmar a curva em outras configurações a repetir indefinidamente esse split.

## 2. Configurações do selecionado

| Família/configuração | RMSE relativo | Correlação | p90 relativo por canal |
|---|---:|---:|---:|
| Doped / 5m SMF | 0,426904 | 0,904167 | 0,758991 |
| Multicore / C1 and C4 | 0,503079 | 0,862803 | 0,627613 |
| Multimode / Configuration 7 | **0,650614** | **0,775780** | 0,904515 |
| Single-mode / Configuration 11 | **0,222695** | **0,974268** | **1,169678** |
| Single-mode / Configuration 3 | 0,398487 | 0,923926 | 0,818968 |
| Single-mode / Configuration 7 | 0,369333 | 0,927769 | 0,859263 |

Melhora de RMSE nas seis configurações frente ao 4x anterior. Multimode permanece
limitante e o p90 de Configuration 11 mostra que bom score global não garante
fidelidade em cada canal. Nenhum modelo final promovido; tolerâncias físicas e
bitrate aceitável ainda não definidos pela aplicação.

## 3. Próximo passo

Confirmar 1x/2x/4x no fold 0 com source exclusivamente desse fold e **referência
robusta comum**. Não retreinar pela proteção isoladamente: os inputs de treino
auditados são invariantes. O motivo do novo treino é testar a curva de taxas
em outras configurações; a proteção corrige o domínio de avaliação problemática.

Protocolo no [relatório 40](40-HIGH_RATE_FOLD0_ROBUST_PROTOCOL.md). Sem score
de teste ou interpretação como auditoria externa independente.

### Correção de proveniência da justificativa anterior

Conferindo conjuntos dos manifestos: treino do fold 1 **não** cruza a validação
do fold 0, mas cruza seus grupos de **teste**. A frase anterior que atribuía a
sobreposição à validação estava incorreta. A decisão de usar source específico
do fold continua correta para preservar a separação completa train/val/test.
O novo protocolo não usa pesos do fold 1 no fold 0, nem avalia teste.
