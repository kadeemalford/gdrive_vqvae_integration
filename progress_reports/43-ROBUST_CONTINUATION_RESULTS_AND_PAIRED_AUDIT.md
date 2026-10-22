# Continuação robusta: ganhos modestos e regressões localizadas

## 1. Resultado completo à mesma taxa

Fontes: `results/experiments/robust_rate_continuation_20261018/`, `COMPLETE`,
`full_validation/COMPLETE`, summaries/groups/windows e históricos completos.
Ambos os braços concluíram epoch 40; selecionados macro também são epoch 40.
Cada braço recebeu 65.536 updates adicionais, com optimizer/RNG preservados.

| Braço | RMSE epoch 24 → 40 | Correlação 24 → 40 | p90 macro por canal 24 → 40 | Compressão pooled |
|---|---:|---:|---:|---:|
| 2x | 0,592612 → **0,577962** | 0,791020 → 0,802679 | 0,937458 → 0,925811 | 84,228x |
| 4x | 0,479352 → **0,466790** | 0,869126 → 0,875729 | 0,717070 → 0,706400 | 42,811x |

Ganhos relativos de RMSE ~2,47% / 2,62%, menores que os ganhos ao aumentar taxa.
Nem amplitude nem todas as configurações melhoram junto com a média.
4x RMS macro **0,901455→0,898395**, fase FFT 0,929120→0,933915.
2x RMS macro 0,820055→0,827215, fase 0,875982→0,885390.

## 2. Configurações completas

| Família/configuração | RMSE 2x: 24 → 40 | RMSE 4x: 24 → 40 |
|---|---:|---:|
| Doped / 5m LSF | 0,537380 → 0,524890 | 0,433689 → 0,419693 |
| Multicore / C1 and C3 | 0,671921 → 0,658105 | 0,526922 → 0,515453 |
| Multimode / Configuration 6 | 0,695120 → 0,671813 | 0,566143 → 0,548358 |
| Single-mode / Configuration 10 | 0,282286 → 0,262637 | 0,215361 → 0,208888 |
| Single-mode / Configuration 2 | 0,688288 → 0,680327 | **0,604817 → 0,612373** |
| Single-mode / Configuration 6 | **0,427504 → 0,428153** | 0,351781 → 0,329706 |

4x Configuration 2: correlação **0,794035→0,790743**, p90 por canal
**0,844405→0,860627**. A perda não é apenas uma interpretação da escala de RMSE.
Configuração 2 permanece a pior configuração; multimode é a pior família média.
Multicore/multimode 4x têm RMS **0,843829/0,832533**, ainda subestimando amplitude.

## 3. Auditoria pareada das 5.920 janelas por braço

Runner: `finetuning/audit_rate_continuation.py`.
Evidência: `results/experiments/robust_rate_paired_audit_20261018/`.
Sem inferência, updates ou acesso a teste. Pareamento por caminho canônico, não
posição da linha. Rejeita duplicatas, coordenadas diferentes, métricas não finitas
e divergências de metadados, side, índices e diagnósticos da referência.
Confere coordenadas/metadados contra o manifesto de validação.

| Caso de regressão | Janelas | Fração com RMSE pior | Arquivos com média pior | Delta médio RMSE |
|---|---:|---:|---:|---:|
| 2x / single-mode Configuration 6 | 480 | 44,8% | 4/6 | +0,000649 |
| 4x / single-mode Configuration 2 | 380 | **78,4%** | **5/6** | **+0,007557** |

Bootstrap descritivo de **5.000 reamostragens de arquivos**, mantendo janelas
agrupadas e pesos por quantidade de janelas: intervalo 2,5–97,5% para delta
2x Configuration 6 **[-0,003580; +0,004611]**; 4x Configuration 2
**[+0,000763; +0,015295]**. Não há evidência de independência entre arquivos,
que podem incluir cópias/correlação; esses intervalos **não** são comprovação
inferencial de generalização para novas aquisições nem um teste confirmatório.

As outras configurações têm delta médio negativo. Mesmo assim, 18,3% das janelas
4x de Configuration 10 pioram enquanto a média melhora. Não confundir melhora
de grupo com melhora universal por janela/canal.

As regressões são de janelas single-mode sem a patologia de faixa dinâmica
multimode. Strata registrados incluem `dt_s` real do manifesto e canais:
Configuration 2 possui 111 canais, Configuration 6 possui 56, ambas dt=5e-5 s.
Essa associação não prova causalidade por geometria; não trocar arquitetura com
base apenas nela. CSVs por arquivo/stratum/janela permanecem disponíveis.

## 4. Limite do seletor e decisão

Na amostra de duas janelas por arquivo, epoch 40 do **2x melhorava RMSE nas seis
configurações** frente a epoch 24; a expansão completa revelou regressão pequena
em Configuration 6. No 4x, a amostra já mostrava regressão de Configuration 2,
mas o seletor macro escolheu epoch 40 mesmo assim. Portanto coexistem:

1. Representatividade limitada da amostra de seleção.
2. Seletor macro que admite perdas por configuração.

Históricos têm épocas intermediárias sem regressão na amostra, porém seus pesos
não estão disponíveis; não alegar que possam ser recuperados/reavaliados. Os
`best.pt` preservados de ambos os braços são epoch 38, selecionados por loss,
não por fidelidade por configuração. São controles, não alternativas presumidas
superiores. Preservar 24, 38 e 40; nenhum checkpoint é promovido ou removido.

**Não continuar epochs indefinidamente.** Próximo passo é um piloto congelado de
estabilidade: avaliar retained loss-selected e midpoint fixo de pesos 24/40, sem
updates ou aumento de bitrate/modelo. Não supor que média de VQ embeddings seja
benéfica. Protocolo no [relatório 44](44-FROZEN_RATE_STABILITY_PROTOCOL.md).
Para futuros treinos, considerar retenção adicional por critérios de pior
configuração e seleção mais representativa; não reescrever retrospectivamente
o seletor dos estudos concluídos ou transformar validação reutilizada em teste.
