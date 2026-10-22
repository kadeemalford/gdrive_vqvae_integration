# Reativação de códigos: resultado negativo para promoção generalista

## 1. Resultado e decisão

O experimento foi concluído, incluindo avaliação completa; a sessão de espera
cancelada não implica perda de resultado. Fontes:
`results/experiments/codebook_refresh_long_20261011/`, `REPORT.md`, histories e
`full_validation/summary.csv`, tabelas por grupo/janela e `COMPLETE`.

**Não promover refresh como default generalista.** A utilização aumentou muito,
mas o ganho de RMSE macro foi pequeno e houve regressão do pior grupo/canais.
O controle selecionou a inicialização epoch 0, mostrando ausência de ganho no
seletor com a nova continuação sem intervenção.

## 2. Validação completa dos selecionados

| Candidato | Epoch | RMSE macro | Correlação macro | Pior família RMSE |
|---|---:|---:|---:|---:|
| Transferido congelado | 13 histórico | **0,719900** | **0,682337** | **0,817242** |
| Temporal source / controle selecionado | 31 histórico / 0 novo | 0,729713 | 0,673097 | 0,831743 |
| Temporal refresh | 24 novo | 0,724288 | 0,681971 | **0,866028** |

Cada braço recebeu 98.304 updates adicionais, mesmo source, optimizer novo/LR
3e-5 e objetivo; somente refresh divergiu. Scores acima são das mesmas 5.820
janelas. Seleção usou o piloto de 102 janelas; não é teste independente.
Taxa nominal pooled segue 165,914x, com mesmo K/parametrização; não inclui codec completo.

## 3. Configurações: ganho não uniforme

| Família/configuração | Source RMSE | Refresh RMSE | Refresh correlação |
|---|---:|---:|---:|
| Doped / 5m SMF | 0,695047 | 0,678486 | 0,738267 |
| Multicore / C1 and C4 | 0,813047 | 0,801501 | 0,597160 |
| Multimode / Configuration 7 | 0,831743 | **0,866028** | 0,582216 |
| Single-mode / Configuration 11 | 0,444085 | **0,332377** | **0,938986** |
| Single-mode / Configuration 3 | 0,698003 | **0,744081** | 0,672166 |
| Single-mode / Configuration 7 | 0,594957 | 0,576953 | 0,819575 |

Configuration 11 ganha muito, mas isso não compensa automaticamente as falhas em
multimode e Configuration 3. Regressões aparecem também no piloto, não apenas
na ampliação de avaliação.

## 4. Quantização e canais

Na época 24, control: **337 códigos / perplexidade 228,7**; refresh:
**1.004 códigos / perplexidade 520,5**, reativando 32 códigos nessa época.
Contagens de treino antecedem a reativação; scores/save medem o estado depois.

RMS ratio macro source→refresh **0,730590→0,797174**, mas p90 macro por canal
**1,238167→1,245205** e erro não fraco **0,952813→0,962960** pioram. Quantis são
estatísticas por janela agregadas macro, não pooled de todos os canais.

Mais códigos e menor perda VQ não demonstram fidelidade maior em todo o domínio.
O resultado limita a hipótese de codebook restrito como explicação única.

## 5. Próxima hipótese: especialização versus exposição

Persistem ganhos/regressões incompatíveis entre famílias/configurações. Avançar
para a etapa de especialistas por metadados, com fallback e controle compartilhado.
Para não confundir especialização com mais exposição a famílias minoritárias,
incluir generalista com dose familiar igual à dos especialistas somados.

Não aumentar indefinidamente a continuação refresh nem promover memorizadores.
Preservar source, refresh e transferido como referências de trade-offs. Protocolo
e testes da próxima comparação no [relatório 27](27-DOMAIN_SPECIALISTS_AND_BALANCED_CONTROL_PROTOCOL.md).
