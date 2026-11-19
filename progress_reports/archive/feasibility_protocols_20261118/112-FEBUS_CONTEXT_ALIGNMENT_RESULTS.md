# FEBUS: alinhamento de contexto melhora o controle, mas não supera a referência

**18/11/2026.** Resultado do [protocolo 111](111-FEBUS_TRAIN_INFERENCE_CONTEXT_ALIGNMENT.md).
O protocolo congelado permanece inalterado.

## Resultado a cobertura e taxa medidas

Avaliação **full-tile** nos mesmos 136 checks históricos de treino, 43 arquivos
e 12.799.000 células por modelo. Menor NRMSE* macro é melhor. A taxa de
transporte inclui bundles e geometria, com receiver e registry instalados.

| Modelo | NRMSE* macro | SSE físico | Transporte | bpc |
|---|---:|---:|---:|---:|
| Controle `part8-train` | 1,04108 | 1,2136 × 10¹⁸ | 1.101.697 B | 0,68861 |
| Candidata `full-train` | 0,62277 | 5,4896 × 10¹⁷ | 1.150.137 B | 0,71889 |
| VQ madura | 0,49034 | 3,9281 × 10¹⁷ | 1.119.421 B | 0,69969 |

- **Contra o controle pareado:** −40,18% de NRMSE* e melhora nas quatro
  famílias, no SSE e em todas as guardas de cauda e espectro predefinidas.
  A taxa aumentou **4,40%**; o critério de bytes foi o único a falhar.
- **Contra a referência prática:** **+27,01% de NRMSE*** e **+2,74% de bytes**.
  A VQ madura também teve menor SSE e menor NRMSE* em todas as famílias.
  O candidato não deslocou a fronteira medida.
- Com instalação cobrada, os objetos totalizaram 115.112.539 B (`part8-train`),
  115.160.979 B (`full-train`) e 115.129.685 B (VQ madura), sem o runtime
  compartilhado. A conclusão de taxa não depende de omitir a instalação.

## Execução e integridade

| Braço | Updates | Apresentações de células | Posições de índices durante treino | Tempo com avaliação |
|---|---:|---:|---:|---:|
| `part8-train` | 8.192 | 592.255.000 | 77.062.000 | 59 min 52 s |
| `full-train` | 8.192 | 592.255.000 | 41.988.000 | 29 min 39 s |
| VQ madura | 0 | — | — | 21 s |

Os dois treinos tiveram inicialização, schedule e exposição física idênticos.
O controle usou mais posições de índices no **treino**; isso não é a taxa
de inferência. O custo computacional não foi pareado por FLOPs. As duas
GPUs remotas estavam ocupadas em todas as checagens; cada execução usou
somente a RTX 4060 local. O pico alocado da candidata foi de cerca de 3,20 GiB.

Conferidos os hashes dos **408 bundles**, das avaliações usadas pela
[decisão automática](../results/experiments/feasibility_context_alignment_20261118/decision.json)
e dos schedules. Todos os bundles passaram na redecodificação bit a bit.
Não houve reseleção de checkpoint nem uso de validação, teste ou EDAS.

## Decisão

**Encerrar este contraste F4-B sem promoção, extensão ou sweep.** Em igual
dose de células, a política full-tile teve um efeito grande sobre a fidelidade
contra o controle de crop. Isso sustenta cuidar da compatibilidade de
contexto em pesquisas futuras, mas não isola a contribuição da normalização,
bordas ou latentes e não demonstra eficiência por bit. O resultado é de
desenvolvimento, não generalização a sessões independentes.

A VQ madura permanece referência; P3 não é elegível. F2 FEBUS continua
bloqueado pelo [contrato temporal](110-FEBUS_TEMPORAL_CONTRACT_SENSITIVITY.md),
e F3, pela falta de sessões documentadas. Não iniciar F4-A sem evidência
de economia condicional no FEBUS.
