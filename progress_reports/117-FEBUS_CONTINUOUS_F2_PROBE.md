# FEBUS: curva F2 em trechos contínuos dos blocos brutos

**Protocolo prospectivo.** Continuação de [107](107-FEASIBILITY_CONTRACT_AND_EXECUTION_HANDOFF.md)
e da [revalidação temporal](110-FEBUS_TEMPORAL_CONTRACT_SENSITIVITY.md).
O usuário aceitou timestamps no centro do bloco e base comum entre arquivos e
canais, sem offsets de exportação. Não é certificação ou confirmação entre sessões.

## Escopo congelado

- Usar exclusivamente o [manifesto revalidado](../results/experiments/feasibility_centered_clock_revalidation_20261118/results.json):
  **20 fit, 10 check, 17.575.000 células**, uma leitura de bloco selecionado
  por arquivo, apenas as regiões retidas e os primeiros 5.000 samples.
  Não ler HDF5 de outros IDs, baixar dados, usar janelas NPZ como substitutas
  ou atravessar blocos. Verificar headers, coordenadas, linhagem e hashes
  antes de ler valores. Se faltar arquivo ou os headers divergirem, abortar.
- Incluir C2/C3; C1/C4 permanece sem contraste. Publicar populações por
  arquivo e configuração, sem inferir representatividade das dez configurações.
- Nove precisões do 107, `ε/σ_cal = .0001, .0003, .001, .003, .01, .03, .1,
  .3, 1`; primárias `.03` e `.1`. Escalas P0 congeladas. Marginal, temporal e
  espaço-temporal na mesma quantização, ordem causal, reset em 1.000 amostras.
  Fit por configuração, K escolhido no fit somente pelo marginal com custo
  instalado; mesma regra de contagem, smoothing e CDF do 107. Se a geometria
  varia, calcular vizinhança em cada trecho e pagar bindings/instalação,
  sem renumerar canais ou concatenar valores para fabricar dependência.
- Medir bytes reais de transporte e objeto com instalação, SSE/RMSE por
  configuração, erro máximo, escapes, tempo e memória. Validar hashes de
  índices e waveform entre coders, decoder causal e bound. Ler os mesmos
  valores uma vez e reutilizá-los; não selecionar pontos pelos checks.
- CPU, teto de 90 minutos. Sem GPU, parâmetros neurais ou reservas.
  Se houver interrupção, preservar o progresso e marcar inconclusão; não
  modificar a seleção nem declarar sucesso com uma curva incompleta.

## Decisão congelada

Aplicar os critérios do 107 nos dois pontos primários, incluindo os custos
completos e guardas por família/configuração. Qualquer aprovação será
**condicional à cobertura realmente medida**, sem declarar C1/C4 testado.
Resultado favorável em uma precisão ou regime é evidência localizada.
F4-A não é disparado pelo runner; sua justificativa depende de revisão
científica da curva, e F3 ainda depende de novas sessões documentadas.

## Resultado

Foram processadas **17.575.000 células**, com 5.905.000 no check, em nove
configurações. As nove precisões geraram 81 pontos configuração–precisão e
243 braços. Execução de 869 s em CPU, pico RSS de aproximadamente 607 MiB.
**50 testes passaram**, 729 ZIPs foram conferidos e não houve violação de ε.
Os índices e waveforms quantizados coincidiram por bits entre os coders.
[Resultados](../results/experiments/feasibility_continuous_probe_20261118/results.json)
e [evidência de conclusão](../results/experiments/feasibility_continuous_probe_20261118/completion_evidence.json).

| ε/σ_cal | Economia no transporte | Economia com instalação uma vez | Economia em objetos autocontidos |
|---|---:|---:|---:|
| 0,03 | 30,38% | 0,83% | −8,57% |
| 0,1 | 41,27% | 27,41% | 22,85% |

Comparação espaço-temporal contra marginal, mesma reconstrução.
Economia negativa indica aumento de bytes. As quatro famílias economizaram
no transporte, mas o efeito com instalação foi heterogêneo. A 0,03,
doped, multicore C2/C3 e multimode regrediram com instalação.
A 0,1, o agregado com instalação passou o mínimo de taxa, mas doped
e multimode regrediram. **O critério conjunto não passou.**

Há dependência causal aproveitável nesse painel, mas o ganho elevado de
transporte não equivale a ganho generalista de implantação. C1/C4 não foi
testado; não houve confirmação de sessões. O F2 contínuo está **concluído
no desenho revalidado**, sem F4-A disparado, sem treino e sem reservas.
