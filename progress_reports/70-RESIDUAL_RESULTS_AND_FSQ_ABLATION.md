# Residual VQ sem ganho relevante; ablação FSQ a bits nominais iguais

**FSQ concluído:** houve extensão até 24 épocas, sem promoção final. Resultado
e controle de bytes/modos estão no
[relatório 71](71-FSQ_RESULTS_AND_MATCHED_MODE_COMPARISON.md).

## 1. Piloto VQ encerrado

`residual_refinement_fold0_20261030` concluiu em **30/10/2026 às 14:31 UTC**.
Recebeu **16.384 updates** e avaliação completa. A regra predefinida recusou
extensão: ganho recente de triagem ~0,00143%, ganho de triagem contra base ~0,0417%,
ambos insuficientes. Checkpoint selecionado na **época 14**, não na última.

| Sistema | RMSE macro | Correlação | Razão real do bundle |
|---|---:|---:|---:|
| Base | 0,667586 | 0,727658 | 171,936× |
| Todos refinados | 0,667287 | 0,727834 | 162,479× |
| Refinamentos com guarda | 0,667322 | 0,727819 | 167,318× |

A redução de RMSE é **0,0447% sem guarda / 0,0395% com guarda**. Com guarda,
50,6% das células recebem refinamento; ambos os tetos 1,25×/1,5× comportaram todos
os refinamentos admitidos e deram o mesmo resultado. Não confundir isso com
atingimento dos requisitos físicos. Não houve promoção.

Uso de códigos na época 1: 467; época 4: 5; época 16: **3**. O ganho pequeno e
esse colapso tornam injustificado apenas prolongar o treino ou ampliar a rede.
Não demonstram que todo resíduo seja incompressível, nem que FSQ necessariamente
resolverá o problema. O piloto tem exposição limitada e menor resolução residual.

## 2. Próxima hipótese isolada: família de quantização

Implementação: `finetuning/fsq_quantizer.py`, integrada ao runner residual por
`--quantizer fsq-8x8x32`. Não trocar corpus, filtro, base, strides, seed, sampler,
orçamento ou objetivo de reconstrução simultaneamente.

- FSQ de **8×8×32 = 2.048 estados**, índice de 11 bits; mesmos 2.016 índices de
  refinamento no shape máximo. Radix misto reversível sem features do encoder.
- Projeção aprendida 128→3, bins fixos via sigmoid/arredondamento/STE e projeção
  aprendida 3→128. Decoder recupera os três dígitos só pelo índice transmitido.
- Não há tabela VQ aprendida, reseeding ou loss de compromisso/codebook; a perda
  auxiliar é zero. É comparação entre famílias de quantizadores, **não** apenas
  mudança de optimizer com todas as demais equações idênticas.
- A troca da tabela 2.048×128 pelas projeções reduz o refiner de 7.355.873 para
  **7.094.628 parâmetros (~3,55%)**. Declare essa diferença, sem atribuir eventual
  resultado exclusivamente à utilização dos estados.
- O encoder/decoder nativo recebe a mesma inicialização de seed 2035 antes da
  troca do módulo. FSQ não requer semear codebook; os 64 exemplos de calibração
  continuam iguais para ajustar tabelas lossless após seleção.
- Configuração dos níveis usa cada dimensão ≥5, conforme orientação qualitativa
  de FSQ/ICLR 2024, mas **8×8×32 e sigmoid são escolhas de engenharia para manter
  exatamente 2.048 estados**, não uma receita publicada como ótima para DAS.

## 3. Orçamento e admissão

O restante segue o relatório 69: **16×1.024 updates**, extensão única de oito
épocas apenas com ganho recente ≥0,2% e ≥1% contra a base; FP32 na **RTX 4060
local**, nenhuma GPU remota. Validação por época, seleção por RMSE macro do sistema
com guarda e avaliação completa dos selecionados em streams reais, com controles
sem guarda, fallback, alocação aninhada e pesos/overhead explícitos.

Comparar primeiro fidelity/bytes com o **VQ residual congelado** e com a base,
não apenas número de códigos ou loss total: VQ e FSQ têm perdas auxiliares
diferentes. Registrar reconstrução e auxiliar separadamente na nova execução.
Mais estados usados sem ganho físico não é critério de avanço.

Código do piloto VQ foi preservado em seu `code_snapshot/` antes da evolução
do runner. Resultados anteriores não são reescritos. Testes verificam todos os
2.048 estados decodificáveis, replay exato, gradientes e checkpoint FSQ.

Diretório novo: `results/experiments/residual_refinement_fsq_fold0_20261030/`.
Unidade: `febus-residual-fsq-20261030.service`. Nenhum teste reservado ou promoção
final. Se FSQ não melhorar a fronteira, encerrar essa hipótese sem nova busca
de níveis na mesma validação; se melhorar, priorizar confirmação por grupos e
validação física antes de integrar.

### Execução iniciada

O smoke FSQ passou na RTX 4060: **7.094.628 parâmetros**, 2.016 índices residuais
no shape máximo, base sem gradientes e pico alocado de **1.456.584.704 bytes**.
Treino confirmado ativo, época 2/16, em 30/10/2026 por volta de 14:39 UTC.
Suite completa: **238 testes aprovados**. Fontes/dados permanecem congelados
durante essa execução; métricas posteriores devem vir dos artefatos, não deste
registro de início.
