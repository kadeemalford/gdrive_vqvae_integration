# Índices lossless e ROI: resultados e continuação de empacotamento/caudas

## 1. Estudo inicial concluído

`frozen_entropy_roi_fold0_20261030` concluiu **7.220 janelas** em 30/10/2026,
13:46 UTC, com 802,5 s de execução. Nenhum treino, teste reservado ou promoção.

- Compacto: payload fixo de índices **15.262.320 bytes → 13.500.181 bytes**,
  redução **11,55%**; marginal escolhido em 6.219 janelas, condicional em 170,
  fallback fixo em 831.
- Denso: **60.555.000 → 54.613.299 bytes**, redução **9,81%**; marginal em 6.867,
  condicional em 353. Todas as sequências de índices e side streams passaram no
  roundtrip exato. Não houve perda adicional de reconstrução por codificação.

| Modo | RMSE macro | Correlação | Razão com mapas/metadados textuais | Células densas |
|---|---:|---:|---:|---:|
| Compacto | 0,667586 | 0,727658 | 131,347× | 0% |
| Denso | 0,445187 | 0,888093 | 42,055× | 100% |
| Uniforme, teto 1,5× | 0,628247 | 0,755609 | 95,372× | 17,2% |
| ROI RMSE, teto 1,5× | 0,624483 | 0,759692 | 95,383× | 17,9% |
| ROI com guarda, teto 1,5× | 0,624518 | 0,759531 | 95,384× | 17,9% |
| Uniforme, teto 2× | 0,587852 | 0,784500 | 74,913× | 35,0% |
| ROI RMSE, teto 2× | 0,582433 | 0,790142 | 74,926× | 35,9% |
| ROI com guarda, teto 2× | 0,582450 | 0,789984 | 74,924× | 35,9% |

Essas razões cobram tabelas, headers, normalização/escapes, mapas e metadados;
não são a mesma contabilidade das razões nominais históricas ~160×/~42×.
A redução do payload dos índices não impede overhead total elevado.

As guardas descritivas RMSE/correlação/p90 por configuração passaram contra o
compacto em todos os modos mistos. Isso não é aceitação física. A guarda ROI
é computada pelo encoder que possui o original; sua monotonicidade é propriedade
da regra e não evidência independente de generalização.

## 2. O que limita o avanço

### Metadados redundantes

A serialização textual conservadora consome **4.677.446 bytes**:
3.677.938 do manifesto, 999.182 dos mapas e 326 da política. As tabelas dos
símbolos somam ~108 kB por modelo. São pequenos frente ao bruto, mas importantes
frente a 14,6 MB de frames compactos. A próxima etapa não removerá coordenadas:
testará transporte lossless de toda essa informação.

### Caudas e objetivo de alocação

O p90 macro continua numericamente patológico nessa coorte: ~5,013 milhões no
compacto. ROI por RMSE praticamente não o altera, enquanto o uniforme 2× o reduz
para ~3,831 milhões. A regra guarda permite evitar regressão sem necessariamente
priorizar a melhoria de caudas. Não declarar ROI superior em todos os critérios
pela pequena vantagem de RMSE/correlação.

Pesos FP32 compartilhados foram cobrados separadamente uma vez: no compacto,
a razão dessa coorte curta cai de 131,347× para ~19,385× ao incluir pesos. Para
modos mistos são necessários os dois modelos; o custo amortizado depende do volume
real de operação. Não embutir os pesos em cada janela, nem omitir sua implantação.

## 3. Continuação predefinida: transporte e ganho relativo de cauda

Implementações: `compression/study_bundle.py` e
`finetuning/run_roi_tail_and_bundle.py`.
Diretório: `results/experiments/roi_tail_bundle_fold0_20261030/`.

**Esta é nova ablação retrospectiva de desenvolvimento**, motivada pelo resultado
anterior; não retuna suas regras, não reescreve seus resultados e não serve de
confirmação independente. Sem GPU, sem inferência nova, sem atualização de pesos.

1. Empacotar manifesto, mapas, política, ordem dos frames e tabelas em um arquivo
   ZIP_DEFLATED nível 9, com verificação de bytes originais após descompressão.
   O uso lossless de um container não substitui o codec neural de reconstrução.
2. Transmitir frames com comprimentos explícitos num bundle `.fsb`, cobrando header
   e comprimento de cada frame. O decoder consegue delimitar todas as janelas sem
   NPZ original, nomes externos ou catálogo gratuito.
3. Criar referências **fixas de 11 bits no mesmo container**. Índices e side são
   recuperados e reemitidos sem alterar o alvo. Comparação justa do ganho lossless
   exige não confrontar novos custos completos com a taxa nominal antiga.
4. Reempacotar todos os controles do primeiro estudo sem mudar sua seleção.
5. Nova ROI `balanced`: metade do ganho fracional de RMSE² e metade do ganho
   fracional de p90, cada um limitado a [0,1], ponderados por células e custo
   adicional. Epsilons fixos 1e-12/1e-6. Mesmos tetos 1,5×/2× por arquivo e
   admissão sem piora de RMSE/correlação/p90 contra o compacto. Não ajustar pesos
   depois de observar a resposta.
6. Verificar byte a byte todos os frames transportados e todos os metadados;
   reportar taxa efetiva, metadados, guarda, espectro e custo dos modelos.

Um ganho fracional limita a influência de valores gigantes de canais fracos,
sem removê-los do alvo. Ainda não representa tolerância física ou garantia para
eventos fracos. Comparar também as métricas não fracas e amplitude/fase; uma melhora
de cauda à custa de RMSE pode ser um trade-off, não dominância.

**Decisão seguinte:** avaliar se a economia lossless permanece no container real e
se a mudança de prioridade da ROI melhora o compromisso, antes de investir em
refinamento residual treinável ou novo quantizador. Não iniciar outra rede maior
com base nesses resultados. Eventual treino posterior usará a RTX 4060 local.

## 4. Continuação concluída: transporte e limites da ROI

Todos os bundles/metadados foram recuperados byte a byte. No **mesmo container**:

| Modo | Razão com índices fixos | Razão com Huffman/fallback |
|---|---:|---:|
| Compacto | 153,680× | **171,994×** |
| Denso | 41,127× | **45,499×** |

Manifesto, mapas, política, sequência e tabelas compactos ocupam **181.156 bytes**
no archive básico e 194.866 bytes nos mistos. Headers e comprimentos dos 7.220
frames também são cobrados. Não se excluiu informação, canal ou ruído.

A nova `balanced_1p5` atingiu **0,625183 / 0,757330** em RMSE/correlação e
**115,597×**, com p90 macro ~3,981 milhões contra ~5,013 milhões da ROI por RMSE.
No teto 2×, seu p90 voltou a ~4,858 milhões, contra ~3,831 milhões do uniforme.
Logo, a nova regra **não domina** todos os critérios. Não selecioná-la como
vencedora universal após observar a validação.

O p90 não monotônico entre tetos evidencia um limite da alocação gulosa refeita
separadamente: o orçamento maior pode trocar refinamentos antes selecionados
por outros. Um codec progressivo deve preservar a seleção de menor orçamento ao
expandir; essa propriedade será testada no próximo piloto, sem reescrever este.

Dois arquivos multimode dominam a cauda relativa: p90 médio por arquivo de ~93,9
milhões e ~32,7 milhões, enquanto o RMSE médio dos canais não fracos fica em ~0,83
em ambos. Isso não autoriza remover canais, mas impede interpretar milhões como
erro uniforme de todos. Detalhes em `multimode_tail_file_diagnostics.json` do
estudo inicial. Erros físicos e caudas permanecem critérios concorrentes.

O próximo piloto é separado no
[relatório 69](69-PROGRESSIVE_RESIDUAL_REFINEMENT_PROTOCOL.md): base compacta
congelada e rede menor para codificar o erro, não outro aumento indiscriminado
de parâmetros nem promessa de ganho antes das avaliações.
