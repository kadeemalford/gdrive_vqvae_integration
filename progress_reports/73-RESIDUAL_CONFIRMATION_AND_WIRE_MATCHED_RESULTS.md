# Confirmação residual e controle por bytes reais concluídos

## Confirmação em outra rotação

`residual_confirmation_fold1_20261030` terminou em **30/10/2026 às 19:19 UTC**.
A nova base compacta completou 24×4.096 updates (~2,99 h de épocas); o refiner
FSQ completou 24×1.024 (~0,47 h), além de streams/avaliações. Só a RTX 4060 local
foi usada, sem teste reservado ou promoção final.

| Sistema | RMSE macro | Correlação | p90 canal macro | Razão real |
|---|---:|---:|---:|---:|
| Base compatível | 0,648189 | 0,749319 | 1,024954 | 164,997× |
| Todos os tiles refinados | 0,626616 | 0,768191 | — | 127,842× |
| Refinamento com guarda | 0,633530 | 0,762865 | 1,004680 | 134,243× |

O benefício contra a base se repete, mas continua pagando bits adicionais. O
híbrido não pode ser promovido com base nessa melhora: a comparação pertinente
é com modo denso à taxa real equivalente, não apenas com a base compacta.
Os folds têm configurações distintas; não comparar diretamente RMSE ou médias
de p90 entre eles como melhora de um único modelo.

## Controle novo: tamanho real do stream, não só teto

`finetuning/run_wire_matched_hybrid.py` usa reconstruções congeladas e os mesmos
critérios de admissão/ganho por byte. Em cada ponto, toma o stream denso anterior
como alvo. Busca uma escala de orçamento por arquivo **somente pelo tamanho
serializado**, sem selecionar a melhor fidelidade entre candidatos da busca.
Retém o tamanho observado mais próximo que não ultrapasse o alvo; não presume
monotonicidade perfeita de um algoritmo guloso. Não há treino ou inferência nova.

O empacotamento inclui os mesmos mapas, tabelas, flags, sequência e comprimentos
reais. O cálculo de tamanho foi conferido contra o bundle original. Todos os
bytes de frames e metadados dos novos streams passaram em roundtrip.

Artefatos: `results/experiments/wire_matched_hybrid_20261030/`.

| Coorte / ponto do controle | Bytes abaixo do alvo | Ganho relativo de RMSE do híbrido | Δ correlação | Δ p90 canal |
|---|---:|---:|---:|---:|
| Fold 0 / 1,1× | 433 | 0,156% | +0,001434 | −26.661 |
| Fold 0 / 1,25× | 101 | 0,289% | +0,002556 | **+814.660** |
| Fold 0 / 1,5× | 100 | 0,356% | +0,002919 | **+1.688.163** |
| Fold 0 / 2× | 29 | 0,341% | +0,002623 | **+2.380.626** |
| Fold 1 / 1,1× | 197 | 0,023% | +0,000877 | −0,000128 |
| Fold 1 / 1,25× | 299 | 0,034% | +0,001584 | −0,000422 |
| Fold 1 / 1,5× | 67 | 0,078% | +0,001865 | −0,000628 |
| Fold 1 / 2× | 363 | 0,113% | +0,001754 | −0,000989 |

Δ = híbrido − denso. Gaps de tamanho são ≤0,0027% do stream, não igualdade
matemática perfeita. O ganho de RMSE persiste ao controlar bytes, mas é **pequeno**
e cai no segundo fold. A regressão de cauda no primeiro fold não desaparece.
Seus valores enormes são dominados por casos de faixa dinâmica/canais fracos;
isso requer contextualização física, não exclusão retrospectiva.

## Decisão

- Manter o refiner como componente de pesquisa, não arquitetura final: não há
  dominância generalista forte que justifique três modelos e maior complexidade.
- Encerrar essa sequência de treinos de reconstrução. Não aumentar parâmetros,
  retunar ROI ou repetir seeds até obter um resultado favorável.
- Priorizar agora um **modelo neural de probabilidades dos índices congelados**,
  separado do reconstrutor: potencial de reduzir bytes sem alterar forma de onda,
  amplitude/fase ou canais. Comparar contra marginal/range e Huffman, cobrando
  o novo prior para não atribuir ao contexto ganhos de outro empacotamento.
- Tolerâncias físicas, aquisições externas e validação operacional permanecem
  requisitos de promoção. A confirmação cruzada não cria dados independentes.

O próximo protocolo está no
[relatório 74](74-LEARNED_LOSSLESS_INDEX_PRIOR_PROTOCOL.md).
