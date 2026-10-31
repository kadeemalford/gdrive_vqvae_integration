# Confirmação local de refinamento FSQ em outra rotação de configurações

**Concluída em 30/10/2026 às 19:19 UTC**, sem promoção final. Síntese e controle
adicional por tamanho real do stream no
[relatório 73](73-RESIDUAL_CONFIRMATION_AND_WIRE_MATCHED_RESULTS.md).

## Pergunta e limites

Investigar se o ganho do refiner FSQ contra uma base compacta e sua complementaridade
com modo denso se mantêm em **test fold 1 / validação fold 2 / treino folds 0 e 3**,
com seed **2036**, sem retunar o objetivo da alocação. Não há codec final promovido.

O primeiro fold mostrou ganhos nas seis configurações contra a base, mas não
dominância sobre modo denso a orçamento comum: o híbrido custa mais modelos e
perde em caudas relativas. Repetir esse confronto, não somente o melhor RMSE,
é requisito para avaliar utilidade do refinamento.

Esta é confirmação de desenvolvimento; as aquisições já fazem parte do histórico
de pesquisa, sem novos dados externos. Mudar configuração/seed não demonstra
independência de toda a seleção anterior. Test fold 1 não é inferido.

## Base compatível antes do refiner

O checkpoint compacto do primeiro fold treinou em configurações agora reservadas
para validação. **Não reutilizá-lo nesta rotação.** Preparar uma base compacta
adaptada de `models/febus_finetune_base/best.pt` (offshore original, sem FEBUS),
com corpus filtrado `spatial_band_v1_fold1_20261026`:

- **14.620 janelas de treino / 6.940 de validação**, mesmos 207 HDF5, filtro
  `persistent-amplitude-v1` e normalização `robust-escape-v1` já auditados.
- CNN nativa de 27.958.081 parâmetros, geometria `spatial-4x`, K=2.048, dimensão
  128, 11 bits por índice. Carregar pesos offshore, não source FEBUS de outro fold.
- AdamW LR 1e-4/weight decay 1e-5, objetivo original 0,5 MSE + 0,5 L1 + VQ;
  sampling window, FP32, activation checkpointing, replay exato.
- **16×4.096 updates**, extensão única de oito épocas se melhora entre os últimos
  blocos de quatro épocas for ≥0,2%; teto 98.304 updates. Não estender por não
  atingir o score do fold anterior. Seleção macro na triagem fixa, não época final.
- Fonte densa congelada compatível: `spatial_abc_fold1_20261026/high_A/best_macro.pt`;
  checkpoint registra exatamente esse corpus. Seus pesos não são novamente treinados.

Exposição dos pesos densos históricos não é pareada ao refiner; comparar sistemas
e controles de bits dentro da coorte, sem reivindicar isolamento causal de todo
o compute. Não comparar diretamente scores entre folds com famílias/configurações
diferentes como se fosse ganho de uma nova arquitetura.

## Fila automática de confirmação

1. Treinar/selecionar a base compacta compatível, com checkpoint/RNG retomáveis.
2. Gerar streams compactos/densos da validação completa, aprendendo tabelas somente
   em treino; reconstruir pelo stream e preservar códigos/side. Controles ROI
   anteriores são mantidos com os mesmos tetos, sem novos pesos.
3. Treinar refiner FSQ **8×8×32**, mesma arquitetura de 7.094.628 parâmetros,
   resíduo em coordenadas normalizadas originais, sem side de escala adicional,
   mesma calibração de 16 janelas por família, seed 2036.
4. Orçamento residual idêntico ao piloto: **16×1.024 updates**, até oito épocas
   extras se ganho recente ≥0,2% e ganho contra a base ≥1%, teto 24.576 updates.
   Mesma seleção por RMSE da triagem guardada; registrar também saída sem guarda.
5. Avaliar todos os selecionados da fase 16 e, se elegível, 24 em 6.940 janelas,
   verificando streams residuais, composição e bundles reais.
6. Confrontar denso, residual e híbrido sob tetos **1,1×/1,25×/1,5×/2×** por arquivo,
   com a mesma redução de RMSE² por célula/byte e guarda RMSE/correlação/p90.
   Alocação aninhada e upgrades sem retirar melhora anterior. Mostrar bytes
   efetivamente usados, espectro, fase, caudas, canais não fracos e registry.

São até **122.880 updates no total** entre nova base e refiner, sem outro braço
scratch ou nova rede grande. A confirmação não autoriza afrouxar guardas ou escolher
um sistema só pelo melhor agregado. Se o ganho não repetir, preservar o negativo
e encerrar essa linha sem buscar novas seeds até obter uma favorável.

## Execução e rastreabilidade

Runner: `finetuning/run_residual_fold_confirmation.py`.
Diretório: `results/experiments/residual_confirmation_fold1_20261030/`.
Unidade: `febus-residual-confirmation-fold1-20261030.service`.

Usar **somente a RTX 4060 local**, nenhum treinamento remoto. Os runners de streams,
residual e comparação foram generalizados por flags de corpus/fold/seed/checkpoints.
Snapshots e hashes dos estudos concluídos preservam suas implementações anteriores;
os resultados não são reescritos.

Auditoria dos metadados é conferida antes de executar, fonte densa deve apontar
para o corpus correto, e o refiner confere hash da base no protocolo dos frames.
Fontes/metadados fixados por hashes e snapshot. Espaço mínimo livre de 12 GiB no
orquestrador e limites próprios nos filhos; sem limpeza automática.

Logs e status separados por fase; `compact_base`, `frozen_frames`, `residual_fsq`
e `matched_modes` conservam resultados completos. Não inferir término por um monitor
de chat cancelado: consultar processos, unidade e `STATUS.json`/`COMPLETE` reais.
O serviço tem limite operacional de 48 h/22 GiB RAM e não atualiza o Git durante
a execução. Nenhum commit/push é necessário para os experimentos, que fixam hashes.

### Início confirmado

A unidade local foi iniciada em **30/10/2026 às 15:32 UTC** e confirmada ativa
no treinamento da base compacta, alvo inicial de 16 épocas. O refiner e os
controles de streams/modos seguem automaticamente depois. **240 testes** passaram
após generalizar os runners; snapshots dos três estudos anteriores foram
conferidos contra seus hashes históricos antes dessa confirmação.
