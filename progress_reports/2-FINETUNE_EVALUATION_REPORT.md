# Avaliação do Fine-Tuning FEBUS: Fold 0

> Atualização 2026-10-05: os folds 1–3 já foram avaliados. O teste de balanceamento
> hierárquico, os resultados preliminares e o plano generalista (incluindo ensemble/MoS
> e treino do zero) estão em [FEBUS_GENERALIST_ROADMAP.md](FEBUS_GENERALIST_ROADMAP.md).
> A comparação completa de balanceamento foi iniciada; não há ainda conclusão final.
> O restante deste documento preserva o relato histórico do fold 0.

## Escopo

Este relatório descreve somente a validação cruzada conservadora do fold 0. O checkpoint-base foi copiado para `models/febus_finetune_base/best.pt`; os checkpoints ajustados e a avaliação ficam em `results/fine_tune_results/fold_0/`. Nenhum HDF5 derivado foi usado: `*_output.h5` e `*_fft_output.h5` foram excluídos porque duplicam ou derivam dos arquivos `Copy of ...h5`.

## Unidade de split

O grupo atômico é `família/configuração`. Todas as frequências de uma configuração ficam na mesma partição. No multicore, os dois núcleos de uma pasta como `C1 and C2` também ficam juntos. Janelas temporais de um mesmo HDF5 nunca são divididas entre treino, validação e teste.

| Papel | Grupos de origem no Drive |
| --- | --- |
| Teste, fold 0 | Single-mode: Configurations 1, 5 e 9; Multimode: Configuration 5; Multicore: C1 and C2 e C2 and C4; Doped: 5m i29. |
| Validação, fold 1 | Single-mode: Configurations 2, 6 e 10; Multimode: Configuration 6; Multicore: C1 and C3; Doped: 5m LSF. |
| Treino, folds 2 e 3 | Single-mode: Configurations 3, 4, 7, 8, 11 e 12; Multimode: Configuration 7; Multicore: C1 and C4 e C2 and C3; Doped: 5m SMF. |

O conjunto preparado contém 9.720 janelas de treino, 5.920 de validação e 8.240 de teste. Cada janela usa 1.000 amostras temporais e a geometria espacial nativa do arquivo FEBUS; canais totalmente não finitos são removidos.

## Treinamento

O VQ-VAE foi inicializado a partir do checkpoint OOI, otimizado com AdamW, taxa de aprendizado `1e-5` e perda `0,5*MSE + 0,5*L1 + VQ`. O treino chegou à época 10; a menor `val_loss` registrada foi `0,75490` na época 8, mas o checkpoint salvo pela versão anterior do critério de `min_delta` foi a época 7 (`val_loss=0,75531`, `val_corr=0,44925`). O critério foi corrigido para execuções futuras: o melhor checkpoint passa a ser sempre a menor `val_loss`, enquanto `min_delta` controla somente a paciência.

## Avaliação no teste isolado

As métricas abaixo vêm de `results/fine_tune_results/fold_0/evaluation/summary.json` e comparam o checkpoint-base com `fold_0/best.pt` no mesmo conjunto de 8.240 janelas de teste.

| Grupo | Janelas | Correlação base | Correlação fine-tuned | RMSE relativo base | RMSE relativo fine-tuned | SNR base (dB) | SNR fine-tuned (dB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Total | 8.240 | 0,1664 | 0,3238 | 0,9896 | 0,9460 | 0,0929 | 0,4979 |
| Single-mode | 1.180 | 0,1509 | 0,4114 | 0,9995 | 0,8954 | 0,0067 | 1,0162 |
| Multicore | 5.240 | 0,1873 | 0,3436 | 0,9819 | 0,9373 | 0,1604 | 0,5650 |
| Doped | 1.300 | 0,1152 | 0,2572 | 1,0095 | 1,0064 | -0,0818 | -0,0529 |
| Multimode | 520 | 0,1188 | 0,0924 | 0,9948 | 0,9974 | 0,0449 | 0,0224 |

O ganho agregado é real no fold 0, sobretudo em single-mode e multicore. Ele não é uniforme: multimode piorou e a melhoria em doped é pequena em RMSE/SNR, apesar de a correlação aumentar. Por isso, a conclusão correta é que o fine-tuning melhora a média do teste isolado, mas ainda requer os demais folds e análise por família/configuração.

## Exemplo de melhoria individual

No arquivo `SMF Configurations/Configuration 5/Copy of 3000Hz.h5`, a correlação média das 40 janelas de teste passou de `0,0828` para `0,7901`. O HDF5 foi copiado para `data/raw/smf_configuration_5__3000Hz.h5` e foi usada a configuração `config/spatial_range_configuration_5_3000hz.json` para produzir inferências base e fine-tuned comparáveis.

## Próximos testes recomendados

1. **Completar os folds 1, 2 e 3.** Repetir o mesmo protocolo conservador, sempre mantendo uma configuração inteira em apenas uma partição. Reportar média e dispersão entre folds, não somente o fold 0.

2. **Avaliar por configuração e frequência.** A média global esconde variação grande: Configuration 5/3000 Hz melhorou muito, enquanto Configuration 1/200 Hz não melhorou. Exportar tabelas por `família`, `configuração`, `frequência` e canal multicore identifica onde a adaptação transfere.

3. **Selecionar checkpoint por menor `val_loss` e testar `val_correlation`.** A implementação atual já foi corrigida para salvar a menor `val_loss`. Comparar o checkpoint selecionado por perda com um checkpoint selecionado por correlação esclarece qual critério acompanha melhor a fidelidade visual e espectral FEBUS.

4. **Ajustar a taxa de aprendizado após estagnação.** Quando a validação estabilizar, retomar de `last.pt` com `1e-5` reduzido para `3e-6` ou `1e-6`, mantendo o estado do AdamW. Comparar o ganho adicional contra o risco de piorar multimode.

5. **Fine-tuning parcial.** Comparar três variantes no mesmo fold: modelo inteiro ajustável; encoder congelado; e encoder+codebook congelados, ajustando somente decoder. Isso mede se a adaptação necessária é principalmente de representação, quantização ou reconstrução.

6. **Balancear famílias e configurações no amostrador.** Multimode tem menos arquivos e piorou no fold 0. Testar amostragem uniforme por grupo `família/configuração`, em vez de deixar grupos com mais janelas dominarem as atualizações.

7. **Adicionar objetivo espectral.** Comparar a perda atual com uma perda que inclua erro de magnitude FFT em bandas relevantes, preservando MSE/L1 como base. O VQ-VAE original já apresentava perda relativa alta em bandas superiores no domínio OOI; isso pode ser agravado no FEBUS.

8. **Revisar geometria de entrada.** Testar janelas temporais equivalentes em duração física e estratégias para acomodar a diferença de `dx` e número de canais entre famílias. Esse teste exige manter splits por configuração para não confundir ganho geométrico com vazamento.

9. **Holdout por família como teste secundário.** Após escolher hiperparâmetros apenas com folds por configuração, treinar sem uma família inteira e testá-la isoladamente. Isso mede transferência para fibra não vista, mas não deve substituir a métrica principal de generalização entre configurações.
