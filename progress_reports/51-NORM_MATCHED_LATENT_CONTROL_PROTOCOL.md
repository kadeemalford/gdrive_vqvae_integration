# Controle latente contínuo com energia quantizada preservada

> Estudo concluído. Resultados e decisão no
> [relatório 52](52-NORM_MATCHED_LATENT_RESULTS.md).

## Hipótese e regra única

O bypass do relatório 50 muda simultaneamente escala e estrutura/direção do
latente. Testar se a diferença de energia total explica parte da regressão
multimode, antes de atribuir tudo à direção do latente ou ao codebook.

Para cada janela, dados z do encoder e zq do quantizer, alimentar o decoder com:

`z_control = z * ||zq||₂ / ||z||₂`.

Norma calculada sobre **todos os componentes/posições latentes**, incluindo
padding. Preserva direção de z e energia total de zq, salvo arredondamento float32.
Ambas as normas zero: transmitir zero; z zero e zq não zero: recusar o controle.
Nenhuma busca de alpha, otimização, ajuste em validação ou parâmetro aprendido.
Não depende de família/configuração nem da reconstrução/alvo para escolher escala.

Isso é um **controle diagnóstico de entrada do decoder**, não expansão da
calibração de ganho de saída do relatório 48. A escala é determinística, calculada
com os dois tensores já disponíveis no encoder, não escolhida para melhorar o score.

## Fontes, comparação e custo

Mesmo midpoint 2x e epoch 24 4x dos controles anteriores, sem mudar pesos.
Duas novas avaliações completas nas mesmas **5.920 janelas de validação**;
comparar com os braços VQ/contínuo já completos, sem repeti-los.
Sources e coordenadas comparados por hash com o protocolo anterior.

Transmitir **todos os componentes escalados float32**: 4.096 bits/posição com
dimensão 128. A escala já está incorporada nesses componentes; o decoder não
precisa receber um escalar extra. Taxa nominal igual à do controle contínuo:
expansão de dados, não codec rate-matched. Não cobrar índices de referência como
payload; eles ainda são calculados para a norma de zq e diagnósticos, logo o
tempo não representa uma implementação de codec sem quantização.

Sem rede treinada nova, especialistas, loss ou política de normalização nova.
Não somar 32 bits por janela como se um ganho precisasse ser transmitido à parte.

## Critérios de interpretação

Comparar RMSE/correlação/p90 e amplitude/fase por configuração, em particular
multimode total e subconjuntos com/sem escapes. Conservar o VQ como referência.

- Se recuperar parte da perda, a energia explica parte do desvio observado,
  não toda a perda de quantização nem uma melhoria atingível no bitrate VQ.
- Se não recuperar, energia total sozinha não explica o bypass: covariance,
  estrutura espacial/temporal, direções e distribuição ainda diferem.
- Em nenhum caso chamar o resultado de limite irreducível ou recomendação
  automática de aumentar codebook/modelo.
- Encerrar esse controle fixo; não expandir uma busca de escalas/misturas na
  mesma validação sem hipótese adicional prévia.

## Implementação e verificação

Helper isolado: `finetuning/norm_matched_probe.py`.
Runner existente ganhou modo opt-in **`--norm-matched-only`**, sem mudar trainer.
Snapshots e hashes do estudo anterior permanecem como evidência histórica.
O modo novo referencia resultados completos anteriores, não os sobrescreve.

**174 testes passaram**, incluindo norma, direção, índices, loss de referência
e comportamento de norma zero. Smoke GPU em ambas as fontes, com janelas reais
512x1.000 multimode com escapes e 111x1.000 single-mode Configuration 2:
normas iguais a <1e-6 absoluto nos exemplos, decoder input verificado,
índices inalterados e restauração física bit-exata dos escapes.
Evidência: `results/experiments/norm_matched_quantization_smoke_20261019/`.

Saída: `results/experiments/norm_matched_quantization_20261019/`.
Serviço: `febus-norm-matched-quantization-20261019`.
Log: `results/experiments/norm_matched_quantization_20261019.pipeline.log`.
Estimativa **15–30 min**, duas avaliações, **zero updates**; resultados pendentes.

```bash
systemctl --user status febus-norm-matched-quantization-20261019
# Apenas sem cópia ativa:
.venv/bin/python finetuning/run_frozen_quantization_probe.py \
  --output-dir results/experiments/norm_matched_quantization_20261019 \
  --norm-matched-only --resume

# Após COMPLETE, gerar pareamento contra VQ anterior em saída nova:
.venv/bin/python finetuning/summarize_quantization_probe.py \
  --experiment-dir results/experiments/norm_matched_quantization_20261019 \
  --reference-dir results/experiments/frozen_quantization_probe_20261019 \
  --output-dir results/experiments/norm_matched_quantization_paired_NEW
```

Locks, hashes e tentativas preservadas; retomada pula candidatos completos.
Inibição de idle temporária; reinício/suspensão manual pode interromper execução
e o aviso no chat. Dados de desenvolvimento reutilizados, sem acesso a teste,
aquisição externa ou promoção automática.
