# Quantização e decoder congelados: controle com latentes contínuos

> Estudo concluído; resultados e auditoria pareada no
> [relatório 50](50-FROZEN_QUANTIZATION_RESULTS.md).

## Pergunta e interpretação limitada

Depois do resultado quase neutro/negativo da calibração global, investigar se
substituir latentes quantizados pelo tensor do encoder altera os limites de
reconstrução, mantendo **os mesmos pesos do encoder e decoder**.

Esse controle não é um autoencoder retreinado nem uma taxa equivalente:

- Uma melhora aponta sensibilidade à quantização/input do decoder, mas não
  garante que seja possível recuperá-la com o orçamento VQ atual.
- Uma piora não prova que o codebook seja adequado: o decoder foi treinado com
  latentes quantizados, e o tensor contínuo pode estar fora dessa distribuição.
- Não chamar o erro contínuo de piso irreducível da arquitetura ou limite
  superior universal de fidelidade. Não decidir aumento de codebook apenas por ele.

## Candidatos e dados

Mesmo midpoint 2x e epoch 24 4x dos relatórios 47–48, **sem ganho calibrado**.
Quatro avaliações nas **5.920 janelas completas de validação do fold 0**:
VQ e contínuo para cada source. Rede congelada, sem ajuste de pesos, seletores,
novos dados ou acesso a teste. Referência física `robust-escape-v1` comum.

Hook no quantizer mede o erro encoder→quantizado, RMS de ambos, loss VQ de
referência, uso de códigos e posições/dimensão latentes. No braço contínuo,
encaminha o tensor original do encoder ao decoder. Os índices VQ ainda calculados
são mantidos **somente para diagnóstico/verificação de geometria**, não payload.
Loss VQ no forward contínuo é zero; a loss de referência fica em campo separado.
Não comparar loss total como critério de promoção entre esses braços.

## Taxas e custo computacional

Payload VQ: posições × ceil(log2 K), K=2.048, **11 bits/posição**.
Payload contínuo: posições × dimensão × **32 bits**, mais side de normalização/
escapes em ambos. Dtype float32 é verificado. Nestes sources a dimensão é **128**:
o controle contínuo tem **4.096 bits/posição**, ~372,36x o payload dos índices.

Em 512x1.000, sem side:

| Taxa / modo | Posições | Bits latentes | Interpretação |
|---|---:|---:|---|
| 2x VQ | 16.000 | 176.000 | Referência comprimida |
| 2x contínuo | 16.000 | 65.536.000 | ~4x os bits float32 de entrada |
| 4x VQ | 32.000 | 352.000 | Referência comprimida |
| 4x contínuo | 32.000 | 131.072.000 | ~8x os bits float32 de entrada |

Ou seja, o contínuo **expande**, não comprime, esses exemplos. Contar shapes e
padding reais por janela e reportar razão pooled de payload+side. O estimador
VQ genérico é removido das tabelas do controle para não apresentar taxa falsa.

Como a quantização de referência ainda é calculada no hook, o tempo medido não
estima compute de um decoder/codec contínuo implantado. Container/modelo/entropia
seguem fora da taxa nominal. Nenhum argumento de superioridade rate-matched.

## Análise prevista

Comparar RMSE/correlação/p90 por configuração, amplitude/fase/bandas e pior
configuração/família. Parear janelas para distinguir mudança global de exceções.
Associar descritivamente erro de quantização aos deltas de reconstrução, sem
confundir correlação com causalidade. Respeitar escapes bit-exatos em todos.

Se VQ e contínuo forem semelhantes, não justifica aumentar codebook sem outro
controle. Se houver melhora heterogênea, documentar os grupos antes de qualquer
hipótese de mudança de quantizador/treino. Se houver regressão generalizada,
considerar o desvio de distribuição antes de atribuí-la à capacidade do decoder.

## Verificação e execução

**172 testes passaram**. Smoke GPU em ambos os sources e duas shapes reais
(512x1.000 com escapes e 111x1.000 single-mode Configuration 2):

- Hook VQ mantém a reconstrução original exatamente.
- Braço contínuo coincide exatamente com decoder aplicado ao z capturado.
- Índices de referência inalterados, saídas finitas e escapes bit-exatos.
- Contagem usa dimensão real 128 e custo float32, não somente índices.

Evidência: `results/experiments/frozen_quantization_probe_smoke_20261019/`.
Runner: `finetuning/run_frozen_quantization_probe.py`.
Helper: `finetuning/quantization_probe.py`; sem mudanças no trainer padrão.
Saída: `results/experiments/frozen_quantization_probe_20261019/`.
Serviço: `febus-frozen-quantization-probe-20261019`.
Log: `results/experiments/frozen_quantization_probe_20261019.pipeline.log`.
Estimativa inicial **25–50 min**, quatro avaliações; zero updates. Resultados pendentes.

```bash
systemctl --user status febus-frozen-quantization-probe-20261019
# Apenas sem cópia ativa:
.venv/bin/python finetuning/run_frozen_quantization_probe.py \
  --output-dir results/experiments/frozen_quantization_probe_20261019 --resume
```

Protocolo e fontes/código/manifest/smoke fixados por hashes, lock e tentativas
parciais preservadas. Retomada pula candidatos completos. Inibição de idle é
temporária, sem garantir sobrevivência a suspensão manual/reboot. Aviso no chat
não é garantido após reinício do OpenCode. Nenhuma promoção automática.
