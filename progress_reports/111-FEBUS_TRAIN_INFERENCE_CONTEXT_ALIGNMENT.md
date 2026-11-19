# FEBUS: contraste pareado de contexto de treino e inferência

**18/11/2026.** Protocolo prospectivo após os
[diagnósticos F1](108-FEASIBILITY_DIAGNOSTICS_EXECUTION.md) e o
[contraste de cobertura](109-FEBUS_FULL_COVERAGE_CONTEXT_CONTRAST.md).

## 1. Hipótese e escolha de uma candidata

**Treinar no mesmo contexto espacial da inferência permite que uma fonte
offshore aprenda um ponto full-tile melhor sem transmitir mais tokens?**
O FiLM de crop melhorou a cobertura com `part8`, mas pagou 140,3% mais bytes.
F4-B muda o contexto **de treino**, não o wire de inferência. O flat de 97
partiu de uma função madura; aqui, a origem é offshore, com a mesma exposição
física entre braços. F4-A não é selecionado: D2c não mediu economia FEBUS e o
[relatório 110](110-FEBUS_TEMPORAL_CONTRACT_SENSITIVITY.md) mantém seu bloqueio.
Não se declara superioridade sobre um mecanismo ainda não testado no alvo.

## 2. Braços e estimando

- Fonte `models/febus_finetune_base/best.pt`, SHA-256
  `7d2da0d1cbd1d3dae579d7fd900850afc9ef15f645ea844168c0f78f18b0a27c`.
  VQ flat K2048, dimensão 128, `spatial4-temporal4`, vetores hard exatos,
  checkpointing de ativações. Sem FiLM, alteração de backbone ou novo coder.
- **Controle part8-train:** fragmentos contíguos de oito canais, com resto e
  todos os valores. Normalização por fragmento, gradiente somado com pesos
  proporcionais às células físicas e um step por janela original.
- **Candidata full-train:** o mesmo arquivo, janela, canais e tempos em um
  tensor full-tile, normalização nativa, um único step.
- Perda `0,5 MSE + 0,5 MAE` normalizada mais VQ. AdamW `lr=1e−4`,
  `weight_decay=1e−5`, clip global 1. Medir tokens e compute: mesmos updates
  e células não equivalem a mesmos tokens, FLOPs ou normalizações.

Treino apenas em NPZ preparados do pool train de 32 arquivos, menos os dois
do envelope de 110 (exclusão conservadora). Os 30 restantes servem ao ajuste,
não autorizam D2c. Incluem geometrias estreitas e largas; não trocar corpus
para caber em memória. Sem interseção de IDs com os checks; isso não certifica
sessões independentes nem remove exposição herdada.

O censo de metadados encontrou **3.420 janelas elegíveis** (288.040.000
células únicas no inventário elegível, antes da amostragem com reposição) e
nenhum caminho local ausente. O hash da fonte offshore foi conferido.

Schedule congelado antes de updates: seed 2117, rodada por família, sorteio
uniforme no grupo→arquivo→janela em ordem canônica. Mesma ordem de janelas
nos dois braços, sem amostragem por energia ou erro.

## 3. Preflight, execução e avaliação

1. Congelar hashes de fonte, código, seleção e schedule. Testes unitários e
   smoke sintético incluindo largura 512 e casos estreitos. Se não couber,
   não recortar: testar o dispositivo previsto ou registrar bloqueio.
2. Antes de **cada** treino, consultar GPUs remotas e usar apenas uma livre
   com memória suficiente. Sem remota disponível, RTX 4060 local. Checar
   novamente no lançamento; braços sequenciais. Registrar o dispositivo
   efetivo e não comparar tempos como pareados se houver troca.
   O seletor não derruba processos existentes e restringe o processo ao UUID
   escolhido via `CUDA_VISIBLE_DEVICES`. A consulta não reserva o dispositivo.
3. **8.192 updates por braço**, sem extensão ou escolha de checkpoint por
   score. Teto inicial de duas horas por braço. Interrupção técnica guarda
   progresso sem reduzir retrospectivamente a dose para passar gate.
4. Avaliar endpoints full-tile nos 136 checks históricos de treino, com
   geometria, CDF de índices congelada, FAK1 e identidade vinculada ao export.
   Medir a VQ madura pelo mesmo código/wire. Cobrar bytes de frames, side,
   mapas e instalação; registrar também tempo e artefatos de redecodificação.
5. Sem validação, teste ou EDAS. Não ajustar CDF, dados ou gates pelos checks.

## 4. Decisão predefinida

Avaliar separadamente a candidata contra **(a) part8-train** e **(b) VQ madura**,
na mesma população. Para passar em cada comparação: ≥5% de redução de NRMSE*
macro, sem aumento de bytes completos de transporte ou SSE agregado, e sem
regressão >1% em NRMSE* de família, P90 por configuração ou erro espectral por
configuração/banda. Denominador zero exige erro absoluto não aumentado.
Publicar objeto com instalação, fase ausente e falhas por subgrupo. Sem
interpolação, fallback seletivo, reseleção ou extensão automática.

Só (a) é evidência de compatibilidade, não avanço prático. Mais bytes é ponto
de viabilidade, não eficiência. Dois negativos válidos encerram o contraste
sem sweep; incompletude é limitação técnica. Nem dois passes liberam P3 sem
confirmação independente. F3 depende de sessões documentadas.

## 5. Resultado

Ambos os braços completaram 8.192 updates e 592.255.000 apresentações de
células na RTX 4060. Na avaliação full-tile (136 checks), `full-train`
atingiu **0,62277 NRMSE* / 0,71889 bpc**, contra **1,04108 / 0,68861**
do controle e **0,49034 / 0,69969** da VQ madura. Foi **40,18% melhor em
erro e 4,40% mais caro** que o controle, mas **27,01% pior em erro e 2,74%
mais caro** que a VQ madura. Os 408 bundles passaram no replay bit a bit.
**F4-B encerrado sem promoção.**
[Decisão e evidência](../results/experiments/feasibility_context_alignment_20261118/decision.json).
