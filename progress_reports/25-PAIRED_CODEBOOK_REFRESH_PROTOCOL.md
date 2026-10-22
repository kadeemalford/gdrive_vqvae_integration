# Reativação de códigos: ablação longa controlada

> Comparação concluída; resultado e decisão de não promover no
> [relatório 26](26-CODEBOOK_REFRESH_RESULTS.md).

## 1. Objetivo e controle

Testar se reativar um número limitado de códigos sem uso ajuda o candidato temporal
a melhorar reconstrução, especialmente grupos/canais difíceis. **Não maximizar
perplexidade como substituto de fidelidade**.

Dois braços partem dos mesmos pesos **scratch_temporal best_macro epoch 31**:

- **Control:** quantizador atual sem reativação.
- **Refresh:** reativação train-only versionada `dead-v1`.

Em ambos, **AdamW novo**, LR **3e-5**, weight decay 1e-5, grad clip 1. A LR mais
conservadora é compartilhada; não é resume de optimizer 1e-4. O contraste entre
braços isola refresh; ganho do control não separa LR, reinício do optimizer e
exposição adicional. Fonte e optimizer antigos permanecem intactos.

## 2. Política dead-v1

- Código elegível após **três épocas consecutivas sem uso nas janelas de treino
  apresentadas**. Isso não prova que seja inutilizado em todo o domínio.
- Reativar **até 32 por época**, seleção determinística entre elegíveis.
- Embeddings substituídos por latentes do encoder em treino: até 128 vetores
  amostrados a cada 64 updates, máximo 64 blocos/8.192 vetores por época.
- Sem atualização/calibração baseada na validação, fase ou erro contra holdout.
- Zerar momentos Adam somente nas linhas substituídas; não reiniciar demais pesos
  nem o contador global de steps do optimizer.
- Reativar após o treino da época e **antes da avaliação/save**: métricas pontuam
  o codebook exato do checkpoint, não uma versão anterior à intervenção.
- Log de usage/perplexidade descreve treino antes da reativação; mudanças são
  avaliadas na validação subsequente e no treino seguinte.
- Ages são salvas/retomadas. RNG local por época não altera a ordem pareada dos dados.

Código em `finetuning/codebook_refresh.py`; integração opcional no trainer.
Default `none` mantém o comportamento anterior. Tamanho K=2.048, embeddings,
parâmetros, geometria 4x e bits de índices não mudam. Receiver precisa do modelo
correspondente; taxa nominal ainda não inclui codec/container completo.

## 3. Orçamento e avaliação

| Item | Protocolo |
|---|---|
| Corpus | mesmo fold 1 FEBUS, global escape-v1, sem novos dados |
| Arquitetura | temporal-dilated-v1, 27.958.081 parâmetros |
| Objetivo | temporal 0,5 MSE + 0,5 L1 + VQ, inalterado |
| Updates adicionais | **24 x 4.096 = 98.304 por braço**, 196.608 totais |
| Parada | patience 25, sem parar antes do orçamento de 24 épocas |
| Seleção | mesmo piloto de 102 janelas, inicialização epoch 0 como candidata |
| Avaliação final | 5.820 janelas dos selecionados, source temporal e transferido congelados |
| Custos | índices reais/side pooled, compute/exposição declarados |
| Teste | fora da seleção e diagnóstico |

Somente melhoria robusta de erro/canais/piores grupos justifica avanço. Um ganho
de uso de códigos com regressão física é resultado negativo, não promoção.

## 4. Testes e smoke

**135 testes passaram**: limite/idade de reativação, proteção de códigos ativos,
limpeza seletiva de momentos, estado determinístico retomável, reservoir limitado
e rejeição de mudança de política no resume.

Smoke GPU em **512x1.000**, com três updates **descartados**, sem salvar pesos:
contagens de reativação **0 / 0 / 32**, gradiente/loss finitos, codebook intacto
durante forward de avaliação e source original inalterado. Pico alocado ~**4,19 GB**.
Evidência: `results/experiments/codebook_refresh_smoke_20261011/`.

## 5. Execução e recuperação

Runner: `finetuning/run_codebook_refresh_experiment.py`.
Saída: `results/experiments/codebook_refresh_long_20261011/`.
Log de sequência: `results/experiments/codebook_refresh_long_20261011.pipeline.log`.
Serviço de usuário: **`febus-codebook-refresh-20261011`**, sequencial na GPU.
**Resultados pendentes nesta atualização**.

Estimativa pelos tempos anteriores: **5h30–7h**, incluindo quatro avaliações
completas ao final. Atualizar com épocas medidas; I/O/carga podem variar.

```bash
systemctl --user status febus-codebook-refresh-20261011

# Somente sem cópia ativa:
.venv/bin/python finetuning/run_codebook_refresh_experiment.py \
  --output-dir results/experiments/codebook_refresh_long_20261011 --resume
```

Fontes, manifest, protocolo e código são fixados por hashes, saves atômicos e lock.
Resume de um braço restaura seu optimizer/RNG e ages; código/política alterados
exigem protocolo novo, não retomada silenciosa. Serviço independe da sessão
interativa, não garante sobrevivência a logout/reboot; epochs parciais são repetidos.

Nenhum novo dataset foi baixado e nenhum modelo final foi promovido. Após a
comparação, decidir entre aprofundar quantização, arquitetura/capacidade ou
diversidade, conforme resultados por configuração/canal, não só média de uso.
