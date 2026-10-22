# Piloto prospectivo full versus decoder-only com admissão por configuração

> Estudo concluído. Resultados e decisão no
> [relatório 54](54-GUARDED_DECODER_PILOT_RESULTS.md).

## Hipótese e source

Verificar se limitar a adaptação ao decoder preserva melhor as configurações
que regrediram em continuação full, sem aumentar taxa/arquitetura. A alternativa
pode também piorar generalização: não presumir vantagem de congelar representação.

Source idêntico: `high_rate_fold0_robust_20261017/rate_4x/best_macro.pt`,
epoch 24, RMSE completo 0,479352. Pesos originais imutáveis.
Dois braços: **full** e **decoder**, com **AdamW novo em ambos**.
Não comparar diretamente com a continuação histórica 3e-5 como controle isolado:
LR, seed, optimizer e seleção daquele estudo eram diferentes.

## Orçamento fixo e variáveis controladas

- **8 épocas × 4.096 = 32.768 updates por braço**, 65.536 no total.
- **LR 3e-6**, weight decay 1e-5, seed 2029 e sampling window em ambos.
- Mesmo corpus completo de treino do fold 0, mesmo source, geometria
  spatial4-temporal4, K=2.048 e objetivo temporal/VQ histórico.
- Sem bypass VQ, ganho calibrado, averaging, loss auxiliar/espectral, refresh,
  balanceamento ou roteamento por metadados.
- Mesma política robust-escape-v1, máscaras de loss e restauração física.
- Patience 9 mantém as oito épocas fixas. Métrica macro histórica continua
  sendo calculada e retida como controle de seleção.
- Full otimiza **27.958.081 parâmetros**, decoder **9.758.273**; igual número de
  updates/seed não significa igual compute ou número de parâmetros otimizados.
- No decoder-only, encoder/quantizer ficam em eval e sem gradiente. Após cada
  época seus tensores são conferidos bit a bit contra o source.

São **épocas 0–8 de um novo piloto com optimizer novo**, não epochs 25–32 com
AdamW/RNG históricos. Dentro de cada braço, resume preserva o optimizer/RNG.
Não alterar o bitrate nominal de índices+side ou o número de parâmetros da rede.

## Seleção prospectiva, sem ocultar perdas locais

Validar **todas as 5.920 janelas a cada época**, incluindo inicialização. Não usar
a amostra de duas janelas/arquivo como gate. Conferir que a fase 0 reproduz os
scores físicos do source; tolerância numérica de paridade 5e-7, sem significado
de tolerância física da aplicação.

Reter snapshots **weights-only de todas as oito épocas**, além de last com
optimizer/RNG e os selecionados macro/loss usuais. O runner executa uma época por
invocação e retoma last; isso não reinicia optimizer/RNG a cada época.
Permite reavaliar pesos intermediários, evitando a lacuna do relatório 43.

Admissão experimental com regra fixada **antes** do treino:

1. Referência fixa = fase 0 do próprio source, não melhor checkpoint móvel.
2. Exigir não regressão de **RMSE, correlação e p90 relativo por canal em cada
   configuração**, comparados à referência completa.
3. Epsilon=1e-9 é somente comparação numérica; não inventa tolerância de aplicação.
4. Entre candidatos elegíveis, reter menor RMSE macro com os pesos históricos
   iguais por família/configuração. Em empate, manter a opção já retida.
5. Se nenhum candidato passar, manter baseline. Um candidato admitido pode ter
   trade-offs frente a outro admitido, pois a garantia é contra baseline fixo.
6. Manter `best_macro.pt` como controle, publicar violações de cada época e
   `best_guarded.pt` como escolha experimental, **sem promoção final automática**.

As tabelas por época usam a validação completa para selecionar; a avaliação
final por janela dos selecionados macro/guardados reutiliza essas mesmas janelas.
**Não é holdout, teste externo ou evidência independente de generalização**.
Uma guarda em médias de grupo tampouco garante melhora de cada canal/janela.

## Integridade, memória e execução

Runner isolado: `finetuning/run_guarded_decoder_pilot.py`.
Guarda: `finetuning/generalist_guard.py`. Trainer/defaults não modificados.
Protocolo fixa source/código/manifest/reference/smoke por hash, mais coordenadas
de validação. Lock impede cópias concorrentes; espaço mínimo 6 GiB, sem exclusões.

Snapshots não contêm optimizer/RNG: **retomar treino somente de last.pt**.
Retomada verifica epoch/política e preserva registros de épocas não commitadas
antes de truncar diagnósticos além do último checkpoint. Falha parcial antes
da criação de last inicial requer inspeção, não descarte/reinício silencioso.

**180 testes passaram**, incluindo sinal da correlação, cauda, fallback baseline,
pesos macro, snapshots e recuperação de histórico diagnóstico.
Smoke GPU real de treino 512x1.000: um update descartado por braço, gradientes
finitos, encoder/codebook decoder-only inalterados. Memória alocada pico
**5,37 GB full / 2,83 GB decoder**. Nenhum peso de smoke salvo/reutilizado.
Evidência: `results/experiments/guarded_decoder_smoke_20261019/`.

Saída: `results/experiments/guarded_decoder_pilot_20261019/`.
Serviço: `febus-guarded-decoder-pilot-20261019`.
Log: `results/experiments/guarded_decoder_pilot_20261019.pipeline.log`.
Estimativa inicial **3–6 horas**, incluindo validação completa por época e
avaliação final dos quatro selecionados. Resultados pendentes.

Retomada em 2026-10-19 após reinício do servidor: nenhum processo do estudo ativo.
Full tinha last.pt de época 5, optimizer/RNG e cinco snapshots preservados;
decoder ainda não iniciado. Protocolo/hash e operação CUDA verificados, sem
alteração de código ou repetição das épocas completas. Serviço da retomada:
`febus-guarded-decoder-resume-20261019`.

```bash
systemctl --user status febus-guarded-decoder-pilot-20261019
# Serviço da retomada:
systemctl --user status febus-guarded-decoder-resume-20261019
# Apenas sem processo ativo:
.venv/bin/python finetuning/run_guarded_decoder_pilot.py \
  --output-dir results/experiments/guarded_decoder_pilot_20261019 --resume
```

Inibição de idle temporária; suspensão manual/reboot pode interromper GPU/serviço.
Aviso no chat não é garantido após reinício do OpenCode. Aquisições independentes,
tolerâncias físicas e bitrate aceitável continuam necessários para decisão final.
