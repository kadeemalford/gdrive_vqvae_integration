# FEBUS: amortização de tabelas congeladas em mais janelas

**Protocolo prospectivo.** Continuação do [relatório 114](114-FEBUS_SHORT_WINDOW_ENTROPY_RESULTS.md).
A pergunta é se os ganhos acumulados de transporte pagam o custo **medido**
da instalação, sem reajustar tabelas. Este é um teste de reutilização nos
mesmos arquivos, não uma confirmação entre arquivos ou sessões.

- Reusar as instalações e as duas precisões do 114 (`ε/σ_cal = 0,03 e 0,1`).
  Manter marginal e espaço-temporal; temporal pode ser publicado como contraste
  auxiliar. Não escolher tabela, suporte ou modo novo pelos resultados.
- Nos oito arquivos de check originais, selecionar até **20 novas janelas por
  arquivo**, cada uma com 1.000 amostras e mesmos dataset, dt e índices físicos
  da janela original. Excluir as janelas originais e interseções no mesmo
  bloco; ordenar candidatos por SHA-256 de
  `feasibility-amortization-v1|sample_path`, com desempate pelo caminho.
  Não concatenar janelas; reset por frame. Não alegar independência ou
  unicidade temporal entre blocos se a semântica continuar incerta.
- Congelar entradas e política antes de ler valores. Pagar bindings, frames,
  geometria, resets, escapes e toda instalação. A instalação compartilhada
  entra uma única vez por configuração, com hashes; não extrapolar para
  outros arquivos como se a taxa fosse constante. Se algum grupo não tiver
  janelas suficientes, publicar sua cobertura real sem substituir arquivos.
- Publicar curvas cumulativas a 1, 5, 10 e 20 janelas disponíveis por arquivo,
  objetos/bytes reais e limiar observado de amortização, por grupo e no
  agregado. Verificar índices/reconstrução bit a bit e bound.

**Decisão predefinida:** economia líquida nos dois pontos sustenta testar
transferência de uma instalação fixa fora desses arquivos, não iniciar F4-A.
Economia em apenas um ponto ou alguns grupos é localizada; ausência de
economia fecha esse uso da instalação sem recalibrar no check. F2 contínuo,
multicore e generalização entre sessões permanecem pendências separadas.
Apenas NPZ de treino e CPU; sem HDF5, GPU ou reservas.
