# Codificação estática de transições dos índices neurais

## Hipótese e controles

Um prior contextual neural melhorou payload, mas não amortizou seus pesos na
coorte avaliada. Testar se repetições exatas dos índices podem ser codificadas
com uma descrição estatística menor, sem outro treinamento de rede e sem qualquer
mudança no sinal reconstruído.

Quatro pontos **predefinidos**: compacto/denso dos folds 0 e 1, todos já avaliados
e congelados. Ajustar parâmetros somente nos respectivos treinos. Não usar teste,
trocar pesos entre folds ou retunar direções/limiares a partir da validação.

- Travessias temporal (row-major canal latente × tempo latente) e espacial
  (matriz transposta), duas opções fixas.
- No primeiro símbolo de cada linha, literal obrigatório; nos demais transmitir
  flag `igual ao anterior`. Depois transmitir somente os literais, sem apagar
  os símbolos repetidos da reconstrução.
- Decoder lê todas as flags, sabe quantos literais faltam e restaura cada índice
  por forward-fill; reset de linha explícito evita cruzar limites indevidos.
- Probabilidade de flag com Beta(1,1), frequências literais com +1 por símbolo;
  contagens globais apenas de treino. Sem rede, updates de waveforms ou ajuste
  por família selecionado após ver resultados.
- Controle: menor stream entre Huffman/contexto/fixo existente e range marginal
  ajustado em todo o treino. Comparar o controle a literal+flag no mesmo container,
  cobrando a diferença de tabelas.
- Selecionar por janela o menor payload real entre controle, repetição temporal
  e espacial. Método é transmitido no header, não inferido gratuitamente.

## Integridade e custos

`compression/transition_symbol_stream.py` usa `constriction==0.5.0`,
`perfect=False` explícito, contagens literais em float64 representáveis exatamente
e palavras little-endian. A descrição de parâmetros é pequena, mas seu tamanho
real ainda precisa ser medido; não omitir contagens, flags ou alinhamentos.

Frames `FTR1` protegem header/payload e índices decodificados por CRC; tabelas e
modelo de origem são identificados por hash. **Todos os índices e side streams**
devem coincidir com os frames originais antes de aceitar resultados. Bundles
incluem mapas, metadados, sequência, comprimentos e tabelas. Pesos de waveform
continuam compartilhados e são contabilizados separadamente uma vez.

Não há denoising, recorte adicional de canais ou alteração de cobertura. A camada
de repetição é lossless sobre **códigos discretos de um reconstrutor neural**,
não método clássico alternativo para os dados DAS brutos.

O cache de índices compactos do prior negativo pode ser reutilizado somente se
hash do checkpoint/corpus/arquivo original coincidir. Outros treinos de contagens
usam encoder congelado na **RTX 4060 local**, sem backpropagation ou updates;
decodificação/contagem/range coding são em CPU. Nenhuma execução na GPU remota.

## Execução e decisão

Runner: `finetuning/run_transition_index_coding.py`.
Diretório: `results/experiments/transition_index_coding_20261030/`.
Unidade: `febus-transition-index-coding-20261030.service`.

Fila automática: quatro pontos e controles, sem novos hiperparâmetros entre eles.
Protocolo/fontes/metadados têm hashes e snapshots; arquivos de treino mantêm SHA
do NPZ. Verificar todos os frames após leitura do transporte. Caches/casos completos
podem ser retomados com o mesmo protocolo, sem sobrescrever frames diferentes.

Se houver economia consistente, ela preserva a fidelidade **existente** e não
promove automaticamente o reconstrutor ou garante tolerâncias físicas. Se não
houver, encerrar sem gerar outra série de thresholds na mesma validação. Em
qualquer caso, manter o negativo do prior neural e a comparação contra o melhor
controle não contextual, em vez de só comparar com 11 bits nominais.
