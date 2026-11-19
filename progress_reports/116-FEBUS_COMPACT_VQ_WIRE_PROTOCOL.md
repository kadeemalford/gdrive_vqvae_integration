# FEBUS: reduzir metadados redundantes sem mudar a reconstrução VQ

**Protocolo prospectivo.** O [estudo das tabelas](115-FEBUS_POOLED_CONTEXT_TABLES.md)
encerrou o compartilhamento de linhas de probabilidade. Aqui o mecanismo
é outro: reduzir redundância estrutural do wire **sem alterar probabilidades,
índices ou pesos**, na própria VQ madura.

O censo só de bytes dos 136 bundles da VQ madura do
[relatório 111](111-FEBUS_TRAIN_INFERENCE_CONTEXT_ALIGNMENT.md) encontrou 134 frames
FAK1 sem zeros no overlay e dois frames literais com 20.000 posições de zero
ao todo. Isso sugere economia de framing, mas também que o ganho agregado
pode depender de poucos tiles constantes. Não excluir nenhum deles.

## Contraste

- Usar somente os bundles existentes da VQ madura no 112, com seus hashes
  e a geometria de transporte. Sem nova seleção, waveform de entrada,
  carregamento de pesos, inferência ou GPU.
- Empacotar por um formato ordenado, com offsets explícitos, integridade
  verificada e acesso independente a cada tile. Cobrar container, índice e
  geometria uma vez; declarar registry e receiver como instalação comum.
- Omitir o wrapper de partição de um único fragmento **somente** quando a
  cobertura e geometria forem deriváveis sem perda. Omitir o overlay quando
  vazio, ou quando o literal já reproduz exatamente todos os bits que ele
  imporia, inclusive zeros com sinal. Casos não demonstrados ficam em fallback
  completo, incluindo seus bytes. A regra não usa score ou ajuste por tile.
- Verificar por um leitor independente a indexação, cobertura, integridade e
  todos os frames. Exigir identidade dos bytes FAK1 internos quando o overlay
  é vazio; testar valores literais exatos e fallback. Para FAK1 isso é
  equivalência do caminho até o decoder instalado, **não** uma nova inferência
  neural nem verificação empírica de waveform no hardware.

## Decisão

**Primário:** economizar pelo menos 3% do transporte completo da VQ madura
no mesmo painel, mantendo cobertura, acesso por tile e equivalência sem
perda. Publicar também a decomposição entre tiles literais e não literais,
para não projetar casos raros para toda aquisição. Ganho menor, se
reproduzível, é melhoria pontual de formato; não implica promoção, ganho
do modelo, nem novo treino. Não substituir a referência histórica nem
usar validação, teste ou EDAS.

## Resultado

Nos mesmos 136 tiles e 12.799.000 células, a taxa passou de **0,69969 para
0,67874 bpc**: **33.520 B (2,9944%)** a menos. Faltaram 63 B para o limiar
de 3%. Dois literais respondem por 84% da economia; nos demais tiles a
redução de bytes foi 0,49%, antes do cabeçalho comum. Os bits dos literais
foram conferidos; para FAK1 a garantia é equivalência da entrada do decoder,
não nova inferência de waveform. Oito testes passaram. Preservar como
melhoria pontual de formato, sem trocar a referência.
[Resultados e hashes](../results/experiments/feasibility_compact_bundle_20261118/results.json).
