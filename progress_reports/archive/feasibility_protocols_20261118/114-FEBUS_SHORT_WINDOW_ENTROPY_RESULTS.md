# FEBUS: redundância de curto alcance existe, mas instalação custa mais no painel

**18/11/2026.** Resultado do [protocolo 113](113-FEBUS_SHORT_WINDOW_ENTROPY_TRIAGE.md),
preservado sem alterações. [Evidência por configuração e precisão](../results/experiments/feasibility_npz_probe_20261118/SUMMARY.md).

## Resultado

Foram avaliadas oito janelas de check (271.000 células), após ajuste em 18
janelas (656.000 células), de oito configurações e três famílias, sem
multicore. Os **48 transportes** foram redecodificados independentemente.
Mesmos índices quantizados e reconstrução bit a bit entre coders;
**zero violações do limite de erro**. A quantização em si não é lossless.

| ε/σ de calibração | Marginal, bpc de transporte | Espaço-temporal, bpc de transporte | Variação de bytes de transporte | Variação do objeto autocontido |
|---|---:|---:|---:|---:|
| 0,03 | 8,250 | 7,782 | −5,68% | +65,87% |
| 0,1 | 5,452 | 4,690 | −13,98% | +20,79% |

Com instalação compartilhada, o contextual espaço-temporal entrega cerca de
**4,1× e 6,8×** sobre float32, respectivamente. Esses pontos têm contratos
de erro e populações diferentes dos números históricos de VQ: não são
comparações diretas de eficiência com a referência madura.

## A economia se paga com instalação compartilhada?

Neste recorte, a instalação incremental espaço-temporal custa **244.020 B**
a 0,03 e **77.755 B** a 0,1. A economia de transporte é **15.876 B** e
**25.819 B**, respectivamente. Em uma projeção aritmética, seriam necessários
**16 ou quatro volumes de check equivalentes** para empatar/superar o custo,
**somente se** a mesma instalação puder ser reutilizada e as economias se
mantiverem sem novos custos. Não medimos essa estabilidade; repetir o painel
não constituiria confirmação. Os objetos autocontidos medidos têm outro
enquadramento, portanto não são somados a esse cálculo.

## Decisão

Há redundância aproveitável de curto alcance neste recorte. Ela não aparece
de forma uniforme: o contexto espaço-temporal não reduz a CDF em multimode
em nenhuma das duas precisões. Somente a precisão menos estrita passa o
limiar descritivo de transporte de 10%; nenhuma passa no objeto autocontido
agregado. Há apenas um arquivo de check por configuração.

**Encerrar a triagem como evidência localizada, sem iniciar F4-A.** Antes de
outro treino, seria necessário demonstrar economia líquida com instalação
reutilizável ou menor, sob horizonte declarado e em arquivos adicionais,
incluindo multicore para qualquer conclusão generalista. O F2 contínuo
continua bloqueado pela semântica temporal, F3 pela falta de novas sessões
e F4-B permanece encerrado. Não substituir essas evidências pelas reservas.

Execução em CPU, **29 testes aprovados**, sem HDF5, reservas, GPU ou treino.
