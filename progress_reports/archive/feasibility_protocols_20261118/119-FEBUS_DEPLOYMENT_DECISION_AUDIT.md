# FEBUS: o que o ganho de transporte resolve e o que ainda falta definir

Auditoria dos resultados congelados de [transferência](118-FEBUS_FROZEN_INSTALLATION_TRANSFER_RESULTS.md).
Não há leitura de sinais, novo treino, novo gate ou extrapolação da taxa.
[Script](../finetuning/audit_feasibility_deployment.py) ·
[evidência por configuração](../results/experiments/feasibility_deployment_audit_20261118/results.json).

## Pontos medidos, com a instalação cobrada

São 13.780.000 ocorrências de células, 380 janelas, 19 arquivos e oito
configurações, sem multicore. A instalação foi paga uma vez por configuração.
O ponto de precisão não é um erro relativo por amostra: ε é fixo para cada
configuração, a partir do desvio de calibração P0.

| ε/σ de calibração | Marginal | Espaço-temporal | Razão sobre float32 (espaço-temporal) | Economia contra marginal |
|---|---:|---:|---:|---:|
| 0,03 | 6,882 bpc | 6,524 bpc | 4,90× | 5,20% |
| 0,1 | 4,738 bpc | 4,070 bpc | 7,86× | 14,10% |

Trocar 0,03 por 0,1 permite erro absoluto **3,33 vezes maior**. Os RMSEs
nativos observados crescem aproximadamente na mesma proporção; não usar essa
troca como ganho à mesma fidelidade. O JSON de evidência separa bound, máximo
observado e RMSE por configuração. Não há medição nova de erro de fase,
amplitude ou evento. Estas taxas não são comparáveis diretamente com os
pontos de VQ sem avaliação sob a mesma tolerância e população.

## A precisão estrita não falhou só por overhead de instalação

A 0,03, o contextual economiza **7,30%** em transporte contra transporte.
Mesmo se fosse possível remover **toda** a sua própria instalação sem alterar
o stream, sua economia contra o marginal com a instalação cobrada seria
de apenas **7,86%**. Ainda faltariam **254.248 B** de redução no transporte
para alcançar 10% neste painel. Com a instalação atual faltam **569.133 B**.
É uma conta sobre estes bytes, não um limite de taxa–distorção do FEBUS.

Assim, alongar a implantação pode amortizar as tabelas, mas não deve ser
apresentado como solução para cruzar o gate estrito sem evidência de que
o próprio transporte também melhora. O resultado de 0,1 é tecnicamente útil
quando a aplicação aceitar esse erro e a instalação for compartilhável;
não justifica sozinho um treino neural ou uma promessa de taxa ultracompacta.

## Próximas decisões, sem reabrir negativos

1. **Contrato de uso:** confirmar o produto/unidade armazenados e fixar o
   menor conjunto de restrições que pode desqualificar uma reconstrução:
   limite absoluto ou quantil extremo, viés, erro de amplitude/fase nas bandas
   relevantes e cobertura por família. Preservar waveform e realização de
   ruído continua sendo o objetivo; mudar para detecção é outro contrato.
2. **Objeto de implantação:** declarar duração típica, frequência de acesso
   aleatório e se decoder/tabelas podem ser instalados e reutilizados. Reportar
   taxa com e sem instalação, sem confundir os dois contratos.
3. **Confirmação:** quando houver tolerâncias e sessões documentadas, fixar
   antes de avaliar um candidato e um painel que inclua multicore. Tratar a
   VQ madura como referência; comparar no mesmo contrato físico, sem reusar
   reservas para escolher candidato. O F2 contínuo ainda exige documentação
   da semântica temporal.

**Interrupção científica, não operacional:** os contrastes de F4-B e da
instalação congelada estão completos. O critério de F4-A não passou e F3
não tem sessões novas. Não há outra campanha de treino justificada pelos
resultados atuais. O insumo necessário agora é o requisito de uso, não mais
updates ou flexibilidade no gate para iniciar outro treino.

## Referências provisórias de progresso

Sem tolerância de produto, usar comparações pareadas: **2–3% menos NRMSE\***
sem mais bytes completos, ou **3–5% menos bytes** com a mesma reconstrução
ou fidelidade não inferior. São marcos de pesquisa, não condição para
reconhecer todo ganho: efeitos menores, especialmente bit-exatos e baratos,
merecem registro se forem consistentes e não vierem de perda de cobertura.
Checar SSE, viés, caudas por canal e amplitude/fase por banda; sem margens
justificadas, não presumir não inferioridade.

Na VQ madura do [112](112-FEBUS_CONTEXT_ALIGNMENT_RESULTS.md), isso equivale
aproximadamente a NRMSE* **0,476–0,481 a até 0,700 bpc**, ou **0,665–0,679 bpc**
à fidelidade de referência (NRMSE* 0,49034 a 0,69969 bpc, mesmo painel e wire).
São números prospectivos, não previsões. Para codificação de erro explícito,
manter `ε/σ_cal=0,03` como âncora mais estrita e `0,1` como alternativa de taxa:
3% e 10% da escala por configuração, não da amplitude de cada amostra.

Os **5,20%** no ponto estrito de 118 já são uma melhoria de engenharia neste
recorte, embora o critério conjunto histórico permaneça reprovado. Sete dos
oito grupos economizaram; retirar um por vez deixa **4,55–6,51%** de economia
agregada ([sensibilidade dos bytes congelados](../results/experiments/feasibility_deployment_audit_20261118/transfer_sensitivity.json)).
Não é intervalo de confiança: o painel não inclui multicore, multimode não
economizou nesse contraste, e a pouca variedade temporal restringe conclusões
sobre outros eventos e sessões. A diversidade de fibras/configurações do
interrogador exige efeitos por grupo; milhares de amostras próximas não
substituem regimes novos. Nenhum desses pontos aprova fidelidade de produto.
