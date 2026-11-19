# FEBUS: sensibilidade do manifesto F2 à posição dos relógios no bloco

**18/11/2026.** Continuação da pendência D2a nos
[relatórios 107](107-FEASIBILITY_CONTRACT_AND_EXECUTION_HANDOFF.md) e
[108](108-FEASIBILITY_DIAGNOSTICS_EXECUTION.md). Só metadados e documentação;
nenhum HDF5, waveform, GPU ou reserva foi acessado.

## 1. As fontes apoiam uma interpretação, mas não a certificam

- A [documentação a1das](https://a1das-v2-olivier-coutant-fa984751404466eb2f1c34128bcd5750d522ff.pages.in2p3.fr/html/module_core.html#about-febus-optics-das-a1-file-format),
  desta vez acessada, é da **versão 2.3.5**. Descreve `[NBLOCK, NTIME, NSPACE]`,
  tempo absoluto por bloco e overlap percentual. Não certifica as coordenadas
  de tempo do exportador **2.3.24** do acervo.
- O [leitor Xdas](https://raw.githubusercontent.com/xdas-dev/xdas/main/xdas/io/febus.py)
  usa `Spacing[1]/1000` e, por default, considera o tick central e o trimming
  simétrico. A [documentação](https://xdas.readthedocs.io/en/stable/user-guide/io/data-formats.html)
  recomenda fornecer esses parâmetros explicitamente, pelas diferenças entre
  versões. O default não é evidência instrumental.
- O [leitor DASCore v0.1.21](https://raw.githubusercontent.com/DASDAE/dascore/v0.1.21/dascore/io/febus/a1utils.py)
  usa `Origin[1]` em milissegundos como offset relativo ao tick e assume
  trimming simétrico. Não valida especificamente 2.3.24 e constrói uma grade
  regular a partir do primeiro timestamp; isso não preserva necessariamente a
  variabilidade observada nos ticks do corpus.

Os 87 cabeçalhos de treino são compatíveis com 21.000 samples a 20 kHz,
`Origin[1] = −525`, `Spacing[1] = 0,05` e passos de clock próximos de 1 s.
Em milissegundos, o origin colocaria o tick no centro do bloco de 1,05 s.
É uma evidência de consistência, não suficiente para descartar outras
referências de tempo ou offsets entre arquivos.

## 2. Teste conservador por envelopes

Se, **condicionalmente**, os ticks forem segundos em base comparável entre
arquivos e cada tick cair dentro do seu próprio bloco de duração `L = 1,05 s`,
todo suporte possível do bloco está contido em `[tick−L, tick+L]`. O teste
dispensa conhecer a posição intrabloco exata e a colocação do overlap; não
dispensa aquelas duas hipóteses. Interseção de envelopes indica possibilidade,
não sobreposição de waveform comprovada.

O [script](../tools/audit_feasibility_clock_envelopes.py) aplica o envelope aos
**486 blocos dos 87 arquivos train**, incluindo os 43 checks históricos. Une
componentes transitivamente, mantém as atribuições originais quando possível
e não escolhe outro check para salvar uma configuração.

[Resultado congelado](../results/experiments/feasibility_clock_envelopes_20261118/results.json):

| Critério condicional | Interseções entre arquivos | Pool remanescente |
|---|---:|---:|
| Posição conforme hipótese histórica (108) | 345 | 32 |
| Qualquer posição intrabloco, por envelope | 534 | 30 |

O envelope cria uma ligação direta a check histórico em multicore/C2 e C3;
por propagação, os dois arquivos check desse componente saem. Restam **20 fit,
oito check e dois já bloqueados**, em papéis candidatos. C1/C4 continua com uma
componente; C2/C3 também perde o contraste. **Oito configurações** mantêm um
contraste candidato. Isso não prova dependência física entre os dois recortes,
nem certifica independência de quem ficou.

## 3. Decisão

### Evidência mínima para retomar

Solicitar ao responsável pela aquisição ou ao fornecedor, para o exportador
**2.3.24** usado neste acervo:

1. Qual é a unidade e a base de tempo de `Source/time`? São comparáveis entre
   arquivos e núcleos? Há resets, relógios não sincronizados ou offsets de
   exportação? Se houver, quais arquivos compartilham a mesma base?
2. A que amostra se refere cada timestamp? Qual é a fórmula que liga o tick
   e o índice temporal do HDF5, incluindo `Origin[1]`, `Spacing[1]` e `Extent`?
3. Como `BlockRate=1000` e `BlockOverlap=5` se traduzem em número e posição de
   amostras repetidas, considerando 21.000 amostras a 20 kHz?

Anexar documentação versionada, declaração do responsável ou exemplo
independente com timestamps de aquisição conhecidos. Com a fórmula completa,
recalcular o grafo de sobreposições antes de qualquer leitura de waveform.
Com confirmação apenas da base comparável e da posição do tick dentro do
bloco, usar o envelope conservador para um diagnóstico restrito, sem alegar
cobertura multicore. Se a base comparável não puder ser estabelecida, não
executar D2c sob esse desenho.

**D2c permanece bloqueado no desenho dos relatórios 107–108.** Não usar a leitura
mais favorável de clocks para manter a cobertura multicore. Para reabrir este
desenho, obter evidência auditável sobre escala, base de tempo e vínculo entre
tick e bloco para essa versão/exportação, depois reconciliar slots e componentes
antes de qualquer leitura. Se for confirmado apenas que o tick pertence ao
bloco e usa tempo comparável, o critério de envelope poderá sustentar um
diagnóstico mais restrito, não um passe de quatro famílias.

F3 continua dependendo de sessões documentadas. O diagnóstico sintético de
108 permanece válido; nenhuma estimativa de oportunidade F2 no FEBUS foi
produzida nesta etapa.

## 4. Revalidação sob as premissas aceitas

O usuário autorizou assumir `Source/time` em segundos, centrado no bloco e
com base comparável entre arquivos e núcleos, sem offsets de exportação.
Isso é uma **premissa de análise**, não certificação do fornecedor.
O [script](../tools/revalidate_feasibility_centered_clocks.py) revalidou os
486 intervalos brutos com `t(i)=tick+(i−10500)/20000`, sem atravessar
fronteiras de blocos. Das **345 interseções entre arquivos**, nenhuma
conecta fit a check, nem um arquivo elegível a componente ligado aos
checks históricos excluídos, sob essas premissas.

[Manifesto revalidado](../results/experiments/feasibility_centered_clock_revalidation_20261118/results.json):
**20 arquivos fit, 10 check, 17.575.000 células planejadas**.
C2/C3 volta ao contraste condicional; C1/C4 segue sem componentes
suficientes. Não há alegação de independência de sessões nem passe nas
dez configurações. Nenhum waveform foi aberto nesta revalidação.
