# Contexto na análise, antes do bottleneck discreto

**Protocolo registrado em 12/11/2026, antes dos resultados.**

## 1. Hipótese distinta dos probes encerrados

O relatório 84 testou adaptar **tokens já quantizados** para melhorar síntese.
Mesmo atenção não atingiu o limiar de avanço. Isso não mede a capacidade de
contexto entre **features contínuas do encoder** para escolher símbolos melhores:
depois da escolha, parte da informação original já não está disponível.

O presente probe adiciona um adaptador antes da quantização e pode mudar os
IDs transmitidos. A fonte é a referência A compacta madura da época 24, com
encoder, embedding e decoder originais **congelados**. Não transmite features
contínuas nem adiciona informação auxiliar do alvo ao receptor.

O negativo do diagnóstico de escala local (relatório 86) não sustenta treinar
hyperprior. A direção volta à qualidade da informação codificada, e não a mais
empacotamento marginal dos mesmos coeficientes.

## 2. Controle e orçamento

- Dois adaptadores residuais inicialmente identidade: **MLP local, 34.052
  parâmetros**, versus **atenção axial, 33.824 parâmetros**. Diferença <1%.
- Mesmas features do encoder, mesmo plano de batches, mesma origem, mesmo
  Adam novo a `1e-3`, sem weight decay, clip1, seed 2106 e FP32/TF32 desligado.
- Somente esses parâmetros novos treinam. Encoder original, codebook e decoder
  não recebem gradientes e são auditados por hash antes/depois.
- Loss histórica `0,5 MSE + 0,5 L1 + VQ`, células ordinárias preservadas. O
  commitment leva gradiente ao adaptador de análise; o codebook não muda.
- Na avaliação, recuperar **vetores exatos dos IDs**, sem resíduos numéricos de
  straight-through. No treino, preservar o gradiente ST até o adaptador. O
  quantizador alterna explicitamente entre esses modos, sem destravar os pesos.
- Mesma grade e K2048: taxa nominal igual, **bytes entrópicos não presumidos
  iguais**. O adaptador é custo de encoder; o receptor continua com o decoder
  original. Uma confirmação posterior deve cobrar stream e todos os assets.

Cada braço recebe **512 updates iniciais**, checks a cada 128. Uma única extensão
pareada a **2.048** exige ganho ≥0,5% de D contra origem e ≥0,2% recente; se a
atenção vencer, deve superar local em ≥0,25%, ou o vencedor local precisa de
≥1% contra origem. A regra não pode ser flexibilizada depois dos resultados.

## 3. Seleção e critérios de saída

Usar os mesmos arquivos internos de checks dos probes recentes (seed de seleção
2100), somente do **treino**; batches novos excluem seus IDs de arquivo. A
inicialização é elegível. Essas checks já foram vistas pelo treinamento antigo
e reutilizadas no desenvolvimento: não são aquisições independentes.

Selecionar menor D normalizada, ponderada por células dentro de configuração,
configurações e famílias iguais. Só abrir uma confirmação de desenvolvimento
se o melhor braço ganhar **≥1% de D contra origem**; alegar ganho específico
de contexto exige ainda **≥0,5% contra local**. Os percentuais de D **não são
RMSE físico**. O gate final de codec permanece ≥5% menos RMSE físico com bytes
completos não maiores e sem regressões nas guardas por configuração.

Se negativo, encerrar sem extensão extra. Isso limita o adaptador de análise
com backbone congelado, não refuta Transformer treinado ponta a ponta ou um
codec inteiramente novo. Não reinterpretar os resultados anteriores de síntese
como se tivessem medido esta hipótese.

## 4. Implementação e execução

`finetuning/analysis_context.py`, `finetuning/run_analysis_context_probe.py` e
`tests/test_analysis_context.py`. Os testes verificam gradiente através de ST
somente para o adaptador e replay sem informação contínua ou resíduo de ST.

Somente uma RTX 4090 remota, GPU física 0, braços sequenciais. Primeiro smoke
de dois updates por braço, incluindo maior tile, depois o orçamento fixado.
Saídas próprias `analysis_context_smoke_fold0_20261112` e
`analysis_context_train_probe_fold0_20261112`, em `results/experiments/`.
Sem inferência de formas de onda de validação ou teste nesta etapa.

## 5. Resultado concluído: vantagem relativa pequena, avanço não atingido

A campanha acionou a extensão predefinida aos 512 updates e completou **2.048
updates por braço**, sequencialmente. A auditoria posterior verificou contadores
Adam iguais a 2.048, linhagem e hashes de fonte/código/vendors. Os parâmetros
base permaneceram imutáveis pelo teste do runner. Históricos, protocolos e
adaptadores selecionados foram copiados para o diretório local correspondente.

| Variante | Melhor D de triagem ↓ | Update selecionado |
|---|---:|---:|
| Origem congelada | 0,371929769 | 0 |
| Adaptador local de análise | 0,371542255 | 512 |
| Atenção axial de análise | **0,369617199** | 1.536 |

Axial melhorou **0,6218% contra a origem** e **0,5181% contra local**. Assim,
ultrapassou o limiar relativo de 0,5% para a vantagem contextual **nesta triagem**,
mas **não** o limiar absoluto de 1% contra origem para abrir confirmação. Não
apresentar esse primeiro critério isolado como sucesso de codec ou significância
estatística. São checks de treino reutilizadas, uma seed e loss normalizada.

Os endpoints foram piores que os selecionados: D local **0,37173046** e axial
**0,37117024** ao update 2.048. Não estender porque o melhor snapshot intermediário
favorece uma hipótese; o teto e a seleção foram fixados antes do resultado.

**Decisão:** encerrar o probe sem abrir validação, sem novo budget e sem promoção.
O contraste não sustenta investir em outro adaptador pequeno sobre esta base
congelada. Contexto antes e depois da quantização teve ganhos modestos; as
diferenças entre seus números não são uma ablação causal de posição, pois planos
de treino e objetivos não são idênticos entre campanhas.

Nenhuma janela de sinal de validação/teste foi lida. Não há medição nova de RMSE
físico ou bytes completos: a igualdade nominal de grade/K não substitui isso.
O resultado continua sem refutar uma arquitetura de análise/síntese contextual
treinada conjuntamente; ela exigiria um protocolo estrutural próprio, não a
continuação oportunista destes adaptadores.
