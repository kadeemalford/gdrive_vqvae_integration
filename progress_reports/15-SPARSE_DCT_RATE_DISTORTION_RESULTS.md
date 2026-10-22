# Controle clássico: resultados rate-distortion com DCT esparsa

## 1. Conclusão e mudança de rota

O controle clássico não domina o VQ nas quatro famílias, mas **supera a rede em
algumas configurações dentro de seu orçamento nominal**, especialmente single-mode.
Mais bits permitem diminuir bastante o erro global, sem resolver automaticamente
os canais de pequena amplitude. Portanto, taxa e representação importam, mas o
benchmark não demonstra um limite informacional nem falta de dados como causa única.

Próxima hipótese neural: comparar a geometria temporal padrão com mais posições
latentes no tempo, mantendo alvo, normalização, parâmetros e stride espacial.
O [relatório 16](16-TEMPORAL_LATENT_DENSITY_PILOT.md) descreve essa ablação.

## 2. Escopo e comparação

Benchmark concluído: `results/experiments/sparse_dct_rate_control_20261008/`.

- CPU, sem treino nem forwards adicionais de rede.
- 210 janelas de treino + 102 de validação do fold 1, duas por arquivo, seed 2026.
- Controle FSD1: DCT 2D ortonormal, top coeficientes, índices explicitamente
  transmitidos, quantização escalar de oito bits e escala, header, normalização e escapes.
- Decodificação do byte stream real e score sobre o sinal original completo.
- Orçamentos 1×/4×/16×/64× do payload nominal VQ global por geometria.
- Referência de rede: scores salvos do piloto global epoch 6 nas mesmas janelas.
  A população é **a amostra de 102 janelas**, não a validação completa do relatório 14.

O DCT usa no máximo o orçamento, com pequenos arredondamentos. Seu stream é real;
a taxa VQ é nominal. Nenhum deles inclui todos os custos de implantação/modelos/
mapas físicos. Não confundir orçamento maior com comparação à mesma taxa.

## 3. Erro e correlação na validação

| Controle | RMSE macro | Correlação macro |
|---|---:|---:|
| VQ global nominal | 0,867924 | 0,448475 |
| DCT ≤1× orçamento | 0,884017 | 0,378736 |
| DCT ≤4× orçamento | 0,831461 | 0,500533 |
| DCT ≤16× orçamento | 0,716508 | 0,671316 |
| DCT ≤64× orçamento | 0,492539 | 0,860782 |

| Família | VQ global | DCT 1× | DCT 4× | DCT 16× | DCT 64× |
|---|---:|---:|---:|---:|---:|
| Doped | 0,832464 | 0,962563 | 0,907687 | 0,778783 | 0,523764 |
| Multicore | 0,924606 | 0,941377 | 0,884911 | 0,755423 | 0,506027 |
| Multimode | 0,964664 | 0,961288 | 0,916511 | 0,810628 | 0,595192 |
| Single-mode | 0,749964 | 0,670840 | 0,616734 | 0,521199 | 0,345174 |

Mesmo a 64× orçamento, RMSE macro ~0,493 ainda não estabelece fidelidade suficiente
sem requisitos de aplicação. Essa curva caracteriza um codec/base/quantizador
simples; outro codec pode oferecer trade-off melhor.

### Configurações no orçamento próximo do VQ

| Single-mode | VQ RMSE | DCT 1× RMSE | VQ correlação | DCT correlação |
|---|---:|---:|---:|---:|
| Configuration 11 | 0,703713 | 0,705179 | 0,609003 | 0,626471 |
| Configuration 3 | 0,796119 | 0,699506 | 0,570519 | 0,638444 |
| Configuration 7 | 0,750059 | 0,607835 | 0,627497 | 0,709786 |

Ganhos relevantes em 3/7; RMSE ligeiramente pior em 11. DCT 1× também supera
ligeiramente o RMSE multimode, mas perde bastante em doped. Não transformá-lo em
codec generalista padrão nem montar um roteador oracle por janela. Um sistema
híbrido por metadados é alternativa de pesquisa, com custos e fallback explícitos.

## 4. Taxa e distribuição por canal

Na validação, DCT 1× apresenta médias de razões por família de ~**543–718×**;
4×, ~**136–180×**; 16×, ~**34–45×**; 64×, ~**8–11×**.
Consultar `pooled_rates.csv` para razão por soma dos bytes, em vez de tratar uma
média de razões por janela como taxa total. Os limites são locais a cada geometria.

Mais bits e melhor RMSE global não garantem menos erro em cada canal. Em multimode,
o erro relativo médio dos canais fora do grupo fraco diagnóstico é ~**1,087 no
DCT 1×**, contra ~**1,054 no VQ**, e pode piorar em orçamentos maiores. DCT global
preserva mais energia dos componentes dominantes e pode gerar erro/leakage em
linhas fracas; a escala global de quantização também é uma limitação deste controle.

Não remover os canais nem mudar o alvo para favorecer esse codec. Medir erro
absoluto, mediana/p90, regiões ordinárias e componentes de baixa variância junto
com bitrate. A fração de energia retida antes da quantização não é qualidade
final garantida: a decodificação e o score raw é que avaliam o stream.

## 5. Consequências para arquitetura, dados e validação

1. A vantagem DCT em single-mode mostra que o VQ transferido não aproveita toda a
   estrutura disponível nesse orçamento; isso não prova que o codebook sozinho
   seja o gargalo, mas enfraquece atribuir toda a falha a uma taxa impossível.
2. A melhora com orçamento maior justifica **ablação neural de densidade/taxa**, não
   apenas mais epochs sob a mesma geometria extremamente comprimida.
3. DCT 4× usa mais bits e também muda a forma de codificação. Ele não demonstra
   que uma rede com quatro vezes mais índices atingirá seu resultado.
4. Os dados extra/treino do zero permanecem opções condicionais, mas não são
   necessários para executar esse primeiro controle de arquitetura/taxa.
5. Candidatos promissores exigem validação completa, outros folds/seeds e novo
   holdout de aquisição/configuração. Testes não foram consultados neste benchmark.

Nenhum modelo/arquivo de dados foi substituído. O benchmark é diagnóstico de
rate-distortion, não codec final nem demonstração de reconstrução suficiente.
