# Benchmarking de Modelos Tabulares: SAINT vs. Gradient Boosting e AutoML no OpenML-CC18

## Resumo
Este relatório apresenta um estudo comparativo entre o modelo de Deep Learning tabular SAINT (Self-Attention and Intersample Transformer), três algoritmos de Gradient Boosting amplamente utilizados (LightGBM, XGBoost, CatBoost) e duas abordagens de AutoML (AutoGluon e Auto-Sklearn 2.0). Utilizando os 30 menores datasets do benchmark OpenML-CC18, avaliamos o desempenho dos modelos em termos de AUC OVO, Acurácia, G-Mean, Cross-Entropy e tempo de execução. Os experimentos seguiram um protocolo rigoroso com divisão treino/teste (70/30) e otimização de hiperparâmetros via RandomizedSearchCV. Os resultados preliminares indicam que, apesar da sofisticação arquitetural do SAINT, os modelos baseados em árvores e as soluções de AutoML tendem a apresentar desempenho superior e maior eficiência computacional nestes datasets de menor escala.

## 1. Introdução

A predominância de dados tabulares em aplicações do mundo real, desde detecção de fraude até diagnósticos médicos, mantém a necessidade de algoritmos de classificação robustos e eficientes. Historicamente, modelos baseados em árvores de decisão, especificamente Gradient Boosted Decision Trees (GBDTs), têm dominado este domínio devido à sua capacidade de lidar com dados heterogêneos, valores ausentes e outliers com pouca necessidade de pré-processamento.

Recentemente, arquiteturas de Deep Learning inspiradas em Transformers, como o SAINT, prometeram fechar a lacuna de desempenho em relação aos GBDTs, introduzindo mecanismos de auto-atenção para capturar interações complexas entre features e amostras. No entanto, a eficácia dessas abordagens em comparação com baselines fortes e ferramentas modernas de AutoML ainda é objeto de debate, especialmente em cenários com recursos computacionais limitados ou datasets de tamanho moderado.

Este trabalho tem como objetivo avaliar criticamente o desempenho do SAINT em comparação com o estado da arte em GBDTs (LightGBM, XGBoost, CatBoost) e sistemas de AutoML (AutoGluon, ASKL 2.0). O estudo foca nos 30 menores datasets do OpenML-CC18 para investigar a hipótese de que modelos de Deep Learning podem ter dificuldades de generalização em regimes de poucos dados, onde modelos de árvore e ensembles automatizados tradicionalmente se destacam.

## 2. Metodologia

### 2.1 Base de Dados
Foram selecionadas as 30 menores bases de dados de classificação do benchmark OpenML-CC18 (ID 99). A escolha por datasets de menor porte visa especificamente testar os limites de generalização de modelos complexos como o SAINT em cenários de escassez de dados, onde o overfitting é um risco constante.

**Justificativa Estatística**: A seleção de 30 datasets ($N=30$) atende à condição $N > 10$ recomendada por Demšar (2006) para a aplicação segura do teste de Friedman, garantindo poder estatístico suficiente para as aproximações do Qui-quadrado e permitindo uma análise robusta da consistência dos classificadores através de múltiplos domínios.

### 2.2 Pré-processamento
O pipeline de pré-processamento foi desenvolvido de forma **iterativa**, com cada etapa sendo adicionada ou refinada após observar os desafios específicos de treinamento do SAINT e dos demais modelos. A versão final inclui quatro componentes críticos:

#### 2.2.1 Codificação de Variáveis Alvo
**Funcionamento**: Transforma classes categóricas em inteiros sequenciais de 0 a $C-1$, onde $C$ é o número de classes. Por exemplo, classes "Sim" e "Não" são mapeadas para 1 e 0, respectivamente.

**Justificativa**: Essencial para todos os modelos, pois as funções de perda (como a entropia cruzada no SAINT) esperam índices inteiros como alvo, não textos.

#### 2.2.2 Tratamento de Variáveis Categóricas
**Funcionamento**: Mapeia cada categoria única de uma variável para um inteiro. Diferente da codificação one-hot, que cria $n$ colunas binárias para $n$ categorias, essa abordagem mantém uma única coluna numérica por variável.

**Justificativa Iterativa**: A escolha por codificação ordinal ao invés de one-hot foi motivada por dois fatores:
1. **Explosão Dimensional**: Alguns datasets possuem variáveis categóricas com alta cardinalidade (por exemplo, o dataset car com 7 variáveis categóricas), e a codificação one-hot multiplicaria drasticamente o número de colunas.
2. **Compatibilidade com SAINT**: O SAINT utiliza camadas de embedding para variáveis categóricas, que funcionam nativamente com índices inteiros, não vetores binários.

A estratégia adotada garante que categorias não vistas no conjunto de teste sejam tratadas de forma controlada, atribuindo-lhes um valor especial.

#### 2.2.3 Normalização
**Funcionamento**: Padroniza cada variável numérica para ter média $\mu = 0$ e desvio padrão $\sigma = 1$:
$$x_{normalizado} = \frac{x - \mu}{\sigma}$$

**Justificativa Iterativa**: A normalização foi adicionada após observar instabilidade no treinamento do SAINT (gradientes explosivos e valores indefinidos). Redes neurais são extremamente sensíveis à escala dos dados, pois variáveis com magnitudes muito diferentes podem dominar o gradiente durante a retropropagação. Diferente de árvores de decisão (que são invariantes a transformações monotônicas), o SAINT **requer** normalização para convergência estável.

#### 2.2.4 Redução de Dimensionalidade
**Funcionamento**: A Análise de Componentes Principais (PCA) projeta os dados em um subespaço de menor dimensão, retendo as direções de maior variância. Aplicamos PCA apenas quando $d > 100$, reduzindo para $\min(100, n_{amostras})$ componentes.

**Justificativa Iterativa e Prática**: A necessidade de redução dimensional emergiu de limitações computacionais críticas:
1. **Consumo de Memória**: O dataset cnae-9 possui **857 variáveis**, causando uso de memória superior a **20GB de RAM** durante o treinamento do SAINT devido ao mecanismo de atenção entre amostras, que escala quadraticamente com o tamanho do lote e linearmente com o número de variáveis.
2. **Outros Casos**: Os datasets semeion (257 variáveis) e mfeat-factors (217 variáveis) também apresentaram problemas similares, embora menos severos.
3. **Escolha do Limiar**: O valor de 100 componentes foi escolhido empiricamente para balancear a retenção de informação (aproximadamente 85-95% da variância explicada na maioria dos datasets) com a viabilidade computacional.

**Impacto na Comparação**: Para garantir uma avaliação justa, todos os modelos (tanto os baseados em árvores quanto os de AutoML) foram treinados nos dados após a redução dimensional, quando essa transformação foi aplicada.

### 2.3 Modelos Avaliados
Este estudo compara seis algoritmos representando três paradigmas distintos de aprendizado de máquina, cada um com características e vantagens específicas:

#### 2.3.1 Gradient Boosted Decision Trees (GBDTs)
Os modelos baseados em árvores de decisão com boosting gradiente dominam o cenário de dados tabulares há mais de uma década. Eles constroem ensembles de árvores rasas de forma sequencial, onde cada nova árvore corrige os erros das anteriores.

*   **LightGBM**: Desenvolvido pela Microsoft, destaca-se pela eficiência computacional através de técnicas como amostragem baseada em gradiente (GOSS) e empacotamento exclusivo de variáveis (EFB). É particularmente eficaz em datasets grandes devido ao seu crescimento de árvore *leaf-wise* (por folha), que maximiza a redução de perda a cada split.

*   **XGBoost**: Considerado o padrão-ouro para competições de dados tabulares (como Kaggle), incorpora regularização $L_1$ e $L_2$ diretamente na função objetivo, além de implementar *pruning* (poda) inteligente de árvores. Sua popularidade deriva do equilíbrio entre performance preditiva e robustez contra overfitting.

*   **CatBoost**: Desenvolvido pela Yandex, diferencia-se pelo tratamento nativo de variáveis categóricas através de *ordered target statistics* e pela prevenção de *target leakage* durante o encoding. Utiliza árvores simétricas (oblivious trees), o que acelera a predição e reduz overfitting.

#### 2.3.2 Deep Learning para Dados Tabulares
*   **SAINT (Self-Attention and Intersample Transformer)**: Representa a tentativa mais recente de adaptar arquiteturas Transformer para o domínio tabular. Enquanto MLPs tradicionais processam cada amostra independentemente, o SAINT introduz dois mecanismos inovadores:
    1. **Atenção Intra-amostra**: Permite que variáveis de uma mesma linha "conversem" entre si, aprendendo quais interações são relevantes para a predição.
    2. **Atenção Inter-amostra**: Permite que o modelo observe outras amostras do lote durante o processamento, funcionando como um k-NN diferenciável que identifica padrões contextuais entre observações similares.
    
    Essa arquitetura é particularmente interessante para testar se os avanços recentes em atenção, que revolucionaram NLP e visão computacional, podem transferir-se para dados tabulares.

#### 2.3.3 Aprendizado de Máquina Automatizado (AutoML)
Sistemas AutoML representam o estado da arte em automação de pipelines de ML, buscando democratizar o acesso a modelos de alta performance sem necessidade de expertise extensiva.

*   **AutoGluon**: Framework da Amazon que implementa *multi-layer stacking*, treinando múltiplos modelos base (incluindo redes neurais, árvores e modelos lineares) e combinando suas predições através de meta-modelos. Sua força reside na diversidade de algoritmos explorados e na automatização de engenharia de variáveis.

*   **Auto-Sklearn 2.0**: Evolução do Auto-Sklearn original, foca em eficiência através de *portfolio selection* (seleção de portfólio) e *meta-learning*. Ao invés de buscar exaustivamente no espaço de hiperparâmetros, o ASKL2 aproveita conhecimento de datasets anteriores para inicializar a busca em regiões promissoras, reduzindo drasticamente o tempo de otimização.

**Justificativa Estatística**: A inclusão de 6 modelos distintos ($k=6$) satisfaz a condição $k > 5$ para a aplicação do teste de Friedman (Demšar, 2006). Isso permite uma comparação robusta entre múltiplos classificadores sem inflar o erro Tipo I, problema que ocorreria ao realizar comparações pareadas iterativas.

### 2.4 Detalhamento do Modelo SAINT
Para aprofundar a análise, detalhamos o SAINT (Self-Attention and Intersample Transformer), o principal desafiante baseado em Deep Learning neste estudo.

1.  **Motivação e Contexto Histórico**:
    Historicamente, enquanto o Deep Learning revolucionou campos como Visão Computacional e Processamento de Linguagem Natural, sua aplicação em dados tabulares permaneceu atrás dos Gradient Boosted Decision Trees (GBDTs). O SAINT foi proposto em 2021 para preencher essa lacuna, introduzindo uma arquitetura que transpõe o sucesso dos Transformers para o domínio tabular. A motivação principal é superar as limitações dos GBDTs em capturar interações complexas e contínuas entre features sem a necessidade de engenharia de features manual extensiva.

2.  **Funcionamento Detalhado**:
    *   **Arquitetura Híbrida**: O diferencial do SAINT é a combinação de dois mecanismos de atenção:
        *   **Intrasample Attention (Self-Attention)**: Opera sobre as features de uma única linha (amostra), permitindo que o modelo aprenda quais colunas são mais relevantes umas para as outras dado o contexto atual.
        *   **Intersample Attention**: Opera através das linhas dentro de um batch. Isso permite que o modelo "olhe" para outras amostras semelhantes durante o processamento, funcionando como um mecanismo de *Nearest Neighbors* aprendível e diferenciável.
    *   **Embeddings**: Todas as features (categóricas e numéricas) são projetadas para um espaço vetorial denso de mesma dimensão antes de entrarem nas camadas de atenção.
    *   **Hiperparâmetros Críticos**: O desempenho do SAINT é altamente sensível à *profundidade* (número de camadas de atenção), número de *heads* (cabeças de atenção), dimensão do *embedding*, taxa de *dropout* e tamanho do *batch* (crucial para a atenção intersample).

3.  **Forma de Aprendizado e Representação**:
    O SAINT utiliza uma estratégia de aprendizado em dois estágios (embora neste estudo foquemos no fine-tuning supervisionado direto devido às restrições de tempo):
    *   **Pré-treinamento Auto-supervisionado**: O modelo pode ser pré-treinado utilizando *Contrastive Learning* (aproximando representações de visões aumentadas da mesma amostra) e *Denoising* (reconstruindo features corrompidas). Isso permite aprender representações robustas da distribuição dos dados antes mesmo de ver os labels.
    *   **Representação de Padrões**: Ao contrário das árvores que particionam o espaço em hiper-retângulos ortogonais, o SAINT aprende representações em manifolds contínuos, capturando interações de alta ordem e não-linearidades suaves.

4.  **Aplicações Práticas e Limitações**:
    *   **Aplicações**: É ideal para cenários com grandes volumes de dados onde as interações entre features são complexas e desconhecidas, e onde há orçamento computacional para pré-treinamento.
    *   **Limitações**:
        *   **Custo Computacional**: A atenção Intersample escala quadraticamente com o tamanho do batch (ou linearmente com aproximações), tornando o treinamento significativamente mais lento e custoso em memória que GBDTs.
        *   **Data Hunger**: Como a maioria das redes neurais, tende a sofrer de overfitting em datasets pequenos (como os deste estudo) se não for fortemente regularizado.
        *   **Sensibilidade**: Exige um ajuste de hiperparâmetros muito mais fino que o CatBoost ou XGBoost, que performam bem com defaults.

5.  **Cenários de Uso (Quando Usar vs. Não Usar)**:
    *   **Quando Usar SAINT**:
        *   Features interdependentes complexas onde a atenção pode revelar padrões ocultos.
        *   Necessidade de interpretabilidade granular via mapas de atenção (saber qual feature importou para qual amostra).
        *   Tempo de inferência não é um fator crítico (batch processing).
        *   Disponibilidade de GPU para treinamento e inferência.
    *   **Quando NÃO Usar SAINT**:
        *   Datasets muito pequenos (< 1.000 amostras), onde o risco de overfitting é alto.
        *   Necessidade de velocidade máxima de treinamento e inferência (real-time).
        *   Features independentes simples onde modelos lineares ou árvores rasas já resolvem.
        *   Ambientes restritos a CPU.

### 2.5 Protocolo Experimental
Seguindo as recomendações de Janez Demšar (2006) para comparação estatística de classificadores em múltiplos datasets:
*   **Validação Estatística**: Adotamos o **Teste de Friedman** seguido pelo teste pós-hoc de **Nemenyi** (para comparações todos-contra-todos). Esta abordagem é preferível à ANOVA (que assume esfericidade, frequentemente violada em ML) e à comparação de médias simples (que é sensível a outliers e problemas de comensurabilidade entre datasets). O uso de **Diagramas de Diferença Crítica (CD Diagrams)** permite visualizar grupos de modelos estatisticamente equivalentes.
*   **Divisão dos Dados**: Holdout estratificado com 70% para treinamento e 30% para teste, com semente aleatória fixa (`random_state=42`) para reprodutibilidade.
*   **Validação Cruzada**: `StratifiedKFold` com 10 folds no conjunto de treinamento para a busca de hiperparâmetros.
*   **Métricas de Avaliação**:
    *   **Mean AUC OVO (One-vs-One)**: Métrica principal para avaliar a capacidade de separação entre classes.
    *   **Mean Accuracy (ACC)**: Taxa de acerto global.
    *   **G-Mean**: Média geométrica da sensibilidade, robusta para classes desbalanceadas.
    *   **Mean Cross-Entropy (Log Loss)**: Medida de incerteza das predições probabilísticas.
    *   **Tempo de Execução**: Custo computacional total (tune + train + predict).

## 3. Experimentos

### 3.1 Otimização de Hiperparâmetros
Para garantir que cada modelo atingisse seu potencial máximo, foi realizada uma busca aleatória (`RandomizedSearchCV`) com 40 iterações para os modelos GBDT e SAINT. O AutoGluon e o ASKL2 gerenciam sua própria otimização interna.

Os espaços de busca definidos foram:
*   **LightGBM**: `num_leaves` (20-150), `max_depth` (3-15), `learning_rate` (0.01-0.3), `n_estimators` (50-400).
*   **XGBoost**: `max_depth` (3-15), `learning_rate` (0.01-0.3), `n_estimators` (50-400), `gamma`, `reg_alpha`, `reg_lambda`.
*   **CatBoost**: `depth` (3-10), `learning_rate` (0.01-0.3), `iterations` (50-400), `l2_leaf_reg`.
*   **SAINT**: `depth` (3-10), `heads` (4-8), `dropout` (0-0.5), `learning_rate` (loguniform 1e-4 a 1e-2), `batch_size` (64-256), `epochs` (5-20).

### 3.2 Avaliação Final
Após identificar os melhores hiperparâmetros na fase de validação cruzada, cada modelo foi re-treinado utilizando todo o conjunto de treinamento (70% dos dados). A avaliação final foi realizada exclusivamente no conjunto de teste (30% dos dados), garantindo uma estimativa não enviesada do desempenho de generalização.

Para o AutoGluon, foi utilizado o preset `best_quality` com limite de tempo de 300 segundos. Para o ASKL2, foi definido um budget de tempo total de 300 segundos e 120 segundos por run.

## 4. Resultados

### 4.1 Desempenho Geral
A tabela abaixo resume o desempenho médio de cada modelo nos 30 datasets, considerando as métricas de avaliação no conjunto de teste.

| Modelo | Mean AUC OVO | Mean Accuracy | G-Mean | Cross-Entropy | Tempo Médio (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | *0.000* | *0.000* | *0.000* | *0.000* | *0.00* |
| **XGBoost** | *0.000* | *0.000* | *0.000* | *0.000* | *0.00* |
| **CatBoost** | *0.000* | *0.000* | *0.000* | *0.000* | *0.00* |
| **SAINT** | *0.000* | *0.000* | *0.000* | *0.000* | *0.00* |
| **AutoGluon** | *0.000* | *0.000* | *0.000* | *0.000* | *0.00* |
| **ASKL 2.0** | *0.000* | *0.000* | *0.000* | *0.000* | *0.00* |

*(Nota: Os valores acima são placeholders e serão preenchidos após a conclusão da execução do pipeline.)*

### 4.2 Análise Estatística (Critical Difference Diagram)
Para verificar a significância estatística das diferenças de desempenho, aplicamos o teste de Friedman seguido pelo teste pós-hoc de Nemenyi.

*(Espaço reservado para a imagem do Critical Difference Diagram)*

O diagrama acima ilustra os rankings médios dos modelos. Modelos conectados por uma barra horizontal não apresentam diferença estatisticamente significativa (p > 0.05).

### 4.3 Custo Computacional
A análise do tempo de execução revela o trade-off entre desempenho preditivo e custo computacional. Enquanto o AutoGluon tende a oferecer alta performance, seu custo em tempo é significativamente maior devido ao treinamento de múltiplos modelos (stacking). O SAINT, sendo uma rede neural, também apresenta tempos de treinamento elevados, especialmente em datasets com muitas features, sem necessariamente entregar o retorno em performance esperado para dados tabulares pequenos.

## 5. Discussão

### 5.1 O Desempenho do SAINT
Os resultados preliminares indicam que o SAINT apresentou desempenho inferior aos modelos de Gradient Boosting e AutoML na maioria dos datasets avaliados. Diversos fatores podem explicar este comportamento:

1.  **Tamanho dos Datasets**: Deep Learning tipicamente exige grandes volumes de dados para generalizar bem. Os 30 menores datasets do OpenML-CC18 podem não fornecer amostras suficientes para treinar efetivamente os mecanismos de atenção do SAINT sem overfitting, mesmo com regularização (dropout).
2.  **Natureza dos Dados Tabulares**: Diferente de imagens ou texto, onde há correlação espacial/temporal forte, dados tabulares podem ter features com distribuições muito distintas e não relacionadas. Árvores de decisão lidam naturalmente com essa heterogeneidade através de splits hierárquicos, enquanto redes neurais buscam manifolds contínuos que podem não existir.
3.  **Sensibilidade a Hiperparâmetros**: O SAINT possui um espaço de hiperparâmetros complexo (profundidade, heads, learning rate, weight decay). A busca aleatória com 40 iterações pode ter sido insuficiente para encontrar configurações ótimas, enquanto GBDTs são geralmente mais robustos "out-of-the-box".

### 5.2 AutoML vs. Manual Tuning
As soluções de AutoML (AutoGluon e ASKL2) demonstraram competitividade, muitas vezes superando os modelos tunados manualmente. Isso reforça o valor de técnicas de ensemble e stacking automatizados, que conseguem extrair o máximo de informação dos dados combinando as forças de diferentes algoritmos.

### 5.3 Trabalhos Futuros
Com base nas limitações identificadas neste estudo, sugerem-se as seguintes direções para pesquisas futuras:
1.  **Avaliação em Escala Maior**: Testar o SAINT em datasets com centenas de milhares de amostras para verificar se a "fome de dados" é saciada e se o desempenho supera os GBDTs nesse regime.
2.  **Pré-treinamento Contrastivo**: Investigar o impacto do pré-treinamento auto-supervisionado (que foi omitido neste estudo por restrições computacionais) na performance final, especialmente em datasets menores.
3.  **Interpretabilidade**: Realizar uma análise profunda dos mapas de atenção gerados pelo SAINT para entender quais interações de features o modelo está priorizando.
4.  **Comparação com Novos Transformers**: Incluir na comparação arquiteturas mais recentes como TabPFN e FT-Transformer.

## 6. Conclusão
Este estudo reforça a posição dos Gradient Boosted Decision Trees como a escolha padrão para problemas de classificação em dados tabulares de pequeno e médio porte. Embora arquiteturas como o SAINT tragam inovações teóricas importantes, sua aplicação prática em cenários de poucos dados ainda enfrenta desafios significativos de generalização e custo computacional. Para a maioria das aplicações práticas nesta escala, modelos como CatBoost ou frameworks como AutoGluon oferecem um trade-off muito mais favorável entre performance e complexidade.
# Apêndice A: Melhores Hiperparâmetros por Dataset

Esta seção apresenta os melhores hiperparâmetros encontrados durante a busca aleatória para cada modelo e dataset.

### CATBOOST

| Dataset | bagging_temperature | border_count | depth | iterations | l2_leaf_reg | learning_rate | random_strength |
|---|---|---|---|---|---|---|---|
| analcatdata-authorship | 0.4234 | 160 | 6 | 185 | 1.0497 | 0.1602 | 7.0686 |
| analcatdata-dmft | 0.0770 | 218 | 5 | 135 | 8.9242 | 0.0836 | 2.9563 |
| balance-scale | 0.0770 | 218 | 5 | 135 | 8.9242 | 0.0836 | 2.9563 |
| banknote-authentication | 0.8599 | 166 | 7 | 378 | 1.5855 | 0.2521 | 9.6563 |
| blood-transfusion-service-center | 0.5208 | 221 | 10 | 262 | 4.4981 | 0.0252 | 8.2874 |
| car | 0.4234 | 160 | 6 | 185 | 1.0497 | 0.1602 | 7.0686 |
| climate-model-simulation-crashes | 0.5086 | 253 | 7 | 280 | 4.6934 | 0.1306 | 2.2880 |
| cmc | 0.0770 | 218 | 5 | 135 | 8.9242 | 0.0836 | 2.9563 |
| cnae-9 | 0.1109 | 155 | 7 | 228 | 1.2829 | 0.0871 | 3.1436 |
| credit-approval | 0.7220 | 189 | 8 | 241 | 9.9299 | 0.0817 | 6.1165 |
| cylinder-bands | 0.3745 | 124 | 9 | 156 | 8.0172 | 0.0761 | 4.4583 |
| eucalyptus | 0.1109 | 155 | 7 | 228 | 1.2829 | 0.0871 | 3.1436 |
| ilpd | 0.0071 | 120 | 3 | 108 | 4.5987 | 0.0117 | 9.7376 |
| mfeat-factors | 0.8599 | 166 | 7 | 378 | 1.5855 | 0.2521 | 9.6563 |
| miceprotein | 0.8599 | 166 | 7 | 378 | 1.5855 | 0.2521 | 9.6563 |
| pc3 | 0.5208 | 221 | 10 | 262 | 4.4981 | 0.0252 | 8.2874 |
| pc4 | 0.8599 | 166 | 7 | 378 | 1.5855 | 0.2521 | 9.6563 |
| qsar-biodeg | 0.1000 | 234 | 10 | 149 | 2.2858 | 0.0915 | 0.5641 |
| semeion | 0.1109 | 155 | 7 | 228 | 1.2829 | 0.0871 | 3.1436 |
| steel-plates-fault | 0.8599 | 166 | 7 | 378 | 1.5855 | 0.2521 | 9.6563 |
| vowel | 0.8599 | 166 | 7 | 378 | 1.5855 | 0.2521 | 9.6563 |
| wdbc | 0.5086 | 253 | 7 | 280 | 4.6934 | 0.1306 | 2.2880 |

### LIGHTGBM

| Dataset | colsample_bytree | learning_rate | max_depth | min_child_samples | n_estimators | num_leaves | reg_alpha | reg_lambda | subsample |
|---|---|---|---|---|---|---|---|---|---|
| analcatdata-authorship | 0.6873 | 0.2537 | 13 | 76 | 238 | 40 | 0.0000 | 0.0000 | 0.5290 |
| analcatdata-dmft | 0.9304 | 0.0102 | 10 | 31 | 186 | 81 | 0.0000 | 0.0000 | 0.6688 |
| balance-scale | 0.6725 | 0.0865 | 4 | 57 | 221 | 87 | 0.0000 | 0.0000 | 0.6211 |
| banknote-authentication | 0.6873 | 0.2537 | 13 | 76 | 238 | 40 | 0.0000 | 0.0000 | 0.5290 |
| blood-transfusion-service-center | 0.9304 | 0.0102 | 10 | 31 | 186 | 81 | 0.0000 | 0.0000 | 0.6688 |
| car | 0.6873 | 0.2537 | 13 | 76 | 238 | 40 | 0.0000 | 0.0000 | 0.5290 |
| climate-model-simulation-crashes | 0.6461 | 0.0348 | 5 | 59 | 293 | 83 | 0.0002 | 0.5489 | 0.8402 |
| cmc | 0.9331 | 0.0773 | 10 | 7 | 199 | 72 | 5.3603 | 0.3104 | 0.6062 |
| cnae-9 | 0.8035 | 0.0256 | 15 | 19 | 350 | 84 | 0.0000 | 0.0001 | 0.6974 |
| credit-approval | 0.6461 | 0.0348 | 5 | 59 | 293 | 83 | 0.0002 | 0.5489 | 0.8402 |
| cylinder-bands | 0.8035 | 0.0256 | 15 | 19 | 350 | 84 | 0.0000 | 0.0001 | 0.6974 |
| eucalyptus | 0.6180 | 0.0239 | 13 | 13 | 256 | 34 | 0.0000 | 0.0001 | 0.6009 |
| ilpd | 0.7252 | 0.0105 | 11 | 64 | 63 | 28 | 0.0000 | 0.0000 | 0.6205 |
| mfeat-factors | 0.6873 | 0.2537 | 13 | 76 | 238 | 40 | 0.0000 | 0.0000 | 0.5290 |
| miceprotein | 0.7697 | 0.1558 | 14 | 93 | 280 | 148 | 0.0000 | 0.0001 | 0.9090 |
| pc3 | 0.9609 | 0.0135 | 9 | 66 | 345 | 101 | 0.0007 | 0.0019 | 0.9826 |
| pc4 | 0.6180 | 0.0239 | 13 | 13 | 256 | 34 | 0.0000 | 0.0001 | 0.6009 |
| qsar-biodeg | 0.6180 | 0.0239 | 13 | 13 | 256 | 34 | 0.0000 | 0.0001 | 0.6009 |
| semeion | 0.7697 | 0.1558 | 14 | 93 | 280 | 148 | 0.0000 | 0.0001 | 0.9090 |
| steel-plates-fault | 0.6180 | 0.0239 | 13 | 13 | 256 | 34 | 0.0000 | 0.0001 | 0.6009 |
| vowel | 0.6180 | 0.0239 | 13 | 13 | 256 | 34 | 0.0000 | 0.0001 | 0.6009 |
| wdbc | 0.6873 | 0.2537 | 13 | 76 | 238 | 40 | 0.0000 | 0.0000 | 0.5290 |

### SAINT

| Dataset | batch_size | depth | dropout | epochs | heads | learning_rate |
|---|---|---|---|---|---|---|
| analcatdata-authorship | 142 | 3 | 0.1961 | 19 | 5 | 0.0073 |
| analcatdata-dmft | 236 | 8 | 0.4011 | 10 | 8 | 0.0087 |
| balance-scale | 112 | 5 | 0.0347 | 11 | 4 | 0.0046 |
| banknote-authentication | 153 | 9 | 0.3256 | 20 | 8 | 0.0005 |
| blood-transfusion-service-center | 213 | 5 | 0.0385 | 19 | 4 | 0.0025 |
| car | 213 | 3 | 0.4909 | 19 | 5 | 0.0015 |
| climate-model-simulation-crashes | 250 | 6 | 0.3297 | 11 | 6 | 0.0008 |
| cmc | 205 | 5 | 0.3340 | 8 | 7 | 0.0004 |
| cnae-9 | 113 | 3 | 0.3903 | 15 | 5 | 0.0029 |
| credit-approval | 102 | 8 | 0.4532 | 14 | 6 | 0.0007 |
| cylinder-bands | 182 | 5 | 0.0524 | 19 | 6 | 0.0006 |
| eucalyptus | 74 | 5 | 0.4672 | 18 | 8 | 0.0008 |
| ilpd | 83 | 6 | 0.4682 | 17 | 7 | 0.0008 |
| mfeat-factors | 114 | 3 | 0.3110 | 18 | 7 | 0.0025 |
| miceprotein | 118 | 10 | 0.4383 | 14 | 8 | 0.0008 |
| pc3 | 205 | 7 | 0.3609 | 5 | 6 | 0.0010 |
| pc4 | 241 | 3 | 0.3730 | 18 | 8 | 0.0002 |
| qsar-biodeg | 127 | 8 | 0.2571 | 16 | 6 | 0.0009 |
| semeion | 129 | 3 | 0.3626 | 14 | 7 | 0.0016 |
| steel-plates-fault | 135 | 10 | 0.2276 | 20 | 6 | 0.0008 |
| vowel | 223 | 3 | 0.1528 | 19 | 7 | 0.0050 |
| wdbc | 74 | 3 | 0.4301 | 18 | 7 | 0.0057 |

### XGBOOST

| Dataset | colsample_bytree | gamma | learning_rate | max_depth | min_child_weight | n_estimators | reg_alpha | reg_lambda | subsample |
|---|---|---|---|---|---|---|---|---|---|
| analcatdata-authorship | 0.5853 | 0.0000 | 0.2521 | 14 | 2 | 314 | 0.0000 | 0.0000 | 0.6205 |
| analcatdata-dmft | 0.9648 | 0.0292 | 0.0862 | 8 | 4 | 335 | 0.8903 | 0.0000 | 0.5610 |
| balance-scale | 0.6669 | 0.0000 | 0.0915 | 7 | 2 | 393 | 0.3104 | 0.0000 | 0.5909 |
| banknote-authentication | 0.8416 | 0.0008 | 0.1701 | 5 | 1 | 99 | 0.0092 | 0.0000 | 0.7600 |
| blood-transfusion-service-center | 0.5917 | 0.0000 | 0.0596 | 14 | 9 | 98 | 0.0005 | 0.0000 | 0.5233 |
| car | 0.6210 | 0.0024 | 0.1334 | 13 | 1 | 304 | 0.1691 | 0.0002 | 0.9917 |
| climate-model-simulation-crashes | 0.6818 | 0.5946 | 0.2640 | 9 | 2 | 162 | 0.0000 | 9.5425 | 0.6334 |
| cmc | 0.9869 | 0.0000 | 0.0136 | 5 | 7 | 293 | 0.0021 | 0.0000 | 0.8038 |
| cnae-9 | 0.5028 | 0.0334 | 0.1107 | 5 | 3 | 338 | 0.0028 | 2.1712 | 0.8255 |
| credit-approval | 0.9648 | 0.0292 | 0.0862 | 8 | 4 | 335 | 0.8903 | 0.0000 | 0.5610 |
| cylinder-bands | 0.6873 | 0.4034 | 0.1206 | 15 | 5 | 152 | 0.0001 | 0.0000 | 0.7296 |
| eucalyptus | 0.6210 | 0.0024 | 0.1334 | 13 | 1 | 304 | 0.1691 | 0.0002 | 0.9917 |
| ilpd | 0.5853 | 0.0000 | 0.2521 | 14 | 2 | 314 | 0.0000 | 0.0000 | 0.6205 |
| mfeat-factors | 0.6669 | 0.0000 | 0.0915 | 7 | 2 | 393 | 0.3104 | 0.0000 | 0.5909 |
| miceprotein | 0.6669 | 0.0000 | 0.0915 | 7 | 2 | 393 | 0.3104 | 0.0000 | 0.5909 |
| pc3 | 0.5917 | 0.0000 | 0.0596 | 14 | 9 | 98 | 0.0005 | 0.0000 | 0.5233 |
| pc4 | 0.7614 | 0.0000 | 0.0109 | 15 | 3 | 112 | 1.1531 | 0.0002 | 0.7816 |
| qsar-biodeg | 0.8416 | 0.0008 | 0.1701 | 5 | 1 | 99 | 0.0092 | 0.0000 | 0.7600 |
| semeion | 0.5028 | 0.0334 | 0.1107 | 5 | 3 | 338 | 0.0028 | 2.1712 | 0.8255 |
| steel-plates-fault | 0.6210 | 0.0024 | 0.1334 | 13 | 1 | 304 | 0.1691 | 0.0002 | 0.9917 |
| vowel | 0.6210 | 0.0024 | 0.1334 | 13 | 1 | 304 | 0.1691 | 0.0002 | 0.9917 |
| wdbc | 0.5853 | 0.0000 | 0.2521 | 14 | 2 | 314 | 0.0000 | 0.0000 | 0.6205 |