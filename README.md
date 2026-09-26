# Detecção de fraudes em cartão de crédito
O objetivo desse exercício é utilizar técnicas de aprendizagem estatística para treinar modelos que detectem fraudes associadas com cartão de crédito. Nesse caso, iremos utilizar alguns modelos clássicos de aprendizagem estatística com o objetivo de que ao fim obtenhamos um modelo capaz de dizer se uma determinada operação com cartão é ou não fraudulenta. O conjunto de dados utilizado está disponível em formato CSV no URL https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv .

## Modelos utilizados
Para realizar a detecção de fraudes, utilizamos três principais modelos, sendo eles a regressão logística, o Random Forest Classifier e o XGBoost Classifier. O objetivo dos três modelos é receber um conjunto de dados relacionados com as operações de cartão de crédito e trazer como resposta se uma nova transação, que não faz parte dos dados de treino, é ou não uma operação fraudulenta. Nesse caso, há duas possibilidades apenas de resposta para se uma operação é fraudulenta: Sim, representado por y=1 e não, representado por y-0. Desses três modelos, a regressão logística se baseia em realizar um tipo de regressão em uma função logística, o que resulta em uma curva contínua semelhante à um S. Nesse caso, uma determinada observação é associada com uma certa estimativa para y de zero à um. Definindo um valor de corte (e que pode ser alterado, assim como é feito mais adiante no arquivo), decidimos que y=Sim (ou y=1) se a estimativa é superior ao corte, ou que y=Não (ou y=0) caso a estimativa seja inferior ao corte.

Para os outros dois modelos classificadores, a abordagem matemática é ligeiramente diferente, visto que tratam-se de modelos que implementam árvores de decisão. Em essência, ambos modelos criam árvores de decisão e classificam uma nova observação não presente no conjunto de treinamento percorrendo essa árvore de decisão. O algoritmo Random Forest tem por objetivo criar árvores sendo que em cada nó introduzimos limitações nos conjuntos de preditores disponíveis para a escolha. O algoritmo então seleciona qual dos preditores possíveis melhor divide a árvore de decisão de maneira que ela se adapte melhor ao conjunto de dados. Essa limitação no conjunto de preditores é randômica e daí que vem o nome do algoritmo. O algoritmo então repete de forma a aumentar a diversidade de árvores randômicas.

O XGBoost Classifier é muito semelhante em espírito, mas cada vez que o algoritmo finaliza uma árvore, então a próxima utilizará os erros cometidos pela árvore anterior como baliza, o que permite com que as próximas árvores se adequem melhor ao conjunto de dados.

Os três algoritmos acima são algoritmos clássicos de aprendizagem estatística e são apenas uma pequena fração de algoritmos de decisão/classificadores. Sendo assim, é possível realizar o mesmo estudo utilizando outros modelos.

## Coleta de dados e análise preliminar
Os dados acima foram coletados pelo link acima utilizando diretamente a função ```pandas.read_csv```, que retorna um DataFrame. Uma análise preliminar utilizando ```.head()``` nos mostra a presença de 28 colunas tratadas para manter a discrição dos dados, além de uma coluna Time, Amount (que representa o valor associado com a operação) e Class, que diz se uma operação é fraudulenta (valor 1) ou não (valor 0). Após isso podemos contar quantos elementos pertencem à cada classe e obter um total de 99.83% de não fraudes e 0.17% de fraudes (aproximadamente). Podemos perceber então que há muito mais dados de não fraude do que de fraudes. Utilizando um ```.describe()```, obtemos que esse arquivo possui 284807 dados e sendo assim há um total de 492 fraudes.

No entanto, no processo de treinamento há uma etapa crucial: a separação dos dados em parte de treinamento e de teste. Isso serve para, com um conjunto de dados, estimarmos qual o comportamento dos dados obtidos em um outro conjunto de dados que não foi utilizado para treiná-lo (essa etapa é conhecida como cross-validation, ou validação cruzada, possuindo diversas técnicas e maneiras de realizá-la, mas aqui utilizaremos apenas o conjunto de teste). No entanto, se separarmos o conjunto de dados há um problema muito grande: existe a possibilidade de o conjunto de treinamento possuir apenas dados sem fraudes. Se isso por ventura ocorrer, então o nosso modelo se torna incapaz de detectar qualquer tipo de fraude. Pior do que isso, testes de hipótese vão nos dizer que o modelo se adaptou bem ao conjunto de dados, visto que uma estratégia de escolher "não fraude" para os dados fornece um resultado de 99.83% de acertos nesse conjunto de dados.

Para que o desbalanceamento no conjunto de dados não seja um problema, ao utilizar o comando de ```train_test_split``` passamos também um argumento ```stratify=df["Class"]``` que faz com que os dados sejam divididos de maneira à sempre incluir fraudes, o que faz com que nosso modelo seja forçado à procurar entender como as quantidades se relacionam com operações fraudulentas independentemente da divisão realizada.

O restante da preparação dos dados segue como de costume: normalizando os dados utilizando a função ```StandardScaler```.

Há outras duas formas de balancear os dados, incluindo novos dados na classe minoritária (gerados seguindo as regras dos dados), técnica chamada de Oversampling, ou diminuindo da classe majoritária, técnica conhecida como Undersampling

## Avaliação dos modelos
Uma vez que possuímos os dados pré tratados, podemos prosseguir com a análise. Em geral, o pacote ```sklearn``` utiliza a mesma sequência: criamos uma nova variável ```MODEL``` que é dada pela função associada com o modelo desejado como por exemplo, 
```python
model=LogisticRegression()
```
e depois utilizamos este objeto ```model``` para realizar o ajuste aos dados utilizando (e esta função é independente do modelo escolhido)
```python
model.fit(x_treino, y_treino)
```
o que retorna o modelo ajustado para o nosso conjunto de dados. Com isso podemos realizar previsões utilizando o comando (que independe do modelo utilizado)
```python
y_predito=model.predict(x_teste)
```
e por fim trazer um pequeno relatório utilizando 
```python
classification_report(y_test,y_pred)
```
que compara o y de teste com os valores preditos. Aqui utilizaremos três quantidades para classificar a qualidade de nosso modelo:

1. Precisão: TP/TP+FP - é a proporção de positivos (nesse caso, leia-se fraude) verdadeiros que o modelo acertou em comparação com todos os positivos obtidos (incluindo aqueles em que o modelo errou).

2. Recall: TP/TP+FN - é a proporção de positivos verdadeiros que o modelo acertou em comparação com todos aqueles que ele deveria classificar como positivos (mas que podem ter sido classificados falsamente como negativos).

3. F1-score: Média harmônica entre a precisão e o recall.

Note que, como queremos descobrir fraudes, gostaríamos do modelo com maior recall possível, ainda que isso diminua a precisão do modelo.

Um outro indicador interessante para nossos modelos é a chamada curva ROC (Receiver Operating Characteristic) é uma curva que descreve a qualidade do modelo colocando no eixo X a taxa de falsos positivos (ou seja, as operações que nosso modelo detecta como fraude quando não se trata de fraude) e o eixo Y a taxa de positivos verdadeiros. Quanto mais próximo do canto superior esquerdo está a curva, melhor é nosso modelo visto que isso quer dizer que há menos falsos positivos e mais positivos verdadeiros. Também temos a chamada AUC, ou Area Under Curve, um parâmetro no intervalo [0,1] em que quanto mais próximo de um, maior a área debaixo da curva (e portanto mais próximo do canto superior esquerdo está a curva).

Abaixo incluímos uma tabela para cada modelo, incluindo a precisão, o recall, o f1-score e AUC associados com a classe de fraude.

|| Regressão Logística | Random Forest | XGBoost |
|--| -------- | -------- | -------- |
| Precisão | 0.86   | 0.74   | 0.94   |
| Recall | 0.64   | 0.80   | 0.78   |
| F1-Score | 0.73   | 0.77   | 0.85   |
| AUC | 0.928   | 0.972   | 0.969   |

Com a tabela acima, podemos perceber que os modelos de Random Forest e XGBoost são mais adequados. O Random Forest possui um recall maior, ainda que sua precisão seja inferior. Por outro lado XGBoost possui um nível levemente inferior no recall, mas possui uma precisão muito maior, o que faz com que seu F1-score seja superior. As áreas de baixo das curvas também são próximas, o que indicam que o melhor equilíbrio obtido é com o modelo de XGBoost, ainda que ele tenha um recall um pouco inferior. Caso aceite-se diminuir significativamente a precisão para aumentar o recall, então Random forest é o modelo adequado.

Por fim, o uso do teste SHAP para os modelos mostrou que 
1. Para o modelo de Regressão Logística, as três variáveis mais importantes são
2. Para o modelo de Random Forest, as três variáveis mais importantes são
3. Para o modelo de XGBoost, as três variáveis mais importantes são V4, V14 e V12.
