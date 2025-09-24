''' SECTION 3 - Aprendizagem Bayesiana
Vantagem: rápido, simples para interpretar, trabalha com altas dimensões, boa previsão em base de dados pequena
Desvantagem: considera atributos independentes (nem sempre são)

- Divisão entre previsores e classe
- Tratamento (Label Enconder) de categorias (sem escalonamento) '''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB #Problemas mais genéricos
naive_df = GaussianNB()
naive_df.fit(X_df_treino, y_df_treino)
previsao = naive_df.predict([0,0,1,2], [2,0,0,0]) #Exemplos de casos
print(previsao) #array(['resultado1'],['resultado2'])

#Códigos adicionais
naive_df.classes_ #Mostra as classes
naive_df.class_count_ #Contagem de cada classe
naive_df.class_prior_

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy_score(y_df_teste, previsao)
confusion_matrix(y_df_teste, previsao)

from yellowbrick.classifier import ConfusionMatrix #Gráfico
cm = ConfusionMatrix(naive_df)
cm.fit(X_df_treino, y_df_treino)
cm.score(X_df_teste, y_df_teste)
print(classification_report(y_df_teste, previsao))

#Precision = identificou corretamente e acertou
#Recall = identificou corretamente

''' SECTION 4 - Aprendizagem por Árvores de Decisão
Não utiliza todos os atributos, depende da decisão da árvore.
Os atributos "acima" na árvore são considerados mais importantes = Identificação com a fórmula de Entropia

- Entropy(S) = Σ-pi*log2pi 
Ex: (Alto 6/14, Moderado 3/14, Baixo 5/14) = -6/14*log(6/14;2) - 3/14*log(3/14;2) - 5/14*log(5/14;2) = 1,53

- Gain(S,A) = Entropy(S) - Σ (Sv)/S * Entropy(Sv) = Maior - atributo mais importante
- Divisão entre previsores e classe
- Tratamento (Label Enconder) de categorias (sem escalonamento) '''