#SECTION 2 - Pré-processamento de dados
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px #Gráficos interativos

#Visualização dos dados
df.head()
df.tail()
df.describe()
df.info()

df[df['coluna'] >= 0]
np.unique(df['coluna'], return_counts=True) #Contagem de valores únicos
sns.countplot(x=df['coluna']); #O símbolo ; tira os Warnings
plt.hist(x=df['coluna']) #Insights iniciais

grafico = px.treemap(df, path=['coluna', 'coluna2'])
grafico.show()

grafico2 = px.parallel_categories(df, dimensions=['coluna', 'coluna2'])
grafico2.show()

grafico3 = px.scatter_matrix(df, dimensions ['coluna', 'coluna2', 'coluna3'], color='default')
grafico3.show()


#Valores inconsistentes
df.loc[filtro] #loc Opcional
df.loc[df['coluna'] < 0]
df = df.drop('coluna', axis=1) #axis - 0: linha 1: coluna

df.index #índices/linhas
df[df['coluna'] < 0].index #índices desse filtro
df = df.drop(df[df['coluna'] < 0]. index) #dropar índices desse filtro

média_valor = df['coluna'].mean()
df.loc[df['coluna'] < 0, 'coluna'] = média_valor #preencher com média


#Valores faltantes
df.isnull.sum()
df.loc[pd.isnull(df['coluna'])] #localizar
df['coluna'].fillna(df['coluna'].mean(), inplace=True) #preencher + inplace=True altera base de dados
df.loc[df['id'].isin([3,4,23])] #localizar


#Divisão entre previsores e classe
type(df) #pandas.core.frame.Dataframe ou numpy.ndarray
X_df = df.iloc[:, 1:4].values # [linhas, colunas]; 1:4 vai do 1 ao 3 = intervalo; values = converte para matriz (numpy)
y_df = df.iloc[:, 4].values


#Escalonamento dos valores
X_df[:, 0].min(), X_df[:, 1].min() ...
X_df[:, 0].max()
#Padronização (Standardisation)   x = (x - x[média])/(x[desvio])   Indicado na presença de outliers
#Normalização (Normalization)   x = (x - x[mín])/(x[máx] - x[mín])
from sklearn.preprocessing import StandardScaler
scaler_df = StandardScaler()
X_df = scaler_df.fir_transform(X_df)


#Tratamento de atributos categóricos
    #Label Enconder (strings para números)
from sklearn.preprocessing import LabelEnconder
le_teste1 = LabelEnconder()
teste = le_teste1.fir_transform(X_df[:,1])
le_teste2 = LabelEnconder()
teste = le_teste2.fir_transform(X_df[:,4]) # [...]

    #OneHotEnconder (codificação)
#ColunaCarro: 1(Gol), 2(Pálio), 3(Uno) --- transformar em 3 Colunas --- Col1(Gol): 1,0,0 + Col2(Pálio): 0,1,0 + Col3(Uno): 0,0,1
len(np.unique(df['coluna'])) #quantidade de categorias
from sklearn.preprocessing import OneHotEnconder
from sklearn.compose import ColumnTransformer
ohe_teste = ColumnTransformer(transformers=[('OneHot', OneHotEnconder(), [1,2,3])], remainder='passthrough') #[1,2,3] = colunas que aplicou LabelEnconder
X_df = ohe_teste.fir_transform(X_df).toarray()
X_df.shape


#Escalonamento dos valores (numéricos)
from sklearn.preprocessing import StandardScaler
ssca = StandardScaler()
X_df = ssca.fit_transform(X_df)


#Divisão Treinamento e Teste
from sklearn.model_selection import train_test_split
X_df_treino, X_df_teste, y_df_treino, y_df_teste = train_test_split(X_df, y_df, test_size=0.25, random_state=0) #maior base de dados = menor test_size

X_df_treino.shape
y_df_treino.shape


#Salvar variáveis
import pickle
with open('df.pkl', mode='wb') as f:
    pickle.dump ([X_df_treino, y_df_treino, X_df_teste, y_df_teste], f)
