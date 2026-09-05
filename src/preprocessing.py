# -*- coding: utf-8 -*-
"""

Arquivo inicial gerado pelo Colab a partir do 02_Engenharia_dados.ipynb

ChangeLog:
    2026-09-04: Alterar para ler o chamados.csv do diretório data/raw e escrever as
                saídas em data/preprocessed (Roberto F Batista)

**1. Carregando o conjunto de dados**
"""

import pandas as pd

df_chamados = pd.read_csv("../data/raw/chamados.csv")

df_original = df_chamados.copy()

"""**1.2 Dimensões do dataset**"""

df_chamados.shape

"""**1.3 Visualização dos dados**"""

df_chamados.head()

"""**2. Limpeza dos dados**

**2.1 Remoção das colunas sem dados**
"""

percentual_nulos = df_chamados.isna().mean()

colunas_vazias = percentual_nulos[percentual_nulos >= 0.99].index

print("Colunas com 99% ou mais de valores ausentes:")
print(colunas_vazias)

df_chamados = df_chamados.drop(columns=colunas_vazias)

print("\nColunas removidas:")
print(colunas_vazias)

"""**2.2 - Remoção de Colunas sem Variabilidade**"""

colunas_sem_variacao = df_chamados.columns[df_chamados.nunique() <= 1]

print("Colunas sem variabilidade:")
print(colunas_sem_variacao)

df_chamados = df_chamados.drop(columns=colunas_sem_variacao)

"""**2.3 Verificação de valores ausentes**"""

df_chamados.isna().sum().sort_values(ascending=False)

"""**2.4 Identificação das variáveis**"""

df_chamados.columns

"""**2.5 Verificar valores duplicados**"""

df_chamados.duplicated().sum()

"""**2.6 Verificação de consistência das categorias**"""

colunas_categoricas = [
    'department',
    'city_x',
    'state_x',
    'device_type',
    'os',
    'browser',
    'network_type',
    'priority',
    'status',
    'assigned_team'
]

print(colunas_categoricas)

colunas_categoricas = [
    'department',
    'city_x',
    'state_x',
    'device_type',
    'os',
    'browser',
    'network_type',
    'priority',
    'status',
    'assigned_team'
]
for coluna in colunas_categoricas:
    print(f"\n{coluna}:")
    print(df_chamados[coluna].value_counts())

"""**2.7 Transformação das variáveis categóricas**"""

for coluna in colunas_categoricas:
    print(f"\n{coluna}:")
    print(df_chamados[coluna].unique())

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

dados_categoricos = encoder.fit_transform(
    df_chamados[colunas_categoricas]
)

dados_categoricos

colunas_codificadas = encoder.get_feature_names_out(colunas_categoricas)

print(colunas_codificadas)

df_categoricas = pd.DataFrame(
    dados_categoricos,
    columns=colunas_codificadas
)

df_categoricas.head()

df_chamados.columns

"""**2.8 Preparação das variáveis numéricas**"""

colunas_numericas = [
    'historical_tickets',
    'sla_hours',
    'resolution_time_hours',
    'image_width',
    'image_height',
    'sentiment_score',
    'num_words',
    'num_images_x',
    'latitude',
    'longitude'
]

print(colunas_numericas)

df_chamados[colunas_numericas].isna().sum()

df_chamados[colunas_numericas].describe()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

dados_numericos = scaler.fit_transform(
    df_chamados[colunas_numericas]
)

df_numericas = pd.DataFrame(
    dados_numericos,
    columns=colunas_numericas
)

df_numericas.head()

dados_processados = pd.concat(
    [df_numericas, df_categoricas],
    axis=1
)

dados_processados.head()

"""### Separação das variáveis de entrada e alvo"""

X = dados_processados
y = df_chamados['target_category']

print("Dimensão das entradas (X):", X.shape)
print("Dimensão do alvo (y):", y.shape)

"""**3.0 Partição de dados**

**3.1 Divisão em treino e teste**

(a validação será separada na fase de treino)
"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,
    random_state=42,
    stratify=y
)
#

print('Treino', X_train.shape, y_train.shape)
print('Teste', X_test.shape, y_test.shape)

X_train.to_csv("../data/preprocessed/X_train.csv", index=False)
y_train.to_csv("../data/preprocessed/y_train.csv", index=False)
X_test.to_csv("../data/preprocessed/X_test.csv", index=False)
y_test.to_csv("../data/preprocessed/y_test.csv", index=False)
