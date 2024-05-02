# manipulação de dataframes (dados tabulares)
import pandas as pd

# separação de treino e teste
from sklearn.model_selection import train_test_split

# preprocessamento
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Modelos de IA
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# preparando os dados para o treinamento 

df = pd.read_csv('tic_tac_toe_dataset.csv')

# Separando o dataset de treino e teste
df_treino, df_teste = train_test_split(df, test_size=0.3, random_state=42)

# Copiando a classificação (qm ganhou), 
# pois iremos utilizar depois para comparar os resultados do modelo e ver se ta certo
df_treino_target = df_treino['class'].copy()
df_teste_target = df_teste['class'].copy()

# Removendo a classificação (qm ganhou)
df_treino = df_treino.drop(columns=['class'])
df_teste = df_teste.drop(columns=['class'])

# fazendo mecanismo de preprocessamento
preproc_completo = ColumnTransformer([
    ('numericos', 'passthrough', []),
    ('categoricos', OneHotEncoder(), ['atr1', 'atr2', 'atr3', 'atr4', 'atr5', 'atr6', 'atr7', 'atr8', 'atr9']),
],
    sparse_threshold=0)

# pre-processamento (apenas dos dados categóricos) do conjunto de treino
X_treino = preproc_completo.fit_transform(df_treino)
Y_treino = df_treino_target.values

# pre-processamento (apenas dos dados categóricos) do conjunto de teste
X_teste = preproc_completo.transform(df_teste)
Y_teste = df_teste_target.values

# carregando os modelos 
param_grid = [{
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'min_samples_leaf': [2, 3, 4, 5, 6, 7, 8, 9]
}]

arvore = DecisionTreeClassifier()

grid_search = GridSearchCV(arvore, param_grid)
grid_search.fit(X_treino, Y_treino)

arvore = grid_search.best_estimator_

mlp_clf = MLPClassifier(hidden_layer_sizes=(30, 15), max_iter=500)
knn_clf = KNeighborsClassifier(5)

print("Estimador: Acur. - Prec. - Rec.  - F1")
print("-------------------------------------")
for estimador in (mlp_clf, knn_clf, arvore):
    estimador.fit(X_treino, Y_treino)
    previsoes = estimador.predict(X_teste)
    print("%.25s: %2.3f - %2.3f - %2.3f - %2.3f"
          % (estimador.__class__.__name__.ljust(25, '.'),
             accuracy_score(Y_teste, previsoes),
             precision_score(Y_teste, previsoes, average='macro'),
             recall_score(Y_teste, previsoes, average='macro'),
             f1_score(Y_teste, previsoes, average='macro'),
             ))


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)


@app.get("/")
def root():
    return "Hello World"


@app.get("/verifyMLP/{board}")
def verify_game(board):
    print(board)
    real_board = board.split(',')
    print(real_board)
    end_game = pd.DataFrame([real_board],
                            columns=['atr1', 'atr2', 'atr3', 'atr4', 'atr5', 'atr6', 'atr7', 'atr8', 'atr9'])
    valid_game = preproc_completo.transform(end_game)
    result = mlp_clf.predict(valid_game)
    return {"result": result[0]}


@app.get("/verifyKNN/{board}")
def verify_game(board):
    real_board = board.split(',')
    end_game = pd.DataFrame([real_board],
                            columns=['atr1', 'atr2', 'atr3', 'atr4', 'atr5', 'atr6', 'atr7', 'atr8', 'atr9'])
    valid_game = preproc_completo.transform(end_game)
    result = knn_clf.predict(valid_game)
    return {"result": result[0]}


@app.get("/verifyTree/{board}")
def verify_game(board):
    real_board = board.split(',')
    end_game = pd.DataFrame([real_board],
                            columns=['atr1', 'atr2', 'atr3', 'atr4', 'atr5', 'atr6', 'atr7', 'atr8', 'atr9'])
    valid_game = preproc_completo.transform(end_game)
    result = arvore.predict(valid_game)
    return {"result": result[0]}
