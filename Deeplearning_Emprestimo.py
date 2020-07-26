import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# IMPORTANDO O DATASET
dataset = pd.read_csv('emprestimo.csv')
X = dataset.iloc[:, 1:12].values
y = dataset.iloc[:,12 ].values

# TRATANDO OS DADOS DE ENTRADA E SAIDA DA REDE
# Encoding (Colocado categorias em códigos numéricos) para os dados de entrada
labelencoder_X_0 = LabelEncoder()
X[:, 0] = labelencoder_X_0.fit_transform(X[:, 0])
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])
labelencoder_X_4 = LabelEncoder()
X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4])

onehotencoder = ColumnTransformer([("Dependents", OneHotEncoder(), [2])],
                                  remainder='passthrough',sparse_threshold=0.3, n_jobs= -1)
X = onehotencoder.fit_transform(X)
onehotencoder1 = ColumnTransformer([("Dependents", OneHotEncoder(), [13])],
                                   remainder='passthrough', sparse_threshold=0.3, n_jobs= -1)
X = onehotencoder1.fit_transform(X)

# Saída da rede (1 se sim e 0 se o banco não conceder o crédito)
labelencoder_y = LabelEncoder()
y= labelencoder_y.fit_transform(y)

# Dividindo o dataset em dados de treino e de teste.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Normalização dos dados.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#  EXECUÇÃO DA REDE PARA OBTENÇÃO DE UM MODELO(APREDIZAGEM) PARA PREDIÇÕES
classifier = Sequential()
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu', input_dim = 16))
#classifier.add(Dropout(p = 0.1))
# Adding the second hidden layer
classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 2)

#PREDIÇÃO COM OS DADOS DO DATASET
# Predições a partir dos dados de teste do dataset
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# FAZENDO UMA PREDIÇÃO SIMPLES, DE UM CLIENTE FICTÍCIO, APÓS O TREINAMENTO DO REDE.

#Variáveis obtidas no open banking e as cadastradas pelo cliente em nossa plataforma.
'''
Loan_ID: ID exclusivo do empréstimo
Gênero: Masculino Feminino
Casado: Requerente casado (S / N)
Dependentes: Número de dependentes
Educação: Educação de candidatos (graduação / pós-graduação)
Autônomo: Autônomo (S / N)
CandidatIncome: Renda do candidato
CoapplicantIncome: Renda do coapplicante
Montante do empréstimo: Montante do empréstimo em milhares
Loan_Amount_Term: Prazo do empréstimo em meses
Credit_History: o histórico de crédito atende às diretrizes
Property_Area: Urbano / Semi-Urbano / Rural

A IA RETORNA:
Status do empréstimo: Empréstimo aprovado (S / N) - de acordo com o perfil do cliente.
'''
Client_01 = classifier.predict(sc.transform(np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                                    1, 1, 0, 0, 4583, 1508.0, 128, 360, 1]])))
Client_01 = (Client_01 >0.5)
print(Client_01)

Client_02 = classifier.predict(sc.transform(np.array([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                                                      1, 1, 0, 1, 3000, 0.0, 66, 360, 1]])))
Client_02 = (Client_02 >0.5)
print(Client_02)

#PLOT
#  Confusion Matrix Para acurácia da rede
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
print(cm)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax)
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Sim', 'Não']); ax.yaxis.set_ticklabels(['Não', 'Sim'])

#MELHORIAS PARA AUMENTAR A ACURÁCIA DA REDE

#implementação da rede com Keras e novas "configurações"
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifieres():
    classifier = Sequential()
    classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifieres = KerasClassifier(build_fn = build_classifieres, batch_size = 10, epochs = 2)
accuracies = cross_val_score(estimator = classifieres, X = X_train, y = y_train, cv = 10, n_jobs = -1)

mean = accuracies.mean()
variance = accuracies.std()

print(mean)
print(variance)

# Regualrização do Dropout para reduzir overfitting, caso seja necessário
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classif(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classif = KerasClassifier(build_fn = build_classif)
parameters = {'batch_size': [25, 32],
              'epochs': [2, 2],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classif,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print(best_accuracy)