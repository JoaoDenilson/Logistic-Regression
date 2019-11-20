from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics

#Carregando os dados do data set.
cancer = load_breast_cancer()

#fazendoa separação de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3)

#Fazendo a classificação
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)

y_pred = log_reg.predict(X_test)

print('Acurácia do conjunto de treinamento: {:.3f}'.format(log_reg.score(X_train,y_train)))
print('Acurácia do conjunto de teste: {:.3f}'.format(log_reg.score(X_test,y_test)))

print("Accuracy: {:.3f}".format(metrics.accuracy_score(y_test, y_pred)))