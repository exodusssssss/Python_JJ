#Exercice 1
#a)
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data    
y = iris.target

#b)
#Il y a 4 variables explicatives et une variable catégorielle


#c)
from sklearn.linear_model import LogisticRegression
import numpy as np
model_log = LogisticRegression(C=np.inf, max_iter=200)
model_log.fit(X, y)

#d)
from sklearn.model_selection import cross_val_score
score_iris_log = cross_val_score(model_log, X, y)
print(score_iris_log)
print(score_iris_log.mean())
#Le score correspond à l'accuracy du modèle, le taux de bonne classification

#e)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
scaler = StandardScaler()
X_normalise = scaler.fit_transform(X)
model_log_L2 = LogisticRegressionCV(penalty='l2', Cs=10, max_iter=200)
model_log_L2.fit(X_normalise, y)
score_iris_log_L2 = cross_val_score(model_log_L2, X_normalise, y)
print(score_iris_log_L2)
print(score_iris_log_L2.mean())

#f)
model_log_L1 = LogisticRegressionCV(penalty='l1', solver='saga', Cs=10, max_iter=200)
model_log_L1.fit(X_normalise, y)
score_iris_log_L1 = cross_val_score(model_log_L1, X_normalise, y)
print(score_iris_log_L1)
print(score_iris_log_L1.mean())

print(model_log_L1.coef_)

#g)
# Pour la matrice des coefficients, on peut dire que certaines variables n'influencent pas du tout 
# certaines classes (les cellules qui contiennent 0). Et à l'inverse, on peut dire que certaines classes sont fortement
# influencés par des variables comme la classe 3 par exemple.

# Exercice 2
#a)
from sklearn import datasets
titanic = datasets.fetch_openml(data_id=40945, as_frame=True, parser='auto')

#b)

#c)
