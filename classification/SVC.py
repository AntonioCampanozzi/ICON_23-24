
'''
|------------------------------------------------------------------------------------------------------------|
| ESTRATTO DELLA SEZIONE DEL NOTEBOOK COLAB CON TUNING IPERPARAMETRI, TRAINING E VALIDAZIONE DEL MODELLO SVC |
|------------------------------------------------------------------------------------------------------------|
'''
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, learning_curve
from sklearn.svm import SVC

svc=SVC()

#tuning

svc_params = {
    'C': [0.1, 1, 10, 100, 1000],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], #non consideriamo precomputed perché prende in input una matrice del tipo (n-_samples, n_samples) al posto di (n_features,n_samples)
    'gamma': ['scale', 'auto'],
    'degree': [0, 1, 2, 3, 4, 5, 6]
}

grid=GridSearchCV(svc, svc_params)

grid.fit(X_train,Y_train)

bestParameters = {
   'C': grid.best_params_['C'],
   'kernels': grid.best_params_['kernel'],
   'gamma': grid.best_params_['gamma'],
   'degree': grid.best_params_['degree'],
    }

'''MIGLIORI IPERPARAETRI PER SVC:
{
  'C': 1,
  'kernels': 'poly',
  'gamma': 'scale',
  'degree': 2}'''

#validazione-cross validation

svc=SVC(C=1, kernel='poly', gamma='scale', degree=2)

cv = StratifiedKFold(n_splits=5)

scores_acc = cross_validate(svc, X, Y, cv=cv, scoring='accuracy',return_train_score=True)
scores_f1_macro=cross_validate(svc, X, Y, cv=cv,scoring="f1_macro",return_train_score=True)
scores_prec_macro=cross_validate(svc, X, Y, cv=cv,scoring="precision_macro",return_train_score=True)
scores_recall_macro=cross_validate(svc, X, Y, cv=cv,scoring="recall_macro",return_train_score=True)
scores_weighted = cross_validate(svc, X, Y, cv=cv,scoring="f1_weighted",return_train_score=True)
scores_prec_weighted = cross_validate(svc, X, Y, cv=cv,scoring="precision_weighted",return_train_score=True)
scores_recall_weighted = cross_validate(svc, X, Y, cv=cv,scoring="recall_weighted",return_train_score=True)

print(f'l\'accuratezza media cross_validata è: {np.mean(scores_acc.get("train_score"))} con una deviazione standard di: {np.std(scores_acc.get("train_score"))} e una varianza pari a: {np.var(scores_acc.get("train_score"))}per il train set')
print(f'l\'accuratezza media cross_validata è: {np.mean(scores_acc.get("test_score"))} con una deviazione standard di: {np.std(scores_acc.get("test_score"))} e una varianza pari a: {np.var(scores_acc.get("test_score"))} per il test set\n')

print(f'l\'f1_score_macro medio cross_validato è: {np.mean(scores_f1_macro.get("train_score"))} con una deviazione standard di: {np.std(scores_f1_macro.get("train_score"))} e una varianza pari a: {np.var(scores_f1_macro.get("train_score"))} per il train set')
print(f'l\'f1_score_macro medio cross_validato è: {np.mean(scores_f1_macro.get("test_score"))} con una deviazione standard di: {np.std(scores_f1_macro.get("test_score"))} e una varianza pari a: {np.var(scores_f1_macro.get("test_score"))} per il test set\n')

print(f'la precision_macro media cross_validata è: {np.mean(scores_prec_macro.get("train_score"))} con una deviazione standard di: {np.std(scores_prec_macro.get("train_score"))} e una varianza pari a: {np.var(scores_prec_macro.get("train_score"))} per il train set')
print(f'la precision_macro media cross_validata è: {np.mean(scores_prec_macro.get("test_score"))} con una deviazione standard di: {np.std(scores_prec_macro.get("test_score"))} e una varianza pari a: {np.var(scores_prec_macro.get("test_score"))} per il test set\n')

print(f'la recall_macro media cross_validata è: {np.mean(scores_recall_macro.get("train_score"))} con una deviazione standard di: {np.std(scores_recall_macro.get("train_score"))} e una varianza pari a: {np.var(scores_recall_macro.get("train_score"))} per il train set')
print(f'la recall_macro media cross_validata è: {np.mean(scores_recall_macro.get("test_score"))} con una deviazione standard di: {np.std(scores_recall_macro.get("test_score"))} e una varianza pari a: {np.var(scores_recall_macro.get("test_score"))} per il test set\n')

print(f'l\'f1_score_weighted medio cross_validato è: {np.mean(scores_weighted.get("train_score"))} con una deviazione standard di: {np.std(scores_weighted.get("train_score"))} e una varianza pari a: {np.var(scores_weighted.get("train_score"))} per il train set')
print(f'l\'f1_score_weighted medio cross_validato è: {np.mean(scores_weighted.get("test_score"))} con una deviazione standard di: {np.std(scores_weighted.get("test_score"))} e una varianza pari a: {np.var(scores_weighted.get("test_score"))} per il test set\n')

print(f'la precision_weighted media cross_validata è: {np.mean(scores_prec_weighted.get("train_score"))} con una deviazione standard di: {np.std(scores_prec_weighted.get("train_score"))} e una varianza pari a: {np.var(scores_prec_weighted.get("train_score"))} per il train set')
print(f'la precision_weighted media cross_validata è: {np.mean(scores_prec_weighted.get("test_score"))} con una deviazione standard di: {np.std(scores_prec_weighted.get("test_score"))} e una varianza pari a: {np.var(scores_prec_weighted.get("test_score"))} per il test set\n')

print(f'la recall_weighted media cross_validata è: {np.mean(scores_recall_weighted.get("train_score"))} con una deviazione standard di: {np.std(scores_recall_weighted.get("train_score"))} e una varianza pari a: {np.var(scores_recall_weighted.get("train_score"))} per il train set')
print(f'la recall_weighted media cross_validata è: {np.mean(scores_recall_weighted.get("test_score"))} con una deviazione standard di: {np.std(scores_recall_weighted.get("test_score"))} e una varianza pari a: {np.var(scores_recall_weighted.get("test_score"))} per il test set')

# learning curve accuracy
train_sizes, train_scores, test_scores = learning_curve(
    svc, X_train, Y_train, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

# Calcola la media e la deviazione standard delle score
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Traccia le learning curves
plt.figure()
plt.title("Learning Curves per la Support Vector Machine")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Test score")

plt.legend(loc="best")
plt.show()

#learning curve errore

train_sizes, train_scores, test_scores = learning_curve(
    svc, X_train, Y_train, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)
train_scores_error = 1 - train_scores
test_scores_error = 1 - test_scores

# Calcola la media e la deviazione standard dell'errore
train_scores_mean = np.mean(train_scores_error, axis=1)
train_scores_std = np.std(train_scores_error, axis=1)
test_scores_mean = np.mean(test_scores_error, axis=1)
test_scores_std = np.std(test_scores_error, axis=1)

#Traccia le learing curves

plt.figure()
plt.title("Error Learning Curve per la Support Vector Machine")
plt.xlabel("Training examples")
plt.ylabel("Errore")
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training error")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Test error")

plt.legend(loc="best")
plt.show()

#validazione-singolo sample

svc.fit(X_train, Y_train)

print(classification_report(Y_test,svc.predict(X_test)))

cm=confusion_matrix(Y_test, svc.predict(X_test))

display=ConfusionMatrixDisplay(cm)
display.plot()

#salvataggio del modello

with open('svc_model', 'wb') as file:
  pickle.dump(svc, file)
file.close()


