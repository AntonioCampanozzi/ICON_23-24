
'''
|------------------------------------------------------------------------------------------------------------------|
| ESTRATTO DELLA SEZIONE DEL NOTEBOOK COLAB CON TUNING IPERPARAMETRI, TRAINING E VALIDAZIONE DEL MODELLO LINEARSVC |
|------------------------------------------------------------------------------------------------------------------|
'''
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, learning_curve
from sklearn.svm import LinearSVC

lsvc=LinearSVC(dual=False) #dual è impostato a false perché la label da predire e n-aria e non binaria

#fase di tuning

lsvc_params = {
    'C': [0.01, 0.1, 1, 10, 100],
    'loss': ['squared_hinge'],
    'tol': [1e-4, 1e-3, 1e-2, 1e-1, 1],
    'max_iter': [3000, 4000, 5000, 10000, 20000, 50000]
}

grid=GridSearchCV(lsvc, lsvc_params)

grid.fit(X_train,Y_train)

bestParameters = {
   'C': grid.best_params_['C'],
   'loss': grid.best_params_['loss'],
   'tol': grid.best_params_['tol'],
   'max_iter': grid.best_params_['max_iter'],
    }

'''MIGLIORI IPERPARAMETRI PER LINEARSVC:
{
  'C': 100,
  'loss': 'squared_hinge',
  'tol': 0.01,
  'max_iter': 3000}'''

#validazione-cross validation

lsvc=LinearSVC(dual=False, C=100, loss='squared_hinge', tol=0.01, max_iter=3000)

cv = StratifiedKFold(n_splits=5)

scores_acc = cross_validate(lsvc, X, Y, cv=cv, scoring='accuracy',return_train_score=True)
scores_f1_macro=cross_validate(lsvc, X, Y, cv=cv,scoring="f1_macro",return_train_score=True)
scores_prec_macro=cross_validate(lsvc, X, Y, cv=cv,scoring="precision_macro",return_train_score=True)
scores_recall_macro=cross_validate(lsvc, X, Y, cv=cv,scoring="recall_macro",return_train_score=True)
scores_weighted = cross_validate(lsvc, X, Y, cv=cv,scoring="f1_weighted",return_train_score=True)
scores_prec_weighted = cross_validate(lsvc, X, Y, cv=cv,scoring="precision_weighted",return_train_score=True)
scores_recall_weighted = cross_validate(lsvc, X, Y, cv=cv,scoring="recall_weighted",return_train_score=True)

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




# Calcolo  della learning curve di accuracy

train_sizes, train_scores, test_scores = learning_curve(
    lsvc, X_train, Y_train, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

# Calcola la media e la deviazione standard delle score
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Traccia le learning curves
plt.figure()
plt.title("Learning Curves per la Support Vector Machine a kernel lineare")
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

# learning curve errore
train_sizes, train_scores, test_scores = learning_curve(
    lsvc, X_train, Y_train, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

train_scores_error = 1 - train_scores
test_scores_error = 1 - test_scores

# Calcola la media e la deviazione standard dell'errore
train_scores_mean = np.mean(train_scores_error, axis=1)
train_scores_std = np.std(train_scores_error, axis=1)
test_scores_mean = np.mean(test_scores_error, axis=1)
test_scores_std = np.std(test_scores_error, axis=1)

# Traccia le learning curves
plt.figure()
plt.title("Error Learning Curve per la Support Vector Machine a kernerl lineare")
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

lsvc.fit(X_train,Y_train)

print(classification_report(Y_test,lsvc.predict(X_test)))

cm=confusion_matrix(Y_test, lsvc.predict(X_test))

display=ConfusionMatrixDisplay(cm)
display.plot()

#salvataggio modello

with open('linear_svc_model', 'wb') as file:
  pickle.dump(lsvc,file)
file.close()