
'''
|----------------------------------------------------------------------------------------------------------------------|
| ESTRATTO DELLA SEZIONE DEL NOTEBOOK COLAB CON TUNING IPERPARAMETRI, TRAINING E VALIDAZIONE DEL MODELLO RANDOM FOREST |
|----------------------------------------------------------------------------------------------------------------------|
'''
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve, cross_validate, StratifiedKFold, GridSearchCV

rf=RandomForestClassifier()
#tuning

RandomForestHyperparameters = {
   'criterion': ['gini', 'entropy', 'log_loss'],
   'n_estimators': [50,100, 200],
   'max_depth': [None, 5, 10],
   'min_samples_split': [2, 5, 10, 20],
   'min_samples_leaf': [1, 2, 5, 10, 20]}

grid = GridSearchCV(rf, RandomForestHyperparameters)
grid.fit(X_train,Y_train)
bestParameters = {
   'criterion': grid.best_params_['criterion'],
   'n_estimators': grid.best_params_['n_estimators'],
   'max_depth': grid.best_params_['max_depth'],
   'min_samples_split': grid.best_params_['min_samples_split'],
   'min_samples_leaf': grid.best_params_['min_samples_leaf'],
    }

print("MIGLIORI IPERPARAMETRI RANDOM FOREST = ", bestParameters)

'''MIGLIORI IPERPARAMETRI PER LA RANDOM FOREST
{'criterion': 'gini',
'n_estimators': 100,
'max_depth': 10,
'min_samples_split': 10,
'min_samples_leaf': 1}'''

#validazione-cross validation

rf=RandomForestClassifier(criterion='gini', n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=1)

cv = StratifiedKFold(shuffle=True,n_splits=5)

cv = StratifiedKFold(n_splits=5)

scores_acc = cross_validate(rf, X, Y, cv=cv, scoring='accuracy',return_train_score=True)
scores_f1_macro=cross_validate(rf, X, Y, cv=cv,scoring="f1_macro",return_train_score=True)
scores_prec_macro=cross_validate(rf, X, Y, cv=cv,scoring="precision_macro",return_train_score=True)
scores_recall_macro=cross_validate(rf, X, Y, cv=cv,scoring="recall_macro",return_train_score=True)
scores_weighted = cross_validate(rf, X, Y, cv=cv,scoring="f1_weighted",return_train_score=True)
scores_prec_weighted = cross_validate(rf, X, Y, cv=cv,scoring="precision_weighted",return_train_score=True)
scores_recall_weighted = cross_validate(rf, X, Y, cv=cv,scoring="recall_weighted",return_train_score=True)

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


#learning curves accuracy
train_sizes, train_scores, test_scores = learning_curve(
    rf, X_train, Y_train, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

# Calcola la media e la deviazione standard delle score
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Traccia le learning curves
plt.figure()
plt.title("Learning Curves per la Random Forest")
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

#learning curves errore
train_sizes, train_scores, test_scores = learning_curve(
    rf, X_train, Y_train, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
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
plt.title("Error Learning Curve per la Random Forest")
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

#vaidazione-singolo sample

rf.fit(X_train,Y_train)

print(classification_report(Y_test,rf.predict(X_test)))

cm=confusion_matrix(Y_test, rf.predict(X_test))

display=ConfusionMatrixDisplay(cm)
display.plot()

#salvataggio del modello

with open('random_forest_classifier_model', 'wb') as file:
  pickle.dump(rf,file)
file.close()