
'''
|------------------------------------------------------------------------------------------------------------|
| ESTRATTO DELLA SEZIONE DEL NOTEBOOK COLAB CON TUNING IPERPARAMETRI, TRAINING E VALIDAZIONE DEL MODELLO SVC |
|------------------------------------------------------------------------------------------------------------|
'''

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

print(f'MIGLIORI IPERPARAMETRI PER SVC: \n{bestParameters}') #{'C': 1, 'kernels': 'poly', 'gamma': 'scale', 'degree': 2}

#validazione-cross validation

svc=SVC(C=1, kernel='poly', gamma='scale', degree=2)

with open('svc_model', 'rb') as file:
  svc=pickle.load(file)
file.close()

cv = StratifiedKFold(n_splits=5)

scores_acc = cross_validate(svc, X, Y, cv=cv, scoring='accuracy',return_train_score=True)
scores_f1_macro=cross_validate(svc, X, Y, cv=cv,scoring="f1_macro",return_train_score=True)
scores_prec_macro=cross_validate(svc, X, Y, cv=cv,scoring="precision_macro",return_train_score=True)
scores_recall_macro=cross_validate(svc, X, Y, cv=cv,scoring="recall_macro",return_train_score=True)
scores_weighted = cross_validate(svc, X, Y, cv=cv,scoring="f1_weighted",return_train_score=True)
scores_prec_weighted = cross_validate(svc, X, Y, cv=cv,scoring="precision_weighted",return_train_score=True)
scores_recall_weighted = cross_validate(svc, X, Y, cv=cv,scoring="recall_weighted",return_train_score=True)

print(f'l\'accuratezza media cross_validata è: {np.mean(scores_acc.get("train_score"))*100:.2f}% con una deviazione standard del: {np.std(scores_acc.get("train_score"))*100:.2f}% per il train set')
print(f'l\'accuratezza media cross_validata è: {np.mean(scores_acc.get("test_score"))*100:.2f}% con una deviazione standard del: {np.std(scores_acc.get("test_score"))*100:.2f}% per il test set\n')

print(f'l\'f1_score_macro medio cross_validato è: {np.mean(scores_f1_macro.get("train_score"))*100:.2f}% con una deviazione standard del: {np.std(scores_f1_macro.get("train_score"))*100:.2f}% per il train set')
print(f'l\'f1_score_macro medio cross_validato è: {np.mean(scores_f1_macro.get("test_score"))*100:.2f}% con una deviazione standard del: {np.std(scores_f1_macro.get("test_score"))*100:.2f}% per il test set\n')

print(f'la precision_macro media cross_validata è: {np.mean(scores_prec_macro.get("train_score"))*100:.2f}% con una deviazione standard del: {np.std(scores_prec_macro.get("train_score"))*100:.2f}% per il train set')
print(f'la precision_macro media cross_validata è: {np.mean(scores_prec_macro.get("test_score"))*100:.2f}% con una deviazione standard del: {np.std(scores_prec_macro.get("test_score"))*100:.2f}% per il test set\n')

print(f'la recall_macro media cross_validata è: {np.mean(scores_recall_macro.get("train_score"))*100:.2f}% con una deviazione standard del: {np.std(scores_recall_macro.get("train_score"))*100:.2f}% per il train set')
print(f'la recall_macro media cross_validata è: {np.mean(scores_recall_macro.get("test_score"))*100:.2f}% con una deviazione standard del: {np.std(scores_recall_macro.get("test_score"))*100:.2f}% per il test set\n')

print(f'l\'f1_score_weighted medio cross_validato è: {np.mean(scores_weighted.get("train_score"))*100:.2f}% con una deviazione standard del: {np.std(scores_weighted.get("train_score"))*100:.2f}% per il train set')
print(f'l\'f1_score_weighted medio cross_validato è: {np.mean(scores_weighted.get("test_score"))*100:.2f}% con una deviazione standard del: {np.std(scores_weighted.get("test_score"))*100:.2f}% per il test set\n')

print(f'la precision_weighted media cross_validata è: {np.mean(scores_prec_weighted.get("train_score"))*100:.2f}% con una deviazione standard del: {np.std(scores_prec_weighted.get("train_score"))*100:.2f}% per il train set')
print(f'la precision_weighted media cross_validata è: {np.mean(scores_prec_weighted.get("test_score"))*100:.2f}% con una deviazione standard del: {np.std(scores_prec_weighted.get("test_score"))*100:.2f}% per il test set\n')

print(f'la recall_weighted media cross_validata è: {np.mean(scores_recall_weighted.get("train_score"))*100:.2f}% con una deviazione standard del: {np.std(scores_recall_weighted.get("train_score"))*100:.2f}% per il train set')
print(f'la recall_weighted media cross_validata è: {np.mean(scores_recall_weighted.get("test_score"))*100:.2f}% con una deviazione standard del: {np.std(scores_recall_weighted.get("test_score"))*100:.2f}% per il test set')

#validazione-singolo sample

svc.fit(X_train.values, Y_train.values)

#salvataggio del modello

with open('svc_model', 'wb') as file:
  pickle.dump(svc, file)
file.close()

print(classification_report(Y_test.values,svc.predict(X_test.values)))

cm=confusion_matrix(Y_test.values, svc.predict(X_test.values))

display=ConfusionMatrixDisplay(cm)
display.plot()