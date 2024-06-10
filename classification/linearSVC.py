
'''
|------------------------------------------------------------------------------------------------------------------|
| ESTRATTO DELLA SEZIONE DEL NOTEBOOK COLAB CON TUNING IPERPARAMETRI, TRAINING E VALIDAZIONE DEL MODELLO LINEARSVC |
|------------------------------------------------------------------------------------------------------------------|
'''
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

grid=GridSearchCV(lsvc, lsvc_params)

grid.fit(X_train,Y_train)

bestParameters = {
   'C': grid.best_params_['C'],
   'loss': grid.best_params_['loss'],
   'tol': grid.best_params_['tol'],
   'max_iter': grid.best_params_['max_iter'],
    }

print(f'MIGLIORI IPERPARAMETRI PER LINEARSVC: \n{bestParameters}')  #{'C': 100, 'loss': 'squared_hinge', 'tol': 0.01, 'max_iter': 3000}

#validazione-cross validation

lsvc=LinearSVC(dual=False, C=100, loss='squared_hinge', tol=0.01, max_iter=3000)

with open('linear_svc_model', 'rb') as file:
  lsvc=pickle.load(file)
file.close()

cv = StratifiedKFold(n_splits=5)

scores_acc = cross_validate(lsvc, X, Y, cv=cv, scoring='accuracy',return_train_score=True)
scores_f1_macro=cross_validate(lsvc, X, Y, cv=cv,scoring="f1_macro",return_train_score=True)
scores_prec_macro=cross_validate(lsvc, X, Y, cv=cv,scoring="precision_macro",return_train_score=True)
scores_recall_macro=cross_validate(lsvc, X, Y, cv=cv,scoring="recall_macro",return_train_score=True)
scores_weighted = cross_validate(lsvc, X, Y, cv=cv,scoring="f1_weighted",return_train_score=True)
scores_prec_weighted = cross_validate(lsvc, X, Y, cv=cv,scoring="precision_weighted",return_train_score=True)
scores_recall_weighted = cross_validate(lsvc, X, Y, cv=cv,scoring="recall_weighted",return_train_score=True)

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

lsvc.fit(X_train.values,Y_train.values)

#salvataggio modello

with open('linear_svc_model', 'wb') as file:
  pickle.dump(lsvc,file)
file.close()

print(classification_report(Y_test.values,lsvc.predict(X_test.values)))

cm=confusion_matrix(Y_test.values, lsvc.predict(X_test.values))

display=ConfusionMatrixDisplay(cm)
display.plot()