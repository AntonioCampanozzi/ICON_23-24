
'''
|----------------------------------------------------------------------------------------------------------------------|
| ESTRATTO DELLA SEZIONE DEL NOTEBOOK COLAB CON TUNING IPERPARAMETRI, TRAINING E VALIDAZIONE DEL MODELLO RANDOM FOREST |
|----------------------------------------------------------------------------------------------------------------------|
'''

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

print("MIGLIORI IPERPARAMETRI RANDOM FOREST = ", bestParameters) #{'criterion': 'gini', 'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 1}

#validazione-cross validation

rf=RandomForestClassifier(criterion='gini', n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=1)

with open('random_forest_model', 'rb') as file:
  rf=pickle.load(file)
file.close()

cv = StratifiedKFold(n_splits=5)

scores_acc = cross_validate(rf, X, Y, cv=cv, scoring='accuracy',return_train_score=True)
scores_f1_macro=cross_validate(rf, X, Y, cv=cv,scoring="f1_macro",return_train_score=True)
scores_prec_macro=cross_validate(rf, X, Y, cv=cv,scoring="precision_macro",return_train_score=True)
scores_recall_macro=cross_validate(rf, X, Y, cv=cv,scoring="recall_macro",return_train_score=True)
scores_weighted = cross_validate(rf, X, Y, cv=cv,scoring="f1_weighted",return_train_score=True)
scores_prec_weighted = cross_validate(rf, X, Y, cv=cv,scoring="precision_weighted",return_train_score=True)
scores_recall_weighted = cross_validate(rf, X, Y, cv=cv,scoring="recall_weighted",return_train_score=True)

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


#vaidazione-singolo sample

rf.fit(X_train.values,Y_train.values)

#salvataggio del modello

with open('random_forest_classifier_model', 'wb') as file:
  pickle.dump(rf,file)
file.close()

print(classification_report(Y_test.values,rf.predict(X_test.values)))

cm=confusion_matrix(Y_test.values, rf.predict(X_test.values))

display=ConfusionMatrixDisplay(cm)
display.plot()