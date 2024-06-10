
'''
|---------------------------------------------------------------------------------------------------------------------------|
| ESTRATTO DELLA SEZIONE DEL NOTEBOOK COLAB CON TUNING IPERPARAMETRI, TRAINING E VALIDAZIONE DEL MODELLO K-NEAREST NEIGHBOR |
|---------------------------------------------------------------------------------------------------------------------------|
'''
#creazione knn

knn=KNeighborsClassifier()

#fase di tuning

parameters_knn = {
        "n_neighbors": [1, 3, 5, 7, 9, 12, 15],
        "weights": ["uniform", "distance"],
        "metric": ["minkowski", "manhattan"],
        "algorithm": ["ball_tree", "kd_tree", "brute"],
        "leaf_size": [20, 30, 40]}

#grid search dei parametri
grid_knn = GridSearchCV(knn,parameters_knn)

#addestramento modello
grid_knn.fit(X_train, Y_train)


print("Best params", grid_knn.best_params_) #{'algorithm': 'ball_tree', 'leaf_size': 20, 'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'distance'}

#validazione-cross validation

knn = KNeighborsClassifier(algorithm='ball_tree', leaf_size=20, metric='manhattan', n_neighbors=3, weights='distance')

with open('knn_model', 'rb') as file:
  knn=pickle.load(file)
file.close()

cv = StratifiedKFold(n_splits=5)

#cross validation
scores_acc = cross_validate(knn, X, Y, cv=cv, scoring='accuracy',return_train_score=True)
scores_f1_macro=cross_validate(knn, X, Y, cv=cv,scoring="f1_macro",return_train_score=True)
scores_prec_macro=cross_validate(knn, X, Y, cv=cv,scoring="precision_macro",return_train_score=True)
scores_recall_macro=cross_validate(knn, X, Y, cv=cv,scoring="recall_macro",return_train_score=True)
scores_weighted = cross_validate(knn, X, Y, cv=cv,scoring="f1_weighted",return_train_score=True)
scores_prec_weighted = cross_validate(knn, X, Y, cv=cv,scoring="precision_weighted",return_train_score=True)
scores_recall_weighted = cross_validate(knn, X, Y, cv=cv,scoring="recall_weighted",return_train_score=True)

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

knn.fit(X_train.values,Y_train.values)

#salvataggio del modello

with open('knn_model', 'wb') as file:
  pickle.dump(knn,file)
file.close()

#report di classificazione
print(classification_report(Y_test.values , knn.predict(X_test.values)))

cm = confusion_matrix(Y_test.values, knn.predict(X_test.values))

display = ConfusionMatrixDisplay(cm)
display.plot()