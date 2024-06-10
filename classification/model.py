import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import pickle


def predict_point():
    values = []
    df = pd.read_csv('./dataset/Crop_Recommendation.csv')
    #raccoglimento degli input
    nitro = input(
        f'inserisci livello di nitrogeno (max individuato: {df["Nitrogen"].max()}-min individuato: {df["Nitrogen"].min()}):')
    values.append(float(nitro))
    phos = input(
        f'inserisci livello di fosforo (max individuato: {df["Phosphorus"].max()}-min individuato: {df["Phosphorus"].min()}):')
    values.append(float(phos))
    potassium = input(
        f'inserisci livello di potassio (max individuato: {df["Potassium"].max()}-min individuato: {df["Potassium"].min()}):')
    values.append(float(potassium))
    temperature = input(
        f'inserisci temperatura(Celsius) (max individuato: {df["Temperature"].max()}-min individuato: {df["Temperature"].min()}):')
    values.append(float(temperature))
    humidity = input(
        f'inserisci livello di umidità(%) (max individuato: {df["Humidity"].max()}-min individuato: {df["Humidity"].min()}):')
    values.append(float(humidity))
    ph = input(
        f'inserisci livello di ph (max individuato: {df["pH_Value"].max()}-min individuato: {df["pH_Value"].min()}):')
    values.append(float(ph))
    rainfall = input(
        f'inserisci livello di piogge (mm) (max individuato: {df["Rainfall"].max()}-min individuato: {df["Rainfall"].min()}):')
    values.append(float(rainfall))
    #scaling del nuovo punto
    features_to_scale = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
    scaled_point = []
    i = 0
    for f in features_to_scale:#non disponendo dell'oggetto minmaxscaler originale in questo file, per non scalare il dataset nuovamente si è optato all'approccio raw
        scaled_feature = (values[i] - df[f].min()) / (df[f].max() - df[f].min())
        scaled_point.append(scaled_feature)
        i += 1

    #chiamo il modello preaddestrato (verrà usata la random forest)
    with open('./pre_trained_models/random_forest_classifier_model', 'rb') as file:
        model = pickle.load(file)
    file.close()
    #effettuo la predzione
    pred=model.predict(np.array(scaled_point).reshape(1, -1))
    le=LabelEncoder()
    le.fit_transform(df['Crop']) #mappo le label per ricavare poi al numero la stringa corrispondente
    crop=le.inverse_transform(pred)
    print(f'la coltura adatta è: {crop[0]}')




def vizualizemetrics(pretrained_model, x, y):
    with open(pretrained_model, 'rb') as file:
        model = pickle.load(file)
    file.close()

    cv = StratifiedKFold(n_splits=5)

    scores_acc = cross_validate(model, x, y, cv=cv, scoring='accuracy', return_train_score=True)
    scores_f1_macro = cross_validate(model, x, y, cv=cv, scoring="f1_macro", return_train_score=True)
    scores_prec_macro = cross_validate(model, x, y, cv=cv, scoring="precision_macro", return_train_score=True)
    scores_recall_macro = cross_validate(model, x, y, cv=cv, scoring="recall_macro", return_train_score=True)
    scores_weighted = cross_validate(model, x, y, cv=cv, scoring="f1_weighted", return_train_score=True)
    scores_prec_weighted = cross_validate(model, x, y, cv=cv, scoring="precision_weighted", return_train_score=True)
    scores_recall_weighted = cross_validate(model, x, y, cv=cv, scoring="recall_weighted", return_train_score=True)

    print(
        f'l\'accuratezza media cross_validata è: {np.mean(scores_acc.get("train_score")) * 100:.2f}% con una deviazione standard del: {np.std(scores_acc.get("train_score")) * 100:.2f}% per il train set')
    print(
        f'l\'accuratezza media cross_validata è: {np.mean(scores_acc.get("test_score")) * 100:.2f}% con una deviazione standard del: {np.std(scores_acc.get("test_score")) * 100:.2f}% per il test set\n')

    print(
        f'l\'f1_score_macro medio cross_validato è: {np.mean(scores_f1_macro.get("train_score")) * 100:.2f}% con una deviazione standard del: {np.std(scores_f1_macro.get("train_score")) * 100:.2f}% per il train set')
    print(
        f'l\'f1_score_macro medio cross_validato è: {np.mean(scores_f1_macro.get("test_score")) * 100:.2f}% con una deviazione standard del: {np.std(scores_f1_macro.get("test_score")) * 100:.2f}% per il test set\n')

    print(
        f'la precision_macro media cross_validata è: {np.mean(scores_prec_macro.get("train_score")) * 100:.2f}% con una deviazione standard del: {np.std(scores_prec_macro.get("train_score")) * 100:.2f}% per il train set')
    print(
        f'la precision_macro media cross_validata è: {np.mean(scores_prec_macro.get("test_score")) * 100:.2f}% con una deviazione standard del: {np.std(scores_prec_macro.get("test_score")) * 100:.2f}% per il test set\n')

    print(
        f'la recall_macro media cross_validata è: {np.mean(scores_recall_macro.get("train_score")) * 100:.2f}% con una deviazione standard del: {np.std(scores_recall_macro.get("train_score")) * 100:.2f}% per il train set')
    print(
        f'la recall_macro media cross_validata è: {np.mean(scores_recall_macro.get("test_score")) * 100:.2f}% con una deviazione standard del: {np.std(scores_recall_macro.get("test_score")) * 100:.2f}% per il test set\n')

    print(
        f'l\'f1_score_weighted medio cross_validato è: {np.mean(scores_weighted.get("train_score")) * 100:.2f}% con una deviazione standard del: {np.std(scores_weighted.get("train_score")) * 100:.2f}% per il train set')
    print(
        f'l\'f1_score_weighted medio cross_validato è: {np.mean(scores_weighted.get("test_score")) * 100:.2f}% con una deviazione standard del: {np.std(scores_weighted.get("test_score")) * 100:.2f}% per il test set\n')

    print(
        f'la precision_weighted media cross_validata è: {np.mean(scores_prec_weighted.get("train_score")) * 100:.2f}% con una deviazione standard del: {np.std(scores_prec_weighted.get("train_score")) * 100:.2f}% per il train set')
    print(
        f'la precision_weighted media cross_validata è: {np.mean(scores_prec_weighted.get("test_score")) * 100:.2f}% con una deviazione standard del: {np.std(scores_prec_weighted.get("test_score")) * 100:.2f}% per il test set\n')

    print(
        f'la recall_weighted media cross_validata è: {np.mean(scores_recall_weighted.get("train_score")) * 100:.2f}% con una deviazione standard del: {np.std(scores_recall_weighted.get("train_score")) * 100:.2f}% per il train set')
    print(
        f'la recall_weighted media cross_validata è: {np.mean(scores_recall_weighted.get("test_score")) * 100:.2f}% con una deviazione standard del: {np.std(scores_recall_weighted.get("test_score")) * 100:.2f}% per il test set')

if __name__ == '__main__':
    df = pd.read_csv('../dataset/Crop_Recommendation.csv')
    with open('../pre_trained_models/random_forest_classifier_model', 'rb') as file:
        model = pickle.load(file)
    file.close()
    pred=model.predict(np.array([0.42857142857142855, 0.20714285714285716, 0.21, 0.4985485176118481, 0.6969118840017569, 0.543555791294453, 0.7029155410877844]).reshape(1, -1))
    le=LabelEncoder()
    le.fit_transform(df['Crop'])
    mario=le.inverse_transform(pred)
    print(mario[0])
