import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np

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
