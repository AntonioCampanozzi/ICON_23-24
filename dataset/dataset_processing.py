import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pickle

'''
|----------------------------------------------------------------------------------|
| ESTRATTO DELLA SEZIONE DEL NOTEBOOK COLAB DOVE IL DATASET E' STATO PREPROCESSATO |
|----------------------------------------------------------------------------------|
'''


df=pd.read_csv('Crop_Recommendation.csv')

labelencoder=LabelEncoder()
stdscaler=MinMaxScaler(feature_range=(0,1))
columns_to_scale = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
df[columns_to_scale] = stdscaler.fit_transform(df[columns_to_scale])
df['Crop']=labelencoder.fit_transform(df['Crop'])

#distribuzione degli n valori della label

label_counts = df['Crop'].value_counts()
plt.pie(label_counts, autopct='%1.1f%%', labeldistance=0.9, radius=2.0)
plt.show()

#divisione dataset in feature e label

X=df.drop('Crop', axis=1)
Y=df['Crop']

#train-test split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2, random_state=42)