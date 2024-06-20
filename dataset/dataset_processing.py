import pandas as pd
import numpy as np
import seaborn as sns
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

#defizione encoder e scaler

labelencoder=LabelEncoder()
stdscaler=MinMaxScaler(feature_range=(0,1))

columns_to_scale = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']

#distribuzioni delle features

for i in columns_to_scale:
  plt.hist(df[i])
  plt.title(f'distribuzione di {i}')
  plt.show()

#boxplot per potassium

plt.boxplot(df['Potassium'])
plt.show()

#distribuzione delle classi

label_counts = df['Crop'].value_counts()
plt.pie(label_counts, autopct='%1.1f%%', labeldistance=0.9, radius=2.0)
plt.show()

# Applica fit_transform su tutto il dataframe per calcolare i parametri di scalatura

df[columns_to_scale] = stdscaler.fit_transform(df[columns_to_scale])
df['Crop']=labelencoder.fit_transform(df['Crop'])

#calcolo della matrice di correlazione e seguente heatmap

corr_matr=df[columns_to_scale].corr()
plt.figure(figsize=(10,8))
plt.title('heatmap su dataset processato')
sns.heatmap(corr_matr, cmap='coolwarm', annot=True)
plt.show()

#divisione dataset in feature e label

X=df.drop('Crop', axis=1)
Y=df['Crop']

#divisione in train e test

X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2, random_state=42)

df.head()
