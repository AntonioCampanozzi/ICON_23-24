from sklearn.model_selection import train_test_split
from kb import pl
import pickle
from classification import model
from  ontology import Ontologia

if __name__ == '__main__':
    with open('dataset/processed_dataset.PICKLE', 'rb') as file:
        df = pickle.load(file)
    file.close()
    # divisione dataset in feature e label
    X = df.drop('Crop', axis=1)
    Y = df['Crop']
    # train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print('Scegli una delle seguenti opzioni(digita il numero corrispondente)')
    print('1.interroga l\'ontologia')
    print('2.effettua una predizione')
    print('3.interroga la knowedge base per informazioni aggiuntive')
    choice = input(':')
    match choice:
        case '1':
            Ontologia.consultOntology()
        case '2':
            model.predict_point()
        case '3':
            pl.consultKB()
