from owlready2 import *


def consultOntology():
    onto = get_ontology("ontology/agricoltura.owl").load()
    print('Scegli una delle seguenti opzioni:')
    print('1.Estrai le classi dell\'ontologia')
    print('2.Estrai le object propertise dell\' ontologia')
    print('3.Estrai le data properties dell\'ontologia')
    print('4.Estrai gli individui')
    print('5.Estrai le informazioni di una coltura')
    print('6.Filtra le colture in base a un fattore ambientale')
    print('7.Filtra le colture in base a un fattore nutrizionale')
    choice = input(':')
    match choice:
        case '1':
            print("LISTA DELLE CLASSI NELL'ONTOLOGIA:")
            print(list(onto.classes()), "\n")
        case '2':
            print("LISTA DELLE OBJECT PROPERTY NELL'ONTOLOGIA:")
            print(list(onto.object_properties()), "\n")
        case '3':
            print("LISTA DELLE DATA PROPERTY NELL'ONTOLOGIA:")
            print(list(onto.data_properties()), "\n")
        case '4':
            print("LISTA DEGLI INDIVIDUI DELLA CLASSE CROP")
            crop = onto.search(type=onto.Crop)
            print(crop, "\n")

            print("LISTA DEGLI INDIVIDUI DELLA CLASSE FATTORI_AMBIENTALI")
            ambientali = onto.search(type=onto.Fattori_ambientali)
            print(ambientali, "\n")

            print("LISTA DEGLI INDIVIDUI DELLA CLASSE FATTORI_NUTRIZIONALI")
            nutrizionali = onto.search(type=onto.Fattori_nutrizionali)
            print(nutrizionali, "\n")
        case '5':
            individual = input('inserisci individuo del quale vuoi conoscere i dati: ')
            query_5(individual, onto)
        case '6':
            factor_name = input(
                'inserisci nome del fattore ambientale (Humidity_percentage, Temperature_Celsius°, Rainfall_mm): ')
            min_val = input('inserisci valore minimo: ')
            max_val = input('inserisci valore massimo: ')
            query_6(factor_name, int(min_val), int(max_val), onto)
        case '7':
            factor_name = input('inserisci nome del fattore nutrizionale (Nitrogen, Phosphorus, Potassium, pH_value): ')
            min_val = input('inserisci valore minimo: ')
            max_val = input('inserisci valore massimo: ')
            query_7(factor_name, int(min_val), int(max_val), onto)


# funzione per recuperare e stampare le proprietà di un determinato individuo in un'ontologia
def getdatas(individual, onto):
    # ricerca dell'individuo
    individual = onto.search(iri=f'*{individual}')
    # Inizializzazione del dizionario delle proprietà
    data_properties = {}
    # Raccolta delle proprietà dell'individuo
    for prop in individual[0].get_properties():
        data_properties[prop.name] = getattr(individual[0], prop.name)
    # Stampa delle proprietà
    for prop_name, prop_values in data_properties.items():
        print(f"{prop_name}: {prop_values[0]}")
    print('\n')
    return data_properties


# QUERY che permette di ottenere tutti i dati relativi a un tipo di Crop dato in input
def query_5(individual, onto):
    ind = onto.search(iri=f'*{individual}')
    print('dati coltura:')
    getdatas(ind[0].name, onto)
    print('fattori ambientali:')
    getdatas(ind[0].hasEnvironmentalFactor[0].name, onto)
    print('fattori nutrizionali:')
    getdatas(ind[0].hasNutritionalFactor[0].name, onto)


# QUERY che permette di ottenere tutte le istanze di crop aventi un valore minore
# rispetto ad un determinato fattore ambientale
def query_6(factor_name, min, max, onto):
    result = []
    try:
        for crop in onto.Crop.instances():
            for environmental_factor in crop.hasEnvironmentalFactor:
                attribute_values = getattr(environmental_factor, factor_name)
                if min <= attribute_values[0] <= max:
                    result.append(crop)
        if result.__len__() == 0:
            print(f'nessuno degli individui ha {factor_name} tra {min} e {max}')
        else:
            print(f'Risultati Crop con {factor_name} tra {min} e {max}:{result}')
    except:
        print('nome del fattore inserito errato')


# QUERY che permette di ottenere tutte le istanze di crop aventi un determinato fattore nutrizionale in
# un dato range
def query_7(factor_name, min, max, onto):
    result = []
    try:
        for crop in onto.Crop.instances():
            for nutritional_factor in crop.hasNutritionalFactor:
                attribute_values = getattr(nutritional_factor, factor_name)
                if min <= attribute_values[0] <= max:
                    result.append(crop)
        if result.__len__() == 0:
            print(f'nessuno degli individui ha {factor_name} tra {min} e {max}')
        else:
            print(f'Risultati Crop con {factor_name} tra {min} e {max}:{result}')
    except:
        print('nome del fattore inserito errato')
