from pyswip import Prolog

def consultKB():
    prolog = Prolog()
    prolog.consult('kb/crop.pl')
    print(
        'Scegli una delle seguenti opzioni per interrogare la base di conoscenza(inserisci il numero corrispondente)\n')
    print('1. coltura consigliata in base a ettari di terreno disponibili, mese di semina, budget')
    print('2. coltura consigliata in base al mese di semina scelto')
    print('3. budget minimo necessario data la coltura e la superficie in ettari')
    print('4. coltura più economica dati gli ettari, il mese sdi semina e il budget')
    print('5. coltura consigliata in base ai dettagli della zona di coltivazione')
    print('6. coltura più simile a quella fornita in input')
    choice = input(':')

    match choice:
        case '1':
            acres = input('inserisci ettari a disposizione: ')
            month = input('inserisci mese di semina: ')
            budget = input('inserisci il tuo budget: ')
            r = list(prolog.query(f'sowing_economical_info({acres},{month},{budget},Crop)'))
            print('colture consigliate:')
            if r.__len__() == 0:
                print('nessuna delle colture al momento presenti nella base di conoscenza è adatta.')
            else:
                for i in r:
                    print(i['Crop'])

        case '2':
            month = input('inserisci mese di semina: ')
            r = list(prolog.query(f'sowingmonth({month},Crop)'))
            print(f'colture consigliate: per il mese di {month}:')
            if r.__len__() == 0:
                print('nessuna delle colture al momento presenti nella base di conoscenza è adatta.')
            else:
                for i in r:
                    print(i['Crop'])
        case '3':
            acres = input('inserisci ettari a disposizione: ')
            crop = input('inserisci coltura: ')
            r = list(prolog.query(f'cropcost({acres},{crop},Result)'))
            print(f'budget minimo richiesto:{r[0].get('Result')}')

        case '4':
            acres = input('inserisci ettari a disposizione: ')
            month = input('inserisci mese di semina: ')
            budget = input('inserisci il tuo budget: ')
            r = list(prolog.query(f'min_cost_crop({acres},{month},{budget},MinCostCrop)'))
            if r.__len__() == 0:
                print(
                    'nessuna delle colture al momento presenti nella base di conoscenza ha soddisfatto i parametri presentati.')
            else:
                c = list(prolog.query(f'cropcost({acres},{r[0].get('MinCostCrop')},Result)'))
                print(
                    f'la coltura più economica è {r[0].get('MinCostCrop')} con un costo di semina pari a: {c[0].get('Result')}')

        case '5':
            ph = input('inserisci il livello di ph del terreno(basico,neutro,acido): ')
            humidity = input('inserisci il livello di umidità della zona(alta, media, bassa): ')
            rain = input('inserisci atmosfers della zona(arida, standard, piovosa): ')
            r = list(prolog.query(f'sowing_zone_info({ph},{humidity},{rain}, Crop)'))
            print(f'colture consigliate per un terreno {ph}, in una zona con {humidity} umidità e {rain}:')
            if r.__len__() == 0:
                print('nessuna delle colture al momento presenti nella base di conoscenza è adatta.')
            else:
                for i in r:
                    print(i['Crop'])
        case '6':
            crop = input('inserisci coltura: ')
            r = list(prolog.query(f'most_sim({crop}, SimilarCrop)'))
            if r.__len__() == 0:
                print('nessuna coltura simile trovata, probabilmente l\'input inserito è errato o assente?')
            else:
                print(f'la coltura più simile a {crop} è: {r[0].get('SimilarCrop')}')
