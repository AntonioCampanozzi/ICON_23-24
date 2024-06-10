from pyswip import Prolog

def consultKB():

    prolog = Prolog()
    prolog.consult('kb/crop.pl')
    print(
        'Scegli una delle seguenti opzioni per interrogare la base di conoscenza(inserisci il numero corrispondente)\n')
    print('1. coltura consigliata in base a ettari di terreno disponibili, mese di semina, budget')
    print('2.coltura consigliata in base al mese di semina scelto')
    print('3.budget minimo necessario data la coltura e la superficie in ettari')
    print('4.dato un budget, una coltura e un ettaraggio, se si hanno abbastanza soldi a disposizione per essa')
    choice = input(':')

    match choice:
        case '1':
            acres = input('inserisci ettari a disposizione: ')
            month = input('inserisci mese di semina: ')
            budget = input('inserisci il tuo budget: ')
            r = list(prolog.query(f'sowinginfo({acres},{month},{budget},Crop)'))
            print('colture consigliate:')
            for i in r:
                print(i['Crop'])

        case '2':
            month = input('inserisci mese di semina: ')
            r = list(prolog.query(f'sowingmonth({month},Crop)'))
            print(f'colture consigliate: per il mese di {month}:')
            for i in r:
                print(i['Crop'])
        case '3':
            acres = input('inserisci ettari a disposizione: ')
            crop = input('inserisci coltura: ')
            r = list(prolog.query(f'cropcost({acres},{crop},Result)'))
            print(f'budget minimo richiesto:{r[0].get('Result')}')
        case '4':
            acres = input('inserisci ettari a disposizione: ')
            crop = input('inserisci coltura: ')
            budget = input('inserisci il tuo budget: ')
            r = bool(list(prolog.query(f'isonbudget({budget},{crop},{acres})')))
            if r:
                print('il budget è sufficiente')
            else:
                print('il budget non è sufficiente')
