%crop(nome, azoto, fosforo, potassio, Temperatura_max, Temperatura_min,
% ph, umidita, piogge, prezzo)
crop(rice, 80 , 47, 40 , 20 , 27 , 6.4 , 82 , 236 , 1000).
crop(maize, 78 , 48 , 20 , 18 , 26 , 6.2 , 65 , 84 , 700).
crop(chickpea, 40 , 68 , 80 , 17 , 21, 7.3 , 16 , 80 , 900).
crop(kidneybeans, 21 , 67 , 20 , 15 , 25 , 5.7 , 95 , 176 , 600).
crop(pigeonpeas, 21 , 68 , 20 , 18 , 37 , 5.7 , 48 , 149 , 400).
crop(mothbeans, 21 , 48 , 20 , 24 , 31 , 6.8 , 53 , 51 , 400).
crop(mungbean, 21 , 47 , 20 , 27 , 30 , 6.7 , 85 , 48 , 400).
crop(blackgram, 40 , 67 , 19 , 25 , 35 , 7.1 , 65 , 68 , 400).
crop(lentil, 19 , 68 , 19 , 18 , 30 , 6.9 , 65 , 46 , 500).
crop(pomegranate, 19 , 19 , 40 , 18 , 25 , 6.4 , 90 , 107 , 2500).
crop(banana, 100 , 82 , 50 , 25 , 30 , 5.9 , 80 , 105 , 3000).
crop(mango, 20 , 27 , 30 , 27 , 36 , 5.7 , 50 , 95 , 2000).
crop(grapes, 23 , 132 , 200 , 8 , 12 , 6.0 , 82 , 69 , 5000).
crop(watermelon, 99 , 17 , 50 , 24 , 27 , 6.5 , 85 , 51 , 2000).
crop(muskmelon, 100 , 18 , 50 , 27 , 27 , 6.3 , 92 , 24.7 , 2000).
crop(apple, 21 , 134 , 200 , 21 , 24 , 5.9 , 92 , 113 , 10000).
crop(orange, 19 , 16 , 10 , 10 , 35 , 7.0 , 92 , 110 , 3000).
crop(papaya, 50 , 59 , 50 , 23 , 44 , 6.7 , 92 , 143 , 1500).
crop(coconut, 22 , 17 , 31 , 25 , 30 , 5.9 , 95 , 176 , 1500).
crop(cotton, 118 , 46 , 20 , 22 , 26 , 6.9 , 80 , 80 , 1500).
crop(jute, 78 , 47 , 40 , 23 , 27 , 6.7 , 80 , 175 , 1000).
crop(coffee, 101 , 29 , 30 , 23 , 28 , 6.8 , 58 , 158 , 4000).

%month(nome, temperatura_max, temperatura_min)
month(gennaio, 3 , 12).
month(febbraio, 3 , 14).
month(marzo, 6 , 17).
month(aprile, 9 , 22).
month(maggio, 13 , 24).
month(giugno, 17 , 28).
month(luglio, 19 , 31).
month(agosto, 22 , 44).
month(settembre, 15 , 27).
month(ottobre, 12 , 22).
month(novembre, 7 , 17).
month(dicembre, 4 , 13).

ph(acido).
ph(neutro).
ph(basico).

humidity(alta).
humidity(media).
humidity(bassa).

rainfall(arida).
rainfall(standard).
rainfall(piovosa).

sowingmonth(Month,Crop):-(month(Month,TMmin,TMmax)),
    (crop(Crop,_,_,_,TCmin,TCmax,_,_,_,_)),
    TCmin >= TMmin,TCmax =< TMmax.

enoughmoney(Budget):-
    findall(Price,crop(_,_,_,_,_,_,_,_,_,Price),Prices),
    min_list(Prices,Basebudget),
    Budget>=Basebudget.

cropcost(Acres,Crop,Result):-Acres>0,
    (crop(Crop,_,_,_,_,_,_,_,_,Price)),
    Result is Acres*Price.

isonbudget(Budget,Crop,Acres):-(enoughmoney(Budget)),
    (cropcost(Acres,Crop,C)),Cost is C,Budget>=Cost.

sowing_economical_info(Acres,Month,Budget,Crop):-(sowingmonth(Month,Crop)),
    (isonbudget(Budget,Crop,Acres)).

checkPH_level(PH_type,Crop):-
    ph(PH_type),
    PH_type=acido,crop(Crop,_,_,_,_,_,PH_level,_,_,_),PH_level<7;
    PH_type=neutro,crop(Crop,_,_,_,_,_,PH_level,_,_,_),PH_level=7;
    PH_type=basico,crop(Crop,_,_,_,_,_,PH_level,_,_,_),PH_level>7.

checkhumidity(Humiditytype,Crop):-
    humidity(Humiditytype),
    Humiditytype=bassa,crop(Crop,_,_,_,_,_,_,Humiditylevel,_,_),Humiditylevel>=0,Humiditylevel<30;
    Humiditytype=media,crop(Crop,_,_,_,_,_,_,Humiditylevel,_,_),Humiditylevel>=30,Humiditylevel<60;
    Humiditytype=alta,crop(Crop,_,_,_,_,_,_,Humiditylevel,_,_),Humiditylevel>=60.

checkrainfall(Rainfalltype,Crop):-
    rainfall(Rainfalltype),
    Rainfalltype=arida,crop(Crop,_,_,_,_,_,_,_,Rainfall_level,_),Rainfall_level>=0,Rainfall_level<55;
    Rainfalltype=standard,crop(Crop,_,_,_,_,_,_,_,Rainfall_level,_),Rainfall_level>=55,Rainfall_level<150;
    Rainfalltype=piovosa,crop(Crop,_,_,_,_,_,_,_,Rainfall_level,_),Rainfall_level>=150.

sowing_zone_info(PH,Humidity,Rainfall,Crop):-checkPH_level(PH,Crop),
    checkhumidity(Humidity,Crop),
    checkrainfall(Rainfall,Crop).

min_cost_crop(Acres, Month, Budget, MinCostCrop) :-
    %cerca tutti i crop che soddisfano i parametri economici
    findall(Crop, sowing_economical_info(Acres, Month, Budget, Crop), Crops),
    % estrai i costi di tutti i crop prima radunati
    findall(Cost, (member(Crop, Crops), cropcost(Acres, Crop, Cost)), Costs),
    %dei costi estra il minimo
    min_list(Costs, MinCost),
    %salva l'indice della lista dove si trova il costo minimo
    nth0(Index, Costs, MinCost),
    %dato che le liste hanno corrispondenza 1 a 1,
    %estre il crop corrispondente a Index estratto in precedenza
    nth0(Index, Crops, MinCostCrop).

crop_eucl_dist(Crop1,Crop2,D):-crop(Crop1,N1,F1,K1,TMX1,TMN1,PH1,H1,R1,P1),crop(Crop2,N2,F2,K2,TMX2,TMN2,PH2,H2,R2,P2),
    %distanza euclidea: sqrt((X1-X2)^2+(Y1-Y2)^2) per punti del tipo P1(X1,Y1), P2(X2,Y2)
    D is
    sqrt((N1-N2)**2+
         (F1-F2)**2+
         (K1-K2)**2+
         (TMX1-TMX2)**2 +
         (TMN1-TMN2)**2 +
         (PH1-PH2)**2 +
         (H1-H2)**2 +
         (R1-R2)**2 +
         (P1-P2)**2).

most_sim(Cropchosen,SimCrop):-
    crop(Cropchosen,_,_,_,_,_,_,_,_,_),
    %estrae tutti i crop escluso quello dato in input
    findall(Crop,(crop(Crop,_,_,_,_,_,_,_,_,_),Crop \= Cropchosen),Crops),
    %estrae tutte le distanze euclidee tra il crop in input e quelli estratti
    findall(Dist,(member(Crop,Crops),crop_eucl_dist(Cropchosen,Crop,Dist)),Distances),
    %trova il la distanza minima
    min_list(Distances,Nearest),
    %salva l'indice dove è presente la distanza minima
    nth0(Index,Distances,Nearest),
    %restituisce il crop presente nell'indice Index della lista dei crop
    nth0(Index,Crops,SimCrop).


