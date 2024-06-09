crop(rice,20,27,1000).
crop(maize,18,26,700).
crop(chickpea,17,21,900).
crop(kidneybeans,15,25,600).
crop(pidgeonpeas,18,37,400).
crop(mothbeans,24,31,400).
crop(mungbean,27,30,400).
crop(blackgram,25,35,400).
crop(lentil,18,30,500).
crop(pomegranate,18,25,2500).
crop(banana,25,30,3000).
crop(mango,27,36,2000).
crop(grapes,8,12,5000).
crop(watermelon,24,27,2000).
crop(muskmelon,27,27,2000).
crop(apple,21,24,10000).
crop(orange,10,35,3000).
crop(papaya,23,44,1500).
crop(coconut,25,30,1500).
crop(cotton,22,26,1500).
crop(jute,23,27,1000).
crop(coffee,23,28,4000).

month(gennaio,3,12).
month(febbraio,3,14).
month(marzo,6,17).
month(aprile,9,22).
month(maggio,13,24).
month(giugno,17,28).
month(luglio,19,31).
month(agosto,22,44).
month(settembre,15,27).
month(ottobre,12,22).
month(novembre,7,17).
month(dicembre,4,13).


sowingmonth(Month,Crop):-(month(Month,TMmin,TMmax)),(crop(Crop,TCmin,TCmax,_)),TCmin >= TMmin,TCmax =< TMmax.
enoughmoney(Budget):-Budget>=400.
cropcost(Acres,Crop,Result):-Acres>0,(crop(Crop,_,_,Price)),Result is Acres*Price.
isonbudget(Budget,Crop,Acres):-enoughmoney(Budget),(cropcost(Acres,Crop,C)),Cost is C,Budget>=Cost.
sowinginfo(Acres,Month,Budget,Crop):-(sowingmonth(Month,Crop)),(isonbudget(Budget,Crop,Acres)).

