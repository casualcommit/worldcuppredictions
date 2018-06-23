from keras.layers import Dense
from keras.models import Sequential
#import numpy as np
#import os
#import tensorflow as tf
import pandas as pd
import glob
from collections import defaultdict
from sklearn import preprocessing
import random

path ='C:/Users/eshve/Desktop/PlayersTeams' # use your path
allFiles2 = glob.glob(path + "/*.csv")
frame2 = pd.DataFrame()
listofteams_ = []
dq =  pd.DataFrame(columns=['Country','PlayerScore'])
for file_ in allFiles2:
    smallist = []
    dr = pd.read_csv(file_, index_col=None, header=None, encoding = "ISO-8859-1")
    score = round(dr.iloc[:,1:2].values.sum(),2)
    smallist.append(file_.replace('.csv','').replace('C:/Users/eshve/Desktop/PlayersTeams\\',''))
    smallist.append(score)
    listofteams_.append(smallist)

dq = pd.DataFrame(listofteams_, columns=['Country','PlayerScore'])

d = defaultdict(preprocessing.LabelEncoder)

path2 ='C:/Users/eshve/Desktop/WorldCupHistorical' # use your path
allFiles = glob.glob(path2 + "/*.csv")
frame = pd.DataFrame()

list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
df = pd.concat(list_)

list1 = []
list2 = []
list3 = []

list1 = df.iloc[:,[0,1]].values.tolist()
list2 = df.iloc[:,[5,4]].values.tolist()

list3 = [list(a) for a in zip(list1, list2)]
for item in list3:
    item = random.shuffle(item)

dk = pd.DataFrame(list3)
ds = pd.DataFrame()

list4 = []
list5 = []
list6 = []
list7 = []

for i in dk[0]:
    list4.append(i[0])
    list5.append(i[1])
    
for i in dk[1]:
    list6.append(i[0])
    list7.append(i[1])
    
ds['home'] = list4
ds['homeresult'] = list5
ds['awayresult'] = list7
ds['away'] = list6
print(ds[0:5])
    
df = ds
print(df[0:5])
df = df.apply(lambda x: d[x.name].fit_transform(x))

dd = pd.read_csv("C:/Users/eshve/Desktop/Project/predict.csv",index_col=None, header=0)
dd = dd.iloc[:,[0,1,4,5]]
dd = dd.apply(lambda x: d[x.name].fit_transform(x))

cleanup_nums = {"homeresult":     {2 : 1, 1 : 0.5},
               "awayresult":     {2 : 1, 1 : 0.5}}

onehotencoder = preprocessing.OneHotEncoder(categorical_features=[0,1])

x_train = df.iloc[:,[0,3]].values
x_train = onehotencoder.fit_transform(x_train).toarray()
y_train = df.iloc[:,[1,2]]
y_train = 0.5 * y_train

x_test = dd.iloc[:,[0,3]].values
x_test = onehotencoder.transform(x_test).toarray()

model = Sequential()

print(x_train.shape[1])
model.add(Dense(units=64, activation='relu', input_dim=x_train.shape[1]))
model.add(Dense(units=2, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=32)
model.train_on_batch(x_train, y_train)

#loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
#print("\n%s: %.2f%%" % (model.metrics_names[1], loss_and_metrics[1]*100))
predictions = model.predict(x_test, batch_size=32)
#print(predictions)

de = pd.read_csv("C:/Users/eshve/Desktop/Project/predict.csv",index_col=None, header=0)
de = de.iloc[:,[0,5]]
df = pd.DataFrame(predictions)
dh = de.join(df)
dh.columns.values[2]='chancehome'
dh.columns.values[3]='chanceaway'

d = dq.set_index('Country')['PlayerScore'].to_dict()
dh['joint1'] = dh['home'].map(d)
dh['joint2'] = dh['away'].map(d)
dh['diffhome'] = dh['joint1'].values/(dh['joint1'].values+dh['joint2'].values)
dh['diffaway'] = dh['joint2'].values/(dh['joint1'].values+dh['joint2'].values)

dh['chancehome2'] = dh['chancehome'].values/(dh['chancehome'].values+dh['chanceaway'].values)
dh['chanceaway2'] = dh['chanceaway'].values/(dh['chancehome'].values+dh['chanceaway'].values)

dh['scorehome'] = (dh['chancehome2']*0.90+dh['diffhome']*1.10)/(dh['chancehome2']+dh['diffhome']+dh['chanceaway2']+dh['diffaway'])
dh['scoreaway'] = (dh['chanceaway2']*0.90+dh['diffaway']*1.10)/(dh['chancehome2']+dh['diffhome']+dh['chanceaway2']+dh['diffaway'])
#print(dh.iloc[:,[0,1,2,3,4,5,6,7,8,9]])
print(dh.iloc[:,[0,1,4,5,10,11]])

d = {range(0, 4901): 'L', range(4901, 5100): 'D', range(5100, 10000): 'W'}
dh['schome2'] = (round(dh['scorehome']*10000)).apply(lambda x: next((v for k, v in d.items() if x in k), 0))
dh['scaway2'] = (round(dh['scoreaway']*10000)).apply(lambda x: next((v for k, v in d.items() if x in k), 0))
print(dh.iloc[:,[0,12,13,1]])
#criteria = [dh['scorehome'].between(0.509999, 1), dh['scorehome'].between(0.490001, 0.509999), dh['scorehome'].between(0, 0.490001)]
#values = ['W', 'D', 'L']
#dh['c'] = dh['scorehome'].apply(criteria,values,0)