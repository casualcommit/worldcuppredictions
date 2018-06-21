from keras.layers import Dense
from keras.models import Sequential
#import numpy as np
#import os
import pandas as pd
import glob
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import defaultdict
d = defaultdict(LabelEncoder)

path ='C:/Users/eshve/Desktop/WorldCupHistorical' # use your path
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()

list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
df = pd.concat(list_)
df = df.iloc[:,[0,2,3,5]]
df = df.apply(lambda x: d[x.name].fit_transform(x))
#print(df.head())
#print(df.tail())

dd = pd.read_csv("C:/Users/eshve/Desktop/Project/predict.csv",index_col=None, header=0)
dd = dd.iloc[:,[0,2,3,5]]
dd = dd.apply(lambda x: d[x.name].fit_transform(x))


#for fname in os.listdir('C:/Users/eshve/Desktop/WorldCupHistorical'):
#    inf_from_every_file = np.genfromtxt('C:/Users/eshve/Desktop/WorldCupHistorical/'+fname, unpack=True, delimiter=',',names=True,dtype=None)
#    #inf_from_every_file = inf_from_every_file[:,5]
#    datasets.append(inf_from_every_file)
##    print(datasets)
#print(datasets)
#
#
#dd = np.genfromtxt("C:/Users/eshve/Desktop/Project/predict.csv", unpack=True, delimiter=",",names=True,dtype=None)
#
onehotencoder = OneHotEncoder(categorical_features=[0,1])

x_train = df.iloc[:,[0,3]].values
x_train = onehotencoder.fit_transform(x_train).toarray()
y_train = df.iloc[:,[1,2]]

x_test = dd.iloc[:,[0,3]].values
x_test = onehotencoder.transform(x_test).toarray()
#y_test = dd.iloc[:,[1,2]]
#print(x_train.shape)
#print(x_test.shape)
#print(y_train)
#print(y_test)
##
##
model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=200))
model.add(Dense(units=2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50, batch_size=32)
model.train_on_batch(x_train, y_train)

#loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
#print("\n%s: %.2f%%" % (model.metrics_names[1], loss_and_metrics[1]*100))
predictions = model.predict(x_test, batch_size=128)
#print(predictions)

de = pd.read_csv("C:/Users/eshve/Desktop/Project/predict.csv",index_col=None, header=0)
de = de.iloc[:,[0,5]]
df = pd.DataFrame(predictions)
dq = de.join(df)
print(dq)