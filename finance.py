import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

data = pd.read_csv('all_stocks_5yr.csv')
print(data.head())

cl = data[data['Name']=='GOOG'].Close#takes closing data
scl=MinMaxScaler()#scales the data (nomalizes)
cl=cl.reshape(cl.shape[0],1)#cl.shape[0] basically returns the number of rows
#print(cl)
cl=scl.fit_transform(cl)#the normalization takes place on cl
#print(cl)

#Create a function to process the data into 7 day look back slices
#so this function has X=[1,2,3,4,5,6,7] where y=[7]
def processData(data,lb):
    X,Y = [],[]
    for i in range(len(data)-lb-1):
        X.append(data[i:(i+lb),0])
        Y.append(data[(i+lb),0])
    return np.array(X),np.array(Y)
X,y = processData(cl,7)
X_train,X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
y_train,y_test = y[:int(y.shape[0]*0.80)],y[int(y.shape[0]*0.80):]
print(X.shape[0])
print(X_train.shape[0])
print(X_test.shape[0])
print(y_train.shape[0])
print(y_test.shape[0])


#BUILDING THE MODEL
model = Sequential()
model.add(LSTM(256,input_shape=(7,1)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
#Reshape data for (Sample,Timestep,Features)
#print(X_train)
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
#converts entire X_train model(100*7) into 7*1*(3d)
#print(X_train)
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
#Fit model with history to check for overfitting
history = model.fit(X_train,y_train,epochs=300,validation_data=(X_test,y_test),shuffle=False)

#plotting
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train','validation'])
plt.show()

Xt = model.predict(X_test)
plt.plot(scl.inverse_transform(y_test.reshape(-1,1)))#undoes the scaling
plt.plot(scl.inverse_transform(Xt))





