import numpy as np
import pandas as pd
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

def critical():
    model = Sequential()
    state_shape = 12
    model.add(Dense(128, input_dim=state_shape, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.001))
    return model

def level_predict():
    model = Sequential()
    state_shape = 15
    model.add(Dense(128, input_dim=state_shape, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(2,activation="softmax"))
    model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.001))
    return model

#data = pd.read_pickle('level.pkl')
'''X = []
Y = []

for i in data['Observations2'].iteritems():
    X.append(i[1])

for i in data['Level'].iteritems():
    if(i[1] == 1):
        Y.append(0)
    else:
        Y.append(1)

np.array(X)
np.array(Y)
print(X)
print(Y)

model = level_predict()
model.fit(X,Y,epochs=5,shuffle=True)

model.save('levelModel.h5')
model.save_weights('levelWeights.h5')'''

data = pd.read_pickle('critical.pkl')
X = []
Y = []

for i in data['Observations'].iteritems():
    X.append(i[1])

for i in data['Critical'].iteritems():
    Y.append(i[1])

np.array(X)
np.array(Y)
print(X)
print(Y)
model = critical()
model.fit(X,Y,epochs=5,shuffle=True)

model.save('criticalModel.h5')
model.save_weights('criticalWeights.h5')