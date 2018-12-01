import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor

base = pd.read_csv('/home/matheus/cars/cars.csv', encoding='ISO-8859-1')

base = base.drop(['City', 'Model'], axis=1)

base = base.dropna(how='any', axis=0)

new_vins = []
for i in base.values:
    new_vins.append(i[4][3:7])

base = base.drop(['Vin'], axis=1)

base.insert(2, 'new_vins', new_vins)

previsors = base.iloc[:, 1:].values
prices = base.iloc[:, 0].values

le_previsors = LabelEncoder()

needs_fit = [1, 3, 4]
oh_encoder = OneHotEncoder(categorical_features=needs_fit)

for i in needs_fit:
    previsors[:, i] = le_previsors.fit_transform(previsors[:, i])



previsors = oh_encoder.fit_transform(previsors).toarray()

train_previsors = previsors[:int(len(previsors) * 0.8)]
train_prices = prices[:int(len(prices) * 0.8)]
test_previsors = previsors[int(((len(previsors) * 0.8)) + 1):]
test_prices = prices[int(((len(previsors) * 0.8) + 1)):]

regressor = Sequential()
regressor.add(Dense(units=int(len(train_previsors[0])), activation='linear', input_dim=len(train_previsors[0])))
regressor.add(Dense(units=int(len(train_previsors[0])), activation='relu'))
regressor.add(Dense(units=int(len(train_previsors[0])), activation='relu'))
regressor.add(Dense(units=int(len(train_previsors[0])), activation='relu'))
regressor.add(Dense(units=int(len(train_previsors[0])), activation='relu'))
regressor.add(Dense(units=1, activation='linear'))

regressor.compile(loss='mean_absolute_percentage_error', optimizer='adam')

h = regressor.fit(train_previsors, train_prices, batch_size=1000, epochs=20, shuffle=True)

previsions = regressor.predict(test_previsors)

with open('dotrainHistoryDict.txt', 'w') as history:
    history.write(json.dumps(h.history))

with open('results.txt', 'a') as results:
    for i in range(len(test_prices)):
        results.write("{}, {}\n".format(test_prices[i], previsions[i][0]))

print(h)