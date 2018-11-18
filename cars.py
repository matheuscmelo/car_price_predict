import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor

base = pd.read_csv('/home/matheus/cars/cars.csv', encoding='ISO-8859-1')

base = base.dropna(how='any', axis=0)[:2000]

previsors = base.iloc[:, 1:].values
prices = base.iloc[:, 0].values

le_previsors = LabelEncoder()

needs_fit = [2, 3, 4, 5, 6]
oh_encoder = OneHotEncoder(categorical_features=needs_fit)

for i in needs_fit:
    previsors[:, i] = le_previsors.fit_transform(previsors[:, i])

previsors = oh_encoder.fit_transform(previsors).toarray()

train_previsors = previsors[:int(len(previsors) * 0.8)]
train_prices = prices[:int(len(prices) * 0.8)]
test_previsors = previsors[int(((len(previsors) * 0.8)) + 1):]
test_prices = prices[int(((len(previsors) * 0.8) + 1)):]

regressor = Sequential()
regressor.add(Dense(units=int(len(train_previsors[0])/2), activation='relu', input_dim=len(train_previsors[0])))
regressor.add(Dense(units=int(len(train_previsors[0])), activation='relu'))
regressor.add(Dense(units=int(len(train_previsors[0])), activation='relu'))
regressor.add(Dense(units=int(len(train_previsors[0])/2), activation='relu'))
regressor.add(Dense(units=1, activation='linear'))

regressor.compile(loss='mean_absolute_error', optimizer='adam',
                  metrics=['mean_absolute_percentage_error'])

h = regressor.fit(train_previsors, train_prices, batch_size=1, epochs=100)

previsions = regressor.predict(test_previsors)