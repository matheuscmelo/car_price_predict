import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense

base = pd.read_csv('/home/matheus/cars/cars.csv', encoding='ISO-8859-1')[:500000]

base = base.drop(['City', 'Model', 'Vin'], axis=1)

base = base.dropna(how='any', axis=0)

base = base.sample(frac=1)

previsors = base.iloc[:, 1:].values
prices = base.iloc[:, 0].values

le_previsors = LabelEncoder()

needs_fit = [2, 3]

oh_encoder = OneHotEncoder(categorical_features=needs_fit)

for i in needs_fit:
    previsors[:, i] = le_previsors.fit_transform(previsors[:, i])

previsors = oh_encoder.fit_transform(previsors).toarray()

train_previsors = previsors[:int(len(previsors) * 0.8)]
train_prices = prices[:int(len(prices) * 0.8)]
test_previsors = previsors[int(((len(previsors) * 0.8)) + 1):]
test_prices = prices[int(((len(previsors) * 0.8) + 1)):]

regressor = LinearRegression()

regressor.fit(train_previsors, train_prices)

predicts = regressor.predict(test_previsors)

for i in range(len(predicts)):
    print predicts[i], test_prices[i]


with open('results-lr.txt', 'a') as results:
    for i in range(len(test_prices)):
        results.write("{}, {}\n".format(predicts[i], test_prices[i]))
