from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import scipy.io as sio
from sklearn.model_selection import train_test_split
from boruta import BorutaPy
import numpy as np
np.set_printoptions(threshold=np.inf)
from sklearn.neural_network import MLPRegressor
import time
time_begin = time.time()

train = np.load()
y = []

def get_data_subset(x_data, columns):
    return x_data[:, columns]

def objective_function_calculation(feature_combination, x_train, x_test, y_train, y_test):
    x_scale, y_scale = StandardScaler(), StandardScaler()
    x_train_scaled = x_scale.fit_transform(x_train)
    x_test_scaled = x_scale.transform(x_test)
    y_train_scaled = y_scale.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scale.transform(y_test.reshape(-1, 1))
    limited_train_data = get_data_subset(x_train_scaled, feature_combination)
    limited_test_data = get_data_subset(x_test_scaled, feature_combination)
    model = MLPRegressor(hidden_layer_sizes=50)
    model.fit(limited_train_data, y_train_scaled.ravel())
    y_test_pred = model.predict(limited_test_data)
    y_test_pred = y_test_pred.reshape(-1, 1)
    y_test_pred = y_scale.inverse_transform(y_test_pred)
    y_test_pred = np.squeeze(y_test_pred)
    return y_test, y_test_pred

feature_combination =[]

X = np.zeros([m1, len(feature_combination)])
for j in range(len(feature_combination)):
    X[:, j] = train[:, feature_combination[j]]

y = np.zeros([m1])
for i in range(m1):
    y[i] = y[i]

forest = RandomForestRegressor(
   n_jobs = -1,
   max_depth = 5,random_state = 42
)

boruta = BorutaPy(
   estimator = forest,
   n_estimators = 'auto',
       max_iter =100,perc =100
)

boruta.fit(np.array(X), np.array(y))

print(boruta.support_)
print(boruta.support_weak_)

a = boruta.support_
s = []
for i in range(len(boruta.support_)):
    if a[i] == False:
        s.append(i)

test = []
pre = []
for i in range(m1):
    if i % m2 == 0:
        x_test = train[i:i + m2, :]
        y_test = y[i:i + m2]
        x_train = np.delete(train, [i,i+1,i+2], axis=0)
        y_train = np.delete(y, [i,i+1,i+2], axis=0)
        costtest, costpred = objective_function_calculation(feature_combination, x_train, x_test, y_train, y_test)
        test.append(costtest)
        pre.append(costpred)

objective_function = 1/(round(mean_squared_error(test, pre), 10) + 1)
best_objective_function = objective_function
best_feature_combination = feature_combination

for i in range(m3):
    feature_combination = []
    all_features = range(2048 - 1)
    not_selected = np.setdiff1d(all_features, feature_combination)
    feature_combination = np.delete(feature_combination, s)
    feature_in = np.random.randint(0, len(not_selected),len(s))
    next_feature_combination = np.insert(feature_combination,s[0],not_selected[feature_in])
    test = []
    pre = []
    for i in range(m1):
        if i % m2 == 0:
            x_test = train[i:i + m2, :]
            y_test = y[i:i + m2]
            x_train = np.delete(train, [i, i + 1, i + 2], axis=0)
            y_train = np.delete(y, [i, i + 1, i + 2], axis=0)
            costtest, costpred = objective_function_calculation(next_feature_combination, x_train, x_test, y_train, y_test)
            test.append(costtest)
            pre.append(costpred)
    next_objective_function = 1/(round(mean_squared_error(test, pre), 10) + 1)

    if next_objective_function > best_objective_function:
        best_objective_function = next_objective_function
        best_feature_combination = next_feature_combination

time_end = time.time()
time = time_end - time_begin
print('time:', time)