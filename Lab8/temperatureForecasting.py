import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


df = pd.read_csv("SummaryofWeather.csv", low_memory=False)


#Part 2
sensor_id = 22508
mask = df['STA'] == sensor_id
df1 = df[mask][['Date', 'MeanTemp']]
df1['Date'] = df1['Date'].astype('datetime64')

#Part4
fig, ax = plt.subplots()
ax.plot(df1['Date'], df1['MeanTemp'])
plt.show()

measurements = df1['MeanTemp'].values


#Part 5
W = 6
df_windows = pd.DataFrame(np.zeros((len(df1['MeanTemp']) - W, W + 1)))
for i in range(len(df1['MeanTemp'])-W):
    df_windows.iloc[i] = measurements[i:i+W+1]



#Part 6
df_windows.index = df1['Date'].values[:len(df_windows)]
mask_1940_4 = df_windows.index.year <= 1944
mask_1945 = df_windows.index.year == 1945

x_train = df_windows[mask_1940_4].iloc[:, -1]
y_train = df_windows[mask_1940_4].iloc[:, : -1]
x_test = df_windows[mask_1945].iloc[:,-1]
y_test = df_windows[mask_1945].iloc[:,:-1]
year_test = df_windows[mask_1945].index

#Part 7
regressors = {
    "linear": LinearRegression(fit_intercept = True),
    "polynomial": make_pipeline(PolynomialFeatures(2), LinearRegression()),
    "random_forest": RandomForestRegressor(n_estimators=20),
    "Ridge":  Ridge(alpha=0.5),
    "Lasso": Lasso(alpha=0.5)
}

df_eval = pd.DataFrame(np.zeros((len(regressors), 3)), index=regressors.keys(), columns=['r2', 'mae', 'mse'])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
best_r2_score = 0

for regressor in regressors.keys():
    regressors[regressor].fit(x_train.reshape(-1, 1), y_train)
    y_pred = regressors[regressor].predict(x_test.reshape(-1, 1))
    df_eval.loc[regressor, 'r2'] = r2_score(y_test, y_pred)
    if df_eval.loc[regressor, 'r2'] > best_r2_score:
        best_r2_score = df_eval.loc[regressor, 'r2']
        best_y_pred = y_pred
    df_eval.loc[regressor, 'mae'] = mean_absolute_error(y_test, y_pred)
    df_eval.loc[regressor, 'mse'] = mean_squared_error(y_test, y_pred)

#Part 8

fig, ax = plt.subplots(figsize=(5,4))
ax.plot(year_test,y_test, c='r')
ax.plot(year_test,best_y_pred, c='b')
plt.show()

print('Finito')
