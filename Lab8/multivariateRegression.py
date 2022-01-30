from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd


x, y = make_regression(n_samples=2000, random_state=14)

regressors = {
    "linear": LinearRegression(fit_intercept = True),
    "polynomial": make_pipeline(PolynomialFeatures(2), LinearRegression()),
    "random_forest": RandomForestRegressor(n_estimators=20),
    "Ridge":  Ridge(alpha=0.5),
    "Lasso": Lasso(alpha=0.5)
}



x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=42, shuffle=True)



df = pd.DataFrame(np.zeros((len(regressors), 3)), index=regressors.keys(), columns=['r2', 'mae', 'mse'])


for regressor in regressors.keys():
    regressors[regressor].fit(x_train, y_train)
    y_pred = regressors[regressor].predict(x_test)
    df.loc[regressor, 'r2'] = r2_score(y_test, y_pred)
    df.loc[regressor, 'mae'] = mean_absolute_error(y_test, y_pred)
    df.loc[regressor, 'mse'] = mean_squared_error(y_test, y_pred)


print(df)
