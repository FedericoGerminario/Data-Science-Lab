import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
df = pd.read_csv("development.csv")
df_eval = pd.read_csv("evaluation.csv")

df.index = df['id']
df.sort_index()

#migliorare fornendo un encoding per il tipo di camera
values = ['minimum_nights','latitude', 'longitude', 'availability_365', 'neighbourhood_group', 'neighbourhood', 'room_type', 'calculated_host_listings_count']

df = df[df.name.isna() == False]
#df = df[df.last_review.isna() == False]
df = df.fillna(0)


df1 = df.loc[:, values]
df1 = pd.get_dummies(df1)

x = np.array(df1)
y = np.array(df.loc[:, 'price'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#regressor_parameters={'alpha': [0.1, 0.2, 0.5],
#            'fit_intercept': [True, False]}

regressor_parameters_forest={'n_estimators': [300],
                      'max_features': ['sqrt']}


regressor_parameters_Lasso_Ridge = {'alpha': [0.1, 0.2, 0.5]}
reg_Forest = RandomForestRegressor()
reg_Lasso = Lasso()
reg_Ridge = Ridge()

gridsearch = GridSearchCV(reg_Forest, regressor_parameters_forest, scoring='r2', cv=5)
gridsearch.fit(x, y)

'''
y_pred = gridsearch.predict(x_test)


r2 = r2_score(y_test, y_pred)
print(r2)
'''

print(gridsearch.best_score_)

#df_eval = df_eval[df_eval.last_review.isna() == False]
df_eval = df_eval.fillna(0)

df_eval.index = df_eval['id']

df1_eval = df_eval.loc[:, values]
df1_eval = pd.get_dummies(df1_eval)

missing_columns = set(df1) - set(df1_eval)  # df_test doesn't contain all the neighbourhoods of X, need to add missing columns
for c in missing_columns:
    df1_eval[c] = 0
df1_eval = df1_eval[df1.columns]

x_eval = np.array(df1_eval)
y_eval = gridsearch.predict(x_eval)

df_to_submit = pd.DataFrame({'Id': df_eval.index, 'Predicted': y_eval})
df_to_submit.to_csv("mypredictions.csv", sep=",", index=False)


print('finito')
