import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def inject_noise(y):
    #Adding gaussian random noise
    return y + np.random.normal(0, 50, size=y.size)



def function1(x):
    return (x * np.sin(x) + 2 * x)

def function2(x):
    return (10*np.sin(x) + x**2)

def function3(x):
    return (np.sign(x) * (x**2 + 300) + 20 * np.sin(x))

tr = 20
n_samples = 100
x = np.linspace(-tr, tr, n_samples)
y1 = function1(x)
y2 = function2(x)
y3 = function3(x)

#Part 6 - Injecting Noise
y1 = inject_noise(y1)
y2 = inject_noise(y2)
y3 = inject_noise(y3)



x_train, x_test, y_train, y_test = train_test_split(
    x, y1, train_size = 0.7, random_state=42, shuffle=True)
y_test = y_test[x_test.argsort()]
x_test.sort()


#Part 3
reg = LinearRegression(fit_intercept=True)
print(x_train.reshape(1, -1))
reg.fit(x_train.reshape(-1, 1), y_train)
y_test_pred_linear = reg.predict(x_test.reshape(-1, 1))

#Polynomial
poly = PolynomialFeatures(2)
reg_poly = make_pipeline(poly, LinearRegression())
reg_poly.fit(x_train.reshape(-1, 1), y_train)
y_test_pred_poly = reg_poly.predict(x_test.reshape(-1, 1))

#RandomForestRegressor
regr = RandomForestRegressor()
regr.fit(x_train.reshape(-1, 1), y_train)
y_test_pred_rfr = regr.predict(x_test.reshape(-1, 1))


#Part 4
#Evaluation through r2
r2 = r2_score(y_test, y_test_pred_rfr)
mae = mean_absolute_error(y_test, y_test_pred_rfr)
mse = mean_squared_error(y_test, y_test_pred_rfr)

#Part 5 (Missing trigonometric functions in the model)







fig, ax = plt.subplots(figsize=(3, 2))
#ax.plot(x_train, y_train)
#ax.plot(x_test, y_test)
ax.plot(x, y1)
#ax.plot(x_test.reshape(-1, 1), y_test_pred_linear)
#ax.plot(x_test.reshape(-1, 1), y_test_pred_poly)
ax.plot(x_test.reshape(-1, 1), y_test_pred_rfr)
plt.show()

print("finito")
