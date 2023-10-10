from  useful_package import hyperbola,  polynom_3
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


X = np.linspace(1,2,50)
Y_hip = hyperbola(X)
Y_poly = polynom_3(X)
X = X.reshape(-1, 1)

X_test = np.linspace(2,3,15) 
Y_hip_test = hyperbola(X_test)
Y_poly_test = polynom_3(X_test)
X_test = X_test.reshape(-1, 1) 

reg_hyp = RandomForestRegressor()
reg_poly = RandomForestRegressor() 

reg_hyp.fit(X, Y_hip)
reg_poly.fit(X, Y_poly)

error_hyp = mean_squared_error(Y_hip_test, reg_hyp.predict(X_test))
error_poly = mean_squared_error(Y_poly_test, reg_poly.predict(X_test))

print(f'Hyp MSE: {error_hyp}\nPoly MSE: {error_poly}')
