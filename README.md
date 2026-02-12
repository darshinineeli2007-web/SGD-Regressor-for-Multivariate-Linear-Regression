# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize the input feature matrix **X** and target variable **y**, then standardize the features using **StandardScaler**.
2. Create an **SGDRegressor** model with appropriate learning parameters.
3. Train the model using the scaled data (**fit the model**).
4. Predict the output values using the trained model and display coefficients, intercept, and predicted values.


## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: 
RegisterNumber:  
*/
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

X = np.array([
    [2, 80, 50],
    [3, 60, 40],
    [5, 90, 70],
    [7, 85, 80],
    [9, 95, 90]
])
y = np.array([50, 45, 70, 80, 95])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sgd_reg = SGDRegressor(max_iter=1000, learning_rate='invscaling', eta0=0.01, random_state=42)
sgd_reg.fit(X_scaled, y)
print("Weights (coefficients):", sgd_reg.coef_)
print("Intercept:", sgd_reg.intercept_)

y_pred = sgd_reg.predict(X_scaled)
print("Predicted values:", y_pred)

```

## Output:
<img width="562" height="60" alt="image" src="https://github.com/user-attachments/assets/30745c2b-619a-4315-98c5-babb3201c475" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
