# Step 1: Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Importing the dataset
dataset = pd.read_csv('data/BBCA.csv')

# Inspect the dataset
print(dataset.head())

# Step 3: Selecting the features (open, high, low, volume) and the target (close)
X = dataset[['open', 'high', 'low', 'volume']].values
y = dataset['close'].values

# Reshaping y to a column vector
y = y.reshape(-1, 1)

# Step 4: Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Step 5: Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 6: Training the Support Vector Regression model on the Training set
regressor = SVR(kernel='rbf')
regressor.fit(X_train, y_train.ravel())

# Step 7: Predicting the Test set Results
y_pred = regressor.predict(X_test)
y_pred = y_pred.reshape(-1, 1)  # Reshaping y_pred to a column vector
y_pred = sc_y.inverse_transform(y_pred)

# Step 8: Comparing the Test Set with Predicted Values
df = pd.DataFrame({'Real Values': sc_y.inverse_transform(y_test).flatten(), 'Predicted Values': y_pred.flatten()})
print(df)

# Step 9: Evaluating the model
mae = mean_absolute_error(sc_y.inverse_transform(y_test), y_pred)
mse = mean_squared_error(sc_y.inverse_transform(y_test), y_pred)
r2 = r2_score(sc_y.inverse_transform(y_test), y_pred)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R²) Score: {r2}')

# Step 10: Visualising the SVR results
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), sc_y.inverse_transform(y_test), color='red', label='Real Values')
plt.scatter(range(len(y_test)), y_pred, color='green', label='Predicted Values')
plt.title('SVR Prediction of BCA Stock Prices')
plt.xlabel('Test Set Index')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
