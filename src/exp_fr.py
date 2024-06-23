
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_csv('data/TLKM.csv')
stock_name = 'TLKM'

# Parse the 'timestamp' column as datetime
# Convert 'timestamp' to datetime format and sort the data by date
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d')
df.sort_values('timestamp', inplace=True)


# Set the 'timestamp' column as the index
df.set_index('timestamp', inplace=True)

# Display the first few rows of the dataframe
print(df.head())

# Define the feature columns and target column
features = ['open', 'low', 'high', 'volume']
target = 'close'

# Train/test split
x = df[features]
y = df[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Train/test data plotting
train = df.loc[df.index < '2021-01-01']
test = df.loc[df.index >= '2021-01-01']

fig, ax = plt.subplots(figsize=(15,5))
train[target].plot(ax=ax, label='Training Set')
test[target].plot(ax=ax, label='Test Set')
ax.axvline('2021-01-01', color='black')
ax.legend(['Train', 'Test'])
plt.show()

# Create Random Forest Regressor object 
rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', bootstrap=True)

# Fit RF Regression to the Training set
rf.fit(x_train, y_train)

# Predicting the Test set results
y_pred = rf.predict(x_test)

# Evaluate model's performance on train data
predict_train = rf.predict(x_train)
r2_train = r2_score(y_train, predict_train)
mse_train = mean_squared_error(y_train, predict_train)
rmse_train = mean_squared_error(y_train, predict_train, squared=False)
mae_train = mean_absolute_error(y_train, predict_train)
print("Train Data Evaluation:")
print("R2 Score:", r2_train)
print("Mean Squared Error:", mse_train)
print("Root Mean Squared Error:", rmse_train)
print("Mean Absolute Error:", mae_train)

# Evaluate model's performance on test data
r2_test = r2_score(y_test, y_pred)
mse_test = mean_squared_error(y_test, y_pred)
rmse_test = mean_squared_error(y_test, y_pred, squared=False)
mae_test = mean_absolute_error(y_test, y_pred)
print("Test Data Evaluation:")
print("R2 Score:", r2_test)
print("Mean Squared Error:", mse_test)
print("Root Mean Squared Error:", rmse_test)
print("Mean Absolute Error:", mae_test)

# Predict close price on test OHLV(open, high, low, volume)
test_ohlv = test[features]
y_pred_test = rf.predict(test_ohlv)

# Plotting actual vs predicted close prices
plt.rcParams["figure.figsize"] = (20,10)

plt.plot(test.index, test[target], color='red', label="Actual Data")
plt.plot(test.index, y_pred_test, color='blue', label="Prediction")

plt.title(f'Actual vs Predicted {stock_name} Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price (IDR)')
plt.legend()
plt.show()
