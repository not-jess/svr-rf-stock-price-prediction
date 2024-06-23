import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
# df = pd.read_csv('data/BBCA.csv')
# stock_name = 'BBCA'
# df = pd.read_csv('data/BBRI.csv')
# stock_name = 'BBRI'
# df = pd.read_csv('data/BMRI.csv')
# stock_name = 'BMRI'
# df = pd.read_csv('data/BYAN.csv')
# stock_name = 'BYAN'
df = pd.read_csv('data/TLKM.csv')
stock_name = 'TLKM'

print("Dataset loaded successfully. Here are the first few rows:")
print(df.head())

# Preprocessing the data
print("\nConverting timestamp to datetime format...")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

print("\nDataset after setting the timestamp as index:")
print(df.head())

# Features and target variable
X = df[['open', 'low', 'high', 'volume']].values
y = df['close'].values

# Splitting the dataset into train and test sets
split_date = '2022-12-06'
print(f"\nSplitting the dataset into train and test sets at {split_date}...")
train = df.loc[df.index < split_date]
test = df.loc[df.index >= split_date]

X_train = train[['open', 'low', 'high', 'volume']].values
y_train = train['close'].values
X_test = test[['open', 'low', 'high', 'volume']].values
y_test = test['close'].values

# Standardizing the data
print("\nStandardizing the data...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFirst 5 rows of standardized training data:")
print(X_train[:5])

# Hyperparameter tuning for SVR using GridSearchCV
print("\nHyperparameter tuning for SVR using GridSearchCV...")
parameters = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 0.5, 0.9, 1, 5],
    'gamma': [0.001, 0.01, 0.1, 1]
}

model = GridSearchCV(SVR(), param_grid=parameters, cv=3)
model.fit(X_train, y_train)

print("\nOptimal parameter list:", model.best_params_)
print("Optimal model:", model.best_estimator_)
print("Optimal R2 value:", model.best_score_)

# Evaluation on training data
print("\nEvaluating the model on training data...")
y_train_pred = model.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
print("Train Data Evaluation:")
print("R2 Score:", r2_train)
print("Mean Squared Error:", mse_train)
print("Root Mean Squared Error:", rmse_train)
print("Mean Absolute Error:", mae_train)

# Evaluation on test data
print("\nEvaluating the model on test data...")
y_test_pred = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
print("Test Data Evaluation:")
print("R2 Score:", r2_test)
print("Mean Squared Error:", mse_test)
print("Root Mean Squared Error:", rmse_test)
print("Mean Absolute Error:", mae_test)

# Writing evaluations to a new .txt file
with open(f'{stock_name}_evaluations.txt', 'w') as f:
    f.write("Train Data Evaluation:\n")
    f.write(f"R2 Score: {r2_train}\n")
    f.write(f"Mean Squared Error: {mse_train}\n")
    f.write(f"Root Mean Squared Error: {rmse_train}\n")
    f.write(f"Mean Absolute Error: {mae_train}\n")
    f.write("\n")
    f.write("Test Data Evaluation:\n")
    f.write(f"R2 Score: {r2_test}\n")
    f.write(f"Mean Squared Error: {mse_test}\n")
    f.write(f"Root Mean Squared Error: {rmse_test}\n")
    f.write(f"Mean Absolute Error: {mae_test}\n")

# Plotting the train/test split
print("\nPlotting the train/test split...")
fig, ax = plt.subplots(figsize=(15, 5))
train['close'].plot(ax=ax, label='Training Set')
test['close'].plot(ax=ax, label='Test Set')
ax.axvline(split_date, color='black', linestyle='--')
ax.legend(['Train', 'Test'])
plt.title('Train/Test Split')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

# Plotting the actual vs predicted stock prices for the test set
print("\nPlotting actual vs predicted stock prices for the test set...")
plt.figure(figsize=(20, 10))
plt.plot(test.index, y_test, color='red', label='Actual Data')
plt.plot(test.index, y_test_pred, color='blue', label='Prediction')
plt.title(f'Actual vs Predicted {stock_name} Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price (IDR)')
plt.legend()

# Save the plot to a file with the stock_name as the title
plt.savefig(f'{stock_name}_plot.png')

plt.show()
