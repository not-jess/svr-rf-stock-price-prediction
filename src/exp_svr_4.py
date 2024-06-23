import csv
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Function to load data
def get_data(filename):
    print(f"Loading data from {filename}...")
    data = pd.read_csv(filename)
    dates = pd.to_datetime(data['timestamp']).map(lambda date: date.toordinal())
    prices = data['close'].values
    print("Data loading complete.")
    return dates, prices

# Function to predict stock prices using SVR
def predict_price(dates, prices, x):
    print("Reshaping dates for model training...")
    # Reshape the dates to a 2D array
    dates = np.reshape(dates, (len(dates), 1))
    
    print("Splitting data into training and testing sets...")
    # Split the data into training and testing sets
    dates_train, dates_test, prices_train, prices_test = train_test_split(dates, prices, test_size=0.2, random_state=42)
    
    print("Initializing and training SVR models...")
    # Initialize and train the SVR models
    svr_lin = SVR(kernel='linear', C=1e3, gamma='auto')
    svr_poly = SVR(kernel='poly', C=1e3, degree=2, gamma='auto')
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    
    svr_lin.fit(dates_train, prices_train)
    svr_poly.fit(dates_train, prices_train)
    svr_rbf.fit(dates_train, prices_train)
    
    print("Predicting prices using the trained models...")
    # Predict prices
    pred_lin_train = svr_lin.predict(dates_train)
    pred_poly_train = svr_poly.predict(dates_train)
    pred_rbf_train = svr_rbf.predict(dates_train)
    
    pred_lin_test = svr_lin.predict(dates_test)
    pred_poly_test = svr_poly.predict(dates_test)
    pred_rbf_test = svr_rbf.predict(dates_test)
    
    print("Plotting the results...")
    # Plot the results
    plt.figure(figsize=(14, 7))
    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates_train, pred_lin_train, color='red', label='Linear Model (train)')
    plt.plot(dates_test, pred_lin_test, color='red', linestyle='--', label='Linear Model (test)')
    plt.plot(dates_train, pred_poly_train, color='green', label='Polynomial Model (train)')
    plt.plot(dates_test, pred_poly_test, color='green', linestyle='--', label='Polynomial Model (test)')
    plt.plot(dates_train, pred_rbf_train, color='blue', label='RBF Model (train)')
    plt.plot(dates_test, pred_rbf_test, color='blue', linestyle='--', label='RBF Model (test)')
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression (SVR)')
    plt.legend()
    plt.show()
    
    print("Evaluating model performance on training data...")
    # Evaluation metrics for training data
    print('Training Data Evaluation:')
    print(f'Linear Kernel MSE: {mean_squared_error(prices_train, pred_lin_train)}')
    print(f'Linear Kernel RMSE: {np.sqrt(mean_squared_error(prices_train, pred_lin_train))}')
    print(f'Linear Kernel MAE: {mean_absolute_error(prices_train, pred_lin_train)}')
    
    print(f'Polynomial Kernel MSE: {mean_squared_error(prices_train, pred_poly_train)}')
    print(f'Polynomial Kernel RMSE: {np.sqrt(mean_squared_error(prices_train, pred_poly_train))}')
    print(f'Polynomial Kernel MAE: {mean_absolute_error(prices_train, pred_poly_train)}')
    
    print(f'RBF Kernel MSE: {mean_squared_error(prices_train, pred_rbf_train)}')
    print(f'RBF Kernel RMSE: {np.sqrt(mean_squared_error(prices_train, pred_rbf_train))}')
    print(f'RBF Kernel MAE: {mean_absolute_error(prices_train, pred_rbf_train)}')
    
    print("\nEvaluating model performance on testing data...")
    # Evaluation metrics for testing data
    print('Testing Data Evaluation:')
    print(f'Linear Kernel MSE: {mean_squared_error(prices_test, pred_lin_test)}')
    print(f'Linear Kernel RMSE: {np.sqrt(mean_squared_error(prices_test, pred_lin_test))}')
    print(f'Linear Kernel MAE: {mean_absolute_error(prices_test, pred_lin_test)}')
    
    print(f'Polynomial Kernel MSE: {mean_squared_error(prices_test, pred_poly_test)}')
    print(f'Polynomial Kernel RMSE: {np.sqrt(mean_squared_error(prices_test, pred_poly_test))}')
    print(f'Polynomial Kernel MAE: {mean_absolute_error(prices_test, pred_poly_test)}')
    
    print(f'RBF Kernel MSE: {mean_squared_error(prices_test, pred_rbf_test)}')
    print(f'RBF Kernel RMSE: {np.sqrt(mean_squared_error(prices_test, pred_rbf_test))}')
    print(f'RBF Kernel MAE: {mean_absolute_error(prices_test, pred_rbf_test)}')
    
    return svr_lin.predict(np.array(x).reshape(-1,1))[0], svr_poly.predict(np.array(x).reshape(-1,1))[0], svr_rbf.predict(np.array(x).reshape(-1,1))[0]

# Example usage
stock_name = 'data/BBCA.csv'
dates, prices = get_data(stock_name)
print("Dates:", dates)
print("Prices:", prices)

print("Making predictions for the next day...")
predicted_price = predict_price(dates, prices, [dates.iloc[-1] + 1])  # Predict the next day's price  # Predict the next day's price
print("The Stock Price Prediction:")
print("Linear Kernel: $", str(predicted_price[0]))
print("Polynomial Kernel: $", str(predicted_price[1]))
print("RBF Kernel: $", str(predicted_price[2]))
