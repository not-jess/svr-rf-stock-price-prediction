def predictionModel(file_path):
    '''
    This function creates an SVR model to predict stock prices using multiple features.
    It splits the data based on a specific date ('2021-01-01') and evaluates the model performance.
    '''

    # Importing necessary libraries
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import matplotlib.pyplot as plt
    from datetime import datetime

    # Read the CSV file
    df = pd.read_csv(file_path)
    print("Data read successfully!")

    # Convert 'timestamp' to datetime format and sort the data by date
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d')
    df.sort_values('timestamp', inplace=True)
    print("Data preprocessing completed!")

    # Set the 'timestamp' column as the index
    df.set_index('timestamp', inplace=True)

    # Define the feature columns and target column
    features = ['open', 'high', 'low']
    target = 'close'

    # Split the data into training and testing sets based on a specific date
    train = df.loc[df.index < '2021-01-01']
    test = df.loc[df.index >= '2021-01-01']

    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]

    print("Data split into training and testing sets!")

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # Define the parameter grid for GridSearchCV
    parameters = {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 1],
        'gamma': [0.001, 0.01, 0.1]
    }

    parameters_faster = {
        'C': [1, 10],  # Reduced from [0.1, 1, 10]
        'epsilon': [0.1, 1],  # Reduced from [0.01, 0.1, 1]
        'gamma': [0.01, 0.1]  # Reduced from [0.001, 0.01, 0.1]
    }

    # Use GridSearchCV to find the best parameters
    gsc = GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid=parameters_faster,
        cv=5,
        scoring='neg_mean_absolute_error',
        verbose=3
    )

    print("Starting GridSearchCV...")
    grid_result = gsc.fit(X_train_scaled, y_train)
    print("Grid search completed!")

    # Get the best parameters and create the SVR model
    best_param = grid_result.best_params_
    svr_model = SVR(kernel='rbf', C=best_param['C'], epsilon=best_param['epsilon'], gamma=best_param['gamma'])
    svr_model.fit(X_train_scaled, y_train)
    print("Model training completed!")

    # Predicting the test set results
    y_pred = svr_model.predict(X_test_scaled)

    # Predicting the test set results
    y_pred = svr_model.predict(X_test)

    # Evaluate model's performance on train data
    predict_train = svr_model.predict(X_train)
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

    # Plotting actual vs predicted close prices
    plt.figure(figsize=(20, 10))
    plt.plot(test.index, test[target], color='red', label="Actual Data")
    plt.plot(test.index, y_pred, color='blue', label="Prediction")
    plt.title('SVR Model - BCA Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

    return

# Example usage:
predictionModel('data/BBCA.csv')
