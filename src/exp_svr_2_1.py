def predictionModel(n_days, file_path):
    '''
    This function creates an ML model to predict stock price based on the provided number of days.
    '''

    # importing necessary libraries
    import pandas as pd
    import plotly.graph_objects as go
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.svm import SVR
    from datetime import datetime, timedelta
    import numpy as np

    # Read the CSV file
    df = pd.read_csv(file_path)
    print("Data read successfully!")

    # Convert 'timestamp' to datetime format and sort the data by date
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d')
    df.sort_values('timestamp', inplace=True)
    print("Data preprocessing completed!")

    # Create a 'Days' column as the number of days from the first date
    df['Days'] = (df['timestamp'] - df['timestamp'].min()).dt.days

    # Prepare the feature (X) and target (y) variables
    X = df[['Days']]
    y = df['close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    print("Data split into training and testing sets!")

    # Define the parameter grid for GridSearchCV
    parameters = {
        'C': [0.001, 0.01, 0.1, 1, 100, 1000],
        'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 150, 1000],
        'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5, 8, 40, 100, 1000]
    }

    # Use GridSearchCV to find the best parameters
    gsc = GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid=parameters,
        cv=10,
        scoring='neg_mean_absolute_error',
        verbose=3
    )

    print("Starting GridSearchCV...")
    grid_result = gsc.fit(X_train, y_train)
    print("Grid search completed!")

    # Get the best parameters and create the SVR model
    best_param = grid_result.best_params_
    svr_model = SVR(kernel='rbf', C=best_param['C'], epsilon=best_param['epsilon'], gamma=best_param['gamma'])
    svr_model.fit(X_train, y_train)
    print("Model training completed!")

    # Generate future days for prediction
    last_day = df['Days'].max()
    future_days = np.arange(last_day + 1, last_day + n_days + 1).reshape(-1, 1)

    # Generate future dates for plotting
    last_date = df['timestamp'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, n_days + 1)]

    # Plot the predictions
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=svr_model.predict(future_days),
            mode='lines+markers',
            name='Predicted Close Price'
        )
    )
    fig.update_layout(
        title=f"Predicted Close Price for the next {n_days} days",
        xaxis_title="Date",
        yaxis_title="Close Price"
    )
    print("Plotting completed!")

    return fig

# Example usage:
# fig = predictionModel(30, 'BBCA.csv')
# fig.show()

fig = predictionModel(30, 'data/BBCA.csv')
fig.show()
