import mlflow
import mlflow.sklearn
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from itertools import product

# Step 1: Set up MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Set your MLflow tracking URI
mlflow.set_experiment("random_forest_experiment-3")
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define hyperparameters to try
hyperparameters = {
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "max_leaf_nodes": [10, 50, 100],
    "min_samples_leaf": [1, 2, 5],
    "n_estimators": [50, 100, 200],
    "bootstrap": [True, False],
    "max_features": ["auto", "sqrt", "log2"]
}
# hgghgfghfgfghghj
# Step 4: Train and evaluate models with different hyperparameters
param_combinations = product(*hyperparameters.values())
print(param_combinations)
for i, params in enumerate(param_combinations):
    print(i,"iiiii")
    print(params,"paramssss")
    with mlflow.start_run(run_name=f"model_{i}"):
        # Map 'auto' to its corresponding value
        mapped_params = {key: value if value != 'auto' else None if key == 'max_features' else value 
                         for key, value in zip(hyperparameters.keys(), params)}
        
        # Train the model
        model = RandomForestRegressor(**mapped_params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r_squared = r2_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_params(dict(zip(hyperparameters.keys(), params)))
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R-squared", r_squared)
        
        # Log model artifact
        mlflow.sklearn.log_model(model, "random_forest_model")
