
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow.models import infer_signature
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ---------------------------------------
# 1. Configure MLflow
# ---------------------------------------

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("California Housing Price Prediction")


# ---------------------------------------
# 2. Load the Dataset
# ---------------------------------------

housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target

print("Dataset loaded successfully!")
print(f"Total samples: {len(X)}")
print(f"Number of features: {X.shape[1]}")


# ---------------------------------------
# 3. Split Data (80% / 20%)
# ---------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")


# ---------------------------------------
# 4. Define Experiment Configurations
# ---------------------------------------

experiments = [
    {"run_name": "Run 1", "max_depth": 3, "learning_rate": 0.1},
    {"run_name": "Run 2", "max_depth": 5, "learning_rate": 0.05},
    {"run_name": "Run 3", "max_depth": 7, "learning_rate": 0.01}
]

results = []


# ---------------------------------------
# 5. Train and Track Experiments
# ---------------------------------------

for config in experiments:

    run_name = config["run_name"]
    max_depth = config["max_depth"]
    learning_rate = config["learning_rate"]

    print(f"\nStarting {run_name}...")

    with mlflow.start_run(run_name=run_name):

        # Create the model
        model = HistGradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=100,
            random_state=42
        )

        # Train
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_val)

        # Calculate validation metrics
        rmse = np.sqrt(
            mean_squared_error(y_val, predictions)
        )
        mae = mean_absolute_error(y_val, predictions)
        r2 = r2_score(y_val, predictions)

        # Log the parameters
        mlflow.log_params({
            "max_depth": max_depth,
            "learning_rate": learning_rate
        })

        # Log the metrics
        mlflow.log_metrics({
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        })

        # Infer the model's input/output signature
        input_example = X_train.head(5)
        signature = infer_signature(
            input_example,
            model.predict(input_example)
        )

        # Log the trained model
        mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        signature=signature,
        input_example=input_example,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
)

        # Save results for comparison
        results.append({
            "Run": run_name,
            "Max Depth": max_depth,
            "Learning Rate": learning_rate,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

        print(f"{run_name} completed!")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R2:   {r2:.4f}")


# ---------------------------------------
# 6. Compare All Experiments
# ---------------------------------------

results_df = pd.DataFrame(results)

print("\nExperiment Comparison:")
print(results_df.to_string(index=False))


# ---------------------------------------
# 7. Select Best Model (Lowest RMSE)
# ---------------------------------------

best_index = results_df["RMSE"].idxmin()
best_run = results_df.loc[best_index]

print("\nBest Model:")
print(f"Run: {best_run['Run']}")
print(f"Max Depth: {best_run['Max Depth']}")
print(f"Learning Rate: {best_run['Learning Rate']}")
print(f"RMSE: {best_run['RMSE']:.4f}")
print(f"MAE: {best_run['MAE']:.4f}")
print(f"R2: {best_run['R2']:.4f}")


# ---------------------------------------
# 8. Save Comparison Results
# ---------------------------------------

results_df.to_csv("results.csv", index=False)

print("\nResults saved to results.csv")
print("All experiments completed successfully!")