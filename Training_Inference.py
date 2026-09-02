"""import xgboost as xgb
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os


# Custom evaluation function for XGBoost
def custom_eval_metric(y_pred, dmatrix):
    y_true = dmatrix.get_label().reshape(-1, 3)
    y_pred = y_pred.reshape(-1, 3)
    mae = np.mean(np.abs(y_true - y_pred))
    return 'custom_mae', mae

# Seed for reproducibility
def set_seed(seed_value=42):
    np.random.seed(seed_value)

set_seed()

def train_model_and_optimize(data,train=True, optimize_on_begin=2024, optimize_on_end=2024, train_begin=2020, train_end=2023):
    # One-hot encode the 'season' feature
    data = pd.get_dummies(data, columns=["season"])

    # Filter data based on specified date ranges
    train_data = data[(data["datetime"].dt.year >= train_begin) & (data["datetime"].dt.year <= train_end) & (data["datetime"].dt.year != 2022)]
    test_data = data[(data["datetime"].dt.year >= optimize_on_begin) & (data["datetime"].dt.year <= optimize_on_end)]

    # Clip 'residuals' values
    train_data["residuals"] = train_data["residuals"].clip(lower=20)
    test_data["residuals"] = test_data["residuals"].clip(lower=20)

    # Prepare features and targets
    X_train = train_data.drop(columns=["Fossil Gas", "Fossil Hard coal", "Fossil Brown coal/Lignite", "datetime"])
    y_train = train_data[["Fossil Gas", "Fossil Hard coal", "Fossil Brown coal/Lignite"]]
    X_test = test_data.drop(columns=["Fossil Gas", "Fossil Hard coal", "Fossil Brown coal/Lignite", "datetime"])
    y_test = test_data[["Fossil Gas", "Fossil Hard coal", "Fossil Brown coal/Lignite"]]

    y_train_flat = y_train.values.reshape(-1)
    y_test_flat = y_test.values.reshape(-1)

    # If train flag is False and model exists, load it
    model_path = "optimized_xgboost_model.json"
    if not train and os.path.exists(model_path):
        print("Loading the best model from saved file.")
        optimized_model = xgb.XGBRegressor()
        optimized_model.load_model(model_path)
        return optimized_model

    # Define objective function for Optuna
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-3, 10),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'n_estimators': 1000,
            'random_state': 42,
        }
        model = xgb.XGBRegressor(**params, objective='reg:squarederror', early_stopping_rounds=50)
        model.fit(X_train, y_train_flat, eval_set=[(X_test, y_test_flat)], eval_metric=custom_eval_metric, verbose=False)
        y_pred = model.predict(X_test).reshape(-1, 3)
        return np.sqrt(mean_squared_error(y_test, y_pred))

    # Optimize hyperparameters with Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    # Train the model with the best parameters found
    best_params = study.best_trial.params
    optimized_model = xgb.XGBRegressor(**best_params, objective='reg:squarederror', early_stopping_rounds=50, random_state=42)
    optimized_model.fit(X_train, y_train_flat, eval_set=[(X_test, y_test_flat)], eval_metric=custom_eval_metric, verbose=50)

    # Save model to disk
    optimized_model.save_model(model_path)
    print("Training and optimization completed. Model saved.")

    return optimized_model


def predict_co2_emissions(best_model, X_test):
    # Generate predictions
    predictions_flat = best_model.predict(X_test)
    
    # Reshape predictions back to (n_samples, 3)
    predictions = predictions_flat.reshape(-1, 3)
    
    # Convert predictions to DataFrame
    predictions_df = pd.DataFrame(predictions, columns=["Fossil Gas_Pred", "Fossil Hard coal_Pred", "Fossil Brown coal/Lignite_Pred"])
    predictions_df["datetime"] = X_test.index.values

    # Combine actual and predicted data for comparison
    comparison_df = predictions_df[["Fossil Gas_Pred", "Fossil Hard coal_Pred", "Fossil Brown coal/Lignite_Pred"]]

    # CO2 emission factors (Kg of CO₂/MWh)
    CO2_emission_factors = {
        "Fossil Gas": 185,
        "Fossil Hard coal": 920,
        "Fossil Brown coal/Lignite": 1183
    }


    comparison_df["Total_CO2_Pred"] = (
        comparison_df["Fossil Gas_Pred"] * CO2_emission_factors["Fossil Gas"] +
        comparison_df["Fossil Hard coal_Pred"] * CO2_emission_factors["Fossil Hard coal"] +
        comparison_df["Fossil Brown coal/Lignite_Pred"] * CO2_emission_factors["Fossil Brown coal/Lignite"]
    )

    return comparison_df[["datetime","Total_CO2_Pred"]]

"""