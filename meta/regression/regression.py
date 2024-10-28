# Import necessary libraries
import warnings

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import entropy, kurtosis, skew
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

import os
import subprocess

import requests


def run_mlflow_server():
    try:
        # Run the mlflow server in the background
        subprocess.Popen(
            ["mlflow", "server", "--host", "127.0.0.1", "--port", "3060"]
        )
        print("MLflow server started on http://127.0.0.1:3060")
    except Exception as e:
        print(f"Failed to start MLflow server: {e}")


# Call the function to start the MLflow server
def init_mlflow():
    # Initialize MLflow
    run_mlflow_server()
    mlflow.set_tracking_uri(uri="http://127.0.0.1:3060")
    mlflow.set_experiment("Meta-Model-Regression")


print("Step 1: Load and preprocess multiple regression datasets.")

# Initialize lists to store datasets
datasets = []

# 1. Diabetes dataset
# print("Loading Diabetes dataset...")
# try:
#     print("Loading Diabetes dataset...")
#     diabetes = load_diabetes()
#     X_diabetes = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
#     y_diabetes = pd.Series(diabetes.target)

#     datasets.append(("Diabetes", X_diabetes, y_diabetes))
#     print("Diabetes dataset loaded successfully.")
# except Exception as e:
#     print(f"Error loading Diabetes dataset: {e}")

# 2. California Housing dataset
print("Loading California Housing dataset...")
try:
    california = fetch_california_housing(as_frame=True)
    X_ca = california.data
    y_ca = california.target
    datasets.append(("California Housing", X_ca, y_ca))
    print("California Housing dataset loaded successfully.")
except Exception as e:
    print(f"Error loading California Housing dataset: {e}")

# 3. Concrete Compressive Strength dataset
print("Loading Concrete Compressive Strength dataset...")
try:
    concrete_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    concrete_column_names = [
        "Cement",
        "BlastFurnaceSlag",
        "FlyAsh",
        "Water",
        "Superplasticizer",
        "CoarseAggregate",
        "FineAggregate",
        "Age",
        "CompressiveStrength",
    ]
    concrete = pd.read_excel(
        concrete_url, header=0, names=concrete_column_names
    )
    X_concrete = concrete.drop("CompressiveStrength", axis=1)
    y_concrete = concrete["CompressiveStrength"]
    datasets.append(("Concrete Compressive Strength", X_concrete, y_concrete))
    print("Concrete Compressive Strength dataset loaded successfully.")
except Exception as e:
    print(f"Error loading Concrete Compressive Strength dataset: {e}")

# 4. Energy Efficiency dataset
print("Loading Energy Efficiency dataset...")
try:
    energy_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    energy_column_names = [
        "Relative_Compactness",
        "Surface_Area",
        "Wall_Area",
        "Roof_Area",
        "Overall_Height",
        "Orientation",
        "Glazing_Area",
        "Glazing_Area_Distribution",
        "Heating_Load",
        "Cooling_Load",
    ]
    energy = pd.read_excel(energy_url, header=0, names=energy_column_names)
    X_energy = energy.drop(["Heating_Load", "Cooling_Load"], axis=1)
    y_energy = energy[
        "Heating_Load"
    ]  # You can choose "Heating_Load" or "Cooling_Load" based on your target
    datasets.append(("Energy Efficiency", X_energy, y_energy))
    print("Energy Efficiency dataset loaded successfully.")
except Exception as e:
    print(f"Error loading Energy Efficiency dataset: {e}")

# 5. Auto MPG dataset
print("Loading Auto MPG dataset...")
try:
    auto_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    auto_columns = [
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
        "car_name",
    ]
    auto = pd.read_csv(
        auto_url, delim_whitespace=True, names=auto_columns, na_values="?"
    )
    auto.dropna(inplace=True)
    auto = auto.drop("car_name", axis=1)
    # One-hot encode 'origin'
    auto = pd.get_dummies(auto, columns=["origin"], drop_first=True)
    X_auto = auto.drop("mpg", axis=1)
    y_auto = auto["mpg"]
    datasets.append(("Auto MPG", X_auto, y_auto))
    print("Auto MPG dataset loaded and preprocessed successfully.")
except Exception as e:
    print(f"Error loading Auto MPG dataset: {e}")

print("\nStep 2: Define diverse regression models to be used.")
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=50, random_state=42, n_jobs=-1
    ),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Support Vector Regressor": SVR(),
    "K-Nearest Neighbors Regressor": KNeighborsRegressor(
        n_neighbors=5, n_jobs=-1
    ),
    "XGBoost Regressor": XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
}

print(
    "\nStep 3: Preprocess, extract meta-features, and record performance for each dataset and model."
)

meta_features_list = []
performance_list = []

dataset_counter = 1

# Initialize MLflow
init_mlflow()

# Start a single MLflow run for the entire experiment
randomnumber = np.random.randint(0, 1000)
with mlflow.start_run(run_name=f"META_RUN_REG_{randomnumber}"):

    for dataset_name, X, y in datasets:
        print(
            f"\nProcessing {dataset_name} ({dataset_counter}/{len(datasets)})..."
        )
        dataset_counter += 1

        # Handle categorical variables
        if isinstance(X, pd.DataFrame):
            categorical_cols = X.select_dtypes(
                include=["object", "category"]
            ).columns
            if len(categorical_cols) > 0:
                X = pd.get_dummies(
                    X, columns=categorical_cols, drop_first=True
                )

        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # Split data into train, validation, and test sets
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42
        )
        # Now, X_train: 60%, X_val: 20%, X_test: 20%

        meta_features = {}
        meta_features["dataset_name"] = dataset_name
        meta_features["n_samples"] = X_train.shape[0]
        meta_features["n_features"] = X_train.shape[1]
        meta_features["feature_mean"] = np.mean(X_train)
        meta_features["feature_std"] = np.std(X_train)
        meta_features["coeff_variation"] = (
            np.std(X_train) / np.mean(X_train) if np.mean(X_train) != 0 else 0
        )

        n_components = min(5, X_train.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        meta_features["pca_explained_variance"] = np.sum(
            pca.explained_variance_ratio_
        )

        mi = mutual_info_regression(
            X_train, y_train, discrete_features=False, random_state=42
        )
        meta_features["avg_mutual_info"] = np.mean(mi)

        # Compute skewness and kurtosis
        skewness = skew(X_train, axis=0)
        kurtosis_values = kurtosis(X_train, axis=0)
        meta_features["avg_skewness"] = np.mean(skewness)
        meta_features["avg_kurtosis"] = np.mean(kurtosis_values)

        # Compute mean absolute correlation between features
        corr_matrix = np.corrcoef(X_train, rowvar=False)
        # Exclude self-correlation by masking the diagonal
        mask = np.ones(corr_matrix.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        abs_corr = corr_matrix[mask]
        meta_features["mean_abs_correlation"] = np.mean(abs_corr)

        # Number of features with zero variance
        zero_variance_features = np.sum(np.var(X_train, axis=0) == 0)
        meta_features["n_zero_variance_features"] = zero_variance_features

        # Mean and median feature variances
        variances = np.var(X_train, axis=0)
        meta_features["mean_variance"] = np.mean(variances)
        meta_features["median_variance"] = np.median(variances)

        # Mean feature entropy
        feature_entropies = [
            entropy(np.histogram(X_train[:, i], bins=10)[0] + 1e-10)
            for i in range(X_train.shape[1])
        ]
        meta_features["mean_feature_entropy"] = np.mean(feature_entropies)

        for model_name, model in models.items():
            print(f"  Training {model_name}...")

            # Log model parameters
            mlflow.log_param(
                f"{dataset_name}_{model_name}_n_samples", X_train.shape[0]
            )
            mlflow.log_param(
                f"{dataset_name}_{model_name}_n_features", X_train.shape[1]
            )

            model.fit(X_train, y_train)

            # Evaluate on validation set
            y_pred_val = model.predict(X_val)

            mse_val = mean_squared_error(y_val, y_pred_val)
            mae_val = mean_absolute_error(y_val, y_pred_val)
            rmse_val = np.sqrt(mse_val)
            r2_val = r2_score(y_val, y_pred_val)

            # Evaluate on test set
            y_pred_test = model.predict(X_test)

            mse_test = mean_squared_error(y_test, y_pred_test)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            rmse_test = np.sqrt(mse_test)
            r2_test = r2_score(y_test, y_pred_test)

            performance_list.append(
                {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "mse_val": mse_val,
                    "mae_val": mae_val,
                    "rmse_val": rmse_val,
                    "r2_val": r2_val,
                    "mse_test": mse_test,
                    "mae_test": mae_test,
                    "rmse_test": rmse_test,
                    "r2_test": r2_test,
                }
            )

            meta_features_entry = meta_features.copy()
            meta_features_entry["model_name"] = model_name
            meta_features_list.append(meta_features_entry)

            print(f"    - Validation MSE: {mse_val:.4f}")
            print(f"    - Validation MAE: {mae_val:.4f}")
            print(f"    - Validation RMSE: {rmse_val:.4f}")
            print(f"    - Validation R²: {r2_val:.4f}")

            # Log metrics
            mlflow.log_metric(f"{dataset_name}_{model_name}_mse_val", mse_val)
            mlflow.log_metric(f"{dataset_name}_{model_name}_mae_val", mae_val)
            mlflow.log_metric(
                f"{dataset_name}_{model_name}_rmse_val", rmse_val
            )
            mlflow.log_metric(f"{dataset_name}_{model_name}_r2_val", r2_val)

            # Save model
            mlflow.sklearn.log_model(
                model, f"{dataset_name}_{model_name}_model"
            )

    print("\nStep 4: Create a meta-dataset for meta-learning.")
    meta_features_df = pd.DataFrame(meta_features_list)
    performance_df = pd.DataFrame(performance_list)
    meta_dataset = pd.merge(
        meta_features_df, performance_df, on=["dataset_name", "model_name"]
    )

    print("Meta-dataset created:")
    print(meta_dataset.head())
    meta_dataset.to_csv("meta_dataset_regression.csv", index=False)
    print(
        "\nStep 5: Train and evaluate meta-models using cross-validation and different algorithms."
    )

    # Prepare data for meta-models
    X_meta = meta_dataset.drop(
        [
            "dataset_name",
            "model_name",
            "mse_val",
            "mae_val",
            "rmse_val",
            "r2_val",
            "mse_test",
            "mae_test",
            "rmse_test",
            "r2_test",
        ],
        axis=1,
    )

    # Encode model names
    model_encoder = LabelEncoder()
    X_meta["model_encoded"] = model_encoder.fit_transform(
        meta_dataset["model_name"]
    )

    # Handle NaN values in meta-features
    X_meta.fillna(X_meta.mean(), inplace=True)

    # Scale the meta-features
    scaler_meta = StandardScaler()
    X_meta_scaled = scaler_meta.fit_transform(X_meta)

    # Targets for test set metrics
    y_meta_mse_test = meta_dataset["mse_test"].values
    y_meta_mae_test = meta_dataset["mae_test"].values
    y_meta_rmse_test = meta_dataset["rmse_test"].values
    y_meta_r2_test = meta_dataset["r2_test"].values

    # Targets for validation set metrics
    y_meta_mse_val = meta_dataset["mse_val"].values
    y_meta_mae_val = meta_dataset["mae_val"].values
    y_meta_rmse_val = meta_dataset["rmse_val"].values
    y_meta_r2_val = meta_dataset["r2_val"].values

    groups = meta_dataset["dataset_name"].values

    # Define neural network architecture
    class MetaModel(nn.Module):
        def __init__(self, input_size):
            super(MetaModel, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.bn1 = nn.BatchNorm1d(64)
            self.fc2 = nn.Linear(64, 32)
            self.bn2 = nn.BatchNorm1d(32)
            self.fc3 = nn.Linear(32, 1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            x = self.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = self.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    # Training function for neural network
    def train_model(model, optimizer, X, y, num_epochs=500):
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X)
            loss = nn.MSELoss()(outputs, y)
            loss.backward()
            optimizer.step()
        return model

    # Function to train and evaluate meta-models
    def train_and_evaluate_meta_model(
        X_meta_scaled, y_meta, groups, model_type="xgb", metric_name=""
    ):
        errors = []
        gkf = GroupKFold(n_splits=len(np.unique(groups)))

        for train_idx, test_idx in gkf.split(
            X_meta_scaled, y_meta, groups=groups
        ):
            X_train_meta, X_test_meta = (
                X_meta_scaled[train_idx],
                X_meta_scaled[test_idx],
            )
            y_train_meta, y_test_meta = y_meta[train_idx], y_meta[test_idx]

            if model_type == "xgb":
                # Initialize XGBoost regressor
                xgb_model = XGBRegressor(
                    random_state=42, n_jobs=-1, verbosity=0
                )
                # Train XGBoost model
                xgb_model.fit(X_train_meta, y_train_meta)
                # Evaluate XGBoost model on test set
                pred_meta = xgb_model.predict(X_test_meta)
                error = mean_absolute_error(y_test_meta, pred_meta)
                errors.append(error)
            elif model_type == "nn":
                # Convert to tensors
                X_train_tensor = torch.tensor(
                    X_train_meta, dtype=torch.float32
                )
                y_train_tensor = torch.tensor(
                    y_train_meta, dtype=torch.float32
                ).unsqueeze(1)
                X_test_tensor = torch.tensor(X_test_meta, dtype=torch.float32)
                y_test_tensor = torch.tensor(
                    y_test_meta, dtype=torch.float32
                ).unsqueeze(1)

                # Initialize model
                input_size = X_train_meta.shape[1]
                meta_model_nn = MetaModel(input_size)

                # Loss function and optimizer
                optimizer_nn = optim.Adam(meta_model_nn.parameters(), lr=0.001)

                # Train neural network model
                meta_model_nn = train_model(
                    meta_model_nn, optimizer_nn, X_train_tensor, y_train_tensor
                )

                # Evaluate neural network model on test set
                meta_model_nn.eval()
                with torch.no_grad():
                    pred_meta = meta_model_nn(X_test_tensor).numpy().flatten()
                error = mean_absolute_error(y_test_meta, pred_meta)
                errors.append(error)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        mean_abs_error = np.mean(errors)
        print(
            f"Mean Absolute Error of {metric_name} Meta-Model ({model_type.upper()}): {mean_abs_error:.4f}"
        )
        return mean_abs_error

    # Function to train final meta-model
    def train_final_meta_model(X_meta_scaled, y_meta, model_type="xgb"):
        if model_type == "xgb":
            # Initialize XGBoost regressor
            xgb_model_final = XGBRegressor(
                random_state=42, n_jobs=-1, verbosity=0
            )
            # Train XGBoost model
            xgb_model_final.fit(X_meta_scaled, y_meta)
            return xgb_model_final
        elif model_type == "nn":
            # Convert to tensors
            X_tensor = torch.tensor(X_meta_scaled, dtype=torch.float32)
            y_tensor = torch.tensor(y_meta, dtype=torch.float32).unsqueeze(1)
            # Initialize model
            input_size = X_meta_scaled.shape[1]
            meta_model_nn_final = MetaModel(input_size)
            # Loss function and optimizer
            optimizer_nn = optim.Adam(
                meta_model_nn_final.parameters(), lr=0.001
            )
            # Train neural network model
            meta_model_nn_final = train_model(
                meta_model_nn_final, optimizer_nn, X_tensor, y_tensor
            )
            return meta_model_nn_final
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    print("\nTraining and evaluating meta-models for validation metrics.")

    # Train and evaluate meta-models for each regression metric on validation set
    metrics_val = {
        "mse_val": ("MSE Validation", "mse_val"),
        "mae_val": ("MAE Validation", "mae_val"),
        "rmse_val": ("RMSE Validation", "rmse_val"),
        "r2_val": ("R² Validation", "r2_val"),
    }

    meta_models_val = {}
    for metric_key, (metric_name, _) in metrics_val.items():
        mean_abs_error_val = train_and_evaluate_meta_model(
            X_meta_scaled,
            meta_dataset[metric_key].values,
            groups,
            model_type="xgb",
            metric_name=metric_name,
        )
        # Log evaluation metrics in MLflow
        mlflow.log_metric(
            f"mean_abs_error_{metric_key}_xgb", mean_abs_error_val
        )
        meta_models_val[metric_key] = mean_abs_error_val

    print("\nTraining and evaluating meta-models for test metrics.")

    # Train and evaluate meta-models for each regression metric on test set
    metrics_test = {
        "mse_test": ("MSE Test", "mse_test"),
        "mae_test": ("MAE Test", "mae_test"),
        "rmse_test": ("RMSE Test", "rmse_test"),
        "r2_test": ("R² Test", "r2_test"),
    }

    meta_models_test = {}
    for metric_key, (metric_name, _) in metrics_test.items():
        mean_abs_error_test = train_and_evaluate_meta_model(
            X_meta_scaled,
            meta_dataset[metric_key].values,
            groups,
            model_type="xgb",
            metric_name=metric_name,
        )
        # Log evaluation metrics in MLflow
        mlflow.log_metric(
            f"mean_abs_error_{metric_key}_xgb", mean_abs_error_test
        )
        meta_models_test[metric_key] = mean_abs_error_test

    # Train final meta-models on the entire dataset for validation metrics
    final_meta_models_val = {}
    for metric_key in metrics_val.keys():
        final_model = train_final_meta_model(
            X_meta_scaled, meta_dataset[metric_key].values, model_type="xgb"
        )
        final_meta_models_val[metric_key] = final_model
        # Save the final models for validation metrics
        mlflow.sklearn.log_model(
            final_model, f"final_meta_model_{metric_key}_xgb"
        )

    # Train final meta-models on the entire dataset for test metrics
    final_meta_models_test = {}
    for metric_key in metrics_test.keys():
        final_model = train_final_meta_model(
            X_meta_scaled, meta_dataset[metric_key].values, model_type="xgb"
        )
        final_meta_models_test[metric_key] = final_model
        # Save the final models for test metrics
        mlflow.sklearn.log_model(
            final_model, f"final_meta_model_{metric_key}_xgb"
        )

    print("Meta-models logged to MLflow.")

    print(
        "\nStep 6: Predict metrics for each dataset and model using the best meta-models."
    )

    # Predict on the meta-dataset for validation metrics
    predictions_df = meta_dataset.copy()
    for metric_key in metrics_val.keys():
        model = final_meta_models_val[metric_key]
        if isinstance(model, XGBRegressor):
            predictions = model.predict(X_meta_scaled)
        else:
            predictions = model(X_meta_scaled).detach().numpy().flatten()
        predictions_df[f"predicted_{metric_key}"] = predictions

    # Predict on the meta-dataset for test metrics
    for metric_key in metrics_test.keys():
        model = final_meta_models_test[metric_key]
        if isinstance(model, XGBRegressor):
            predictions = model.predict(X_meta_scaled)
        else:
            predictions = model(X_meta_scaled).detach().numpy().flatten()
        predictions_df[f"predicted_{metric_key}"] = predictions

    for idx, row in predictions_df.iterrows():
        print(f"{row['dataset_name']} - {row['model_name']}:")
        for metric_key, (metric_name, actual_key) in {
            **metrics_val,
            **metrics_test,
        }.items():
            predicted = row[f"predicted_{metric_key}"]
            actual = row[actual_key]
            print(f"  Predicted {metric_name}: {predicted:.4f}")
            print(f"  Actual {metric_name}: {actual:.4f}")

    print(
        "\nStep 7: Compile predictions and compare predicted metrics with actual metrics."
    )

    print(
        predictions_df[
            [
                "dataset_name",
                "model_name",
                "predicted_mse_val",
                "mse_val",
                "predicted_mae_val",
                "mae_val",
                "predicted_rmse_val",
                "rmse_val",
                "predicted_r2_val",
                "r2_val",
                "predicted_mse_test",
                "mse_test",
                "predicted_mae_test",
                "mae_test",
                "predicted_rmse_test",
                "rmse_test",
                "predicted_r2_test",
                "r2_test",
            ]
        ]
    )

    print("\nStep 8: Evaluate the meta-models' performance.")

    # Compute absolute errors for validation metrics
    predictions_df["mse_abs_error_val"] = abs(
        predictions_df["predicted_mse_val"] - predictions_df["mse_val"]
    )
    predictions_df["mae_abs_error_val"] = abs(
        predictions_df["predicted_mae_val"] - predictions_df["mae_val"]
    )
    predictions_df["rmse_abs_error_val"] = abs(
        predictions_df["predicted_rmse_val"] - predictions_df["rmse_val"]
    )
    predictions_df["r2_abs_error_val"] = abs(
        predictions_df["predicted_r2_val"] - predictions_df["r2_val"]
    )

    mean_mse_abs_error_val_final = predictions_df["mse_abs_error_val"].mean()
    mean_mae_abs_error_val_final = predictions_df["mae_abs_error_val"].mean()
    mean_rmse_abs_error_val_final = predictions_df["rmse_abs_error_val"].mean()
    mean_r2_abs_error_val_final = predictions_df["r2_abs_error_val"].mean()

    print(
        f"\nMean Absolute Error of MSE Meta-Model on Validation Set: {mean_mse_abs_error_val_final:.4f}"
    )
    print(
        f"Mean Absolute Error of MAE Meta-Model on Validation Set: {mean_mae_abs_error_val_final:.4f}"
    )
    print(
        f"Mean Absolute Error of RMSE Meta-Model on Validation Set: {mean_rmse_abs_error_val_final:.4f}"
    )
    print(
        f"Mean Absolute Error of R² Meta-Model on Validation Set: {mean_r2_abs_error_val_final:.4f}"
    )

    # Log evaluation metrics in MLflow
    mlflow.log_metric(
        "mean_mse_abs_error_val_final", mean_mse_abs_error_val_final
    )
    mlflow.log_metric(
        "mean_mae_abs_error_val_final", mean_mae_abs_error_val_final
    )
    mlflow.log_metric(
        "mean_rmse_abs_error_val_final", mean_rmse_abs_error_val_final
    )
    mlflow.log_metric(
        "mean_r2_abs_error_val_final", mean_r2_abs_error_val_final
    )

    # Compute absolute errors for test metrics
    predictions_df["mse_abs_error_test"] = abs(
        predictions_df["predicted_mse_test"] - predictions_df["mse_test"]
    )
    predictions_df["mae_abs_error_test"] = abs(
        predictions_df["predicted_mae_test"] - predictions_df["mae_test"]
    )
    predictions_df["rmse_abs_error_test"] = abs(
        predictions_df["predicted_rmse_test"] - predictions_df["rmse_test"]
    )
    predictions_df["r2_abs_error_test"] = abs(
        predictions_df["predicted_r2_test"] - predictions_df["r2_test"]
    )

    mean_mse_abs_error_test = predictions_df["mse_abs_error_test"].mean()
    mean_mae_abs_error_test = predictions_df["mae_abs_error_test"].mean()
    mean_rmse_abs_error_test = predictions_df["rmse_abs_error_test"].mean()
    mean_r2_abs_error_test = predictions_df["r2_abs_error_test"].mean()

    print(
        f"\nMean Absolute Error of MSE Meta-Model on Test Set: {mean_mse_abs_error_test:.4f}"
    )
    print(
        f"Mean Absolute Error of MAE Meta-Model on Test Set: {mean_mae_abs_error_test:.4f}"
    )
    print(
        f"Mean Absolute Error of RMSE Meta-Model on Test Set: {mean_rmse_abs_error_test:.4f}"
    )
    print(
        f"Mean Absolute Error of R² Meta-Model on Test Set: {mean_r2_abs_error_test:.4f}"
    )

    # Log evaluation metrics in MLflow
    mlflow.log_metric("mean_mse_abs_error_test_final", mean_mse_abs_error_test)
    mlflow.log_metric("mean_mae_abs_error_test_final", mean_mae_abs_error_test)
    mlflow.log_metric(
        "mean_rmse_abs_error_test_final", mean_rmse_abs_error_test
    )
    mlflow.log_metric("mean_r2_abs_error_test_final", mean_r2_abs_error_test)

    # Save the predictions DataFrame to a CSV file
    output_file_path = "meta_model_predictions_regression.csv"
    predictions_df.to_csv(output_file_path, index=False)
    print(f"\nPredictions saved to {output_file_path}")

    # Log the CSV file as an artifact in MLflow
    mlflow.log_artifact(output_file_path)
    print(f"Predictions CSV saved and logged to MLflow as an artifact.")

    # Function to plot and log comparisons
    def plot_metric_comparison(
        predictions_df, metric, split, data_split_ratio, mlflow
    ):
        plt.figure(figsize=(12, 8))
        for dataset_name in predictions_df["dataset_name"].unique():
            df_subset = predictions_df[
                predictions_df["dataset_name"] == dataset_name
            ]
            plt.plot(
                df_subset["model_name"],
                df_subset[
                    f"{metric}_val" if split == "val" else f"{metric}_test"
                ],
                label=f"{dataset_name} - Actual",
                marker="o",
            )
            plt.plot(
                df_subset["model_name"],
                df_subset[
                    (
                        f"predicted_{metric}_val"
                        if split == "val"
                        else f"predicted_{metric}_test"
                    )
                ],
                "--",
                marker="x",
                label=f"{dataset_name} - Predicted",
            )

        plt.title(
            f"Predicted vs Actual {metric.upper()} on {split.capitalize()} Set ({data_split_ratio})"
        )
        plt.xlabel("Model")
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plot_filename = f"{metric}_{split}_comparison_plot.png"
        plt.savefig(plot_filename)
        plt.close()

        # Log the plot as an artifact
        mlflow.log_artifact(plot_filename)

        print(
            f"{metric.upper()} comparison plot for {split} set saved and logged to MLflow as an artifact."
        )

    data_split_ratio = "Training:60%, Validation:20%, Test:20%"

    # Plot comparisons for validation split
    for metric in ["mse", "mae", "rmse", "r2"]:
        plot_metric_comparison(
            predictions_df,
            metric,
            split="val",
            data_split_ratio=data_split_ratio,
            mlflow=mlflow,
        )

    # Plot comparisons for test split
    for metric in ["mse", "mae", "rmse", "r2"]:
        plot_metric_comparison(
            predictions_df,
            metric,
            split="test",
            data_split_ratio=data_split_ratio,
            mlflow=mlflow,
        )

    print(
        "\nStep 9: Complete. All metrics and models have been logged to MLflow."
    )
