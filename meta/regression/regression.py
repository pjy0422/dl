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
from sklearn.datasets import (
    fetch_california_housing,
    fetch_openml,
    load_diabetes,
)
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

# Initialize MLflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:3050")
mlflow.set_experiment("Meta-Model-Regression")

print("Step 1: Load and preprocess multiple regression datasets.")

# Initialize lists to store datasets
datasets = []

# 1. Diabetes dataset
print("Loading Diabetes dataset...")
diabetes = load_diabetes()
X_diabetes = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y_diabetes = pd.Series(diabetes.target)
datasets.append(("Diabetes", X_diabetes, y_diabetes))
print("Diabetes dataset loaded successfully.")

# 2. California Housing dataset
print("Loading California Housing dataset...")
california = fetch_california_housing()
X_california = pd.DataFrame(california.data, columns=california.feature_names)
y_california = pd.Series(california.target)
datasets.append(("California Housing", X_california, y_california))
print("California Housing dataset loaded successfully.")

# 3. Concrete Compressive Strength dataset
print("Loading Concrete Compressive Strength dataset...")
concrete = fetch_openml(name="Concrete_Compressive_Strength", as_frame=True)
X_concrete = concrete.data
y_concrete = concrete.target.astype(float)
datasets.append(("Concrete Compressive Strength", X_concrete, y_concrete))
print("Concrete Compressive Strength dataset loaded successfully.")

# 4. Energy Efficiency dataset
print("Loading Energy Efficiency dataset...")
energy = fetch_openml(name="Energy_efficiency", as_frame=True)
X_energy = energy.data
y_energy = energy.target.astype(float)
datasets.append(("Energy Efficiency", X_energy, y_energy))
print("Energy Efficiency dataset loaded successfully.")

# 5. Auto MPG dataset
print("Loading Auto MPG dataset...")
auto_mpg = fetch_openml(name="autoMpg", as_frame=True)
X_auto_mpg = auto_mpg.data.select_dtypes(include=[np.number]).dropna(axis=1)
y_auto_mpg = auto_mpg.target.astype(float)
datasets.append(("Auto MPG", X_auto_mpg, y_auto_mpg))
print("Auto MPG dataset loaded successfully.")

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
    "XGBoost Regressor": XGBRegressor(random_state=42, n_jobs=-1),
}

print(
    "\nStep 3: Preprocess, extract meta-features, and record performance for each dataset and model."
)

meta_features_list = []
performance_list = []

dataset_counter = 1

# Start a single MLflow run for the entire experiment
randomnumber = np.random.randint(0, 1000)
with mlflow.start_run(run_name=f"META_RUN_{randomnumber}"):

    for dataset_name, X, y in datasets:
        print(
            f"\nProcessing {dataset_name} ({dataset_counter}/{len(datasets)})..."
        )
        dataset_counter += 1

        # Handle missing values
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # Split data into train, validation, and test sets
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25 / 0.9, random_state=42
        )
        # Now, X_train: 60%, X_val: 20%, X_test: 20%

        # Extract meta-features
        meta_features = {}
        meta_features["dataset_name"] = dataset_name
        meta_features["n_samples"] = X_train.shape[0]
        meta_features["n_features"] = X_train.shape[1]
        meta_features["feature_mean"] = np.mean(X_train)
        meta_features["feature_std"] = np.std(X_train)
        meta_features["coeff_variation"] = (
            np.std(X_train) / np.mean(X_train) if np.mean(X_train) != 0 else 0
        )

        # Add target variable statistics
        meta_features["target_mean"] = np.mean(y_train)
        meta_features["target_std"] = np.std(y_train)
        meta_features["target_skewness"] = pd.Series(y_train).skew()
        meta_features["target_kurtosis"] = pd.Series(y_train).kurt()

        # Number of outliers in target variable
        q1 = np.percentile(y_train, 25)
        q3 = np.percentile(y_train, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = y_train[(y_train < lower_bound) | (y_train > upper_bound)]
        meta_features["n_target_outliers"] = len(outliers)

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
        abs_corr = np.abs(corr_matrix[mask])
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

        n_components = min(5, X_train.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        meta_features["pca_explained_variance"] = np.sum(
            pca.explained_variance_ratio_
        )

        # Compute average correlation between features and target
        corr_matrix_full = np.corrcoef(np.c_[X_train, y_train].T)
        corr = corr_matrix_full[-1, :-1]
        meta_features["avg_feature_target_correlation"] = np.mean(np.abs(corr))

        mi = mutual_info_regression(X_train, y_train, random_state=42)
        meta_features["avg_mutual_info"] = np.mean(mi)

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
            rmse_val = np.sqrt(mse_val)
            mae_val = mean_absolute_error(y_val, y_pred_val)
            r2_val = r2_score(y_val, y_pred_val)

            # Evaluate on test set
            y_pred_test = model.predict(X_test)
            mse_test = mean_squared_error(y_test, y_pred_test)
            rmse_test = np.sqrt(mse_test)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            r2_test = r2_score(y_test, y_pred_test)

            performance_list.append(
                {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "mae_val": mae_val,
                    "mse_val": mse_val,
                    "rmse_val": rmse_val,
                    "r2_val": r2_val,
                    "mae_test": mae_test,
                    "mse_test": mse_test,
                    "rmse_test": rmse_test,
                    "r2_test": r2_test,
                }
            )

            meta_features_entry = meta_features.copy()
            meta_features_entry["model_name"] = model_name
            meta_features_list.append(meta_features_entry)

            print(f"    - Validation MAE: {mae_val:.4f}")
            print(f"    - Validation MSE: {mse_val:.4f}")
            print(f"    - Validation RMSE: {rmse_val:.4f}")
            print(f"    - Validation R2 Score: {r2_val:.4f}")

            # Log metrics
            mlflow.log_metric(f"{dataset_name}_{model_name}_mae_val", mae_val)
            mlflow.log_metric(f"{dataset_name}_{model_name}_mse_val", mse_val)
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
            "mae_val",
            "mse_val",
            "rmse_val",
            "r2_val",
            "mae_test",
            "mse_test",
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

    # Targets
    y_meta_mae = meta_dataset["mae_test"].values
    y_meta_mse = meta_dataset["mse_test"].values
    y_meta_rmse = meta_dataset["rmse_test"].values
    y_meta_r2 = meta_dataset["r2_test"].values

    groups = meta_dataset["dataset_name"].values

    gkf = GroupKFold(n_splits=len(np.unique(groups)))

    # Initialize lists to store results
    mae_errors_nn = []
    mse_errors_nn = []
    rmse_errors_nn = []
    r2_errors_nn = []

    mae_errors_xgb = []
    mse_errors_xgb = []
    rmse_errors_xgb = []
    r2_errors_xgb = []

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
            loss = torch.sqrt(nn.MSELoss()(outputs, y))
            loss.backward()
            optimizer.step()
        return model

    for train_idx, test_idx in gkf.split(
        X_meta_scaled, y_meta_mae, groups=groups
    ):
        X_train_meta, X_test_meta = (
            X_meta_scaled[train_idx],
            X_meta_scaled[test_idx],
        )
        y_train_mae, y_test_mae = y_meta_mae[train_idx], y_meta_mae[test_idx]
        y_train_mse, y_test_mse = y_meta_mse[train_idx], y_meta_mse[test_idx]
        y_train_rmse, y_test_rmse = (
            y_meta_rmse[train_idx],
            y_meta_rmse[test_idx],
        )
        y_train_r2, y_test_r2 = y_meta_r2[train_idx], y_meta_r2[test_idx]

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_meta, dtype=torch.float32)
        y_train_mae_tensor = torch.tensor(
            y_train_mae, dtype=torch.float32
        ).unsqueeze(1)
        y_train_mse_tensor = torch.tensor(
            y_train_mse, dtype=torch.float32
        ).unsqueeze(1)
        y_train_rmse_tensor = torch.tensor(
            y_train_rmse, dtype=torch.float32
        ).unsqueeze(1)
        y_train_r2_tensor = torch.tensor(
            y_train_r2, dtype=torch.float32
        ).unsqueeze(1)

        X_test_tensor = torch.tensor(X_test_meta, dtype=torch.float32)
        y_test_mae_tensor = torch.tensor(
            y_test_mae, dtype=torch.float32
        ).unsqueeze(1)
        y_test_mse_tensor = torch.tensor(
            y_test_mse, dtype=torch.float32
        ).unsqueeze(1)
        y_test_rmse_tensor = torch.tensor(
            y_test_rmse, dtype=torch.float32
        ).unsqueeze(1)
        y_test_r2_tensor = torch.tensor(
            y_test_r2, dtype=torch.float32
        ).unsqueeze(1)

        # Initialize models
        input_size = X_train_meta.shape[1]
        meta_model_mae = MetaModel(input_size)
        meta_model_mse = MetaModel(input_size)
        meta_model_rmse = MetaModel(input_size)
        meta_model_r2 = MetaModel(input_size)

        # Loss function and optimizer
        optimizer_mae = optim.Adam(meta_model_mae.parameters(), lr=0.001)
        optimizer_mse = optim.Adam(meta_model_mse.parameters(), lr=0.001)
        optimizer_rmse = optim.Adam(meta_model_rmse.parameters(), lr=0.001)
        optimizer_r2 = optim.Adam(meta_model_r2.parameters(), lr=0.001)

        # Train neural network models
        meta_model_mae = train_model(
            meta_model_mae, optimizer_mae, X_train_tensor, y_train_mae_tensor
        )
        meta_model_mse = train_model(
            meta_model_mse, optimizer_mse, X_train_tensor, y_train_mse_tensor
        )
        meta_model_rmse = train_model(
            meta_model_rmse,
            optimizer_rmse,
            X_train_tensor,
            y_train_rmse_tensor,
        )
        meta_model_r2 = train_model(
            meta_model_r2, optimizer_r2, X_train_tensor, y_train_r2_tensor
        )

        # Evaluate neural network models on test set
        meta_model_mae.eval()
        meta_model_mse.eval()
        meta_model_rmse.eval()
        meta_model_r2.eval()

        with torch.no_grad():
            pred_mae_nn = meta_model_mae(X_test_tensor).numpy().flatten()
            pred_mse_nn = meta_model_mse(X_test_tensor).numpy().flatten()
            pred_rmse_nn = meta_model_rmse(X_test_tensor).numpy().flatten()
            pred_r2_nn = meta_model_r2(X_test_tensor).numpy().flatten()

        mae_error_nn = mean_absolute_error(y_test_mae, pred_mae_nn)
        mse_error_nn = mean_absolute_error(y_test_mse, pred_mse_nn)
        rmse_error_nn = mean_absolute_error(y_test_rmse, pred_rmse_nn)
        r2_error_nn = mean_absolute_error(y_test_r2, pred_r2_nn)

        mae_errors_nn.append(mae_error_nn)
        mse_errors_nn.append(mse_error_nn)
        rmse_errors_nn.append(rmse_error_nn)
        r2_errors_nn.append(r2_error_nn)

        # Initialize XGBoost regressors
        xgb_model_mae = XGBRegressor(random_state=42)
        xgb_model_mse = XGBRegressor(random_state=42)
        xgb_model_rmse = XGBRegressor(random_state=42)
        xgb_model_r2 = XGBRegressor(random_state=42)

        # Train XGBoost models
        xgb_model_mae.fit(X_train_meta, y_train_mae)
        xgb_model_mse.fit(X_train_meta, y_train_mse)
        xgb_model_rmse.fit(X_train_meta, y_train_rmse)
        xgb_model_r2.fit(X_train_meta, y_train_r2)

        # Evaluate XGBoost models on test set
        pred_mae_xgb = xgb_model_mae.predict(X_test_meta)
        pred_mse_xgb = xgb_model_mse.predict(X_test_meta)
        pred_rmse_xgb = xgb_model_rmse.predict(X_test_meta)
        pred_r2_xgb = xgb_model_r2.predict(X_test_meta)

        mae_error_xgb = mean_absolute_error(y_test_mae, pred_mae_xgb)
        mse_error_xgb = mean_absolute_error(y_test_mse, pred_mse_xgb)
        rmse_error_xgb = mean_absolute_error(y_test_rmse, pred_rmse_xgb)
        r2_error_xgb = mean_absolute_error(y_test_r2, pred_r2_xgb)

        mae_errors_xgb.append(mae_error_xgb)
        mse_errors_xgb.append(mse_error_xgb)
        rmse_errors_xgb.append(rmse_error_xgb)
        r2_errors_xgb.append(r2_error_xgb)

    mean_mae_abs_error_nn = np.mean(mae_errors_nn)
    mean_mse_abs_error_nn = np.mean(mse_errors_nn)
    mean_rmse_abs_error_nn = np.mean(rmse_errors_nn)
    mean_r2_abs_error_nn = np.mean(r2_errors_nn)

    mean_mae_abs_error_xgb = np.mean(mae_errors_xgb)
    mean_mse_abs_error_xgb = np.mean(mse_errors_xgb)
    mean_rmse_abs_error_xgb = np.mean(rmse_errors_xgb)
    mean_r2_abs_error_xgb = np.mean(r2_errors_xgb)

    print("\nMeta-models training completed with cross-validation.")

    # Log evaluation metrics in MLflow
    mlflow.log_metric("mean_mae_abs_error_nn", mean_mae_abs_error_nn)
    mlflow.log_metric("mean_mse_abs_error_nn", mean_mse_abs_error_nn)
    mlflow.log_metric("mean_rmse_abs_error_nn", mean_rmse_abs_error_nn)
    mlflow.log_metric("mean_r2_abs_error_nn", mean_r2_abs_error_nn)

    mlflow.log_metric("mean_mae_abs_error_xgb", mean_mae_abs_error_xgb)
    mlflow.log_metric("mean_mse_abs_error_xgb", mean_mse_abs_error_xgb)
    mlflow.log_metric("mean_rmse_abs_error_xgb", mean_rmse_abs_error_xgb)
    mlflow.log_metric("mean_r2_abs_error_xgb", mean_r2_abs_error_xgb)

    print(
        f"\nMean Absolute Error of MAE Meta-Model (Neural Network): {mean_mae_abs_error_nn:.4f}"
    )
    print(
        f"Mean Absolute Error of MSE Meta-Model (Neural Network): {mean_mse_abs_error_nn:.4f}"
    )
    print(
        f"Mean Absolute Error of RMSE Meta-Model (Neural Network): {mean_rmse_abs_error_nn:.4f}"
    )
    print(
        f"Mean Absolute Error of R2 Meta-Model (Neural Network): {mean_r2_abs_error_nn:.4f}"
    )

    print(
        f"\nMean Absolute Error of MAE Meta-Model (XGBoost): {mean_mae_abs_error_xgb:.4f}"
    )
    print(
        f"Mean Absolute Error of MSE Meta-Model (XGBoost): {mean_mse_abs_error_xgb:.4f}"
    )
    print(
        f"Mean Absolute Error of RMSE Meta-Model (XGBoost): {mean_rmse_abs_error_xgb:.4f}"
    )
    print(
        f"Mean Absolute Error of R2 Meta-Model (XGBoost): {mean_r2_abs_error_xgb:.4f}"
    )

    # Log meta-models in MLflow
    mlflow.pytorch.log_model(meta_model_mae, "meta_model_mae_nn")
    mlflow.pytorch.log_model(meta_model_mse, "meta_model_mse_nn")
    mlflow.pytorch.log_model(meta_model_rmse, "meta_model_rmse_nn")
    mlflow.pytorch.log_model(meta_model_r2, "meta_model_r2_nn")

    # Log XGBoost models
    mlflow.sklearn.log_model(xgb_model_mae, "meta_model_mae_xgb")
    mlflow.sklearn.log_model(xgb_model_mse, "meta_model_mse_xgb")
    mlflow.sklearn.log_model(xgb_model_rmse, "meta_model_rmse_xgb")
    mlflow.sklearn.log_model(xgb_model_r2, "meta_model_r2_xgb")

    print("Meta-models logged to MLflow.")

    print(
        "\nStep 6: Predict metrics for each dataset and model using the best meta-models."
    )

    # For simplicity, let's assume XGBoost performed better and use it for predictions
    # Retrain XGBoost models on the entire meta-dataset
    xgb_model_mae_final = XGBRegressor(random_state=42)
    xgb_model_mse_final = XGBRegressor(random_state=42)
    xgb_model_rmse_final = XGBRegressor(random_state=42)
    xgb_model_r2_final = XGBRegressor(random_state=42)

    xgb_model_mae_final.fit(X_meta_scaled, y_meta_mae)
    xgb_model_mse_final.fit(X_meta_scaled, y_meta_mse)
    xgb_model_rmse_final.fit(X_meta_scaled, y_meta_rmse)
    xgb_model_r2_final.fit(X_meta_scaled, y_meta_r2)

    # Save the final models
    mlflow.sklearn.log_model(xgb_model_mae_final, "final_meta_model_mae_xgb")
    mlflow.sklearn.log_model(xgb_model_mse_final, "final_meta_model_mse_xgb")
    mlflow.sklearn.log_model(xgb_model_rmse_final, "final_meta_model_rmse_xgb")
    mlflow.sklearn.log_model(xgb_model_r2_final, "final_meta_model_r2_xgb")

    # Predict on the meta-dataset
    predicted_mae = xgb_model_mae_final.predict(X_meta_scaled)
    predicted_mse = xgb_model_mse_final.predict(X_meta_scaled)
    predicted_rmse = xgb_model_rmse_final.predict(X_meta_scaled)
    predicted_r2 = xgb_model_r2_final.predict(X_meta_scaled)

    predictions_df = meta_dataset.copy()
    predictions_df["predicted_mae"] = predicted_mae
    predictions_df["predicted_mse"] = predicted_mse
    predictions_df["predicted_rmse"] = predicted_rmse
    predictions_df["predicted_r2"] = predicted_r2

    for idx, row in predictions_df.iterrows():
        print(f"{row['dataset_name']} - {row['model_name']}:")
        print(f"  Predicted MAE: {row['predicted_mae']:.4f}")
        print(f"  Actual MAE: {row['mae_test']:.4f}")
        print(f"  Predicted MSE: {row['predicted_mse']:.4f}")
        print(f"  Actual MSE: {row['mse_test']:.4f}")
        print(f"  Predicted RMSE: {row['predicted_rmse']:.4f}")
        print(f"  Actual RMSE: {row['rmse_test']:.4f}")
        print(f"  Predicted R2 Score: {row['predicted_r2']:.4f}")
        print(f"  Actual R2 Score: {row['r2_test']:.4f}")

    print(
        "\nStep 7: Compile predictions and compare predicted metrics with actual metrics."
    )

    print(
        predictions_df[
            [
                "dataset_name",
                "model_name",
                "predicted_mae",
                "mae_test",
                "predicted_mse",
                "mse_test",
                "predicted_rmse",
                "rmse_test",
                "predicted_r2",
                "r2_test",
            ]
        ]
    )

    print("\nStep 8: Evaluate the meta-models' performance.")

    predictions_df["mae_abs_error"] = abs(
        predictions_df["predicted_mae"] - predictions_df["mae_test"]
    )
    predictions_df["mse_abs_error"] = abs(
        predictions_df["predicted_mse"] - predictions_df["mse_test"]
    )
    predictions_df["rmse_abs_error"] = abs(
        predictions_df["predicted_rmse"] - predictions_df["rmse_test"]
    )
    predictions_df["r2_abs_error"] = abs(
        predictions_df["predicted_r2"] - predictions_df["r2_test"]
    )

    mean_mae_abs_error = predictions_df["mae_abs_error"].mean()
    mean_mse_abs_error = predictions_df["mse_abs_error"].mean()
    mean_rmse_abs_error = predictions_df["rmse_abs_error"].mean()
    mean_r2_abs_error = predictions_df["r2_abs_error"].mean()

    print(f"\nMean Absolute Error of MAE Meta-Model: {mean_mae_abs_error:.4f}")
    print(f"Mean Absolute Error of MSE Meta-Model: {mean_mse_abs_error:.4f}")
    print(f"Mean Absolute Error of RMSE Meta-Model: {mean_rmse_abs_error:.4f}")
    print(
        f"Mean Absolute Error of R2 Score Meta-Model: {mean_r2_abs_error:.4f}"
    )

    # Log evaluation metrics in MLflow
    mlflow.log_metric("mean_mae_abs_error_final", mean_mae_abs_error)
    mlflow.log_metric("mean_mse_abs_error_final", mean_mse_abs_error)
    mlflow.log_metric("mean_rmse_abs_error_final", mean_rmse_abs_error)
    mlflow.log_metric("mean_r2_abs_error_final", mean_r2_abs_error)

    # Save the predictions DataFrame to a CSV file
    output_file_path = "meta_model_predictions_regression.csv"
    predictions_df.to_csv(output_file_path, index=False)
    print(f"\nPredictions saved to {output_file_path}")

    # Log the CSV file as an artifact in MLflow
    mlflow.log_artifact(output_file_path)
    print(f"Predictions CSV saved and logged to MLflow as an artifact.")

    # Plot and log comparisons between predicted and actual metrics
    for metric in ["mae", "mse", "rmse", "r2"]:
        plt.figure(figsize=(12, 8))
        for dataset_name in predictions_df["dataset_name"].unique():
            df_subset = predictions_df[
                predictions_df["dataset_name"] == dataset_name
            ]
            plt.plot(
                df_subset["model_name"],
                df_subset[f"{metric}_test"],
                label=f"{dataset_name} - Actual",
                marker="o",
            )
            plt.plot(
                df_subset["model_name"],
                df_subset[f"predicted_{metric}"],
                "--",
                marker="x",
                label=f"{dataset_name} - Predicted",
            )

        plt.title(f"Predicted vs Actual {metric.upper()} on Test Set")
        plt.xlabel("Model")
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plot_filename = f"{metric}_comparison_plot.png"
        plt.savefig(plot_filename)

        # Log the plot as an artifact
        mlflow.log_artifact(plot_filename)

        print(
            f"{metric.upper()} comparison plot saved and logged to MLflow as an artifact."
        )
