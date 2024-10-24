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


def extract_meta_features(X_train, y_train):
    meta_features = {}
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
    return meta_features


meta_features_list = []
performance_list = []

dataset_counter = 1

# Data split ratios
split_ratios = {"train": 0.6, "val": 0.2, "test": 0.2}

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
            X_scaled, y, test_size=split_ratios["test"], random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=split_ratios["val"]
            / (split_ratios["train"] + split_ratios["val"]),
            random_state=42,
        )
        # Now, X_train: 60%, X_val: 20%, X_test: 20%

        # Extract meta-features
        meta_features = extract_meta_features(X_train, y_train)
        meta_features["dataset_name"] = dataset_name

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

            for split_name, X_split, y_split in [
                ("val", X_val, y_val),
                ("test", X_test, y_test),
            ]:
                # Evaluate on split set
                y_pred = model.predict(X_split)

                mse = mean_squared_error(y_split, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_split, y_pred)
                r2 = r2_score(y_split, y_pred)

                performance_list.append(
                    {
                        "dataset_name": dataset_name,
                        "model_name": model_name,
                        f"mae_{split_name}": mae,
                        f"mse_{split_name}": mse,
                        f"rmse_{split_name}": rmse,
                        f"r2_{split_name}": r2,
                    }
                )

                print(f"    - {split_name.capitalize()} MAE: {mae:.4f}")
                print(f"    - {split_name.capitalize()} MSE: {mse:.4f}")
                print(f"    - {split_name.capitalize()} RMSE: {rmse:.4f}")
                print(f"    - {split_name.capitalize()} R2 Score: {r2:.4f}")

                # Log metrics
                mlflow.log_metric(
                    f"{dataset_name}_{model_name}_mae_{split_name}", mae
                )
                mlflow.log_metric(
                    f"{dataset_name}_{model_name}_mse_{split_name}", mse
                )
                mlflow.log_metric(
                    f"{dataset_name}_{model_name}_rmse_{split_name}", rmse
                )
                mlflow.log_metric(
                    f"{dataset_name}_{model_name}_r2_{split_name}", r2
                )

            meta_features_entry = meta_features.copy()
            meta_features_entry["model_name"] = model_name
            meta_features_list.append(meta_features_entry)

            # Save model
            mlflow.sklearn.log_model(
                model, f"{dataset_name}_{model_name}_model"
            )

    print("\nStep 4: Create a meta-dataset for meta-learning.")
    performance_df = (
        pd.DataFrame(performance_list)
        .groupby(["dataset_name", "model_name"])
        .first()
        .reset_index()
    )
    meta_features_df = pd.DataFrame(meta_features_list)
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
        ["dataset_name", "model_name"]
        + [
            col
            for col in meta_dataset.columns
            if "mae_" in col or "mse_" in col or "rmse_" in col or "r2_" in col
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
    y_meta = {}
    for split in ["val", "test"]:
        for metric in ["mae", "mse", "rmse", "r2"]:
            key = f"{metric}_{split}"
            y_meta[key] = meta_dataset[key].values

    groups = meta_dataset["dataset_name"].values

    metric_names = list(y_meta.keys())

    def train_evaluate_meta_models(
        X_meta_scaled, y_meta, groups, metric_names
    ):
        # Initialize dictionaries to store errors
        errors_nn = {metric: [] for metric in metric_names}
        errors_xgb = {metric: [] for metric in metric_names}

        gkf = GroupKFold(n_splits=len(np.unique(groups)))

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
            X_meta_scaled, y_meta[metric_names[0]], groups=groups
        ):
            X_train_meta, X_test_meta = (
                X_meta_scaled[train_idx],
                X_meta_scaled[test_idx],
            )
            X_train_tensor = torch.tensor(X_train_meta, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test_meta, dtype=torch.float32)

            input_size = X_train_meta.shape[1]

            # Initialize models for each metric
            meta_models_nn = {}
            meta_models_xgb = {}
            optimizers = {}
            y_train_tensors = {}
            y_test_values = {}

            for metric in metric_names:
                y_train, y_test = (
                    y_meta[metric][train_idx],
                    y_meta[metric][test_idx],
                )
                y_train_tensor = torch.tensor(
                    y_train, dtype=torch.float32
                ).unsqueeze(1)
                y_test_values[metric] = y_test
                y_train_tensors[metric] = y_train_tensor

                # Neural Network model
                model_nn = MetaModel(input_size)
                optimizer = optim.Adam(model_nn.parameters(), lr=0.001)
                meta_models_nn[metric] = model_nn
                optimizers[metric] = optimizer

                # XGBoost model
                model_xgb = XGBRegressor(random_state=42)
                meta_models_xgb[metric] = model_xgb

            # Train neural network models
            for metric in metric_names:
                model_nn = meta_models_nn[metric]
                optimizer = optimizers[metric]
                y_train_tensor = y_train_tensors[metric]
                model_nn = train_model(
                    model_nn, optimizer, X_train_tensor, y_train_tensor
                )

            # Train XGBoost models
            for metric in metric_names:
                y_train = y_meta[metric][train_idx]
                model_xgb = meta_models_xgb[metric]
                model_xgb.fit(X_train_meta, y_train)

            # Evaluate models on test set
            for metric in metric_names:
                y_test = y_test_values[metric]

                # Neural Network predictions
                model_nn = meta_models_nn[metric]
                model_nn.eval()
                with torch.no_grad():
                    pred_nn = model_nn(X_test_tensor).numpy().flatten()
                error_nn = mean_absolute_error(y_test, pred_nn)
                errors_nn[metric].append(error_nn)

                # XGBoost predictions
                model_xgb = meta_models_xgb[metric]
                pred_xgb = model_xgb.predict(X_test_meta)
                error_xgb = mean_absolute_error(y_test, pred_xgb)
                errors_xgb[metric].append(error_xgb)

        # Compute mean errors
        mean_errors_nn = {
            metric: np.mean(errors_nn[metric]) for metric in metric_names
        }
        mean_errors_xgb = {
            metric: np.mean(errors_xgb[metric]) for metric in metric_names
        }

        return mean_errors_nn, mean_errors_xgb

    mean_errors_nn, mean_errors_xgb = train_evaluate_meta_models(
        X_meta_scaled, y_meta, groups, metric_names
    )

    print("\nMeta-models training completed with cross-validation.")

    # Log evaluation metrics in MLflow
    for metric in metric_names:
        mlflow.log_metric(
            f"mean_{metric}_abs_error_nn", mean_errors_nn[metric]
        )
        mlflow.log_metric(
            f"mean_{metric}_abs_error_xgb", mean_errors_xgb[metric]
        )

    for metric in metric_names:
        print(
            f"\nMean Absolute Error of {metric.replace('_', ' ').upper()} Meta-Model (Neural Network): {mean_errors_nn[metric]:.4f}"
        )
        print(
            f"Mean Absolute Error of {metric.replace('_', ' ').upper()} Meta-Model (XGBoost): {mean_errors_xgb[metric]:.4f}"
        )

    print(
        "\nStep 6: Predict metrics for each dataset and model using the best meta-models."
    )

    # Retrain XGBoost models on the entire meta-dataset
    final_meta_models = {}
    for metric in metric_names:
        xgb_model_final = XGBRegressor(random_state=42)
        xgb_model_final.fit(X_meta_scaled, y_meta[metric])
        final_meta_models[metric] = xgb_model_final
        # Save the final models
        mlflow.sklearn.log_model(
            xgb_model_final, f"final_meta_model_{metric}_xgb"
        )

    # Predict on the meta-dataset
    predictions_df = meta_dataset.copy()
    for metric in metric_names:
        predicted = final_meta_models[metric].predict(X_meta_scaled)
        predictions_df[f"predicted_{metric}"] = predicted

    for idx, row in predictions_df.iterrows():
        print(f"{row['dataset_name']} - {row['model_name']}:")
        for split in ["val", "test"]:
            print(
                f"  Predicted MAE ({split}): {row[f'predicted_mae_{split}']:.4f}"
            )
            print(f"  Actual MAE ({split}): {row[f'mae_{split}']:.4f}")
            print(
                f"  Predicted MSE ({split}): {row[f'predicted_mse_{split}']:.4f}"
            )
            print(f"  Actual MSE ({split}): {row[f'mse_{split}']:.4f}")
            print(
                f"  Predicted RMSE ({split}): {row[f'predicted_rmse_{split}']:.4f}"
            )
            print(f"  Actual RMSE ({split}): {row[f'rmse_{split}']:.4f}")
            print(
                f"  Predicted R2 Score ({split}): {row[f'predicted_r2_{split}']:.4f}"
            )
            print(f"  Actual R2 Score ({split}): {row[f'r2_{split}']:.4f}")

    print(
        "\nStep 7: Compile predictions and compare predicted metrics with actual metrics."
    )

    print(predictions_df.head())

    print("\nStep 8: Evaluate the meta-models' performance.")

    for split in ["val", "test"]:
        for metric in ["mae", "mse", "rmse", "r2"]:
            predictions_df[f"{metric}_abs_error_{split}"] = abs(
                predictions_df[f"predicted_{metric}_{split}"]
                - predictions_df[f"{metric}_{split}"]
            )

        mean_mae_abs_error = predictions_df[f"mae_abs_error_{split}"].mean()
        mean_mse_abs_error = predictions_df[f"mse_abs_error_{split}"].mean()
        mean_rmse_abs_error = predictions_df[f"rmse_abs_error_{split}"].mean()
        mean_r2_abs_error = predictions_df[f"r2_abs_error_{split}"].mean()

        print(
            f"\nMean Absolute Error of MAE Meta-Model ({split} set): {mean_mae_abs_error:.4f}"
        )
        print(
            f"Mean Absolute Error of MSE Meta-Model ({split} set): {mean_mse_abs_error:.4f}"
        )
        print(
            f"Mean Absolute Error of RMSE Meta-Model ({split} set): {mean_rmse_abs_error:.4f}"
        )
        print(
            f"Mean Absolute Error of R2 Meta-Model ({split} set): {mean_r2_abs_error:.4f}"
        )

        # Log evaluation metrics in MLflow
        mlflow.log_metric(
            f"mean_mae_abs_error_{split}_final", mean_mae_abs_error
        )
        mlflow.log_metric(
            f"mean_mse_abs_error_{split}_final", mean_mse_abs_error
        )
        mlflow.log_metric(
            f"mean_rmse_abs_error_{split}_final", mean_rmse_abs_error
        )
        mlflow.log_metric(
            f"mean_r2_abs_error_{split}_final", mean_r2_abs_error
        )

    # Save the predictions DataFrame to a CSV file
    output_file_path = "meta_model_predictions_regression.csv"
    predictions_df.to_csv(output_file_path, index=False)
    print(f"\nPredictions saved to {output_file_path}")

    # Log the CSV file as an artifact in MLflow
    mlflow.log_artifact(output_file_path)
    print(f"Predictions CSV saved and logged to MLflow as an artifact.")

    # Plot and log comparisons between predicted and actual metrics
    for split in ["val", "test"]:
        for metric in ["mae", "mse", "rmse", "r2"]:
            plt.figure(figsize=(12, 8))
            for dataset_name in predictions_df["dataset_name"].unique():
                df_subset = predictions_df[
                    predictions_df["dataset_name"] == dataset_name
                ]
                plt.plot(
                    df_subset["model_name"],
                    df_subset[f"{metric}_{split}"],
                    label=f"{dataset_name} - Actual ({split})",
                    marker="o",
                )
                plt.plot(
                    df_subset["model_name"],
                    df_subset[f"predicted_{metric}_{split}"],
                    "--",
                    marker="x",
                    label=f"{dataset_name} - Predicted ({split})",
                )

            split_ratio = split_ratios.get(split, "Unknown")
            plt.title(
                f"Predicted vs Actual {metric.upper()} on {split.capitalize()} Set ({split_ratio*100:.0f}% of data)"
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

            # Log the plot as an artifact
            mlflow.log_artifact(plot_filename)

            print(
                f"{metric.upper()} comparison plot for {split} set saved and logged to MLflow as an artifact."
            )
