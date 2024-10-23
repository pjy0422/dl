# Import necessary libraries
import warnings

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    fetch_openml,
    load_diabetes,
)
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# Initialize MLflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
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
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
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

        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )

        # Additional target scaling for better performance metrics
        y_train_scaled = y_train
        y_val_scaled = y_val

        # Extract meta-features
        meta_features = {}
        meta_features["dataset_name"] = dataset_name
        meta_features["n_samples"] = X_train.shape[0]
        meta_features["n_features"] = X_train.shape[1]
        meta_features["feature_mean"] = np.mean(X_train)
        meta_features["feature_std"] = np.std(X_train)
        # Add target variable statistics
        meta_features["target_mean"] = np.mean(y_train_scaled)
        meta_features["target_std"] = np.std(y_train_scaled)
        meta_features["target_skewness"] = pd.Series(y_train_scaled).skew()
        meta_features["target_kurtosis"] = pd.Series(y_train_scaled).kurt()

        # Number of outliers in target variable
        q1 = np.percentile(y_train_scaled, 25)
        q3 = np.percentile(y_train_scaled, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = y_train_scaled[
            (y_train_scaled < lower_bound) | (y_train_scaled > upper_bound)
        ]
        meta_features["n_target_outliers"] = len(outliers)

        # Feature skewness and kurtosis
        meta_features["avg_feature_skewness"] = np.mean(
            pd.DataFrame(X_train).skew()
        )
        meta_features["avg_feature_kurtosis"] = np.mean(
            pd.DataFrame(X_train).kurt()
        )

        n_components = min(5, X_train.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        meta_features["pca_explained_variance"] = np.sum(
            pca.explained_variance_ratio_
        )

        # Compute average correlation between features and target
        corr_matrix = np.corrcoef(np.c_[X_train, y_train_scaled].T)
        corr = corr_matrix[-1, :-1]
        meta_features["avg_feature_target_correlation"] = np.mean(np.abs(corr))

        mi = mutual_info_regression(X_train, y_train_scaled, random_state=42)
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

            model.fit(X_train, y_train_scaled)
            y_pred = model.predict(X_val)

            # Calculate performance metrics
            mse = mean_squared_error(y_val_scaled, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val_scaled, y_pred)
            r2 = r2_score(y_val_scaled, y_pred)

            # Normalize RMSE and MAE by target standard deviation
            rmse_normalized = rmse / np.std(y_train_scaled)
            mae_normalized = mae / np.std(y_train_scaled)

            performance_list.append(
                {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "rmse_normalized": rmse_normalized,
                    "mae_normalized": mae_normalized,
                    "r2_score": r2,
                }
            )

            meta_features_entry = meta_features.copy()
            meta_features_entry["model_name"] = model_name
            meta_features_list.append(meta_features_entry)

            print(f"    - Normalized RMSE: {rmse_normalized:.4f}")
            print(f"    - Normalized MAE: {mae_normalized:.4f}")
            print(f"    - R2 Score: {r2:.4f}")

            # Log metrics
            mlflow.log_metric(
                f"{dataset_name}_{model_name}_rmse_normalized", rmse_normalized
            )
            mlflow.log_metric(
                f"{dataset_name}_{model_name}_mae_normalized", mae_normalized
            )
            mlflow.log_metric(f"{dataset_name}_{model_name}_r2_score", r2)

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
        "\nStep 5: Train meta-models to predict Normalized RMSE, Normalized MAE, and R2 Score from meta-features."
    )

    X_meta = meta_dataset.drop(
        [
            "dataset_name",
            "model_name",
            "rmse_normalized",
            "mae_normalized",
            "r2_score",
        ],
        axis=1,
    )

    # Standardize meta-features
    scaler_meta = StandardScaler()
    X_meta_scaled = scaler_meta.fit_transform(X_meta)

    model_encoder = LabelEncoder()
    model_encoded = model_encoder.fit_transform(meta_dataset["model_name"])

    # Add model_encoded to features
    X_meta_scaled = np.hstack([X_meta_scaled, model_encoded.reshape(-1, 1)])

    # Targets
    y_meta_rmse = meta_dataset["rmse_normalized"]
    y_meta_mae = meta_dataset["mae_normalized"]
    y_meta_r2 = meta_dataset["r2_score"]

    # Standardize targets
    scaler_target_rmse = StandardScaler()
    y_meta_rmse_scaled = scaler_target_rmse.fit_transform(
        y_meta_rmse.values.reshape(-1, 1)
    ).ravel()

    scaler_target_mae = StandardScaler()
    y_meta_mae_scaled = scaler_target_mae.fit_transform(
        y_meta_mae.values.reshape(-1, 1)
    ).ravel()

    # R2 score is already bounded between -inf and 1, no need to scale

    meta_model_rmse = GradientBoostingRegressor(random_state=42)
    meta_model_rmse.fit(X_meta_scaled, y_meta_rmse_scaled)

    meta_model_mae = GradientBoostingRegressor(random_state=42)
    meta_model_mae.fit(X_meta_scaled, y_meta_mae_scaled)

    meta_model_r2 = GradientBoostingRegressor(random_state=42)
    meta_model_r2.fit(X_meta_scaled, y_meta_r2)

    print("Meta-models training completed.")

    # Log meta-models in MLflow
    mlflow.sklearn.log_model(meta_model_rmse, "meta_model_rmse_normalized")
    mlflow.sklearn.log_model(meta_model_mae, "meta_model_mae_normalized")
    mlflow.sklearn.log_model(meta_model_r2, "meta_model_r2_score")
    print("Meta-models logged to MLflow.")

    print(
        "\nStep 6: Predict metrics for each dataset and model using the meta-models."
    )
    predictions = []
    for idx, row in meta_dataset.iterrows():
        X_meta_row = row.drop(
            [
                "dataset_name",
                "model_name",
                "rmse_normalized",
                "mae_normalized",
                "r2_score",
            ]
        ).values.reshape(1, -1)

        # Standardize the features
        X_meta_row_scaled = scaler_meta.transform(X_meta_row)

        model_encoded_value = model_encoder.transform([row["model_name"]])
        X_meta_row_scaled = np.hstack(
            [X_meta_row_scaled, model_encoded_value.reshape(-1, 1)]
        )

        # Predict and inverse transform the targets
        predicted_rmse_scaled = meta_model_rmse.predict(X_meta_row_scaled)
        predicted_rmse = scaler_target_rmse.inverse_transform(
            predicted_rmse_scaled.reshape(-1, 1)
        ).ravel()[0]

        predicted_mae_scaled = meta_model_mae.predict(X_meta_row_scaled)
        predicted_mae = scaler_target_mae.inverse_transform(
            predicted_mae_scaled.reshape(-1, 1)
        ).ravel()[0]

        predicted_r2 = meta_model_r2.predict(X_meta_row_scaled)[0]

        predictions.append(
            {
                "dataset_name": row["dataset_name"],
                "model_name": row["model_name"],
                "predicted_rmse_normalized": predicted_rmse,
                "actual_rmse_normalized": row["rmse_normalized"],
                "predicted_mae_normalized": predicted_mae,
                "actual_mae_normalized": row["mae_normalized"],
                "predicted_r2_score": predicted_r2,
                "actual_r2_score": row["r2_score"],
            }
        )

        print(f"{row['dataset_name']} - {row['model_name']}:")
        print(f"  Predicted Normalized RMSE: {predicted_rmse:.4f}")
        print(f"  Actual Normalized RMSE: {row['rmse_normalized']:.4f}")
        print(f"  Predicted Normalized MAE: {predicted_mae:.4f}")
        print(f"  Actual Normalized MAE: {row['mae_normalized']:.4f}")
        print(f"  Predicted R2 Score: {predicted_r2:.4f}")
        print(f"  Actual R2 Score: {row['r2_score']:.4f}")

    print(
        "\nStep 7: Compile predictions and compare predicted metrics with actual metrics."
    )
    predictions_df = pd.DataFrame(predictions)
    print(predictions_df)

    print("\nStep 8: Evaluate the meta-models' performance.")

    predictions_df["rmse_abs_error"] = abs(
        predictions_df["predicted_rmse_normalized"]
        - predictions_df["actual_rmse_normalized"]
    )
    predictions_df["mae_abs_error"] = abs(
        predictions_df["predicted_mae_normalized"]
        - predictions_df["actual_mae_normalized"]
    )
    predictions_df["r2_abs_error"] = abs(
        predictions_df["predicted_r2_score"]
        - predictions_df["actual_r2_score"]
    )

    mean_rmse_abs_error = predictions_df["rmse_abs_error"].mean()
    mean_mae_abs_error = predictions_df["mae_abs_error"].mean()
    mean_r2_abs_error = predictions_df["r2_abs_error"].mean()

    print(
        f"\nMean Absolute Error of Normalized RMSE Meta-Model: {mean_rmse_abs_error:.4f}"
    )
    print(
        f"Mean Absolute Error of Normalized MAE Meta-Model: {mean_mae_abs_error:.4f}"
    )
    print(
        f"Mean Absolute Error of R2 Score Meta-Model: {mean_r2_abs_error:.4f}"
    )

    # Log evaluation metrics in MLflow
    mlflow.log_metric("mean_rmse_abs_error", mean_rmse_abs_error)
    mlflow.log_metric("mean_mae_abs_error", mean_mae_abs_error)
    mlflow.log_metric("mean_r2_abs_error", mean_r2_abs_error)

    # Save the predictions DataFrame to a CSV file
    output_file_path = "meta_model_predictions_regression.csv"
    predictions_df.to_csv(output_file_path, index=False)
    print(f"\nPredictions saved to {output_file_path}")

    # Log the CSV file as an artifact in MLflow
    mlflow.log_artifact(output_file_path)
    print(f"Predictions CSV saved and logged to MLflow as an artifact.")

    # Plot and log comparisons between predicted and actual metrics for each dataset
    for metric in ["rmse_normalized", "mae_normalized", "r2_score"]:
        for dataset_name in predictions_df["dataset_name"].unique():
            plt.figure(figsize=(12, 8))  # Adjusting the figure size

            # Filter the DataFrame for the current dataset
            df_subset = predictions_df[
                predictions_df["dataset_name"] == dataset_name
            ]

            # Increase the line width and marker size for better visibility
            plt.plot(
                df_subset["model_name"],
                df_subset[f"actual_{metric}"],
                label=f"{dataset_name} - Actual",
                linestyle="-",  # Solid line for actual
                marker="o",  # Circle marker for actual
                linewidth=2,  # Thicker line
                markersize=8,  # Larger marker
            )
            plt.plot(
                df_subset["model_name"],
                df_subset[f"predicted_{metric}"],
                label=f"{dataset_name} - Predicted",
                linestyle="--",  # Dashed line for predicted
                marker="x",  # Cross marker for predicted
                linewidth=2,  # Thicker line
                markersize=8,  # Larger marker
            )

            plt.title(
                f"Predicted vs Actual {metric.replace('_', ' ').capitalize()} for {dataset_name}"
            )
            plt.xlabel("Model")
            plt.ylabel(metric.replace("_", " ").capitalize())
            plt.xticks(rotation=45)

            # Adjust the legend to be in the upper right corner
            plt.legend(loc="upper right")

            plt.grid(True)
            plt.tight_layout()

            # Save the plot for each dataset and metric
            plot_filename = f"{dataset_name}_{metric}_comparison_plot.png"
            plt.savefig(plot_filename)
            plt.close()  # Close the figure to free memory

            # Log the plot as an artifact
            mlflow.log_artifact(plot_filename)

            print(
                f"{metric.replace('_', ' ').capitalize()} comparison plot for {dataset_name} saved and logged to MLflow as an artifact."
            )
