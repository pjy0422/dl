# Import necessary libraries
import os
import subprocess
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
from sklearn.datasets import load_digits, load_wine
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier
from torchvision import datasets, transforms
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")


# Function to run MLflow server
def run_mlflow_server():
    try:
        # Check if MLflow server is already running
        result = subprocess.run(
            ["lsof", "-i", ":3060"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.stdout:
            print("MLflow server is already running on port 3060.")
        else:
            # Run the mlflow server in the background
            subprocess.Popen(
                ["mlflow", "server", "--host", "127.0.0.1", "--port", "3060"]
            )
            print("MLflow server started on http://127.0.0.1:3060")
    except Exception as e:
        print(f"Failed to start MLflow server: {e}")


# Initialize MLflow
def init_mlflow():
    run_mlflow_server()
    mlflow.set_tracking_uri(uri="http://127.0.0.1:3060")
    mlflow.set_experiment("Meta-Model_Multiclass")


# Call the function to start the MLflow server
init_mlflow()

print(
    "Step 1: Load and preprocess multiple multi-class classification datasets."
)

# Initialize lists to store datasets
datasets_list = []

# 1. Digits dataset
print("Loading Digits dataset...")
digits = load_digits()
# Handle feature names
feature_names = (
    digits.feature_names
    if hasattr(digits, "feature_names")
    else [f"pixel_{i}" for i in range(digits.data.shape[1])]
)
X_digits = pd.DataFrame(digits.data, columns=feature_names)
y_digits = pd.Series(digits.target)
datasets_list.append(("Digits", X_digits, y_digits))
print("Digits dataset loaded successfully.")

# 2. Wine dataset
print("Loading Wine dataset...")
wine = load_wine()
X_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
y_wine = pd.Series(wine.target)
datasets_list.append(("Wine", X_wine, y_wine))
print("Wine dataset loaded successfully.")

# 3. Fashion MNIST dataset
print("Loading Fashion MNIST dataset...")
try:
    transform = transforms.Compose([transforms.ToTensor()])
    fashion_mnist_train = datasets.FashionMNIST(
        root=".", train=True, download=True, transform=transform
    )
    fashion_mnist_test = datasets.FashionMNIST(
        root=".", train=False, download=True, transform=transform
    )

    # Combine train and test datasets
    X_fashion = np.concatenate(
        [fashion_mnist_train.data.numpy(), fashion_mnist_test.data.numpy()],
        axis=0,
    )
    y_fashion = np.concatenate(
        [
            fashion_mnist_train.targets.numpy(),
            fashion_mnist_test.targets.numpy(),
        ],
        axis=0,
    )

    # Flatten images
    X_fashion = X_fashion.reshape(X_fashion.shape[0], -1)
    X_fashion = pd.DataFrame(
        X_fashion, columns=[f"pixel_{i}" for i in range(X_fashion.shape[1])]
    )
    y_fashion = pd.Series(y_fashion)

    # Optionally, sample a subset if dataset is too large
    sample_size = 20000  # Adjust based on available resources
    if X_fashion.shape[0] > sample_size:
        X_fashion = X_fashion.sample(n=sample_size, random_state=42)
        y_fashion = y_fashion[X_fashion.index]

    datasets_list.append(("Fashion MNIST", X_fashion, y_fashion))
    print("Fashion MNIST dataset loaded and preprocessed successfully.")
except Exception as e:
    print(f"Error loading Fashion MNIST dataset: {e}")

print("\nStep 2: Define diverse classification models to be used.")
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=200, solver="lbfgs", n_jobs=-1, multi_class="auto"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=50, random_state=42, n_jobs=-1
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "XGBoost": XGBClassifier(
        use_label_encoder=False, eval_metric="mlogloss", random_state=42
    ),
}

print(
    "\nStep 3: Preprocess, extract meta-features, and record performance for each dataset and model."
)

meta_features_list = []
performance_list = []

dataset_counter = 1

# Start a single MLflow run for the entire experiment
randomnumber = np.random.randint(0, 1000)
with mlflow.start_run(run_name=f"META_RUN_MULTICLASS_{randomnumber}"):

    for dataset_name, X, y in datasets_list:
        print(
            f"\nProcessing {dataset_name} ({dataset_counter}/{len(datasets_list)})..."
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
            X_scaled, y, test_size=0.2, stratify=y, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
        )
        # Now, X_train: 60%, X_val: 20%, X_test: 20%

        meta_features = {}
        meta_features["dataset_name"] = dataset_name
        meta_features["n_samples"] = X_train.shape[0]
        meta_features["n_features"] = X_train.shape[1]

        # Replace class_balance with class_entropy
        class_counts = y_train.value_counts(normalize=True)
        meta_features["class_entropy"] = entropy(class_counts)

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

        mi = mutual_info_classif(
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
            if hasattr(model, "predict_proba"):
                y_proba_val = model.predict_proba(X_val)
            else:
                y_proba_val = model.decision_function(X_val)
                if len(y_proba_val.shape) == 1:
                    y_proba_val = np.vstack([1 - y_proba_val, y_proba_val]).T

            acc_val = accuracy_score(y_val, y_pred_val)
            precision_val = precision_score(
                y_val, y_pred_val, average="macro", zero_division=0
            )
            recall_val = recall_score(
                y_val, y_pred_val, average="macro", zero_division=0
            )
            f1_val = f1_score(
                y_val, y_pred_val, average="macro", zero_division=0
            )
            try:
                if len(np.unique(y_val)) < 2:
                    auc_val = np.nan
                else:
                    auc_val = roc_auc_score(
                        label_binarize(y_val, classes=np.unique(y)),
                        y_proba_val,
                        average="macro",
                        multi_class="ovr",
                    )
            except ValueError:
                auc_val = np.nan

            # Evaluate on test set
            y_pred_test = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_proba_test = model.predict_proba(X_test)
            else:
                y_proba_test = model.decision_function(X_test)
                if len(y_proba_test.shape) == 1:
                    y_proba_test = np.vstack(
                        [1 - y_proba_test, y_proba_test]
                    ).T

            acc_test = accuracy_score(y_test, y_pred_test)
            precision_test = precision_score(
                y_test, y_pred_test, average="macro", zero_division=0
            )
            recall_test = recall_score(
                y_test, y_pred_test, average="macro", zero_division=0
            )
            f1_test = f1_score(
                y_test, y_pred_test, average="macro", zero_division=0
            )
            try:
                if len(np.unique(y_test)) < 2:
                    auc_test = np.nan
                else:
                    auc_test = roc_auc_score(
                        label_binarize(y_test, classes=np.unique(y)),
                        y_proba_test,
                        average="macro",
                        multi_class="ovr",
                    )
            except ValueError:
                auc_test = np.nan

            performance_list.append(
                {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "accuracy_val": acc_val,
                    "precision_val": precision_val,
                    "recall_val": recall_val,
                    "f1_score_val": f1_val,
                    "auc_roc_val": auc_val,
                    "accuracy_test": acc_test,
                    "precision_test": precision_test,
                    "recall_test": recall_test,
                    "f1_score_test": f1_test,
                    "auc_roc_test": auc_test,
                }
            )

            meta_features_entry = meta_features.copy()
            meta_features_entry["model_name"] = model_name
            meta_features_list.append(meta_features_entry)

            print(f"    - Validation Accuracy: {acc_val:.4f}")
            print(f"    - Validation Precision: {precision_val:.4f}")
            print(f"    - Validation Recall: {recall_val:.4f}")
            print(f"    - Validation F1-Score: {f1_val:.4f}")
            print(f"    - Validation AUC-ROC: {auc_val:.4f}")

            # Log metrics
            mlflow.log_metric(
                f"{dataset_name}_{model_name}_accuracy_val", acc_val
            )
            mlflow.log_metric(
                f"{dataset_name}_{model_name}_precision_val", precision_val
            )
            mlflow.log_metric(
                f"{dataset_name}_{model_name}_recall_val", recall_val
            )
            mlflow.log_metric(
                f"{dataset_name}_{model_name}_f1_score_val", f1_val
            )
            mlflow.log_metric(
                f"{dataset_name}_{model_name}_auc_roc_val", auc_val
            )

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
    meta_dataset.to_csv("meta_dataset_multiclass.csv", index=False)
    print(
        "\nStep 5: Train and evaluate meta-models using cross-validation and different algorithms."
    )

    # Prepare data for meta-models
    X_meta = meta_dataset.drop(
        [
            "dataset_name",
            "model_name",
            "accuracy_val",
            "precision_val",
            "recall_val",
            "f1_score_val",
            "auc_roc_val",
            "accuracy_test",
            "precision_test",
            "recall_test",
            "f1_score_test",
            "auc_roc_test",
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
    y_meta_acc_test = meta_dataset["accuracy_test"].values
    y_meta_precision_test = meta_dataset["precision_test"].values
    y_meta_recall_test = meta_dataset["recall_test"].values
    y_meta_f1_test = meta_dataset["f1_score_test"].values
    y_meta_auc_test = (
        meta_dataset["auc_roc_test"]
        .fillna(meta_dataset["auc_roc_test"].mean())
        .values
    )

    # Targets for validation set metrics
    y_meta_acc_val = meta_dataset["accuracy_val"].values
    y_meta_precision_val = meta_dataset["precision_val"].values
    y_meta_recall_val = meta_dataset["recall_val"].values
    y_meta_f1_val = meta_dataset["f1_score_val"].values
    y_meta_auc_val = (
        meta_dataset["auc_roc_val"]
        .fillna(meta_dataset["auc_roc_val"].mean())
        .values
    )

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
            loss = torch.sqrt(nn.MSELoss()(outputs, y))
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
                xgb_model = XGBRegressor(random_state=42)
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
            xgb_model_final = XGBRegressor(random_state=42)
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

    # Training meta-models for validation metrics
    metrics_val = {
        "accuracy_val": y_meta_acc_val,
        "precision_val": y_meta_precision_val,
        "recall_val": y_meta_recall_val,
        "f1_score_val": y_meta_f1_val,
        "auc_roc_val": y_meta_auc_val,
    }

    meta_models_val = {}
    for metric, y_meta in metrics_val.items():
        if "auc" in metric:
            metric_name = "AUC-ROC Validation"
        else:
            metric_name = (
                metric.replace("_val", "").capitalize() + " Validation"
            )

        mean_abs_error = train_and_evaluate_meta_model(
            X_meta_scaled,
            y_meta,
            groups,
            model_type="xgb",
            metric_name=metric_name,
        )
        meta_models_val[metric] = mean_abs_error

        # Log evaluation metrics in MLflow
        mlflow.log_metric(f"mean_abs_error_{metric}_xgb", mean_abs_error)

    print("\nTraining and evaluating meta-models for test metrics.")

    # Training meta-models for test metrics
    metrics_test = {
        "accuracy_test": y_meta_acc_test,
        "precision_test": y_meta_precision_test,
        "recall_test": y_meta_recall_test,
        "f1_score_test": y_meta_f1_test,
        "auc_roc_test": y_meta_auc_test,
    }

    meta_models_test = {}
    for metric, y_meta in metrics_test.items():
        if "auc" in metric:
            metric_name = "AUC-ROC Test"
        else:
            metric_name = metric.replace("_test", "").capitalize() + " Test"

        mean_abs_error = train_and_evaluate_meta_model(
            X_meta_scaled,
            y_meta,
            groups,
            model_type="xgb",
            metric_name=metric_name,
        )
        meta_models_test[metric] = mean_abs_error

        # Log evaluation metrics in MLflow
        mlflow.log_metric(f"mean_abs_error_{metric}_xgb", mean_abs_error)

    # Train final meta-models on the entire dataset for validation metrics
    print("\nTraining final meta-models for validation metrics.")
    final_meta_models_val = {}
    for metric in metrics_val.keys():
        meta_model_final = train_final_meta_model(
            X_meta_scaled, metrics_val[metric], model_type="xgb"
        )
        final_meta_models_val[metric] = meta_model_final
        # Save the final models for validation metrics
        mlflow.sklearn.log_model(
            meta_model_final, f"final_meta_model_{metric}_xgb"
        )

    # Train final meta-models on the entire dataset for test metrics
    print("\nTraining final meta-models for test metrics.")
    final_meta_models_test = {}
    for metric in metrics_test.keys():
        meta_model_final = train_final_meta_model(
            X_meta_scaled, metrics_test[metric], model_type="xgb"
        )
        final_meta_models_test[metric] = meta_model_final
        # Save the final models for test metrics
        mlflow.sklearn.log_model(
            meta_model_final, f"final_meta_model_{metric}_xgb"
        )

    print("Meta-models logged to MLflow.")

    print(
        "\nStep 6: Predict metrics for each dataset and model using the best meta-models."
    )

    # Predict on the meta-dataset for validation metrics
    for metric in metrics_val.keys():
        meta_model_final = final_meta_models_val[metric]
        predictions = meta_model_final.predict(X_meta_scaled)
        predictions_df_col = f"predicted_{metric}"
        meta_dataset[predictions_df_col] = predictions

    # Predict on the meta-dataset for test metrics
    for metric in metrics_test.keys():
        meta_model_final = final_meta_models_test[metric]
        predictions = meta_model_final.predict(X_meta_scaled)
        predictions_df_col = f"predicted_{metric}"
        meta_dataset[predictions_df_col] = predictions

    for idx, row in meta_dataset.iterrows():
        print(f"{row['dataset_name']} - {row['model_name']}:")
        # Validation Metrics
        print(
            f"  Predicted Validation Accuracy: {row['predicted_accuracy_val']:.4f}"
        )
        print(f"  Actual Validation Accuracy: {row['accuracy_val']:.4f}")
        print(
            f"  Predicted Validation Precision: {row['predicted_precision_val']:.4f}"
        )
        print(f"  Actual Validation Precision: {row['precision_val']:.4f}")
        print(
            f"  Predicted Validation Recall: {row['predicted_recall_val']:.4f}"
        )
        print(f"  Actual Validation Recall: {row['recall_val']:.4f}")
        print(
            f"  Predicted Validation F1-Score: {row['predicted_f1_score_val']:.4f}"
        )
        print(f"  Actual Validation F1-Score: {row['f1_score_val']:.4f}")
        print(
            f"  Predicted Validation AUC-ROC: {row['predicted_auc_roc_val']:.4f}"
        )
        print(f"  Actual Validation AUC-ROC: {row['auc_roc_val']:.4f}")
        # Test Metrics
        print(
            f"  Predicted Test Accuracy: {row['predicted_accuracy_test']:.4f}"
        )
        print(f"  Actual Test Accuracy: {row['accuracy_test']:.4f}")
        print(
            f"  Predicted Test Precision: {row['predicted_precision_test']:.4f}"
        )
        print(f"  Actual Test Precision: {row['precision_test']:.4f}")
        print(f"  Predicted Test Recall: {row['predicted_recall_test']:.4f}")
        print(f"  Actual Test Recall: {row['recall_test']:.4f}")
        print(
            f"  Predicted Test F1-Score: {row['predicted_f1_score_test']:.4f}"
        )
        print(f"  Actual Test F1-Score: {row['f1_score_test']:.4f}")
        print(f"  Predicted Test AUC-ROC: {row['predicted_auc_roc_test']:.4f}")
        print(f"  Actual Test AUC-ROC: {row['auc_roc_test']:.4f}")

    print(
        "\nStep 7: Compile predictions and compare predicted metrics with actual metrics."
    )

    print(
        meta_dataset[
            [
                "dataset_name",
                "model_name",
                "predicted_accuracy_val",
                "accuracy_val",
                "predicted_precision_val",
                "precision_val",
                "predicted_recall_val",
                "recall_val",
                "predicted_f1_score_val",
                "f1_score_val",
                "predicted_auc_roc_val",
                "auc_roc_val",
                "predicted_accuracy_test",
                "accuracy_test",
                "predicted_precision_test",
                "precision_test",
                "predicted_recall_test",
                "recall_test",
                "predicted_f1_score_test",
                "f1_score_test",
                "predicted_auc_roc_test",
                "auc_roc_test",
            ]
        ]
    )

    print("\nStep 8: Evaluate the meta-models' performance.")

    # Compute absolute errors for validation metrics
    meta_dataset["acc_abs_error_val"] = abs(
        meta_dataset["predicted_accuracy_val"] - meta_dataset["accuracy_val"]
    )
    meta_dataset["precision_abs_error_val"] = abs(
        meta_dataset["predicted_precision_val"] - meta_dataset["precision_val"]
    )
    meta_dataset["recall_abs_error_val"] = abs(
        meta_dataset["predicted_recall_val"] - meta_dataset["recall_val"]
    )
    meta_dataset["f1_abs_error_val"] = abs(
        meta_dataset["predicted_f1_score_val"] - meta_dataset["f1_score_val"]
    )
    meta_dataset["auc_abs_error_val"] = abs(
        meta_dataset["predicted_auc_roc_val"] - meta_dataset["auc_roc_val"]
    )

    mean_acc_abs_error_val_final = meta_dataset["acc_abs_error_val"].mean()
    mean_precision_abs_error_val_final = meta_dataset[
        "precision_abs_error_val"
    ].mean()
    mean_recall_abs_error_val_final = meta_dataset[
        "recall_abs_error_val"
    ].mean()
    mean_f1_abs_error_val_final = meta_dataset["f1_abs_error_val"].mean()
    mean_auc_abs_error_val_final = meta_dataset["auc_abs_error_val"].mean()

    print(
        f"\nMean Absolute Error of Accuracy Meta-Model on Validation Set: {mean_acc_abs_error_val_final:.4f}"
    )
    print(
        f"Mean Absolute Error of Precision Meta-Model on Validation Set: {mean_precision_abs_error_val_final:.4f}"
    )
    print(
        f"Mean Absolute Error of Recall Meta-Model on Validation Set: {mean_recall_abs_error_val_final:.4f}"
    )
    print(
        f"Mean Absolute Error of F1-Score Meta-Model on Validation Set: {mean_f1_abs_error_val_final:.4f}"
    )
    print(
        f"Mean Absolute Error of AUC-ROC Meta-Model on Validation Set: {mean_auc_abs_error_val_final:.4f}"
    )

    # Log evaluation metrics in MLflow
    mlflow.log_metric(
        "mean_acc_abs_error_val_final", mean_acc_abs_error_val_final
    )
    mlflow.log_metric(
        "mean_precision_abs_error_val_final",
        mean_precision_abs_error_val_final,
    )
    mlflow.log_metric(
        "mean_recall_abs_error_val_final", mean_recall_abs_error_val_final
    )
    mlflow.log_metric(
        "mean_f1_abs_error_val_final", mean_f1_abs_error_val_final
    )
    mlflow.log_metric(
        "mean_auc_abs_error_val_final", mean_auc_abs_error_val_final
    )

    # Compute absolute errors for test metrics
    meta_dataset["acc_abs_error_test"] = abs(
        meta_dataset["predicted_accuracy_test"] - meta_dataset["accuracy_test"]
    )
    meta_dataset["precision_abs_error_test"] = abs(
        meta_dataset["predicted_precision_test"]
        - meta_dataset["precision_test"]
    )
    meta_dataset["recall_abs_error_test"] = abs(
        meta_dataset["predicted_recall_test"] - meta_dataset["recall_test"]
    )
    meta_dataset["f1_abs_error_test"] = abs(
        meta_dataset["predicted_f1_score_test"] - meta_dataset["f1_score_test"]
    )
    meta_dataset["auc_abs_error_test"] = abs(
        meta_dataset["predicted_auc_roc_test"] - meta_dataset["auc_roc_test"]
    )

    mean_acc_abs_error_test = meta_dataset["acc_abs_error_test"].mean()
    mean_precision_abs_error_test = meta_dataset[
        "precision_abs_error_test"
    ].mean()
    mean_recall_abs_error_test = meta_dataset["recall_abs_error_test"].mean()
    mean_f1_abs_error_test = meta_dataset["f1_abs_error_test"].mean()
    mean_auc_abs_error_test = meta_dataset["auc_abs_error_test"].mean()

    print(
        f"\nMean Absolute Error of Accuracy Meta-Model on Test Set: {mean_acc_abs_error_test:.4f}"
    )
    print(
        f"Mean Absolute Error of Precision Meta-Model on Test Set: {mean_precision_abs_error_test:.4f}"
    )
    print(
        f"Mean Absolute Error of Recall Meta-Model on Test Set: {mean_recall_abs_error_test:.4f}"
    )
    print(
        f"Mean Absolute Error of F1-Score Meta-Model on Test Set: {mean_f1_abs_error_test:.4f}"
    )
    print(
        f"Mean Absolute Error of AUC-ROC Meta-Model on Test Set: {mean_auc_abs_error_test:.4f}"
    )

    # Log evaluation metrics in MLflow
    mlflow.log_metric("mean_acc_abs_error_test_final", mean_acc_abs_error_test)
    mlflow.log_metric(
        "mean_precision_abs_error_test_final", mean_precision_abs_error_test
    )
    mlflow.log_metric(
        "mean_recall_abs_error_test_final", mean_recall_abs_error_test
    )
    mlflow.log_metric("mean_f1_abs_error_test_final", mean_f1_abs_error_test)
    mlflow.log_metric("mean_auc_abs_error_test_final", mean_auc_abs_error_test)

    # Save the predictions DataFrame to a CSV file
    output_file_path = "meta_model_predictions_multiclass.csv"
    meta_dataset.to_csv(output_file_path, index=False)
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
                df_subset[f"{metric}_{split}"],
                label=f"{dataset_name} - Actual",
                marker="o",
            )
            plt.plot(
                df_subset["model_name"],
                df_subset[f"predicted_{metric}_{split}"],
                "--",
                marker="x",
                label=f"{dataset_name} - Predicted",
            )

        plt.title(
            f"Predicted vs Actual {metric.replace('_', ' ').capitalize()} on {split.capitalize()} Set ({data_split_ratio})"
        )
        plt.xlabel("Model")
        plt.ylabel(metric.replace("_", " ").capitalize())
        plt.xticks(rotation=45)
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plot_filename = f"{metric}_{split}_comparison_plot_multiclass.png"
        plt.savefig(plot_filename)
        plt.close()

        # Log the plot as an artifact
        mlflow.log_artifact(plot_filename)

        print(
            f"{metric.replace('_', ' ').capitalize()} comparison plot for {split} set saved and logged to MLflow as an artifact."
        )

    data_split_ratio = "Training:60%, Validation:20%, Test:20%"

    # Plot comparisons for validation split
    for metric in ["accuracy", "precision", "recall", "f1_score", "auc_roc"]:
        plot_metric_comparison(
            meta_dataset,
            metric,
            split="val",
            data_split_ratio=data_split_ratio,
            mlflow=mlflow,
        )

    # Plot comparisons for test split
    for metric in ["accuracy", "precision", "recall", "f1_score", "auc_roc"]:
        plot_metric_comparison(
            meta_dataset,
            metric,
            split="test",
            data_split_ratio=data_split_ratio,
            mlflow=mlflow,
        )

print("\nAll steps completed successfully.")
