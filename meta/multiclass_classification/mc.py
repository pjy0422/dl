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
from sklearn.datasets import fetch_openml, load_digits, load_wine
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
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

# Initialize MLflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:3050")
mlflow.set_experiment("Meta-Model-MultiClass")

print("Step 1: Load and preprocess multiple multi-class datasets.")

# Initialize lists to store datasets
datasets = []

# 1. Digits dataset
print("Loading Digits dataset...")
digits = load_digits()
X_digits = pd.DataFrame(digits.data)
y_digits = pd.Series(digits.target)
datasets.append(("Digits", X_digits, y_digits))
print("Digits dataset loaded successfully.")

# 2. Wine dataset
print("Loading Wine dataset...")
wine = load_wine()
X_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
y_wine = pd.Series(wine.target)
datasets.append(("Wine", X_wine, y_wine))
print("Wine dataset loaded successfully.")

# 3. Fashion MNIST dataset (from OpenML)
print("Loading Fashion MNIST dataset...")
try:
    fashion_mnist = fetch_openml("Fashion-MNIST", version=1)
    X_fashion = pd.DataFrame(fashion_mnist.data)
    y_fashion = pd.Series(fashion_mnist.target).astype(int)
    # Sample a subset to reduce computation time
    X_fashion_sampled = X_fashion.sample(n=10000, random_state=42)
    y_fashion_sampled = y_fashion.loc[X_fashion_sampled.index]
    datasets.append(("Fashion MNIST", X_fashion_sampled, y_fashion_sampled))
    print("Fashion MNIST dataset loaded and sampled successfully.")
except Exception as e:
    print(f"Failed to load Fashion MNIST dataset: {e}")

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


def extract_meta_features(X_train, y_train):
    meta_features = {}
    meta_features["n_samples"] = X_train.shape[0]
    meta_features["n_features"] = X_train.shape[1]
    class_counts = np.bincount(y_train)
    meta_features["n_classes"] = len(class_counts)
    class_balance = class_counts / class_counts.sum()
    meta_features["class_balance"] = class_balance.tolist()
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
    skewness = skew(X_train, axis=0)
    kurtosis_values = kurtosis(X_train, axis=0)
    meta_features["avg_skewness"] = np.mean(skewness)
    meta_features["avg_kurtosis"] = np.mean(kurtosis_values)
    corr_matrix = np.corrcoef(X_train, rowvar=False)
    mask = np.ones(corr_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    abs_corr = np.abs(corr_matrix[mask])
    meta_features["mean_abs_correlation"] = np.mean(abs_corr)
    zero_variance_features = np.sum(np.var(X_train, axis=0) == 0)
    meta_features["n_zero_variance_features"] = zero_variance_features
    variances = np.var(X_train, axis=0)
    meta_features["mean_variance"] = np.mean(variances)
    meta_features["median_variance"] = np.median(variances)
    feature_entropies = [
        entropy(np.histogram(X_train[:, i], bins=10)[0] + 1e-10)
        for i in range(X_train.shape[1])
    ]
    meta_features["mean_feature_entropy"] = np.mean(feature_entropies)
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
            X_scaled,
            y,
            test_size=split_ratios["test"],
            stratify=y,
            random_state=42,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=split_ratios["val"]
            / (split_ratios["train"] + split_ratios["val"]),
            stratify=y_temp,
            random_state=42,
        )
        # Now, X_train: 60%, X_val: 20%, X_test: 20%

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
                y_pred = model.predict(X_split)
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_split)
                else:
                    # For models without predict_proba, use decision_function and apply softmax
                    decision_vals = model.decision_function(X_split)
                    y_proba = np.exp(decision_vals) / np.sum(
                        np.exp(decision_vals), axis=1, keepdims=True
                    )

                # Binarize the labels for multiclass metrics
                lb = LabelBinarizer()
                lb.fit(y_train)
                y_binarized = lb.transform(y_split)

                acc = accuracy_score(y_split, y_pred)
                precision = precision_score(
                    y_split, y_pred, average="macro", zero_division=0
                )
                recall = recall_score(
                    y_split, y_pred, average="macro", zero_division=0
                )
                f1 = f1_score(
                    y_split, y_pred, average="macro", zero_division=0
                )
                try:
                    auc = roc_auc_score(
                        y_binarized,
                        y_proba,
                        average="macro",
                        multi_class="ovr",
                    )
                except ValueError:
                    auc = np.nan

                performance_list.append(
                    {
                        "dataset_name": dataset_name,
                        "model_name": model_name,
                        f"accuracy_{split_name}": acc,
                        f"precision_{split_name}": precision,
                        f"recall_{split_name}": recall,
                        f"f1_score_{split_name}": f1,
                        f"auc_roc_{split_name}": auc,
                    }
                )

                print(f"    - {split_name.capitalize()} Accuracy: {acc:.4f}")
                print(
                    f"    - {split_name.capitalize()} Precision: {precision:.4f}"
                )
                print(f"    - {split_name.capitalize()} Recall: {recall:.4f}")
                print(f"    - {split_name.capitalize()} F1-Score: {f1:.4f}")
                print(f"    - {split_name.capitalize()} AUC-ROC: {auc:.4f}")

                # Log metrics
                mlflow.log_metric(
                    f"{dataset_name}_{model_name}_accuracy_{split_name}", acc
                )
                mlflow.log_metric(
                    f"{dataset_name}_{model_name}_precision_{split_name}",
                    precision,
                )
                mlflow.log_metric(
                    f"{dataset_name}_{model_name}_recall_{split_name}", recall
                )
                mlflow.log_metric(
                    f"{dataset_name}_{model_name}_f1_score_{split_name}", f1
                )
                mlflow.log_metric(
                    f"{dataset_name}_{model_name}_auc_roc_{split_name}", auc
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
    meta_dataset.to_csv("meta_dataset_multiclass.csv", index=False)
    print(
        "\nStep 5: Train and evaluate meta-models using cross-validation and different algorithms."
    )

    # Prepare data for meta-models
    # Expand class_balance into separate features and fill NaNs with zeros
    class_balance_df = pd.DataFrame(meta_dataset["class_balance"].tolist())
    class_balance_df.columns = [
        f"class_balance_{i}" for i in class_balance_df.columns
    ]
    class_balance_df = class_balance_df.fillna(0)  # Fill NaNs with zeros

    meta_dataset_expanded = pd.concat(
        [meta_dataset.drop("class_balance", axis=1), class_balance_df], axis=1
    )

    # Prepare the features
    X_meta = meta_dataset_expanded.drop(
        ["dataset_name", "model_name"]
        + [
            col
            for col in meta_dataset_expanded.columns
            if "accuracy" in col
            or "precision" in col
            or "recall" in col
            or "f1_score" in col
            or "auc_roc" in col
        ],
        axis=1,
    )

    # Encode model names
    model_encoder = LabelEncoder()
    X_meta["model_encoded"] = model_encoder.fit_transform(
        meta_dataset_expanded["model_name"]
    )

    # Handle NaN values in meta-features
    X_meta.fillna(X_meta.mean(), inplace=True)

    # Scale the meta-features
    scaler_meta = StandardScaler()
    X_meta_scaled = scaler_meta.fit_transform(X_meta)

    # Targets
    y_meta = {}
    for split in ["val", "test"]:
        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "auc_roc",
        ]:
            key = f"{metric}_{split}"
            y_meta[key] = (
                meta_dataset_expanded[key]
                .fillna(meta_dataset_expanded[key].mean())
                .values
            )

    groups = meta_dataset_expanded["dataset_name"].values

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
            f"\nMean Absolute Error of {metric.replace('_', ' ').title()} Meta-Model (Neural Network): {mean_errors_nn[metric]:.4f}"
        )
        print(
            f"Mean Absolute Error of {metric.replace('_', ' ').title()} Meta-Model (XGBoost): {mean_errors_xgb[metric]:.4f}"
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
    predictions_df = meta_dataset_expanded.copy()
    for metric in metric_names:
        predicted = final_meta_models[metric].predict(X_meta_scaled)
        predictions_df[f"predicted_{metric}"] = predicted

    for idx, row in predictions_df.iterrows():
        print(f"{row['dataset_name']} - {row['model_name']}:")
        for split in ["val", "test"]:
            print(
                f"  Predicted Accuracy ({split}): {row[f'predicted_accuracy_{split}']:.4f}"
            )
            print(
                f"  Actual Accuracy ({split}): {row[f'accuracy_{split}']:.4f}"
            )
            print(
                f"  Predicted Precision ({split}): {row[f'predicted_precision_{split}']:.4f}"
            )
            print(
                f"  Actual Precision ({split}): {row[f'precision_{split}']:.4f}"
            )
            print(
                f"  Predicted Recall ({split}): {row[f'predicted_recall_{split}']:.4f}"
            )
            print(f"  Actual Recall ({split}): {row[f'recall_{split}']:.4f}")
            print(
                f"  Predicted F1-Score ({split}): {row[f'predicted_f1_score_{split}']:.4f}"
            )
            print(
                f"  Actual F1-Score ({split}): {row[f'f1_score_{split}']:.4f}"
            )
            print(
                f"  Predicted AUC-ROC ({split}): {row[f'predicted_auc_roc_{split}']:.4f}"
            )
            print(f"  Actual AUC-ROC ({split}): {row[f'auc_roc_{split}']:.4f}")

    print(
        "\nStep 7: Compile predictions and compare predicted metrics with actual metrics."
    )

    print(predictions_df.head())

    print("\nStep 8: Evaluate the meta-models' performance.")

    for split in ["val", "test"]:
        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "auc_roc",
        ]:
            predictions_df[f"{metric}_abs_error_{split}"] = abs(
                predictions_df[f"predicted_{metric}_{split}"]
                - predictions_df[f"{metric}_{split}"]
            )

        mean_acc_abs_error = predictions_df[
            f"accuracy_abs_error_{split}"
        ].mean()
        mean_precision_abs_error = predictions_df[
            f"precision_abs_error_{split}"
        ].mean()
        mean_recall_abs_error = predictions_df[
            f"recall_abs_error_{split}"
        ].mean()
        mean_f1_abs_error = predictions_df[
            f"f1_score_abs_error_{split}"
        ].mean()
        mean_auc_abs_error = predictions_df[
            f"auc_roc_abs_error_{split}"
        ].mean()

        print(
            f"\nMean Absolute Error of Accuracy Meta-Model ({split} set): {mean_acc_abs_error:.4f}"
        )
        print(
            f"Mean Absolute Error of Precision Meta-Model ({split} set): {mean_precision_abs_error:.4f}"
        )
        print(
            f"Mean Absolute Error of Recall Meta-Model ({split} set): {mean_recall_abs_error:.4f}"
        )
        print(
            f"Mean Absolute Error of F1-Score Meta-Model ({split} set): {mean_f1_abs_error:.4f}"
        )
        print(
            f"Mean Absolute Error of AUC-ROC Meta-Model ({split} set): {mean_auc_abs_error:.4f}"
        )

        # Log evaluation metrics in MLflow
        mlflow.log_metric(
            f"mean_acc_abs_error_{split}_final", mean_acc_abs_error
        )
        mlflow.log_metric(
            f"mean_precision_abs_error_{split}_final", mean_precision_abs_error
        )
        mlflow.log_metric(
            f"mean_recall_abs_error_{split}_final", mean_recall_abs_error
        )
        mlflow.log_metric(
            f"mean_f1_abs_error_{split}_final", mean_f1_abs_error
        )
        mlflow.log_metric(
            f"mean_auc_abs_error_{split}_final", mean_auc_abs_error
        )

    # Save the predictions DataFrame to a CSV file
    output_file_path = "meta_model_predictions_multiclass.csv"
    predictions_df.to_csv(output_file_path, index=False)
    print(f"\nPredictions saved to {output_file_path}")

    # Log the CSV file as an artifact in MLflow
    mlflow.log_artifact(output_file_path)
    print(f"Predictions CSV saved and logged to MLflow as an artifact.")

    # Plot and log comparisons between predicted and actual metrics
    for split in ["val", "test"]:
        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "auc_roc",
        ]:
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
                f"Predicted vs Actual {metric.replace('_', ' ').capitalize()} on {split.capitalize()} Set ({split_ratio*100:.0f}% of data)"
            )
            plt.xlabel("Model")
            plt.ylabel(metric.replace("_", " ").capitalize())
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
                f"{metric.replace('_', ' ').capitalize()} comparison plot for {split} set saved and logged to MLflow as an artifact."
            )
