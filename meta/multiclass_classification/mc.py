# Import necessary libraries
import warnings

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_digits, load_wine
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Initialize MLflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
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
    "SVM": SVC(probability=True, random_state=42),
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
with mlflow.start_run(run_name=f"META_RUN_{randomnumber}"):

    for dataset_name, X, y in datasets:
        print(
            f"\nProcessing {dataset_name} ({dataset_counter}/{len(datasets)})..."
        )
        dataset_counter += 1

        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.3, stratify=y, random_state=42
        )

        meta_features = {}
        meta_features["dataset_name"] = dataset_name
        meta_features["n_samples"] = X_train.shape[0]
        meta_features["n_features"] = X_train.shape[1]
        class_counts = np.bincount(y_train)
        meta_features["n_classes"] = len(class_counts)
        class_balance = class_counts / class_counts.sum()
        meta_features["class_balance"] = class_balance.tolist()
        meta_features["feature_mean"] = np.mean(X_train)
        meta_features["feature_std"] = np.std(X_train)

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
            y_pred = model.predict(X_val)

            acc = accuracy_score(y_val, y_pred)
            f1_macro = f1_score(
                y_val, y_pred, average="macro", zero_division=0
            )
            f1_weighted = f1_score(
                y_val, y_pred, average="weighted", zero_division=0
            )

            performance_list.append(
                {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "accuracy": acc,
                    "f1_macro": f1_macro,
                    "f1_weighted": f1_weighted,
                }
            )

            meta_features_entry = meta_features.copy()
            meta_features_entry["model_name"] = model_name
            meta_features_list.append(meta_features_entry)

            print(f"    - Accuracy: {acc:.4f}")
            print(f"    - F1-Score (Macro): {f1_macro:.4f}")
            print(f"    - F1-Score (Weighted): {f1_weighted:.4f}")

            # Log metrics
            mlflow.log_metric(f"{dataset_name}_{model_name}_accuracy", acc)
            mlflow.log_metric(
                f"{dataset_name}_{model_name}_f1_macro", f1_macro
            )
            mlflow.log_metric(
                f"{dataset_name}_{model_name}_f1_weighted", f1_weighted
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
    # Save the meta-dataset to a CSV file
    meta_dataset.to_csv("meta_dataset_multiclass.csv", index=False)
    print(
        "\nStep 5: Train meta-models to predict Accuracy and F1-Scores from meta-features."
    )

    # Expand class_balance into separate features and fill NaNs with zeros
    class_balance_df = pd.DataFrame(meta_dataset["class_balance"].tolist())
    class_balance_df.columns = [
        f"class_balance_{i}" for i in class_balance_df.columns
    ]
    class_balance_df = class_balance_df.fillna(0)  # Fill NaNs with zeros

    meta_dataset_expanded = pd.concat(
        [meta_dataset.drop("class_balance", axis=1), class_balance_df], axis=1
    )

    X_meta = meta_dataset_expanded.drop(
        ["dataset_name", "model_name", "accuracy", "f1_macro", "f1_weighted"],
        axis=1,
    )

    model_encoder = LabelEncoder()
    X_meta["model_encoded"] = model_encoder.fit_transform(
        meta_dataset_expanded["model_name"]
    )

    y_meta_acc = meta_dataset_expanded["accuracy"]
    y_meta_f1_macro = meta_dataset_expanded["f1_macro"]
    y_meta_f1_weighted = meta_dataset_expanded["f1_weighted"]

    meta_model_acc = GradientBoostingRegressor(random_state=42)
    meta_model_acc.fit(X_meta, y_meta_acc)

    meta_model_f1_macro = GradientBoostingRegressor(random_state=42)
    meta_model_f1_macro.fit(X_meta, y_meta_f1_macro)

    meta_model_f1_weighted = GradientBoostingRegressor(random_state=42)
    meta_model_f1_weighted.fit(X_meta, y_meta_f1_weighted)

    print("Meta-models training completed.")

    # Log meta-models in MLflow
    mlflow.sklearn.log_model(meta_model_acc, "meta_model_accuracy")
    mlflow.sklearn.log_model(meta_model_f1_macro, "meta_model_f1_macro")
    mlflow.sklearn.log_model(meta_model_f1_weighted, "meta_model_f1_weighted")
    print("Meta-models logged to MLflow.")

    print(
        "\nStep 6: Predict metrics for each dataset and model using the meta-models."
    )
    predictions = []
    for idx, row in meta_dataset_expanded.iterrows():
        X_meta_row = (
            row.drop(
                [
                    "dataset_name",
                    "model_name",
                    "accuracy",
                    "f1_macro",
                    "f1_weighted",
                ]
            )
            .to_frame()
            .T
        )
        model_encoded_value = model_encoder.transform([row["model_name"]])[0]
        X_meta_row["model_encoded"] = model_encoded_value

        predicted_acc = meta_model_acc.predict(X_meta_row)[0]
        predicted_f1_macro = meta_model_f1_macro.predict(X_meta_row)[0]
        predicted_f1_weighted = meta_model_f1_weighted.predict(X_meta_row)[0]

        predictions.append(
            {
                "dataset_name": row["dataset_name"],
                "model_name": row["model_name"],
                "predicted_accuracy": predicted_acc,
                "actual_accuracy": row["accuracy"],
                "predicted_f1_macro": predicted_f1_macro,
                "actual_f1_macro": row["f1_macro"],
                "predicted_f1_weighted": predicted_f1_weighted,
                "actual_f1_weighted": row["f1_weighted"],
            }
        )

        print(f"{row['dataset_name']} - {row['model_name']}:")
        print(f"  Predicted Accuracy: {predicted_acc:.4f}")
        print(f"  Actual Accuracy: {row['accuracy']:.4f}")
        print(f"  Predicted F1-Score (Macro): {predicted_f1_macro:.4f}")
        print(f"  Actual F1-Score (Macro): {row['f1_macro']:.4f}")
        print(f"  Predicted F1-Score (Weighted): {predicted_f1_weighted:.4f}")
        print(f"  Actual F1-Score (Weighted): {row['f1_weighted']:.4f}")

    print(
        "\nStep 7: Compile predictions and compare predicted metrics with actual metrics."
    )
    predictions_df = pd.DataFrame(predictions)
    print(predictions_df)

    print("\nStep 8: Evaluate the meta-models' performance.")

    predictions_df["acc_abs_error"] = abs(
        predictions_df["predicted_accuracy"]
        - predictions_df["actual_accuracy"]
    )
    predictions_df["f1_macro_abs_error"] = abs(
        predictions_df["predicted_f1_macro"]
        - predictions_df["actual_f1_macro"]
    )
    predictions_df["f1_weighted_abs_error"] = abs(
        predictions_df["predicted_f1_weighted"]
        - predictions_df["actual_f1_weighted"]
    )

    mean_acc_abs_error = predictions_df["acc_abs_error"].mean()
    mean_f1_macro_abs_error = predictions_df["f1_macro_abs_error"].mean()
    mean_f1_weighted_abs_error = predictions_df["f1_weighted_abs_error"].mean()

    print(
        f"\nMean Absolute Error of Accuracy Meta-Model: {mean_acc_abs_error:.4f}"
    )
    print(
        f"Mean Absolute Error of F1-Score (Macro) Meta-Model: {mean_f1_macro_abs_error:.4f}"
    )
    print(
        f"Mean Absolute Error of F1-Score (Weighted) Meta-Model: {mean_f1_weighted_abs_error:.4f}"
    )

    # Log evaluation metrics in MLflow
    mlflow.log_metric("mean_acc_abs_error", mean_acc_abs_error)
    mlflow.log_metric("mean_f1_macro_abs_error", mean_f1_macro_abs_error)
    mlflow.log_metric("mean_f1_weighted_abs_error", mean_f1_weighted_abs_error)

    # Save the predictions DataFrame to a CSV file
    output_file_path = "meta_model_predictions_multiclass.csv"
    predictions_df.to_csv(output_file_path, index=False)
    print(f"\nPredictions saved to {output_file_path}")

    # Log the CSV file as an artifact in MLflow
    mlflow.log_artifact(output_file_path)
    print(f"Predictions CSV saved and logged to MLflow as an artifact.")

    # Plot and log comparisons between predicted and actual metrics
    for metric in ["accuracy", "f1_macro", "f1_weighted"]:
        plt.figure(
            figsize=(12, 8)
        )  # Adjusting the figure size with more margin
        for dataset_name in predictions_df["dataset_name"].unique():
            df_subset = predictions_df[
                predictions_df["dataset_name"] == dataset_name
            ]
            plt.plot(
                df_subset["model_name"],
                df_subset[f"actual_{metric}"],
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

        plt.title(
            f"Predicted vs Actual {metric.replace('_', ' ').capitalize()}"
        )
        plt.xlabel("Model")
        plt.ylabel(metric.replace("_", " ").capitalize())
        plt.xticks(rotation=45)

        # Adjust the legend to be in the upper right corner
        plt.legend(loc="upper right")

        # Add margin to the plot
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plot_filename = f"{metric}_comparison_plot.png"
        plt.savefig(plot_filename)

        # Log the plot as an artifact
        mlflow.log_artifact(plot_filename)

        print(
            f"{metric.replace('_', ' ').capitalize()} comparison plot saved and logged to MLflow as an artifact."
        )
