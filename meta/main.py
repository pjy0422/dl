# Import necessary libraries
import warnings

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# Initialize MLflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("Meta-Model")

print("Step 1: Load and preprocess multiple datasets.")

# Initialize lists to store datasets
datasets = []

# 1. Kaggle Credit Card Fraud Detection dataset
print("Loading Credit Card Fraud Detection dataset...")
try:
    cc_data = pd.read_csv("creditcard.csv")
    cc_data_sampled = cc_data.sample(n=10000, random_state=42)
    X_cc = cc_data_sampled.drop("Class", axis=1)
    y_cc = cc_data_sampled["Class"]
    datasets.append(("Credit Card Fraud", X_cc, y_cc))
    print(
        "Credit Card Fraud Detection dataset loaded and sampled successfully."
    )
except FileNotFoundError:
    print(
        "Credit Card Fraud Detection dataset not found. Please ensure 'creditcard.csv' is in the working directory."
    )

# 2. Breast Cancer Wisconsin Diagnostic dataset
print("Loading Breast Cancer dataset...")
breast_cancer = load_breast_cancer()
X_bc = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
y_bc = pd.Series(breast_cancer.target)
datasets.append(("Breast Cancer", X_bc, y_bc))
print("Breast Cancer dataset loaded successfully.")

# 3. Pima Indians Diabetes dataset
print("Loading Diabetes dataset...")
try:
    diabetes = pd.read_csv("diabetes.csv")
    X_diabetes = diabetes.drop("Outcome", axis=1)
    y_diabetes = diabetes["Outcome"]
    datasets.append(("Diabetes", X_diabetes, y_diabetes))
    print("Diabetes dataset loaded successfully.")
except FileNotFoundError:
    print(
        "Diabetes dataset not found. Please ensure 'diabetes.csv' is in the working directory."
    )

# 4. Synthetic datasets with varying parameters
print("Generating synthetic datasets...")
for i in range(2):
    X_syn, y_syn = make_classification(
        n_samples=500 + i * 250,
        n_features=15 + i * 5,
        n_informative=2 + i,
        n_redundant=2,
        n_classes=2,
        weights=[0.7, 0.3],
        flip_y=0.01,
        random_state=42 + i,
    )
    datasets.append(
        (f"Synthetic Dataset {i+1}", pd.DataFrame(X_syn), pd.Series(y_syn))
    )
    print(f"Synthetic Dataset {i+1} generated.")

print("\nStep 2: Define classification models to be used.")
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=200, solver="lbfgs", n_jobs=-1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=50, random_state=42, n_jobs=-1
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
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
        meta_features["class_balance"] = y_train.mean()
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

            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_val)[:, 1]
            else:
                y_proba = model.predict(X_val)

            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_val, y_proba)
            except ValueError:
                auc = np.nan

            performance_list.append(
                {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "accuracy": acc,
                    "f1_score": f1,
                    "auc_roc": auc,
                }
            )

            meta_features_entry = meta_features.copy()
            meta_features_entry["model_name"] = model_name
            meta_features_list.append(meta_features_entry)

            print(f"    - Accuracy: {acc:.4f}")
            print(f"    - F1-Score: {f1:.4f}")
            print(f"    - AUC-ROC: {auc:.4f}")

            # Log metrics
            mlflow.log_metric(f"{dataset_name}_{model_name}_accuracy", acc)
            mlflow.log_metric(f"{dataset_name}_{model_name}_f1_score", f1)
            mlflow.log_metric(f"{dataset_name}_{model_name}_auc_roc", auc)

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

    print(
        "\nStep 5: Train meta-models to predict Accuracy, F1-Score, and AUC-ROC from meta-features."
    )

    X_meta = meta_dataset.drop(
        ["dataset_name", "model_name", "accuracy", "f1_score", "auc_roc"],
        axis=1,
    )

    model_encoder = LabelEncoder()
    X_meta["model_encoded"] = model_encoder.fit_transform(
        meta_dataset["model_name"]
    )

    y_meta_acc = meta_dataset["accuracy"]
    y_meta_f1 = meta_dataset["f1_score"]
    y_meta_auc = meta_dataset["auc_roc"].fillna(meta_dataset["auc_roc"].mean())

    meta_model_acc = GradientBoostingRegressor(random_state=42)
    meta_model_acc.fit(X_meta, y_meta_acc)

    meta_model_f1 = GradientBoostingRegressor(random_state=42)
    meta_model_f1.fit(X_meta, y_meta_f1)

    meta_model_auc = GradientBoostingRegressor(random_state=42)
    meta_model_auc.fit(X_meta, y_meta_auc)

    print("Meta-models training completed.")

    # Log meta-models in MLflow
    mlflow.sklearn.log_model(meta_model_acc, "meta_model_accuracy")
    mlflow.sklearn.log_model(meta_model_f1, "meta_model_f1_score")
    mlflow.sklearn.log_model(meta_model_auc, "meta_model_auc_roc")
    print("Meta-models logged to MLflow.")

    print(
        "\nStep 6: Predict metrics for each dataset and model using the meta-models."
    )
    predictions = []
    for idx, row in meta_dataset.iterrows():
        X_meta_row = (
            row.drop(
                [
                    "dataset_name",
                    "model_name",
                    "accuracy",
                    "f1_score",
                    "auc_roc",
                ]
            )
            .to_frame()
            .T
        )
        model_encoded_value = model_encoder.transform([row["model_name"]])[0]
        X_meta_row["model_encoded"] = model_encoded_value

        predicted_acc = meta_model_acc.predict(X_meta_row)[0]
        predicted_f1 = meta_model_f1.predict(X_meta_row)[0]
        predicted_auc = meta_model_auc.predict(X_meta_row)[0]

        predictions.append(
            {
                "dataset_name": row["dataset_name"],
                "model_name": row["model_name"],
                "predicted_accuracy": predicted_acc,
                "actual_accuracy": row["accuracy"],
                "predicted_f1_score": predicted_f1,
                "actual_f1_score": row["f1_score"],
                "predicted_auc_roc": predicted_auc,
                "actual_auc_roc": row["auc_roc"],
            }
        )

        print(f"{row['dataset_name']} - {row['model_name']}:")
        print(f"  Predicted Accuracy: {predicted_acc:.4f}")
        print(f"  Actual Accuracy: {row['accuracy']:.4f}")
        print(f"  Predicted F1-Score: {predicted_f1:.4f}")
        print(f"  Actual F1-Score: {row['f1_score']:.4f}")
        print(f"  Predicted AUC-ROC: {predicted_auc:.4f}")
        print(f"  Actual AUC-ROC: {row['auc_roc']:.4f}")

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
    predictions_df["f1_abs_error"] = abs(
        predictions_df["predicted_f1_score"]
        - predictions_df["actual_f1_score"]
    )
    predictions_df["auc_abs_error"] = abs(
        predictions_df["predicted_auc_roc"] - predictions_df["actual_auc_roc"]
    )

    mean_acc_abs_error = predictions_df["acc_abs_error"].mean()
    mean_f1_abs_error = predictions_df["f1_abs_error"].mean()
    mean_auc_abs_error = predictions_df["auc_abs_error"].mean()

    print(
        f"\nMean Absolute Error of Accuracy Meta-Model: {mean_acc_abs_error:.4f}"
    )
    print(
        f"Mean Absolute Error of F1-Score Meta-Model: {mean_f1_abs_error:.4f}"
    )
    print(
        f"Mean Absolute Error of AUC-ROC Meta-Model: {mean_auc_abs_error:.4f}"
    )

    # Log evaluation metrics in MLflow
    mlflow.log_metric("mean_acc_abs_error", mean_acc_abs_error)
    mlflow.log_metric("mean_f1_abs_error", mean_f1_abs_error)
    mlflow.log_metric("mean_auc_abs_error", mean_auc_abs_error)

    # Save the predictions DataFrame to a CSV file
    output_file_path = "meta_model_predictions.csv"
    predictions_df.to_csv(output_file_path, index=False)
    print(f"\nPredictions saved to {output_file_path}")

    # Log the CSV file as an artifact in MLflow
    mlflow.log_artifact(output_file_path)
    print(f"Predictions CSV saved and logged to MLflow as an artifact.")

    # Plot and log comparisons between predicted and actual metrics
    for metric in ["accuracy", "f1_score", "auc_roc"]:
        plt.figure(figsize=(10, 6))
        for dataset_name in predictions_df["dataset_name"].unique():
            df_subset = predictions_df[
                predictions_df["dataset_name"] == dataset_name
            ]
            plt.plot(
                df_subset["model_name"],
                df_subset[f"actual_{metric}"],
                label=f"{dataset_name} - Actual",
            )
            plt.plot(
                df_subset["model_name"],
                df_subset[f"predicted_{metric}"],
                "--",
                label=f"{dataset_name} - Predicted",
            )

        plt.title(f"Predicted vs Actual {metric.capitalize()}")
        plt.xlabel("Model")
        plt.ylabel(metric.capitalize())
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        # Save the plot
        plot_filename = f"{metric}_comparison_plot.png"
        plt.savefig(plot_filename)

        # Log the plot as an artifact
        mlflow.log_artifact(plot_filename)

        print(
            f"{metric.capitalize()} comparison plot saved and logged to MLflow as an artifact."
        )
