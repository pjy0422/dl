# Import necessary libraries
import warnings

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import entropy, kurtosis, skew
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")


def load_and_preprocess_datasets():
    print(
        "Step 1: Load and preprocess multiple binary classification datasets."
    )

    datasets = []

    # Dataset 1: Kaggle Credit Card Fraud Detection dataset
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

    # Dataset 2: Breast Cancer Wisconsin Diagnostic dataset
    print("Loading Breast Cancer dataset...")
    breast_cancer = load_breast_cancer()
    X_bc = pd.DataFrame(
        breast_cancer.data, columns=breast_cancer.feature_names
    )
    y_bc = pd.Series(breast_cancer.target)
    datasets.append(("Breast Cancer", X_bc, y_bc))
    print("Breast Cancer dataset loaded successfully.")

    # Dataset 3: Pima Indians Diabetes dataset
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

    # Dataset 4: Titanic dataset
    print("Downloading and loading Titanic dataset...")
    try:
        titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        titanic = pd.read_csv(titanic_url)
        # Preprocess Titanic dataset
        titanic = titanic.drop(
            ["PassengerId", "Name", "Ticket", "Cabin"], axis=1
        )
        titanic["Age"].fillna(titanic["Age"].median(), inplace=True)
        titanic["Embarked"].fillna(titanic["Embarked"].mode()[0], inplace=True)
        titanic = pd.get_dummies(
            titanic, columns=["Sex", "Embarked"], drop_first=True
        )
        X_titanic = titanic.drop("Survived", axis=1)
        y_titanic = titanic["Survived"]
        datasets.append(("Titanic", X_titanic, y_titanic))
        print("Titanic dataset loaded and preprocessed successfully.")
    except Exception as e:
        print(f"Error loading Titanic dataset: {e}")

    # Dataset 5: Adult Income dataset
    print("Downloading and loading Adult Income dataset...")
    try:
        adult_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        adult_column_names = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "income",
        ]
        adult = pd.read_csv(
            adult_url,
            names=adult_column_names,
            na_values=" ?",
            skipinitialspace=True,
        )
        # Preprocess Adult dataset
        adult.dropna(inplace=True)
        adult = adult.drop("fnlwgt", axis=1)
        categorical_cols = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
        adult = pd.get_dummies(
            adult, columns=categorical_cols, drop_first=True
        )
        adult["income"] = adult["income"].apply(
            lambda x: 1 if x == ">50K" else 0
        )
        X_adult = adult.drop("income", axis=1)
        y_adult = adult["income"]
        datasets.append(("Adult Income", X_adult, y_adult))
        print("Adult Income dataset loaded and preprocessed successfully.")
    except Exception as e:
        print(f"Error loading Adult Income dataset: {e}")

    # Dataset 6: Heart Disease dataset
    print("Downloading and loading Heart Disease dataset...")
    try:
        heart = pd.read_csv("heart.csv")
        X_heart = heart.drop("target", axis=1)
        y_heart = heart["target"]
        datasets.append(("Heart Disease", X_heart, y_heart))
        print("Heart Disease dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading Heart Disease dataset: {e}")

    return datasets


def define_models():
    print("\nStep 2: Define diverse classification models to be used.")
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=200, solver="lbfgs", n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=50, random_state=42, n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        # "SVM": SVC(kernel="linear", random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", random_state=42
        ),
    }
    return models


def extract_meta_features(X):
    meta_features = {}
    meta_features["n_samples"] = X.shape[0]
    meta_features["n_features"] = X.shape[1]
    meta_features["feature_mean"] = np.mean(X)
    meta_features["feature_std"] = np.std(X)
    meta_features["coeff_variation"] = (
        np.std(X) / np.mean(X) if np.mean(X) != 0 else 0
    )
    n_components = min(5, X.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X)
    meta_features["pca_explained_variance"] = np.sum(
        pca.explained_variance_ratio_
    )
    # Compute skewness and kurtosis
    skewness = skew(X, axis=0)
    kurtosis_values = kurtosis(X, axis=0)
    meta_features["avg_skewness"] = np.mean(skewness)
    meta_features["avg_kurtosis"] = np.mean(kurtosis_values)
    # Compute mean absolute correlation between features
    corr_matrix = np.corrcoef(X, rowvar=False)
    # Exclude self-correlation by masking the diagonal
    mask = np.ones(corr_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    abs_corr = np.abs(corr_matrix[mask])
    meta_features["mean_abs_correlation"] = np.mean(abs_corr)
    # Number of features with zero variance
    zero_variance_features = np.sum(np.var(X, axis=0) == 0)
    meta_features["n_zero_variance_features"] = zero_variance_features
    # Mean and median feature variances
    variances = np.var(X, axis=0)
    meta_features["mean_variance"] = np.mean(variances)
    meta_features["median_variance"] = np.median(variances)
    # Mean feature entropy
    feature_entropies = [
        entropy(np.histogram(X[:, i], bins=10)[0] + 1e-10)
        for i in range(X.shape[1])
    ]
    meta_features["mean_feature_entropy"] = np.mean(feature_entropies)
    return meta_features


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    mlflow,
    dataset_name,
    model_name,
):
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
        y_proba_val = model.predict_proba(X_val)[:, 1]
    else:
        y_proba_val = model.decision_function(X_val)
    acc_val = accuracy_score(y_val, y_pred_val)
    f1_val = f1_score(y_val, y_pred_val, zero_division=0)
    try:
        if len(np.unique(y_val)) == 1:
            auc_val = np.nan
        else:
            auc_val = roc_auc_score(y_val, y_proba_val)
    except ValueError:
        auc_val = np.nan
    # Evaluate on test set
    y_pred_test = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba_test = model.predict_proba(X_test)[:, 1]
    else:
        y_proba_test = model.decision_function(X_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test, zero_division=0)
    try:
        if len(np.unique(y_test)) == 1:
            auc_test = np.nan
        else:
            auc_test = roc_auc_score(y_test, y_proba_test)
    except ValueError:
        auc_test = np.nan
    performance = {
        "accuracy_val": acc_val,
        "f1_score_val": f1_val,
        "auc_roc_val": auc_val,
        "accuracy_test": acc_test,
        "f1_score_test": f1_test,
        "auc_roc_test": auc_test,
    }
    print(f"    - Validation Accuracy: {acc_val:.4f}")
    print(f"    - Validation F1-Score: {f1_val:.4f}")
    print(f"    - Validation AUC-ROC: {auc_val:.4f}")
    # Log metrics
    mlflow.log_metric(f"{dataset_name}_{model_name}_accuracy_val", acc_val)
    mlflow.log_metric(f"{dataset_name}_{model_name}_f1_score_val", f1_val)
    mlflow.log_metric(f"{dataset_name}_{model_name}_auc_roc_val", auc_val)
    # Save model
    mlflow.sklearn.log_model(model, f"{dataset_name}_{model_name}_model")
    return performance


def train_and_evaluate_meta_model(
    X_meta_scaled,
    y_meta,
    groups,
    model_type="xgb",
    metric_name="",
    mlflow=None,
):
    errors = []
    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    for train_idx, test_idx in gkf.split(X_meta_scaled, y_meta, groups=groups):
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
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    mean_abs_error = np.mean(errors)
    print(
        f"Mean Absolute Error of {metric_name} Meta-Model ({model_type.upper()}): {mean_abs_error:.4f}"
    )
    # Log evaluation metrics in MLflow if mlflow is provided
    if mlflow is not None:
        mlflow.log_metric(
            f"mean_abs_error_{metric_name}_{model_type}", mean_abs_error
        )
    return mean_abs_error


def train_final_meta_model(X_meta_scaled, y_meta, model_type="xgb"):
    if model_type == "xgb":
        # Initialize XGBoost regressor
        xgb_model_final = XGBRegressor(random_state=42)
        # Train XGBoost model
        xgb_model_final.fit(X_meta_scaled, y_meta)
        return xgb_model_final
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def plot_metric_comparison(
    predictions_df, metric, split, data_split_ratio, mlflow
):
    plt.figure(figsize=(12, 8))
    for dataset_name in predictions_df["dataset_name"].unique():
        df_subset = predictions_df[
            (predictions_df["dataset_name"] == dataset_name)
            & (predictions_df["split"] == split)
        ]
        if df_subset.empty:
            continue  # Skip if there is no data for this dataset and split
        plt.plot(
            df_subset["model_name"],
            df_subset[metric],
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
        f"Predicted vs Actual {metric.capitalize()} on {split.capitalize()} Set ({data_split_ratio})"
    )
    plt.xlabel("Model")
    plt.ylabel(metric.capitalize())
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
        f"{metric.capitalize()} comparison plot for {split} set saved and logged to MLflow as an artifact."
    )


def main():
    # Initialize MLflow
    mlflow.set_tracking_uri(uri="http://127.0.0.1:3050")
    mlflow.set_experiment("Meta-Model")
    datasets = load_and_preprocess_datasets()
    models = define_models()
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
                X_temp,
                y_temp,
                test_size=0.25,
                stratify=y_temp,
                random_state=42,
            )
            # Now, X_train: 60%, X_val: 20%, X_test: 20%

            # Extract meta-features for each split
            meta_features_train = extract_meta_features(X_train)
            meta_features_train["dataset_name"] = dataset_name
            meta_features_train["split"] = "train"
            meta_features_val = extract_meta_features(X_val)
            meta_features_val["dataset_name"] = dataset_name
            meta_features_val["split"] = "val"
            meta_features_test = extract_meta_features(X_test)
            meta_features_test["dataset_name"] = dataset_name
            meta_features_test["split"] = "test"

            for model_name, model in models.items():
                performance = train_model(
                    model,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    mlflow,
                    dataset_name,
                    model_name,
                )
                # Collect performance metrics for each split
                performance_train = {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "split": "train",
                    # Assuming we can get training performance (could use cross-validation)
                    # For simplicity, we'll use training performance as NaN
                    "accuracy": np.nan,
                    "f1_score": np.nan,
                    "auc_roc": np.nan,
                }
                performance_val = {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "split": "val",
                    "accuracy": performance["accuracy_val"],
                    "f1_score": performance["f1_score_val"],
                    "auc_roc": performance["auc_roc_val"],
                }
                performance_test = {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "split": "test",
                    "accuracy": performance["accuracy_test"],
                    "f1_score": performance["f1_score_test"],
                    "auc_roc": performance["auc_roc_test"],
                }
                performance_list.extend(
                    [performance_train, performance_val, performance_test]
                )
                # Meta-features for each split
                meta_features_entry_train = meta_features_train.copy()
                meta_features_entry_train["model_name"] = model_name
                meta_features_entry_val = meta_features_val.copy()
                meta_features_entry_val["model_name"] = model_name
                meta_features_entry_test = meta_features_test.copy()
                meta_features_entry_test["model_name"] = model_name
                meta_features_list.extend(
                    [
                        meta_features_entry_train,
                        meta_features_entry_val,
                        meta_features_entry_test,
                    ]
                )
        print("\nStep 4: Create a meta-dataset for meta-learning.")
        meta_features_df = pd.DataFrame(meta_features_list)
        performance_df = pd.DataFrame(performance_list)
        meta_dataset = pd.merge(
            meta_features_df,
            performance_df,
            on=["dataset_name", "model_name", "split"],
        )
        print("Meta-dataset created:")
        print(meta_dataset.head())
        meta_dataset.to_csv("meta_dataset_binaryclass.csv", index=False)
        print(
            "\nStep 5: Train and evaluate meta-models using cross-validation and different algorithms."
        )
        # Prepare data for meta-models
        X_meta = meta_dataset.drop(
            [
                "dataset_name",
                "model_name",
                "split",
                "accuracy",
                "f1_score",
                "auc_roc",
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
        # Targets for metrics
        y_meta_acc = meta_dataset["accuracy"].values
        y_meta_f1 = meta_dataset["f1_score"].values
        y_meta_auc = (
            meta_dataset["auc_roc"]
            .fillna(meta_dataset["auc_roc"].mean())
            .values
        )
        groups = (
            meta_dataset["dataset_name"].values
            + "_"
            + meta_dataset["split"].values
        )
        print("\nTraining and evaluating meta-models for metrics.")
        # Use only validation and test splits
        valid_splits = meta_dataset["split"].isin(["val", "test"])
        X_meta_scaled_valid = X_meta_scaled[valid_splits]
        y_meta_acc_valid = y_meta_acc[valid_splits]
        y_meta_f1_valid = y_meta_f1[valid_splits]
        y_meta_auc_valid = y_meta_auc[valid_splits]
        groups_valid = groups[valid_splits]
        mean_acc_abs_error = train_and_evaluate_meta_model(
            X_meta_scaled_valid,
            y_meta_acc_valid,
            groups_valid,
            model_type="xgb",
            metric_name="Accuracy",
            mlflow=mlflow,
        )
        mean_f1_abs_error = train_and_evaluate_meta_model(
            X_meta_scaled_valid,
            y_meta_f1_valid,
            groups_valid,
            model_type="xgb",
            metric_name="F1-Score",
            mlflow=mlflow,
        )
        mean_auc_abs_error = train_and_evaluate_meta_model(
            X_meta_scaled_valid,
            y_meta_auc_valid,
            groups_valid,
            model_type="xgb",
            metric_name="AUC-ROC",
            mlflow=mlflow,
        )
        # Log evaluation metrics in MLflow
        mlflow.log_metric("mean_acc_abs_error_xgb", mean_acc_abs_error)
        mlflow.log_metric("mean_f1_abs_error_xgb", mean_f1_abs_error)
        mlflow.log_metric("mean_auc_abs_error_xgb", mean_auc_abs_error)
        # Train final meta-models on the validation and test data
        xgb_model_acc_final = train_final_meta_model(
            X_meta_scaled_valid, y_meta_acc_valid, model_type="xgb"
        )
        xgb_model_f1_final = train_final_meta_model(
            X_meta_scaled_valid, y_meta_f1_valid, model_type="xgb"
        )
        xgb_model_auc_final = train_final_meta_model(
            X_meta_scaled_valid, y_meta_auc_valid, model_type="xgb"
        )
        # Save the final models
        mlflow.sklearn.log_model(
            xgb_model_acc_final, "final_meta_model_accuracy_xgb"
        )
        mlflow.sklearn.log_model(
            xgb_model_f1_final, "final_meta_model_f1_score_xgb"
        )
        mlflow.sklearn.log_model(
            xgb_model_auc_final, "final_meta_model_auc_roc_xgb"
        )
        print("Meta-models logged to MLflow.")
        print(
            "\nStep 6: Predict metrics for each dataset and model using the best meta-models."
        )
        # Predict on the validation and test splits
        predictions_df = meta_dataset.copy()
        predictions_df["predicted_accuracy"] = np.nan
        predictions_df["predicted_f1_score"] = np.nan
        predictions_df["predicted_auc_roc"] = np.nan
        # Make predictions
        predicted_acc = xgb_model_acc_final.predict(X_meta_scaled_valid)
        predicted_f1 = xgb_model_f1_final.predict(X_meta_scaled_valid)
        predicted_auc = xgb_model_auc_final.predict(X_meta_scaled_valid)
        # Assign predictions to the appropriate rows
        predictions_df.loc[valid_splits, "predicted_accuracy"] = predicted_acc
        predictions_df.loc[valid_splits, "predicted_f1_score"] = predicted_f1
        predictions_df.loc[valid_splits, "predicted_auc_roc"] = predicted_auc
        for idx, row in predictions_df[valid_splits].iterrows():
            print(
                f"{row['dataset_name']} - {row['model_name']} ({row['split']}):"
            )
            print(f"  Predicted Accuracy: {row['predicted_accuracy']:.4f}")
            print(f"  Actual Accuracy: {row['accuracy']:.4f}")
            print(f"  Predicted F1-Score: {row['predicted_f1_score']:.4f}")
            print(f"  Actual F1-Score: {row['f1_score']:.4f}")
            print(f"  Predicted AUC-ROC: {row['predicted_auc_roc']:.4f}")
            print(f"  Actual AUC-ROC: {row['auc_roc']:.4f}")
        print(
            "\nStep 7: Compile predictions and compare predicted metrics with actual metrics."
        )
        print(
            predictions_df[
                [
                    "dataset_name",
                    "model_name",
                    "split",
                    "predicted_accuracy",
                    "accuracy",
                    "predicted_f1_score",
                    "f1_score",
                    "predicted_auc_roc",
                    "auc_roc",
                ]
            ]
        )
        print("\nStep 8: Evaluate the meta-models' performance.")
        # Compute absolute errors
        valid_predictions_df = predictions_df[valid_splits].copy()
        valid_predictions_df["acc_abs_error"] = abs(
            valid_predictions_df["predicted_accuracy"]
            - valid_predictions_df["accuracy"]
        )
        valid_predictions_df["f1_abs_error"] = abs(
            valid_predictions_df["predicted_f1_score"]
            - valid_predictions_df["f1_score"]
        )
        valid_predictions_df["auc_abs_error"] = abs(
            valid_predictions_df["predicted_auc_roc"]
            - valid_predictions_df["auc_roc"]
        )
        mean_acc_abs_error_final = valid_predictions_df["acc_abs_error"].mean()
        mean_f1_abs_error_final = valid_predictions_df["f1_abs_error"].mean()
        mean_auc_abs_error_final = valid_predictions_df["auc_abs_error"].mean()
        print(
            f"\nMean Absolute Error of Accuracy Meta-Model: {mean_acc_abs_error_final:.4f}"
        )
        print(
            f"Mean Absolute Error of F1-Score Meta-Model: {mean_f1_abs_error_final:.4f}"
        )
        print(
            f"Mean Absolute Error of AUC-ROC Meta-Model: {mean_auc_abs_error_final:.4f}"
        )
        # Log evaluation metrics in MLflow
        mlflow.log_metric("mean_acc_abs_error_final", mean_acc_abs_error_final)
        mlflow.log_metric("mean_f1_abs_error_final", mean_f1_abs_error_final)
        mlflow.log_metric("mean_auc_abs_error_final", mean_auc_abs_error_final)
        # Save the predictions DataFrame to a CSV file
        output_file_path = "meta_model_predictions.csv"
        predictions_df.to_csv(output_file_path, index=False)
        print(f"\nPredictions saved to {output_file_path}")
        # Log the CSV file as an artifact in MLflow
        mlflow.log_artifact(output_file_path)
        print(f"Predictions CSV saved and logged to MLflow as an artifact.")
        data_split_ratio = "Training:60%, Validation:20%, Test:20%"
        # Plot comparisons for splits
        for split in ["val", "test"]:
            for metric in ["accuracy", "f1_score", "auc_roc"]:
                plot_metric_comparison(
                    predictions_df,
                    metric,
                    split=split,
                    data_split_ratio=data_split_ratio,
                    mlflow=mlflow,
                )


if __name__ == "__main__":
    main()
