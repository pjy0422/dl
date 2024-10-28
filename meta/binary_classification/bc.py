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


class MLflowManager:
    def __init__(
        self, host="127.0.0.1", port=3060, experiment_name="Meta-Model"
    ):
        self.host = host
        self.port = port
        self.experiment_name = experiment_name
        self.run = None

    def run_mlflow_server(self):
        try:
            subprocess.Popen(
                [
                    "mlflow",
                    "server",
                    "--host",
                    self.host,
                    "--port",
                    str(self.port),
                ]
            )
            print(f"MLflow server started on http://{self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to start MLflow server: {e}")

    def init_mlflow(self):
        self.run_mlflow_server()
        mlflow.set_tracking_uri(uri=f"http://{self.host}:{self.port}")
        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name):
        self.run = mlflow.start_run(run_name=run_name)
        print(f"MLflow run '{run_name}' started.")

    def end_run(self):
        if self.run:
            mlflow.end_run()
            print("MLflow run ended.")

    def log_param(self, key, value):
        mlflow.log_param(key, value)

    def log_metric(self, key, value):
        mlflow.log_metric(key, value)

    def log_artifact(self, file_path):
        mlflow.log_artifact(file_path)

    def log_model(self, model, artifact_path):
        mlflow.sklearn.log_model(model, artifact_path)


class DatasetLoader:
    def __init__(self):
        self.datasets = []

    def load_credit_card_fraud(
        self, filepath="creditcard.csv", sample_size=10000, random_state=42
    ):
        print("Loading Credit Card Fraud Detection dataset...")
        try:
            cc_data = pd.read_csv(filepath)
            cc_data_sampled = cc_data.sample(
                n=sample_size, random_state=random_state
            )
            X_cc = cc_data_sampled.drop("Class", axis=1)
            y_cc = cc_data_sampled["Class"]
            self.datasets.append(("Credit Card Fraud", X_cc, y_cc))
            print(
                "Credit Card Fraud Detection dataset loaded and sampled successfully."
            )
        except FileNotFoundError:
            print(
                f"Credit Card Fraud Detection dataset not found at '{filepath}'."
            )

    def load_breast_cancer(self):
        print("Loading Breast Cancer dataset...")
        breast_cancer = load_breast_cancer()
        X_bc = pd.DataFrame(
            breast_cancer.data, columns=breast_cancer.feature_names
        )
        y_bc = pd.Series(breast_cancer.target)
        self.datasets.append(("Breast Cancer", X_bc, y_bc))
        print("Breast Cancer dataset loaded successfully.")

    def load_diabetes(self, filepath="diabetes.csv"):
        print("Loading Diabetes dataset...")
        try:
            diabetes = pd.read_csv(filepath)
            X_diabetes = diabetes.drop("Outcome", axis=1)
            y_diabetes = diabetes["Outcome"]
            self.datasets.append(("Diabetes", X_diabetes, y_diabetes))
            print("Diabetes dataset loaded successfully.")
        except FileNotFoundError:
            print(f"Diabetes dataset not found at '{filepath}'.")

    def load_titanic(
        self,
        url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    ):
        print("Downloading and loading Titanic dataset...")
        try:
            titanic = pd.read_csv(url)
            titanic = titanic.drop(
                ["PassengerId", "Name", "Ticket", "Cabin"], axis=1
            )
            titanic["Age"].fillna(titanic["Age"].median(), inplace=True)
            titanic["Embarked"].fillna(
                titanic["Embarked"].mode()[0], inplace=True
            )
            titanic = pd.get_dummies(
                titanic, columns=["Sex", "Embarked"], drop_first=True
            )
            X_titanic = titanic.drop("Survived", axis=1)
            y_titanic = titanic["Survived"]
            self.datasets.append(("Titanic", X_titanic, y_titanic))
            print("Titanic dataset loaded and preprocessed successfully.")
        except Exception as e:
            print(f"Error loading Titanic dataset: {e}")

    def load_adult_income(
        self,
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    ):
        print("Downloading and loading Adult Income dataset...")
        try:
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
                url,
                names=adult_column_names,
                na_values=" ?",
                skipinitialspace=True,
            )
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
            self.datasets.append(("Adult Income", X_adult, y_adult))
            print("Adult Income dataset loaded and preprocessed successfully.")
        except Exception as e:
            print(f"Error loading Adult Income dataset: {e}")

    def load_heart_disease(self, filepath="heart.csv"):
        print("Downloading and loading Heart Disease dataset...")
        try:
            heart = pd.read_csv(filepath)
            X_heart = heart.drop("target", axis=1)
            y_heart = heart["target"]
            self.datasets.append(("Heart Disease", X_heart, y_heart))
            print("Heart Disease dataset loaded successfully.")
        except Exception as e:
            print(f"Error loading Heart Disease dataset: {e}")

    def load_all_datasets(self):
        self.load_credit_card_fraud()
        self.load_breast_cancer()
        self.load_diabetes()
        self.load_titanic()
        self.load_adult_income()
        self.load_heart_disease()
        print("\nAll datasets loaded.\n")


class ModelTrainer:
    def __init__(self, models):
        self.models = models
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="mean")

    def preprocess(self, X):
        # Handle categorical variables
        if isinstance(X, pd.DataFrame):
            categorical_cols = X.select_dtypes(
                include=["object", "category"]
            ).columns
            if len(categorical_cols) > 0:
                X = pd.get_dummies(
                    X, columns=categorical_cols, drop_first=True
                )
        # Impute missing values
        X_imputed = self.imputer.fit_transform(X)
        # Scale features
        X_scaled = self.scaler.fit_transform(X_imputed)
        return X_scaled

    def split_data(self, X, y, random_state=42):
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=0.25,
            stratify=y_temp,
            random_state=random_state,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_and_evaluate(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        model,
        model_name,
        dataset_name,
        mlflow_manager,
    ):
        print(f"  Training {model_name}...")
        # Log model parameters
        mlflow_manager.log_param(
            f"{dataset_name}_{model_name}_n_samples", X_train.shape[0]
        )
        mlflow_manager.log_param(
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

        print(f"    - Validation Accuracy: {acc_val:.4f}")
        print(f"    - Validation F1-Score: {f1_val:.4f}")
        print(f"    - Validation AUC-ROC: {auc_val:.4f}")

        # Log metrics
        mlflow_manager.log_metric(
            f"{dataset_name}_{model_name}_accuracy_val", acc_val
        )
        mlflow_manager.log_metric(
            f"{dataset_name}_{model_name}_f1_score_val", f1_val
        )
        mlflow_manager.log_metric(
            f"{dataset_name}_{model_name}_auc_roc_val", auc_val
        )

        # Save model
        mlflow_manager.log_model(model, f"{dataset_name}_{model_name}_model")

        performance = {
            "accuracy_val": acc_val,
            "f1_score_val": f1_val,
            "auc_roc_val": auc_val,
        }

        return performance


class MetaFeatureExtractor:
    def __init__(self):
        pass

    def extract_meta_features(self, X_train, y_train):
        meta_features = {}
        meta_features["n_samples"] = X_train.shape[0]
        meta_features["n_features"] = X_train.shape[1]
        meta_features["class_balance"] = y_train.mean()
        meta_features["feature_mean"] = np.mean(X_train)
        meta_features["feature_std"] = np.std(X_train)
        meta_features["coeff_variation"] = (
            (np.std(X_train) / np.mean(X_train))
            if np.mean(X_train) != 0
            else 0
        )

        # PCA
        n_components = min(5, X_train.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        meta_features["pca_explained_variance"] = np.sum(
            pca.explained_variance_ratio_
        )

        # Mutual Information
        mi = mutual_info_classif(
            X_train, y_train, discrete_features=False, random_state=42
        )
        meta_features["avg_mutual_info"] = np.mean(mi)

        # Skewness and Kurtosis
        skewness = skew(X_train, axis=0)
        kurtosis_values = kurtosis(X_train, axis=0)
        meta_features["avg_skewness"] = np.mean(skewness)
        meta_features["avg_kurtosis"] = np.mean(kurtosis_values)

        # Mean Absolute Correlation
        corr_matrix = np.corrcoef(X_train, rowvar=False)
        mask = np.ones(corr_matrix.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        abs_corr = np.abs(corr_matrix[mask])
        meta_features["mean_abs_correlation"] = np.mean(abs_corr)

        # Zero Variance Features
        zero_variance_features = np.sum(np.var(X_train, axis=0) == 0)
        meta_features["n_zero_variance_features"] = zero_variance_features

        # Variance Statistics
        variances = np.var(X_train, axis=0)
        meta_features["mean_variance"] = np.mean(variances)
        meta_features["median_variance"] = np.median(variances)

        # Feature Entropy
        feature_entropies = [
            entropy(np.histogram(X_train[:, i], bins=10)[0] + 1e-10)
            for i in range(X_train.shape[1])
        ]
        meta_features["mean_feature_entropy"] = np.mean(feature_entropies)

        return meta_features


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


class MetaModelManager:
    def __init__(self):
        pass

    def train_model_nn(self, model, optimizer, X, y, num_epochs=500):
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X)
            loss = torch.sqrt(nn.MSELoss()(outputs, y))
            loss.backward()
            optimizer.step()
        return model

    def train_and_evaluate_meta_model(
        self,
        X_meta_scaled,
        y_meta,
        groups,
        model_type="xgb",
        metric_name="",
        mlflow_manager=None,
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
                meta_model_nn = self.train_model_nn(
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

        if mlflow_manager:
            mlflow_manager.log_metric(
                f"mean_abs_error_{metric_name}_{model_type}", mean_abs_error
            )

        return mean_abs_error

    def train_final_meta_model(self, X_meta_scaled, y_meta, model_type="xgb"):
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
            meta_model_nn_final = self.train_model_nn(
                meta_model_nn_final, optimizer_nn, X_tensor, y_tensor
            )
            return meta_model_nn_final
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class MetaLearningPipeline:
    def __init__(self):
        # Initialize MLflow Manager
        self.mlflow_manager = MLflowManager()
        self.mlflow_manager.init_mlflow()

        # Initialize Dataset Loader
        self.dataset_loader = DatasetLoader()
        self.dataset_loader.load_all_datasets()

        # Define Models
        self.models = {
            "Logistic Regression": LogisticRegression(
                max_iter=200, solver="lbfgs", n_jobs=-1
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=50, random_state=42, n_jobs=-1
            ),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(
                n_neighbors=5, n_jobs=-1
            ),
            "Naive Bayes": GaussianNB(),
            "XGBoost": XGBClassifier(
                use_label_encoder=False, eval_metric="logloss", random_state=42
            ),
        }

        # Initialize Model Trainer
        self.model_trainer = ModelTrainer(self.models)

        # Initialize Meta Feature Extractor
        self.meta_feature_extractor = MetaFeatureExtractor()

        # Initialize Meta Model Manager
        self.meta_model_manager = MetaModelManager()

        # Lists to store meta-data
        self.meta_features_list = []
        self.performance_list = []
        self.test_performance_list = []

    def start(self):
        # Start MLflow run
        random_number = np.random.randint(0, 1000)
        run_name = f"META_RUN_{random_number}"
        self.mlflow_manager.start_run(run_name=run_name)

        # Iterate over each dataset
        for idx, (dataset_name, X, y) in enumerate(
            self.dataset_loader.datasets, 1
        ):
            print(
                f"\nProcessing {dataset_name} ({idx}/{len(self.dataset_loader.datasets)})..."
            )

            # Preprocess data
            X_scaled = self.model_trainer.preprocess(X)
            X_train, X_val, X_test, y_train, y_val, y_test = (
                self.model_trainer.split_data(X_scaled, y)
            )

            # Extract meta-features
            meta_features = self.meta_feature_extractor.extract_meta_features(
                X_train, y_train
            )
            meta_features["dataset_name"] = dataset_name

            # Iterate over each model
            for model_name, model in self.models.items():
                performance = self.model_trainer.train_and_evaluate(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    model,
                    model_name,
                    dataset_name,
                    self.mlflow_manager,
                )

                # Append performance metrics
                performance_entry = {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "accuracy_val": performance["accuracy_val"],
                    "f1_score_val": performance["f1_score_val"],
                    "auc_roc_val": performance["auc_roc_val"],
                }
                self.performance_list.append(performance_entry)

                # Append meta-features
                meta_features_entry = meta_features.copy()
                meta_features_entry["model_name"] = model_name
                self.meta_features_list.append(meta_features_entry)

        # Create meta-dataset
        meta_features_df = pd.DataFrame(self.meta_features_list)
        performance_df = pd.DataFrame(self.performance_list)
        meta_dataset = pd.merge(
            meta_features_df, performance_df, on=["dataset_name", "model_name"]
        )
        meta_dataset.to_csv("meta_dataset_binaryclass.csv", index=False)
        print(
            "\nMeta-dataset created and saved to 'meta_dataset_binaryclass.csv'."
        )
        self.mlflow_manager.log_artifact("meta_dataset_binaryclass.csv")

        # Prepare data for meta-models
        X_meta = meta_dataset.drop(
            [
                "dataset_name",
                "model_name",
                "accuracy_val",
                "f1_score_val",
                "auc_roc_val",
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

        # Targets for validation set metrics
        y_meta_acc_val = meta_dataset["accuracy_val"].values
        y_meta_f1_val = meta_dataset["f1_score_val"].values
        y_meta_auc_val = (
            meta_dataset["auc_roc_val"]
            .fillna(meta_dataset["auc_roc_val"].mean())
            .values
        )

        groups = meta_dataset["dataset_name"].values

        # Train and evaluate meta-models
        print("\nTraining and evaluating meta-models for validation metrics.")
        mean_acc_abs_error_val = (
            self.meta_model_manager.train_and_evaluate_meta_model(
                X_meta_scaled,
                y_meta_acc_val,
                groups,
                model_type="xgb",
                metric_name="Accuracy Validation",
                mlflow_manager=self.mlflow_manager,
            )
        )

        mean_f1_abs_error_val = (
            self.meta_model_manager.train_and_evaluate_meta_model(
                X_meta_scaled,
                y_meta_f1_val,
                groups,
                model_type="xgb",
                metric_name="F1-Score Validation",
                mlflow_manager=self.mlflow_manager,
            )
        )

        mean_auc_abs_error_val = (
            self.meta_model_manager.train_and_evaluate_meta_model(
                X_meta_scaled,
                y_meta_auc_val,
                groups,
                model_type="xgb",
                metric_name="AUC-ROC Validation",
                mlflow_manager=self.mlflow_manager,
            )
        )

        # Train final meta-models
        xgb_model_acc_val_final = (
            self.meta_model_manager.train_final_meta_model(
                X_meta_scaled, y_meta_acc_val, model_type="xgb"
            )
        )
        xgb_model_f1_val_final = (
            self.meta_model_manager.train_final_meta_model(
                X_meta_scaled, y_meta_f1_val, model_type="xgb"
            )
        )
        xgb_model_auc_val_final = (
            self.meta_model_manager.train_final_meta_model(
                X_meta_scaled, y_meta_auc_val, model_type="xgb"
            )
        )

        # Save the final models
        self.mlflow_manager.log_model(
            xgb_model_acc_val_final, "final_meta_model_accuracy_val_xgb"
        )
        self.mlflow_manager.log_model(
            xgb_model_f1_val_final, "final_meta_model_f1_score_val_xgb"
        )
        self.mlflow_manager.log_model(
            xgb_model_auc_val_final, "final_meta_model_auc_roc_val_xgb"
        )
        print("Meta-models for validation metrics logged to MLflow.")

        # Create meta-dataset predictions
        print("\nPredicting validation metrics using meta-models.")
        predicted_acc_val = xgb_model_acc_val_final.predict(X_meta_scaled)
        predicted_f1_val = xgb_model_f1_val_final.predict(X_meta_scaled)
        predicted_auc_val = xgb_model_auc_val_final.predict(X_meta_scaled)

        predictions_df = meta_dataset.copy()
        predictions_df["predicted_accuracy_val"] = predicted_acc_val
        predictions_df["predicted_f1_score_val"] = predicted_f1_val
        predictions_df["predicted_auc_roc_val"] = predicted_auc_val

        # Display predictions
        for idx, row in predictions_df.iterrows():
            print(f"{row['dataset_name']} - {row['model_name']}:")
            print(
                f"  Predicted Validation Accuracy: {row['predicted_accuracy_val']:.4f}"
            )
            print(f"  Actual Validation Accuracy: {row['accuracy_val']:.4f}")
            print(
                f"  Predicted Validation F1-Score: {row['predicted_f1_score_val']:.4f}"
            )
            print(f"  Actual Validation F1-Score: {row['f1_score_val']:.4f}")
            print(
                f"  Predicted Validation AUC-ROC: {row['predicted_auc_roc_val']:.4f}"
            )
            print(f"  Actual Validation AUC-ROC: {row['auc_roc_val']:.4f}")

        # Save predictions
        predictions_df.to_csv("meta_model_predictions.csv", index=False)
        self.mlflow_manager.log_artifact("meta_model_predictions.csv")
        print(
            "\nPredictions saved to 'meta_model_predictions.csv' and logged to MLflow."
        )

        # Plot and log comparisons
        self.plot_and_log_comparisons(predictions_df)

        # Evaluate meta-model performance
        print("\nEvaluating the meta-models' performance.")
        predictions_df["acc_abs_error_val"] = abs(
            predictions_df["predicted_accuracy_val"]
            - predictions_df["accuracy_val"]
        )
        predictions_df["f1_abs_error_val"] = abs(
            predictions_df["predicted_f1_score_val"]
            - predictions_df["f1_score_val"]
        )
        predictions_df["auc_abs_error_val"] = abs(
            predictions_df["predicted_auc_roc_val"]
            - predictions_df["auc_roc_val"]
        )

        mean_acc_abs_error_val_final = predictions_df[
            "acc_abs_error_val"
        ].mean()
        mean_f1_abs_error_val_final = predictions_df["f1_abs_error_val"].mean()
        mean_auc_abs_error_val_final = predictions_df[
            "auc_abs_error_val"
        ].mean()

        print(
            f"\nMean Absolute Error of Accuracy Meta-Model on Validation Set: {mean_acc_abs_error_val_final:.4f}"
        )
        print(
            f"Mean Absolute Error of F1-Score Meta-Model on Validation Set: {mean_f1_abs_error_val_final:.4f}"
        )
        print(
            f"Mean Absolute Error of AUC-ROC Meta-Model on Validation Set: {mean_auc_abs_error_val_final:.4f}"
        )

        # Log evaluation metrics
        self.mlflow_manager.log_metric(
            "mean_acc_abs_error_val_final", mean_acc_abs_error_val_final
        )
        self.mlflow_manager.log_metric(
            "mean_f1_abs_error_val_final", mean_f1_abs_error_val_final
        )
        self.mlflow_manager.log_metric(
            "mean_auc_abs_error_val_final", mean_auc_abs_error_val_final
        )

        # Final summary
        print("\nFinal Summary of Meta-Model Performance on Validation Set:")
        print(
            predictions_df[
                [
                    "dataset_name",
                    "model_name",
                    "predicted_accuracy_val",
                    "accuracy_val",
                    "predicted_f1_score_val",
                    "f1_score_val",
                    "predicted_auc_roc_val",
                    "auc_roc_val",
                ]
            ]
        )

        print("\nFinal Evaluation Metrics:")
        print(
            f"Mean Absolute Error of Accuracy Meta-Model: {mean_acc_abs_error_val_final:.4f}"
        )
        print(
            f"Mean Absolute Error of F1-Score Meta-Model: {mean_f1_abs_error_val_final:.4f}"
        )
        print(
            f"Mean Absolute Error of AUC-ROC Meta-Model: {mean_auc_abs_error_val_final:.4f}"
        )

        # Evaluate on Test Set
        print("\nFinal Evaluation on Test Set.")
        self.evaluate_on_test_set()

        # End MLflow run
        self.mlflow_manager.end_run()

    def plot_and_log_comparisons(self, predictions_df):
        def plot_metric_comparison(
            predictions_df, metric, split, data_split_ratio, mlflow_manager
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
            plt.close()

            # Log the plot as an artifact
            mlflow_manager.log_artifact(plot_filename)

            print(
                f"{metric.capitalize()} comparison plot for {split} set saved and logged to MLflow."
            )

        data_split_ratio = "Training:60%, Validation:20%, Test:20%"

        # Plot comparisons for validation split
        for metric in ["accuracy", "f1_score", "auc_roc"]:
            plot_metric_comparison(
                predictions_df,
                metric,
                split="val",
                data_split_ratio=data_split_ratio,
                mlflow_manager=self.mlflow_manager,
            )

    def evaluate_on_test_set(self):
        print("\nEvaluating base models on the test set...")
        for dataset_name, X, y in self.dataset_loader.datasets:
            print(f"\nDataset: {dataset_name}")
            # Preprocess data
            X_scaled = self.model_trainer.preprocess(X)
            X_train, X_val, X_test, y_train, y_val, y_test = (
                self.model_trainer.split_data(X_scaled, y)
            )

            for model_name, model in self.models.items():
                print(f"  Evaluating {model_name} on test set...")
                # Train model on train set
                model.fit(X_train, y_train)
                # Predict on test set
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

                print(f"    - Test Accuracy: {acc_test:.4f}")
                print(f"    - Test F1-Score: {f1_test:.4f}")
                print(f"    - Test AUC-ROC: {auc_test:.4f}")

                # Log test metrics
                self.mlflow_manager.log_metric(
                    f"{dataset_name}_{model_name}_accuracy_test", acc_test
                )
                self.mlflow_manager.log_metric(
                    f"{dataset_name}_{model_name}_f1_score_test", f1_test
                )
                self.mlflow_manager.log_metric(
                    f"{dataset_name}_{model_name}_auc_roc_test", auc_test
                )

                # Append test performance
                test_performance = {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "accuracy_test": acc_test,
                    "f1_score_test": f1_test,
                    "auc_roc_test": auc_test,
                }
                self.test_performance_list.append(test_performance)

        # Save test performance
        test_performance_df = pd.DataFrame(self.test_performance_list)
        test_performance_df.to_csv(
            "test_performance_binaryclass.csv", index=False
        )
        self.mlflow_manager.log_artifact("test_performance_binaryclass.csv")
        print(
            "\nTest performance metrics saved to 'test_performance_binaryclass.csv' and logged to MLflow."
        )


if __name__ == "__main__":
    pipeline = MetaLearningPipeline()
    pipeline.start()
