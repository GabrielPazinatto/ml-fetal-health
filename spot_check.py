import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
N_JOBS = 1

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    cross_validate,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    fbeta_score,
    make_scorer,
    confusion_matrix,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    StackingClassifier,
)
import warnings

warnings.filterwarnings("ignore")


def get_spot_check_models():
    models = {
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=42
        ),
        "KNN": KNeighborsClassifier(n_jobs=N_JOBS),
        "Neural Network": MLPClassifier(max_iter=1000, random_state=42),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42, n_jobs=N_JOBS
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=N_JOBS
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Ada Boosting": AdaBoostClassifier(random_state=42),
    }

    stacking_estimators = [
        (
            "rf",
            RandomForestClassifier(
                class_weight="balanced", random_state=42, n_jobs=N_JOBS
            ),
        ),
        ("gb", GradientBoostingClassifier(random_state=42)),
    ]
    models["Stacking (RF+GB)"] = StackingClassifier(
        estimators=stacking_estimators,
        final_estimator=LogisticRegression(class_weight="balanced", random_state=42),
        n_jobs=N_JOBS,
    )

    pipelines = {}
    for name, model in models.items():
        pipelines[name] = Pipeline([("scaler", StandardScaler()), ("model", model)])

    return pipelines


class FetalHealthSpotCheck:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.models = get_spot_check_models()
        self.results = []
        self.cv_fold_results = []
        self.confusion_matrices = {}
        self.duplicate_rows_removed = 0

    def load_and_preprocess(self):
        df = pd.read_csv(self.filepath)
        self.duplicate_rows_removed = int(df.duplicated().sum())
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Removed {self.duplicate_rows_removed} duplicate rows.")

        X = df.drop(columns=["fetal_health"])
        y = df["fetal_health"].astype(int)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return self

    def evaluate_models(self):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = {
            "accuracy": "accuracy",
            "recall": make_scorer(recall_score, average="macro"),
            "f1": make_scorer(f1_score, average="macro"),
            "f2": make_scorer(fbeta_score, beta=2, average="macro"),
        }

        for name, pipeline in self.models.items():
            print(f"Spot-checking {name}...")

            cv_results = cross_validate(
                pipeline,
                self.X_train,
                self.y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=N_JOBS,
            )

            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)

            self._record_metrics(name, cv_results, y_pred)

        return self

    def _record_metrics(self, model_name: str, cv_results: dict, y_pred):
        test_acc = accuracy_score(self.y_test, y_pred)
        test_rec = recall_score(self.y_test, y_pred, average="macro")
        test_f1 = f1_score(self.y_test, y_pred, average="macro")
        test_f2 = fbeta_score(self.y_test, y_pred, beta=2, average="macro")

        cv_metrics = {
            "Accuracy": cv_results["test_accuracy"],
            "Recall": cv_results["test_recall"],
            "F1-Score": cv_results["test_f1"],
            "F2-Score": cv_results["test_f2"],
        }

        for fold_idx in range(len(cv_results["test_accuracy"])):
            self.cv_fold_results.append(
                {
                    "Model": model_name,
                    "Fold": fold_idx + 1,
                    "Accuracy": cv_metrics["Accuracy"][fold_idx],
                    "Recall": cv_metrics["Recall"][fold_idx],
                    "F1-Score": cv_metrics["F1-Score"][fold_idx],
                    "F2-Score": cv_metrics["F2-Score"][fold_idx],
                }
            )

        summary = {}
        for metric_name, values in cv_metrics.items():
            summary[f"CV {metric_name} Mean"] = np.mean(values)
            summary[f"CV {metric_name} Std"] = np.std(values, ddof=1)

        self.results.append(
            {
                "Model": model_name,
                **summary,
                "Test Accuracy": test_acc,
                "Test Recall": test_rec,
                "Test F1-Score": test_f1,
                "Test F2-Score": test_f2,
            }
        )
        self.confusion_matrices[model_name] = confusion_matrix(
            self.y_test, y_pred, labels=[1, 2, 3]
        )

    def export_results_to_csv(self, filename: str = "spot_check_results.csv"):
        df_results = pd.DataFrame(self.results)
        df_results = df_results.sort_values(by="CV F2-Score Mean", ascending=False)
        df_results.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")
        return self

    def export_cv_fold_results_to_csv(
        self, filename: str = "spot_check_cv_fold_results.csv"
    ):
        df_results = pd.DataFrame(self.cv_fold_results)
        df_results = df_results.sort_values(by=["Model", "Fold"])
        df_results.to_csv(filename, index=False)
        print(f"Fold-level CV results saved to {filename}")
        return self

    def plot_cv_metric_distribution(
        self,
        metric: str = "F2-Score",
        filename: str = "spot_check_cv_f2_distribution.png",
    ):
        df_results = pd.DataFrame(self.cv_fold_results)
        model_order = (
            pd.DataFrame(self.results)
            .sort_values(by=f"CV {metric} Mean", ascending=False)["Model"]
            .tolist()
        )

        plt.figure(figsize=(14, 7))
        sns.boxplot(data=df_results, x="Model", y=metric, order=model_order)
        sns.stripplot(
            data=df_results,
            x="Model",
            y=metric,
            order=model_order,
            color="black",
            alpha=0.55,
            size=4,
        )
        plt.title(f"Cross-validation distribution by model: {metric}")
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"CV metric distribution plot saved to {filename}")
        return self

    def plot_confusion_matrices(
        self, filename: str = "spot_check_confusion_matrices.png"
    ):
        num_models = len(self.confusion_matrices)
        cols = 3
        rows = (num_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        axes = axes.flatten()

        class_labels = ["Normal", "Suspect", "Pathological"]

        for idx, (name, cm) in enumerate(self.confusion_matrices.items()):
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=axes[idx],
                cbar=False,
                xticklabels=class_labels,
                yticklabels=class_labels,
            )
            axes[idx].set_title(name)
            axes[idx].set_xlabel("Predicted")
            axes[idx].set_ylabel("Actual")

        for i in range(num_models, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        return self


if __name__ == "__main__":
    (
        FetalHealthSpotCheck("fetal_health.csv")
        .load_and_preprocess()
        .evaluate_models()
        .export_results_to_csv()
        .export_cv_fold_results_to_csv()
        .plot_cv_metric_distribution()
        .plot_confusion_matrices()
    )
