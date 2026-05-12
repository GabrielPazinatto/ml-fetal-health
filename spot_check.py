import pandas as pd
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
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    StackingClassifier,
)
import warnings

warnings.filterwarnings("ignore")


def fetal_health_cost(y_true, y_pred):
    cost_matrix = np.array(
        [
            [0, 1, 2],
            [3, 0, 1],
            [10, 5, 0],
        ]
    )
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])
    total_cost = np.sum(cm * cost_matrix)
    return total_cost / len(y_true)


def get_spot_check_models():
    models = {
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=42
        ),
        "KNN": KNeighborsClassifier(n_jobs=-1),
        "Neural Network": MLPClassifier(max_iter=1000, random_state=42),
        "Ridge": RidgeClassifier(class_weight="balanced", random_state=42),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Ada Boosting": AdaBoostClassifier(random_state=42),
    }

    stacking_estimators = [
        (
            "rf",
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        ),
        ("gb", GradientBoostingClassifier(random_state=42)),
    ]
    models["Stacking (RF+GB)"] = StackingClassifier(
        estimators=stacking_estimators,
        final_estimator=LogisticRegression(class_weight="balanced", random_state=42),
        n_jobs=-1,
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
        self.confusion_matrices = {}

    def load_and_preprocess(self):
        df = pd.read_csv(self.filepath)
        X = df.drop(columns=["fetal_health"])
        y = df["fetal_health"].astype(int)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return self

    def evaluate_models(self):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = {
            "cost": make_scorer(fetal_health_cost, greater_is_better=False),
            "accuracy": "accuracy",
            "recall": make_scorer(recall_score, average="macro"),
            "f1": make_scorer(f1_score, average="macro"),
            "f2": make_scorer(fbeta_score, beta=2, average="macro"),
        }

        for name, pipeline in self.models.items():
            print(f"Spot-checking {name}...")

            cv_results = cross_validate(
                pipeline, self.X_train, self.y_train, cv=cv, scoring=scoring, n_jobs=-1
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
        test_cost = fetal_health_cost(self.y_test, y_pred)

        self.results.append(
            {
                "Model": model_name,
                "CV Accuracy": np.mean(cv_results["test_accuracy"]),
                "CV Recall": np.mean(cv_results["test_recall"]),
                "CV F1-Score": np.mean(cv_results["test_f1"]),
                "CV F2-Score": np.mean(cv_results["test_f2"]),
                "CV Avg Penalty Cost": -np.mean(
                    cv_results["test_cost"]
                ),  # Revert negative sign
                "Test Accuracy": test_acc,
                "Test Recall": test_rec,
                "Test F1-Score": test_f1,
                "Test F2-Score": test_f2,
                "Test Avg Penalty Cost": test_cost,
            }
        )
        self.confusion_matrices[model_name] = confusion_matrix(self.y_test, y_pred)

    def export_results_to_csv(self, filename: str = "spot_check_results.csv"):
        df_results = pd.DataFrame(self.results)
        # Sort by best (lowest) CV Cost
        df_results = df_results.sort_values(by="CV Avg Penalty Cost")
        df_results.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")
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
    FetalHealthSpotCheck(
        "fetal_health.csv"
    ).load_and_preprocess().evaluate_models().export_results_to_csv().plot_confusion_matrices()
