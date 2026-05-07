import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
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


class ModelBuilder:
    def __init__(self):
        self.models = {}
        self._current_model_name = None

    def add_model(self, name: str, classifier):
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", classifier)])
        self.models[name] = {"pipeline": pipeline, "params": {}}
        self._current_model_name = name
        return self

    def add_parameter(self, param_name: str, values: list):
        if self._current_model_name is None:
            raise ValueError("Must call add_model before add_parameter.")

        formatted_key = f"model__{param_name}"
        self.models[self._current_model_name]["params"][formatted_key] = values
        return self

    def build(self) -> dict:
        return {
            name: (info["pipeline"], info["params"])
            for name, info in self.models.items()
        }


class FetalHealthPipeline:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.models = self._initialize_models()
        self.results = []
        self.confusion_matrices = {}

    def _initialize_models(self) -> dict:
        builder = ModelBuilder()

        # fmt: off
        builder.add_model("Decision Tree", DecisionTreeClassifier(random_state=42)) \
               .add_parameter("criterion", ["gini", "entropy"]) \
               .add_parameter("max_depth", [3, 5, 10])
               
        builder.add_model("KNN", KNeighborsClassifier(n_jobs=-1)) \
               .add_parameter("n_neighbors", [3, 5, 7]) \
               .add_parameter("weights", ["uniform", "distance"])
               
        builder.add_model("Neural Network", MLPClassifier(max_iter=1000, random_state=42)) \
               .add_parameter("hidden_layer_sizes", [(50,), (100,)]) \
               .add_parameter("activation", ["relu", "tanh"])
               
        builder.add_model("Linear Regression (Ridge)", RidgeClassifier(random_state=42)) \
               .add_parameter("alpha", [0.1, 1.0, 10.0]) \
               .add_parameter("class_weight", [None, "balanced"])
               
        builder.add_model("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)) \
               .add_parameter("penalty", ["l1", "l2"]) \
               .add_parameter("solver", ["saga"]) \
               .add_parameter("class_weight", [None, "balanced"])
               
        builder.add_model("Random Forest", RandomForestClassifier(random_state=42, n_jobs=-1)) \
               .add_parameter("n_estimators", [50, 100]) \
               .add_parameter("max_depth", [None, 10, 20])
               
        builder.add_model("Gradient Boosting", GradientBoostingClassifier(random_state=42)) \
               .add_parameter("n_estimators", [50, 100]) \
               .add_parameter("learning_rate", [0.1, 0.2])
               
        builder.add_model("Ada Boosting", AdaBoostClassifier(random_state=42)) \
               .add_parameter("n_estimators", [50, 100]) \
               .add_parameter("learning_rate", [0.1, 1.0])

        # fmt: on

        return builder.build()

    def load_and_preprocess(self):
        df = pd.read_csv(self.filepath)
        X = df.drop(columns=["fetal_health"])
        y = df["fetal_health"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    def train_and_evaluate(self):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cost_scorer = make_scorer(fetal_health_cost, greater_is_better=False)

        for name, (pipeline, params) in self.models.items():
            grid = GridSearchCV(pipeline, params, cv=cv, scoring=cost_scorer, n_jobs=-1)
            grid.fit(self.X_train, self.y_train)

            best_model = grid.best_estimator_
            y_pred = best_model.predict(self.X_test)

            self._record_metrics(name, y_pred, grid.best_params_)

    def _record_metrics(self, model_name: str, y_pred, best_params: dict):
        acc = accuracy_score(self.y_test, y_pred)
        rec = recall_score(self.y_test, y_pred, average="macro")
        f1 = f1_score(self.y_test, y_pred, average="macro")
        f2 = fbeta_score(self.y_test, y_pred, beta=2, average="macro")
        avg_cost = fetal_health_cost(self.y_test, y_pred)

        self.results.append(
            {
                "Model": model_name,
                "Accuracy": acc,
                "Recall": rec,
                "F1-Score": f1,
                "F2-Score": f2,
                "Avg Penalty Cost": avg_cost,  # Lower is better
                "Best Parameters": str(best_params),
            }
        )
        self.confusion_matrices[model_name] = confusion_matrix(self.y_test, y_pred)

    def export_results_to_csv(self, filename: str = "model_results.csv"):
        pd.DataFrame(self.results).to_csv(filename, index=False)

    def plot_confusion_matrices(self, filename: str = "confusion_matrices.png"):
        num_models = len(self.confusion_matrices)
        cols = 3
        rows = (num_models + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        axes = axes.flatten()
        for idx, (name, cm) in enumerate(self.confusion_matrices.items()):
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx], cbar=False)
            axes[idx].set_title(name)
        for i in range(num_models, len(axes)):
            fig.delaxes(axes[i])
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


if __name__ == "__main__":
    pipeline = FetalHealthPipeline("fetal_health.csv")
    pipeline.load_and_preprocess()
    pipeline.train_and_evaluate()
    pipeline.export_results_to_csv()
    pipeline.plot_confusion_matrices()
