import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    StratifiedKFold,
    ParameterGrid,
    GridSearchCV,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import mlflow
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    fbeta_score,
    make_scorer,
    confusion_matrix,
)
from sklearn.svm import SVC
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

        # Initialize MLflow experiment
        mlflow.set_experiment("Fetal_Health_Classification")
        self.experiment = mlflow.get_experiment_by_name("Fetal_Health_Classification")

        self.optimized_base_estimators = {}

        self.models = self._initialize_models()
        self.results = []
        self.confusion_matrices = {}

    def _initialize_models(self) -> dict:
        builder = ModelBuilder()

        # fmt: off
        builder.add_model("Decision Tree", DecisionTreeClassifier(random_state=42)) \
               .add_parameter("criterion", ["gini", "entropy"]) \
               .add_parameter("max_depth", [7, 10, 11, 13, 14]) \
               .add_parameter("min_samples_split", [1, 2, 3, 5, 7, 10, 20]) \
               .add_parameter("min_samples_leaf", [1, 2, 3, 5]) \
               .add_parameter("class_weight", ["balanced"])
               
        builder.add_model("KNN", KNeighborsClassifier(n_jobs=-1)) \
               .add_parameter("n_neighbors", [3, 5, 7, 11, 15]) \
               .add_parameter("weights", ["uniform", "distance"]) \
               .add_parameter("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
               
        builder.add_model("Neural Network", MLPClassifier(max_iter=1000, random_state=42)) \
               .add_parameter("hidden_layer_sizes", [(20,),(50,),(25, 10), (50, 20)]) \
               .add_parameter("learning_rate_init", [0.001, 0.01, 0.05, 0.1]) \
               .add_parameter("solver", ["lbfgs"]) \
               .add_parameter("learning_rate", ["constant", "adaptive", "invscaling"]) \
               .add_parameter("activation", ["relu", "tanh"]) \
               .add_parameter("alpha", [0.001, 0.01])
               
        builder.add_model("Linear Regression (Ridge)", RidgeClassifier(random_state=42)) \
               .add_parameter("alpha", [0.1, 0.5, 1.0, 3.0, 10.0]) \
               .add_parameter("class_weight", [None, "balanced"])
               
        builder.add_model("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)) \
               .add_parameter("penalty", ["l1", "l2"]) \
               .add_parameter("solver", ["saga"]) \
               .add_parameter("class_weight", ["balanced"]) \
               .add_parameter("C", [0.01, 0.1, 1.0, 10.0])
               
        builder.add_model("Random Forest", RandomForestClassifier(random_state=42, n_jobs=-1)) \
               .add_parameter("n_estimators", [50, 100, 150, 200]) \
               .add_parameter("class_weight", [None, "balanced", "balanced_subsample"]) \
               .add_parameter("max_depth", [None, 15, 20, 25, 30])
               
        builder.add_model("Gradient Boosting", GradientBoostingClassifier(random_state=42)) \
               .add_parameter("n_estimators", [350, 400, 450]) \
               .add_parameter("learning_rate", [0.005, 0.01, 0.05, 0.1]) \
               .add_parameter("max_depth", [3, 5, 7])
               
        builder.add_model("Ada Boosting", AdaBoostClassifier(random_state=42)) \
               .add_parameter("n_estimators", [50, 100]) \
               .add_parameter("learning_rate", [0.1, 1.0])

        rf_opt = self.optimized_base_estimators.get("Random Forest", RandomForestClassifier(random_state=42))
        gb_opt = self.optimized_base_estimators.get("Gradient Boosting", GradientBoostingClassifier(random_state=42))
        knn_opt = self.optimized_base_estimators.get("KNN", KNeighborsClassifier())
        gb_opt = self.optimized_base_estimators.get("Gradient Boosting", GradientBoostingClassifier(random_state=42))
        ridge_opt = self.optimized_base_estimators.get("Linear Regression (Ridge)", RidgeClassifier(random_state=42))
        nn_opt = self.optimized_base_estimators.get("Neural Network", MLPClassifier(random_state=42))
        lr_opt = self.optimized_base_estimators.get("Logistic Regression", LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1))

        stacking_estimators = [
            ("rf", rf_opt),
            ("gb", gb_opt)
        ]
        
        builder.add_model("Stacking Classifier rf_gb", StackingClassifier(
                                estimators=stacking_estimators, 
                                final_estimator=lr_opt,
                                cv=5,
                                n_jobs=-1
                           )) \
               .add_parameter("final_estimator__C", [0.01, 0.1, 1.0, 10.0]) \
               .add_parameter("final_estimator__penalty", ["l2"])
               
        stacking_estimators_2 = [
            ("rf", rf_opt),
            ("gb", gb_opt),
            ("nn", nn_opt),
        ]
        
        builder.add_model("Stacking Classifier rf_gb_nn", StackingClassifier(
                                estimators=stacking_estimators_2, 
                                final_estimator=LogisticRegression(class_weight="balanced", random_state=42, solver='saga'),
                                cv=5,
                                n_jobs=-1
                           )) \
               .add_parameter("final_estimator__C", [0.05, 0.1, 0.5]) \
               .add_parameter("final_estimator__penalty", ["l1", "l2"]) \
               .add_parameter("passthrough", [False, True]) 

        stacking_estimators_diverse = [
            ("gb", gb_opt),
            ("knn", knn_opt),
            ("ridge", ridge_opt),
        ]
        
        builder.add_model("Stacking Classifier Diverse", StackingClassifier(
                                estimators=stacking_estimators_diverse, 
                                final_estimator=LogisticRegression(class_weight="balanced", random_state=42, solver='saga'),
                                cv=5,
                                n_jobs=-1
                           )) \
               .add_parameter("final_estimator__C", [0.05, 0.1, 0.5]) \
               .add_parameter("passthrough", [False, True])
               
        builder.add_model("Stacking Classifier gb_knn_ridge", StackingClassifier(
                                estimators=stacking_estimators_2, 
                                final_estimator=RidgeClassifier(class_weight="balanced", random_state=42),
                                cv=5,
                                n_jobs=-1
                           )) \
               .add_parameter("final_estimator__alpha", [0.1, 1.0, 10.0]) \
               .add_parameter("passthrough", [False, True])

        # fmt: on

        return builder.build()

    def load_and_preprocess(self):
        df = pd.read_csv(self.filepath)
        X = df.drop(columns=["fetal_health"])
        y = df["fetal_health"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        return self

    def _param_dict_to_signature(self, param_dict: dict) -> str:
        return str(sorted([(str(k), str(v)) for k, v in param_dict.items()]))

    def _get_existing_runs(self, model_name: str) -> set:
        if not self.experiment:
            return set()

        df_runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            filter_string=f"tags.model_name = '{model_name}'",
        )

        if df_runs.empty:
            return set()

        existing_signatures = set()
        param_cols = [c for c in df_runs.columns if c.startswith("params.")]

        for _, row in df_runs.iterrows():
            run_params = {}
            for col in param_cols:
                val = row[col]
                if pd.notna(val):
                    run_params[col.replace("params.", "")] = val
            existing_signatures.add(self._param_dict_to_signature(run_params))

        return existing_signatures

    def train_and_evaluate(self):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = self._get_scoring_metrics()

        for name, (pipeline, params) in self.models.items():
            print(f"Checking configurations for {name}...")

            param_grid = list(ParameterGrid(params))
            untested_grids = self._filter_untested_grids(name, param_grid)

            if untested_grids:
                self._train_and_log_new_configs(
                    name, pipeline, untested_grids, cv, scoring
                )
            else:
                print(
                    f"All configurations for {name} are already logged. Skipping training."
                )

            self._evaluate_and_record_best_model(name, pipeline, param_grid)

        return self

    def _get_scoring_metrics(self) -> dict:
        return {
            "cost": make_scorer(fetal_health_cost, greater_is_better=False),
            "accuracy": "accuracy",
            "recall": make_scorer(recall_score, average="macro"),
            "f1": make_scorer(f1_score, average="macro"),
            "f2": make_scorer(fbeta_score, beta=2, average="macro"),
        }

    def _filter_untested_grids(self, model_name: str, param_grid: list) -> list:
        existing_runs = self._get_existing_runs(model_name)

        untested_params = [
            p
            for p in param_grid
            if self._param_dict_to_signature(p) not in existing_runs
        ]

        return [{k: [v] for k, v in p.items()} for p in untested_params]

    def _train_and_log_new_configs(
        self, model_name: str, pipeline, untested_grids, cv, scoring
    ):
        print(f"Training {len(untested_grids)} new configurations for {model_name}...")

        grid = GridSearchCV(
            pipeline,
            untested_grids,
            cv=cv,
            scoring=scoring,
            refit="cost",
            n_jobs=-1,
        )
        grid.fit(self.X_train, self.y_train)

        print(f"Logging {model_name} batches to MLflow...\n")
        self._log_grid_results(model_name, grid.cv_results_)

    def _log_grid_results(self, model_name: str, cv_results: dict):
        for i in range(len(cv_results["params"])):
            run_params = cv_results["params"][i]
            run_metrics = {
                "cv_cost": -cv_results["mean_test_cost"][i],
                "cv_accuracy": cv_results["mean_test_accuracy"][i],
                "cv_recall": cv_results["mean_test_recall"][i],
                "cv_f1": cv_results["mean_test_f1"][i],
                "cv_f2": cv_results["mean_test_f2"][i],
            }

            with mlflow.start_run(
                experiment_id=self.experiment.experiment_id,
                tags={"model_name": model_name},
            ):
                log_params = {k: str(v) for k, v in run_params.items()}
                mlflow.log_params(log_params)
                mlflow.log_metrics(run_metrics)

    def _evaluate_and_record_best_model(
        self, model_name: str, pipeline, param_grid: list
    ):
        df_all = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            filter_string=f"tags.model_name = '{model_name}'",
        )

        if df_all.empty:
            return

        best_run = df_all.sort_values("metrics.cv_cost", ascending=True).iloc[0]

        best_run_sig = self._param_dict_to_signature(
            {
                k.replace("params.", ""): v
                for k, v in best_run.items()
                if k.startswith("params.") and pd.notna(v)
            }
        )

        best_param_dict = next(
            (p for p in param_grid if self._param_dict_to_signature(p) == best_run_sig),
            None,
        )

        if best_param_dict is None:
            best_param_dict = self._coerce_logged_params(best_run)

        pipeline.set_params(**best_param_dict)
        pipeline.fit(self.X_train, self.y_train)
        y_best_pred = pipeline.predict(self.X_test)

        self.optimized_base_estimators[model_name] = pipeline.named_steps["model"]

        self._record_metrics(
            model_name, y_best_pred, best_param_dict, best_run["metrics.cv_cost"]
        )

    def _param_dict_to_signature(self, param_dict: dict) -> str:
        return str(sorted([(str(k), str(v)) for k, v in param_dict.items()]))

    def _coerce_logged_params(self, run_row: pd.Series) -> dict:
        coerced = {}

        for key, value in run_row.items():
            if not key.startswith("params.") or pd.isna(value):
                continue

            param_name = key.replace("params.", "")
            if isinstance(value, str):
                lowered = value.lower()
                if lowered == "none":
                    coerced[param_name] = None
                    continue
                if lowered == "true":
                    coerced[param_name] = True
                    continue
                if lowered == "false":
                    coerced[param_name] = False
                    continue

                try:
                    coerced[param_name] = ast.literal_eval(value)
                    continue
                except (ValueError, SyntaxError):
                    pass

                try:
                    coerced[param_name] = int(value)
                    continue
                except ValueError:
                    pass

                try:
                    coerced[param_name] = float(value)
                    continue
                except ValueError:
                    pass

            coerced[param_name] = value

        return coerced

    def _record_metrics(
        self, model_name: str, y_pred, best_params: dict, cv_cost: float
    ):
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
                "Avg Penalty Cost": avg_cost,
                "CV Cost": cv_cost,
                "Best Parameters": str(best_params),
            }
        )
        self.confusion_matrices[model_name] = confusion_matrix(self.y_test, y_pred)

    def export_results_to_csv(self, filename: str = "model_results.csv"):
        pd.DataFrame(self.results).to_csv(filename, index=False)
        return self

    def plot_confusion_matrices(self, filename: str = "confusion_matrices.png"):
        num_models = len(self.confusion_matrices)
        cols = 3
        rows = (num_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        axes = axes.flatten()

        class_labels = ["Normal", "Suspeito", "Patológico"]

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
            axes[idx].set_xlabel("Previsão")
            axes[idx].set_ylabel("Valor Real")

        # Hide empty subplots
        for i in range(num_models, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        return self


if __name__ == "__main__":
    # fmt: off
    FetalHealthPipeline("fetal_health.csv") \
    .load_and_preprocess() \
    .train_and_evaluate() \
    .export_results_to_csv() \
    .plot_confusion_matrices()
