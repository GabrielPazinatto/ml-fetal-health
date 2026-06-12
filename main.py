import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "10")
N_JOBS = -1
RANDOM_STATE = 42
CLASS_LABELS = [1, 2, 3]
CLASS_NAMES = ["Normal", "Suspeito", "Patologico"]
TOP_N_CONFIGS = 10

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def macro_f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2, average="macro")


class FetalHealthOptimizer:
    def __init__(
        self,
        filepath: str,
        output_dir: str = "t2",
        random_state: int = RANDOM_STATE,
    ):
        self.filepath = filepath
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.scoring = self._build_scoring()
        self.search_results = []
        self.best_model_rows = []
        self.classification_report_rows = []
        self.confusion_matrices = {}
        self.best_estimators = {}
        self.duplicate_rows_removed = 0

    def load_and_preprocess(self):
        df = pd.read_csv(self.filepath)
        self.duplicate_rows_removed = int(df.duplicated().sum())
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Removed {self.duplicate_rows_removed} duplicate rows.")

        X = df.drop(columns=["fetal_health"])
        y = df["fetal_health"].astype(int)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y,
        )
        return self

    def optimize_base_models(self):
        searches = {
            "Random Forest": (
                self._pipeline(
                    RandomForestClassifier(
                        random_state=self.random_state, n_jobs=N_JOBS
                    )
                ),
                self._random_forest_grid(),
                120,
            ),
            "Gradient Boosting": (
                self._pipeline(
                    GradientBoostingClassifier(random_state=self.random_state)
                ),
                self._gradient_boosting_grid(),
                160,
            ),
        }

        for model_name, (pipeline, param_grid, n_iter) in searches.items():
            search = self._run_randomized_search(
                model_name=model_name,
                pipeline=pipeline,
                param_grid=param_grid,
                n_iter=n_iter,
            )
            self._store_search_results(model_name, search, "randomized")

            focused_grid = self._focused_grid_for_model(model_name, search.best_params_)
            focused_search = self._run_grid_search(
                model_name=model_name,
                pipeline=pipeline,
                param_grid=focused_grid,
                search_stage="focused_grid",
            )
            self._store_search_results(model_name, focused_search, "focused_grid")
            self._evaluate_best_model(
                model_name,
                focused_search.best_estimator_,
                focused_search.best_params_,
                focused_search.best_score_,
            )
            self.best_estimators[model_name] = clone(
                focused_search.best_estimator_.named_steps["model"]
            )

        return self

    def optimize_stacking_model(self):
        self._ensure_base_models_are_optimized()

        stacking = StackingClassifier(
            estimators=[
                ("rf", self.best_estimators["Random Forest"]),
                ("gb", self.best_estimators["Gradient Boosting"]),
            ],
            final_estimator=LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=self.random_state,
            ),
            n_jobs=N_JOBS,
        )
        pipeline = self._pipeline(stacking)
        param_grid = {
            "model__final_estimator__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            "model__passthrough": [False, True],
            "model__cv": [3, 5],
        }

        search = self._run_grid_search(
            model_name="Stacking (Tuned RF+GB)",
            pipeline=pipeline,
            param_grid=param_grid,
            search_stage="stacking_grid",
        )
        self._store_search_results("Stacking (Tuned RF+GB)", search, "stacking_grid")
        self._evaluate_best_model(
            "Stacking (Tuned RF+GB)",
            search.best_estimator_,
            search.best_params_,
            search.best_score_,
        )
        self.best_estimators["Stacking (Tuned RF+GB)"] = clone(
            search.best_estimator_.named_steps["model"]
        )

        return self

    def export_results(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        pd.concat(self.search_results, ignore_index=True).to_csv(
            self.output_dir / "t2_optimization_results.csv", index=False
        )

        self._top_configurations().to_csv(
            self.output_dir / "t2_top_configurations.csv", index=False
        )

        pd.DataFrame(self.best_model_rows).sort_values(
            by="Test F2-Score", ascending=False
        ).to_csv(self.output_dir / "t2_best_models.csv", index=False)

        pd.DataFrame(self.classification_report_rows).to_csv(
            self.output_dir / "t2_classification_report.csv", index=False
        )

        return self

    def plot_metric_comparison(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df_results = pd.DataFrame(self.best_model_rows)
        df_long = df_results.melt(
            id_vars="Model",
            value_vars=[
                "Test Accuracy",
                "Test Recall",
                "Test F1-Score",
                "Test F2-Score",
            ],
            var_name="Metric",
            value_name="Score",
        )

        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_long, x="Model", y="Score", hue="Metric")
        plt.ylim(0, 1)
        plt.title("Optimized Model Performance on Test Set")
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.xticks(rotation=20, ha="right")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(self.output_dir / "t2_metric_comparison.png", dpi=150)
        plt.close()
        return self

    def plot_optimization_results(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df_results = pd.concat(self.search_results, ignore_index=True)

        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=df_results,
            x="Model",
            y="CV F2-Score Mean",
            hue="Search Stage",
        )
        sns.stripplot(
            data=df_results,
            x="Model",
            y="CV F2-Score Mean",
            hue="Search Stage",
            dodge=True,
            alpha=0.35,
            size=3,
        )
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_labels = list(dict.fromkeys(labels))
        unique_handles = [handles[labels.index(label)] for label in unique_labels]
        plt.legend(unique_handles, unique_labels, title="Search Stage")
        plt.title("CV F2 Distribution Across Tried Configurations")
        plt.xlabel("Model")
        plt.ylabel("CV F2-Score Mean")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(self.output_dir / "t2_cv_f2_distribution_by_model.png", dpi=150)
        plt.close()

        top_configs = self._top_configurations()
        plt.figure(figsize=(14, 7))
        sns.barplot(
            data=top_configs,
            x="Top Config Rank",
            y="CV F2-Score Mean",
            hue="Model",
        )
        plt.ylim(0, 1)
        plt.title(f"Top {TOP_N_CONFIGS} Configurations by CV F2")
        plt.xlabel("Rank within model")
        plt.ylabel("CV F2-Score Mean")
        plt.tight_layout()
        plt.savefig(self.output_dir / "t2_top_configurations_f2.png", dpi=150)
        plt.close()
        return self

    def plot_confusion_matrices(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        num_models = len(self.confusion_matrices)
        cols = num_models

        fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 4))
        if num_models == 1:
            axes = [axes]

        for ax, (model_name, cm) in zip(axes, self.confusion_matrices.items()):
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                cbar=False,
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
            )
            ax.set_title(model_name)
            ax.set_xlabel("Predito")
            ax.set_ylabel("Real")

        plt.tight_layout()
        plt.savefig(self.output_dir / "t2_confusion_matrices.png", dpi=150)
        plt.close()
        return self

    def _build_scoring(self):
        return {
            "accuracy": "accuracy",
            "recall": make_scorer(recall_score, average="macro"),
            "f1": make_scorer(f1_score, average="macro"),
            "f2": make_scorer(macro_f2_score),
        }

    def _pipeline(self, classifier):
        return Pipeline([("scaler", StandardScaler()), ("model", classifier)])

    def _run_randomized_search(self, model_name, pipeline, param_grid, n_iter):
        print(f"Optimizing {model_name} with {n_iter} sampled configurations...")
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=self.scoring,
            refit="f2",
            cv=StratifiedKFold(
                n_splits=5, shuffle=True, random_state=self.random_state
            ),
            random_state=self.random_state,
            n_jobs=N_JOBS,
            return_train_score=False,
        )
        search.fit(self.X_train, self.y_train)
        return search

    def _run_grid_search(self, model_name, pipeline, param_grid, search_stage):
        total_configs = 1
        for values in param_grid.values():
            total_configs *= len(values)

        print(
            f"Optimizing {model_name} ({search_stage}) "
            f"with {total_configs} grid configurations..."
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=self.scoring,
            refit="f2",
            cv=StratifiedKFold(
                n_splits=5, shuffle=True, random_state=self.random_state
            ),
            n_jobs=N_JOBS,
            return_train_score=False,
        )
        search.fit(self.X_train, self.y_train)
        return search

    def _store_search_results(self, model_name, search, search_stage):
        cv_results = pd.DataFrame(search.cv_results_)
        columns = ["params"] + [
            column
            for column in cv_results.columns
            if column.startswith("mean_test_") or column.startswith("std_test_")
        ]
        result = cv_results[columns].copy()
        result.insert(0, "Model", model_name)
        result.insert(1, "Search Stage", search_stage)
        result = result.rename(
            columns={
                "mean_test_accuracy": "CV Accuracy Mean",
                "std_test_accuracy": "CV Accuracy Std",
                "mean_test_recall": "CV Recall Mean",
                "std_test_recall": "CV Recall Std",
                "mean_test_f1": "CV F1-Score Mean",
                "std_test_f1": "CV F1-Score Std",
                "mean_test_f2": "CV F2-Score Mean",
                "std_test_f2": "CV F2-Score Std",
            }
        )
        result["Best Rank F2"] = cv_results["rank_test_f2"]
        self.search_results.append(result)

    def _evaluate_best_model(self, model_name, best_pipeline, best_params, cv_best_f2):
        print(f"Evaluating best {model_name} on held-out test set...")
        y_pred = best_pipeline.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        test_recall = recall_score(self.y_test, y_pred, average="macro")
        test_f1 = f1_score(self.y_test, y_pred, average="macro")
        test_f2 = fbeta_score(self.y_test, y_pred, beta=2, average="macro")

        self.best_model_rows.append(
            {
                "Model": model_name,
                "Best Parameters": str(best_params),
                "CV Best F2-Score": cv_best_f2,
                "Test Accuracy": test_accuracy,
                "Test Recall": test_recall,
                "Test F1-Score": test_f1,
                "Test F2-Score": test_f2,
                "F2 Generalization Gap": cv_best_f2 - test_f2,
            }
        )
        self.confusion_matrices[model_name] = confusion_matrix(
            self.y_test, y_pred, labels=CLASS_LABELS
        )
        self._record_classification_report(model_name, y_pred)

    def _record_classification_report(self, model_name, y_pred):
        report = classification_report(
            self.y_test,
            y_pred,
            labels=CLASS_LABELS,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        for label, metrics in report.items():
            if not isinstance(metrics, dict):
                continue

            row = {"Model": model_name, "Class": label}
            row.update(metrics)
            self.classification_report_rows.append(row)

    def _top_configurations(self):
        columns = [
            "Model",
            "Search Stage",
            "Top Config Rank",
            "params",
            "CV F2-Score Mean",
            "CV F2-Score Std",
            "CV Recall Mean",
            "CV Recall Std",
            "CV F1-Score Mean",
            "CV F1-Score Std",
            "CV Accuracy Mean",
            "CV Accuracy Std",
        ]
        df_results = pd.concat(self.search_results, ignore_index=True)
        top_configs = (
            df_results.sort_values(
                ["Model", "CV F2-Score Mean"], ascending=[True, False]
            )
            .groupby("Model", as_index=False)
            .head(TOP_N_CONFIGS)
            .copy()
        )
        top_configs["Top Config Rank"] = (
            top_configs.groupby("Model")["CV F2-Score Mean"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        return top_configs.sort_values(["Model", "Top Config Rank"])[columns]

    def _ensure_base_models_are_optimized(self):
        missing = [
            name
            for name in ("Random Forest", "Gradient Boosting")
            if name not in self.best_estimators
        ]
        if missing:
            raise RuntimeError(
                "Base models must be optimized before stacking: " + ", ".join(missing)
            )

    def _random_forest_grid(self):
        return {
            "model__n_estimators": [100, 200, 300, 500],
            "model__criterion": ["gini", "entropy", "log_loss"],
            "model__max_depth": [None, 8, 12, 16, 24, 32],
            "model__min_samples_split": [2, 4, 8, 12],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", "log2", None],
            "model__class_weight": ["balanced", "balanced_subsample"],
        }

    def _gradient_boosting_grid(self):
        return {
            "model__loss": ["log_loss"],
            "model__n_estimators": [100, 200, 300, 500],
            "model__learning_rate": [0.005, 0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4, 5],
            "model__min_samples_split": [2, 4, 8, 12],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__max_features": [None, "sqrt", "log2"],
        }

    def _focused_grid_for_model(self, model_name, best_params):
        if model_name == "Random Forest":
            return self._focused_random_forest_grid(best_params)
        if model_name == "Gradient Boosting":
            return self._focused_gradient_boosting_grid(best_params)
        raise ValueError(f"No focused grid configured for {model_name}.")

    def _focused_random_forest_grid(self, best_params):
        return {
            "model__n_estimators": self._nearby_values(
                best_params["model__n_estimators"], [100, 200, 300, 500]
            ),
            "model__max_depth": self._nearby_values(
                best_params["model__max_depth"], [None, 8, 12, 16, 24, 32]
            ),
            "model__min_samples_split": self._nearby_values(
                best_params["model__min_samples_split"], [2, 4, 8, 12]
            ),
            "model__min_samples_leaf": self._nearby_values(
                best_params["model__min_samples_leaf"], [1, 2, 4, 8]
            ),
            "model__criterion": [best_params["model__criterion"]],
            "model__max_features": [best_params["model__max_features"]],
            "model__class_weight": [best_params["model__class_weight"]],
        }

    def _focused_gradient_boosting_grid(self, best_params):
        return {
            "model__n_estimators": self._nearby_values(
                best_params["model__n_estimators"], [100, 200, 300, 500]
            ),
            "model__learning_rate": self._nearby_values(
                best_params["model__learning_rate"], [0.005, 0.01, 0.03, 0.05, 0.1]
            ),
            "model__max_depth": self._nearby_values(
                best_params["model__max_depth"], [2, 3, 4, 5]
            ),
            "model__min_samples_split": self._nearby_values(
                best_params["model__min_samples_split"], [2, 4, 8, 12]
            ),
            "model__min_samples_leaf": self._nearby_values(
                best_params["model__min_samples_leaf"], [1, 2, 4, 8]
            ),
            "model__loss": [best_params["model__loss"]],
            "model__subsample": [best_params["model__subsample"]],
            "model__max_features": [best_params["model__max_features"]],
        }

    def _nearby_values(self, best_value, candidates):
        if best_value not in candidates:
            return [best_value]

        best_index = candidates.index(best_value)
        start = max(0, best_index - 1)
        end = min(len(candidates), best_index + 2)
        return candidates[start:end]


if __name__ == "__main__":
    (
        FetalHealthOptimizer("fetal_health.csv")
        .load_and_preprocess()
        .optimize_base_models()
        .optimize_stacking_model()
        .export_results()
        .plot_optimization_results()
        .plot_metric_comparison()
        .plot_confusion_matrices()
    )
