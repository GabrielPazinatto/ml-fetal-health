import argparse
import ast
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "10")
N_JOBS = -1
RANDOM_STATE = 42
CLASS_LABELS = [1, 2, 3]
CLASS_NAMES = ["Normal", "Suspeito", "Patologico"]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    fbeta_score,
    make_scorer,
    roc_auc_score,
)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


def macro_f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2, average="macro")


class FetalHealthInterpreter:
    def __init__(
        self,
        filepath: str,
        best_models_path: str = "t2/t2_best_models.csv",
        output_dir: str = "t2/interpret",
        model_name: str | None = None,
        focus_class: int = 3,
        top_features: int = 8,
        random_state: int = RANDOM_STATE,
    ):
        self.filepath = filepath
        self.best_models_path = Path(best_models_path)
        self.output_dir = Path(output_dir)
        self.requested_model_name = model_name
        self.focus_class = focus_class
        self.top_features = top_features
        self.random_state = random_state
        self.best_model_rows = {}
        self.selected_model_name = None
        self.pipeline = None
        self.permutation_importance_rows = None
        self.report_output_dir = Path(best_models_path).parent

    def load_and_preprocess(self):
        df = pd.read_csv(self.filepath)
        duplicate_rows_removed = int(df.duplicated().sum())
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Removed {duplicate_rows_removed} duplicate rows.")

        self.X = df.drop(columns=["fetal_health"])
        self.y = df["fetal_health"].astype(int)
        self.feature_names = self.X.columns.tolist()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=self.y,
        )
        return self

    def load_best_model_metadata(self):
        df_best = pd.read_csv(self.best_models_path)
        self.best_model_rows = {
            row["Model"]: row.to_dict() for _, row in df_best.iterrows()
        }

        if self.requested_model_name is None:
            best_row = df_best.sort_values("Test F2-Score", ascending=False).iloc[0]
            self.selected_model_name = best_row["Model"]
        else:
            self.selected_model_name = self.requested_model_name

        print(f"Selected model for interpretation: {self.selected_model_name}")
        return self

    def fit_selected_model(self):
        model = self._build_model(self.selected_model_name)
        self.pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
        self.pipeline.fit(self.X_train, self.y_train)

        y_pred = self.pipeline.predict(self.X_test)
        print(f"Test accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print(f"Test macro F2: {macro_f2_score(self.y_test, y_pred):.4f}")
        return self

    def export_permutation_importance(self):
        self._ensure_output_dir()
        print("Computing permutation importance with macro F2...")
        result = permutation_importance(
            self.pipeline,
            self.X_test,
            self.y_test,
            scoring=make_scorer(macro_f2_score),
            n_repeats=30,
            random_state=self.random_state,
            n_jobs=N_JOBS,
        )
        df_importance = pd.DataFrame(
            {
                "Feature": self.feature_names,
                "Importance Mean": result.importances_mean,
                "Importance Std": result.importances_std,
            }
        ).sort_values("Importance Mean", ascending=False)
        df_importance.to_csv(
            self.output_dir / "interpret_permutation_importance.csv", index=False
        )
        self.permutation_importance_rows = df_importance

        self._plot_feature_importance(
            df_importance.head(self.top_features),
            title="Permutation Importance on Test Set (Macro F2)",
            output_path=self.output_dir / "interpret_permutation_importance.png",
        )
        return self

    def export_native_feature_importance(self):
        self._ensure_output_dir()
        model = self.pipeline.named_steps["model"]
        importances = self._native_feature_importances(model)

        if importances is None:
            print(
                f"{self.selected_model_name} does not expose native feature "
                "importances. Writing an empty CSV."
            )
            df_importance = pd.DataFrame(
                columns=["Feature", "Importance", "Source Model"]
            )
            df_importance.to_csv(
                self.output_dir / "interpret_native_feature_importance.csv",
                index=False,
            )
            return self

        df_importance = pd.DataFrame(
            {
                "Feature": self.feature_names,
                "Importance": importances,
                "Source Model": self.selected_model_name,
            }
        ).sort_values("Importance", ascending=False)
        df_importance.to_csv(
            self.output_dir / "interpret_native_feature_importance.csv", index=False
        )

        self._plot_feature_importance(
            df_importance.head(self.top_features),
            value_column="Importance",
            title="Native Tree Feature Importance",
            output_path=self.output_dir / "interpret_native_feature_importance.png",
        )
        return self

    def export_partial_dependence(self):
        self._ensure_output_dir()
        self._ensure_permutation_importance()

        top_feature_names = self.permutation_importance_rows.head(self.top_features)[
            "Feature"
        ].tolist()
        top_feature_indices = [
            self.feature_names.index(feature) for feature in top_feature_names
        ]

        fig, ax = plt.subplots(figsize=(12, max(6, 2.4 * len(top_feature_indices))))
        PartialDependenceDisplay.from_estimator(
            self.pipeline,
            self.X_test,
            features=top_feature_indices,
            target=self.focus_class,
            response_method="predict_proba",
            feature_names=self.feature_names,
            n_cols=2,
            grid_resolution=50,
            ax=ax,
        )
        plt.suptitle(
            f"Partial Dependence for Class {self.focus_class} "
            f"({self._class_name(self.focus_class)})",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(
            self.output_dir
            / f"interpret_partial_dependence_class_{self.focus_class}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        return self

    def export_shap_summary(self):
        import shap
        self._ensure_output_dir()

        print(
            f"Computing SHAP summary for class {self.focus_class} "
            f"({self._class_name(self.focus_class)})..."
        )
        scaler = self.pipeline.named_steps["scaler"]
        model = self.pipeline.named_steps["model"]
        X_test_scaled = pd.DataFrame(
            scaler.transform(self.X_test),
            columns=self.feature_names,
            index=self.X_test.index,
        )
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_scaled)
        class_index = self._class_index(model, self.focus_class)
        shap_values_for_class = self._shap_values_for_class(
            shap_values,
            class_index,
            expected_features=len(self.feature_names),
        )

        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_values_for_class,
            X_test_scaled,
            max_display=self.top_features,
            show=False,
        )
        plt.title(
            f"SHAP Summary for Class {self.focus_class} "
            f"({self._class_name(self.focus_class)})"
        )
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"interpret_shap_summary_class_{self.focus_class}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        return self

    def export_roc_pr_curves(self):
        self.report_output_dir.mkdir(parents=True, exist_ok=True)
        y_score = self.pipeline.predict_proba(self.X_test)
        y_true_binary = pd.get_dummies(self.y_test).reindex(
            columns=CLASS_LABELS, fill_value=0
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        for class_label in CLASS_LABELS:
            class_index = self._class_index(
                self.pipeline.named_steps["model"],
                class_label,
            )
            auc = roc_auc_score(y_true_binary[class_label], y_score[:, class_index])
            RocCurveDisplay.from_predictions(
                y_true_binary[class_label],
                y_score[:, class_index],
                name=f"{self._class_name(class_label)} (AUC={auc:.3f})",
                ax=ax,
            )
        ax.set_title(f"One-vs-Rest ROC Curves ({self.selected_model_name})")
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(self.report_output_dir / "t2_roc_curves.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        for class_label in CLASS_LABELS:
            class_index = self._class_index(
                self.pipeline.named_steps["model"],
                class_label,
            )
            ap = average_precision_score(
                y_true_binary[class_label], y_score[:, class_index]
            )
            PrecisionRecallDisplay.from_predictions(
                y_true_binary[class_label],
                y_score[:, class_index],
                name=f"{self._class_name(class_label)} (AP={ap:.3f})",
                ax=ax,
            )
        ax.set_title(
            f"One-vs-Rest Precision-Recall Curves ({self.selected_model_name})"
        )
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(self.report_output_dir / "t2_pr_curves.png", dpi=150)
        plt.close(fig)
        return self

    def export_error_analysis(self):
        self._ensure_output_dir()
        y_pred = self.pipeline.predict(self.X_test)
        is_correct = self.y_test.to_numpy() == y_pred

        df_errors = pd.DataFrame(
            {
                "True Class": self.y_test.to_numpy(),
                "Predicted Class": y_pred,
                "Correct": is_correct,
            }
        )
        error_summary = (
            df_errors.groupby(["True Class", "Predicted Class", "Correct"])
            .size()
            .reset_index(name="Count")
            .sort_values(["True Class", "Predicted Class"])
        )
        error_summary.to_csv(
            self.output_dir / "interpret_error_analysis_by_class.csv", index=False
        )

        df_test = self.X_test.copy()
        df_test["True Class"] = self.y_test.to_numpy()
        df_test["Predicted Class"] = y_pred
        df_test["Prediction Outcome"] = [
            "Correct" if correct else "Incorrect" for correct in is_correct
        ]

        summary_rows = []
        for feature in self.feature_names:
            grouped = df_test.groupby(["True Class", "Prediction Outcome"])[feature]
            feature_summary = grouped.agg(
                ["count", "mean", "std", "median"]
            ).reset_index()
            feature_summary.insert(0, "Feature", feature)
            summary_rows.append(feature_summary)

        pd.concat(summary_rows, ignore_index=True).to_csv(
            self.output_dir / "interpret_error_feature_summary.csv", index=False
        )
        return self

    def export_surrogate_tree(self):
        self._ensure_output_dir()
        teacher_train_pred = self.pipeline.predict(self.X_train)
        teacher_test_pred = self.pipeline.predict(self.X_test)

        surrogate = DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=20,
            random_state=self.random_state,
        )
        surrogate.fit(self.X_train, teacher_train_pred)
        surrogate_test_pred = surrogate.predict(self.X_test)

        fidelity = accuracy_score(teacher_test_pred, surrogate_test_pred)
        pd.DataFrame(
            [
                {
                    "Selected Model": self.selected_model_name,
                    "Surrogate Max Depth": surrogate.max_depth,
                    "Surrogate Min Samples Leaf": surrogate.min_samples_leaf,
                    "Test Fidelity": fidelity,
                }
            ]
        ).to_csv(self.output_dir / "interpret_surrogate_fidelity.csv", index=False)

        rules = export_text(surrogate, feature_names=self.feature_names)
        (self.output_dir / "interpret_surrogate_tree_rules.txt").write_text(
            rules, encoding="utf-8"
        )

        plt.figure(figsize=(18, 10))
        plot_tree(
            surrogate,
            feature_names=self.feature_names,
            class_names=CLASS_NAMES,
            filled=True,
            rounded=True,
            fontsize=8,
        )
        plt.title(
            f"Surrogate Decision Tree for {self.selected_model_name} "
            f"(fidelity={fidelity:.3f})"
        )
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "interpret_surrogate_tree.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        return self

    def _build_model(self, model_name):
        builders = {
            "Random Forest": self._build_random_forest,
            "Gradient Boosting": self._build_gradient_boosting,
            "Stacking (Tuned RF+GB)": self._build_stacking,
        }
        return builders[model_name]()

    def _build_random_forest(self):
        params = self._model_params("Random Forest")
        return RandomForestClassifier(
            **params, random_state=self.random_state, n_jobs=N_JOBS
        )

    def _build_gradient_boosting(self):
        params = self._model_params("Gradient Boosting")
        return GradientBoostingClassifier(**params, random_state=self.random_state)

    def _build_stacking(self):
        stacking_params = self._model_params("Stacking (Tuned RF+GB)")
        final_estimator = LogisticRegression(
            C=stacking_params.pop("final_estimator__C"),
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_state,
        )
        return StackingClassifier(
            estimators=[
                ("rf", self._build_random_forest()),
                ("gb", self._build_gradient_boosting()),
            ],
            final_estimator=final_estimator,
            n_jobs=N_JOBS,
            **stacking_params,
        )

    def _model_params(self, model_name):
        raw_params = self.best_model_rows[model_name]["Best Parameters"]
        params = ast.literal_eval(raw_params)
        return {key.removeprefix("model__"): value for key, value in params.items()}

    def _native_feature_importances(self, model):
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_

        if isinstance(model, StackingClassifier):
            rf = model.named_estimators_.get("rf")
            if rf is not None and hasattr(rf, "feature_importances_"):
                return rf.feature_importances_

        return None

    def _class_index(self, model, class_label):
        classes = list(model.classes_)
        if class_label not in classes:
            raise ValueError(f"Class {class_label} is not available in the model.")
        return classes.index(class_label)

    def _shap_values_for_class(self, shap_values, class_index, expected_features):
        if isinstance(shap_values, list):
            return shap_values[class_index]

        values = getattr(shap_values, "values", shap_values)
        values = np.asarray(values)

        if values.ndim == 2:
            return values
        if values.ndim != 3:
            raise ValueError(f"Unsupported SHAP values shape: {values.shape}")
        if values.shape[2] != expected_features:
            return values[:, :, class_index]
        return values[class_index, :, :]

    def _plot_feature_importance(
        self,
        df_importance,
        title,
        output_path,
        value_column="Importance Mean",
    ):
        plt.figure(figsize=(10, 6))
        plot_data = df_importance.sort_values(value_column, ascending=True)
        sns.barplot(data=plot_data, x=value_column, y="Feature", color="#3b6ea8")
        plt.title(title)
        plt.xlabel(value_column)
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _ensure_output_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_permutation_importance(self):
        if self.permutation_importance_rows is None:
            raise RuntimeError(
                "Permutation importance must be computed before partial dependence."
            )

    def _class_name(self, class_label):
        if class_label in CLASS_LABELS:
            return CLASS_NAMES[CLASS_LABELS.index(class_label)]
        return str(class_label)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interpret optimized fetal-health classification models."
    )
    parser.add_argument("--data", default="fetal_health.csv")
    parser.add_argument("--best-models", default="t2/t2_best_models.csv")
    parser.add_argument("--output-dir", default="t2/interpret")
    parser.add_argument(
        "--model",
        choices=["Random Forest", "Gradient Boosting", "Stacking (Tuned RF+GB)"],
        default=None,
    )
    parser.add_argument("--focus-class", type=int, default=3, choices=CLASS_LABELS)
    parser.add_argument("--top-features", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    (
        FetalHealthInterpreter(
            filepath=args.data,
            best_models_path=args.best_models,
            output_dir=args.output_dir,
            model_name=args.model,
            focus_class=args.focus_class,
            top_features=args.top_features,
        )
        .load_and_preprocess()
        .load_best_model_metadata()
        .fit_selected_model()
        .export_permutation_importance()
        .export_native_feature_importance()
        .export_partial_dependence()
        .export_shap_summary()
        .export_roc_pr_curves()
        .export_error_analysis()
        .export_surrogate_tree()
    )
