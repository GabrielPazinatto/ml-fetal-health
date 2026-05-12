from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class FetalHealthEDA:
    def __init__(
        self,
        filepath: str,
        target_column: str = "fetal_health",
        output_dir: str = "outputs/eda",
    ):
        self.filepath = Path(filepath)
        self.target_column = target_column
        self.output_dir = Path(output_dir)
        self.reports = {}
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        return self

    def analyze(self):
        self._ensure_loaded()

        numeric_columns = self.df.select_dtypes(include="number").columns
        missing_values = (
            self.df.isna()
            .sum()
            .rename("missing_count")
            .reset_index()
            .rename(columns={"index": "column"})
        )
        missing_values["missing_percent"] = (
            missing_values["missing_count"] / len(self.df) * 100
        )
        numeric_summary = (
            self.df[numeric_columns]
            .describe()
            .transpose()
            .reset_index()
            .rename(columns={"index": "column"})
        )

        self.reports = {
            "dataset_shape": pd.DataFrame(
                [
                    {
                        "rows": self.df.shape[0],
                        "columns": self.df.shape[1],
                        "feature_columns": self.df.shape[1] - 1,
                    }
                ]
            ),
            "missing_values": missing_values,
            "duplicate_rows": pd.DataFrame(
                [{"duplicate_rows": int(self.df.duplicated().sum())}]
            ),
            "target_class_distribution": self._target_distribution(),
            "numeric_summary": numeric_summary,
        }

        return self

    def export_tables(self):
        self._ensure_analyzed()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for name, report in self.reports.items():
            report.to_csv(self.output_dir / f"{name}.csv", index=False)

        self._write_summary_text()
        return self

    def export_plots(self):
        self._ensure_analyzed()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 5))
        ax = sns.countplot(data=self.df, x=self.target_column, color="#4C78A8")
        ax.set_title("Target Class Distribution")
        ax.set_xlabel(self.target_column)
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(self.output_dir / "target_class_distribution.png", dpi=150)
        plt.close()

        return self

    def _target_distribution(self):
        counts = self.df[self.target_column].value_counts().sort_index()
        distribution = counts.rename("count").reset_index()
        distribution = distribution.rename(columns={"index": self.target_column})
        distribution["percent"] = distribution["count"] / len(self.df) * 100
        return distribution

    def _write_summary_text(self):
        shape = self.reports["dataset_shape"].iloc[0]
        duplicates = self.reports["duplicate_rows"].iloc[0]["duplicate_rows"]
        missing_total = int(self.reports["missing_values"]["missing_count"].sum())

        lines = [
            "Fetal Health EDA Summary",
            "",
            f"Dataset: {self.filepath}",
            f"Rows: {shape['rows']}",
            f"Columns: {shape['columns']}",
            f"Feature columns: {shape['feature_columns']}",
            f"Missing values: {missing_total}",
            f"Duplicate rows: {duplicates}",
            "",
            "Target class distribution:",
        ]

        for _, row in self.reports["target_class_distribution"].iterrows():
            lines.append(
                f"- {int(row[self.target_column])}: {int(row['count'])} "
                f"({row['percent']:.2f}%)"
            )

        (self.output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    def _ensure_loaded(self):
        if self.df is None:
            raise RuntimeError("Data has not been loaded. Call load_data() first.")

    def _ensure_analyzed(self):
        self._ensure_loaded()
        if not self.reports:
            raise RuntimeError("EDA has not been analyzed. Call analyze() first.")


if __name__ == "__main__":
    (
        FetalHealthEDA("fetal_health.csv")
        .load_data()
        .analyze()
        .export_tables()
        .export_plots()
    )
