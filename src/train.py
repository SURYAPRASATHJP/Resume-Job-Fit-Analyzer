from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Make `preprocess` importable whether run as a script or a module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import clean_series  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "UpdatedResumeDataSet.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Common column-name variants seen in resume datasets.
_TEXT_CANDIDATES = ["Resume", "resume", "Resume_str", "text", "Text"]
_LABEL_CANDIDATES = ["Category", "category", "label", "Label"]


def _pick_column(df: pd.DataFrame, preferred: str | None, candidates: list[str], kind: str) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(
        f"Could not find a {kind} column. Looked for {candidates} in {list(df.columns)}. "
        f"Pass --{kind}-col explicitly."
    )


def load_data(csv_path: Path, text_col: str | None, label_col: str | None) -> tuple[pd.Series, pd.Series]:
    if not csv_path.exists():
        raise SystemExit(
            f"Dataset not found at {csv_path}.\n"
            "Download the Kaggle 'Resume Dataset' and place UpdatedResumeDataSet.csv in data/."
        )
    df = pd.read_csv(csv_path)
    text_col = _pick_column(df, text_col, _TEXT_CANDIDATES, "text")
    label_col = _pick_column(df, label_col, _LABEL_CANDIDATES, "label")

    df = df[[label_col, text_col]].dropna()
    df = df[df[text_col].astype(str).str.strip().astype(bool)]
    # Drop classes with a single example so a stratified split is possible.
    counts = df[label_col].value_counts()
    too_small = counts[counts < 2].index.tolist()
    if too_small:
        print(f"[warn] dropping {len(too_small)} class(es) with <2 examples: {too_small}")
        df = df[~df[label_col].isin(too_small)]
    print(f"Loaded {len(df)} resumes across {df[label_col].nunique()} categories "
          f"(text='{text_col}', label='{label_col}').")
    return df[text_col].reset_index(drop=True), df[label_col].reset_index(drop=True)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            # `validate=False` keeps text passing through untouched as a list.
            ("clean", FunctionTransformer(clean_series, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    sublinear_tf=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    stop_words="english",
                    max_features=20000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",  # classes are imbalanced
                    max_iter=2000,
                    C=10.0,
                ),
            ),
        ]
    )


def evaluate(pipe: Pipeline, X_test, y_test) -> str:
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, zero_division=0)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    header = (
        f"Accuracy : {acc:.3f}\n"
        f"Macro-F1 : {macro_f1:.3f}  (per-class average; trust this over raw accuracy)\n"
        f"{'-' * 60}\n"
    )
    full = header + report
    (REPORTS_DIR / "classification_report.txt").write_text(full)

    # Confusion matrix figure — read this, not the headline accuracy.
    labels = sorted(pd.unique(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.5), max(6, n * 0.5)))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, xticks_rotation=90, colorbar=False, values_format="d"
    )
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)
    return full


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the resume-category classifier.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to the dataset CSV.")
    parser.add_argument("--text-col", default=None, help="Name of the resume-text column.")
    parser.add_argument("--label-col", default=None, help="Name of the category column.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y = load_data(args.csv, args.text_col, args.label_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    pipe = build_pipeline()
    print("Fitting pipeline (clean -> TF-IDF -> LogisticRegression)...")
    pipe.fit(X_train, y_train)

    report = evaluate(pipe, X_test, y_test)
    print("\n" + report)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / "pipeline.joblib"
    joblib.dump(pipe, out)
    print(f"Saved fitted pipeline -> {out}")
    print(f"Reports -> {REPORTS_DIR / 'classification_report.txt'} and confusion_matrix.png")


if __name__ == "__main__":
    main()
