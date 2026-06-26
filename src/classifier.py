
from __future__ import annotations

import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess  

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "pipeline.joblib"


class ResumeClassifier:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run `python src/train.py` first."
            )
        self.pipe = joblib.load(model_path)
        self.classes_ = list(self.pipe.classes_)

    def predict(self, text: str, top_k: int = 3) -> dict:
        proba = self.pipe.predict_proba([text])[0]
        order = proba.argsort()[::-1]
        top = [(self.classes_[i], float(proba[i])) for i in order[:top_k]]
        best = order[0]
        return {
            "category": self.classes_[best],
            "confidence": float(proba[best]),
            "top_k": top,
        }


if __name__ == "__main__":
    import sys as _sys

    clf = ResumeClassifier()
    sample = " ".join(_sys.argv[1:]) or "python machine learning tensorflow data pipelines sql"
    print(clf.predict(sample))
