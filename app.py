from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from classifier import ResumeClassifier  # noqa: E402
from llm_critique import DEFAULT_MODEL, critique  # noqa: E402

MODEL_PATH = os.environ.get("MODEL_PATH", str(ROOT / "models" / "pipeline.joblib"))

st.set_page_config(page_title="Resume Job-Fit Analyzer", page_icon="🧭", layout="wide")


@st.cache_resource(show_spinner=False)
def load_classifier(path: str) -> ResumeClassifier:
    return ResumeClassifier(path)


st.title("🧭 Resume Job-Fit Analyzer")
st.caption("ML category prediction + local-LLM critique for how well a resume fits a job.")

with st.sidebar:
    st.header("Settings")
    ollama_model = st.text_input("Ollama model", value=DEFAULT_MODEL)
    run_llm = st.checkbox("Run LLM critique", value=True)
    st.divider()
    model_exists = Path(MODEL_PATH).exists()
    if model_exists:
        st.success("Classifier model loaded.")
    else:
        st.warning("No trained model found.\nRun `python src/train.py` after adding the dataset.")
    st.caption(f"Model path: `{MODEL_PATH}`")

col_in1, col_in2 = st.columns(2)
with col_in1:
    resume = st.text_area("Resume", height=320, placeholder="Paste the candidate's resume here...")
with col_in2:
    jd = st.text_area("Job description", height=320, placeholder="Paste the target job description here...")

analyze = st.button("Analyze fit", type="primary", use_container_width=True)


def render_classifier(resume_text: str) -> None:
    st.subheader("📊 ML classifier")
    if not Path(MODEL_PATH).exists():
        st.info("Train the classifier first to see a predicted category.")
        return
    try:
        clf = load_classifier(MODEL_PATH)
        result = clf.predict(resume_text)
    except Exception as exc:
        st.error(f"Classifier error: {exc}")
        return
    st.metric("Predicted category", result["category"], f"{result['confidence'] * 100:.1f}% confidence")
    st.caption("Top categories")
    for label, prob in result["top_k"]:
        st.write(f"**{label}** — {prob * 100:.1f}%")
        st.progress(min(max(prob, 0.0), 1.0))


def render_critique(resume_text: str, jd_text: str, model: str) -> None:
    st.subheader("🤖 LLM critique")
    with st.spinner("Asking the local model..."):
        data = critique(resume_text, jd_text, model=model)
    if data.get("error"):
        st.error(data["error"])
        return

    verdict = data.get("fit_verdict", {})
    score = verdict.get("score")
    cols = st.columns([1, 3])
    with cols[0]:
        st.metric("Fit score", f"{score}/100" if score is not None else "—", verdict.get("label", ""))
    with cols[1]:
        if verdict.get("summary"):
            st.write(verdict["summary"])

    strengths = data.get("matching_strengths", [])
    gaps = data.get("skill_gaps", [])
    if strengths:
        st.markdown("**✅ Matching strengths**")
        for s in strengths:
            st.markdown(f"- {s}")
    if gaps:
        st.markdown("**⚠️ Skill gaps**")
        for g in gaps:
            st.markdown(f"- {g}")

    rewrites = data.get("bullet_rewrites", [])
    if rewrites:
        st.markdown("**✍️ Suggested bullet rewrites**")
        for r in rewrites:
            if isinstance(r, dict):
                st.markdown(f"- ~~{r.get('original', '')}~~\n  → **{r.get('improved', '')}**")
            else:
                st.markdown(f"- {r}")

    with st.expander("Raw JSON"):
        st.json(data)


if analyze:
    if not resume.strip():
        st.warning("Please paste a resume.")
    else:
        left, right = st.columns(2)
        with left:
            render_classifier(resume)
        with right:
            if run_llm:
                if not jd.strip():
                    st.warning("Paste a job description to get an LLM critique.")
                else:
                    render_critique(resume, jd, ollama_model)
            else:
                st.info("LLM critique disabled in the sidebar.")
