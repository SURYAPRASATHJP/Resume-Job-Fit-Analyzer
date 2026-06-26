
from __future__ import annotations

import json
import os

try:
    import ollama
except ImportError:  # pragma: no cover - dependency missing
    ollama = None

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# JSON shape we instruct the model to return. Keep this in sync with the prompt.
_JSON_SCHEMA_HINT = """{
  "fit_verdict": {
    "score": <integer 0-100, how well the resume fits the job>,
    "label": "<one of: Strong fit, Moderate fit, Weak fit>",
    "summary": "<two-sentence overall assessment>"
  },
  "matching_strengths": ["<resume point that aligns with the job>", "..."],
  "skill_gaps": ["<specific skill/requirement in the job missing from the resume>", "..."],
  "bullet_rewrites": [
    {"original": "<a weak bullet from the resume>", "improved": "<stronger, quantified rewrite tailored to the job>"}
  ]
}"""

_SYSTEM_PROMPT = (
    "You are an expert technical recruiter and resume coach. You compare a "
    "candidate's resume against a target job description and return precise, "
    "actionable feedback. Be concrete and specific to the texts provided; never "
    "invent experience the candidate does not have. Respond with JSON only."
)


def _build_user_prompt(resume: str, job_description: str) -> str:
    return (
        "Compare the RESUME against the JOB DESCRIPTION and respond with a single "
        "JSON object matching exactly this schema (no markdown, no commentary):\n\n"
        f"{_JSON_SCHEMA_HINT}\n\n"
        "Guidance:\n"
        "- List 3-6 skill_gaps, each a real requirement from the job missing or weak in the resume.\n"
        "- List 2-5 matching_strengths.\n"
        "- Provide 2-4 bullet_rewrites that quantify impact and mirror the job's language.\n\n"
        f"=== RESUME ===\n{resume.strip()}\n\n"
        f"=== JOB DESCRIPTION ===\n{job_description.strip()}\n"
    )


def _empty_result(error: str) -> dict:
    return {
        "fit_verdict": {"score": None, "label": "Unavailable", "summary": error},
        "matching_strengths": [],
        "skill_gaps": [],
        "bullet_rewrites": [],
        "error": error,
    }


def critique(
    resume: str,
    job_description: str,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    temperature: float = 0.2,
) -> dict:
    """Return a structured critique dict. On failure, returns a result whose
    `error` key explains what went wrong (so the UI can degrade gracefully)."""
    if ollama is None:
        return _empty_result("The `ollama` package is not installed (pip install ollama).")
    if not resume.strip() or not job_description.strip():
        return _empty_result("Both a resume and a job description are required.")

    client = ollama.Client(host=host)
    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(resume, job_description)},
            ],
            format="json",  # constrain Ollama to emit valid JSON
            options={"temperature": temperature},
        )
    except Exception as exc:  # connection refused, model not pulled, etc.
        return _empty_result(
            f"Could not reach Ollama at {host} with model '{model}': {exc}. "
            "Is Ollama running and the model pulled (`ollama pull {model}`)?".format(model=model)
        )

    content = response["message"]["content"]
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return _empty_result(f"Model did not return valid JSON. Raw output:\n{content[:800]}")

    # Normalize to guarantee the keys the UI expects.
    return {
        "fit_verdict": data.get("fit_verdict", {}) or {},
        "matching_strengths": data.get("matching_strengths", []) or [],
        "skill_gaps": data.get("skill_gaps", []) or [],
        "bullet_rewrites": data.get("bullet_rewrites", []) or [],
    }


if __name__ == "__main__":
    demo = critique(
        resume="Software engineer with 3 years building Python web apps and REST APIs.",
        job_description="Seeking an ML engineer skilled in scikit-learn, model evaluation, and AWS.",
    )
    print(json.dumps(demo, indent=2))
