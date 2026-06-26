
from __future__ import annotations

import re

# Keep a few symbol chars that carry signal in tech resumes: c++, c#, .net, node.js
_URL_RE = re.compile(r"http\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+")
_KEEP_RE = re.compile(r"[^a-z0-9\s+#.]")
_MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: object) -> str:
    """Lowercase, strip urls/emails/punctuation, and collapse whitespace."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _KEEP_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def clean_series(texts) -> list[str]:
    """Vectorized helper for a list/Series of documents -> list[str].

    Used as the first stage of the model pipeline (FunctionTransformer), so it
    must accept any array-like of raw documents and return a list of strings.
    """
    return [clean_text(t) for t in texts]
