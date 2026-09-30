"""Data loading and text preprocessing (stdlib only, no external libraries)."""

import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

_TOKEN_RE = re.compile(r"[a-z0-9']+")

STOPWORDS = frozenset(
    """a an the and or but if is are was were be been being of to in on for with at by
    from as it its this that these those i me my we our us you your yours they them
    their he she do does did doing have has had having not no nor so than then there
    here what when where which who whom why how all any some just please can could
    would should will shall may might get got also about into out up down more most""".split()
)


def normalize(text):
    """Lowercase, strip accents, mask long numbers (ids, amounts) to <num>."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("'", "'")
    text = re.sub(r"\b\d{4,}\b", "<num>", text)
    return text


def _stem(token):
    """Small suffix stripper, applied until stable so plurals/verbs collide.

    'charges' -> 'charg', 'charged' -> 'charg', 'invoices' -> 'invoic'.
    """
    for _ in range(3):
        for suffix in ("ies", "ing", "ed", "s", "e"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                token = token[:-3] + "y" if suffix == "ies" else token[: -len(suffix)]
                break
        else:
            break
    return token


def tokenize(text, keep_stopwords=False):
    tokens = [t.strip("'") for t in _TOKEN_RE.findall(normalize(text)) if len(t) > 1]
    if not keep_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [_stem(t) for t in tokens]


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_split(name="train"):
    return load_json(DATA_DIR / f"{name}.json")


def load_examples():
    return load_json(DATA_DIR / "example_inputs.json")
