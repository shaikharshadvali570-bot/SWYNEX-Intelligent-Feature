"""Engine-agnostic pipeline: local model, public LLM API, or an ensemble.

Every engine returns the same triage object so the CLI, the examples script and
the tests do not care which one produced it:

    {category, urgency, team, sla_hours, category_confidence, requires_review, ...}

Business rules (routing + SLA) live here, not in the model: the model decides
WHAT the ticket is, the pipeline decides WHERE it goes.
"""

from .api_client import APIError, TriageAPIClient
from .labels import AUTOREPLY_CUES, ROUTING, SLA_HOURS
from .model_local import train_local_model


def looks_like_autoreply(text):
    flat = " ".join(text.lower().split())
    return next((cue for cue in AUTOREPLY_CUES if cue in flat), None)


def apply_business_rules(result):
    """Deterministic routing + SLA, so those fields are never hallucinated."""
    result["team"] = ROUTING.get(result["category"], "support-l1")
    result["sla_hours"] = SLA_HOURS.get(result["urgency"], 24)
    return result


class LocalEngine:
    """Baseline engine: trained on the bundled JSON, runs fully offline."""

    name = "local"

    def __init__(self, records=None, **kwargs):
        self.model = train_local_model(records, **kwargs)

    def triage(self, text, context=""):
        result = self.model.predict(text)
        result["engine"] = self.name
        return apply_business_rules(result)

    def info(self):
        return self.model.info()


class ApiEngine:
    """Remote engine: a public OpenAI-compatible chat endpoint."""

    name = "api"

    def __init__(self, client=None, **kwargs):
        self.client = client or TriageAPIClient(**kwargs)

    def triage(self, text, context=""):
        result = self.client.triage(text, context)
        result["engine"] = self.name
        return apply_business_rules(result)

    def info(self):
        return self.client.describe()


class EnsembleEngine:
    """Local first; only call the API when the local model is unsure."""

    name = "ensemble"

    def __init__(self, threshold=0.45, local=None, api=None):
        self.threshold = threshold
        self.local = local or LocalEngine()
        self.api = api or ApiEngine()
        self.sent_to_api = 0

    def triage(self, text, context=""):
        auto = looks_like_autoreply(text)
        base = self.local.triage(text, context)
        if auto:
            base["note"] = f"detected automated message ({auto!r}), no API call"
            base["requires_review"] = True
            return base
        if not base["requires_review"] and base["category_confidence"] >= self.threshold:
            base["note"] = "confident local answer, API skipped"
            return base
        self.sent_to_api += 1
        try:
            remote = self.api.triage(text, context)
        except APIError as exc:
            base["note"] = f"API unavailable ({str(exc)[:80]}), kept local answer"
            base["requires_review"] = True
            return base
        return combine(base, remote)

    def info(self):
        return {"engine": self.name, "threshold": self.threshold,
                "local": self.local.info(), "api": self.api.info(),
                "sent_to_api": self.sent_to_api}


def combine(local, remote):
    """Merge the two views and flag disagreements for a human."""
    same_cat = local["category"] == remote["category"]
    same_urg = local["urgency"] == remote["urgency"]
    merged = dict(remote)
    merged.update({
        "engine": "ensemble",
        "agree_category": same_cat,
        "agree_urgency": same_urg,
        "requires_review": not (same_cat and same_urg),
        "local_view": {
            "category": local["category"],
            "urgency": local["urgency"],
            "category_confidence": local["category_confidence"],
            "top_keywords": local.get("top_keywords", []),
        },
        "note": "engines agree" if same_cat and same_urg else "engines disagree - human review",
    })
    return apply_business_rules(merged)


ENGINES = {"local": LocalEngine, "api": ApiEngine, "ensemble": EnsembleEngine}


def build_engine(name="local", **kwargs):
    if name not in ENGINES:
        raise ValueError(f"unknown engine {name!r}, choose from {sorted(ENGINES)}")
    return ENGINES[name](**kwargs)


def run_batch(engine, records):
    """Triage a list of {text, context?} dicts and summarise the outcome."""
    results, errors = [], []
    for record in records:
        try:
            out = engine.triage(record["text"], record.get("context", ""))
        except APIError as exc:  # remote engine unreachable
            errors.append({"id": record.get("id"), "error": str(exc)[:200]})
            continue
        out["id"] = record.get("id")
        out["gold"] = {k: record[k] for k in ("category", "urgency") if k in record}
        results.append(out)
    summary = {
        "count": len(results),
        "errors": len(errors),
        "needs_review": sum(1 for r in results if r.get("requires_review")),
        "categories": _counts(results, "category"),
        "urgencies": _counts(results, "urgency"),
    }
    return results, summary, errors


def _counts(results, key):
    out = {}
    for result in results:
        out[result[key]] = out.get(result[key], 0) + 1
    return dict(sorted(out.items()))


def format_row(result):
    """One compact ASCII line for the CLI."""
    return "| {id:<5} | {category:<9} | {urgency:<8} | {team:<24} | {conf:>5} | {review:<3} |".format(
        id=str(result.get("id") or "")[:5],
        category=result["category"],
        urgency=result["urgency"],
        team=result["team"],
        conf=result.get("category_confidence"),
        review="yes" if result.get("requires_review") else "no",
    )
