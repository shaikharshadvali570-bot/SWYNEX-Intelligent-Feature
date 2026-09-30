"""Metrics, confusion matrices and an ablation over the two urgency heads."""

from collections import Counter

from .labels import CATEGORIES, URGENCIES


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return round(precision, 3), round(recall, 3), round(f1, 3)


def classification_report(gold, pred, labels):
    """Per-class precision/recall/F1 plus overall accuracy."""
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    report = {"accuracy": round(correct / max(1, len(gold)), 3), "n": len(gold), "classes": {}}
    macro = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        p, r, f1 = _prf(tp, fp, fn)
        report["classes"][label] = {
            "precision": p, "recall": r, "f1": f1, "support": tp + fn,
            "errors": sorted({g for g, p_ in zip(gold, pred) if p_ == label and g != label}),
        }
        macro.append(f1)
    report["macro_f1"] = round(sum(macro) / len(macro), 3)
    return report


def confusion_matrix(gold, pred, labels):
    matrix = {g: Counter() for g in labels}
    for g, p in zip(gold, pred):
        matrix[g][p] += 1
    return {g: {p: matrix[g].get(p, 0) for p in labels} for g in labels}


def format_matrix(title, matrix, labels):
    width = max(max(len(l) for l in labels), 6)
    lines = [title, "".ljust(width + 2) + "".join(l[:6].ljust(7) for l in labels)]
    for g in labels:
        row = matrix[g]
        lines.append(g[:width].ljust(width + 2) + "".join(str(row[l]).ljust(7) for l in labels))
    return "\n".join(lines)


def evaluate(model, records, **kwargs):
    gold_cat = [r["category"] for r in records]
    gold_urg = [r["urgency"] for r in records]
    preds = [model.predict(r["text"], **kwargs) for r in records]
    pred_cat = [p["category"] for p in preds]
    pred_urg = [p["urgency"] for p in preds]
    return {
        "category": classification_report(gold_cat, pred_cat, CATEGORIES),
        "urgency": classification_report(gold_urg, pred_urg, URGENCIES),
        "category_confusion": confusion_matrix(gold_cat, pred_cat, CATEGORIES),
        "urgency_confusion": confusion_matrix(gold_urg, pred_urg, URGENCIES),
        "review_rate": round(sum(p["requires_review"] for p in preds) / max(1, len(preds)), 3),
        "mismatches": [
            {
                "id": r["id"], "text": r["text"],
                "gold_category": r["category"], "pred_category": p["category"],
                "gold_urgency": r["urgency"], "pred_urgency": p["urgency"],
                "top_keywords": p["top_keywords"],
            }
            for r, p in zip(records, preds)
            if r["category"] != p["category"] or r["urgency"] != p["urgency"]
        ],
    }


def _accuracy(gold, pred):
    return round(sum(g == p for g, p in zip(gold, pred)) / max(1, len(gold)), 3)


FEATURE_VARIANTS = {
    "tfidf-only": {"features": ("words",), "fuse_rules": False},
    "tfidf+lexicon": {"features": ("words", "lex"), "fuse_rules": False},
    "tfidf+cue": {"features": ("words", "cue"), "fuse_rules": False},
    "tfidf+cue+lexicon": {"features": ("words", "cue", "lex"), "fuse_rules": False},
    "+rules-first urgency (shipped)": {"features": ("words", "cue", "lex")},
}


def ablation(train_records, eval_records, alpha=1.0):
    """Refit the same model with fewer feature groups and compare."""
    from .model_local import cue_score, score_to_level, train_local_model

    gold_cat = [r["category"] for r in eval_records]
    gold_urg = [r["urgency"] for r in eval_records]
    out = {}
    for name, kwargs in FEATURE_VARIANTS.items():
        model = train_local_model(train_records, alpha=alpha, **kwargs)
        preds = [model.predict(r["text"]) for r in eval_records]
        out[name] = {
            "category_accuracy": _accuracy(gold_cat, [p["category"] for p in preds]),
            "urgency_accuracy": _accuracy(gold_urg, [p["urgency"] for p in preds]),
            "category_macro_f1": classification_report(
                gold_cat, [p["category"] for p in preds], CATEGORIES)["macro_f1"],
        }
    rules_only = [score_to_level(cue_score(" ".join(r["text"].lower().split()))[0])
                  or "medium" for r in eval_records]
    out["rules-only"] = {
        "category_accuracy": None,
        "urgency_accuracy": _accuracy(gold_urg, rules_only),
        "category_macro_f1": None,
    }
    return out


def leave_one_out(records, alpha=1.0):
    """LOO accuracy on the training set (no separate validation split needed)."""
    from .model_local import MultinomialNB, TfidfVectorizer

    texts = [r["text"] for r in records]
    vectorizer = TfidfVectorizer().fit(texts)
    weights = [vectorizer.transform(t) for t in texts]
    results = {}
    for field, classes in (("category", CATEGORIES), ("urgency", URGENCIES)):
        labels = [r[field] for r in records]
        hits = 0
        for i in range(len(records)):
            idx = [j for j in range(len(records)) if j != i]
            nb = MultinomialNB(alpha=alpha).fit(
                [weights[j] for j in idx], [labels[j] for j in idx], classes,
                vectorizer.vocabulary_size,
            )
            probs, _ = nb.probabilities(weights[i])
            if max(probs, key=lambda k: probs[k]) == labels[i]:
                hits += 1
        results[field] = round(hits / len(records), 3)
    return results
