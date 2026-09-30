"""Local baseline model: TF-IDF weighting + Multinomial Naive Bayes.

Written from scratch with the standard library so the prototype runs with zero
installs and zero API keys. Two learned heads (category, urgency) are fused with
two small, auditable rule heads (urgency cues + a domain lexicon prior).
"""

import math
import re
from collections import Counter, defaultdict

from .dataset import _stem, tokenize
from .labels import (
    CATEGORY_LEXICON,
    CATEGORIES,
    CUE_WEIGHTS,
    LEXICON_BONUS,
    ROUTING,
    RULE_THRESHOLDS,
    RULE_TRUST,
    SLA_HOURS,
    URGENCIES,
    URGENCY_RANK,
)


def damped_softmax(scores, matched_tokens, divisor=6.0):
    """Softmax over class scores, shrunk toward uniform when signal is thin."""
    classes = sorted(scores)
    k = len(classes)
    uniform = 1.0 / k
    values = [scores[c] for c in classes]
    mean = sum(values) / k
    var = sum((v - mean) ** 2 for v in values) / k
    temperature = max(0.45, math.sqrt(var))
    exps = {c: math.exp((scores[c] - mean) / temperature) for c in classes}
    total = sum(exps.values()) or 1.0
    probs = {c: exps[c] / total for c in classes}
    weight = min(1.0, matched_tokens / divisor)  # few known words -> stay unsure
    return {c: uniform * (1 - weight) + probs[c] * weight for c in classes}


class TfidfVectorizer:
    """Sublinear-TF x IDF vectorizer returning sparse dict vectors."""

    def __init__(self, tokenizer=tokenize):
        self.tokenizer = tokenizer
        self.df = Counter()
        self.idf = {}
        self.n_docs = 0

    def fit(self, texts):
        self.df = Counter()
        self.n_docs = 0
        for text in texts:
            self.n_docs += 1
            for tok in set(self.tokenizer(text)):
                self.df[tok] += 1
        self.idf = {
            tok: math.log((1 + self.n_docs) / (1 + df)) + 1.0
            for tok, df in self.df.items()
        }
        return self

    @property
    def vocabulary_size(self):
        return len(self.idf)

    def transform(self, text):
        weights = {}
        for tok, count in Counter(self.tokenizer(text)).items():
            idf = self.idf.get(tok)
            if idf is None:
                continue  # unseen word carries no learned signal
            weights[tok] = (1.0 + math.log(count)) * idf
        return weights


class MultinomialNB:
    """Multinomial NB over non-negative tf-idf weights."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = ()
        self.feature = defaultdict(Counter)
        self.total = Counter()
        self.class_count = Counter()
        self.log_prior = {}
        self.vocab_size = 1

    def fit(self, weighted_docs, labels, classes, vocab_size=None):
        self.classes = tuple(classes)
        self.feature = defaultdict(Counter)
        self.total = Counter()
        self.class_count = Counter(labels)
        self.vocab_size = vocab_size or (len(set().union(*self.feature.values())) or 1)
        n = max(1, len(labels))
        for weights, label in zip(weighted_docs, labels):
            for tok, value in weights.items():
                self.feature[label][tok] += value
                self.total[label] += value
        self.log_prior = {
            c: math.log((self.class_count[c] + 1.0) / (n + len(self.classes)))
            for c in self.classes
        }
        return self

    def _log_theta(self, cls, token):
        denom = self.total[cls] + self.alpha * self.vocab_size
        return math.log((self.feature[cls].get(token, 0.0) + self.alpha) / denom)

    def scores(self, weights):
        out = {}
        for cls in self.classes:
            score = self.log_prior[cls]
            for tok, value in weights.items():
                score += value * self._log_theta(cls, tok)
            out[cls] = score
        return out

    def probabilities(self, weights):
        return damped_softmax(self.scores(weights), len(weights)), self.scores(weights)


def cue_score(flat_text, weights=None):
    """Sum the urgency cue weights found in the text -> (score, [cues])."""
    weights = CUE_WEIGHTS if weights is None else weights
    hits = [(cue, w) for cue, w in weights.items() if cue in flat_text]
    return round(sum(w for _, w in hits), 3), sorted(c for c, _ in hits)


def score_to_level(score, thresholds=None):
    """Map a cue score onto an urgency level, or None for 'rules abstain'."""
    t = RULE_THRESHOLDS if thresholds is None else thresholds
    if score >= t["critical"]:
        return "critical"
    if score >= t["high"]:
        return "high"
    if score <= t["low"]:
        return "low"
    return None


# Lexicon keys are stemmed exactly like incoming text so lookups line up.
LEXICON = {c: {_stem(k): w for k, w in table.items()} for c, table in CATEGORY_LEXICON.items()}


def slug(text):
    return re.sub(r"\W+", "_", text.strip()).strip("_")


def featurize(text, vectorizer, groups=("words", "cue", "lex"), lexicon_bonus=LEXICON_BONUS):
    """Turn raw text into NB features.

    words = tf-idf of the ticket text
    cue   = named urgency phrases, so NB learns which phrases mean what
    lex   = per-category lexicon hit counts, again learned rather than hand-set
    """
    flat = " ".join(text.lower().split())
    features = {}
    if "words" in groups:
        features.update(vectorizer.transform(text))
    words = vectorizer.transform(text) if "words" not in groups else features
    if "cue" in groups:
        for cue, weight in CUE_WEIGHTS.items():
            if cue in flat:
                key = "cue:" + slug(cue)
                features[key] = max(features.get(key, 0.0), abs(weight))
    if "lex" in groups:
        for category, table in LEXICON.items():
            hits = sum(table.get(tok, 0) for tok in words)
            if hits:
                features["lex:" + category] = lexicon_bonus * min(hits, 6) / 3.0
    return features


class TriageModel:
    """Local triage engine: learned heads fused with rule heads."""

    def __init__(self, vectorizer, category, urgency, train_size,
                 lexicon_bonus=LEXICON_BONUS, thresholds=RULE_THRESHOLDS,
                 features=("words", "cue", "lex"), fuse_rules=True):
        self.vectorizer = vectorizer
        self.category = category
        self.urgency = urgency
        self.train_size = train_size
        self.lexicon_bonus = lexicon_bonus
        self.thresholds = thresholds
        self.features = tuple(features)
        self.fuse_rules = fuse_rules

    def info(self):
        return {
            "engine": "local-tfidf-nb",
            "train_examples": self.train_size,
            "vocabulary_size": self.vectorizer.vocabulary_size,
            "feature_groups": list(self.features),
            "lexicon_bonus": self.lexicon_bonus,
            "rule_fusion": self.fuse_rules,
            "category_counts": dict(sorted(self.category.class_count.items())),
            "urgency_counts": dict(sorted(self.urgency.class_count.items())),
        }

    def _features(self, text, use_lexicon=True, use_cues=True):
        groups = tuple(g for g in self.features
                       if (g != "lex" or use_lexicon) and (g != "cue" or use_cues))
        return featurize(text, self.vectorizer, groups, self.lexicon_bonus)

    def _lexicon_hits(self, features):
        """Per-category lexicon evidence, kept for the explainability output."""
        return {c: round(features.get("lex:" + c, 0.0), 3) for c in CATEGORIES}

    def category_scores(self, features):
        return self.category.scores(features)

    def _keywords(self, features, category, runner_up):
        def contribution(tok):
            delta = (self.category._log_theta(category, tok)
                     - self.category._log_theta(runner_up, tok))
            return features[tok] * delta

        candidates = [t for t in features if not t.startswith("lex:")]
        return sorted(candidates, key=contribution, reverse=True)[:4]

    def predict(self, text, use_lexicon=True, use_cues=True, fuse_rules=None):
        features = self._features(text, use_lexicon=use_lexicon, use_cues=use_cues)
        matched = len(features)

        cat_probs = damped_softmax(self.category_scores(features), matched)
        category = max(sorted(cat_probs), key=lambda k: cat_probs[k])
        runner_up = max((c for c in cat_probs if c != category), key=lambda k: cat_probs[k])

        urg_probs = damped_softmax(self.urgency.scores(features), matched)
        # argmax over the learned head; ties resolve upward (escalate when torn)
        model_urgency = max(URGENCIES, key=lambda u: (urg_probs[u], URGENCY_RANK[u]))

        # Urgency is a business rule as much as a learned pattern. On ~100
        # labelled tickets the learned head is weaker than the escalation
        # dictionary (see the ablation table in the README), so the rules lead
        # and the model is the fallback. fuse_rules=False flips that.
        fuse = self.fuse_rules if fuse_rules is None else fuse_rules
        flat = " ".join(text.lower().split())
        score, evidence = cue_score(flat)
        level = score_to_level(score, self.thresholds) if fuse else None
        if level is None:
            urgency, source = model_urgency, "model"
        else:
            weight = min(1.0, abs(score) / RULE_TRUST)
            value = ((1 - weight) * URGENCY_RANK[model_urgency]
                     + weight * URGENCY_RANK[level])
            urgency = min(URGENCIES,
                          key=lambda u: (abs(URGENCY_RANK[u] - value), -URGENCY_RANK[u]))
            source = "rules" if weight >= 0.99 else "model+rules"

        return {
            "text": text,
            "category": category,
            "category_confidence": round(cat_probs[category], 3),
            "urgency": urgency,
            "urgency_model_guess": model_urgency,
            "urgency_source": source,
            "urgency_rule_score": score,
            "urgency_confidence": round(urg_probs[urgency], 3),
            "team": ROUTING[category],
            "sla_hours": SLA_HOURS[urgency],
            "top_keywords": self._keywords(features, category, runner_up),
            "lexicon_hits": self._lexicon_hits(features),
            "rule_evidence": evidence,
            "requires_review": cat_probs[category] < 0.40,
            "category_probabilities": {k: round(v, 3) for k, v in sorted(cat_probs.items())},
            "urgency_probabilities": {k: round(v, 3) for k, v in sorted(urg_probs.items())},
        }


def train_local_model(records=None, alpha=0.5, lexicon_bonus=LEXICON_BONUS,
                      thresholds=RULE_THRESHOLDS, features=("words", "cue", "lex"),
                      fuse_rules=True):
    """Fit the vectorizer and both learned heads on labelled records."""
    from .dataset import load_split

    records = records if records is not None else load_split("train")
    texts = [r["text"] for r in records]
    vectorizer = TfidfVectorizer().fit(texts)
    vectors = [featurize(t, vectorizer, features, lexicon_bonus) for t in texts]
    category = MultinomialNB(alpha).fit(
        vectors, [r["category"] for r in records], CATEGORIES, vectorizer.vocabulary_size
    )
    urgency = MultinomialNB(alpha).fit(
        vectors, [r["urgency"] for r in records], URGENCIES, vectorizer.vocabulary_size
    )
    return TriageModel(vectorizer, category, urgency, len(records), lexicon_bonus,
                       thresholds, features, fuse_rules)

