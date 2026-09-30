"""Offline unit tests: no network calls, no API key, no pip installs.

    python -m unittest discover -s tests -v
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from triage.api_client import TriageAPIClient, coerce, extract_json  # noqa: E402
from triage.dataset import load_split, tokenize  # noqa: E402
from triage.evaluate import ablation, evaluate  # noqa: E402
from triage.model_local import cue_score, score_to_level, train_local_model  # noqa: E402
from triage.pipeline import (  # noqa: E402
    EnsembleEngine,
    LocalEngine,
    apply_business_rules,
    combine,
    looks_like_autoreply,
    run_batch,
)

TRAIN, TEST = load_split("train"), load_split("test")
MODEL = train_local_model(TRAIN)

GOOD_REPLY = ('{"category": "billing", "urgency": "high", '
              '"summary": "charged twice", "confidence": 0.9}')


class FakeAPI(TriageAPIClient):
    """Canned replies so the tests never touch the network."""

    def __init__(self, replies=(GOOD_REPLY,)):
        super().__init__()
        self.replies = list(replies)
        self.calls = []

    def _post(self, messages):
        self.calls.append(messages)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return reply, {"total_tokens": 42}


class ExplodingAPI(FakeAPI):
    """Raises if the ensemble ever decides to call the API."""

    def _post(self, messages):
        self.calls.append(messages)
        raise AssertionError("the API must not be called here")


class TokenizerTests(unittest.TestCase):
    def test_suffix_stripping_collides_forms(self):
        for word in ("charged", "charges", "charging", "charge"):
            self.assertEqual(tokenize(word), ["charg"], word)

    def test_stopwords_removed(self):
        self.assertNotIn("the", tokenize("the invoice for the order"))

    def test_long_numbers_collapse_to_one_token(self):
        self.assertEqual(tokenize("invoice 4471 missing"), tokenize("invoice 88231 missing"))
        self.assertIn("num", tokenize("invoice 4471 missing"))

    def test_accents_folded(self):
        self.assertEqual(tokenize("caf\u00e9 charge"), tokenize("cafe charge"))


class ModelTests(unittest.TestCase):
    KEYS = ("category", "urgency", "team", "sla_hours", "category_confidence",
            "urgency_source", "top_keywords", "requires_review")

    def test_prediction_has_full_schema(self):
        out = MODEL.predict("I was charged twice for my plan")
        for key in self.KEYS:
            self.assertIn(key, out)
        self.assertIn(out["category"],
                      ("billing", "technical", "shipping", "account", "other"))

    def test_held_out_category_accuracy(self):
        hits = sum(1 for r in TEST if MODEL.predict(r["text"])["category"] == r["category"])
        self.assertGreaterEqual(hits / len(TEST), 0.65)

    def test_outage_is_critical_and_routed_to_engineering(self):
        out = MODEL.predict("Emergency: total outage, all users are down, we are losing money")
        self.assertEqual(out["urgency"], "critical")
        self.assertEqual(out["category"], "technical")
        self.assertEqual(out["team"], "support-l2-engineering")
        self.assertEqual(out["sla_hours"], 1)

    def test_gibberish_is_low_confidence(self):
        out = MODEL.predict("zxcvb qwerty asdfgh jkl")
        self.assertTrue(out["requires_review"])
        self.assertLess(out["category_confidence"], 0.4)

    def test_cue_rules_thresholds(self):
        self.assertEqual(score_to_level(cue_score("emergency outage")[0]), "critical")
        self.assertIsNone(score_to_level(cue_score("the box arrived this morning")[0]))
        self.assertEqual(score_to_level(cue_score("how do i reset it")[0]), "low")

    def test_cue_evidence_is_reported(self):
        out = MODEL.predict("URGENT: our platform is down, we are losing money")
        self.assertIn("urgent", out["rule_evidence"])
        self.assertEqual(out["urgency"], "high")  # one cue: high, not critical
        self.assertEqual(out["urgency_source"], "rules")

    def test_urgency_head_beats_random_guessing(self):
        # 4 classes -> 0.25 by chance; the shipped engine should clear that
        self.assertGreater(evaluate(MODEL, TEST)["urgency"]["accuracy"], 0.35)



class ApiParsingTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(extract_json(GOOD_REPLY)["category"], "billing")

    def test_json_in_code_fence(self):
        self.assertEqual(extract_json("```json\n" + GOOD_REPLY + "\n```")["urgency"], "high")

    def test_json_buried_in_prose(self):
        raw = "Sure, here you go: " + GOOD_REPLY + " Hope that helps!"
        self.assertEqual(extract_json(raw)["confidence"], 0.9)

    def test_missing_object_raises(self):
        with self.assertRaises(ValueError):
            extract_json("this looks like billing and it seems urgent")

    def test_nested_object_is_balanced(self):
        raw = 'not json yet {"a": {"b": "}"}, "category": "billing"} trailing'
        self.assertEqual(extract_json(raw)["category"], "billing")

    def test_coerce_fuzzy_labels_recovered_but_off_schema(self):
        out = coerce({"category": "Billing issues", "urgency": "HIGH"})
        self.assertEqual((out["category"], out["urgency"]), ("billing", "high"))
        self.assertFalse(out["valid"])  # recovered, but triggers one retry

    def test_coerce_rejects_unknown_labels(self):
        out = coerce({"category": "alien", "urgency": "low"})
        self.assertEqual(out["category"], "other")
        self.assertFalse(out["valid"])

    def test_coerce_tolerates_bad_confidence(self):
        self.assertEqual(coerce({"category": "billing", "confidence": "high"})["confidence"], 0.5)


class ApiEngineTests(unittest.TestCase):
    def test_default_config_is_key_free(self):
        client = TriageAPIClient(api_key="")
        self.assertFalse(client.uses_key)
        self.assertTrue(client.endpoint().endswith("/openai"))

    def test_custom_gateway_endpoint(self):
        client = TriageAPIClient(base_url="https://ai-gateway.vercel.sh/v1", api_key="sk-test")
        self.assertEqual(client.endpoint(),
                         "https://ai-gateway.vercel.sh/v1/chat/completions")

    def test_key_never_appears_in_description(self):
        client = TriageAPIClient(api_key="sk-secret-value")
        self.assertNotIn("sk-secret-value", json.dumps(client.describe()))

    def test_triage_returns_routed_result(self):
        out = FakeAPI().triage("charged twice today")
        self.assertEqual(out["category"], "billing")
        self.assertEqual(out["team"], "billing-team")
        self.assertEqual(out["sla_hours"], 4)

    def test_bad_reply_triggers_one_retry(self):
        client = FakeAPI(["I think this is billing related, quite urgent", GOOD_REPLY])
        out = client.triage("charged twice")
        self.assertEqual(out["attempts"], 2)
        self.assertEqual(out["category"], "billing")
        self.assertEqual(len(client.calls), 2)


class PipelineTests(unittest.TestCase):
    def test_business_rules_are_deterministic(self):
        out = apply_business_rules({"category": "shipping", "urgency": "low"})
        self.assertEqual((out["team"], out["sla_hours"]), ("logistics-team", 72))

    def test_autoreply_detection(self):
        self.assertTrue(looks_like_autoreply("Out of office until Monday"))
        self.assertIsNone(looks_like_autoreply("my parcel is missing"))

    def test_ensemble_skips_api_when_local_is_confident(self):
        local = LocalEngine(TRAIN)
        text = "I was charged twice on the same invoice, please refund the duplicate charge"
        self.assertGreaterEqual(local.triage(text)["category_confidence"], 0.45)
        engine = EnsembleEngine(threshold=0.45, local=local, api=ExplodingAPI())
        self.assertEqual(engine.triage(text)["note"], "confident local answer, API skipped")

    def test_ensemble_escalates_uncertain_text(self):
        api = FakeAPI()
        engine = EnsembleEngine(threshold=0.45, local=LocalEngine(TRAIN), api=api)
        out = engine.triage("asdf zxcv qwer thing happened ok")
        self.assertTrue(api.calls, "the API should have been consulted")
        self.assertEqual(out["engine"], "ensemble")

    def test_ensemble_survives_api_failure(self):
        from triage.api_client import APIError

        class DeadAPI(FakeAPI):
            def _post(self, messages):
                raise APIError("HTTP 402: quota exceeded")

        local = LocalEngine(TRAIN)
        engine = EnsembleEngine(threshold=0.99, local=local, api=DeadAPI())
        out = engine.triage("my order has not moved in nine days")
        self.assertIn("kept local answer", out["note"])
        self.assertEqual(out["category"], local.triage("my order has not moved in nine days")["category"])

    def test_ensemble_detects_autoreply_without_api(self):
        api = FakeAPI()
        engine = EnsembleEngine(local=LocalEngine(TRAIN), api=api)
        out = engine.triage("This is an auto-reply, I am out of office all week")
        self.assertEqual(api.calls, [])
        self.assertTrue(out["requires_review"])

    def test_combine_flags_disagreement(self):
        local = {"category": "billing", "urgency": "medium", "category_confidence": 0.5,
                 "top_keywords": ["charg"]}
        remote = {"category": "technical", "urgency": "medium", "category_confidence": 0.8}
        merged = combine(local, remote)
        self.assertTrue(merged["requires_review"])
        self.assertFalse(merged["agree_category"])
        self.assertEqual(merged["local_view"]["category"], "billing")

    def test_run_batch_summarises(self):
        results, summary, errors = run_batch(
            LocalEngine(TRAIN),
            [{"id": "1", "text": "charged twice"}, {"id": "2", "text": "where is my parcel"}])
        self.assertEqual(len(results), 2)
        self.assertEqual(summary["count"], 2)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

