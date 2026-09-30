"""Label schema, routing rules and urgency escalation cues.

Keeping the schema in one place makes it easy to point the prototype at a
different problem domain: change these constants + the JSON datasets.
"""

CATEGORIES = ("billing", "technical", "shipping", "account", "other")
URGENCIES = ("low", "medium", "high", "critical")

URGENCY_RANK = {u: i for i, u in enumerate(URGENCIES)}

# Where a ticket goes once it is classified. Business rule, not a model.
ROUTING = {
    "billing": "billing-team",
    "technical": "support-l2-engineering",
    "shipping": "logistics-team",
    "account": "account-security-team",
    "other": "support-l1",
}

# Target first-response time per urgency, used for the SLA field.
SLA_HOURS = {"critical": 1, "high": 4, "medium": 24, "low": 72}

# ---------------------------------------------------------------- rule heads
# Keyword cues are weighted and summed into a single urgency score, then fused
# with the learned head. A single weak word must not escalate a ticket; a
# strong phrase (or two medium ones) can.
CUE_WEIGHTS = {
    # strong emergency signals (+3 each)
    "outage": 3.0, "emergency": 3.0, "critical": 3.0, "data loss": 3.0,
    "unauthori": 3.0, "not authorise": 3.0, "not authorize": 3.0, "suspic": 3.0,
    "stolen": 3.0, "money keeps leaving": 3.0, "all users": 3.0,
    "whole team": 3.0, "entire team": 3.0, "cannot log in": 3.0,
    "cant login": 3.0, "completely down": 3.0, "blocking a live": 3.0,
    # real time pressure (+1.5 each)
    "urgent": 1.5, "asap": 1.5, "immediately": 1.5, "right now": 1.5,
    "deadline": 1.5, "blocked": 1.5, "escalat": 1.5, "supervisor": 1.5,
    "locked out": 1.5, "crash": 1.5, "keeps": 1.5, "again": 1.0,
    "today": 1.0, "tomorrow": 1.0, "this hour": 1.5,
    # calm / informational phrasing (negative = de-escalate)
    "feature idea": -1.5, "feedback": -1.0, "how do i": -1.5, "how many": -1.5,
    "can i": -1.5, "where do i": -1.5, "where can i": -1.5, "no problem": -1.5,
    "not urgent": -2.0, "thanks": -1.0, "love": -1.5, "just wanted": -2.0,
    "question": -1.0, "whenever": -1.0, "tip": -1.0, "unsubscribe": -1.0,
}

# score -> level thresholds (>= CRITICAL -> critical, >= HIGH -> high,
# <= LOW -> low, otherwise the rules abstain and the learned head decides)
RULE_THRESHOLDS = {"critical": 3.0, "high": 1.5, "low": -1.0}

# |score| at which the keyword rules fully override the learned urgency head
RULE_TRUST = 1.5

# Domain lexicon used as a prior on top of the learned category scores.
# Keys are stemmed the same way tokenize() stems the incoming text.
CATEGORY_LEXICON = {
    "billing": {"charg": 2, "invoic": 2, "receipt": 2, "refund": 2, "payment": 2,
                "bill": 2, "price": 1, "pricing": 1, "subscription": 1, "renew": 1,
                "card": 1, "statement": 2, "vat": 2, "invoice": 2, "charge": 2, "paid": 1},
    "technical": {"error": 2, "crash": 2, "bug": 2, "api": 2, "app": 1, "slow": 1,
                  "timeout": 2, "500": 2, "503": 2, "401": 2, "webhook": 2, "sync": 2,
                  "outage": 2, "down": 1, "update": 1, "upload": 1, "export": 1,
                  "blank": 1, "fail": 1, "failur": 2, "index": 1, "unreach": 2},
    "shipping": {"order": 2, "parcel": 3, "package": 3, "track": 3, "tracking": 3,
                 "deliveri": 2, "delivered": 2, "delivery": 2, "courier": 3, "ship": 2,
                 "custom": 2, "dispatch": 3, "address": 1, "return": 1, "refund": 0,
                 "truck": 2, "warehouse": 2, "label": 1, "item": 1},
    "account": {"account": 3, "password": 3, "login": 3, "log": 1, "signin": 3,
                "two": 1, "factor": 3, "2fa": 3, "otp": 3, "locked": 2, "lock": 2,
                "device": 2, "session": 2, "email": 1, "profil": 2, "gdpr": 3,
                "delete": 1, "seat": 1, "team": 1, "verif": 2},
    "other": {"feature": 2, "idea": 2, "feedback": 2, "suggestion": 2, "compliment": 3,
              "review": 2, "unsubscribe": 2, "spam": 2, "manager": 1, "question": 1,
              "discount": 1, "quote": 1, "hours": 1, "thanks": 2, "praise": 3},
}

LEXICON_BONUS = 1.5  # weight of a lexicon hit, tuned by LOO (see README)

# Cues that indicate the ticket is NOT a real actionable request.
AUTOREPLY_CUES = ("out of office", "auto-reply", "delivery status", "do not reply")

