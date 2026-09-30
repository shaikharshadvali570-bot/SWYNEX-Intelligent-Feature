# ticket-triage-ai

> Turn a raw customer message into a routing decision - **category, urgency,
> owning team, SLA** - in one call, with a confidence score and a human-review
> flag when the model is not sure.
>
> A Naive Bayes model trained **in-process on 100 labelled tickets** (no
> downloads, no `pip install`) plus a **public LLM API** the prototype calls for
> the tickets the small model is unsure about.
>
> Python standard library only. No API key needed. Works offline.

---

## The problem

Support teams spend real time on the first few seconds of every ticket: which
queue does it belong in, how urgent is it, who has to answer, and how fast.
Getting that wrong has a direct cost - a payment fraud ticket sitting in the
general queue for a day, or a developer being paged for a "how do I add a
colleague" question.

So the problem this prototype defines is:

> **Given one free-text support message, output a single triage decision:
> category, urgency, owning team and response SLA - fast enough to sit in front
> of a human, with a confidence score that tells the human when to distrust
> the result.**

Why this is a good prototype problem: the output contract is crisp, it can be
scored objectively against labelled tickets, and it fails in interesting ways -
bag-of-words is good at topic and weak at urgency, an LLM is good at both but
is slower and costs money per call. That tension is what the two engines are
for.

**Input** - one customer message:

```
We were billed 3 times today for the same invoice and our bank is calling us,
this needs fixing this hour.
```

**Output** - one JSON object:

```json
{
  "category": "billing",
  "category_confidence": 0.62,
  "urgency": "critical",
  "urgency_source": "rules",
  "team": "billing-team",
  "sla_hours": 1,
  "requires_review": false,
  "top_keywords": ["charg", "bill"]
}
```

## Quick start

```bash
git clone https://github.com/<your-username>/ticket-triage-ai.git
cd ticket-triage-ai

python run.py classify "I was charged twice and my parcel is late"   # offline model
python run.py classify "warehouse is down, trucks waiting" --engine ensemble
python run.py batch data/example_inputs.json --engine local          # table of results
python run.py eval --ablation --loo                                  # metrics
python run.py api-test                                               # check the public AI API
python examples/run_examples.py                                      # regenerate examples
python -m unittest discover -s tests                                 # offline test suite
```

Requires Python 3.9+ (developed and verified on CPython 3.13).
`requirements.txt` is intentionally empty - there is nothing to install.

## How it works

```
  customer message
        |
        v
  +--------------+  words  ->  sublinear TF x IDF
  |  featurize   |  cues   ->  named urgency phrases as features
  +--------------+  lex    ->  per-category lexicon hit counts
        |
        v
  +------------------+        +--------------------------+
  | Multinomial NB   |        | keyword escalation cues  |
  | head 1: category |        | (outage, asap, locked    |
  | head 2: urgency  |        |  out, ... ) -> urgency   |
  +------------------+        +--------------------------+
        |                               |
        +---------------+---------------+
                        v
        +--------------------------------------+
        | business rules: category -> team,     |
        | urgency -> SLA, requires_review flag |
        +--------------------------------------+
                        v
              one triage object (JSON)
```

| Module | Responsibility |
|--------|----------------|
| `triage/dataset.py` | load JSON, lowercase, strip accents, mask invoice ids, light suffix stemming, stopwords |
| `triage/model_local.py` | TF-IDF vectorizer, Multinomial NB, the three feature groups, cue rules, explainability output |
| `triage/api_client.py` | OpenAI-compatible HTTP client: prompt, tolerant JSON parsing, one repair retry, disk cache, backoff |
| `triage/labels.py` | the label schema, routing table, SLA table, cue weights, lexicon - all in one place |
| `triage/pipeline.py` | the three engines, business rules, ensemble logic, batch runs |
| `triage/evaluate.py` | accuracy / precision / recall / F1, confusion matrices, LOO, feature ablation |
| `triage/cli.py` | `classify`, `batch`, `eval`, `info`, `api-test` |

**The model** (`triage/model_local.py`) is a plain multinomial Naive Bayes over
three feature groups, all learned rather than hand-tuned:

- `words` - sublinear TF x IDF of the ticket text;
- `cue` - named urgency phrases (`outage`, `asap`, `locked out`, ...) turned into
  features, so the *data* decides how much each phrase matters;
- `lex` - per-category lexicon hit counts (billing words, logistics words, ...),
  also learned rather than asserted.

Confidence is a softmax over the class scores, shrunk toward uniform when the
text contains few known words, so the model cannot be confident about a sentence
full of words it has never seen. It is a *relative agreement* score, not a
calibrated probability - the README does not pretend otherwise.

**Two deliberate non-ML decisions:**

1. *Urgency is decided by an escalation dictionary first*
   (`labels.py:CUE_WEIGHTS`). With ~100 labelled tickets the learned urgency
   head is weaker than a hand-audited rule list - the ablation table below has
   the numbers. Shipping the worse of two available options would have been the
   wrong call, and the rules are auditable, which matters for on-call paging.
2. *Routing and SLA are business rules, never model output*
   (`pipeline.apply_business_rules`). An LLM inventing a plausible-looking team
   name is a bug, not a feature.

Swapping in scikit-learn would be a ~10 line change (`TfidfVectorizer` +
`MultinomialNB(alpha=0.5)`). The reason for the hand-rolled version is that the
demo then runs with zero installs and zero model downloads, so a reviewer can
`git clone && python run.py` and it just works.

## The AI API

The remote engine talks to any **OpenAI-compatible chat-completions** endpoint.
Out of the box that is [Pollinations AI](https://pollinations.ai) - a public
text API that needs **no account and no key** - serving `openai-fast`
(GPT-OSS 20B) on its anonymous tier.

```python
# triage/api_client.py, in essence
POST https://text.pollinations.ai/openai
{"model": "openai-fast", "temperature": 0, "seed": 11,
 "messages": [{"role": "system", "content": <schema instructions>},
              {"role": "user",   "content": <ticket + context>}]}
```

What the client does around that call, because a demo that only works on a
lucky reply is not a demo:

- **tolerant parsing** - strips ``` fences and prose, balances braces without
  breaking on `{`/`"` inside strings, then validates the labels;
- **one repair retry** - an off-schema reply is re-sent with the schema
  repeated, and the attempt count is reported;
- **fuzzy label recovery** - `"Billing issues"` -> `billing`, but the reply is
  still marked off-schema so a human sees it;
- **backoff + pacing** - 3 s between calls (`AI_API_MIN_INTERVAL`) because the
  anonymous tier is rate limited; `402`/`429` are retried with exponential
  backoff while `400`/`401`/`403` fail fast;
- **disk cache** (`.cache/api/`) keyed by endpoint+prompt+model, so re-running
  the examples is free, deterministic and works offline. Set `AI_API_CACHE=0`
  to bypass it;
- **no secrets, ever** - keys are read from the environment only, never written
  to disk, never printed, and there is a test asserting the key cannot leak into
  the engine description.

Point it somewhere else with environment variables (see `.env.example`):

```bash
export AI_API_BASE_URL=https://ai-gateway.vercel.sh/v1
export AI_API_MODEL=openai/gpt-4o-mini
export AI_API_KEY=sk-ai-...          # stays in your shell, never in the repo
python run.py api-test
```

Also works unchanged with OpenRouter, OpenAI, a local Ollama
(`AI_API_BASE_URL=http://localhost:11434/v1`), or any gateway that speaks
`/chat/completions`.

## Three engines, one output format

| Engine | What it is | Network | Cost | Use it when |
|--------|------------|---------|------|-------------|
| `local` | TF-IDF + Multinomial Naive Bayes; trains and triages 100 tickets in well under a second | none | free | offline, private data, bulk triage |
| `api` | public LLM API (OpenAI-compatible chat completions) | yes | free tier / your key | messy wording, richer summaries |
| `ensemble` | local first, API only when `confidence < 0.45` | sometimes | a fraction of `api` | production default |

```bash
python run.py classify "..." --engine ensemble
```

The ensemble also short-circuits automated messages (out-of-office auto-replies)
and falls back to the local answer if the API is unreachable, so a rate-limited
or offline run degrades instead of failing.

## Results

Every number below is produced by `python run.py eval` on the **30 held-out
tickets** in `data/test.json`, with the model trained on the **100 tickets** in
`data/train.json`. Reproduce with `python run.py eval --ablation --loo
--mismatches`.

### Local engine on the held-out set (n=30)

| head | accuracy | macro-F1 | chance level |
|------|----------|----------|--------------|
| category (5 classes) | **0.667** | 0.651 | 0.200 |
| urgency (4 classes) | **0.500** | 0.460 | 0.250 |

Per class, which is where the real information is:

| category | precision | recall | F1 | support |  | urgency | precision | recall | F1 | support |
|----------|-----------|--------|----|---------|--|----------|-----------|--------|----|---------|
| account | 0.750 | 1.000 | 0.857 | 6 |  | low | 0.667 | 0.600 | 0.632 | 10 |
| billing | 0.714 | 0.833 | 0.769 | 6 |  | medium | 0.500 | 0.625 | 0.556 | 8 |
| shipping | 0.800 | 0.667 | 0.727 | 6 |  | high | 0.429 | 0.375 | 0.400 | 8 |
| technical | 0.500 | 0.500 | 0.500 | 6 |  | critical | 0.250 | 0.250 | 0.250 | 4 |
| other | 0.500 | 0.333 | 0.400 | 6 |  | | | | | |

`other` is the classic catch-all and behaves like one: it absorbs anything
unusual, which is why its recall is the worst of the five. `critical` is the
hardest urgency level - it is rare (4/30) and needs to be *earned* by the model,
so most misses land one level down in `high`. The escalation dictionary exists
precisely because of that column.

### Feature ablation - the same data, fewer features

| variant | category acc | urgency acc | category macro-F1 |
|---------|--------------|-------------|-------------------|
| `tfidf-only` | 0.667 | 0.367 | 0.651 |
| `tfidf+lexicon` | 0.700 | 0.367 | 0.681 |
| `tfidf+cue` | 0.633 | 0.400 | 0.625 |
| `tfidf+cue+lexicon` | 0.700 | 0.367 | 0.681 |
| **`+rules-first urgency` (shipped)** | **0.700** | **0.467** | 0.681 |
| `rules-only` (no learning at all) | – | 0.433 | – |

Reading it honestly:

- the **lexicon** is the only feature group that clearly helps category
  (+3.3 points, +3.0 macro-F1); raw TF-IDF alone confuses `other` with
  `technical`;
- **cue features help urgency only slightly** (0.367 -> 0.400);
- **switching urgency to rules-first is the single biggest win**
  (0.400 -> 0.467), and the rules alone (0.433) beat the learned head
  (0.367). That is why it ships, and why the README says so instead of
  pretending the model does everything.

Leave-one-out over the 100 training tickets (TF-IDF words only, no lexicon):
`category 0.740`, `urgency 0.320` - consistent with the held-out numbers and a
reminder that 100 tickets is a small sample, not a benchmark.

### Where it actually fails

`python run.py eval --mismatches` prints all 20; three worth calling out because
they are the *interesting* errors, not noise:

- `"Images render blurry in the editor on the desktop app after the newest
  release."` -> `other` instead of `technical`. No lexicon word, no error code,
  just a noun phrase.
- `"I paid for two seats but only one appears active in the licence list."` ->
  `shipping`. "active in the list" reads like a manifest.
- `"Can I add a second delivery address to my account for gifts?"` -> `account`
  instead of `shipping`. Genuinely ambiguous: it mentions both.

### The AI API engine, on the 6 demo tickets

Real run, recorded in `examples/outputs_api.json` and `examples/EXAMPLES.md`:

| ticket | local | public API | agree? |
|--------|-------|------------|--------|
| X-1 double charge | billing / high (0.70) | billing / critical (0.95) | category yes |
| X-2 app crash | technical / critical (0.66) | technical / high (0.95) | category yes |
| X-3 tracking stuck | shipping / medium (0.75) | shipping / high (0.95) | category yes |
| X-4 possible account takeover | account / critical (0.71) | account / critical (0.95) | both |
| X-5 praise for the UI | account / low (0.47) | **other / low (0.99)** | **no** - API right |
| X-6 warehouse down | technical / critical (0.41) | shipping / high (0.95) | **no** |

Two things worth reading twice:

- **X-6 is the escalation working.** The local model was at 0.41, below the
  0.45 threshold, so the ensemble called the API; the engines disagreed and the
  ticket was flagged `requires_review: true` instead of being silently routed.
  "Warehouse cannot print labels, trucks waiting" is honestly ambiguous between
  an ops incident and a broken feature, and a human should look.
- **X-5 is the escalation failing.** Local confidence 0.474 was just above the
  threshold, so the API was never called - and the local model was wrong
  (a compliment about the onboarding flow is `other`, not `account`). This is
  the real cost of a fixed threshold, and the fix is calibration on a dev set
  (see *What I would do next*), not a bigger threshold: raising it would have
  pushed more volume to the paid engine and still not caught this case.

Latency and availability, from the recorded run: the local engine triaged all
six tickets in well under a second total; the API engine needed ~1.7 s per
ticket, 5 of them served from the disk cache. Earlier attempts on the same day
*did* hit `HTTP 402` on the shared anonymous tier, which is why the client
paces itself, retries with backoff and caches - the run is reproducible rather
than luck. The LLM reported `0.95` confidence on every ticket, including X-6
where it disagreed with the local model, so its self-reported confidence should
not be used as a routing signal.



- **The dataset is 100 synthetic tickets written for this prototype.** The
  numbers below measure "does this pipeline learn the task", not "would this
  work on Zendesk". Real data is noisier, multilingual, and the labels are
  genuinely contested (`high` vs `medium` is a judgement call). The 30-ticket
  test set means one mistake is worth 3.3 points, so treat the last digit as
  noise.
- **Urgency is the weak head.** Bag-of-words features cannot see that
  "we're fine, just a question" is calm and "everything is on fire" is not.
  The rule dictionary carries that field until there is more labelled data.
- **The confidence numbers are not calibrated probabilities.** They are softmax
  scores, damped when the text is unfamiliar. They are useful for *ranking*
  and for the `requires_review` flag, not for saying "92% sure".
- **The free API tier is shared and rate limited.** A run may see `HTTP 402` on
  some tickets; the client retries, paces itself, caches, and the ensemble
  falls back to the local answer. `examples/EXAMPLES.md` shows exactly that
  happening, which is a fairer picture than a cherry-picked run.
- **The LLM engine is overconfident.** The anonymous model answered `0.95` on
  every ticket, including ones the local model rated 0.41 and disagreed with.
  Self-reported LLM confidence should not be trusted as a routing signal.
- **Not a security boundary.** The prototype sends ticket text to a third-party
  API when the `api`/`ensemble` engine is used. For real customer data that
  needs a DPA, an allow-list, and PII scrubbing before anything is sent.

## What I would do next

1. **Replace synthetic data with 1-2k real tickets** and add a proper
   `train/dev/test` split plus a k-fold evaluation; the tuning loop here is a
   grid over two hyperparameters, which is enough to show the process but not
   enough to trust the optimum.
2. **Ordinal regression for urgency** (e.g. a cumulative-link model, or
   Krippendorff's alpha to measure label agreement first - if humans disagree
   about the gold labels, no model will look good).
3. **Calibrate the confidence** on a dev split (isotonic or Platt scaling) and
   measure whether `requires_review` actually catches errors: the interesting
   number is recall on the flag, not accuracy.
4. **Log drift**: track feature/token drift and the flag rate weekly, so a
   queue that slowly fills with `other` shows up before users notice.
5. **Add a small FastAPI service** so the model is reachable by a real
   ticketing system, with a batch endpoint and a 100 ms cached-response path.
6. **PII redaction before the API hop**, plus a per-tenant opt-out.

## Repository layout

```
ticket-triage-ai/
├── README.md               this file
├── SETUP_GITHUB.md         how to publish it (browser, git, or Desktop)
├── LICENSE                 MIT
├── requirements.txt        intentionally empty - standard library only
├── .env.example            optional provider settings, no secrets
├── run.py                  single entry point for the CLI
├── data/
│   ├── train.json          100 labelled tickets
│   ├── test.json           30 held-out labelled tickets
│   └── example_inputs.json 6 demo inputs
├── triage/
│   ├── labels.py           schema, routing, SLA, cue weights, lexicon
│   ├── dataset.py          loading, normalisation, stemming
│   ├── model_local.py      TF-IDF + NB + feature groups + rules
│   ├── api_client.py       OpenAI-compatible client (cache, retry, parsing)
│   ├── pipeline.py         local / api / ensemble engines, business rules
│   ├── evaluate.py         metrics, confusion matrices, LOO, ablation
│   └── cli.py              command line interface
├── examples/
│   ├── run_examples.py     regenerates the outputs below
│   ├── EXAMPLES.md         side-by-side results for the 6 demo tickets
│   └── outputs_*.json      raw engine output, committed for review
└── tests/
    └── test_triage.py      32 offline tests (no network, no keys)
```

## Using it on a different problem

The prototype is deliberately domain-agnostic. To retarget it (spam filtering,
bug triage, expense categorisation):

1. edit `CATEGORIES` / `URGENCIES` / `ROUTING` / `SLA_HOURS` / `CUE_WEIGHTS` /
   `CATEGORY_LEXICON` in `triage/labels.py`;
2. regenerate `data/train.json` and `data/test.json` with your own labels;
3. run `python run.py eval --ablation --loo` and read the confusion matrix.

Nothing else in the code knows what a support ticket is - the JSON datasets are
the only place the domain lives.

## License

MIT - see [LICENSE](LICENSE).
