"""Command line interface.

  python run.py classify "I was charged twice" --engine ensemble
  python run.py batch data/example_inputs.json --engine api
  python run.py eval --ablation
  python run.py info
  python run.py api-test
"""

import argparse
import json
import sys
from pathlib import Path

from .api_client import APIError, TriageAPIClient
from .dataset import DATA_DIR, load_split
from .evaluate import ablation, evaluate, format_matrix, leave_one_out
from .model_local import train_local_model
from .pipeline import build_engine, format_row, run_batch

ENGINES = ("local", "api", "ensemble")


def _print(result):
    print(json.dumps(result, indent=2, default=str))


def cmd_classify(args):
    engine = build_engine(args.engine)
    result = engine.triage(args.text, args.context)
    if args.json:
        _print(result)
        return 0
    print(f"  text        : {result['text']}")
    print(f"  engine      : {result['engine']}")
    print(f"  category    : {result['category']} ({result['category_confidence']})")
    print(f"  urgency     : {result['urgency']} -> SLA {result['sla_hours']}h")
    print(f"  route to    : {result['team']}")
    print(f"  needs review: {'yes' if result['requires_review'] else 'no'}")
    if result.get("summary"):
        print(f"  summary     : {result['summary']}")
    if result.get("top_keywords"):
        print(f"  evidence    : {', '.join(result['top_keywords'])}")
    if result.get("rule_evidence"):
        print(f"  cues        : {', '.join(result['rule_evidence'])}")
    if result.get("note"):
        print(f"  note        : {result['note']}")
    return 0


def cmd_batch(args):
    path = Path(args.path) if args.path else DATA_DIR / "example_inputs.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    engine = build_engine(args.engine)
    results, summary, errors = run_batch(engine, records)
    print(f"\nBatch: {path.name}  engine={args.engine}")
    print("| id    | category  | urgency  | team                     |  conf | rev |")
    print("|-------|-----------|----------|--------------------------|-------|-----|")
    for result in results:
        print(format_row(result))
    print(f"\n{summary['count']} triaged, {summary['needs_review']} flagged for review, "
          f"{summary['errors']} errors")
    print(f"categories: {summary['categories']}")
    print(f"urgencies : {summary['urgencies']}")
    for error in errors:
        print(f"  error on {error['id']}: {error['error']}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"results": results, "summary": summary, "errors": errors},
            indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.out}")
    return 1 if errors else 0


def cmd_eval(args):
    train, test = load_split("train"), load_split(args.split)
    model = train_local_model(train)
    report = evaluate(model, test)
    print(f"model: {json.dumps(model.info())}\n")
    for key in ("category", "urgency"):
        rep = report[key]
        print(f"{key}: accuracy={rep['accuracy']} macro_f1={rep['macro_f1']} n={rep['n']}")
        for label, stats in sorted(rep["classes"].items()):
            print("  {:<10} p={:<6} r={:<6} f1={:<6} n={}".format(
                label, stats["precision"], stats["recall"], stats["f1"], stats["support"]))
        print()
    print(format_matrix("category confusion (gold rows)", report["category_confusion"],
                        sorted(report["category_confusion"])))
    print()
    print(format_matrix("urgency confusion (gold rows)", report["urgency_confusion"],
                        sorted(report["urgency_confusion"])))
    print(f"\nflagged for human review: {report['review_rate']:.0%} of "
          f"{report['category']['n']}")
    if args.mismatches:
        print("\nmismatches:")
        for item in report["mismatches"]:
            print("  {} gold={}/{} pred={}/{} {}".format(
                item["id"], item["gold_category"], item["gold_urgency"],
                item["pred_category"], item["pred_urgency"], item["text"][:50]))
    if args.ablation:
        print("\nfeature ablation (same data, fewer feature groups):")
        for name, stats in ablation(train, test).items():
            print("  {:<32} category={} urgency={} macro_f1={}".format(
                name, stats["category_accuracy"], stats["urgency_accuracy"],
                stats["category_macro_f1"]))
    if args.loo:
        print("\nleave-one-out on train (tf-idf words only):",
              json.dumps(leave_one_out(train)))
    return 0


def cmd_info(args):
    print(json.dumps(build_engine(args.engine).info(), indent=2, default=str))
    return 0


def cmd_api_test(args):
    client = TriageAPIClient()
    print(f"endpoint: {client.endpoint()}")
    print(f"model   : {client.model}")
    print(f"api key : {'from environment' if client.uses_key else 'not set (public, key-free mode)'}")
    health = client.health()
    print(json.dumps(health, indent=2, default=str))
    if not health["reachable"]:
        print("The endpoint is unreachable. Check the network, or export "
              "AI_API_BASE_URL / AI_API_KEY to use another OpenAI-compatible provider.")
        return 1
    _print(client.triage("We were billed twice today and our card company called us."))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="run.py", description="Support-ticket triage prototype")
    sub = parser.add_subparsers(dest="command", required=True)

    classify = sub.add_parser("classify", help="triage one ticket")
    classify.add_argument("text")
    classify.add_argument("--context", default="", help="extra context for the API engine")
    classify.add_argument("--engine", default="local", choices=ENGINES)
    classify.add_argument("--json", action="store_true", help="print raw triage object")
    classify.set_defaults(func=cmd_classify)

    batch = sub.add_parser("batch", help="triage a JSON file of tickets")
    batch.add_argument("path", nargs="?", default=None)
    batch.add_argument("--engine", default="local", choices=ENGINES)
    batch.add_argument("--out", default=None, help="write results to this JSON file")
    batch.set_defaults(func=cmd_batch)

    report = sub.add_parser("eval", help="metrics on the held-out split")
    report.add_argument("--split", default="test")
    report.add_argument("--mismatches", action="store_true")
    report.add_argument("--ablation", action="store_true")
    report.add_argument("--loo", action="store_true")
    report.set_defaults(func=cmd_eval)

    info = sub.add_parser("info", help="show model / engine configuration")
    info.add_argument("--engine", default="local", choices=ENGINES)
    info.set_defaults(func=cmd_info)

    api = sub.add_parser("api-test", help="check the public AI API end to end")
    api.set_defaults(func=cmd_api_test)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except APIError as exc:
        print(f"API error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

