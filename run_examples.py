"""Run every engine over the bundled example tickets and save the real output.

    python examples/run_examples.py                    # local + api + ensemble
    python examples/run_examples.py --engines local    # offline only

Writes, next to this file:
    outputs_local.json / outputs_api.json / outputs_ensemble.json
    EXAMPLES.md   (a readable side-by-side of the same results)

Everything in those files is produced by the code in this repository, so the
README can quote real numbers instead of invented ones.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from triage.pipeline import build_engine, run_batch  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engines", default="local,api,ensemble",
                        help="comma separated: local, api, ensemble")
    parser.add_argument("--inputs", default=str(ROOT / "data" / "example_inputs.json"))
    args = parser.parse_args()

    records = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
    summaries, all_results, all_errors = {}, {}, {}

    for name in [e.strip() for e in args.engines.split(",") if e.strip()]:
        engine = build_engine(name)
        started = time.time()
        results, summary, errors = run_batch(engine, records)
        elapsed = time.time() - started
        summary["elapsed_seconds"] = round(elapsed, 2)
        summary["seconds_per_ticket"] = round(elapsed / max(1, len(results)), 2)
        summary["config"] = engine.info()
        summaries[name] = summary
        all_results[name] = results
        all_errors[name] = errors

        (HERE / f"outputs_{name}.json").write_text(
            json.dumps({"engine": name, "summary": summary, "results": results,
                        "errors": errors}, indent=2, default=str), encoding="utf-8")
        print(f"{name:<9} {summary['count']} triaged, {summary['errors']} errors, "
              f"{elapsed:5.1f}s total ({summary['seconds_per_ticket']}s/ticket) "
              f"-> outputs_{name}.json")
        for error in errors:
            print(f"          error on {error['id']}: {error['error']}")

    write_markdown(records, all_results, summaries, all_errors)
    print("wrote EXAMPLES.md")
    return 0


def _cell(result):
    return (f"{result['category']} / {result['urgency']} -> {result['team']} "
            f"({result['category_confidence']})")


def write_markdown(records, all_results, summaries, all_errors):
    engines = list(all_results)
    lines = [
        "# Example inputs and outputs",
        "",
        "Regenerate with `python examples/run_examples.py`. Every value below is",
        "real program output, so the README can quote it verbatim.",
        "",
        "## Inputs",
        "",
    ]
    for record in records:
        lines.append(f"**{record['id']}** - {record['text']}")
        if record.get("context"):
            lines.append(f"  - context: {record['context']}")
        lines.append("")

    lines += ["## Outputs", ""]
    for record in records:
        lines += [f"### {record['id']}", "",
                  f"> {record['text']}", ""]
        lines.append("| engine | result | review |")
        lines.append("|--------|--------|--------|")
        for engine in engines:
            match = next((r for r in all_results[engine] if r.get("id") == record["id"]), None)
            if match is None:
                lines.append(f"| {engine} | _error_ | - |")
                continue
            lines.append(f"| {engine} | {_cell(match)} | "
                         f"{'yes' if match.get('requires_review') else 'no'} |")
        notes = [f"{engine}: {r.get('note')}" for engine in engines
                 for r in all_results[engine] if r.get("id") == record["id"] and r.get("note")]
        for note in notes:
            lines.append("")
            lines.append(f"> {note}")
        lines.append("")

    lines += ["## Summary", "",
              "| engine | tickets | flagged | category counts | urgency counts | s/ticket |",
              "|--------|---------|---------|-----------------|----------------|---------|"]
    for engine in engines:
        summary = summaries[engine]
        lines.append(
            f"| {engine} | {summary['count']} | {summary['needs_review']} | "
            f"{summary['categories']} | {summary['urgencies']} | "
            f"{summary['seconds_per_ticket']} |")
    failures = {e: errs for e, errs in all_errors.items() if errs}
    if failures:
        lines += ["", "## Errors", ""]
        for engine, errs in failures.items():
            lines.append(f"- **{engine}**: {errs}")
    lines.append("")
    (HERE / "EXAMPLES.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
