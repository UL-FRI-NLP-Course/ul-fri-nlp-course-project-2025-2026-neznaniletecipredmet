"""Apply audit verdicts to produce a cleaned evaluation question set.

Reads the output of audit_references.py (e.g. questions_audited.json) and
emits a cleaned questions_full_v2.json by applying a deterministic policy:

  - supported            -> keep original reference
  - partial + rewrite    -> use suggested rewrite (audit_status: auto_rewritten)
  - partial + no rewrite -> keep original, mark needs_review
  - unsupported + rewrite-> use suggested rewrite (audit_status: auto_rewritten)
  - unsupported + none   -> drop reference (audit_status: dropped, needs_review)
  - skipped_negative     -> keep original (negatives are unaffected)
  - skipped_no_reference -> keep as-is
  - error                -> keep original, mark needs_review

The script preserves the original reference under reference_answer_original
for full traceability, and writes a small markdown triage report listing the
unsupported/error cases for an optional human pass.

Usage:
  python scripts/apply_reference_audit.py \
      --audited ../code/questions_audited.json \
      --out     ../code/questions_full_v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply audit verdicts to produce cleaned eval set")
    p.add_argument("--audited", required=True,
                   help="Input audit JSON (output of audit_references.py)")
    p.add_argument("--out", default=None,
                   help="Output cleaned questions JSON (default: <audited stem>_v2.json)")
    p.add_argument("--triage-md", default=None,
                   help="Path to markdown triage report (default: <out stem>_triage.md)")
    p.add_argument("--prefer-original", action="store_true",
                   help="On 'partial', keep the original (only auto-rewrite on 'unsupported'). Default off.")
    p.add_argument("--drop-needs-review", action="store_true",
                   help="Drop questions marked needs_review entirely from the output (default: keep them, judge will skip)")
    return p.parse_args()


def _resolve_input(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    candidates = [
        Path(__file__).resolve().parents[1] / path,
        Path(__file__).resolve().parents[2] / path,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return p


def _decide_action(verdict: str, has_rewrite: bool, prefer_original: bool) -> tuple[str, str]:
    """Return (action, audit_status). action in {'keep', 'rewrite', 'drop'}."""
    if verdict == "supported":
        return "keep", "supported"
    if verdict == "partial":
        if prefer_original or not has_rewrite:
            return "keep", "partial_kept_original"
        return "rewrite", "auto_rewritten"
    if verdict == "unsupported":
        if has_rewrite:
            return "rewrite", "auto_rewritten"
        return "drop", "unsupported_dropped"
    if verdict == "skipped_negative":
        return "keep", "skipped_negative"
    if verdict == "skipped_no_reference":
        return "keep", "skipped_no_reference"
    return "keep", f"unhandled_verdict_{verdict or 'none'}"


def _markdown_triage(items: list[dict]) -> str:
    lines = ["# Reference audit triage", ""]
    if not items:
        lines.append("No questions need manual review.")
        return "\n".join(lines)
    lines.append(f"{len(items)} question(s) need manual review:")
    lines.append("")
    for it in items:
        q = it.get("question") or "(no question)"
        verdict = (it.get("audit") or {}).get("verdict")
        unsupported = (it.get("audit") or {}).get("unsupported_claims") or []
        reasoning = (it.get("audit") or {}).get("reasoning") or ""
        lines.append(f"## {q}")
        lines.append("")
        lines.append(f"- **verdict**: `{verdict}`")
        lines.append(f"- **status**: `{it.get('audit_status')}`")
        if unsupported:
            lines.append("- **unsupported claims**:")
            for u in unsupported:
                lines.append(f"  - {u}")
        if reasoning:
            lines.append(f"- **judge reasoning**: {reasoning}")
        ref_orig = it.get("reference_answer_original") or it.get("reference_answer") or ""
        if ref_orig:
            lines.append(f"- **original reference**: {ref_orig}")
        ref_now = it.get("reference_answer") or ""
        if ref_now and ref_now != ref_orig:
            lines.append(f"- **rewritten reference**: {ref_now}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    audited_path = _resolve_input(args.audited)
    if not audited_path.exists():
        print(f"Audited file not found: {audited_path}", file=sys.stderr)
        raise SystemExit(1)

    with open(audited_path, "r", encoding="utf-8") as f:
        audited = json.load(f)
    if not isinstance(audited, list):
        print(f"{audited_path}: expected a JSON array", file=sys.stderr)
        raise SystemExit(1)

    out_records: list[dict] = []
    triage_items: list[dict] = []

    counts = {
        "kept_supported": 0,
        "auto_rewritten": 0,
        "kept_partial_original": 0,
        "dropped_reference_unsupported": 0,
        "negatives": 0,
        "no_reference": 0,
        "errors": 0,
    }

    for q in audited:
        if not isinstance(q, dict):
            continue
        out = dict(q)
        audit = q.get("audit") or {}
        verdict = audit.get("verdict") or "unknown"
        rewrite = audit.get("suggested_rewrite")
        has_rewrite = isinstance(rewrite, str) and rewrite.strip()
        action, status = _decide_action(verdict, bool(has_rewrite), bool(args.prefer_original))

        original_ref = q.get("reference_answer")
        out["reference_answer_original"] = original_ref
        out["audit_status"] = status

        needs_review = False

        if action == "keep":
            if verdict == "skipped_negative":
                counts["negatives"] += 1
            elif verdict == "skipped_no_reference":
                counts["no_reference"] += 1
            elif verdict == "supported":
                counts["kept_supported"] += 1
            elif verdict == "partial":
                counts["kept_partial_original"] += 1
                needs_review = True
            else:
                counts["errors"] += 1
                needs_review = True
        elif action == "rewrite":
            out["reference_answer"] = rewrite.strip()
            counts["auto_rewritten"] += 1
            needs_review = True
        elif action == "drop":
            out["reference_answer"] = ""
            out["needs_review"] = True
            counts["dropped_reference_unsupported"] += 1
            needs_review = True

        if needs_review:
            out["needs_review"] = True
            triage_items.append(out)

        if not (args.drop_needs_review and out.get("needs_review") and not out.get("reference_answer")):
            out_records.append(out)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = audited_path.with_name(audited_path.stem.replace("_audited", "") + "_v2.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)
    print(f"Wrote cleaned eval set -> {out_path}")

    triage_path = Path(args.triage_md) if args.triage_md else out_path.with_name(out_path.stem + "_triage.md")
    triage_path.parent.mkdir(parents=True, exist_ok=True)
    triage_path.write_text(_markdown_triage(triage_items), encoding="utf-8")
    print(f"Wrote triage report -> {triage_path}")

    print("=" * 50)
    print("Audit application summary:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"  total -> output rows: {len(out_records)}")
    print(f"  triage items (needs_review): {len(triage_items)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
