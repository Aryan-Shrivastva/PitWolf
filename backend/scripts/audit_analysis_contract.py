"""Verify observed-battle rows cannot contaminate model training rows."""

from __future__ import annotations

import json
from pathlib import Path

from extract_decision_points import OUT_ROOT, SCHEMA_VERSION


def key(row: dict) -> tuple:
    return (row.get("year"), row.get("round"), row.get("session"),
            row.get("lap"), row.get("driver"), row.get("defender"))


def main() -> None:
    files = sorted(OUT_ROOT.glob("*/*_r.json"))
    failures: list[dict] = []
    total_analysis = 0
    total_clean = 0

    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append({"file": str(path), "issues": ["UNREADABLE"], "detail": str(error)})
            continue
        analysis = payload.get("analysisRows")
        clean = payload.get("rows") or []
        issues: list[str] = []
        if payload.get("schemaVersion") != SCHEMA_VERSION:
            issues.append("STALE_SCHEMA")
        if not isinstance(analysis, list):
            issues.append("MISSING_ANALYSIS_ROWS")
            analysis = []
        analysis_by_key = {key(row): row for row in analysis}
        for row in clean:
            observed = analysis_by_key.get(key(row))
            if observed is None:
                issues.append("CLEAN_ROW_MISSING_FROM_ANALYSIS")
                break
            if not observed.get("outcomeLabelEligible"):
                issues.append("UNCLEAN_ROW_IN_TRAINING_SET")
                break
            if not observed.get("label"):
                issues.append("UNLABELLED_ROW_IN_TRAINING_SET")
                break
        if any(row.get("outcomeLabelEligible") and not row.get("label") for row in analysis):
            issues.append("ELIGIBLE_ANALYSIS_ROW_MISSING_LABEL")
        if any(not row.get("outcomeLabelEligible") and row.get("eligibleForTraining") for row in analysis):
            issues.append("UNSCORABLE_ROW_MARKED_TRAINING_ELIGIBLE")
        if issues:
            failures.append({"file": str(path), "year": payload.get("year"), "round": payload.get("round"), "issues": sorted(set(issues))})
        total_analysis += len(analysis)
        total_clean += len(clean)

    print(json.dumps({
        "status": "PASS" if not failures else "FAIL",
        "expectedDecisionPointSchema": SCHEMA_VERSION,
        "raceFiles": len(files),
        "observedAnalysisRows": total_analysis,
        "cleanTrainingRows": total_clean,
        "outcomeUnscorableRows": total_analysis - total_clean,
        "failures": failures,
        "guarantee": "Observed-but-unscorable rows are available to analysis/replay but never enter training or held-out classification scoring.",
    }, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
