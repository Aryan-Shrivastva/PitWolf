"""Audit selector rosters in every cached decision-point race.

The UI selector must use the official session participant roster, never the
filtered collection of model-training battle rows. This catches races where a
retirement, an excluded label window, or no close battle would otherwise make
an entrant disappear from Strategy, Energy, or Overtake.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from extract_decision_points import OUT_ROOT, SCHEMA_VERSION


def main() -> None:
    files = sorted(OUT_ROOT.glob("*/*_r.json"))
    failures: list[dict[str, object]] = []
    participant_counts: Counter[int] = Counter()

    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append({"file": str(path), "reason": "UNREADABLE", "detail": str(error)})
            continue

        participants = payload.get("participants")
        finish_positions = payload.get("finishPositions") or {}
        issues: list[str] = []
        if payload.get("schemaVersion") != SCHEMA_VERSION:
            issues.append("STALE_SCHEMA")
        if not isinstance(participants, list) or not participants:
            issues.append("MISSING_PARTICIPANT_ROSTER")
            participants = []
        elif any(not isinstance(driver, str) or not driver for driver in participants):
            issues.append("INVALID_PARTICIPANT_ID")
        elif len(participants) != len(set(participants)):
            issues.append("DUPLICATE_PARTICIPANT")

        participant_set = set(participants)
        classified_not_selectable = sorted(set(finish_positions) - participant_set)
        if classified_not_selectable:
            issues.append("CLASSIFIED_DRIVER_MISSING_FROM_ROSTER")
        declared_count = payload.get("participantCount")
        if declared_count is not None and declared_count != len(participants):
            issues.append("PARTICIPANT_COUNT_MISMATCH")

        if issues:
            failures.append({
                "file": str(path),
                "year": payload.get("year"),
                "round": payload.get("round"),
                "issues": issues,
                "classifiedNotSelectable": classified_not_selectable,
            })
        else:
            participant_counts[len(participants)] += 1

    report = {
        "status": "PASS" if not failures else "FAIL",
        "expectedDecisionPointSchema": SCHEMA_VERSION,
        "raceFiles": len(files),
        "validRosters": sum(participant_counts.values()),
        "participantCountDistribution": dict(sorted(participant_counts.items())),
        "failures": failures,
        "guarantee": (
            "Every participant recorded in a valid cache roster is selectable by the UI, "
            "even when no retained decision row exists for that driver."
        ),
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
