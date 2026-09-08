"""Audit observed driver-position summaries in cached race payloads.

This intentionally validates structure and arithmetic only. A lap
classification change is evidence of movement, not proof of an on-track pass.
"""

import json
from pathlib import Path

from extract_decision_points import OUT_ROOT


def main():
    files = sorted(OUT_ROOT.glob('*/*_r.json'))
    failures = []
    summary_count = 0
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f'{path}: unreadable ({error})')
            continue
        participants = set(payload.get('participants') or [])
        summaries = payload.get('driverSummaries')
        if not isinstance(summaries, list):
            failures.append(f'{path}: missing driverSummaries')
            continue
        by_driver = {item.get('driver'): item for item in summaries if isinstance(item, dict)}
        if set(by_driver) != participants or len(by_driver) != len(summaries):
            failures.append(f'{path}: participant/summary roster mismatch')
            continue
        for driver, item in by_driver.items():
            grid, finish, net = item.get('gridPosition'), item.get('finalPosition'), item.get('netPlacesGained')
            if grid is not None and finish is not None and net != grid - finish:
                failures.append(f'{path}: {driver} net position arithmetic is inconsistent')
            for direction, sign in (('positionGainEvents', -1), ('positionLossEvents', 1)):
                for event in item.get(direction) or []:
                    before, after, places = event.get('fromPosition'), event.get('toPosition'), event.get('places')
                    if not all(isinstance(value, int) for value in (before, after, places)):
                        failures.append(f'{path}: {driver} malformed {direction}')
                        continue
                    if before == after or abs(before - after) != places or (after - before) * sign <= 0:
                        failures.append(f'{path}: {driver} inconsistent {direction} movement')
                    if event.get('source') != 'LAP_CLASSIFICATION':
                        failures.append(f'{path}: {driver} movement missing LAP_CLASSIFICATION source')
        summary_count += len(summaries)
    print(json.dumps({
        'status': 'PASS' if not failures else 'FAIL',
        'raceFiles': len(files),
        'driverSummaries': summary_count,
        'failures': failures,
    }, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
