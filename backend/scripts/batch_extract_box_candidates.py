"""Cache-first backfill for the independent BOX candidate dataset."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

from extract_box_candidates import OUT_ROOT, build_candidates


ROOT = OUT_ROOT.parent


def available_races() -> list[tuple[int, int]]:
    races = []
    for path in sorted((ROOT / 'decision-points').glob('*/*_r.json')):
        try:
            year = int(path.parent.name)
            round_number = int(path.name.split('_', 1)[0])
        except ValueError:
            continue
        races.append((year, round_number))
    return races


def extract_one(item: tuple[int, int]) -> dict:
    year, round_number = item
    payload = build_candidates(year, round_number, 'R')
    target = OUT_ROOT / str(year) / f'{round_number}_r.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    return {'year': year, 'round': round_number, 'candidateCount': payload['candidateCount']}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--year', type=int)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    races = available_races()
    if args.year is not None:
        races = [item for item in races if item[0] == args.year]
    if not args.force:
        races = [item for item in races if not (OUT_ROOT / str(item[0]) / f'{item[1]}_r.json').exists()]
    completed, failures = [], []
    if args.workers <= 1:
        # Sequential cache reads are reliable in restricted Windows sessions
        # where child-process pipes may be unavailable. It is also gentler on
        # the shared FastF1 HTTP cache.
        for item in races:
            try:
                completed.append(extract_one(item))
            except Exception as error:
                failures.append({'year': item[0], 'round': item[1], 'error': str(error)})
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(extract_one, item): item for item in races}
            for future in concurrent.futures.as_completed(futures):
                item = futures[future]
                try:
                    completed.append(future.result())
                except Exception as error:
                    failures.append({'year': item[0], 'round': item[1], 'error': str(error)})
    print(json.dumps({
        'requested': len(races), 'completed': len(completed), 'failures': failures,
        'candidateCount': sum(item['candidateCount'] for item in completed),
    }))
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
