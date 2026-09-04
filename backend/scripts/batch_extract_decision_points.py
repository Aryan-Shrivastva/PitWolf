"""Batch decision-point extraction across seasons.

Single-threaded and paced (F1 API budget: 500 calls/hour), resumable via the
JSON files under data/f1-cache/decision-points/, and yields to on-demand
requests through the .ondemand.lock used by the API server.
"""

import argparse
import json
import pathlib
import time

import fastf1

from extract_decision_points import OUT_ROOT, extract_race
from fetch_f1_session import CACHE_DIR

CACHE_ROOT = CACHE_DIR.parent


def is_rate_limited(error):
    text = str(error)
    return 'calls/h' in text or '429' in text or 'has not been loaded yet' in text


def wait_for_ondemand():
    lock = CACHE_ROOT / '.ondemand.lock'
    while lock.exists():
        try:
            if time.time() - lock.stat().st_mtime > 600:
                lock.unlink()
                break
        except OSError:
            break
        time.sleep(5)


def run_task(year, round_number, session_name, max_gap, hold_laps, attempts=5):
    target = OUT_ROOT / str(year) / f'{round_number}_{session_name.lower()}.json'
    if target.exists():
        try:
            cached = json.loads(target.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            cached = None
        if isinstance(cached, dict) and cached.get('schemaVersion') == 'decision-point.v5':
            return 'skip'
    delay = 5
    for attempt in range(attempts):
        wait_for_ondemand()
        try:
            payload = extract_race(year, round_number, session_name, max_gap, hold_laps)
            if not payload.get('rows'):
                raise RuntimeError('no decision points extracted')
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + '.tmp')
            tmp.write_text(json.dumps(payload), encoding='utf-8')
            tmp.replace(target)
            time.sleep(3)
            return 'ok'
        except Exception as error:
            if attempt == attempts - 1:
                print(f'FAIL {year} R{round_number} {session_name}: {error}', flush=True)
                return 'fail'
            if is_rate_limited(error):
                # A rate-limit window is shared by the FastF1 process. Waiting
                # and repeating the same request five times only blocks the
                # whole batch; leave this race resumable instead.
                print(f'RATE-LIMITED {year} R{round_number}, skipping until the next window', flush=True)
                return 'fail'
            else:
                time.sleep(delay)
                delay *= 2
    return 'fail'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-year', type=int, default=2018)
    parser.add_argument('--end-year', type=int, default=2025)
    parser.add_argument('--session', default='R')
    parser.add_argument('--max-gap', type=float, default=1.2)
    parser.add_argument('--hold-laps', type=int, default=6)
    args = parser.parse_args()

    fastf1.set_log_level('ERROR')
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    tasks = []
    for year in range(args.start_year, args.end_year + 1):
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as error:
            print(f'SCHEDULE FAIL {year}: {error}', flush=True)
            continue
        for _, row in schedule.iterrows():
            tasks.append((year, int(row['RoundNumber'])))

    counts = {'ok': 0, 'skip': 0, 'fail': 0}
    started = time.time()
    for index, (year, round_number) in enumerate(tasks, 1):
        status = run_task(year, round_number, args.session, args.max_gap, args.hold_laps)
        counts[status] += 1
        if status != 'skip':
            print(f'[{index}/{len(tasks)}] {status} {year} R{round_number} ({time.time() - started:.0f}s)', flush=True)
    print(f'SUMMARY {args.start_year}-{args.end_year} ok={counts["ok"]} skip={counts["skip"]} fail={counts["fail"]}', flush=True)


if __name__ == '__main__':
    main()
