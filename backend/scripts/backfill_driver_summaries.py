"""Add observed driver-position summaries to existing decision-point caches.

This deliberately loads only cached lap and result timing. It does not rebuild
features, labels, energy traces, or model artefacts, so the clean training
split remains untouched.
"""

import argparse
import json
import math
from pathlib import Path

import fastf1

from extract_decision_points import OUT_ROOT, build_timelines, driver_race_summaries, num
from fetch_f1_session import CACHE_DIR
from overtake_feature_schema import FEATURE_SCHEMA_VERSION


def json_safe(value):
    """Replace legacy non-finite cache values before atomic re-serialization."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def result_context(session):
    participants, grid_positions, finish_positions, result_status = [], {}, {}, {}
    for _, result in session.results.iterrows():
        abbreviation = result.get('Abbreviation')
        if abbreviation is None:
            continue
        driver = str(abbreviation)
        participants.append(driver)
        grid = num(result.get('GridPosition'))
        finish = num(result.get('Position'))
        if grid is not None:
            grid_positions[driver] = int(grid)
        if finish is not None:
            finish_positions[driver] = int(finish)
        if result.get('Status') is not None:
            result_status[driver] = str(result.get('Status'))
    return participants, grid_positions, finish_positions, result_status


def backfill(path, offline):
    payload = json.loads(path.read_text(encoding='utf-8'))
    needs_summaries = not isinstance(payload.get('driverSummaries'), list)
    needs_feature_schema = payload.get('featureSchemaVersion') != FEATURE_SCHEMA_VERSION
    if not needs_summaries and not needs_feature_schema:
        return 'skip'
    if needs_summaries:
        year = int(payload['year'])
        round_number = int(payload['round'])
        session_name = payload.get('session', 'R')
        event = fastf1.get_event(year, round_number)
        session = event.get_session(session_name)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        participants, grid_positions, finish_positions, result_status = result_context(session)
        payload['participants'] = sorted(set(participants))
        payload['participantCount'] = len(payload['participants'])
        payload['finishPositions'] = finish_positions
        payload['driverSummaries'] = driver_race_summaries(
            participants, build_timelines(session), grid_positions, finish_positions, result_status)
    payload['featureSchemaVersion'] = FEATURE_SCHEMA_VERSION
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(json_safe(payload), allow_nan=False), encoding='utf-8')
    temporary.replace(path)
    return 'ok'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-year', type=int, default=2018)
    parser.add_argument('--end-year', type=int, default=2026)
    parser.add_argument('--offline', action='store_true', help='use only local FastF1 cache')
    args = parser.parse_args()

    fastf1.set_log_level('ERROR')
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    if args.offline:
        fastf1.Cache.offline_mode(True)

    counts = {'ok': 0, 'skip': 0, 'fail': 0}
    for path in sorted(OUT_ROOT.glob('*/*_r.json')):
        try:
            year = int(path.parent.name)
            if not args.start_year <= year <= args.end_year:
                continue
            status = backfill(path, args.offline)
            counts[status] += 1
            if status == 'ok':
                print(f'UPDATED {path.relative_to(OUT_ROOT)}', flush=True)
        except Exception as error:
            counts['fail'] += 1
            print(f'FAIL {path}: {error}', flush=True)
    print(json.dumps(counts), flush=True)
    raise SystemExit(1 if counts['fail'] else 0)


if __name__ == '__main__':
    main()
