"""Audit cached decision-point labels without fetching any external data.

This script deliberately audits only local decision-point caches.  It does not
claim that the cache contains every scheduled race: schedule completeness must
be established by the extractor when source data is available.  Its purpose is
to prevent a training run from silently mixing label policies.
"""

import argparse
import collections
import datetime as dt
import json
import pathlib

from extract_decision_points import OUT_ROOT, SCHEMA_VERSION
from fetch_f1_session import CACHE_DIR


DEFAULT_OUTPUT = pathlib.Path(CACHE_DIR).parent / 'models' / 'decision-label-audit.v1.json'


def counter_dict(counter):
    return dict(sorted(counter.items()))


def load_payload(path):
    try:
        return json.loads(path.read_text(encoding='utf-8')), None
    except (OSError, json.JSONDecodeError) as error:
        return None, str(error)


def audit_year(year):
    root = OUT_ROOT / str(year)
    files = sorted(root.glob('*_r.json')) if root.exists() else []
    summary = {
        'cachedRaceFiles': len(files),
        'currentSchemaFiles': 0,
        'staleSchemaFiles': [],
        'invalidFiles': [],
        'rows': 0,
        'eligibleRows': 0,
        'labelCounts': collections.Counter(),
        'labelEvidenceCounts': collections.Counter(),
        'exclusionReasonCounts': collections.Counter(),
    }

    for path in files:
        payload, error = load_payload(path)
        if error:
            summary['invalidFiles'].append({'file': path.name, 'error': error})
            continue
        if payload.get('schemaVersion') != SCHEMA_VERSION:
            summary['staleSchemaFiles'].append({
                'file': path.name,
                'schemaVersion': payload.get('schemaVersion'),
            })
            continue

        summary['currentSchemaFiles'] += 1
        rows = payload.get('rows') or []
        summary['rows'] += len(rows)
        for row in rows:
            if row.get('eligibleForTraining'):
                summary['eligibleRows'] += 1
                label = row.get('label')
                if label:
                    summary['labelCounts'][label] += 1
                evidence = row.get('labelEvidence')
                if evidence:
                    summary['labelEvidenceCounts'][evidence] += 1
        for excluded in payload.get('excludedRows') or []:
            for reason in excluded.get('exclusionReasons') or ['UNSPECIFIED']:
                summary['exclusionReasonCounts'][reason] += 1

    for key in ('labelCounts', 'labelEvidenceCounts', 'exclusionReasonCounts'):
        summary[key] = counter_dict(summary[key])
    return summary


def build_report(years):
    by_year = {str(year): audit_year(year) for year in years}
    stale = sum(len(item['staleSchemaFiles']) for item in by_year.values())
    invalid = sum(len(item['invalidFiles']) for item in by_year.values())
    current = sum(item['currentSchemaFiles'] for item in by_year.values())
    cached = sum(item['cachedRaceFiles'] for item in by_year.values())
    return {
        'schemaVersion': 'decision-label-audit.v1',
        'createdAtUtc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'expectedDecisionPointSchema': SCHEMA_VERSION,
        'years': list(years),
        'scopeDisclosure': (
            'Local cache audit only. It does not prove schedule completeness; '
            'a missing source race is not represented as a cache file.'
        ),
        'readyForTraining': cached > 0 and stale == 0 and invalid == 0,
        'summary': {
            'cachedRaceFiles': cached,
            'currentSchemaFiles': current,
            'staleSchemaFiles': stale,
            'invalidFiles': invalid,
        },
        'byYear': by_year,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-year', type=int, default=2018)
    parser.add_argument('--end-year', type=int, default=2026)
    parser.add_argument('--output', type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.end_year < args.start_year:
        parser.error('--end-year must be at least --start-year')

    report = build_report(range(args.start_year, args.end_year + 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({
        'output': str(args.output),
        'readyForTraining': report['readyForTraining'],
        **report['summary'],
    }))


if __name__ == '__main__':
    main()
