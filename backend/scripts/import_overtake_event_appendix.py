"""Safely import one FIA event Overtake Mode appendix into PitWolf's registry.

This script never downloads or guesses values. An operator transcribes values
from the official FIA Competition Notes / Race Director document into a small
JSON file, then this importer validates its provenance before atomically
adding it to ``data/overtake-rule-context.json``.
"""

import argparse
import json
import re
from pathlib import Path

from fetch_f1_session import CACHE_DIR

ROOT = CACHE_DIR.parent
DEFAULT_REGISTRY = ROOT / 'overtake-rule-context.json'
EVENT_KEY = re.compile(r'^2026:\d{1,2}:r$')
REQUIRED_NUMBERS = (
    'officialDetectionGapS', 'detectionLineDistanceM',
    'activationLineDistanceM', 'rechargeLimitMj',
)


def finite_non_negative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def validate(entry):
    errors = []
    event_key = entry.get('eventKey')
    if not isinstance(event_key, str) or not EVENT_KEY.fullmatch(event_key):
        errors.append('eventKey must be a race key like 2026:3:r')
    if entry.get('eventSpecificDataLoaded') is not True:
        errors.append('eventSpecificDataLoaded must be true')
    for field in REQUIRED_NUMBERS:
        if not finite_non_negative(entry.get(field)):
            errors.append(f'{field} must be a non-negative number')
    if not isinstance(entry.get('powerLimitProfile'), dict) or not entry['powerLimitProfile']:
        errors.append('powerLimitProfile must be a non-empty object copied from the FIA event document')
    source = entry.get('eventAppendixSource')
    if not isinstance(source, dict):
        errors.append('eventAppendixSource is required')
    else:
        url = source.get('url')
        if not isinstance(url, str) or not url.startswith('https://www.fia.com/'):
            errors.append('eventAppendixSource.url must be an official https://www.fia.com/ URL')
        if not isinstance(source.get('published'), str) or not source['published']:
            errors.append('eventAppendixSource.published is required')
        if not isinstance(source.get('documentTitle'), str) or not source['documentTitle']:
            errors.append('eventAppendixSource.documentTitle is required')
        if not isinstance(source.get('articleOrSection'), str) or not source['articleOrSection']:
            errors.append('eventAppendixSource.articleOrSection is required')
    if entry.get('analysisWindowGapS') is not None:
        errors.append('analysisWindowGapS is a model filter, not an FIA appendix value')
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='verified appendix JSON record')
    parser.add_argument('--registry', default=str(DEFAULT_REGISTRY), help='rule registry JSON path')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    record_path = Path(args.input)
    registry_path = Path(args.registry)
    entry = json.loads(record_path.read_text(encoding='utf-8'))
    errors = validate(entry)
    result = {
        'eventKey': entry.get('eventKey'),
        'valid': not errors,
        'errors': errors,
        'written': False,
    }
    if errors or args.dry_run:
        print(json.dumps(result, indent=2))
        raise SystemExit(1 if errors else 0)

    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    registry.setdefault('events', {})[entry['eventKey']] = {
        key: value for key, value in entry.items() if key != 'eventKey'
    }
    temporary = registry_path.with_name(registry_path.name + '.tmp')
    temporary.write_text(json.dumps(registry, indent=2) + '\n', encoding='utf-8')
    temporary.replace(registry_path)
    result['written'] = True
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
