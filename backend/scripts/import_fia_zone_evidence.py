"""Import cited FIA detection/activation-line evidence for retrospective replay.

Unlike the full event-appendix importer, this accepts only the information
needed to align telemetry at an official Overtake Detection Line. It does not
set ``eventSpecificDataLoaded`` and can therefore never enable a live command.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from fetch_f1_session import CACHE_DIR


ROOT = Path(CACHE_DIR).parent
DEFAULT_REGISTRY = ROOT / 'overtake-rule-context.json'
EVENT_KEY = re.compile(r'^2026:\d{1,2}:r$')


def finite_non_negative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def validate(entry):
    errors = []
    if not isinstance(entry.get('eventKey'), str) or not EVENT_KEY.fullmatch(entry['eventKey']):
        errors.append('eventKey must be a race key like 2026:3:r')
    if entry.get('zoneEvidenceLoaded') is not True:
        errors.append('zoneEvidenceLoaded must be true')
    for field in ('officialDetectionGapS', 'detectionLineDistanceM', 'activationLineDistanceM'):
        if not finite_non_negative(entry.get(field)):
            errors.append(f'{field} must be a non-negative number')
    source = entry.get('eventAppendixSource')
    if not isinstance(source, dict):
        errors.append('eventAppendixSource is required')
    else:
        if not isinstance(source.get('url'), str) or not source['url'].startswith('https://www.fia.com/'):
            errors.append('eventAppendixSource.url must be an official FIA URL')
        for field in ('published', 'documentTitle', 'articleOrSection'):
            if not isinstance(source.get(field), str) or not source[field]:
                errors.append(f'eventAppendixSource.{field} is required')
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--registry', default=str(DEFAULT_REGISTRY))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    record = json.loads(Path(args.input).read_text(encoding='utf-8'))
    errors = validate(record)
    result = {'eventKey': record.get('eventKey'), 'valid': not errors, 'errors': errors, 'written': False}
    if errors or args.dry_run:
        print(json.dumps(result, indent=2))
        raise SystemExit(1 if errors else 0)
    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    event = registry.setdefault('events', {}).setdefault(record['eventKey'], {})
    event['zoneEvidence'] = {key: value for key, value in record.items() if key != 'eventKey'}
    # This importer must never create a live command state.
    event.setdefault('eventSpecificDataLoaded', False)
    temporary = registry_path.with_name(registry_path.name + '.tmp')
    temporary.write_text(json.dumps(registry, indent=2) + '\n', encoding='utf-8')
    temporary.replace(registry_path)
    result['written'] = True
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
