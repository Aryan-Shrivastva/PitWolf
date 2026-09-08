"""Shared helpers for FIA Overtake/straight-mode zone evidence.

The helpers do not create an FIA rule from a generic close-gap observation.
They only turn a cited event configuration plus FastF1 telemetry into an
auditable, same-track-distance opportunity record.
"""

from __future__ import annotations

from typing import Any


def number(value: Any, default: float | None = None) -> float | None:
    try:
        parsed = float(value)
        return parsed if parsed == parsed else default
    except (TypeError, ValueError):
        return default


def normalize_distance(distance_m: Any, track_length_m: Any) -> float | None:
    """Normalise an FIA lap distance to the real telemetry lap length."""
    distance = number(distance_m)
    length = number(track_length_m)
    if distance is None or length is None or length <= 0:
        return None
    return distance % length


def official_overtake_config(rule_context: Any) -> dict[str, Any] | None:
    """Return a complete live-command config or ``None`` without guessing."""
    context = rule_context if isinstance(rule_context, dict) else {}
    required = (
        'officialDetectionGapS', 'detectionLineDistanceM',
        'activationLineDistanceM', 'rechargeLimitMj', 'powerLimitProfile',
    )
    if context.get('eventSpecificDataLoaded') is not True:
        return None
    if any(context.get(field) is None for field in required):
        return None
    if not isinstance(context.get('powerLimitProfile'), dict):
        return None
    return context


def zone_evidence_config(rule_context: Any) -> dict[str, Any] | None:
    """Return cited line/gap evidence sufficient for retrospective alignment.

    This is deliberately weaker than ``official_overtake_config``: matching a
    recorded car to a cited Detection Line does not require knowing private
    deployment, but it also cannot make a prediction an authorised command.
    """
    context = rule_context if isinstance(rule_context, dict) else {}
    evidence = context.get('zoneEvidence') if isinstance(context.get('zoneEvidence'), dict) else {}
    required = ('officialDetectionGapS', 'detectionLineDistanceM', 'activationLineDistanceM', 'eventAppendixSource')
    if evidence.get('zoneEvidenceLoaded') is not True or any(evidence.get(field) is None for field in required):
        return None
    source = evidence.get('eventAppendixSource')
    if not isinstance(source, dict) or not isinstance(source.get('url'), str):
        return None
    return evidence


def straight_mode_evidence(rule_context: Any) -> dict[str, Any] | None:
    """Return map-reference evidence without turning it into FIA rule data.

    These user-supplied turn ranges help a future map align a Straight Mode
    zone to telemetry corner distances. They never contain a legal detection
    gap or metre coordinate and therefore cannot open a command gate.
    """
    context = rule_context if isinstance(rule_context, dict) else {}
    evidence = context.get('straightModeEvidence')
    if not isinstance(evidence, dict):
        return None
    if evidence.get('coordinateStatus') not in {
        'TURN_RANGE_ONLY_NOT_TRACK_DISTANCE_MAPPED',
        'NO_STRAIGHT_MODE_ZONES',
    }:
        return None
    if not isinstance(evidence.get('zones'), list):
        return None
    return evidence


def zone_descriptor(rule_context: Any, track_length_m: Any) -> dict[str, Any] | None:
    """Build the one FIA Overtake Mode zone descriptor currently supported.

    FIA documents may later define multiple independent Overtake structures.
    Keeping a stable ``OM-1`` identity now lets the extractor extend to a list
    of zones without changing downstream consumer keys.
    """
    # Prefer the complete live-command record when present, otherwise use the
    # narrower zone evidence record for retrospective telemetry alignment.
    config = official_overtake_config(rule_context) or zone_evidence_config(rule_context)
    if config is None:
        return None
    detection = normalize_distance(config.get('detectionLineDistanceM'), track_length_m)
    activation = normalize_distance(config.get('activationLineDistanceM'), track_length_m)
    if detection is None or activation is None:
        return None
    return {
        'zoneId': 'OM-1',
        'type': 'OVERTAKE_MODE',
        'detectionGapS': number(config.get('officialDetectionGapS')),
        'detectionLineDistanceM': round(detection, 1),
        'activationLineDistanceM': round(activation, 1),
        'rechargeLimitMj': number(config.get('rechargeLimitMj')),
        'powerLimitProfile': config.get('powerLimitProfile'),
        'eventAppendixSource': config.get('eventAppendixSource'),
    }


def zone_alignment_status(rule_context: Any, track_length_m: Any) -> dict[str, Any]:
    """Explain availability for API/UI consumers without changing a prediction."""
    zone = zone_descriptor(rule_context, track_length_m)
    straight_mode = straight_mode_evidence(rule_context)
    if zone is None:
        return {
            'status': 'EVENT_APPENDIX_REQUIRED',
            'telemetryAlignmentAvailable': False,
            'reason': 'A complete cited FIA Power Unit Information record is required before zone alignment is computed.',
            'straightModeMapReference': straight_mode,
        }
    fully_authorised = official_overtake_config(rule_context) is not None
    return {
        'status': 'READY_TO_EXTRACT' if fully_authorised else 'RETROSPECTIVE_ZONE_EVIDENCE_ONLY',
        'telemetryAlignmentAvailable': True,
        'liveCommandEligible': fully_authorised,
        'zone': zone,
        'method': 'SAME_TRACK_DISTANCE_INTERPOLATION',
        'straightModeMapReference': straight_mode,
    }
