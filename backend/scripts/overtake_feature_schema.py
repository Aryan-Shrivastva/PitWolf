"""Canonical, versioned feature contract for the overtake action model."""

FEATURE_SCHEMA_VERSION = 'overtake-features.v1'

FEATURES = [
    'gapS', 'closingRateS', 'speedDeltaKph', 'tyreAgeDiff', 'lapFraction',
    'position', 'raceMeanSpeedKph', 'attackerCompoundOrd', 'defenderCompoundOrd',
    'drsEligible', 'attackerMassKg', 'massDeltaKg', 'attackerSoCMj',
    'defenderSoCMj', 'energyDeltaMj', 'trafficAheadCount',
    'trafficBehindCount', 'packDensity', 'slipstreamProxy', 'dirtyAirRisk',
    'attackerTyreDegProxy', 'defenderTyreDegProxy',
]

WEATHER_FEATURES = [
    'airTempC', 'trackTempC', 'humidityPct', 'windSpeedMps', 'rainfall',
    'weatherMissing',
]

COMPOUND_ORDINAL = {
    'SOFT': 0,
    'MEDIUM': 1,
    'HARD': 2,
    'INTERMEDIATE': 3,
    'WET': 4,
}
