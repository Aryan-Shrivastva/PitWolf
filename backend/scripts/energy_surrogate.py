"""Cheap, race-level battery-state surrogate for overtake decisions.

FastF1 exposes the inputs needed to estimate energy use, but not the team's
actual battery state or deployment map.  This module deliberately uses only
lap-level battle features, so it is cheap enough to attach to every historical
decision point and honest enough to be labelled MODELLED.
"""

import numpy as np
import pandas as pd

try:
    from energy_transition import era_for_year, observed_action, transition_soc
except ImportError:  # allow package-style unit tests as well as direct scripts
    from .energy_transition import era_for_year, observed_action, transition_soc


SOC_CAPACITY_MJ = 4.0
SOC_START_MJ = 2.8


def _clip(value, low=0.0, high=SOC_CAPACITY_MJ):
    return float(np.clip(value, low, high))


def add_surrogate_energy(df):
    """Add attackerSoCMj, defenderSoCMj and energyDeltaMj to a row frame.

    State is carried once per race/lap/driver.  The update uses pace pressure,
    closing rate, tyre age differential and pit context, none of which are
    outcome labels.  Repeated battle rows on the same lap therefore see the
    same pre-lap state rather than draining the battery multiple times.

    The pace reference is an expanding, past-only median.  Using the median of
    the entire race here would let a training feature see future lap times.
    A pit stop changes tyres and lap-time context; it is never treated as a
    battery recharge because this surrogate models on-track SoC continuity.
    """
    if df.empty:
        return df
    df = df.copy()
    for column in ('attackerSoCMj', 'defenderSoCMj', 'energyDeltaMj'):
        df[column] = np.nan

    def numeric_column(frame, column):
        return pd.to_numeric(frame[column], errors='coerce') if column in frame else pd.Series(dtype=float)

    def bounded_mean(frame, column, scale, default):
        values = numeric_column(frame, column).dropna().abs()
        if values.empty:
            return default
        return float(np.clip(float(values.mean()) / scale, 0.0, 1.0))

    race_columns = [column for column in ('year', 'round', 'session') if column in df]
    for race_key, race in df.groupby(race_columns, sort=False):
        race_year = race_key[0] if isinstance(race_key, tuple) else race_key
        race_era = era_for_year(race_year)
        states = {}
        past_lap_times = []
        for lap, lap_rows in race.groupby('lap', sort=True):
            drivers = set(lap_rows['driver'].dropna()) | set(lap_rows['defender'].dropna())
            before = {driver: states.get(driver, SOC_START_MJ) for driver in drivers}
            for index, row in lap_rows.iterrows():
                attacker = row.get('driver')
                defender = row.get('defender')
                attacker_soc = before.get(attacker, states.get(attacker, SOC_START_MJ))
                defender_soc = before.get(defender, states.get(defender, SOC_START_MJ))
                df.at[index, 'attackerSoCMj'] = round(attacker_soc, 3)
                df.at[index, 'defenderSoCMj'] = round(defender_soc, 3)
                df.at[index, 'energyDeltaMj'] = round(attacker_soc - defender_soc, 3)

            for driver in drivers:
                driver_rows = lap_rows[(lap_rows['driver'] == driver) | (lap_rows['defender'] == driver)]
                pressure = bounded_mean(driver_rows, 'speedDeltaKph', 30.0, 0.35)
                closing = bounded_mean(driver_rows, 'closingRateS', 3.0, 0.25)
                tyre_load = min(0.4, bounded_mean(driver_rows, 'tyreAgeDiff', 20.0, 0.0))
                lap_time = pd.concat([
                    numeric_column(driver_rows, 'attackerLapTimeS'),
                    numeric_column(driver_rows, 'defenderLapTimeS'),
                ]).dropna()
                # At a lap-start decision point, only completed earlier laps
                # are allowed to establish the race pace reference.
                reference_lap_time = float(np.median(past_lap_times)) if past_lap_times else 95.0
                pace = 0.5
                if not lap_time.empty:
                    pace = float(np.clip(0.5 + ((reference_lap_time - float(lap_time.mean())) / 10.0), 0.1, 0.9))
                surrogate_row = {
                    'gapS': float(numeric_column(driver_rows, 'gapS').mean()) if not numeric_column(driver_rows, 'gapS').dropna().empty else 1.2,
                    'closingRateS': closing * 3.0,
                    'tyreAgeDiff': tyre_load * 20.0,
                    'pace': pace,
                }
                action = observed_action(surrogate_row)
                next_soc, _, _ = transition_soc(
                    before.get(driver, states.get(driver, SOC_START_MJ)),
                    action,
                    surrogate_row,
                    defending=False,
                    era=race_era,
                )
                states[driver] = next_soc

            # Make this lap available only to later decision points.  This is
            # deliberately after all feature calculations for the lap.
            lap_times = pd.concat([
                numeric_column(lap_rows, 'attackerLapTimeS'),
                numeric_column(lap_rows, 'defenderLapTimeS'),
            ]).dropna()
            past_lap_times.extend(lap_times.tolist())

    return df
