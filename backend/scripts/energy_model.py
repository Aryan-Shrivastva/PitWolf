"""Physics energy model for PitWolf.

Derives instantaneous power demand from real FastF1 telemetry (P = F*v with
F = m*a + drag + rolling resistance), then splits demand between ICE and
electrical power using the 2026 FIA regulation ceilings (fia_2026_regs.py) and
integrates battery state of charge within the 4 MJ ES window.

Every constant that is not from the regulations is listed in ASSUMPTIONS and
must be surfaced in the UI as MODELLED.
"""

import numpy as np

import fia_2026_regs as regs
from fia_compliance import event_context as fia_event_context, propulsion_ceiling_kw, recharge_ceiling_mj, regulation_payload, session_kind

G = 9.81

# Non-regulation physics constants. All of these are assumptions and are
# surfaced in API output so the frontend can label them MODELLED.
ASSUMPTIONS = {
    'dragAreaCda': {'value': 1.30, 'unit': 'm^2', 'note': 'Cd*A in race trim, low-downforce ~ medium'},
    'rollingResistance': {'value': 0.008, 'unit': '-', 'note': 'Crr for racing slicks on smooth tarmac'},
    'drivetrainEfficiency': {'value': 0.95, 'unit': '-', 'note': 'wheel power / engine+ERS power when propelling'},
    'iceThermalEfficiency': {'value': 0.48, 'unit': '-', 'note': 'fuel energy -> mechanical, modern F1 ICE'},
    'regenEfficiency': {'value': 0.85, 'unit': '-', 'note': 'wheel braking power -> harvested DC energy'},
    'dischargeEfficiency': {'value': 0.92, 'unit': '-', 'note': 'ES DC energy -> ERS-K output'},
    'chargeEfficiency': {'value': 0.88, 'unit': '-', 'note': 'harvested DC energy -> stored ES energy'},
    'airDensityFallback': {'value': 1.15, 'unit': 'kg/m^3', 'note': 'used when no session weather data'},
    'auxiliaryLoadKw': {'value': 30.0, 'unit': 'kW', 'note': 'cooling/hydraulics/electronics carried by the ICE'},
    'fuelStartMassKg': {
        'value': None, 'unit': 'kg',
        'note': 'year-dependent: 2026 ~70 kg (reg target), 2018-2025 110 kg allowance',
    },
}


def fuel_start_mass_kg(year):
    entry = regs.MAX_FUEL_START_KG.get(year) or regs.MAX_FUEL_START_KG['default']
    return entry['value']


def air_density_from_weather(weather):
    """Ideal-gas density with humidity correction; returns None if unusable."""
    if weather is None:
        return None
    try:
        temp_c = float(weather.get('airTempC'))
        pressure_mbar = float(weather.get('pressureMbar'))
    except (TypeError, ValueError):
        return None
    if not (0.0 < temp_c < 50.0 and 800.0 < pressure_mbar < 1100.0):
        return None
    temp_k = temp_c + 273.15
    humidity = float(weather.get('humidityPct') or 0.0) / 100.0
    sat_pressure_pa = 610.94 * np.exp(17.625 * temp_c / (temp_c + 243.04))
    vapor_pa = humidity * sat_pressure_pa
    dry_pa = pressure_mbar * 100.0 - vapor_pa
    return dry_pa / (287.058 * temp_k) + vapor_pa / (461.495 * temp_k)


def _smooth(values, window_samples):
    window_samples = max(1, int(window_samples))
    kernel = np.ones(window_samples) / window_samples
    return np.convolve(values, kernel, mode='same')


def _qualifying_ers_inference(throttle, brake, acceleration, wheel_power_w, sample_hz):
    """Create a public-telemetry deployment *likelihood*, not an ERS measurement.

    Qualifying has no public ERS-K power, battery SoC, fuel-flow or driver-mode
    channel.  The most defensible signal available is therefore the observed
    propulsive demand: high throttle, positive longitudinal acceleration and
    high wheel-power demand.  The returned multiplier lets the legal energy
    controller prioritise those locations while its SoC, power and recharge
    constraints remain authoritative.
    """
    positive_accel = np.maximum(0.0, acceleration)
    accel_reference = max(0.35, float(np.percentile(positive_accel, 85)))
    demand_kw = np.maximum(0.0, wheel_power_w) / 1000.0
    demand_reference = max(1.0, float(np.percentile(demand_kw, 85)))

    full_throttle = np.clip((throttle - 0.72) / 0.28, 0.0, 1.0)
    # A short smoothing window prevents isolated telemetry samples from
    # creating artificial full-deployment islands on the circuit map.
    sustained_throttle = _smooth(full_throttle, max(3, round(sample_hz * 0.18)))
    accel_score = np.clip(positive_accel / accel_reference, 0.0, 1.0)
    demand_score = np.clip(demand_kw / demand_reference, 0.0, 1.0)
    score = np.clip(
        0.50 * sustained_throttle + 0.30 * accel_score + 0.20 * demand_score,
        0.0, 1.0,
    )
    propulsive = (~brake) & (throttle >= 0.55) & (wheel_power_w > 0.0)
    score = np.where(propulsive, score, 0.0)
    # Low-score powered sections can still receive ERS after the higher-value
    # segments; they simply do not receive the same inferred priority.
    multiplier = np.where(propulsive, 0.20 + 1.80 * score, 0.0)
    return score, multiplier


def _in_distance_windows(distance_m, windows):
    """Return whether a point is inside a supplied start/end distance range.

    A range whose end precedes its start crosses the timing line.  These
    windows are a PitWolf scenario policy from user-supplied circuit maps,
    rather than FIA Overtake activation data.
    """
    for window in windows:
        try:
            start = float(window['startDistanceM'])
            end = float(window['endDistanceM'])
        except (KeyError, TypeError, ValueError):
            continue
        if end >= start and start <= distance_m <= end:
            return True
        if end < start and (distance_m >= start or distance_m <= end):
            return True
    return False


def _straight_mode_window_index(distance_m, windows):
    """Return the configured Straight Mode window containing this point."""
    for index, window in enumerate(windows):
        if _in_distance_windows(distance_m, [window]):
            return index
    return -1


def compute_lap_energy(trace, *, year, round_number=None, session_name=None, lap_fraction=0.5, weather=None,
                       soc_start_mj=None, override_windows=None, high_speed_kph=140.0,
                       compliance_context=None, qualifying_battery_calibration=None,
                       deployment_scale=1.0, soc_reserve_taper=True,
                       qualifying_ers_inference=True,
                       qualifying_straight_mode_policy=None):
    """Compute the energy trace for one lap.

    trace: dict with arrays 'time' (s), 'speed' (kph), 'throttle' (0-100),
           'brake' (bool), 'rpm'.
    lap_fraction: fraction of the race already completed (0-1), used only to
        estimate fuel mass on board.
    override_windows: list of [start_m, end_m] distance ranges where Override
        Mode is assumed active (per-event lines are unpublished; when None the
        envelope is evaluated in normal mode and flagged as an assumption).
    qualifying_straight_mode_policy: an explicit PitWolf user policy. When
        supplied for qualifying, ERS deployment is enabled only in its mapped
        circuit windows. Recovery under braking is not a deployment action and
        remains available outside those ranges.
    Returns a dict with per-sample arrays (aligned to the input sampling) and
    lap-level totals.
    """
    time = np.asarray(trace['time'], dtype=float)
    speed_kph = np.asarray(trace['speed'], dtype=float)
    throttle = np.asarray(trace['throttle'], dtype=float) / 100.0
    brake = np.asarray(trace['brake'], dtype=bool)
    n = len(time)
    rpm_source = trace.get('rpm')
    rpm = np.asarray(rpm_source if rpm_source is not None else np.full(n, 10500.0), dtype=float)
    distance_source = trace.get('distance')
    distance = np.asarray(distance_source if distance_source is not None else np.zeros(n), dtype=float)
    if n < 5:
        raise ValueError('telemetry trace too short')

    dt = np.diff(time, prepend=time[0] - 0.02)
    dt = np.clip(dt, 0.01, 0.5)

    v = speed_kph / 3.6
    sample_hz = 1.0 / float(np.median(dt[1:]) if n > 1 else 0.02)
    v_smooth = _smooth(v, max(3, round(sample_hz * 0.25)))
    # Strictly increasing coordinates guard against duplicate telemetry stamps.
    t_coord = np.maximum.accumulate(time) + np.arange(n) * 1e-4
    a = np.gradient(v_smooth, t_coord)

    context = compliance_context or fia_event_context(year, round_number, session_name)
    qualifying = session_kind(session_name) == 'QUALIFYING'
    fuel_kg = 0.0 if qualifying else fuel_start_mass_kg(year) * max(0.0, 1.0 - lap_fraction)
    minimum_mass = (regs.CONSTANTS['minimum_mass_qualifying_kg']['value'] if qualifying
                    else regs.CONSTANTS['minimum_mass_race_kg']['value'])
    # C4 defines Car Mass as the car including tyres, Driver and Driver
    # Ballast, without fuel.  The 726/724 kg C4.1 figure therefore must not
    # have the 82 kg driver reference mass added a second time.
    mass = minimum_mass + regs.MINIMUM_TYRE_MASS_KG['value'] + fuel_kg

    rho = air_density_from_weather(weather)
    rho_assumed = rho is None
    if rho is None:
        rho = ASSUMPTIONS['airDensityFallback']['value']

    cda = ASSUMPTIONS['dragAreaCda']['value']
    crr = ASSUMPTIONS['rollingResistance']['value']
    eta_drive = ASSUMPTIONS['drivetrainEfficiency']['value']
    eta_ice = ASSUMPTIONS['iceThermalEfficiency']['value']
    eta_regen = ASSUMPTIONS['regenEfficiency']['value']
    eta_out = ASSUMPTIONS['dischargeEfficiency']['value']
    eta_in = ASSUMPTIONS['chargeEfficiency']['value']
    standard_ecu_eta = regs.CONSTANTS['standard_ecu_efficiency_correction']['value']

    p_wheel = (mass * a + 0.5 * rho * cda * v ** 2 + crr * mass * G) * v  # W

    if qualifying and qualifying_ers_inference:
        ers_inference_score, qualifying_deployment_multiplier = _qualifying_ers_inference(
            throttle, brake, a, p_wheel, sample_hz,
        )
    else:
        ers_inference_score = np.zeros(n)
        qualifying_deployment_multiplier = np.ones(n)

    ers_cap_kw = regs.CONSTANTS['ers_k_dc_power_max_kw']['value']
    soc_window = regs.CONSTANTS['es_soc_window_mj']['value']
    harvest_cap_mj = recharge_ceiling_mj(context)

    # A real team ES capacity and starting SoC are not public. For every
    # qualifying reference lap, PitWolf therefore uses the FIA-permitted 4 MJ
    # on-track SoC window as a *normalised usable window*: 100% at the line and
    # 0% at the end. Recovery is still calculated only from braking and may
    # never be forced to refill the store. A small bounded search finds the
    # deployment intensity that consumes the available initial + recovered
    # energy by lap end without violating any instantaneous cap.
    # Ten kilojoules is a numerical end-of-lap tolerance (0.25% of the FIA
    # usable window), not hidden battery capacity.
    qualifying_terminal_tolerance_mj = 0.01
    if qualifying_battery_calibration is None:
        qualifying_battery_calibration = qualifying
    if qualifying_battery_calibration:
        low, high = 0.0, 1.0
        trial = compute_lap_energy(
            trace, year=year, round_number=round_number, session_name=session_name,
            lap_fraction=lap_fraction, weather=weather, soc_start_mj=soc_window,
            override_windows=override_windows, high_speed_kph=high_speed_kph,
            compliance_context=context, qualifying_battery_calibration=False,
            deployment_scale=high, soc_reserve_taper=False,
            qualifying_ers_inference=qualifying_ers_inference,
            qualifying_straight_mode_policy=qualifying_straight_mode_policy,
        )
        # Increase only until the lap can consume the whole usable window or
        # all deployment points have saturated. This is a modelled calibration,
        # not an attempt to reconstruct hidden team control commands.
        while trial['summary']['socEndMj'] > qualifying_terminal_tolerance_mj and high < 32.0:
            low, high = high, high * 2.0
            trial = compute_lap_energy(
                trace, year=year, round_number=round_number, session_name=session_name,
                lap_fraction=lap_fraction, weather=weather, soc_start_mj=soc_window,
                override_windows=override_windows, high_speed_kph=high_speed_kph,
                compliance_context=context, qualifying_battery_calibration=False,
                deployment_scale=high, soc_reserve_taper=False,
                qualifying_ers_inference=qualifying_ers_inference,
                qualifying_straight_mode_policy=qualifying_straight_mode_policy,
            )
        if trial['summary']['socEndMj'] <= qualifying_terminal_tolerance_mj:
            for _ in range(12):
                midpoint = (low + high) / 2.0
                candidate = compute_lap_energy(
                    trace, year=year, round_number=round_number, session_name=session_name,
                    lap_fraction=lap_fraction, weather=weather, soc_start_mj=soc_window,
                    override_windows=override_windows, high_speed_kph=high_speed_kph,
                    compliance_context=context, qualifying_battery_calibration=False,
                    deployment_scale=midpoint, soc_reserve_taper=False,
                    qualifying_ers_inference=qualifying_ers_inference,
                    qualifying_straight_mode_policy=qualifying_straight_mode_policy,
                )
                if candidate['summary']['socEndMj'] <= qualifying_terminal_tolerance_mj:
                    high = midpoint
                else:
                    low = midpoint
            selected_scale = high
            calibrated = True
        else:
            selected_scale = high
            calibrated = False
        result = compute_lap_energy(
            trace, year=year, round_number=round_number, session_name=session_name,
            lap_fraction=lap_fraction, weather=weather, soc_start_mj=soc_window,
            override_windows=override_windows, high_speed_kph=high_speed_kph,
            compliance_context=context, qualifying_battery_calibration=False,
            deployment_scale=selected_scale, soc_reserve_taper=False,
            qualifying_ers_inference=qualifying_ers_inference,
            qualifying_straight_mode_policy=qualifying_straight_mode_policy,
        )
        result['summary'].update({
            'socDisplayMode': 'QUALIFYING_NORMALISED_USABLE_WINDOW',
            'socStartPercent': 100.0,
            'socEndPercent': round(100.0 * result['summary']['socEndMj'] / soc_window, 2),
            'qualifyingBatteryCalibrated': calibrated,
            'qualifyingDeploymentScale': round(selected_scale, 4),
            'qualifyingBatteryNote': 'MODELLED: 100–0% is the FIA 4 MJ usable on-track SoC window, not a measured team battery percentage.',
            'qualifyingErsInference': {
                'status': 'PUBLIC_TELEMETRY_CONSTRAINED_INFERENCE',
                'mode': ('FIA_OVERTAKE_CURVE_WITHIN_USER_STRAIGHT_MODE_ZONES'
                         if qualifying_straight_mode_policy is not None else 'FIA_OVERTAKE_ACTIVE_LAP_WIDE'),
                'signals': ['throttle', 'longitudinal acceleration', 'tractive power demand', 'braking recovery'],
                'isMeasuredTeamTelemetry': False,
            },
        })
        if qualifying_straight_mode_policy is not None:
            result['summary']['qualifyingBatteryDeploymentPolicy'] = {
                'policy': qualifying_straight_mode_policy.get('policy', 'USER_STRAIGHT_MODE_ZONES_ONLY'),
                'status': qualifying_straight_mode_policy.get('status', 'USER_STRAIGHT_MODE_REFERENCE_MISSING'),
                'referenceKey': qualifying_straight_mode_policy.get('referenceKey'),
                'zoneCount': len(qualifying_straight_mode_policy.get('windows') or []),
                'mappingBasis': qualifying_straight_mode_policy.get('mappingBasis'),
                'isFiaRule': False,
                'enforced': True,
            }
        return result

    soc = soc_window * 0.7 if soc_start_mj is None else float(soc_start_mj)
    soc = float(np.clip(soc, 0.0, soc_window))

    p_elec_kw = np.zeros(n)
    p_ice_kw = np.zeros(n)
    p_harvest_kw = np.zeros(n)
    soc_trace = np.zeros(n)
    clipping = np.zeros(n, dtype=bool)
    harvest_lap_mj = 0.0
    deploy_full_throttle_mj = 0.0
    harvest_high_speed_mj = 0.0

    override_windows = override_windows or []
    strict_qualifying_straight_mode = qualifying and qualifying_straight_mode_policy is not None
    qualifying_straight_mode_windows = (
        qualifying_straight_mode_policy.get('windows') or []
        if strict_qualifying_straight_mode else []
    )
    straight_mode_eligible = np.ones(n, dtype=bool)
    if strict_qualifying_straight_mode:
        straight_mode_eligible = np.asarray([
            _in_distance_windows(point, qualifying_straight_mode_windows)
            for point in distance
        ], dtype=bool)
        straight_mode_window_indices = np.asarray([
            _straight_mode_window_index(point, qualifying_straight_mode_windows)
            for point in distance
        ], dtype=int)
    else:
        straight_mode_window_indices = np.full(n, -1, dtype=int)
    # Keep a small, explicit energy reserve for every later Straight Mode
    # window. Without it, a telemetry-weighted optimiser can spend the full
    # usable window in the first long straight, then misrepresent later
    # allowed deployment zones as clipping. This is a model allocation rule,
    # not a claim about private team ERS commands.
    straight_mode_minimum_reserve_mj = 0.16
    # This small continuous floor is reserved over the remaining portion of
    # the current Straight Mode window as well. It avoids a blue deployment
    # segment collapsing to orange midway through a long straight.
    straight_mode_minimum_deploy_kw = 20.0
    # The FIA B7.2 Competition document can nominate C5.2.8iii sectors
    # where the alternative ERS-K curve applies in a race.  They are not
    # qualifying restrictions; an active Overtake window takes priority.
    alternative_windows = [
        (float(zone['startDistanceM']), float(zone['endDistanceM']))
        for zone in (context.get('alternativePowerSectors') or [])
        if zone.get('startDistanceM') is not None and zone.get('endDistanceM') is not None
    ] if session_kind(session_name) == 'RACE_OR_RUNNING' else []

    for i in range(n):
        in_override = any(start <= distance[i] <= end for start, end in override_windows)
        in_alternative_sector = any(start <= distance[i] <= end for start, end in alternative_windows)
        envelope_kw = propulsion_ceiling_kw(
            speed_kph[i], context,
            overtake=in_override,
            power_limited=in_alternative_sector and not in_override,
        )
        omega_rad_s = max(1e-6, rpm[i] * 2.0 * np.pi / 60.0)
        mechanical_torque_cap_kw = regs.CONSTANTS['mgu_k_torque_max_nm']['value'] * omega_rad_s / 1000.0
        # C5.2.21 converts electrical and mechanical quantities with a fixed
        # 0.97 factor for compliance. The separate component efficiencies are
        # retained only as explicitly modelled physical assumptions.
        deploy_torque_cap_kw = mechanical_torque_cap_kw / standard_ecu_eta
        harvest_torque_cap_kw = mechanical_torque_cap_kw * standard_ecu_eta

        braking = brake[i] and a[i] < -5.0
        if braking:
            p_reg_kw = min(ers_cap_kw, harvest_torque_cap_kw, max(0.0, -p_wheel[i]) / 1000.0 * eta_regen)
            if harvest_lap_mj + p_reg_kw * dt[i] / 1000.0 > harvest_cap_mj:
                p_reg_kw = max(0.0, (harvest_cap_mj - harvest_lap_mj) * 1000.0 / dt[i])
            p_harvest_kw[i] = p_reg_kw
            harvest_lap_mj += p_reg_kw * dt[i] / 1000.0
            if speed_kph[i] > high_speed_kph:
                harvest_high_speed_mj += p_reg_kw * dt[i] / 1000.0
            soc = min(soc_window, soc + p_reg_kw * dt[i] / 1000.0 * eta_in)
        else:
            demand_kw = max(0.0, p_wheel[i]) / 1000.0 / eta_drive if p_wheel[i] > 0 else 0.0
            ice_ceiling_kw = regs.ice_power_ceiling_kw(rpm[i], eta_ice) * max(0.2, throttle[i])
            propulsion_kw = min(ice_ceiling_kw, demand_kw)
            p_ice_kw[i] = min(ice_ceiling_kw, propulsion_kw + ASSUMPTIONS['auxiliaryLoadKw']['value'])
            remaining = max(0.0, demand_kw - propulsion_kw)
            # On a qualifying lap ERS-K can replace part of the ICE's wheel
            # contribution; it is not limited to a shortfall after the ICE
            # ceiling. This keeps observed wheel demand unchanged while giving
            # the controller a physically meaningful deployment budget. Race
            # behaviour deliberately retains the conservative legacy rule
            # until the separate tyre/traffic/race-control phase is designed.
            qualifying_displacement_kw = (
                demand_kw * max(0.0, min(1.0, (throttle[i] - 0.50) / 0.50))
                if qualifying and straight_mode_eligible[i] else 0.0
            )
            electric_opportunity_kw = max(remaining, qualifying_displacement_kw)
            if strict_qualifying_straight_mode and not straight_mode_eligible[i]:
                electric_opportunity_kw = 0.0
            deploy_target_kw = min(
                electric_opportunity_kw * max(0.0, float(deployment_scale))
                * (qualifying_deployment_multiplier[i] if qualifying else 1.0),
                ers_cap_kw, deploy_torque_cap_kw, envelope_kw,
            )
            # Battery protection: taper deployment below 15% SoC to keep an
            # end-of-straight reserve (avoids unrealistic deep clipping).
            if soc_reserve_taper and soc < 0.15 * soc_window:
                deploy_target_kw *= soc / (0.15 * soc_window)
            available_soc_mj = soc
            if strict_qualifying_straight_mode and straight_mode_window_indices[i] >= 0:
                current_window = straight_mode_window_indices[i]
                future_indices = straight_mode_window_indices[i + 1:]
                later_windows = np.unique(
                    future_indices[(future_indices >= 0) & (future_indices != current_window)]
                )
                future_can_deploy = (
                    (future_indices == current_window)
                    & (~brake[i + 1:])
                    & (throttle[i + 1:] >= 0.55)
                    & (p_wheel[i + 1:] > 0.0)
                )
                current_window_seconds = float(np.sum(dt[i + 1:][future_can_deploy]))
                current_window_reserve_mj = (
                    straight_mode_minimum_deploy_kw * current_window_seconds / 1000.0 / eta_out
                )
                available_soc_mj = max(
                    0.0,
                    soc - straight_mode_minimum_reserve_mj * len(later_windows) - current_window_reserve_mj,
                )
            deploy_kw = min(deploy_target_kw, available_soc_mj / dt[i] * 1000.0 * eta_out) if dt[i] > 0 else 0.0
            # A deliberately retained allocation for a later Straight Mode
            # zone is not "superclipping". Flag clipping only when the usable
            # window is actually exhausted, so the map does not paint a
            # planned battery distribution orange.
            if deploy_kw < deploy_target_kw - 1.0 and soc <= 1e-4 and throttle[i] > 0.9:
                clipping[i] = True
            p_elec_kw[i] = deploy_kw
            if qualifying:
                # ERS deployment substitutes ICE wheel power rather than
                # adding unobserved acceleration to the recorded trace.
                p_ice_kw[i] = min(
                    ice_ceiling_kw,
                    max(0.0, demand_kw - deploy_kw) + ASSUMPTIONS['auxiliaryLoadKw']['value'],
                )
            if throttle[i] > 0.9:
                deploy_full_throttle_mj += deploy_kw * dt[i] / 1000.0
            soc = max(0.0, soc - deploy_kw * dt[i] / 1000.0 / eta_out)
        soc_trace[i] = soc

    fuel_mj = np.sum(p_ice_kw * dt) / 1000.0 / eta_ice
    deploy_mj = np.sum(p_elec_kw * dt) / 1000.0
    harvest_mj = np.sum(p_harvest_kw * dt) / 1000.0
    clip_s = float(np.sum(dt[clipping]))
    inferred_full_deploy = p_elec_kw >= 0.95 * ers_cap_kw
    inferred_high_priority = (ers_inference_score >= 0.75) & (p_elec_kw > 1.0)

    return {
        'trace': {
            'time': time,
            'distance': distance,
            'speedKph': speed_kph,
            'pElecKw': p_elec_kw,
            'pIceKw': p_ice_kw,
            'pHarvestKw': p_harvest_kw,
            'socMj': soc_trace,
            'clipping': clipping,
            'ersInferenceScore': ers_inference_score,
            'straightModeEligible': straight_mode_eligible,
            'straightModeWindowIndex': straight_mode_window_indices,
        },
        'summary': {
            'carMassKg': round(mass, 1),
            'fuelOnBoardKg': round(fuel_kg, 1),
            'airDensityKgM3': round(rho, 3),
            'airDensityAssumed': rho_assumed,
            'deployMj': round(deploy_mj, 3),
            'harvestMj': round(harvest_mj, 3),
            'fuelEnergyMj': round(fuel_mj, 2),
            'netSocDeltaMj': round(float(soc_trace[-1] - (soc_window * 0.7 if soc_start_mj is None else soc_start_mj)), 3),
            'socStartMj': round(soc_window * 0.7 if soc_start_mj is None else soc_start_mj, 3),
            'socEndMj': round(float(soc_trace[-1]), 3),
            'socWindowMj': soc_window,
            'socDisplayMode': 'MODELLED_SOC_MJ',
            'clipSeconds': round(clip_s, 2),
            'harvestCapMj': harvest_cap_mj,
            'eventRechargeLimitLoaded': bool(context.get('eventSpecificDataLoaded')),
            'standardEcuEfficiencyCorrection': standard_ecu_eta,
            'deployFullThrottleMj': round(deploy_full_throttle_mj, 3),
            'harvestHighSpeedMj': round(harvest_high_speed_mj, 3),
            'inferredFull350KwSeconds': round(float(np.sum(dt[inferred_full_deploy])), 2),
            'inferredHighPriorityDeploySeconds': round(float(np.sum(dt[inferred_high_priority])), 2),
            'straightModeMinimumReserveMj': (straight_mode_minimum_reserve_mj if strict_qualifying_straight_mode else None),
            'straightModeMinimumDeployKw': (straight_mode_minimum_deploy_kw if strict_qualifying_straight_mode else None),
        },
        'assumptions': ASSUMPTIONS,
        'regulation': regulation_payload(context),
        'citations': {
            'ersCapKw': regs.CONSTANTS['ers_k_dc_power_max_kw']['citation'],
            'socWindowMj': regs.CONSTANTS['es_soc_window_mj']['citation'],
            'harvestCapMj': regs.CONSTANTS['harvest_max_mj_per_lap']['citation'],
            'propulsionEnvelope': regs.CONSTANTS['override_activation']['citation'],
            'standardEcuEfficiencyCorrection': regs.CONSTANTS['standard_ecu_efficiency_correction']['citation'],
            'minimumMass': regs.CONSTANTS['minimum_mass_race_kg']['citation'],
            'carMassDefinition': f'{regs.TECHNICAL_C} Art. C4.1 and Appendix C1 Car Mass definition',
        },
    }
