"""FIA 2026 regulation constants used by the PitWolf energy engine.

Every number carries its source document and article. Values the regulations
delegate to per-event appendices (which competitions harvest at 8 MJ, detection
gaps/lines) are NOT published in the PDFs, so this module uses the regulation
defaults and flags such values as assumption-prone.

The cached documents in ``backend/data/fia-docs/`` are historical drafts only.
The source URLs below are the current published FIA documents and must be
rechecked whenever the FIA publishes a later issue.
"""

REGULATION_VERSION = {
    'sporting_b': {
        'document': '2026 Formula 1 Regulations — Section B [Sporting], Issue 08',
        'published': '2026-08-05',
        'url': 'https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_08_-_2026-08-05_7.pdf',
    },
    'technical_c': {
        'document': '2026 Formula 1 Regulations — Section C [Technical], Issue 20',
        'published': '2026-08-05',
        'url': 'https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_c_technical_-_iss_20_-_2026-08-05.pdf',
    },
}

TECHNICAL_C = '2026 FIA F1 Regulations — Section C [Technical], Issue 20 (2026-08-05)'
SPORTING_B = '2026 FIA F1 Regulations — Section B [Sporting], Issue 08 (2026-08-05)'


def reg(value, unit, citation, note=None):
    return {'value': value, 'unit': unit, 'citation': citation, 'note': note}


CONSTANTS = {
    'fuel_energy_flow_max_mj_h': reg(
        3000.0, 'MJ/h', f'{TECHNICAL_C} Art. C5.2.3',
        'Fuel energy flow must not exceed 3000 MJ/h.'),
    'fuel_energy_flow_rpm_ramp_limit': reg(
        10500.0, 'rpm', f'{TECHNICAL_C} Art. C5.2.4',
        'Below this rpm the fuel energy flow limit is EF = 0.27*N + 165 MJ/h.'),
    'ers_k_dc_power_max_kw': reg(
        350.0, 'kW', f'{TECHNICAL_C} Art. C5.2.7',
        'Absolute electrical DC power of the ERS-K may not exceed 350 kW.'),
    'es_soc_window_mj': reg(
        4.0, 'MJ', f'{TECHNICAL_C} Art. C5.2.9',
        'Max minus min state of charge of the ES may not exceed 4 MJ on track.'),
    'harvest_max_mj_per_lap': reg(
        8.5, 'MJ/lap', f'{TECHNICAL_C} Art. C5.2.10',
        'Recharge measured at the CU-K HV DC bus, per lap.'),
    'harvest_max_mj_per_lap_designated': reg(
        7.0, 'MJ/lap', f'{TECHNICAL_C} Art. C5.2.10 i',
        'Reduced cap at FIA-designated competitions; the event configuration must supply where it applies.'),
    'override_extra_harvest_mj_per_lap': reg(
        0.5, 'MJ/lap', f'{TECHNICAL_C} Art. C5.2.10 iii',
        'Additional Recharge is conditional on the Sporting Regulations and event configuration.'),
    'standard_ecu_efficiency_correction': reg(
        0.97, '-', f'{TECHNICAL_C} Art. C5.2.21',
        'Fixed correction applied by Standard ECU software when converting electrical and mechanical quantities.'),
    'mgu_k_torque_max_nm': reg(
        500.0, 'Nm', f'{TECHNICAL_C} Art. C5.2.11',
        'MGU-K mechanical torque magnitude, efficiency-corrected by 0.97.'),
    'mgu_k_standing_start_speed_kph': reg(
        50.0, 'kph', f'{TECHNICAL_C} Art. C5.2.12',
        'MGU-K usable during a standing start only once the car reaches 50 km/h.'),
    'garage_charge_max_kj_qualifying': reg(
        100.0, 'kJ', f'{TECHNICAL_C} Art. C5.2.13',
        'Qualifying/Sprint Qualifying garage limit only; this must not be treated as a race pit-stop recharge allowance.'),
    'minimum_mass_race_kg': reg(
        724.0, 'kg', f'{TECHNICAL_C} Art. C4.1',
        'Race minimum mass excluding Nominal Tyre Mass; qualifying is 726 kg.'),
    'minimum_mass_qualifying_kg': reg(
        726.0, 'kg', f'{TECHNICAL_C} Art. C4.1',
        'Sprint Qualifying and Qualifying minimum mass excluding Nominal Tyre Mass.'),
    'heat_hazard_mass_increase_kg': reg(
        5.0, 'kg', f'{TECHNICAL_C} Art. C4.6',
        'Heat Hazard Mass Increase during a TTCS; other Competition sessions have a separate 2 kg requirement.'),
    'driver_reference_mass_min_kg': reg(
        82.0, 'kg', f'{TECHNICAL_C} Art. C4.5.2',
        'Driver reference mass plus driver ballast must not be less than 82 kg.'),
    'override_activation': reg(
        None, None, f'{SPORTING_B} Art. B7.2',
        'Per event the FIA publishes the Detection Gap, Detection Line, Activation Line, '
        'power limits and Recharge limits. Overtake requires the control-electronics '
        'enabled/activated state; no event-specific value may be inferred.'),
}

MINIMUM_TYRE_MASS_KG = {
    'value': 44.0,
    'unit': 'kg',
    'citation': 'ASSUMPTION — Car TR C Appendix defines Nominal Tyre Mass as a set of '
                'new dry-weather tyres measured by the tyre provider; the published 2026 '
                'figure is not available in the regulations, so a nominal set mass is '
                'assumed (labelled MODELLED).',
    'note': 'Per-set mass of four 18-inch slicks; update if the FIA publishes the value.',
}

MAX_FUEL_START_KG = {
    2026: {'value': 70.0, 'citation': 'ASSUMPTION — 2026 regs target ~70 kg race fuel '
                                       'start (regulation figure to be confirmed; labelled MODELLED).'},
    'default': {'value': 110.0, 'citation': 'ASSUMPTION — 110 kg max fuel allowance was '
                                            'the 2019-2025 limit (historical regs; labelled MODELLED).'},
}


def propulsion_envelope_kw(speed_kph, override=False):
    """ERS-K propulsion power ceiling vs car speed — Technical C5.2.8."""
    v = max(0.0, speed_kph)
    if override:
        return max(0.0, 7100.0 - 20.0 * v) if v < 355.0 else 0.0
    if v < 340.0:
        return 1800.0 - 5.0 * v
    if v < 345.0:
        return 6900.0 - 20.0 * v
    return 0.0


def fuel_energy_flow_mj_h(rpm):
    """Fuel energy flow ceiling — PU TR Art. 5.4.3 / 5.4.4."""
    if rpm >= CONSTANTS['fuel_energy_flow_rpm_ramp_limit']['value']:
        return CONSTANTS['fuel_energy_flow_max_mj_h']['value']
    return min(CONSTANTS['fuel_energy_flow_max_mj_h']['value'], 0.27 * rpm + 165.0)


def ice_power_ceiling_kw(rpm, ice_efficiency):
    """Maximum ICE power implied by the fuel-flow ceiling.

    The efficiency is an assumption (typical modern F1 thermal efficiency);
    the fuel-flow limit itself is Technical C5.2.3/C5.2.4.
    """
    return fuel_energy_flow_mj_h(rpm) / 3.6 * ice_efficiency
