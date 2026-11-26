"""Bloch-Siegert phase (PULSIM.bloch_siegert), checked against the values of
Figure S13 (preset 1: Q3 180 deg on C' at 176 ppm, carrier on C-alpha at 56 ppm,
13C at 600 MHz) and against the relations the SI states."""

import numpy as np

from PULSIM import RFShape, Pulse, NumpyBackend
from PULSIM.spin_system import gyro_ratio
from PULSIM.bloch_siegert import (OffsetPulse, ppm_to_khz, nucleus_mhz, run, bs_phase,
                                  bs_theory, mirror_echo, equivalent_delay)

MHZ = nucleus_mhz(600.0, "13C")
PPM = np.linspace(200.0, 0.0, 801)
CA_BAND = np.linspace(70.0, 40.0, 121)


def q3(at_ppm, carrier_ppm):
    """0.35 ms Gaussian-cascade Q3 180 deg on 13C at `at_ppm`, phase zero at its center."""
    shape = RFShape.create("gausscasq3", duration=0.35, points=500)
    return OffsetPulse(shape, np.pi, axis="x", backend=NumpyBackend(Gamma=gyro_ratio("13C")),
                       offset_khz=ppm_to_khz(at_ppm, carrier_ppm, MHZ), align=0.5)


def at(df, value):
    return int(np.argmin(np.abs(df - value)))


def test_units():
    assert nucleus_mhz(600.0, "H") == 600.0
    assert abs(MHZ - 150.9) < 0.1
    assert abs(ppm_to_khz(176.0, 56.0, MHZ) - 18.11) < 0.01


def test_offset_pulse_acts_at_its_own_frequency():
    """A 90 deg pulse played +5 kHz from the carrier excites +5 kHz, not -5 kHz,
    for positive (1H) and negative (15N) gamma alike."""
    for nucleus in ("H", "15N"):
        shape = RFShape.create("gausscasq5", duration=2.0, points=500)
        P = OffsetPulse(shape, np.pi / 2, backend=NumpyBackend(Gamma=gyro_ratio(nucleus)),
                        offset_khz=5.0)
        mz = run([P], [5.0, -5.0], "z")[2]
        assert abs(mz[0]) < 0.05 and mz[1] > 0.99, nucleus


def test_phase_of_preset_1():
    """C-alpha at 56 ppm turns by about -97 deg (Figure S13a); both expressions
    follow the calculation, and the approximation improves far from the pulse."""
    P, df = q3(176.0, 56.0), ppm_to_khz(PPM, 56.0, MHZ)
    phi, th = bs_phase([P], df)["phi"], bs_theory([P], df, "13C")
    i = at(df, 0.0)
    assert abs(phi[i] + 97.4) < 1.0
    assert abs(th["integral"][i] - phi[i]) < 8.0 and abs(th["approx"][i] - phi[i]) < 3.0
    far = at(df, ppm_to_khz(0.0, 56.0, MHZ))
    assert abs(th["approx"][far] - phi[far]) < 0.03 * abs(phi[far])


def test_excited_offsets_are_masked():
    P, df = q3(176.0, 56.0), ppm_to_khz(PPM, 56.0, MHZ)
    r = bs_phase([P], df)
    i = at(df, ppm_to_khz(176.0, 56.0, MHZ))
    assert r["excited"][i] and np.isnan(r["phi"][i])
    assert not r["excited"][at(df, 0.0)]


def test_carrier_does_not_change_the_phase():
    """Presets 1 and 2: the same pulse with the carrier on C-alpha or on C' turns
    each spin by the same phi_BS (it depends on the pulse and the spin only)."""
    p1 = bs_phase([q3(176.0, 56.0)], ppm_to_khz(CA_BAND, 56.0, MHZ))["phi"]
    p2 = bs_phase([q3(176.0, 176.0)], ppm_to_khz(CA_BAND, 176.0, MHZ))["phi"]
    assert np.nanmax(np.abs(p1 - p2)) < 0.5


def test_mirror_case_has_the_opposite_sign():
    """Preset 3: a pulse 120 ppm below the spins turns them by about as much as a
    pulse 120 ppm above, with the opposite sign."""
    above = bs_phase([q3(176.0, 56.0)], [0.0])["phi"][0]
    below = bs_phase([q3(56.0, 176.0)], [0.0])["phi"][0]
    assert above < 0 < below
    assert abs(above + below) < 0.05 * abs(above)


def test_method_1_cancels_the_phase():
    """The identical pulse at the mirror position of an echo (Figure S13c, left)."""
    P = q3(176.0, 56.0)
    h180 = Pulse(RFShape.create("hard", duration=0.001, points=10), np.pi,
                 backend=NumpyBackend(Gamma=gyro_ratio("13C")))
    m = mirror_echo(P, ppm_to_khz(CA_BAND, 56.0, MHZ), 0.5, h180)
    assert np.abs(m["one"]).min() > 80.0
    assert np.abs(m["two"]).max() < 0.5


def test_methods_2_and_3_preset_1():
    """Delay +15.2 us after the pulse removes the slope, the next pulse at -97 deg
    the rest (Figure S13c, right)."""
    eq = equivalent_delay(q3(176.0, 56.0), ppm_to_khz(CA_BAND, 56.0, MHZ))
    assert abs(eq["t_eq"] * 1000 + 15.2) < 0.5
    assert abs(eq["const"] + 97.4) < 1.0
    assert np.abs(eq["e_fix"]).max() < 2.0
    assert np.abs(eq["e_unc"]).max() > 80.0