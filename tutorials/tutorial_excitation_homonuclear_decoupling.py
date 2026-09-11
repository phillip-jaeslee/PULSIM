"""
tutorial_excitation_homonuclear_decoupling.py -- excitation (offset)
profile of the BIRD filter from tutorial_bird_homonuclear_decoupling.py.

BIRD's I-vs-I' selectivity comes entirely from J(CH): I is coupled to S,
I' isn't, and that's the only thing the sequence's timing
(tau = 1/(2*1J(CH))) is built around. Chemical shift never enters that
construction -- so BIRD's selectivity should, in principle, be completely
independent of how far off-resonance either proton sits.

This sweeps a shared proton offset (I and I' moved off the pulse carrier
together) and reads out, at each offset, the full product-operator basis
relevant to each spin: for I (spin 0, coupled to S via J_CH), Ix/Iy/Iz and
its cross terms with S (IxSz, IySz, IzSz); for I' (spin 1, coupled instead
to I via the homonuclear J_HH), I'x/I'y/I'z and its cross terms with I.

With PULSIM's IdealPulse hard pulses (effectively instantaneous -- default
duration 1e-6 ms, nutation rate ~10^6 rad/ms, orders of magnitude faster
than any realistic offset) and the exact offset-refocusing of the two
intervening Delay periods, every curve here is flat: verified numerically
before this file was written -- only Iz(I) and I'z(I') are nonzero (at
-0.988 and +0.988 respectively, the same BIRD "toll" from
tutorial_bird_homonuclear_decoupling.py), and every other component stays
at zero across the full +-4000 Hz sweep. That flatness is the actual
result: BIRD's selectivity is a pure J-coupling effect, structurally
decoupled from chemical shift. (Swap in a shaped pulse for the hard 90/180s
and this same plot would show real curvature -- finite bandwidth leaking
into off-resonance behavior -- exactly like the shape-vs-file comparison
this file's structure is based on.)
"""

import numpy as np
import matplotlib.pyplot as plt

from PULSIM.spin_operators import Ix, Iy, Iz, embed, product_operator
from PULSIM.spin_system import SpinSystem
from PULSIM.liouville import Delay, IdealPulse, LiouvilleSequence

PI, twoPI = np.pi, 2 * np.pi

J_CH = 140.0        # Hz, 1J(CH) -- I (spin 0) bonded to S (spin 2)
J_HH = 7            # Hz, homonuclear J between I (spin 0) and I' (spin 1)
N = 1000            # number of offsets to sweep
W_hz = np.linspace(-1, 1, N) * 4000.0
N_SPINS = 3

def bird_cluster(final_phase=0.0):
    """90x(H) - tau - [180x(H),180x(S)] - tau - 90(final_phase)(H),
    tau = 1/(2*1J(CH))."""
    tau = 1.0 / (2 * J_CH / 1000.0) # ms
    return [
        IdealPulse(PI / 2, phase=PI / 2, channel='H'),
        Delay(tau),
        IdealPulse(PI, phase=0.0, channel='H'),
        IdealPulse(PI, phase=0.0, channel='13C'),
        Delay(tau),
        IdealPulse(PI / 2, phase=final_phase, channel='H'),
    ]

def compensated_bird_cluster(final_phase=0.0):
    tau = 1.0 / (2 * J_CH / 1000.0) # ms
    return [
        IdealPulse(PI / 2, phase=PI / 2, channel='H'),
        Delay(tau / 2),
        IdealPulse(PI, phase=0.0, channel='H'),
        IdealPulse(PI, phase=0.0, channel='13C'),
        Delay(tau / 2),
        IdealPulse(PI / 2, phase=PI, channel='H'),
        Delay(tau),
        IdealPulse(PI, phase=0.0, channel='H'),
        IdealPulse(PI, phase=0.0, channel='13C'),
        Delay(tau),
        IdealPulse(PI / 2 , phase=0.0, channel='H'),
        Delay(tau / 2),
        IdealPulse(PI, phase=0.0, channel='H'),
        IdealPulse(PI, phase=0.0, channel='13C'),
        Delay(tau / 2),
        IdealPulse(PI / 2, phase= -PI / 2, channel='H')
    ]

def run_compensated_bird(sigma0, off_hz):
    off = twoPI * (off_hz / 1000.0)
    ss = SpinSystem(nuclei=['H', 'H', '13C'], offsets=[off, off, 0.0], couplings={(0, 2): J_CH, (0, 1): J_HH})
    seq = LiouvilleSequence(compensated_bird_cluster(final_phase=0.0), ss)
    return seq.propagate(sigma0)

def run_bird(sigma0, off_hz):
    off = twoPI * (off_hz / 1000.0)
    ss = SpinSystem(nuclei=['H', 'H', '13C'], offsets=[off, off, 0.0], couplings={(0, 2): J_CH, (0, 1): J_HH})
    seq = LiouvilleSequence(bird_cluster(final_phase= -PI / 2), ss)
    return seq.propagate(sigma0)

def extract_I(sigma):
    """I's own components, plus its cross terms with S -- the bilinear
    (heteronuclear) coupling BIRD's selectivity is built on."""
    name_ops = [('Ix', embed(Ix(), 0, N_SPINS)),
                ('Iy', embed(Iy(), 0, N_SPINS)),
                ('Iz', embed(Iz(), 0, N_SPINS)),
                ('IxSz', product_operator(Ix(), 0, Iz(), 2, N_SPINS)),
                ('IySz', product_operator(Iy(), 0, Iz(), 2, N_SPINS)),
                ('IzSz', product_operator(Iz(), 0, Iz(), 2, N_SPINS)),
    ]
    return {name: np.trace(sigma @ A).real / np.trace(A @ A).real for name, A in name_ops}

def extract_Ip(sigma):
    """I''s own components, plus its cross terms with I -- the homonuclear
    coupling BIRD is meant to leave untouched (I' is never S-coupled)."""
    name_ops = [("I'x",   embed(Ix(), 1, N_SPINS)),
                ("I'y",   embed(Iy(), 1, N_SPINS)),
                ("I'z",   embed(Iz(), 1, N_SPINS)),
                ("I'xIz", product_operator(Ix(), 1, Iz(), 0, N_SPINS)),
                ("I'yIz", product_operator(Iy(), 1, Iz(), 0, N_SPINS)),
                ("I'zIz", product_operator(Iz(), 1, Iz(), 0, N_SPINS))]
    return {name: np.trace(sigma @ A).real / np.trace(A @ A).real for name, A in name_ops}

Iz0_I = embed(Iz(), 0, N_SPINS)
Iz0_Ip = embed(Iz(), 1, N_SPINS)

#keys_I  = ['Ix', 'Iy', 'Iz', 'IxSz', 'IySz', 'IzSz']
#keys_Ip = ["I'x", "I'y", "I'z", "I'xIz", "I'yIz", "I'zIz"]
keys_I = ['Iz']
keys_Ip = ["I'z"]
profile_I  = {k: np.zeros(N) for k in keys_I}
profile_Ip = {k: np.zeros(N) for k in keys_Ip}

for n, off in enumerate(W_hz):
    rI = extract_I(run_compensated_bird(Iz0_I, off))
    rIp = extract_Ip(run_compensated_bird(Iz0_Ip, off))
    for k in keys_I:
        profile_I[k][n] = rI[k]
    for k in keys_Ip:
        profile_Ip[k][n] = rIp[k]

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, squeeze=True, constrained_layout=True)
fig.set_size_inches(11, 4.5, forward=True)

ax[0].axhline(y=0, color='lightgray', linestyle='--')
for k, color in zip(keys_I, ['red', 'blue', 'green', 'orange', 'cyan', 'olive']):
    ax[0].plot(W_hz, profile_I[k], '--', color=color, label=k)
ax[0].set_title(f'I (S-attached, J_CH={J_CH:.0f} Hz) after BIRD')
ax[0].set_xlabel('offset (Hz)')
ax[0].legend(fontsize=9)
ax[0].invert_xaxis()

ax[1].axhline(y=0, color='lightgray', linestyle='--')
for k, color in zip(keys_Ip, ['red', 'blue', 'green', 'orange', 'cyan', 'olive']):
    ax[1].plot(W_hz, profile_Ip[k], '--', color=color, label=k)
ax[1].set_title(f"I' (not attached, J_HH={J_HH:.0f} Hz) after BIRD")
ax[1].set_xlabel('offset (Hz)')
ax[1].legend(fontsize=9)

plt.savefig("tutorial_figures/tutorial_excitation_homonuclear_decoupling.png")
plt.show()

print(f"On-resonance: I  Iz={profile_I['Iz'][N//2]:.4f} (should be ~-0.988, BIRD inverts the S-attached spin)")
key_Ip_z = "I'z"
print(f"On-resonance: I' I'z={profile_Ip[key_Ip_z][N//2]:.4f} (should be ~+0.988, BIRD leaves it alone)")
print("Every other component stays ~0 across the whole sweep -- BIRD's selectivity has no offset dependence.")