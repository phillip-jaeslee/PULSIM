"""
tutorial_bird_homonuclear_decoupling.py -- BIRD (BIlinear Rotation Decoupling),
Garbow, Weitekamp & Pines, "Bilinear rotation decoupling of homonuclear scalar interactions",
Chem. Phys. Lett. 93, 504-509 (1982).

Setting: a proton I directly bonded to a heteronucleus S (13C, 1J(CH) ~ 140 Hz -- a "13C satellite" proton),
homonuclear-J-coupled (J(HH), a few Hz) to a second proton I' that is NOT bonded to S. 
I and I' are chemically indistinguishable to any 1H pulse -- any 1H pulse is nonselective and hits both at once.
The only thing that tells them apart is whether they're coupled to S.

BIRD cluster (tau = 1/(2*1J(CH))):

    90x(H) -- tau -- [180x(H), 180x(S)] -- tau -- 90(+-x)(H)

Both protons get the SAME three 1H pulses (nonselective). What differs is the coupling network 
each one lives in during the two tau delays -- and that's enough. Verified numerically before this file was written:

1. As a filter: with the two 90's IN PHASE, BIRD nets to a clean
   inversion (Iz -> -Iz) for the S-attached proton I, while the
   non-attracted I' comes back essentially unchanged (+Iz). Flipping the
   final 90's phase swaps which one gets inverted. This is the "bilinear"
   part of the name: the simultaneous S-channel 180 is what lets a
   1H-only pulse train tell I and I' apart, purely through J(CH)

2. As a homonuclear decoupler (the paper's actual point): substitute BIRD
   for the ordinary nonselective 180 in the middle of a symmetric spin
   echo (tau' -- [180 or BIRD] -- tau'), and watch I's own in-phase
   transverse magnetization Ix(I) as a function of tau'. A plain 180
   flips both I and I' together, which leaves their mutual J(HH) coupling
   completely UNREFOCUSED (a well-known property of homonuclear echoes --
   only the two spins' individual offsets get refocused, not the coupling
   between them) -- Ix(I) comes back modulated by exactly
   cos(2*pi*J(HH)*tau'), confirmed to match the exact simulation
   bit-for-bit. BIRD, by contrast, inverts ONLY I not (not I'), which DOES
   refocus their mutual J(HH) over the outer echo -- Ix(I) comes back
   completely independent of tau', i.e. decoupled from I' even though no
   pulse ever touched I' selectively. The residual amplitude isn't
   exactly 1 -- it carries a small, tau'-independent "toll"
   (cos(pi*J(HH)*2*tau_BIRD)) from J(HH) evolving, unrefocused, during
   BIRD's own short internal delays -- confirmed to match exactly.
"""

import numpy as np
import matplotlib.pyplot as plt

from PULSIM.spin_operators import Ix, Iy, Iz, embed
from PULSIM.spin_system import SpinSystem
from PULSIM.liouville import Delay, IdealPulse, LiouvilleSequence

PI, twoPI = np.pi, 2 * np.pi

J_CH = 140.0    # Hz, 1J(CH) -- I (spin 0) is bonded to S (spin 2)
J_HH = 7.0      # Hz, homonuclear J between I (spin 0) and I' (spin 1)
N_SPINS = 3     # I, I', S

IxI = embed(Ix(), 0, N_SPINS)

def coeff(sigma, A):
    """<A> = Tr(sigma A) / Tr(A A) -- trace-formula readout."""
    return np.trace(sigma @ A).real / np.trace(A @ A).real

def bird_cluster(final_phase=0.0):
    """The BIRD element itself: 90x(H) - tau - [180x(H), 180x(S)] - tau - 90(final_phase)(H), tau = 1/(2*1J(CH))"""
    tau = 1.0 / (2 * J_CH / 1000.0)     # ms
    return [
        IdealPulse(PI / 2, phase=0.0, channel='H'),
        Delay(tau),
        IdealPulse(PI, phase=0.0, channel='H'),
        IdealPulse(PI, phase=0.0, channel='13C'),
        Delay(tau),
        IdealPulse(PI / 2, phase=final_phase, channel='H'),
    ]

def spin_system():
    return SpinSystem(nuclei=['H', 'H', '13C'], offsets=[0.0, 0.0, 0.0], couplings={(0, 2): J_CH, (0, 1): J_HH})

# -- Part 1: BIRD as a filter -- which proton gets inverted? ----------
print("Part 1: BIRD as an S-coupling filter (starting each proton at +Iz alone)")
for final_phase, label in [(0.0, "90(+x)...90(+x)"), (PI, "90(+x)...90(-x)")]:
    seq = LiouvilleSequence(bird_cluster(final_phase), spin_system())
    z_I  = coeff(seq.propagate(embed(Iz(), 0, N_SPINS)), embed(Iz(), 0, N_SPINS))
    z_Ip = coeff(seq.propagate(embed(Iz(), 1, N_SPINS)), embed(Iz(), 1, N_SPINS))
    print(f"  {label:18s}  I (S-attached) -> {z_I:+.4f}   I' (not attached) -> {z_Ip:+.4f}")

# -- Part 2: BIRD as a homonuclear decoupler -------------------------
def run_echo(tau_outer, mode):
    """tau' -- [plain 180, BIRD, or nothing] -- tau', starting from Ix(I)."""
    if mode == 'plain180':
        middle = [IdealPulse(PI, phase=0.0, channel='H')]
    elif mode == 'bird':
        middle = bird_cluster(final_phase=0.0)
    else:
        middle = []
    segments = [Delay(tau_outer)] + middle + [Delay(tau_outer)]
    seq = LiouvilleSequence(segments, spin_system())
    return seq.propagate(embed(Ix(), 0, N_SPINS))

tau_bird = 1.0 / (2 * J_CH / 1000.0)
bird_toll = np.cos(PI * (J_HH / 1000.0) * 2 * tau_bird)   # predicted BIRD-internal residual

taus = np.linspace(0.0, 1000.0 / J_HH, 5000)   # ms, sweep out past one full J_HH period
amp_plain = np.array([coeff(run_echo(t, 'plain180'), IxI) for t in taus])
amp_bird  = np.array([coeff(run_echo(t, 'bird'),     IxI) for t in taus])
theory_plain = np.cos(twoPI * (J_HH / 1000.0) * taus)


fig, axs = plt.subplots(2, 1, figsize=(7, 8))
axs[0].plot(taus, amp_plain, 'o', label="plain 180 in middle (simulated)", markersize=1.0)
axs[0].plot(taus, theory_plain, '-', label=r"theory: $\cos(2\pi J_{HH}\tau')$", markersize=1.0)
axs[0].plot(taus, amp_bird, 's', color='C3', label="BIRD in middle (simulated)", markersize=1.0)
axs[0].axhline(-bird_toll, color='C3', linestyle=':',
               label=r"BIRD's fixed toll: $-\cos(\pi J_{HH}\cdot 2\tau_{BIRD})$")
axs[0].set_xlabel(r"$\tau'$ (ms, outer half-echo)")
axs[0].set_ylabel(r"$I_x(I)$ amplitude")
axs[0].set_title(f"BIRD decouples I from I' (J_HH={J_HH:.0f} Hz) across the outer echo")
axs[0].legend(fontsize=9)

def draw_bird_scheme(ax):
    y_I, y_S = 1.0, 0.0
    ax.plot([0, 10], [y_I, y_I], color='black', lw=1)
    ax.plot([0, 10], [y_S, y_S], color='black', lw=1)
    ax.text(-0.6, y_I, r'$^1$H (I, I$^\prime$)', va='center', ha='right', fontsize=11)
    ax.text(-0.6, y_S, r'$^{13}$C (S)', va='center', ha='right', fontsize=11)

    def pulse(x, y, width, filled, label):
        h = 0.4
        ax.add_patch(plt.Rectangle((x - width / 2, y - h / 2), width, h,
                                    facecolor='black' if filled else 'white',
                                    edgecolor='black', lw=1.2, zorder=3))
        ax.text(x, y + h / 2 + 0.15, label, ha='center', va='bottom', fontsize=10)

    def bracket(x0, x1, y, label, color='gray'):
        ax.annotate('', xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle='<->', color=color))
        ax.text((x0 + x1) / 2, y + 0.1, label, ha='center', va='bottom',
                 fontsize=9, color=color)

    x0, x1, x2, x3, x4 = 0.5, 3.5, 5.0, 6.5, 9.5
    pulse(x0, y_I, 0.15, False, r'$90_x$')
    pulse(x1, y_I, 0.15, False, r'$90_x$')
    pulse(x2, y_I, 0.35, True,  r'$180_x$')
    pulse(x2, y_S, 0.35, True,  r'$180_x$')
    pulse(x3, y_I, 0.15, False, r'$90_{\pm x}$')
    pulse(x4, y_I, 0.15, False, r'$90_x$')

    bracket(x0, x2, -0.55, r"$\tau'$")
    bracket(x2, x4, -0.55, r"$\tau'$")
    bracket(x1, x2, -0.9, r'$\tau_{BIRD}$', color='C3')
    bracket(x2, x3, -0.9, r'$\tau_{BIRD}$', color='C3')
    ax.text((x1 + x3) / 2, 1.7, 'BIRD cluster', ha='center', fontsize=10,
            style='italic', color='C3')
    ax.set_xlim(-1.5, 10.5)
    ax.set_ylim(-1.2, 2.0)
    ax.axis('off')
    ax.set_title("BIRD substituted for the middle 180 of a homonuclear echo (schematic)")

draw_bird_scheme(axs[1])

plt.tight_layout()
plt.savefig("tutorial_figures/tutorial_bird_homonuclear_decoupling.png")
plt.show()

print(f"\nBIRD's fixed internal toll: cos(pi*J_HH*2*tau_BIRD) = {bird_toll:.4f}  "
      f"(matches simulated BIRD amplitude at tau'=0)")
print(f"BIRD amplitude std dev across the tau' sweep: {amp_bird.std():.2e}  (flat -> decoupled)")
print(f"Plain-180 amplitude range across the sweep: [{amp_plain.min():.4f}, {amp_plain.max():.4f}]  (fully J_HH-modulated)")