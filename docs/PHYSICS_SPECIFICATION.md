# PULSIM Physics and Numerical Conventions

**Document status:** Draft for review  
**Purpose:** Authoritative physics specification for the OOP reconstruction of PULSIM and for the methods and validation sections of future publications.

This document defines the physical model independently of any particular software backend. Implementations must conform to the equations, units, signs, and validation requirements stated here. Legacy PULSIM behavior is not considered authoritative when it conflicts with this specification.

## Document roadmap

1. **Notation, units, rotating frame, and sign conventions** — draft completed below.
2. Exact constant-field propagation without relaxation. *(not yet drafted; the rotation contract it will formalize is stated in Sections 1.6–1.9.)*
3. Relaxation and exact affine propagation.
4. RF phase, pulse calibration, offsets, and negative-gyromagnetic-ratio nuclei.
5. Sampled waveforms, numerical convergence, and validation.
6. Shaped and adiabatic pulses, imported formats, and model limitations.
7. Mapping of the physics to the OOP interfaces.
8. Gradients, spatial ensembles, and spoiling.
9. Vendor shape files and declared intent.

---

## 1. Notation, units, and sign conventions

### 1.1 Scope of the model

The initial reconstruction models a classical bulk magnetization vector

$$
\mathbf M(t)
=
\begin{pmatrix}
M_x(t)\\
M_y(t)\\
M_z(t)
\end{pmatrix}
$$

in a frame rotating about the laboratory $z$ axis. The validated model includes RF fields, resonance offsets, phenomenological $T_1$ and $T_2$ relaxation, and static or piecewise-constant field gradients represented as position-dependent offsets (Section 8). Chemical exchange, diffusion, flow, radiation damping, concomitant (Maxwell) fields, gradient nonlinearity, and quantum-mechanical coupled-spin evolution are outside the model and must be introduced later as separately validated extensions.

### 1.2 Canonical internal units

The reference implementation shall use the following units without implicit conversion:

| Quantity | Symbol | Internal unit |
|---|---:|---:|
| Time | $t$, $\Delta t$, $T_1$, $T_2$ | ms |
| Magnetic field | $B_x$, $B_y$, $B_z$, $B_1$ | mT |
| Cyclic frequency | $f$, $\Delta f$ | kHz |
| Angular frequency | $\omega$, $\Omega$ | rad ms$^{-1}$ |
| Gyromagnetic ratio divided by $2\pi$ | $\bar\gamma=\gamma/(2\pi)$ | kHz mT$^{-1}$ |
| Gyromagnetic ratio | $\gamma=2\pi\bar\gamma$ | rad ms$^{-1}$ mT$^{-1}$ |
| Rotation angle and internal RF phase | $\theta$, $\phi$ | rad |

The identity

$$
1\ \mathrm{kHz}=1\ \mathrm{cycle\,ms^{-1}}
$$

makes $2\pi\bar\gamma B\,\Delta t$ dimensionless and expressed in radians when $\bar\gamma$ is in kHz/mT, $B$ is in mT, and $\Delta t$ is in ms.

RF phases may be read from or written to instrument files in degrees, but they must be converted explicitly to radians at the software boundary.

### 1.3 Gyromagnetic-ratio convention

Both $\bar\gamma$ and $\gamma$ are signed nuclear constants. Their sign must not be removed by an absolute value. The nuclear-constant table used by the software must cite its source and version.

The symbol $\bar\gamma$ always denotes cyclic frequency per field,

$$
\bar\gamma=\frac{\gamma}{2\pi},
$$

whereas $\gamma$ denotes angular frequency per field. Code, documentation, and manuscript equations must not use the same symbol for both quantities.

### 1.4 Complex RF convention

The complex RF waveform is defined as

$$
B_1(t)=B_x(t)+iB_y(t).
$$

For an RF magnitude $A(t)\geq0$ and phase $\phi(t)$,

$$
B_1(t)=A(t)e^{i\phi(t)},
$$

so that

$$
B_x(t)=A(t)\cos\phi(t),
\qquad
B_y(t)=A(t)\sin\phi(t).
$$

Consequently, phase $0$ represents a field along $+x$, and phase $+\pi/2$ represents a field along $+y$. A pulse axis is therefore determined by the complex RF phase; the physical propagation interface should not require a separate ambiguous `axis` argument.

### 1.5 Resonance-offset convention

The signed rotating-frame frequency offset is defined as

$$
\Delta f=f_{\mathrm{spin}}-f_{\mathrm{RF}}.
$$

Both frequencies must use the same signed-frequency and rotating-frame convention. This point is especially important for nuclei with negative $\gamma$; a positive frequency magnitude reported by an instrument is not automatically a signed Larmor frequency.

For the numerical model, $\Delta f$ is the authoritative rotating-frame parameter: a positive $\Delta f$ produces a positive $z$ component of the angular-frequency vector, independent of the sign of $\bar\gamma$.

### 1.6 Effective field and angular-frequency vector

The rotating-frame effective magnetic field is

$$
\mathbf B_{\mathrm{eff}}(t)
=
\begin{pmatrix}
B_x(t)\\[4pt]
B_y(t)\\[4pt]
\dfrac{\Delta f}{\bar\gamma}
\end{pmatrix}.
$$

This effective-field representation is valid for $\bar\gamma\neq0$. The corresponding angular-frequency vector is

$$
\boldsymbol\Omega_{\mathrm{eff}}(t)
=
2\pi\bar\gamma\,\mathbf B_{\mathrm{eff}}(t)
=
\begin{pmatrix}
2\pi\bar\gamma B_x(t)\\[4pt]
2\pi\bar\gamma B_y(t)\\[4pt]
2\pi\Delta f
\end{pmatrix}.
$$

The angular-frequency-vector form is preferred in the numerical kernel because it makes the offset convention explicit and avoids dividing by $\bar\gamma$.

### 1.7 Bloch equation

PULSIM adopts the cross-product convention

$$
\frac{d\mathbf M}{dt}
=
\boldsymbol\Omega_{\mathrm{eff}}\times\mathbf M
-
\begin{pmatrix}
M_x/T_2\\[4pt]
M_y/T_2\\[4pt]
(M_z-M_0)/T_1
\end{pmatrix}.
$$

Equivalently,

$$
\frac{d\mathbf M}{dt}
=
\gamma\,\mathbf B_{\mathrm{eff}}\times\mathbf M
-
\begin{pmatrix}
M_x/T_2\\[4pt]
M_y/T_2\\[4pt]
(M_z-M_0)/T_1
\end{pmatrix}.
$$

The component equations are

$$
\frac{dM_x}{dt}
=
\Omega_y M_z-\Omega_z M_y-\frac{M_x}{T_2},
$$

$$
\frac{dM_y}{dt}
=
\Omega_z M_x-\Omega_x M_z-\frac{M_y}{T_2},
$$

$$
\frac{dM_z}{dt}
=
\Omega_x M_y-\Omega_y M_x-\frac{M_z-M_0}{T_1}.
$$

Here $M_0$ is the equilibrium longitudinal magnetization in the selected normalization. The usual normalized initial condition is

$$
\mathbf M(0)=
\begin{pmatrix}
0\\0\\M_0
\end{pmatrix},
\qquad M_0=1.
$$

### 1.8 Observable consequences of the convention

The following are defining tests, not optional examples:

1. For positive $\bar\gamma$, an on-resonance positive $x$-directed $90^\circ$ pulse maps

   $$
   +M_z\longrightarrow-M_y.
   $$

2. For positive $\bar\gamma$, an on-resonance positive $y$-directed $90^\circ$ pulse maps

   $$
   +M_z\longrightarrow+M_x.
   $$

3. Define the complex transverse magnetization as

   $$
   M_+=M_x+iM_y.
   $$

   During free precession with $\Delta f>0$ and without relaxation,

   $$
   M_+(t)=M_+(0)e^{+i2\pi\Delta f t}.
   $$

   This must be tested at a phase that distinguishes the two sign conventions.
   At $\Delta f\,t=0.5$ both give $-1$, and at integer values both give $+1$.
   Use $\Delta f\,t=0.25$, where the conventions give $+i$ and $-i$.

4. With zero RF, zero offset, and infinite relaxation times, the magnetization is unchanged.

5. With finite relaxation and no applied field,

   $$
   M_x(t)=M_x(0)e^{-t/T_2},
   $$

   $$
   M_y(t)=M_y(0)e^{-t/T_2},
   $$

   $$
   M_z(t)=M_0+[M_z(0)-M_0]e^{-t/T_1}.
   $$

Every backend must reproduce these results within a documented numerical tolerance.

### 1.9 Implementation contract derived from Section 1

The future OOP implementation shall obey the following rules:

- The reference backend uses floating-point 64-bit arithmetic.
- RF waveforms entering the propagator represent physical $B_1$ values in mT, not unspecified normalized amplitudes.
- RF phase is represented through the real and imaginary parts of $B_1$.
- The propagator receives $\Delta f$ explicitly in kHz.
- No internal function silently converts between Hz and kHz, seconds and milliseconds, or degrees and radians.
- Signed $\bar\gamma$ is retained.
- The zero-field case is handled analytically and must never generate a division-by-zero result.
- Coordinate-changing pulse operations are constructed by RF phase, not by changing the mathematical meaning of the field components.
- Display conventions used by an instrument or plotting package are kept separate from the physical convention used by the numerical kernel.

### 1.10 Items to confirm before Section 1 is finalized

The following choices should be explicitly approved before implementation begins:

1. Retain $B_1=B_x+iB_y$.
2. RESOLVED (2026-09-11). Retain $d\mathbf M/dt=\boldsymbol\Omega\times\mathbf M$,
   the sense already implemented by `bloch_rotate` and by the density-matrix
   path, and covered by the product-operator oracle in
   `tests/test_bloch_rotate_levitt_convention.py`. This draft previously wrote
   $\mathbf M\times\boldsymbol\Omega$, the opposite sense; Sections 1.7, 1.8
   and 1.11 have been corrected to match the implementation. No computed
   result changed.
3. Retain $\Delta f=f_{\mathrm{spin}}-f_{\mathrm{RF}}$, using signed frequencies.
4. Use radians internally and permit degrees only at input/output boundaries.
5. Treat $M_+=M_x+iM_y$ as the canonical complex transverse magnetization.

### 1.11 Convention statement for comparison with other software

PULSIM's rotation sense is

$$
\frac{d\mathbf M}{dt}=\boldsymbol\Omega_{\mathrm{eff}}\times\mathbf M,
$$

equivalently, in product-operator form,

$$
I_z\;\xrightarrow{\;\theta I_x\;}\;I_z\cos\theta-I_y\sin\theta,
$$

so a $90^\circ$ pulse of phase $0$ takes $+I_z$ to $-I_y$, a $90^\circ$ pulse of
phase $\pi/2$ takes $+I_z$ to $+I_x$, and during free precession
$M_+(t)=M_+(0)e^{+i2\pi\Delta f t}$. This is the Ernst/Levitt convention
$R_\alpha(\beta)=\exp(-i\beta I_\alpha)$ used by the density-matrix path.

**What is and is not convention-dependent.** $|M_+|$, $M_z$, excitation and
inversion profiles, bandwidths, and every pulse-calibration quantity are
independent of this choice. Only the sign of the transverse components and the
sense of the phase profile depend on it.

**Comparing with another simulator or a spectrometer.** If the other tool's
$90^\circ$ phase-$0$ pulse takes $+I_z$ to $+I_y$, or it reports $M_+$ with the
opposite imaginary sign, it uses the conjugate convention. Compare magnitudes
first; to compare phase, take the complex conjugate of the transverse
magnetization, equivalently negate $M_y$. Display conventions in instrument
software may differ from that software's own internal convention, so verify
against a known case rather than assuming.

**A test that distinguishes them.** Free precession with $\Delta f\,t=0.25$
gives $M_+=+i$ under PULSIM's convention and $-i$ under the conjugate. Values
such as $\Delta f\,t=0.5$, or any integer, are degenerate and cannot tell the
two apart.

---

## References for Section 1

1. F. Bloch, “Nuclear Induction,” *Physical Review* **70**, 460–474 (1946). DOI: [10.1103/PhysRev.70.460](https://doi.org/10.1103/PhysRev.70.460).
2. R. K. Harris et al., “NMR nomenclature. Nuclear spin properties and conventions for chemical shifts,” *Pure and Applied Chemistry* **73**, 1795–1818 (2001). DOI: [10.1351/pac200173111795](https://doi.org/10.1351/pac200173111795).
3. NIST, “2022 CODATA Recommended Values of the Fundamental Physical Constants,” [complete constants listing](https://physics.nist.gov/cuu/pdf/all.pdf).

---

## 3. Relaxation and exact affine propagation

**Status:** Draft completed 2026-09-11, implemented in `PULSIM/bloch.py`.

### 3.1 Scope

This section specifies phenomenological $T_1$ and $T_2$ relaxation of a single
bulk magnetization vector, and the propagation of the resulting inhomogeneous
linear equation. It covers relaxation during free evolution and during an
applied RF field.

Not in scope, and not claimed by any function specified here: diffusion,
chemical exchange, cross-relaxation and the nuclear Overhauser effect,
radiation damping, scalar-coupled multi-spin relaxation, and any distinction
between $T_2$ and $T_2^{*}$. Relaxation is phenomenological throughout: the
rates are inputs, never derived from a motional model.

### 3.2 Relaxation matrix and equilibrium

$$
\mathbf R=\operatorname{diag}\!\left(\frac{1}{T_2},\ \frac{1}{T_2},\ \frac{1}{T_1}\right),
\qquad
\mathbf M_{\mathrm{eq}}=\begin{pmatrix}0\\0\\M_0\end{pmatrix}.
$$

$T_1$ and $T_2$ are in milliseconds, so the entries of $\mathbf R$ are in
$\mathrm{ms}^{-1}$, consistent with Section 1.2.

A relaxation time of `None` (equivalently $\infty$) means that component does
not relax and contributes a rate of exactly zero. A non-positive relaxation
time is an error, not shorthand for instantaneous relaxation, and must raise.

With this notation the Bloch equation of Section 1.7 is

$$
\frac{d\mathbf M}{dt}
=\boldsymbol\Omega_{\mathrm{eff}}\times\mathbf M-\mathbf R\left(\mathbf M-\mathbf M_{\mathrm{eq}}\right).
$$

### 3.3 Free relaxation

With $\boldsymbol\Omega_{\mathrm{eff}}=\mathbf 0$ the three components decouple
and the solution is closed-form:

$$
M_x(t)=M_x(0)\,e^{-t/T_2},\qquad
M_y(t)=M_y(0)\,e^{-t/T_2},
$$

$$
M_z(t)=M_0+\left[M_z(0)-M_0\right]e^{-t/T_1}.
$$

Implementations must use this expression during delays rather than any
stepped approximation. Two consequences are required to hold exactly rather
than approximately: with both relaxation times infinite the result is the
identity, and starting from $\mathbf M_{\mathrm{eq}}$ the result is
$\mathbf M_{\mathrm{eq}}$.

### 3.4 Exact affine propagation over a constant field

Over an interval on which $\boldsymbol\Omega_{\mathrm{eff}}$ is constant, the
equation of Section 3.2 is linear and inhomogeneous,

$$
\frac{d\mathbf M}{dt}=\mathbf L\,\mathbf M+\mathbf c,
\qquad
\mathbf L=\hat{\boldsymbol\Omega}-\mathbf R,
\qquad
\mathbf c=\mathbf R\,\mathbf M_{\mathrm{eq}},
$$

where $\hat{\boldsymbol\Omega}$ is the matrix representing the cross product,
$\hat{\boldsymbol\Omega}\mathbf M=\boldsymbol\Omega_{\mathrm{eff}}\times\mathbf M$:

$$
\hat{\boldsymbol\Omega}=
\begin{pmatrix}
0 & -\Omega_z & \Omega_y\\
\Omega_z & 0 & -\Omega_x\\
-\Omega_y & \Omega_x & 0
\end{pmatrix}.
$$

The exact solution over a step $\Delta t$ is therefore affine,
$\mathbf M\mapsto \mathbf A\mathbf M+\mathbf b$. Both $\mathbf A$ and
$\mathbf b$ are obtained from a single matrix exponential of the augmented
$4\times4$ generator

$$
\mathbf G=\begin{pmatrix}\mathbf L & \mathbf c\\ \mathbf 0 & 0\end{pmatrix},
\qquad
\exp(\mathbf G\,\Delta t)=
\begin{pmatrix}\mathbf A & \mathbf b\\ \mathbf 0 & 1\end{pmatrix}.
$$

The augmented form is required rather than the algebraically equivalent

$$
\mathbf M(\Delta t)=e^{\mathbf L\Delta t}\left(\mathbf M(0)+\mathbf L^{-1}\mathbf c\right)-\mathbf L^{-1}\mathbf c,
$$

because $\mathbf L$ is singular in precisely the case that matters most:
$T_1=T_2=\infty$, where $\mathbf L=\hat{\boldsymbol\Omega}$ and
$\hat{\boldsymbol\Omega}\,\boldsymbol\Omega_{\mathrm{eff}}=\mathbf 0$. The
augmented propagator never inverts anything and is well conditioned in that
limit.

This propagator is the **reference implementation**. It is exact but costs a
$4\times4$ matrix exponential per step, and is not the production path.

### 3.5 Operator splitting: the production path

Rotation and relaxation do not commute in general. Writing
$\mathbf R=r_2\mathbf I+(r_1-r_2)\,\mathbf e_z\mathbf e_z^{\mathsf T}$,

$$
\left[\hat{\boldsymbol\Omega},\mathbf R\right]=\mathbf 0
\quad\Longleftrightarrow\quad
T_1=T_2
\quad\text{or}\quad
\Omega_x=\Omega_y=0 .
$$

So the two generators commute exactly during a delay, where the effective
field is purely longitudinal, and in the isotropic-relaxation case; during a
transverse RF field with $T_1\neq T_2$ they do not.

The production path is therefore the symmetric (Strang) splitting

$$
\mathcal S(\Delta t)
=\mathcal E\!\left(\tfrac{\Delta t}{2}\right)\circ
\mathcal R(\Delta t)\circ
\mathcal E\!\left(\tfrac{\Delta t}{2}\right),
$$

where $\mathcal R$ is the pure rotation of Section 1 and $\mathcal E$ is the
closed-form relaxation of Section 3.3. Its local error is
$\mathcal O(\Delta t^{3})$ and its global error $\mathcal O(\Delta t^{2})$.
Relaxation is placed in the half-steps because it is the cheaper operation:
it is a closed form with no matrix.

Required behavior at the boundary: when both relaxation times are `None`, the
splitting must return the pure-rotation result **bit for bit**, guaranteed
structurally by an early return rather than by relying on a relaxation factor
evaluating to exactly $1.0$.

### 3.6 Implementation contract

| Requirement | Function |
|---|---|
| $\mathbf R$ from $T_1,T_2$; reject non-positive times | `relaxation_matrix` |
| Closed-form free relaxation, single vector or batch | `bloch_relax` |
| Exact affine step, reference implementation | `affine_propagate` |
| Strang-split step, production path | `bloch_relax_rotate_batch` |
| Relaxation-aware backend | `NumpyBackend(Gamma, T1, T2, M0)` |

`TorchBackend` implements rotation only. Passing $T_1$ or $T_2$ to it must
raise `NotImplementedError`: silently discarding them would return a wrong
answer rather than a slow one.

### 3.7 Validation requirements

Section 3 is satisfied only if all of the following hold. They are implemented
in `tests/test_relaxation.py`; none is a golden test.

1. $T_2$ decay and $T_1$ recovery reproduce Section 3.3 at several times,
   including the inversion-recovery null at $t=T_1\ln 2$.
2. $T_2$ leaves $M_z$ unchanged and $T_1$ leaves $M_x,M_y$ unchanged.
3. With relaxation off, `bloch_relax` is the exact identity and the split
   step is bit-identical to the pure rotation; with either time given alone,
   it is not.
4. `affine_propagate` reduces to `bloch_rotate` when $\mathbf R=\mathbf 0$
   and to `bloch_relax` when $\boldsymbol\Omega_{\mathrm{eff}}=\mathbf 0$,
   the latter across $\Delta t/T_2$ from $10^{-3}$ to $5$.
5. The Strang step converges to `affine_propagate` at second order, measured
   over three successive halvings of $\Delta t$ with $T_1\neq T_2$ and a
   tilted effective field.
6. Free precession at $\Delta f\,t=0.25$ carries $+x$ to $+y$, and to $-y$
   for the opposite offset sign, per Section 1.8 item 3. Values such as
   $\Delta f\,t=0.5$ must not be used: both phase conventions agree there.
7. Equilibrium is a fixed point; with RF off the long-time limit is
   $\mathbf M_{\mathrm{eq}}$; with RF left on it is the continuous-wave
   saturation steady state
   $M_z=M_0/(1+\omega_1^2T_1T_2)$, $M_y=-\omega_1T_2M_z$, $M_x=0$.
8. Non-positive relaxation times raise `ValueError` from both entry points.

### 3.8 Known limitations

Relaxation during a shaped pulse is second-order accurate, not exact, unless
$T_1=T_2$ or the field is purely longitudinal (Section 3.5). Users requiring
an exact answer for a single constant-field step should call the reference
propagator directly. Both $T_1$ and $T_2$ are scalar and isotropic; no
orientation or site dependence is modeled.

---

## 8. Gradients, spatial ensembles, and spoiling

**Status:** Draft completed 2026-09-14, implemented in `PULSIM/gradients.py`.

### 8.1 The reduction to offsets

A linear field gradient adds a position-dependent longitudinal field. In the
rotating frame this is indistinguishable from a resonance offset, so a
gradient requires **no new propagator**: it is a map from position to offset
applied to the input, and an average over positions applied to the output.
The existing batched kernel already propagates many offsets in one call.

$$
\Delta f(\mathbf r)=\Delta f_0+\bar\gamma\,(\mathbf G\cdot\mathbf r)
$$

Units follow Section 1.2 without conversion: $\bar\gamma$ in
$\mathrm{kHz\,mT^{-1}}$, $\mathbf G$ in $\mathrm{mT\,m^{-1}}$, $\mathbf r$ in
$\mathrm m$, so $\mathbf G\cdot\mathbf r$ is in mT and $\Delta f$ in kHz. The
sign convention of Section 1.5 carries over unchanged: a positive
$\Delta f(\mathbf r)$ produces a positive $\Omega_z$ at that position.

The phase accumulated over an interval of constant gradient is

$$
\varphi(\mathbf r,t)=2\pi\,\bar\gamma\,(\mathbf G\cdot\mathbf r)\,t .
$$

### 8.2 Scope

**Static or piecewise-constant gradients only.** `Pulse.apply` holds its
offset array fixed for every timestep, so a gradient varying *within* one
call is not representable and is not claimed. Piecewise-constant means one
propagator call per constant segment. Supporting an arbitrary gradient
waveform requires the propagator to accept timestep-dependent offsets, i.e.
`df` of shape `(n_steps, n_positions)`; until that exists the capability
must not be advertised.

Diffusion, flow, concomitant (Maxwell) fields, gradient nonlinearity and
eddy currents are outside the model.

### 8.3 Spatial sampling and the ensemble average

The observable is the volume average, not any individual isochromat:

$$
\bar{\mathbf M}(t)=\frac{1}{V}\int_V \mathbf M(\mathbf r,t)\,d^3r
\;\approx\;
\frac{\sum_j w_j\,\mathbf M(\mathbf r_j,t)}{\sum_j w_j}.
$$

For a one-dimensional slab of thickness $L$ centered at $c$, positions are
sampled at **cell midpoints**:

$$
z_j=c+L\left(\frac{j+\tfrac12}{N}-\frac12\right),\qquad j=0,\dots,N-1 .
$$

This choice is load-bearing, not cosmetic. Over $k$ complete twists the
sampled phases are $N$-th roots of unity, so their sum is **identically
zero** whenever $N\nmid k$; complete dephasing is then exact to machine
precision rather than leaving an $\mathcal O(1/N)$ residue. Endpoint
sampling double-counts one face and does not have this property.

A consequence to be aware of when reading offsets: the sampled offsets span
$\bar\gamma|\mathbf G|L\,(1-1/N)$, not the full $\bar\gamma|\mathbf G|L$.
The slab spans the whole range; the samples sit half a cell inside each face.

### 8.4 Dephasing of a uniform slab

With transverse magnetization initially uniform over the slab and a gradient
$g$ along $z$, the continuum average is

$$
\bar M_+(t)=M_+(0)\,\frac{\sin(\pi k)}{\pi k},
\qquad
k \equiv \bar\gamma\,g\,L\,t ,
$$

where $k$ is the number of complete phase twists across the slab. The finite
midpoint sample has its own exact closed form,

$$
\bar M_x = M_x(0)\,\frac{\sin(\pi k)}{N\sin(\pi k/N)},
\qquad \bar M_y = 0 ,
$$

which tends to the continuum result as $N$ grows and is real for all $k$.
Both forms vanish at integer $k$: **one complete winding cancels exactly.**

### 8.5 Refocusing

A gradient of $+\mathbf G$ for a duration $t$ followed by $-\mathbf G$ for
the same duration returns every isochromat to its starting phase, so
$\bar{\mathbf M}$ is restored exactly. With relaxation enabled the echo
returns attenuated but undistorted: $T_2$ scales every isochromat equally
regardless of phase, so the amplitude is $e^{-2t/T_2}$ and the direction is
unchanged.

Refocusing is exact here, not second-order, because the effective field
during such a delay is purely longitudinal and therefore commutes with
$\mathbf R$ (Section 3.5).

### 8.6 Ideal spoiling is an idealization, stated separately

Setting $M_x=M_y=0$ while leaving $M_z$ untouched is **not** what a crusher
gradient does, and the two must not be conflated:

| | physical dephasing | ideal spoiling |
|---|---|---|
| bulk $\bar{\mathbf M}_\perp$ | zero at integer $k$ | zero |
| per-isochromat $\lvert M_\perp\rvert$ | **unchanged** | zero |
| recoverable by a later gradient | **yes** | no |

Physical dephasing makes transverse magnetization invisible to the receiver;
it does not destroy it. Ideal spoiling destroys it. A specification that
offered only the second would silently give the wrong answer for any
sequence containing two gradients.

### 8.7 Implementation contract

| Requirement | Function |
|---|---|
| Offsets from a linear gradient | `gradients.gradient_offsets` |
| Midpoint slab sampling | `gradients.uniform_positions` |
| Ensemble average, optionally weighted | `gradients.ensemble_average` |
| Ideal spoiling (idealization) | `gradients.ideal_spoil` |
| Free evolution over a batch of offsets | `bloch.bloch_delay` |

`gradients.py` imports nothing but NumPy, so it is available on the
base install and in Pyodide.

### 8.8 Validation requirements

Implemented in `tests/test_gradients.py`; none is a golden test.

1. A gradient of $10\ \mathrm{mT\,m^{-1}}$ at $5\ \mathrm{mm}$ gives
   $\bar\gamma\times0.05\ \mathrm{kHz}$, checked as a hand-computable number.
2. Zero gradient gives zero offset; the offset is odd about the slab center;
   a $z$-gradient is blind to $x$ and $y$ displacement.
3. The sampled offset span is $\bar\gamma gL(1-1/N)$, asserting the
   midpoint convention.
4. Slab dephasing follows the continuum $\mathrm{sinc}$ law, and matches the
   exact discrete sum of Section 8.4 to machine precision.
5. At integer twists the bulk transverse magnetization vanishes to better
   than $10^{-12}$.
6. A gradient echo refocuses exactly, verified to be genuinely dephased
   in between; under $T_2$ it returns at $e^{-2t/T_2}$ with its phase intact.
7. Physical dephasing and ideal spoiling agree on the bulk vector and
   differ completely per isochromat.
8. `ideal_spoil` zeroes both transverse components and does not mutate its
   argument.

### 8.9 Known limitations

Beyond the exclusions of Section 8.2: the ensemble is a deterministic
quadrature over positions, not a stochastic simulation, so it captures
reversible dephasing only. $T_2^{*}$ arising from static field
inhomogeneity can be modeled by supplying the corresponding offset
distribution, but irreversible loss from molecular motion cannot.

---

## 9. Vendor shape files and declared intent

**Status:** Draft completed 2026-09-15, implemented in `PULSIM/bruker.py`.

### 9.1 Why the header matters

A Bruker/TopSpin shape file is not just a column of numbers. Above the
`##XYPOINTS=` block sits a header carrying the vendor's own statement of what
the pulse is for, what rotation it performs, its bandwidth factor, and — in
files written by ShapeTool — the full parameter set that generated the
waveform. PULSIM previously read the data block and discarded all of it.

This matters because one of those fields, `SHAPE_EXMODE`, is authoritative
where PULSIM was previously guessing. Section 6 of this document records a
heuristic, since deleted, that inferred adiabatic behaviour from a total
rotation above 350 degrees. That heuristic was wrong in both directions on
real files.

### 9.2 Two tiers, measured rather than assumed

Of the 205 entries in `wave/`: 203 are shape files, one is a directory, and
one (`Update_wave.info`) is not a shape file at all.

| Field group | Files | Content |
|---|---|---|
| `SHAPE_TOTROT`, `MODE`, `EXMODE`, `INTEGFAC`, `BWFAC` | 203 | universal |
| `SHAPE_TYPE`, `USER_DEF`, `REPHFAC`, `BWFAC50` | 155 | common |
| `SHAPE_PARAMETERS` | 141 | free-text design string |
| `SHL_*` | 18 | full ShapeTool design block |

Tier 1 is the first two rows: always present, always parseable. Tier 2 is the
`SHL_*` block. The specification makes no claim about fields outside these
groups.

### 9.3 Normalization is mandatory

The same semantic value occurs in four spellings in one corpus:

```
##$SHAPE_EXMODE= Excitation
##$SHAPE_EXMODE= <Excitation>
##$SHAPE_EXMODE= Excitation
##$SHAPE_EXMODE= Excitation\
```

A conforming reader must strip surrounding whitespace, a trailing
line-continuation backslash, and surrounding angle brackets, in that order.
`<>` normalizes to the empty string, which is how ShapeTool pads unused array
slots.

`##$SHAPE_EXMODE= None` is the literal string `None`, a real Bruker value
meaning "no excitation mode declared". It must not be conflated with the
field being absent.

`SHL_*` fields are not scalars. They are 16-slot arrays declared `(0..15)`
with elements on the following lines; slot 0 is the active segment and the
rest are padding. An array terminates at the next line beginning `##`,
which includes plain `##MINX=` lines, not only `##$` fields.

### 9.4 `SHAPE_EXMODE` and `SHAPE_TYPE` are different axes

```
EXMODE : Excitation 76 | None 44 | Adiabatic 44 | Inversion 10 | Universal 9
         Universal180 8 | Refocussing 4 | CompositeAdiabatic 3 | Decoupling 1 | BOP 1
TYPE   : Excitation 83 | Inversion 50 | Refocussing 18 | NoRotation 1
```

`Adiabatic`, `Decoupling` and `BOP` appear only in EXMODE; `NoRotation` only
in TYPE. The vendor separates *how a pulse works* from *what rotation it
performs*. PULSIM adopts the same separation: `RFShape.intent` labels only
the mechanism, and the rotation a pulse performs is expressed by its flip
angle, not by a label.

### 9.5 Intent is read from the file

$$
\texttt{intent} =
\begin{cases}
\texttt{"adiabatic"} & \texttt{SHAPE\_EXMODE} \in \{\texttt{Adiabatic},\ \texttt{CompositeAdiabatic}\}\\
\texttt{None} & \text{otherwise}
\end{cases}
$$

47 of the 203 files declare themselves adiabatic on this rule.

`FileShape` resolves intent by precedence: an explicit `intent=` argument
from the caller wins, so a user simulating a modified or mislabelled waveform
can contradict the file — including passing `intent=None` to clear the
file's claim. Otherwise the header decides. A file that is not a Bruker shape
file parses to an empty header and contributes no intent.

Because "the caller did not say" and "the caller said None" must be
distinguished, the default value of the argument is a sentinel rather than
`None`.

The two files that exposed the old heuristic now classify correctly from the
file alone:

| File | EXMODE | TYPE | intent | old heuristic |
|---|---|---|---|---|
| `Burbop-180.1` | BOP | Refocussing | None | adiabatic (wrong) |
| `BadCop1` | Inversion | Inversion | None | adiabatic (wrong) |
| `HypSec` | Adiabatic | Inversion | adiabatic | adiabatic |

### 9.6 Tier 2 closes the loop on the adiabatic design relations

A ShapeTool file carries both the design *inputs* and Bruker's *derived*
constants, so it is simultaneously the input and the oracle for Section 4's
design relations. For `wave/HypSec`:

| Quantity | In the file | PULSIM computes | Relative difference |
|---|---|---|---|
| `SHL_TRUNCLEV` | 1 % | (input) | — |
| `SHL_SW` | 20 Hz | (input) | — |
| `SHL_BETA` | 5.298292365610480 | $\operatorname{arccosh}(100/\text{trunclev})$ | $8.4\times10^{-16}$ |
| `SHL_MU` | 5.929443746783140 | $\pi\,\text{SW}/(2\beta)$ | $1.5\times10^{-16}$ |

Agreement is at the last bit of double precision. This is the strongest
available confirmation that PULSIM's adiabatic design relations are Bruker's.

### 9.7 `SHAPE_INTEGFAC` is a diagnostic, not ground truth

It is tempting to treat the vendor's stored integral as a cross-check on
`signed_integral_of`. Measured across the corpus, it is not one:

| Normalization used for PULSIM's integral | Files agreeing within $10^{-5}$ |
|---|---|
| by the waveform's own peak | 43 / 201 |
| by 100 | 88 / 201 |

Neither convention explains the corpus, and the two disagree in opposite
directions on different files: `zsqua005.10` matches only under
normalization by 100, `sin600.jfy` only under normalization by its own peak,
each to six digits. The likeliest reading — untested — is that the stored
value reflects the original design and is not recomputed when a file is later
rescaled, so for user-modified files it describes a waveform no longer
present in the file. Composite `.jfy` files are a separate case again:
PULSIM obtains approximately zero by phase cancellation where Bruker reports
0.63.

Implementations must expose this field for inspection and must not calibrate
from it. Conformance requires reporting the disagreement, not resolving it.

### 9.8 Consequence: adiabatic files refuse area calibration

A file declaring `SHAPE_EXMODE= Adiabatic` acquires
`calibration_mode == "adiabatic"`, and the area calibration of Section 4 is
then refused rather than applied. This is a deliberate behaviour change
affecting 47 files. A chirp has no meaningful pulse-area flip angle, and the
number the previous code returned was not a number about anything.

The amplitude may still be supplied directly, which is what a spectrometer
does: `Pulse(shape, nu1_max=...)`. In that case `realized_q` returns `None` —
the adiabaticity factor is genuinely unknown, because PULSIM cannot yet read
the design parameters of such a file. `None` states that; it does not
indicate a failure.

### 9.9 Implementation contract

| Requirement | Function |
|---|---|
| Split a header into scalar and array fields | `bruker.parse_header` |
| Normalize a field value | `bruker._normalize` |
| Read a file's header | `bruker.read_bruker_header` |
| Typed access, intent, tier-2 design | `bruker.BrukerHeader` |
| Intent resolution on import | `rf_shape.FileShape.__init__` |

`bruker.py` imports only the standard library, so it is available on the base
install and under Pyodide.

### 9.10 Validation requirements

Implemented in `tests/test_bruker.py`.

1. All four observed spellings of a value normalize identically; `<>` becomes
   the empty string; the literal `None` survives as a string.
2. Array fields do not leak into the scalar mapping, and terminate on a plain
   `##` line as well as on `##$`.
3. Every one of the 203 shape files parses without raising and yields an
   exmode; `Update_wave.info` yields an empty header rather than an error.
4. A census of the corpus — 203 with an exmode, 47 adiabatic, 18 with a
   design block — asserted exactly, as a tripwire against `wave/` changing
   without anyone noticing.
5. Every tier-2 design block carries `mu`, `beta`, `sw`, `trunclev`,
   `length`, `npoints`.
6. The `HypSec` round trip of Section 9.6, to a relative tolerance of
   $10^{-12}$.
7. `FileShape` takes its intent from the file for `HypSec`, a chirp,
   `Burbop-180.1` and `BadCop1`; an explicit `intent=` overrides in both
   directions.
8. Area calibration of an adiabatic file raises, with a message naming
   `nu1_max`; supplying `nu1_max` works and yields `realized_q is None`.

### 9.11 Not yet implemented

`SHAPE_PARAMETERS` is a free-text design string present in 141 files,
including 43 of the 47 adiabatic ones — far more than the 18 carrying a
tier-2 block, and therefore the practical route to calibrating imported
adiabatic pulses:

```
Type: SmoothedChirp ; Total Sweep-Width [Hz] 100000.0 ;
Length of Pulse [usec] 500.0 ; % to be smoothed 20.0
```

Each pulse family names its parameters differently, so parsing it is a
separate piece of work and is not claimed here. Until it exists, an imported
adiabatic pulse must be given its amplitude explicitly. Two files
(`Bip720,100,10.1`, `Bip720,50,20.1`) carry neither a design block nor
`SHAPE_PARAMETERS` and will always require it.
