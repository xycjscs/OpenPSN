# Introduction

Two deterministic paradigms dominate light-water-reactor core
calculations. *Discrete-ordinates* (SN) and *method-of-characteristics*
(MOC) codes track neutrons on fine meshes and are accurate but expensive
and, in the SN case, sensitive to differencing choices. *Nodal*
diffusion codes work on coarse meshes and are cheap and robust, but
classical diffusion under-predicts the transport corrections (shielding,
pin-wise self shielding) that control $\kappa_{\mathrm{eff}}$ and power
shape.

The **Phase Space Nodal (PSN)** method [@psn2027] bridges the two.
Within each node the scalar flux is represented by a separable
hyperbolic profile in the two axes, plus a five-term quadratic
particular solution of the source, whose angular dependence is carried
through a small set of phase-space (angular) moments with closed-form
correction coefficients. The node is eliminated to give explicit
relations between surface currents and node sources, and neighbouring
nodes are coupled through those currents. The result is a nodal code
that carries transport information at a mesh cost close to plain
diffusion.

Two gaps remained. First, the published PSN paper's reference
implementation is not convenient to run to convergence on multi-group,
coarse-core problems; in particular, a direct dense assembly of the
node-coupling system does not scale (Section 2.2). Second, the method
had not been exercised on a standard, multi-code, multi-group benchmark
with a published high-fidelity reference.

In this work we (i) re-implement PSN as a clean, multi-group,
YAML-driven Python package (**OpenPSN**), with sparse assembly and
vectorised node operators that agree with the validated loop form to
$10^{-15}$ (floating-point rounding); (ii) verify the implementation by
reproducing the published PSN results on the checkerboard and BWR bundle
problems; (iii) derive a directional angular-moment extension of the
nodal closure: the odd-parity flux is eliminated exactly before the
even-parity equation is projected onto piecewise Legendre modes in
azimuth, and the resulting matrix-valued diffusion coefficients define
the local body response and face currents while the spatial trial family
stays separated; and (iv) validate on the C5G7 MOX benchmark and the
published checkerboard, comparing against directly computed OpenMC
references (both homogenised and explicit-pin geometry) and the full
published code landscape for C5G7-2D.

The remainder is organised as follows. Section 2 summarises the PSN
formulation and the specific implementation choices, including the new
rectangular-node extension. Section 3 documents the reproduction of the
published results. Section 4 describes the C5G7 configuration and three
benchmark studies: (i) the pinwise-homogenised nodal model with a full
spatial/angular ($S\times M$) convergence sweep against directly
computed OpenMC references; (ii) the rectangular-node model with
explicit (unhomogenised) intra-node fuel/water structure under the
original $A{=}0$ angular closure; and (iii) the same rectangular-node
model closed at the $A{=}3$ directional angular order, including flux
comparisons against OpenMC and an eigenvalue comparison with the
published code landscape. Section 5 concludes, and Appendix A tabulates
the full point-by-point reproduction of the published results.

# The PSN method and the OpenPSN implementation

## Method in brief

The full theory is given in Ref. [@psn2027]; we summarise only what is
needed to read the implementation. The angular flux within a node is
expanded in a small basis of angular moments; the scalar flux is taken
as a separable hyperbolic profile in space (plus the five-term source
particular solution) whose coefficients are fixed by the node-averaged
flux and a set of surface-current (moment) unknowns. Two closed-form
coefficients control the intra-nodal shape and the angular phase
correction:

$$\begin{align}
\alpha_{imn} &\quad\text{(face-flux weight, }\overline{\Phi}_{imn}
=\alpha_{imn}\,\overline{\phi}_{im}^{\,n}\text{, Eq.~(2.8a) of
Ref.~\cite{psn2027}),}\\
\beta_i  &= 1-\tfrac{1}{4}\cos\Delta\theta_i-\tfrac{1}{4}\cos 2\theta_i
          -\tfrac{1}{2}\cos 2\theta_i\,\cos\Delta\theta_i
          \quad\text{(generic phase-space coefficient, Eq.~2.8d).}
\end{align}$$

The closed form for $\beta_i$ in Eq. (2.8d) is an equivalent closed-form
expression re-derived from the integral form of the phase-space moment,
and the two are algebraically identical; we verified the two to
$2.2\times10^{-16}$ and use the closed form in the implementation. In
the fine-angle limit the coefficient reduces to the single-direction P1
closure $(3/2)\sin^2\theta_i$.

Eliminating the node unknowns yields a sparse system coupling the
surface-current (and node-source) unknowns of neighbouring nodes, which
is iterated with a standard fission-source (power) iteration; the sparse
node-coupling system is solved directly at each outer sweep.

## Implementation and the high-precision formulation

OpenPSN (run as `python -m psn2d run problem.yaml`) is a pure-Python
implementation (NumPy/SciPy) accepting any number of energy groups, an
arbitrary material grid with independent per-edge reflect/vacuum
boundary conditions, and a user-set angular order $M$, spatial
sub-division $S$, angular model, and convergence tolerance. The four
implementation components are:

1.  **Sparse assembly (removes the quadratic growth).** The nodal
    coupling matrix has only $\sim$`<!-- -->`{=html}9 non-zeroes per row
    (the node couples to its own face-current unknowns and to its
    neighbours), so the assembly scales linearly in the node count where
    the natural dense form scales quadratically. We assemble in
    coordinate (COO) format and solve in compressed-sparse-column (CSC)
    format; the path agrees with the plain dense reference to
    $10^{-15}$, and the $10{,}404$-node refinement ($250{,}920$
    unknowns) converges to $10^{-10}$ in the relative eigenvalue change.

2.  **Vectorised node operators, verified to $10^{-15}$.** The
    node-state map is strictly linear in the $(J_4,q)$ unknowns. We
    probe it with nine basis vectors once to build the $9\times9$ linear
    map, then apply it as a batched matrix multiply over all nodes and
    all groups. The vectorised path is verified identical to the
    validated scalar loop to $10^{-15}$ on five representative problems
    (the shipped verification script `verify_vectorized_equiv.py`): it
    changes no number of the validated solver.

3.  **Rectangular nodes with per-node widths (no homogenisation).** The
    original formulation carries a single material per node, with no
    intra-node structure, so a pin cell enters either as a homogenised
    node or through an explicit multi-node geometry. OpenPSN extends the
    node operators to arbitrary per-node widths: each rectangular node
    carries its own width in $x$ and $y$, and nodes meet
    corner-to-corner only (no T-junctions). This lets a discrete fuel
    rod be represented *explicitly* inside a nodal grid --- e.g. the
    C5G7 1.26 cm circular rod is replaced by an equal-area
    $s=\sqrt{\pi r^{2}}=0.95713$ cm fuel square centred on the same
    pitch, giving a $3\times3$ rectangle per pin and a globally
    edge-aligned tiling (17$\times$`<!-- -->`{=html}17 assembly
    $=51\times51$ nodes; 1/4 core $=153\times153$). Un-homogenised
    7-group cross sections can then be assigned per rectangle (fuel
    vs. water), and the homogenisation model --- the usual source of
    nodal bias on MOX benchmarks --- is removed from the problem
    entirely. The rectangular node operators are the same closed-form
    parabolic/phase-space forms as the square case evaluated on the
    actual node widths; they are verified node-by-node against the
    square-node path (which they reduce to exactly when all widths are
    equal).

4.  **High-order directional moment closure ($A$-order angular
    fidelity).** The nodal moment closure is generalised to an arbitrary
    polynomial order $A$ ($K{=}A{+}1$ face modes per direction): the
    original-scale branch at $A{=}0$ reproduces the legacy state matrix
    element-for- element (dedicated regression; the directional branch
    at $A{=}0$ is the distinct segment-resolved variant of Section 4.4),
    and $A{=}1,2,3$ extend it to higher angular fidelity without
    changing any other part of the solver. Each $(i,g)$ node system is
    then a fixed sparse linear system in $K$ face-mode unknowns per face
    ($4K$ per node), so the high-order closure converges every point to
    the same $10^{-7}$ gate as the original method. The accuracy of the
    assembly is certified against an independent full-mesh direct solve:
    at full convergence the keffs agree to ${\le}2\times10^{-14}$ for
    $A{=}1,2,3$.

These are the implementation components behind the results that follow:
not a change to the physics, but the choices that (a) make the published
method reproducible to machine precision and (b) make it converge to
$10^{-7}$--$10^{-10}$ in relative eigenvalue change on multi-group
coarse cores, at any subdivision the dense formulation cannot address.

# Reproduction of the published PSN results

Before any new application we verify the implementation against the
source paper [@psn2027]. We reproduce the two checkerboard eigenvalue
convergence tables (Fig. 3, weak/strong absorption) and the BWR 12-pin
bundle tables (Figs. 10/12, with and without Gd) to within a few pcm.
Table [1](#tab:repro){reference-type="ref" reference="tab:repro"}
summarises the maximum deviations.

  Published problem                       points   max $|\Delta|$ (pcm)
  -------------------------------------- -------- ----------------------
  Fig. 3 checkerboard, weak ($I=30$)        60             1.2
  Fig. 3 checkerboard, strong ($I=30$)      60             1.4
  Fig. 10 BWR bundle, no Gd (2g)            60             3.8
  Fig. 12 BWR bundle, with Gd (2g)          60     within plot reading

  : Reproduction of the published PSN paper (Ref. [@psn2027]). "err" is
  the absolute difference
  $(\kappa_{\mathrm{eff}}-\kappa_{\mathrm{eff}}^{\mathrm{ref}})\times10^5$
  in pcm. The reference values are the OpenMC values used in the
  paper [@openmc2015]. {#tab:repro}

All 248 data points are reproduced --- the 240 cells of
Tables [5](#tab:appA){reference-type="ref"
reference="tab:appA"}--[8](#tab:appD){reference-type="ref"
reference="tab:appD"} plus the 8 restricted $M{=}24$ points listed in
Appendix A. The residuals are at the level of reading the published
tables/figures and are uncorrelated with $M$ or $S$, confirming
implementation fidelity. The full point-by-point comparison is tabulated
in Appendix A (Tables [5](#tab:appA){reference-type="ref"
reference="tab:appA"}--[8](#tab:appD){reference-type="ref"
reference="tab:appD"}); the complete per-point data are shipped with the
code.

# C5G7 MOX benchmark

## Configuration

C5G7 is the OECD Nuclear Energy Agency (NEA) mixed-oxide (MOX)
benchmark [@nea2003; @smith2004], the standard test for uranium-oxide
(UOX)/MOX inter-cell transport: 17$\times$`<!-- -->`{=html}17-pin fuel
assemblies on a 1.26 cm pin pitch (UO$_2$ plus three MOX enrichments),
guide tubes and fission chambers, in a 7-group structure. (The C5G7
family was later extended to time-dependent calculations [@c5g7td]; we
solve the steady 2-D problem.) We solve the **C5G7-2D 1/4-core**
problem: a 2$\times$`<!-- -->`{=html}2 block of fuel assemblies (UO2 on
the diagonal, MOX on the anti-diagonal) plus an L-shaped one-assembly
water reflector, 51$\times$`<!-- -->`{=html}51 pins, 64.26 cm on a side.
Following Ref. [@nea2003] we apply *vacuum* boundaries on the two open
(right, top) edges and *reflective* boundaries on the two symmetry
(left, bottom) edges. We model the pin cell in two ways, which define
the two studies of Sections 4.2--4.3: *(i)* the standard pin-homogenised
nodal model, in which each pin cell is a single homogenised node
(fuel$+$water for fuel pins; pure water for reflector and guide pins),
and *(ii)* the rectangular-node model of Section 2.2, in which each pin
cell is a $3\times3$ array of rectangles carrying the un-homogenised
fuel/water materials. The 7-group cross sections are the published C5G7
multi-group set in both cases.

Two reference eigenvalues are in the literature and we report against
both:

- $\kappa_{\mathrm{eff}}^{\mathrm{MCNP}}=1.18655$ ($\pm0.008\%$), the
  2-D multi-group MCNP value in the original NEA
  report [@nea2003; @smith2004];

- $\kappa_{\mathrm{eff}}=1.18646$ (1.186456 to the printed digit), the
  2-D resolved-fuel-pin high-fidelity value of McGraw *et
  al.* [@mcgraw2014], quoted as the reference in the INL Rattlesnake
  convergence study [@rattlesnake], which explicitly notes that the
  benchmark-report value 1.18655 is $\sim$`<!-- -->`{=html}10 pcm high
  and converges to $\kappa_{\mathrm{eff}}=1.186446$
  ($\sim$`<!-- -->`{=html}1 pcm from the McGraw reference).

We report all deviations against 1.18646 (the more accurate
high-fidelity value). The original NEA report tabulated relative errors
against its MCNP reference 1.18655; re-referenced to 1.18646, those
values shift by $\sim$`<!-- -->`{=html}9 pcm.

## Study I --- the pinwise-homogenised nodal model

The standard nodal treatment folds each pin's fuel and moderator into a
single homogenised node. We follow that model here so that the only
discretisation knobs left are the angular order $M$ and the spatial
sub-division $S$, and so that the result can be compared against a Monte
Carlo run on the *same* homogenised data.

#### Pinwise homogenisation.

The homogenised 7-group cross sections are built pin by pin with the
**flux-weighted self-shielding** recipe: an infinite-medium pin cell
($r=0.54$ cm, 1.26 cm pitch) is solved with an explicit fuel/water MOC
calculation, and the fuel-group cross sections are weighted by the
self-shielded infinite-medium flux (scattering by the source-group
flux). Guide tubes are area-weighted. This is the same
self-shielding-aware homogenisation that nodal practice uses in
production; it removes most of the resonance self-shielding error that a
plain area-weighted average leaves behind.

#### Monte Carlo reference, computed directly.

Rather than lean on a published reference for the homogenised model, we
compute the reference ourselves with OpenMC [@openmc2015] on the exact
same pinwise-homogenised cross sections, so that the two sides differ
only in the transport method. Each calculation uses 20000 active batches
of 100000 particles (collision estimator); the eigenvalue is the mean
over the last 20000 batches and its standard error is
$\sigma_{\mathrm{batch}}/\sqrt{N}$, which we audit to be
$\approx$`<!-- -->`{=html}3.4--3.5 pcm. We run the single UO$_2$
assembly, the single MOX assembly and the 1/4 core, each with
*homogenised* (pin-node) and, where needed, *explicit* (resolved
fuel/water) geometry. The three *homogenised* runs are the reference for
the pinwise-homogenised PSN sweep below:

- single UO$_2$ assembly: $1.3336285$ ($\pm3.5$ pcm),

- single MOX assembly: $1.1853993$ ($\pm3.3$ pcm),

- 1/4 core: $1.1868230$ ($\pm3.4$ pcm).

#### OpenPSN $S\times M$ sweep.

We sweep $S\in\{1,\dots,6\}$ and $M\in\{2,4,8,12,16,24\}$ on all three
geometries (nodes $(17S)^2$ for both assemblies, $(51S)^2$ for the
core), with the pinwise-homogenised C5G7 cross sections, and converge
every point to $10^{-10}$ in the relative eigenvalue change --- all 108
points of Table [2](#tab:bww){reference-type="ref" reference="tab:bww"}.
The largest system (core, $S{=}6$, $93{,}636$ nodes) is solved with the
sparse direct factorisation shipped with OpenPSN, verified to $10^{-15}$
against the plain reference path.

+---------+---------+---------+---------+----------+----------+----------+
|         | $M{=}2$ | $M{=}4$ | $M{=}8$ | $M{=}12$ | $M{=}16$ | $M{=}24$ |
+:========+========:+========:+========:+=========:+=========:+=========:+
| *Single UO$_2$ assembly (ref. $1.3336285$)*                            |
+---------+---------+---------+---------+----------+----------+----------+
| $S{=}1$ | $+13.9$ | $-15.0$ | $-22.4$ | $-23.4$  | $-23.7$  | $-23.8$  |
+---------+---------+---------+---------+----------+----------+----------+
| $S{=}2$ | $+18.8$ | $-8.0$  | $-15.2$ | $-15.3$  | $-15.0$  | $-14.5$  |
+---------+---------+---------+---------+----------+----------+----------+
| $S{=}3$ | $+22.6$ | $-0.3$  | $-6.3$  | $-5.8$   | $-5.3$   | $-4.6$   |
+---------+---------+---------+---------+----------+----------+----------+
| $S{=}4$ | $+24.2$ | $+3.0$  | $-2.5$  | $-2.0$   | $-1.4$   | $-0.6$   |
+---------+---------+---------+---------+----------+----------+----------+
| $S{=}5$ | $+25.0$ | $+4.6$  | $-0.6$  | $+0.0$   | $+0.6$   | $+1.5$   |
+---------+---------+---------+---------+----------+----------+----------+
| $S{=}6$ | $+25.4$ | $+5.5$  | $+0.6$  | $+1.2$   | $+1.8$   | $+2.7$   |
+---------+---------+---------+---------+----------+----------+----------+
| *Single MOX assembly (ref. $1.1853993$)*                               |
+---------+---------+---------+---------+----------+----------+----------+
| $S{=}1$ | $-65.1$ | $-53.3$ | $-30.1$ | $-22.3$  | $-18.6$  | $-15.3$  |
+---------+---------+---------+---------+----------+----------+----------+
| $S{=}2$ | $-62.2$ | $-49.8$ | $-26.1$ | $-16.6$  | $-12.3$  | $-8.4$   |
+---------+---------+---------+---------+----------+----------+----------+
| $S{=}3$ | $-60.3$ | $-45.8$ | $-22.4$ | $-12.5$  | $-8.4$   | $-4.4$   |
+---------+---------+---------+---------+----------+----------+----------+
| $S{=}4$ | $-59.6$ | $-44.2$ | $-20.7$ | $-10.9$  | $-6.7$   | $-2.7$   |
+---------+---------+---------+---------+----------+----------+----------+
| $S{=}5$ | $-59.2$ | $-43.5$ | $-19.8$ | $-10.2$  | $-6.0$   | $-1.9$   |
+---------+---------+---------+---------+----------+----------+----------+
| $S{=}6$ | $-59.0$ | $-43.0$ | $-19.3$ | $-9.8$   | $-5.5$   | $-1.5$   |
+---------+---------+---------+---------+----------+----------+----------+
| *C5G7-2D 1/4 core (ref. $1.1868230$)*                                  |
+---------+---------+---------+---------+----------+----------+----------+
| $S{=}1$ | $-97.6$ | $-59.5$ | $-37.1$ | $-31.3$  | $-28.8$  | $-26.7$  |
+---------+---------+---------+---------+----------+----------+----------+
| $S{=}2$ | $-96.3$ | $-56.1$ | $-33.6$ | $-26.6$  | $-23.6$  | $-20.9$  |
+---------+---------+---------+---------+----------+----------+----------+
| $S{=}3$ | $-93.8$ | $-50.1$ | $-26.8$ | $-19.3$  | $-16.2$  | $-13.3$  |
+---------+---------+---------+---------+----------+----------+----------+
| $S{=}4$ | $-92.8$ | $-47.6$ | $-23.9$ | $-16.3$  | $-13.2$  | $-10.2$  |
+---------+---------+---------+---------+----------+----------+----------+
| $S{=}5$ | $-92.3$ | $-46.3$ | $-22.3$ | $-14.8$  | $-11.6$  | $-8.6$   |
+---------+---------+---------+---------+----------+----------+----------+
| $S{=}6$ | $-92.0$ | $-45.6$ | $-21.4$ | $-13.9$  | $-10.7$  | $-7.7$   |
+---------+---------+---------+---------+----------+----------+----------+

: Pinwise-homogenised OpenPSN (self-shielded, flux-weighted pin XS),
$\Delta$ in pcm versus the OpenMC *homogenised* references: UO$_2$
assembly vs. $1.3336285$, MOX assembly vs. $1.1853993$, 1/4 core
vs. $1.1868230$ (each $\approx$`<!-- -->`{=html}3.3--3.5 pcm standard
error). All 108 points; the minimum-\|$\Delta$\| point per geometry is
marked in Fig. [1](#fig:bww){reference-type="ref" reference="fig:bww"}.
{#tab:bww}

Three features stand out (Fig. [1](#fig:bww){reference-type="ref"
reference="fig:bww"}). *Both assemblies:* the $M{=}2$ column sits high
and positive on UO$_2$ ($+14\ldots+25$ pcm, the angular cliff) and low
on MOX ($-65\ldots-59$ pcm), while the $M\ge4$ region forms a floor that
climbs monotonically with $S$ back toward zero on both; the best points
of the two assemblies ($UO_2$: $S{=}5,M{=}12$, $+0.0$ pcm; MOX:
$S{=}6,M{=}24$, $-1.5$ pcm) sit inside their $\pm3.3$--$3.5$ pcm
statistical uncertainty. *Core:* the $M{=}2$ column is pinned low
($-98\ldots-92$ pcm) and the error recovers monotonically in both $S$
and $M$; the best attainable point ($S{=}6,M{=}24$, $-7.7$ pcm) still
sits outside the $\pm3.4$ pcm reference standard error --- a small,
smooth, method-level bias that we check against the fission field below.

<figure id="fig:bww" data-latex-placement="H">
<img src="c5g7_bww_sweep_heatmap.png" style="width:98.0%" />
<figcaption>Pinwise-homogenised OpenPSN
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi mathvariant="normal">Δ</mi><msub><mi>κ</mi><mrow><mi mathvariant="normal">e</mi><mi mathvariant="normal">f</mi><mi mathvariant="normal">f</mi></mrow></msub></mrow><annotation encoding="application/x-tex">\Delta\kappa_{\mathrm{eff}}</annotation></semantics></math>
in pcm versus the OpenMC homogenised reference, as a function of angular
order
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>M</mi><annotation encoding="application/x-tex">M</annotation></semantics></math>
(horizontal) and spatial sub-division
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>S</mi><annotation encoding="application/x-tex">S</annotation></semantics></math>
(vertical). Left: single
UO<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><msub><mi></mi><mn>2</mn></msub><annotation encoding="application/x-tex">_2</annotation></semantics></math>
assembly
(ref. <math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>1.3336285</mn><annotation encoding="application/x-tex">1.3336285</annotation></semantics></math>);
centre: C5G7-2D 1/4 core
(ref. <math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>1.1868230</mn><annotation encoding="application/x-tex">1.1868230</annotation></semantics></math>);
right: single MOX assembly
(ref. <math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>1.1853993</mn><annotation encoding="application/x-tex">1.1853993</annotation></semantics></math>).
All 36 points per panel; green ring =
minimum-|<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi mathvariant="normal">Δ</mi><annotation encoding="application/x-tex">\Delta</annotation></semantics></math>|
point of each panel.</figcaption>
</figure>

#### Flux distribution at the best points.

Figure [2](#fig:bwwbest){reference-type="ref" reference="fig:bwwbest"}
compares the pin-averaged fission source $\nu\Sigma_f\phi$ of each
geometry's best point against the OpenMC homogenised-geometry reference
on the same homogenised C5G7 cross sections (the two sides differ only
in the transport method; the OpenMC field is the converged 7-group flux
contracted with the homogenised $\nu\Sigma_f$ --- the C5G7 library's
fission cross-section slot carries absorption, so the true fission
source is reconstructed from the flux, identically on both assemblies
and the core --- making both maps the same quantity, each normalised to
the unit fuel sum). On fuel pins the pin-by-pin relative error is
$0.03\%$ RMS (max $0.08\%$) on the UO$_2$ assembly, $0.04\%$ RMS (max
$0.10\%$) on the MOX assembly and $0.07\%$ RMS (max $0.25\%$) over all
$1056$ fuel pins of the 1/4 core. The fields are smooth and slowly
varying across the assembly; a small power-distribution RMS of this kind
does not by itself rule out partial cancellation of eigenvalue errors.

<figure id="fig:bwwbest" data-latex-placement="H">
<img src="c5g7_bww_best_fission.png" style="width:98.0%" />
<figcaption>Pinwise-homogenised best
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>×</mo><mi>M</mi></mrow><annotation encoding="application/x-tex">S\times M</annotation></semantics></math>
point per geometry — top:
UO<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><msub><mi></mi><mn>2</mn></msub><annotation encoding="application/x-tex">_2</annotation></semantics></math>
assembly,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>=</mo><mn>5</mn><mo>,</mo><mi>M</mi><mo>=</mo><mn>12</mn></mrow><annotation encoding="application/x-tex">S{=}5,M{=}12</annotation></semantics></math>,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi mathvariant="normal">Δ</mi><mi>k</mi><mo>=</mo><mi>+</mi><mn>0.0</mn></mrow><annotation encoding="application/x-tex">\Delta k=+0.0</annotation></semantics></math> pcm;
middle: MOX assembly,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>=</mo><mn>6</mn><mo>,</mo><mi>M</mi><mo>=</mo><mn>24</mn></mrow><annotation encoding="application/x-tex">S{=}6,M{=}24</annotation></semantics></math>,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi mathvariant="normal">Δ</mi><mi>k</mi><mo>=</mo><mi>−</mi><mn>1.5</mn></mrow><annotation encoding="application/x-tex">\Delta k=-1.5</annotation></semantics></math> pcm;
bottom: 1/4 core,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>=</mo><mn>6</mn><mo>,</mo><mi>M</mi><mo>=</mo><mn>24</mn></mrow><annotation encoding="application/x-tex">S{=}6,M{=}24</annotation></semantics></math>,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi mathvariant="normal">Δ</mi><mi>k</mi><mo>=</mo><mi>−</mi><mn>7.7</mn></mrow><annotation encoding="application/x-tex">\Delta k=-7.7</annotation></semantics></math> pcm,
cropped to the
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2</mn><mo>×</mo><mn>2</mn></mrow><annotation encoding="application/x-tex">2\times2</annotation></semantics></math>
fuel block (the outer reflector-water region is omitted): pin-averaged
fission source
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>ν</mi><msub><mi mathvariant="normal">Σ</mi><mi>f</mi></msub><mi>ϕ</mi></mrow><annotation encoding="application/x-tex">\nu\Sigma_f\phi</annotation></semantics></math>,
OpenPSN (left), OpenMC same-geometry reference (centre), pin-by-pin
relative error on fuel pins (right; both maps
fuel-normalised).</figcaption>
</figure>

## Study II --- the rectangular-node model with explicit intra-node structure

#### Motivation and layout.

Study I isolates the discretisation error of the nodal PSN model, but
its eigenvalue is still anchored to whatever homogenisation model feeds
it. To remove that model from the problem we use the rectangular-node
extension of Section 2.2: the circular C5G7 fuel rod ($r=0.54$ cm) is
replaced by an equal-area fuel square of side
$s=\sqrt{\pi r^{2}}=0.95713$ cm centred on the same 1.26 cm pitch, so
that every pin cell becomes a $3\times3$ array of rectangles (one fuel
square plus eight water strips) and the whole core is a globally
edge-aligned rectangular tiling.
Figure [3](#fig:layout){reference-type="ref" reference="fig:layout"}
shows the layout. The pin pitch, the rod positions (including Gd pins,
which are the centre rectangle of their $3\times3$ block) and the total
fuel area are unchanged; the layout was verified node by node --- all
2601 pin-cell centre rectangles of the 1/4 core coincide with the 2601
pin cells of the homogenised grid, and the total area reproduces
$4129.35~\mathrm{cm}^2$ exactly. Un-homogenised 7-group cross sections
(fuel, water, Gd, MOX) are then assigned directly to the rectangles. Two
discretisation knobs are now open: the angular order $M$ and the uniform
spatial subdivision $S$ (Section 2.2), and we sweep both.

<figure id="fig:layout" data-latex-placement="H">
<img src="c5g7_rect_layout.png" style="width:98.0%" />
<figcaption>Rectangularisation of the C5G7 pin cell. (a) Original
circular rod in its pitch cell (water gap 0.09 cm). (b) Equal-area fuel
square
(<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>s</mi><mo>=</mo><mn>0.95713</mn></mrow><annotation encoding="application/x-tex">s=0.95713</annotation></semantics></math> cm,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msup><mi>s</mi><mn>2</mn></msup><mo>=</mo><mi>π</mi><msup><mi>r</mi><mn>2</mn></msup></mrow><annotation encoding="application/x-tex">s^2=\pi r^2</annotation></semantics></math>)
with the pin cell split into
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>3</mn><mo>×</mo><mn>3</mn></mrow><annotation encoding="application/x-tex">3\times3</annotation></semantics></math>
rectangles; the four corner water blocks of neighbouring pins meet at a
single node corner, so the tiling has no T-junctions. (c) Global tiling:
a
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>17</mn><mo>×</mo><mn>17</mn></mrow><annotation encoding="application/x-tex">17\times17</annotation></semantics></math>
assembly is
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>51</mn><mo>×</mo><mn>51</mn><mo>=</mo><mn>2601</mn></mrow><annotation encoding="application/x-tex">51\times51=2601</annotation></semantics></math>
rectangles and the 1/4 core is
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>153</mn><mo>×</mo><mn>153</mn><mo>=</mo><mn>23409</mn></mrow><annotation encoding="application/x-tex">153\times153=23409</annotation></semantics></math>.</figcaption>
</figure>

#### $S\times M$ convergence of the $3\times3$ tiling.

Table [3](#tab:rectsm){reference-type="ref" reference="tab:rectsm"} and
Figure [4](#fig:rectsm){reference-type="ref" reference="fig:rectsm"}
give the full $S\times M$ sweep (UO$_2$/MOX assemblies: $S{=}1\ldots6$;
1/4 core: $S{=}1,2$, the $S{=}3$ core grid being $9\times$ larger in
node count and only covered to $M{=}8$; un-homogenised material XS,
every point converged to $10^{-10}$ in relative eigenvalue change
$|\Delta k|/k$) against the directly computed OpenMC run on the
*identical* rectangular-rod geometry (100k particles $\times$ 20200
batches, collision estimator, fixed seed): the single UO$_2$ assembly
($51\times51$ rect. nodes, all edges reflective, ref. $k=1.3347260$),
the MOX assembly (same size, ref. $k=1.1851053$), and the 1/4 core
($153\times153$ rect. nodes, reflective on the symmetry edges, vacuum on
the open edges, ref. $k=1.1872176$).

+---------+----------+----------+----------+----------+----------+----------+
|         | $M{=}2$  | $M{=}4$  | $M{=}8$  | $M{=}12$ | $M{=}16$ | $M{=}24$ |
+:========+=========:+=========:+=========:+=========:+=========:+=========:+
| *UO$_2$ assembly*                                                         |
+---------+----------+----------+----------+----------+----------+----------+
| $S{=}1$ | $+106.6$ | $-43.9$  | $-101.0$ | $-69.3$  | $-37.2$  | $+1.2$   |
+---------+----------+----------+----------+----------+----------+----------+
| $S{=}2$ | $+93.1$  | $-71.7$  | $-132.6$ | $-99.3$  | $-66.9$  | $-28.1$  |
+---------+----------+----------+----------+----------+----------+----------+
| $S{=}3$ | $+90.2$  | $-74.0$  | $-136.9$ | $-108.4$ | $-80.2$  | $-45.2$  |
+---------+----------+----------+----------+----------+----------+----------+
| $S{=}4$ | $+89.2$  | $-74.3$  | $-138.4$ | $-111.2$ | $-85.8$  | $-51.9$  |
+---------+----------+----------+----------+----------+----------+----------+
| $S{=}5$ | $+89.0$  | $-74.1$  | $-138.8$ | $-111.9$ | $-88.0$  | $-55.0$  |
+---------+----------+----------+----------+----------+----------+----------+
| $S{=}6$ | $+88.9$  | $-74.1$  | $-139.0$ | $-112.6$ | $-89.0$  | $-56.5$  |
+---------+----------+----------+----------+----------+----------+----------+
| *MOX assembly*                                                            |
+---------+----------+----------+----------+----------+----------+----------+
| $S{=}1$ | $-893.6$ | $-769.8$ | $-503.5$ | $-319.0$ | $-207.9$ | $-98.9$  |
+---------+----------+----------+----------+----------+----------+----------+
| $S{=}2$ | $-885.5$ | $-754.9$ | $-464.9$ | $-265.5$ | $-148.3$ | $-33.5$  |
+---------+----------+----------+----------+----------+----------+----------+
| $S{=}3$ | $-889.5$ | $-760.7$ | $-488.3$ | $-304.3$ | $-198.3$ | $-92.2$  |
+---------+----------+----------+----------+----------+----------+----------+
| $S{=}4$ | $-891.2$ | $-762.8$ | $-498.0$ | $-316.9$ | $-216.8$ | $-111.2$ |
+---------+----------+----------+----------+----------+----------+----------+
| $S{=}5$ | $-891.9$ | $-763.9$ | $-502.6$ | $-322.9$ | $-226.0$ | $-122.4$ |
+---------+----------+----------+----------+----------+----------+----------+
| $S{=}6$ | $-892.3$ | $-764.6$ | $-505.1$ | $-327.4$ | $-231.0$ | $-128.4$ |
+---------+----------+----------+----------+----------+----------+----------+
| *1/4 core*                                                                |
+---------+----------+----------+----------+----------+----------+----------+
| $S{=}1$ | $-254.3$ | $-246.6$ | $-184.3$ | $-110.9$ | $-61.1$  | $-10.2$  |
+---------+----------+----------+----------+----------+----------+----------+
| $S{=}2$ | $-269.5$ | $-266.7$ | $-201.0$ | $-122.7$ | $-71.3$  | $-18.8$  |
+---------+----------+----------+----------+----------+----------+----------+

: Rectangular-node OpenPSN, $3\times3$ pin tiling: UO$_2$/MOX assemblies
at $S{=}1\ldots6$, 1/4 core at $S{=}1,2$ (at $S{=}3$ the core grid grows
$9\times$, only $M{=}2,4,8$ were computed). $\Delta k$ in pcm versus the
same-geometry pure-XS OpenMC reference (UO$_2$ assembly $1.3347260$, MOX
assembly $1.1851053$, 1/4 core $1.1872176$). Green marker in
Figure [4](#fig:rectsm){reference-type="ref" reference="fig:rectsm"} =
smallest $|\Delta k|$. {#tab:rectsm}

<figure id="fig:rectsm" data-latex-placement="H">
<img src="rect_b4_sm_heatmap.png" style="width:98.0%" />
<figcaption>Same data as Table <a href="#tab:rectsm"
data-reference-type="ref" data-reference="tab:rectsm">3</a>,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi mathvariant="normal">Δ</mi><mi>k</mi></mrow><annotation encoding="application/x-tex">\Delta k</annotation></semantics></math>
in pcm versus the same-geometry pure-XS OpenMC reference; colour bars
symmetric about zero, green ring = point of smallest
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mo stretchy="false" form="prefix">|</mo><mi mathvariant="normal">Δ</mi><mi>k</mi><mo stretchy="false" form="prefix">|</mo></mrow><annotation encoding="application/x-tex">|\Delta k|</annotation></semantics></math>
(UO<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><msub><mi></mi><mn>2</mn></msub><annotation encoding="application/x-tex">_2</annotation></semantics></math>:
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>=</mo><mn>1</mn><mo>,</mo><mi>M</mi><mo>=</mo><mn>24</mn></mrow><annotation encoding="application/x-tex">S{=}1,M{=}24</annotation></semantics></math>,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>+</mi><mn>1.2</mn></mrow><annotation encoding="application/x-tex">+1.2</annotation></semantics></math> pcm;
MOX:
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>=</mo><mn>2</mn><mo>,</mo><mi>M</mi><mo>=</mo><mn>24</mn></mrow><annotation encoding="application/x-tex">S{=}2,M{=}24</annotation></semantics></math>,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>33.5</mn></mrow><annotation encoding="application/x-tex">-33.5</annotation></semantics></math> pcm;
core:
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>=</mo><mn>1</mn><mo>,</mo><mi>M</mi><mo>=</mo><mn>24</mn></mrow><annotation encoding="application/x-tex">S{=}1,M{=}24</annotation></semantics></math>,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>10.2</mn></mrow><annotation encoding="application/x-tex">-10.2</annotation></semantics></math> pcm).</figcaption>
</figure>

Figure [5](#fig:rect33best){reference-type="ref"
reference="fig:rect33best"} shows the fission field at the best
available configuration per geometry
(Table [3](#tab:rectsm){reference-type="ref" reference="tab:rectsm"}):
the UO$_2$ and core rows agree with OpenMC at the sub-percent level,
while the MOX row --- at its best subdivision $S{=}2$, still $-33.5$ pcm
off in $\kappa_{\mathrm{eff}}$ --- carries that eigenvalue bias as a
spatially structured field-level deviation.

<figure id="fig:rect33best" data-latex-placement="H">
<img src="c5g7_rect33_best_fission.png" style="width:71.0%" />
<figcaption><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>3</mn><mo>×</mo><mn>3</mn></mrow><annotation encoding="application/x-tex">3\times3</annotation></semantics></math>
rectangular tiling, best
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>×</mo><mi>M</mi></mrow><annotation encoding="application/x-tex">S\times M</annotation></semantics></math>
configuration per geometry (Table <a href="#tab:rectsm"
data-reference-type="ref" data-reference="tab:rectsm">3</a>):
pin-averaged fission
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi mathvariant="normal">Σ</mi><mi>f</mi></msub><mi>ϕ</mi></mrow><annotation encoding="application/x-tex">\Sigma_f\phi</annotation></semantics></math>,
OpenPSN (left), OpenMC on the identical rectangular-rod geometry
(centre), pin-by-pin relative error on fuel pins (right; both maps
normalised to unit sum). Top:
UO<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><msub><mi></mi><mn>2</mn></msub><annotation encoding="application/x-tex">_2</annotation></semantics></math>
assembly,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>=</mo><mn>1</mn><mo>,</mo><mi>M</mi><mo>=</mo><mn>24</mn></mrow><annotation encoding="application/x-tex">S{=}1,M{=}24</annotation></semantics></math>,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi mathvariant="normal">Δ</mi><mi>k</mi><mo>=</mo><mi>+</mi><mn>1.2</mn></mrow><annotation encoding="application/x-tex">\Delta k=+1.2</annotation></semantics></math> pcm;
middle: MOX assembly,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>=</mo><mn>2</mn><mo>,</mo><mi>M</mi><mo>=</mo><mn>24</mn></mrow><annotation encoding="application/x-tex">S{=}2,M{=}24</annotation></semantics></math>,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi mathvariant="normal">Δ</mi><mi>k</mi><mo>=</mo><mi>−</mi><mn>33.5</mn></mrow><annotation encoding="application/x-tex">\Delta k=-33.5</annotation></semantics></math> pcm;
bottom: 1/4 core,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>=</mo><mn>1</mn><mo>,</mo><mi>M</mi><mo>=</mo><mn>24</mn></mrow><annotation encoding="application/x-tex">S{=}1,M{=}24</annotation></semantics></math>,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi mathvariant="normal">Δ</mi><mi>k</mi><mo>=</mo><mi>−</mi><mn>10.2</mn></mrow><annotation encoding="application/x-tex">\Delta k=-10.2</annotation></semantics></math> pcm
(fuel region).</figcaption>
</figure>

The sweep shows three things. (i) On the UO$_2$ assembly the best point
($S{=}1,M{=}24$, $+1.2$ pcm) lies within the OpenMC standard error
($\pm3.6$ pcm): with a low-absorption fuel the nodal model is
essentially reference-grade on this geometry, and every deeper
subdivision is worse, monotonically from $-28.1$ pcm at $S{=}2$ to
$-56.5$ pcm at $S{=}6$ --- within this tiling and the tested range,
subdivision alone moves the result away from the reference. (ii) On the
MOX assembly the tiling as constructed ($S{=}1$) is badly off even at
$M{=}24$ ($-98.9$ pcm); a single subdivision to $S{=}2$ improves it to
its best point ($-33.5$ pcm), and any further subdivision
($S{=}3\ldots6$) degrades it again, back to $-128.4$ pcm --- the
non-monotonic $S$-response with a single interior optimum at $S{=}2$ is
the clearest signature of the gap-node problem below. (iii) On all three
geometries the $M$-convergence is non-monotonic at low order (an $M{=}8$
dip of $-101\ldots-504$ pcm) and only flattens out at $M{=}24$; on the
core the best point is $S{=}1,M{=}24$ ($-10.2$ pcm), again with $S{=}2$
slightly worse.

#### The slender water gap.

The $S$-saturation pattern above points to a geometric cause. In the
$3\times3$ tiling the water-gap nodes are intrinsically slender: each
edge strip node measures $s \times w = 0.9571\times0.1514$ cm, an aspect
ratio of $6.3{:}1$ (the corner nodes, $w\times w$, are square but carry
little physics). The uniform $S$ subdivision divides *both* dimensions
by $S$, so every sub-node inherits the parent's aspect ratio ---
$S$-subdivision shrinks the strip size and optical thickness but never
reshapes the strips. The residual eigenvalue bias of
Table [3](#tab:rectsm){reference-type="ref" reference="tab:rectsm"} is
therefore plausibly anchored in the strip nodes; within the tested
range, $S$-subdivision alone does not remove the deviation from the
reference, which is consistent with the UO$_2$ rows that degrade
smoothly and monotonically as $S$ grows at fixed $M$ (weak water
coupling: subdivision adds nodes but never re-shapes the strips), the
MOX rows whose only relief is a single subdivision --- best $S{=}2$,
degrading again for $S{\ge}3$ (strong fuel--water coupling in the gap)
--- and the 1/4 core, whose $S{=}2$ row is uniformly worse than $S{=}1$
across all $M$.

## Study III --- higher-order angular modes: the $A{=}3$ directional closure

#### The angular-convergence difficulty of Study II.

The results of the preceding paragraphs show that the rectangular-node
model still carries a large eigenvalue bias on the MOX assembly even at
$M{=}24$ ($-98.9$ pcm at $S{=}1$, $-33.5$ pcm at its best $S{=}2$), and
that on every geometry the $M$-convergence is non-monotonic at low
order, flattening only at the highest segment count. The $S$-response is
non-monotonic as well: on the MOX row a single subdivision buys the best
point and any further subdivision degrades it again. Neither knob can be
pushed to convergence in the old closure. The diagnosis is that the
original PSN carries *one* angular unknown per azimuthal segment: the
intra-segment angular distribution is averaged into a single
coefficient, and for a geometry with strong fuel--water coupling ---
where the angular flux varies sharply across a $7.5^{\circ}$ segment as
directions move from fuel-grazing to water streaming --- that single
mode carries a representation error that segment subdivision $M$ and
spatial subdivision $S$ dilute but do not remove within the range of
this work (Study II: a $-99$ pcm MOX bias persisting to $M{=}24$).

### The $A{=}3$ directional closure, derived from the PSN formulation

*Where the original PSN approximates.* Start from the steady multi-group
transport equation inside a homogeneous node of the original phase-space
nodal method [@psn2027], Eq. (2.1a) of that paper: $$\begin{equation}
\boldsymbol{\Omega}\cdot\nabla\psi(\boldsymbol{\Omega},\mathbf{r})
+\Sigma_t\psi(\boldsymbol{\Omega},\mathbf{r}) = \frac{Q(\mathbf{r})}{4\pi},
\label{eq:transport}
\end{equation}$$ with $Q$ the isotropic (scattering $+$ fission) source
and $\phi=\int\psi\,d\Omega$. The lowest GSPN ansatz, Eq. (2.3a) of
Ref. [@psn2027], $$\begin{equation}
\psi_{P_1} = \frac{1}{4\pi}\Big[\phi_0
-\frac{1}{\Sigma_t}\boldsymbol{\Omega}\cdot\nabla\phi_0\Big],
\end{equation}$$ annihilates the first-derivative term when substituted
back into Eq. [\[eq:transport\]](#eq:transport){reference-type="eqref"
reference="eq:transport"}, leaving
$-\frac{1}{\Sigma_t}(\boldsymbol{\Omega}\cdot\nabla)^2\phi_0
+\Sigma_t\phi_0 = Q$, which no single angle-independent $\phi_0$ can
satisfy for every direction. The original paper therefore integrates
over the full solid angle, using
$\langle\Omega_j\Omega_l\rangle_{4\pi}=\delta_{jl}/3$, and obtains the
isotropic diffusion equation, its Eq. (2.3c), $$\begin{equation}
-\frac{1}{3\Sigma_t}\nabla^2\phi_0+\Sigma_t\phi_0 = Q,
\label{eq:diff3c}
\end{equation}$$ whose wave number is $k^2 = 3\Sigma_t^2$ (Eq. (2.11b)).
The reconstruction
$\psi = \frac{1}{4\pi}[\phi - \Sigma_t^{-1}\boldsymbol{\Omega}\cdot\nabla\phi]$
with $\phi=\phi_0+C$ (Eqs. (2.3d,e)) adds one node-constant degree of
freedom fixed by the phase-space node balance (Eq. (2.9a)), but the
*body* of the node is still governed by the full-solid-angle average of
Eq. [\[eq:diff3c\]](#eq:diff3c){reference-type="eqref"
reference="eq:diff3c"}: the direction-dependent operator
$(\boldsymbol{\Omega}\cdot\nabla)^2$ has been replaced by the
angle-averaged Laplacian. The face quantities, by contrast, retain the
true angular second moments: the original face current (Eq. (2.8b))
carries the polar factor $\beta_i/3=\tau_i/2$ with
$\beta_i=\tfrac32\,\tau_i$ (Eq. (2.8d)) and the azimuthal
chordal-averaging factor $\gamma_m=\sin\Delta\varphi_m/\Delta\varphi_m$,
i.e.\
a direction-dependent diffusion tensor, with $c_m$ the segment centre,
$$D^{(0)}_{im} = \frac{\tau_i}{2\Sigma_t}
\begin{pmatrix}
1+\gamma_m\cos 2c_m & \gamma_m\sin 2c_m\\
\gamma_m\sin 2c_m & 1-\gamma_m\cos 2c_m
\end{pmatrix}$$ with $\tau_i=\langle\sin^2\theta\rangle_i$. **The
original method thus evaluates the body on the isotropic scale while the
faces carry the anisotropic angular moments; the $A{=}3$ directional
closure makes the body use the same angular second moments as the faces,
at every angular order.**

*Parity elimination and the exact even-parity equation.* Write
$f=4\pi\psi$ and split into even and odd parity,
$e(\Omega)=\tfrac12[f(\Omega)+f(-\Omega)]$,
$o(\Omega)=\tfrac12[f(\Omega)-f(-\Omega)]$:
$\boldsymbol{\Omega}\cdot\nabla o+\Sigma_t e=Q$ and
$\boldsymbol{\Omega}\cdot\nabla e+\Sigma_t o=0$. Inside a node of
constant $\Sigma_t$ the odd parity is eliminated *exactly*,
$o=-\Sigma_t^{-1}\boldsymbol{\Omega}\cdot\nabla e$, giving the
directional second-order equation $$\begin{equation}
-\boldsymbol{\Omega}\cdot\nabla\Big(\Sigma_t^{-1}
\boldsymbol{\Omega}\cdot\nabla e\Big)+\Sigma_t e = Q.
\label{eq:A1}
\end{equation}$$ This is the mother equation of the closure. If the
even-parity trial field is restricted to be angle-independent and
Eq. [\[eq:A1\]](#eq:A1){reference-type="eqref" reference="eq:A1"} is
then averaged over the full solid angle,
$\langle\boldsymbol{\Omega}\boldsymbol{\Omega}\rangle_{4\pi}
=\tfrac13\mathbf{I}$, the result is
Eq. [\[eq:diff3c\]](#eq:diff3c){reference-type="eqref"
reference="eq:diff3c"}, and the original reconstruction (2.3a) is
recovered as the exact parity-elimination identity. The projection order
matters: the even parity is projected *after* the exact elimination of
the odd parity, so the second moments that enter below are the *full*
moments $\langle b_a\,\Omega_x^2\, b_b\rangle$; truncating $e$ and $o$
to the same finite angular space first and then eliminating would give
the product
$\langle b_a\Omega_x b_c\rangle\langle b_c\Omega_x b_b\rangle$ instead,
a different (and weaker) closure. The eliminated odd flux is
reconstructable from $\boldsymbol{\Omega}\cdot\nabla e$ and is not
required to stay inside the truncated angular space.

*Polar reduction and the two polar moments.* For a $z$-invariant 2-D
problem the unit direction keeps its three components
$\boldsymbol{\Omega}=(\sin\theta\cos\varphi,
\sin\theta\sin\varphi,\cos\theta)$ and the operator uses $\partial_z=0$;
with even-parity pairing over opposite directions the $z$-component
drops out of every even-parity moment below, so only the planar
projection $\boldsymbol{\Omega}_\perp=(\sin\theta\cos\varphi,
\sin\theta\sin\varphi,0)$ enters. The scalar flux is the half-plane
average $\phi=\frac{1}{\pi}\int_0^1\!\!\int_0^\pi e\,d\varphi\,d\zeta$,
$\zeta=\cos\theta$; this half-plane folding uses, beyond the even-parity
pairing, reflection symmetry of the source and the boundaries in $z$ (or
an equivalent mirrored $z$-folded field) --- the isotropic-source,
vacuum/mirror 2-D configurations of this paper satisfy it, and the
even-parity pairing alone does not extend to arbitrary asymmetric
incident boundaries. Two polar moments are carried per polar sector $i$:
the folded solid-angle weight
$W_i=\int_{\theta_-}^{\theta_+}\sin\theta\,d\theta$ (normalized,
$\sum_i W_i=1$), the second moment $\tau_i=\langle\sin^2\theta\rangle_i$
and the first moment $\kappa_i=\langle\sin\theta\rangle_i$. With the
generic even $\theta$-partition $\Delta\theta=\pi/I$, $$\begin{align}
\tau_i &= 1-\frac{\zeta_+^2+\zeta_+\zeta_-+\zeta_-^2}{3}, \\
\kappa_i &= \frac{[\zeta\sqrt{1-\zeta^2}+\arcsin\zeta]_{\zeta_-}^{\zeta_+}}
{2(\zeta_+-\zeta_-)},
\end{align}$$ which are exactly the original node parameters
$\tau_i=2\beta_i/3=2\Sigma_t p_i$ and $\kappa_i$ of Ref. [@psn2027]
(Eq. (2.8d) for $\beta_i$; the generic polar set --- the source paper's
three-polar protocol is the same structure with fixed quadrature nodes).
These are P0 moments per polar sector --- the polar quadrature is not
extended here; $I$ stays a partition parameter.

*Azimuthal projection: the $K$-mode segment space.* The essential new
degree of freedom is the intra-segment angular representation. Within
segment $m$, $\varphi\in[\varphi_-,\varphi_+]$ of width $\Delta\varphi$,
centre $c_m$, half-width $\delta_m=\Delta\varphi/2$, set
$\xi=(\varphi-c_m)/\delta_m\in[-1,1]$ and expand the even-parity angular
flux in the orthonormal Legendre basis of the segment,
$$\begin{equation}
e(\mathbf{r},\varphi) \simeq \sum_{a=0}^{A} b_a(\xi)\,e_a(\mathbf{r}),
\qquad b_a(\xi)=\sqrt{2a+1}\,P_a(\xi),
\label{eq:azexp}
\end{equation}$$ with $K=A{+}1$ unknowns per face per segment ($K{=}4$
at $A{=}3$) and $\langle f\rangle=\tfrac12\int_{-1}^{1}f\,d\xi$, so
$\langle b_a b_b\rangle=\delta_{ab}$ and
$\langle b_a\rangle=\delta_{a0}$. Projecting
Eq. [\[eq:A1\]](#eq:A1){reference-type="eqref" reference="eq:A1"}:
multiply by each test basis function $b_a$, average over the segment and
the polar sector; the basis is angle-space only, so the spatial
derivatives come out of the angular integrals. This is precisely the
step that extends the original angular node balance (Eq. (2.9a) of
Ref. [@psn2027]) --- one segment-mean amplitude per segment --- into $K$
simultaneous angular test conditions per segment. The collision term is
diagonal by orthogonality, the isotropic source feeds only the $a{=}0$
mode, and the streaming term produces the three segment-averaged moment
matrices $$\begin{equation}
\mathbf{C}_m{}_{ab} = \big\langle b_a b_b\cos^2\varphi\big\rangle,
\qquad
\mathbf{S}_m{}_{ab} = \big\langle b_a b_b\sin^2\varphi\big\rangle,
\qquad
\mathbf{H}_m{}_{ab} = \big\langle b_a b_b\sin\varphi\cos\varphi\big\rangle,
\label{eq:CSH}
\end{equation}$$ evaluated once per segment by Gauss quadrature and
cached. The matrices are symmetric,
$\mathbf{C}+\mathbf{S}=\mathbf{I}_K$, $\mathbf{C}$ and $\mathbf{S}$
positive definite for any positive-width segment (grazing-adjacent
segments can be ill-conditioned, which the spectral construction below
monitors), and $\mathbf{H}$ is generically non-commuting with both.

*The directional diffusion coefficients.* Substituting the expansion
[\[eq:azexp\]](#eq:azexp){reference-type="eqref" reference="eq:azexp"}
into the mother equation [\[eq:A1\]](#eq:A1){reference-type="eqref"
reference="eq:A1"} and expanding the directional operator inside a node
of constant $\Sigma_t$: the in-plane projection
$\boldsymbol{\Omega}_\perp=(\sin\theta\cos\varphi,\sin\theta\sin\varphi,0)$
is independent of $(x,y)$, so the streaming term acts purely on the
spatial derivatives of the angular field, $$\begin{equation}
\boldsymbol{\Omega}\cdot\nabla\bigl(\boldsymbol{\Omega}\cdot\nabla e\bigr)
=\sin^2\theta\Bigl(\cos^2\varphi\, e_{xx}
+2\sin\varphi\cos\varphi\, e_{xy}+\sin^2\varphi\, e_{yy}\Bigr),
\label{eq:stream}
\end{equation}$$ where $e_{xx}$, $e_{xy}$, $e_{yy}$ are the second
spatial derivatives of $e$.

The projection then separates exactly. Multiply
[\[eq:stream\]](#eq:stream){reference-type="eqref"
reference="eq:stream"} by the test basis $b_a$ and average over the
segment $m$ and the polar sector $i$. The collision term is diagonal by
orthogonality ($\langle b_a b_b\rangle=\delta_{ab}$), the isotropic
source feeds only the $a{=}0$ mode, and the streaming term admits the
expansion $e=\sum_b b_b\,e_b$ with the spatial derivatives passing
through the angular average unchanged. Since $\sin^2\theta$ depends only
on the polar angle, the $\theta$- and $\varphi$-averages factorize, and
the contribution of the streaming term to the $a$-th test equation is
$$\begin{equation}
-\frac{\tau_i}{\Sigma_t}\sum_{b=0}^{A}\Bigl(
C_{ab}\,e_{b,xx}+2H_{ab}\,e_{b,xy}+S_{ab}\,e_{b,yy}\Bigr),
\label{eq:proj}
\end{equation}$$ where $\tau_i=\langle\sin^2\theta\rangle_i$ is the
sector's second polar moment and the angular integrals are exactly the
moment matrices [\[eq:CSH\]](#eq:CSH){reference-type="eqref"
reference="eq:CSH"}: the three directional second derivatives $e_{xx}$,
$e_{xy}$, $e_{yy}$ select $\mathbf{C}$, $\mathbf{H}$, $\mathbf{S}$
respectively, and no other combination enters. In vector form, with
$\mathbf{e}_{xx}=(e_{0,xx},\ldots,e_{A,xx})^T$ and analogously for
$e_{xy}$, $e_{yy}$, this is the projected body equation
$$\begin{equation}
-\frac{\tau_i}{\Sigma_t}\Bigl(\mathbf{C}\,\mathbf{e}_{xx}
+2\mathbf{H}\,\mathbf{e}_{xy}+\mathbf{S}\,\mathbf{e}_{yy}\Bigr)
+\Sigma_t\,\mathbf{e}=\mathbf{v}_0\,q(\mathbf{r}),
\label{eq:A7}
\end{equation}$$ where $\mathbf{e}=(e_0,\ldots,e_A)^T$,
$\mathbf{v}_0=(1,0,\ldots,0)^T$, and $q$ is the five-term source
(below). Identifying the *directional diffusion coefficients*
$$\begin{equation}
\mathbf{D}_x = \frac{\tau_i}{\Sigma_t}\mathbf{C} = 2p\,\mathbf{C},
\qquad
\mathbf{D}_y = \frac{\tau_i}{\Sigma_t}\mathbf{S} = 2p\,\mathbf{S},
\qquad
\mathbf{D}_{xy} = \frac{\tau_i}{\Sigma_t}\mathbf{H} = 2p\,\mathbf{H},
\label{eq:A8}
\end{equation}$$ with $p=\beta_i/(3\Sigma_t)=\tau_i/(2\Sigma_t)$ the
original node parameter (Eq. (2.8d) of Ref. [@psn2027] gives
$\beta_i=\tfrac32\,\tau_i$).

**The $A{=}0$ limit reproduces the original face current exactly.**
Restricting Eq. [\[eq:A8\]](#eq:A8){reference-type="eqref"
reference="eq:A8"} to the $a{=}0$ component and writing $c_m$ and
$\Delta\varphi_m$ for the segment centre and width, the three scalar
coefficients become $$\begin{equation}
(\mathbf{D}_x)_{00} = \frac{\tau_i}{2\Sigma_t}\,
\bigl(1+\gamma_m\cos 2c_m\bigr),\qquad
(\mathbf{D}_{xy})_{00} = \frac{\tau_i}{2\Sigma_t}\,
\gamma_m\sin 2c_m,\qquad
(\mathbf{D}_y)_{00} = \frac{\tau_i}{2\Sigma_t}\,
\bigl(1-\gamma_m\cos 2c_m\bigr),
\label{eq:CSH-A0}
\end{equation}$$ $\gamma_m=\sin\Delta\varphi_m/\Delta\varphi_m$ as in
Eq. (2.8b) of Ref. [@psn2027]. With the face-current relation
(Section [\[sec:face\]](#sec:face){reference-type="ref"
reference="sec:face"} below),
$J_n=-(n_x\mathbf{D}_x+n_y\mathbf{D}_{xy})\bar{e}_x^{\,n}
-(n_x\mathbf{D}_{xy}+n_y\mathbf{D}_y)\bar{e}_y^{\,n}$, the $a{=}0$
component is then $J_n=-\hat{\mathbf{n}}^T D^{(0)}_{im}
\overline{\nabla\phi}^{\,n}$ with $D^{(0)}_{im}=\frac{\tau_i}{2\Sigma_t}
\bigl[(1-\gamma_m)\mathbf{I}+2\gamma_m\mathbf{d}_m\mathbf{d}_m^T\bigr]$
($\mathbf{d}_m=(\cos c_m,\sin c_m)^T$) the original face-current tensor,
*for any face normal and any segment* --- the directional face current
at $A{=}0$ is the original Eq. (2.8b), not an approximation to it.

**The original wave number.** The scalar $k^2=3\Sigma_t^2$ of
Eq. [\[eq:diff3c\]](#eq:diff3c){reference-type="eqref"
reference="eq:diff3c"} is the $A{=}0$ special case of
Eq. [\[eq:A8\]](#eq:A8){reference-type="eqref" reference="eq:A8"} in
which both averages are taken over their full ranges --- the azimuthal
segment over the complete half plane ($\Delta\varphi_m=\pi$,
$\gamma_m=0$, $(\mathbf{C})_{00}=(\mathbf{S})_{00}=\tfrac12$,
$\mathbf{H}=\mathbf{0}$) *and* the polar range over the full solid
angle, $\tau_i=\tfrac23$. Then
$\mathbf{D}=\frac{\tau_i}{2\Sigma_t}\mathbf{I}=\frac{1}{3\Sigma_t}\mathbf{I}$
and the single decay constant
$k=\sqrt{\Sigma_t/\tfrac{1}{3\Sigma_t}}=\sqrt{3}\,\Sigma_t$ is
recovered. For a truncated polar range the decay rates are
$\sqrt{\Sigma_t/\lambda_j}$ with $\lambda_j$ the spectrum of
$(\tau_i/\Sigma_t)\mathbf{C}$: at $A{=}0$ over the full half-plane that
spectrum is the single value $\tau_i/(2\Sigma_t)$, so the two scales
coincide only at $\tau_i=\tfrac23$ --- a full-half-plane $A{=}0$
statement; for a finite segment the whole $\mathbf{C}$ spectrum, not the
polar moment alone, sets the rates. The directional closure generalizes
that one decay constant to the spectrum of the segment-resolved matrix;
at $A{=}0$ the two discretizations still differ, because the original
*body* uses a single scalar wave number (full-solid-angle for the
generic set, per-sector for the three-polar protocol) while the
directional body uses the *segment-resolved* decay rates
(Section [4.4.1.1](#sec:a3a0){reference-type="ref" reference="sec:a3a0"}
below).

*Rectangular nodes: the per-node widths enter only through the spatial
scales.* The angular construction above is width-independent: neither
the projected equation [\[eq:A7\]](#eq:A7){reference-type="eqref"
reference="eq:A7"} nor the moment matrices
[\[eq:CSH\]](#eq:CSH){reference-type="eqref" reference="eq:CSH"}
contains a node width, and the same is true of the original method,
whose spatial shape (Eq. (2.11a) of Ref. [@psn2027]) uses the scalar
wave number $k=\sqrt{3}\,\Sigma_t$ for the generic polar set and
$k_i=\sqrt{2}\,\Sigma_t/\mu_i$ for the three-polar protocol, with the
node widths entering only through the spatial coordinates and the
hyperbolic arguments. For rectangular nodes the source basis is rescaled
by the per-axis coordinates $\eta_x=2x/h_x$, $\eta_y=2y/h_y$ (the same
five spatial moments), and the hyperbolic family below carries per-axis
arguments $U_x=h_xK_x/2$, $U_y=h_yK_y/2$, so each angular mode decays
across the node at its own rate $k_j=\sqrt{\Sigma_t/\lambda_j}$ over the
actual width of each axis. Nodes meet corner-to-corner only (no
T-junctions), so on this conforming grid every face has a single shared
tangential length and the face-current matching is unchanged (the two
nodes on either side of a face need not have the same width in the
normal direction). When all widths are equal the rectangular operators
reduce exactly to the square-node path, and the two are verified
node-by-node to machine precision (the regression of Section 2.2).

*The body function: hyperbolic functions of the matrix spectrum.* With
the five-term spatial source $$\begin{equation}
q(x,y) = q_0+q_{1x}P_1(\eta_x)+q_{1y}P_1(\eta_y)+q_{2x}P_2(\eta_x)
+q_{2y}P_2(\eta_y),\qquad
\eta_x=\frac{2x}{h_x},\quad \eta_y=\frac{2y}{h_y},
\label{eq:source5}
\end{equation}$$ (the same five spatial source moments as the original
method; the spatial function family is unchanged, written in the
rescaled per-axis coordinates), the even-parity body is sought in the
separated hyperbolic family $$\begin{equation}
\mathbf{e}(x,y) = F_x(x)\mathbf{a}_x+G_x(x)\mathbf{b}_x
+F_y(y)\mathbf{a}_y+G_y(y)\mathbf{b}_y+\mathbf{c}
+\mathbf{v}_0\,q(x,y)/\Sigma_t,
\label{eq:A10}
\end{equation}$$ $F_x(x)=\cosh(K_x x)\cosh(U_x)^{-1}$,
$G_x(x)=\sinh(K_x x)\cosh(U_x)^{-1}$ (and y-analogues), with the
positive-definite square roots $K_x=(\Sigma_t\mathbf{D}_x^{-1})^{1/2}$,
$U_x=h_xK_x/2$ and similarly for $y$. Each scalar mode of the
eigen-decomposition of $\mathbf{D}_x$ contributes a hyperbolic pair at
its own decay constant $k_j=\sqrt{\Sigma_t/\lambda_j}$; the
scaled-hyperbolic form $\cosh v/\cosh u$ (all exponents non-positive)
guards the grazing directions against overflow. The quadratic source
fixes the constant vector by $\partial_x^2 P_2(\eta_x)=12/h_x^2$:
$$\begin{equation}
\mathbf{c} = \frac{12}{\Sigma_t^2}\Big(
\mathbf{D}_x\mathbf{v}_0\,\frac{q_{2x}}{h_x^2}
+\mathbf{D}_y\mathbf{v}_0\,\frac{q_{2y}}{h_y^2}\Big).
\label{eq:A11}
\end{equation}$$ The family has $e_{xy}=0$; it satisfies the *projected*
node equation Eq. [\[eq:A7\]](#eq:A7){reference-type="eqref"
reference="eq:A7"} inside this separated space --- the general
mixed-derivative response is not recovered, and $\mathbf{D}_{xy}$ acts
through the face tangential-gradient current (below), not through the
body. All matrix functions of one axis ($\tanh U$, $\tanh U/U$, and the
space-integrals $I_1,I_2$ defined below) are built from the *same*
eigen-decomposition; in particular $\mathbf{D}_{xy}\tanh U_y$ is not
commutable and keeps that order.

*Face currents: the $4K\times4K$ local system.*[]{#sec:face
label="sec:face"} With $t_x=\tanh U_x$ and the face operators
$A_x=\mathbf{D}_xK_xt_x$, $B_x=\mathbf{D}_xK_x$,
$H_y=\frac{2}{h_y}\mathbf{D}_{xy}\tanh U_y$,
$H_x=\frac{2}{h_x}\mathbf{D}_{xy}\tanh U_x$ (y-analogues of $A_x,B_x$),
the outward face-current coefficients of the four faces satisfy the
block relation $$\begin{equation}
\underbrace{\begin{pmatrix}\mathbf{J}_{x+}\\\mathbf{J}_{x-}\\
\mathbf{J}_{y+}\\\mathbf{J}_{y-}\end{pmatrix}}_{\mathbf{J}}
= \underbrace{\begin{pmatrix}
-A_x&-B_x&0&-H_y\\
-A_x& B_x&0& H_y\\
0&-H_x&-A_y&-B_y\\
0& H_x&-A_y& B_y
\end{pmatrix}}_{\mathcal{K}}
\begin{pmatrix}\mathbf{a}_x\\\mathbf{b}_x\\\mathbf{a}_y\\\mathbf{b}_y\end{pmatrix}
+ \mathcal{O}\,\mathbf{q},
\label{eq:A13}
\end{equation}$$ which is the tensor face-current relation
$\mathbf{J}_n = -\big(n_x\mathbf{D}_x+n_y\mathbf{D}_{xy}\big)\overline{\mathbf{e}}_x^{\,n}
-\big(n_x\mathbf{D}_{xy}+n_y\mathbf{D}_y\big)\overline{\mathbf{e}}_y^{\,n}$:
*every angular mode sees its own directional diffusion coefficient*,
which is the origin of the name. The source-offset block $\mathcal{O}$
is the face evaluation of the five-term particular solution, with an
overall $1/\Sigma_t$ factor: its rows are the columns
$\mathbf{D}_x\mathbf{v}_0$, $\mathbf{D}_y\mathbf{v}_0$,
$\mathbf{D}_{xy}\mathbf{v}_0$ scaled by $(1/\Sigma_t)(2/h)$ and
$(1/\Sigma_t)(6/h)$, so *high angular modes do receive source
contribution* even though the source itself is purely $a{=}0$. Inverting
Eq. [\[eq:A13\]](#eq:A13){reference-type="eqref" reference="eq:A13"} (by
direct solve, not explicit inversion) yields the local response mapping
$\mathbf{q}$ and $\mathbf{J}\mapsto$ face means and the five spatial
moments. At $K{=}4$ the local coefficient system is $16\times16$ per
node (plus the five source columns); the angular quadrature points used
to build Eq. [\[eq:CSH\]](#eq:CSH){reference-type="eqref"
reference="eq:CSH"} are construction points, not transport unknowns. The
five spatial moments returned to the source update are $$\begin{align}
m_0 &= \mathbf{v}_0^T\overline{\mathbf{e}}, \qquad
m_{1x} = 3\,\mathbf{v}_0^T I_1(U_x)\,\mathbf{b}_x + q_{1x}/\Sigma_t,\\
m_{2x} &= 5\,\mathbf{v}_0^T I_2(U_x)\,\mathbf{a}_x + q_{2x}/\Sigma_t,
\end{align}$$ with $I_1(U)=U^{-1}(I-U^{-1}\tanh U)$,
$I_2(U)=U^{-1}(\tanh U-3I_1(U))$ and y-analogues. The angular-zeroth
extraction is performed *after* the full matrix--vector product
(e.g. $(3\,I_1\mathbf{b}_x)_0$, not $3(I_1)_{00}(\mathbf{b}_x)_0$): high
angular modes feed the scalar source through the off-diagonal entries,
which is exactly the coupling that the single-mode closure cannot
represent. The node volume balance
$\overline{\mathbf{e}}=\mathbf{v}_0 q_0/\Sigma_t
-(\mathbf{J}_{x+}+\mathbf{J}_{x-})/(h_x\Sigma_t)
-(\mathbf{J}_{y+}+\mathbf{J}_{y-})/(h_y\Sigma_t)$ holds for every
angular mode simultaneously.

*Assembly: face fluxes, boundaries, and the eigenproblem.* The signed
face flux entering the global connection is the original PSN convention
with the basis promoted to a matrix:
$\Phi_n = \frac{\kappa_i}{2}\mathbf{E}_n\,\mathbf{L}_n$,
$\big(\mathbf{E}_n\big)_{ab}=\langle b_a b_b\cos(\varphi-\varphi_n)\rangle$,
with $\mathbf{L}_n$ the face-mean even-parity vector,
$\big(\mathbf{L}_n\big)_a = \frac{1}{|\partial n|}\int_{\partial n}
e_a\,ds$; at $A{=}0$ this is exactly the original $\alpha\,\times$
face-mean. The internal face closure uses the same raw face-mean
continuity and anti-symmetric outward-current pairing as the original
method; on a uniform even $M$ grid with axis-aligned nodes every segment
has a single sign of $\Omega\cdot n$, so the connection is exact in this
$L$-representation. Mirror reflection maps segment $m$ to $M{-}1{-}m$
with local $\xi\mapsto-\xi$, which in the $K$-mode representation is
$L_m = R\,L_{m'}$, $J_m = -R\,J_{m'}$ with
$R=\mathrm{diag}(1,-1,1,-1,\ldots)$ --- odd modes reverse their
reflection sign; the moment matrices obey the corresponding identities
$C_{m'}=RC_mR$, $H_{m'}=-RH_mR$. Vacuum is the hemispheric current
$J = B\,L$ with $B$ the positive-definite matrix
$\bigl(B\bigr)_{ab}=\kappa_i\langle b_a b_b
|\cos(\varphi-\varphi_n)|\rangle$ (the same normal-relative form on
every face: $\varphi_n=0$ on $x$ faces, $\pi/2$ on $y$ faces),
$\kappa_i$ the first polar moment (not $\sqrt{\tau_i}$); at $A{=}0$ this
is $B_{00}=2|\alpha_{imn}|$, the original vacuum condition. Multi-group
assembly keeps the original structure: each group's five-moment response
$R_g$ is coupled through the isotropic scattering and fission blocks,
the discrete eigenproblem is non-symmetric and is driven by the same
outer fission-source (power) iteration as the original implementation;
each outer sweep solves the node-coupling system with a Krylov method
(GMRES). The outer iteration stops when the relative eigenvalue change
$|\Delta k|/k$ and the fission-source iterate change both fall below
$10^{-7}$: the iterate change is the $\ell_2$ norm of the difference
between successive fission sources divided by the larger of the previous
source's norm and unity,
$r_F=\|s^{(n+1)}-s^{(n)}\|_2/\max(\|s^{(n)}\|_2,1)$, a scaled iterate
change --- relative when the source norm exceeds unity, absolute below
it --- with both iterates produced on the same normalisation convention;
the GMRES inner solve runs to a relative linear-residual tolerance of
$10^{-7}$ (absolute tolerance zero) --- the same value as the outer
gate, as used by the production sweep driver; the library default of
$10^{-11}$ applies to standalone CLI runs --- restarted every 20
iterations, at most $800$ internal Krylov iterations per linear system.

#### Relation to the original closure at $A{=}0$ {#sec:a3a0}

At $A{=}0$ the two branches of the implementation share the
*face-current constitutive law* --- the $(0,0)$ entries of
Eq. [\[eq:CSH\]](#eq:CSH){reference-type="eqref" reference="eq:CSH"} are
the original segment means $\langle\cos^2\varphi\rangle_m$ etc. --- and
the interface flux/current connection. The *body* differs. The original
branch keeps the published scalar wave number, $k=\sqrt{3}\,\Sigma_t$
for the generic polar set and $k_i=\sqrt{2}\,\Sigma_t/\mu_i$
($\mu_i=\sin\theta_i$) for the three-polar protocol, while the
directional branch evaluates its hyperbolic functions at the
segment-resolved spectrum of $\mathbf{D}_x,\mathbf{D}_y$, i.e. at
$\sqrt{\Sigma_t/\lambda_j}$ with $\lambda_j$ the segment-dependent
eigenvalues. A shared constitutive law and connection do not make the
condensed nodal response $J,q\mapsto L,m$ identical --- the body wave
number enters the face-current block matrices directly --- so the two
discretizations agree on the face current at equal gradients but not
inside the node: this is precisely the body/face scale split of the
original method (Section above) carried into the new branch, and it is
why the directional $A{=}0$ solution is a distinct, segment-resolved
discretization rather than a reproduction of the legacy state matrix
(verified numerically over all 2016 node-response configurations of a
C5G7 pin cell: 3 polar points $\times$ 7 groups $\times$ 24 azimuthal
segments $\times$ 4 material groups: the original branch at $A{=}0$
reproduces the legacy node response to $4.8\times10^{-13}$, while the
directional branch at $A{=}0$ differs by $\mathcal{O}(1)$ in the body
scales, with a decay-rate spread of $0.62$--$45.9$ across the half-plane
segments). The $A{=}0$ directional solution is the correct limiting
member of the new family; the $A{=}1,2,3$ results below are reported
with it as the reference closure.

#### Verification of the local construction.

Before any benchmark was re-run, the local node response was verified
against independently constructed solutions: for $A{=}0\ldots3$, five
azimuthal segment widths, both aspect ratios of a $2$-cm node, and all
five unit sources (plus one mixed source), a spatial field is built from
arbitrary homogeneous coefficients and source, the four face
flux/current vectors and the five spatial moments are computed by
independent numerical integration, and compared against the local
response matrix. Maximum normalized differences over 240 cases: face
response and five moments $4.5\times10^{-12}$, constant-vector recovery
from the volume balance $8.5\times10^{-12}$, strong residual of the
projected body equation $1.0\times10^{-15}$. These verify the
discretization *as written*; they are not a transport-accuracy
statement.

#### Motivation: where the missing modes go.

Before re-running any benchmark, it is worth seeing what the added modes
actually represent. Figure [6](#fig:pinangular){reference-type="ref"
reference="fig:pinangular"} takes the synthetic one-group square-pin
problem (fuel square $0.9571$ cm in a $1.26$ cm pitch, purely absorbing,
vacuum boundary) and computes the even-parity angular flux at a
water-region observation point by piecewise-analytic ray transport; the
same reference is then orthogonally projected, sector by sector, onto
the $A{=}0\ldots3$ Legendre spaces. Three $7.5^{\circ}$ sectors of an
$M{=}24$ partition are shown: a smooth sector (a), a sector containing
two geometric corner events (b), and a fuel-grazing sector (c). The
segment mean is exactly $A{=}0$; the smooth sector is already captured
by $A{=}1$, while the corner and grazing sectors need the full cubic to
track the reference --- a direct picture of why the *same* closure that
is harmless on a weakly coupled geometry is insufficient on a pin cell
with a slender water gap. (These curves are projections of one fixed
reference function, not four PSN solutions; the figure demonstrates
representation, not eigenvalue accuracy.)

<figure id="fig:pinangular" data-latex-placement="H">
<embed src="pin_angular_three_sectors_en.pdf" style="width:98.0%" />
<figcaption>Intra-segment angular representation at a water-region
observation point of a synthetic, purely absorbing square-pin problem
(pin pitch
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>1.26</mn><annotation encoding="application/x-tex">1.26</annotation></semantics></math> cm,
fuel side
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>0.9571</mn><annotation encoding="application/x-tex">0.9571</annotation></semantics></math> cm,
observation point
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mo stretchy="false" form="prefix">(</mo><mn>0.535</mn><mo>,</mo><mn>0.15</mn><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">(0.535,0.15)</annotation></semantics></math> cm,
polar angle
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><msup><mn>90</mn><mo>∘</mo></msup><annotation encoding="application/x-tex">90^{\circ}</annotation></semantics></math>;
fuel/water
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mo stretchy="false" form="prefix">(</mo><msub><mi mathvariant="normal">Σ</mi><mi>t</mi></msub><mo>,</mo><mi>Q</mi><mo stretchy="false" form="postfix">)</mo><mo>=</mo><mo stretchy="false" form="prefix">(</mo><mn>1.2</mn><mo>,</mo><mn>1</mn><mo stretchy="false" form="postfix">)</mo><mi>/</mi><mo stretchy="false" form="prefix">(</mo><mn>0.2</mn><mo>,</mo><mn>0.04</mn><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">(\Sigma_t,Q)=(1.2,1)/(0.2,0.04)</annotation></semantics></math>,
vacuum boundary). Reference: even parity
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>u</mi><mo>=</mo><mo stretchy="false" form="prefix">(</mo><msub><mi>F</mi><mi mathvariant="normal">Ω</mi></msub><mo>+</mo><msub><mi>F</mi><mrow><mi>−</mi><mi mathvariant="normal">Ω</mi></mrow></msub><mo stretchy="false" form="postfix">)</mo><mi>/</mi><mn>2</mn></mrow><annotation encoding="application/x-tex">u=(F_{\Omega}+F_{-\Omega})/2</annotation></semantics></math>
from piecewise-analytic ray transport, normalised by its own sector
mean. Curves: orthogonal Legendre projections of the <em>same</em>
reference onto
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>A</mi><mo>=</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn></mrow><annotation encoding="application/x-tex">A{=}0,1,2,3</annotation></semantics></math>
— i.e. what the closure with
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>A</mi><annotation encoding="application/x-tex">A</annotation></semantics></math>
degrees of freedom per segment can represent. Top: geometry, the fan is
the full angular sector, the solid ray the representative direction (a)
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>15</mn><annotation encoding="application/x-tex">15</annotation></semantics></math>–<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><msup><mn>22.5</mn><mo>∘</mo></msup><annotation encoding="application/x-tex">22.5^{\circ}</annotation></semantics></math>
(smooth), (b)
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>30</mn><annotation encoding="application/x-tex">30</annotation></semantics></math>–<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><msup><mn>37.5</mn><mo>∘</mo></msup><annotation encoding="application/x-tex">37.5^{\circ}</annotation></semantics></math>
(fuel-corner at
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><msup><mn>31.805</mn><mo>∘</mo></msup><annotation encoding="application/x-tex">31.805^{\circ}</annotation></semantics></math>
and outer-boundary corner at
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><msup><mn>33.803</mn><mo>∘</mo></msup><annotation encoding="application/x-tex">33.803^{\circ}</annotation></semantics></math>),
(c)
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>82.5</mn><annotation encoding="application/x-tex">82.5</annotation></semantics></math>–<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><msup><mn>90</mn><mo>∘</mo></msup><annotation encoding="application/x-tex">90^{\circ}</annotation></semantics></math>
(fuel-grazing at
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><msup><mn>84.869</mn><mo>∘</mo></msup><annotation encoding="application/x-tex">84.869^{\circ}</annotation></semantics></math>).
Illustration of angular representation only; not an eigenvalue or
accuracy result.</figcaption>
</figure>

#### The checkerboard of Ref. [@psn2027] at $A{=}3$.

The first re-computation of Study II's protocol is the strong/weak
absorber checkerboard of Figure 3 of Ref. [@psn2027]: a $2\times2$ base
cell of $1$ cm nodes (all edges reflective), one group, fuel
$\Sigma_t{=}1.5$, $\Sigma_s{=}1.35$, $\nu\Sigma_f{=}0.24$ and absorber
$\Sigma_t{=}1.0$ with $\Sigma_s{=}0.93$ (weak) or $0.30$ (strong). We
uniformly subdivide to $2S\times2S$ nodes and sweep $S{=}1\ldots20$
against $M{=}2,4,8,12,16,24$ --- 240 points per angular model --- with
the $A{=}3$ closure and a $10^{-7}$ convergence gate. The polar
partition is the generic even set with $I{=}30$ segments (15 independent
sectors); the source paper's three-polar protocol was swept as well
($S{=}1\ldots12$). References: the OpenMC values of the source paper,
$1.12974$ (weak) and $0.51673$ (strong), quoted to five decimals
($\approx\pm0.5$ pcm precision limit).

Figure [7](#fig:cbA3){reference-type="ref" reference="fig:cbA3"} gives
the $I{=}30$ sweep. Four readings. (i) The coarse-grid part converges as
second order in the node size: $S{=}1\to6$ reduces $|\Delta k|$ by a
factor $\sim$`<!-- -->`{=html}28 at $M{=}24$ on both absorbers (weak
$+458.6\to+16.1$, strong $+2430.4\to+89.5$ pcm), and the $M$-columns
flatten from $M{=}8$ on --- the angular-segment count is not the
bottleneck once the segment count is above 8. (ii) The best point of the
whole sweep is the weak absorber at $S{=}12$, $M{=}16/24$: $-0.1$ pcm,
i.e. at the precision limit of the five-decimal reference; on the strong
absorber the best point is $S{=}14$, $M{=}24$ ($-0.5$ pcm) / $S{=}13$,
$M{=}8$ ($-1.9$ pcm). (iii) Beyond $S{\approx}12$ the deviation from the
reference stops decreasing, and the fine-grid rows settle near a small
residual within this scan: over $S{=}10\ldots20$ each $M{=}24$ column is
fit by $b+c_2/S^2$ to $0.02$ pcm RMS (weak) and $0.08$ pcm RMS (strong),
whereas the one-term $b+c_1/S$ form fits the same data at
$0.16$/$0.84$ pcm RMS (the fit uses the unrounded sweep values) ---
roughly a factor of $9$ (weak) to $10$ (strong) worse, i.e. the decay is
second-order-like rather than linear. The finite-range data are thus
consistent with a spatial contribution decaying as $S^{-2}$ plus a
residual that persists at the fixed angular discretization --- a reading
of a finite scan, not a proven asymptotic error decomposition; the
polar-set comparison below ($I{=}60$ moves the $S{=}20$ point by
$+2.9$ pcm weak, $+15.3$ pcm strong) shows that residual responds to the
polar partition, but does not by itself establish $I{=}60$ as converged.
Within this scan the points closest to the reference sit at
$S{=}12$--$14$; beyond that range a larger $S$ adds cost without further
reducing the deviation from the reference. (iv) The $M{=}2$ column's
negative bias deepens with $S$ (weak $-51.8$, strong $-259.5$ at
$S{=}20$): the two-segment azimuthal representation carries a structural
streaming error that refinement exposes rather than hides.

<figure id="fig:cbA3" data-latex-placement="H">
<img src="checkerboard_a3_i30_sm.png" style="width:98.0%" />
<figcaption>Checkerboard of Ref. <span class="citation"
data-cites="psn2027"></span> Fig. 3,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>A</mi><mo>=</mo><mn>3</mn></mrow><annotation encoding="application/x-tex">A{=}3</annotation></semantics></math>
closure, generic
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>I</mi><mo>=</mo><mn>30</mn></mrow><annotation encoding="application/x-tex">I{=}30</annotation></semantics></math>
polar set:
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi mathvariant="normal">Δ</mi><mi>k</mi></mrow><annotation encoding="application/x-tex">\Delta k</annotation></semantics></math>
versus
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>M</mi><annotation encoding="application/x-tex">M</annotation></semantics></math>
for each
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>S</mi><annotation encoding="application/x-tex">S</annotation></semantics></math>
(<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>=</mo><mn>1</mn><mi>…</mi><mn>20</mn></mrow><annotation encoding="application/x-tex">S{=}1\ldots20</annotation></semantics></math>,
one curve per subdivision), weak absorber (left) and strong absorber
(right). Vertical axis in symlog: the apparent “jump” of the curves as
they cross
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi mathvariant="normal">Δ</mi><mi>k</mi><mo>=</mo><mn>0</mn></mrow><annotation encoding="application/x-tex">\Delta k=0</annotation></semantics></math>
is an axis artefact of
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mo stretchy="false" form="prefix">|</mo><mi mathvariant="normal">Δ</mi><mi>k</mi><mo stretchy="false" form="prefix">|</mo><mo>&lt;</mo><mn>1</mn></mrow><annotation encoding="application/x-tex">|\Delta k|&lt;1</annotation></semantics></math> pcm
being drawn on the linear branch of the symlog scale — the underlying
data are continuous. All 240 points converged to
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><msup><mn>10</mn><mrow><mi>−</mi><mn>7</mn></mrow></msup><annotation encoding="application/x-tex">10^{-7}</annotation></semantics></math>;
best points: weak
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>=</mo><mn>12</mn></mrow><annotation encoding="application/x-tex">S{=}12</annotation></semantics></math>,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>M</mi><mo>=</mo><mn>16</mn><mi>/</mi><mn>24</mn></mrow><annotation encoding="application/x-tex">M{=}16/24</annotation></semantics></math>
(<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.1</mn></mrow><annotation encoding="application/x-tex">-0.1</annotation></semantics></math> pcm),
strong
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>=</mo><mn>14</mn></mrow><annotation encoding="application/x-tex">S{=}14</annotation></semantics></math>,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>M</mi><mo>=</mo><mn>24</mn></mrow><annotation encoding="application/x-tex">M{=}24</annotation></semantics></math>
(<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.5</mn></mrow><annotation encoding="application/x-tex">-0.5</annotation></semantics></math> pcm).</figcaption>
</figure>

Two effects are visible in the fine-grid rows and worth isolating.
First, the signed-error crossing of zero between the coarse-grid
positive bias and the fine-grid offset (e.g. weak $S{=}11$--$14$ at
$M{=}16/24$) is consistent with cancellation between a decaying spatial
contribution and a finite-angular-discretization offset: repeating the
$I{=}30$ point at $S{=}20$, $M{=}24$, $A{=}3$ with the even polar set
doubled to $I{=}60$ moves the weak-absorber value from $-3.8$ to
$-0.9$ pcm and the strong-absorber value from $-11.6$ to $+3.7$ pcm.
That comparison demonstrates polar-sector sensitivity; it does not
establish a loss of second-order spatial convergence. The two polar sets
do not by themselves prove the $I{=}60$ residual is converged, nor do
they, on the same finite-range data, separate a polar-quadrature offset
from any residual spatial-basis contribution --- they show the fine-grid
residual responds to the polar partition, and no more. Second, the polar
partition also explains the coarse/fine gap between the two protocols:
on the source paper's three-polar protocol the same fine-grid points sit
at $+3.9$ (weak) and $+28.6$ (strong) pcm at $S{=}12$, $M{=}24$, versus
$-0.1$ and $+7.1$ pcm with $I{=}30$ --- the extra polar sectors pay off
only on the finer meshes, where the sensitivity to the polar
discretisation is more visible than the spatial contribution; at $S{=}1$
the two polar sets agree to within a few pcm.

#### The $2\times2$ fuel-split tiling.

The $A{=}3$ closure is run on the C5G7 pin cell with the fuel square
split $2\times2$: each pin cell becomes a $4\times4$ block of rectangles
--- four fuel sub-nodes of side $s/2{=}0.4786$ cm and a water ring of
corner blocks ($0.1514^2$ cm$^2$) and edge strips
($0.4786\times0.1514$ cm, aspect $3.2{:}1$) ---
Figure [8](#fig:rect4layout){reference-type="ref"
reference="fig:rect4layout"}. The $17\times17$ assembly is
$68\times68{=}4624$ rectangles and the 1/4 core $204\times204{=}41616$;
pin positions, fuel area and total area are unchanged. The $2\times2$
split is the tiling in which the edge water strips --- the channels that
carry streaming across the pin --- carry a tangential extent of $s/2$
(aspect $3.2{:}1$), halved from $s$ (aspect $6.3{:}1$) in the $3\times3$
tiling, while remaining dedicated water nodes on the fuel/water
interface as they already were. It is the tiling the $A{=}3$ closure is
run on: the angular correction and the intra-node spatial split act on
different parts of the error, and the $S$ sub-division of Section 2.2
remains available on top of it.

<figure id="fig:rect4layout" data-latex-placement="H">
<img src="rect4_pin_geometry.png" style="width:98.0%" />
<figcaption>The
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2</mn><mo>×</mo><mn>2</mn></mrow><annotation encoding="application/x-tex">2\times2</annotation></semantics></math>
fuel-split tiling of the C5G7 pin cell. (a)
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>17</mn><mo>×</mo><mn>17</mn></mrow><annotation encoding="application/x-tex">17\times17</annotation></semantics></math>
assembly, one pin circled. (b) One pin at
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>=</mo><mn>1</mn></mrow><annotation encoding="application/x-tex">S{=}1</annotation></semantics></math>:
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>4</mn><mo>×</mo><mn>4</mn></mrow><annotation encoding="application/x-tex">4\times4</annotation></semantics></math>
blocks
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mo stretchy="false" form="prefix">[</mo><annotation encoding="application/x-tex">[</annotation></semantics></math>water
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mo stretchy="false" form="prefix">|</mo><annotation encoding="application/x-tex">|</annotation></semantics></math>
fuel<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>/</mi><mn>2</mn></mrow><annotation encoding="application/x-tex">/2</annotation></semantics></math>
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mo stretchy="false" form="prefix">|</mo><annotation encoding="application/x-tex">|</annotation></semantics></math>
fuel<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>/</mi><mn>2</mn></mrow><annotation encoding="application/x-tex">/2</annotation></semantics></math>
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mo stretchy="false" form="prefix">|</mo><annotation encoding="application/x-tex">|</annotation></semantics></math>
water<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mo stretchy="false" form="postfix">]</mo><mo>×</mo><mn>2</mn></mrow><annotation encoding="application/x-tex">]\times2</annotation></semantics></math>;
water gap
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>0.1514</mn><annotation encoding="application/x-tex">0.1514</annotation></semantics></math> cm,
fuel square
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>0.957</mn><annotation encoding="application/x-tex">0.957</annotation></semantics></math> cm
(<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mo>=</mo><mn>2</mn><mo>×</mo><mn>0.479</mn></mrow><annotation encoding="application/x-tex">=2\times0.479</annotation></semantics></math>).
(c) The same pin at
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>=</mo><mn>2</mn></mrow><annotation encoding="application/x-tex">S{=}2</annotation></semantics></math>:
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>8</mn><mo>×</mo><mn>8</mn></mrow><annotation encoding="application/x-tex">8\times8</annotation></semantics></math>
blocks.</figcaption>
</figure>

#### C5G7 at $A{=}3$: assemblies and core.

Figure [9](#fig:a3mconv){reference-type="ref" reference="fig:a3mconv"}
gives the $A{=}3$ sweep on the two $17\times17$ assemblies, $S{=}1,2$
against $M{=}2,4,8,12,16,24$ (24 points, every point converged to
$10^{-7}$), and the $S{=}1$ row of the 1/4-core sweep on the same
tiling, against the same-geometry pure-XS OpenMC references ($1.3347260$
UO$_2$, $1.1851053$ MOX, $1.1872176$ core; each
$\approx\pm3.3$--$3.6$ pcm). The contrast with the old closure of Study
II is direct, but with a caveat: the old closure ran on the $3\times3$
tiling, where the MOX assembly sat at $-98.9$ pcm ($S{=}1$, $M{=}24$)
and $-33.5$ pcm at its best, while the $A{=}3$ closure runs on the
$2\times2$ fuel-split tiling, where it sits at $-0.9$ pcm at $M{=}12$,
$S{=}1$, inside the Monte Carlo standard error; on the $S{=}2$ row the
same tiling sits at $-9.8$ pcm at $M{=}24$. The two tilings are not the
same mesh ($4624$ vs $2601$ rectangles per assembly at $S{=}1$), so this
contrast measures the new setting --- finer fuel split together with the
$A{=}3$ angular closure --- as a whole rather than the angular order in
isolation. On the UO$_2$ assembly the best point is $-0.5$ pcm ($S{=}2$,
$M{=}24$) and $-2.8$ pcm on the $S{=}1$ row ($M{=}8$), both inside the
$\pm3.6$ pcm reference uncertainty; the $M$-convergence is monotonic in
signed $\Delta k$ from $M{=}2$ onward on both assemblies; with only
$S{=}1,2$ computed, this scan does not rule out a $S{\ge}3$
non-monotonicity of the Study-II type. The 1/4 core shows the same fast
angular convergence on its $S{=}1$ row --- $-135.6$, $-34.2$ and
$-1.7$ pcm at $M{=}2$, $4$, $8$ --- the core reaches the Monte Carlo
standard error at $M{=}8$, where the assemblies need $M{=}12$--$24$; the
row then overshoots mildly ($+6.2$ pcm at $M{=}12$, $+9.1$ pcm at
$M{=}16$ and $+11.0$ pcm at $M{=}24$). The whole $S{=}2$ row is in
progress.

<figure id="fig:a3mconv" data-latex-placement="H">
<img src="c5g7_rect4_a3_m_conv.png" style="width:98.0%" />
<figcaption><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>A</mi><mo>=</mo><mn>3</mn></mrow><annotation encoding="application/x-tex">A{=}3</annotation></semantics></math>
closure,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2</mn><mo>×</mo><mn>2</mn></mrow><annotation encoding="application/x-tex">2\times2</annotation></semantics></math>
fuel-split tiling. Top:
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi mathvariant="normal">Δ</mi><mi>k</mi></mrow><annotation encoding="application/x-tex">\Delta k</annotation></semantics></math>
versus
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>M</mi><annotation encoding="application/x-tex">M</annotation></semantics></math>
for
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>=</mo><mn>1</mn><mo>,</mo><mn>2</mn></mrow><annotation encoding="application/x-tex">S{=}1,2</annotation></semantics></math>
on the
UO<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><msub><mi></mi><mn>2</mn></msub><annotation encoding="application/x-tex">_2</annotation></semantics></math>
and MOX
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>17</mn><mo>×</mo><mn>17</mn></mrow><annotation encoding="application/x-tex">17\times17</annotation></semantics></math>
assemblies (all 24 points converged to
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><msup><mn>10</mn><mrow><mi>−</mi><mn>7</mn></mrow></msup><annotation encoding="application/x-tex">10^{-7}</annotation></semantics></math>)
and the completed
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>=</mo><mn>1</mn></mrow><annotation encoding="application/x-tex">S{=}1</annotation></semantics></math>
row of the 1/4 core
(<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>204</mn><mo>×</mo><mn>204</mn></mrow><annotation encoding="application/x-tex">204\times204</annotation></semantics></math>
grid; the
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>=</mo><mn>2</mn></mrow><annotation encoding="application/x-tex">S{=}2</annotation></semantics></math>
row is in progress). Bottom: pin-averaged fission-rate
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi mathvariant="normal">Σ</mi><mi>f</mi></msub><mi>ϕ</mi></mrow><annotation encoding="application/x-tex">\Sigma_f\phi</annotation></semantics></math>
maximum relative error (percent) versus
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>M</mi><annotation encoding="application/x-tex">M</annotation></semantics></math>,
against the same-geometry pure-XS OpenMC reference power maps, fuel pins
only
(<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>264</mn><annotation encoding="application/x-tex">264</annotation></semantics></math>
assembly pins; core rows cropped to the
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2</mn><mo>×</mo><mn>2</mn></mrow><annotation encoding="application/x-tex">2\times2</annotation></semantics></math>
fuel block,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>1056</mn><annotation encoding="application/x-tex">1056</annotation></semantics></math>
pins). Same-geometry references
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>1.3347260</mn><annotation encoding="application/x-tex">1.3347260</annotation></semantics></math>
/
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>1.1851053</mn><annotation encoding="application/x-tex">1.1851053</annotation></semantics></math>
/
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>1.1872176</mn><annotation encoding="application/x-tex">1.1872176</annotation></semantics></math>
(each
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mo>≈</mo><mi>±</mi><mn>3.3</mn></mrow><annotation encoding="application/x-tex">\approx\pm3.3</annotation></semantics></math>–<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>3.6</mn><annotation encoding="application/x-tex">3.6</annotation></semantics></math> pcm).
Best points:
UO<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><msub><mi></mi><mn>2</mn></msub><annotation encoding="application/x-tex">_2</annotation></semantics></math>
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>=</mo><mn>2</mn></mrow><annotation encoding="application/x-tex">S{=}2</annotation></semantics></math>,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>M</mi><mo>=</mo><mn>24</mn></mrow><annotation encoding="application/x-tex">M{=}24</annotation></semantics></math>
(<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.5</mn></mrow><annotation encoding="application/x-tex">-0.5</annotation></semantics></math> pcm),
MOX
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>=</mo><mn>1</mn></mrow><annotation encoding="application/x-tex">S{=}1</annotation></semantics></math>,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>M</mi><mo>=</mo><mn>12</mn></mrow><annotation encoding="application/x-tex">M{=}12</annotation></semantics></math>
(<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>0.9</mn></mrow><annotation encoding="application/x-tex">-0.9</annotation></semantics></math> pcm),
core
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>=</mo><mn>1</mn></mrow><annotation encoding="application/x-tex">S{=}1</annotation></semantics></math>,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>M</mi><mo>=</mo><mn>8</mn></mrow><annotation encoding="application/x-tex">M{=}8</annotation></semantics></math>
(<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>−</mi><mn>1.7</mn></mrow><annotation encoding="application/x-tex">-1.7</annotation></semantics></math> pcm,
best core point).</figcaption>
</figure>

#### Fission fields.

Figure [10](#fig:a3fission){reference-type="ref"
reference="fig:a3fission"} compares the pin-averaged fission
distributions against the OpenMC reference on the identical geometry:
the pin-by-pin relative error on the 264 fuel pins is $0.055\%$ RMS (max
$0.19\%$) on the UO$_2$ assembly ($\Delta k=-0.5$ pcm, $S{=}2$,
$M{=}24$) and $0.075\%$ RMS (max $0.22\%$) on the MOX assembly
($\Delta k=-9.8$ pcm) --- the field is reference-grade on both materials
at assembly size. On the 1/4 core at $S{=}1$, $M{=}8$
($\Delta k=-1.7$ pcm) the error over all 1056 fuel pins of the
$2\times2$ fuel block is $0.286\%$ RMS (max $0.64\%$).

<figure id="fig:a3fission" data-latex-placement="H">
<img src="c5g7_rect4_a3_fission.png" style="width:98.0%" />
<figcaption><math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>A</mi><mo>=</mo><mn>3</mn></mrow><annotation encoding="application/x-tex">A{=}3</annotation></semantics></math>
closure,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2</mn><mo>×</mo><mn>2</mn></mrow><annotation encoding="application/x-tex">2\times2</annotation></semantics></math>
fuel-split tiling: pin-averaged fission
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi mathvariant="normal">Σ</mi><mi>f</mi></msub><mi>ϕ</mi></mrow><annotation encoding="application/x-tex">\Sigma_f\phi</annotation></semantics></math>,
OpenPSN (left), OpenMC on the identical rectangular-rod geometry
(centre), pin-by-pin relative error on fuel pins (right; both maps
normalised to unit sum). Top:
UO<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><msub><mi></mi><mn>2</mn></msub><annotation encoding="application/x-tex">_2</annotation></semantics></math>
assembly,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>=</mo><mn>2</mn></mrow><annotation encoding="application/x-tex">S{=}2</annotation></semantics></math>,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>M</mi><mo>=</mo><mn>24</mn></mrow><annotation encoding="application/x-tex">M{=}24</annotation></semantics></math>,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi mathvariant="normal">Δ</mi><mi>k</mi><mo>=</mo><mi>−</mi><mn>0.5</mn></mrow><annotation encoding="application/x-tex">\Delta k=-0.5</annotation></semantics></math> pcm,
RMS
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.055</mn><mi>%</mi></mrow><annotation encoding="application/x-tex">0.055\%</annotation></semantics></math>
(max
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.19</mn><mi>%</mi></mrow><annotation encoding="application/x-tex">0.19\%</annotation></semantics></math>).
Middle: MOX assembly,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>=</mo><mn>2</mn></mrow><annotation encoding="application/x-tex">S{=}2</annotation></semantics></math>,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>M</mi><mo>=</mo><mn>24</mn></mrow><annotation encoding="application/x-tex">M{=}24</annotation></semantics></math>,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi mathvariant="normal">Δ</mi><mi>k</mi><mo>=</mo><mi>−</mi><mn>9.8</mn></mrow><annotation encoding="application/x-tex">\Delta k=-9.8</annotation></semantics></math> pcm,
RMS
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.075</mn><mi>%</mi></mrow><annotation encoding="application/x-tex">0.075\%</annotation></semantics></math>
(max
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.22</mn><mi>%</mi></mrow><annotation encoding="application/x-tex">0.22\%</annotation></semantics></math>).
Bottom: 1/4 core,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>=</mo><mn>1</mn></mrow><annotation encoding="application/x-tex">S{=}1</annotation></semantics></math>,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>M</mi><mo>=</mo><mn>8</mn></mrow><annotation encoding="application/x-tex">M{=}8</annotation></semantics></math>,
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi mathvariant="normal">Δ</mi><mi>k</mi><mo>=</mo><mi>−</mi><mn>1.7</mn></mrow><annotation encoding="application/x-tex">\Delta k=-1.7</annotation></semantics></math>
pcm, RMS
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.286</mn><mi>%</mi></mrow><annotation encoding="application/x-tex">0.286\%</annotation></semantics></math>
(max
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>0.64</mn><mi>%</mi></mrow><annotation encoding="application/x-tex">0.64\%</annotation></semantics></math>)
over all 1056 fuel pins, cropped to the
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mn>2</mn><mo>×</mo><mn>2</mn></mrow><annotation encoding="application/x-tex">2\times2</annotation></semantics></math>
fuel block.</figcaption>
</figure>

#### Fission-rate convergence across the sweep.

The bottom row of Figure [9](#fig:a3mconv){reference-type="ref"
reference="fig:a3mconv"} repeats the field comparison of
Figure [10](#fig:a3fission){reference-type="ref"
reference="fig:a3fission"} for every computed point of the sweep, with
the identical pin-averaged fission-rate convention ($\Sigma_f\phi$,
unit-sum normalised, fuel pins only). On the assemblies the field error
is essentially flat in $M$ --- $0.053$--$0.057\%$ RMS on UO$_2$ and
$0.075$--$0.081\%$ on MOX --- even though the eigenvalue error moves by
more than $300$ pcm over the same rows: the pin-averaged power shape is
established by $M{=}4$ and finer angular orders move it by at most
$0.007\%$ RMS across the rows; the maximum pin deviation stays within
$0.18$--$0.19\%$ (UO$_2$) and $0.21$--$0.23\%$ (MOX) over the whole
sweep. The 1/4-core field, cropped to the $2\times2$ fuel block as in
Figure [10](#fig:a3fission){reference-type="ref"
reference="fig:a3fission"}, sits a factor ${\sim}5$ higher
($0.28$--$0.43\%$ RMS, $0.64$--$0.86\%$ max) and decays from $M{=}2$ to
$M{=}8$ before plateauing along the full $S{=}1$ row ($0.638$--$0.647\%$
max from $M{=}8$ to $M{=}24$). This sweep does not separate the OpenMC
power-tally statistics from the residual method error at that floor; the
numbers are reported as the measured field agreement, not as a proven
limit. The $S{=}2$ core row beyond $M{=}2$ is still in progress.

#### Where the method now stands.

The sequence of this paper is, in retrospect, a sequence of error
sources being identified and removed one by one. The implementation
*reproduces* the published PSN results to a few pcm (Section 3). The
rectangular-node extension *removes the homogenisation model* and
exposes the next error: the $A{=}0$ angular closure is incomplete for
strongly coupled pin geometries (Study II: non-monotonic $M$ and $S$
convergence, a $-99$ pcm MOX bias at $M{=}24$ that the $S\times M$
refinement does not drive toward zero). The $A{=}3$ directional closure
of this section --- derived above from the original transport equation
through the exact parity elimination, the segment-resolved angular
moments and the matrix hyperbolic body --- *narrowed the angular
component of the error*: on the published checkerboard, its coarse-grid
error decays close to second order in $S$
($\sim$`<!-- -->`{=html}28-fold from $S{=}1$ to $6$ at $M{=}24$); the
sweep crosses the five-decimal reference within $0.1$ pcm (weak
absorber, $S{\approx}12$) --- a finite-range crossing, not a proven
asymptotic limit --- and the $S{=}20$ point shifts from $-3.8$ to
$-0.9$ pcm when the polar set is doubled to $I{=}60$, and on C5G7 the
same closure brings the MOX assembly --- the hardest case of the
benchmark --- to $-0.9$ pcm at $M{=}12$ ($S{=}1$), with
$0.055$--$0.075\%$-RMS fission fields at $S{=}2$, $M{=}24$, and the 1/4
core to $-1.7$ pcm at $M{=}8$ ($S{=}1$); on these points the method sits
within the same-geometry OpenMC standard error, although the scans do
not yet establish the limiting accuracy. The result is a diffusion-based
higher-order angular-mode directional PSN algorithm --- the
hyperbolic-plus-source spatial basis, Legendre angular expansion within
each segment, direction-matrix body closure --- on a compact nodal grid,
without a discrete-ordinates or characteristics mesh.

## Comparison with the published diffusion / SN / nodal landscape

The original NEA report [@nea2003] tabulates 20 deterministic codes on
C5G7-2D. Table [4](#tab:lit){reference-type="ref" reference="tab:lit"}
reproduces the diffusion- and nodal-family values we could extract from
it, together with recent P1/SP3/SN/nodal literature results and our own
OpenPSN result.

+----------------+-------------------------+-------------------------+----------------+
| method class   | code                    | $\kappa_{\mathrm{eff}}$ | $\Delta$ (pcm) |
+:===============+:========================+:=======================:+:==============:+
| *Pure P1 diffusion*                                                                 |
+----------------+-------------------------+-------------------------+----------------+
| diffusion      | CRONOS2                 | 1.18323                 | $-323$         |
| (FEM)          |                         |                         |                |
+----------------+-------------------------+-------------------------+----------------+
| P1 (pin-hom)   | STELLA-P1 [@stella2017] | 1.18426                 | $-290$ (vs     |
|                |                         |                         | PEACH)         |
+----------------+-------------------------+-------------------------+----------------+
| *SN*                                                                                |
+----------------+-------------------------+-------------------------+----------------+
| SN-FEM         | CRONOS2-SN              | 1.18338                 | $-308$         |
+----------------+-------------------------+-------------------------+----------------+
| SN-FDM         | DORT-GRS                | 1.18482                 | $-164$         |
+----------------+-------------------------+-------------------------+----------------+
| SN-FDM         | DORT-ORNL               | 1.18496                 | $-150$         |
+----------------+-------------------------+-------------------------+----------------+
| SN-FDM         | TWODANT                 | 1.18668                 | $+22$          |
+----------------+-------------------------+-------------------------+----------------+
| SN (diamond    | PARTISN                 | 1.18637                 | $-9$           |
| diff.)         |                         |                         |                |
+----------------+-------------------------+-------------------------+----------------+
| SN             | PERICLES                | 1.18658                 | $+12$          |
| (unstructured) |                         |                         |                |
+----------------+-------------------------+-------------------------+----------------+
| *High-order nodal $P_n$ / surf. harmonics*                                          |
+----------------+-------------------------+-------------------------+----------------+
| $P_n$-nodal    | VARIANT-SE              | 1.18495                 | $-151$         |
| (FEM)          |                         |                         |                |
+----------------+-------------------------+-------------------------+----------------+
| $P_n$-nodal +  | VARIANT-ISE             | 1.18745                 | $+99$          |
| int. transp.   |                         |                         |                |
+----------------+-------------------------+-------------------------+----------------+
| surface        | SUHAM-2D                | 1.18628                 | $-18$          |
| harm. (G3/P2)  |                         |                         |                |
+----------------+-------------------------+-------------------------+----------------+
| *MOC*                                                                               |
+----------------+-------------------------+-------------------------+----------------+
| MOC            | CHAPLET                 | 1.18656                 | $+10$          |
+----------------+-------------------------+-------------------------+----------------+
| MOC            | MCCG3D                  | 1.18657                 | $+11$          |
+----------------+-------------------------+-------------------------+----------------+
| MOC            | DeCART                  | 1.18660                 | $+14$          |
+----------------+-------------------------+-------------------------+----------------+
| MOC            | APOLLO2                 | 1.18618                 | $-28$          |
+----------------+-------------------------+-------------------------+----------------+
| MOC            | CRX                     | 1.18813                 | $+167$         |
+----------------+-------------------------+-------------------------+----------------+
| MOC            | UNKGRO                  | 1.18523                 | $-123$         |
| (stoch. rays)  |                         |                         |                |
+----------------+-------------------------+-------------------------+----------------+
| *This work*                                                                         |
+----------------+-------------------------+-------------------------+----------------+
| PSN nodal      | OpenPSN                 | 1.1872011               | $+74.1$        |
| (rect. node,   |                         |                         |                |
| $A{=}3$,       |                         |                         |                |
| $S{=}1$,       |                         |                         |                |
| $M{=}8$)       |                         |                         |                |
+----------------+-------------------------+-------------------------+----------------+

: C5G7-2D $\kappa_{\mathrm{eff}}$ by method class (2-D multi-group,
homogenised pin model unless noted). NEA-2003 $k$ values (its Table 17)
plus recent P1/SP3 literature. $\Delta$ = $(k-1.18646)\times10^{5}$ in
pcm, except where stated (STELLA vs its PEACH reference 1.18716). The
original NEA table prints the relative per-cent error against its MCNP
reference 1.18655; the pcm values here are recomputed from the printed
$k$ against 1.18646. {#tab:lit}

*Readings.*

- **Pure P1 diffusion is systematically low** --- CRONOS2 $-323$ pcm
  ($-0.27\%$), STELLA-P1 $-290$ pcm vs its PEACH MOC reference 1.18716:
  no transport correction and no self shielding.

- **SN spans $-308$ (CRONOS2-SN) to $+22$ pcm (TWODANT); DORT
  $-164$/$-150$; the better-differenced / unstructured variants PARTISN
  ($-9$) and PERICLES ($+12$) within $\pm0.01\%$.**

- **High-order nodal $P_n$ / surface harmonics** spans $-151$
  (VARIANT-SE) to $+99$ pcm (VARIANT-ISE); SUHAM-2D $-18$; VARIANT-ISE
  also has the best pin power in the original table (MRE 0.11%).

- **MOC (explicit pin geometry) clusters within $\pm28$ pcm** (CHAPLET
  $+10$, MCCG3D $+11$, DeCART $+14$, APOLLO2 $-28$), with two outliers
  (CRX $+167$, UNKGRO $-123$); TWODANT is SN-FDM ($+22$); the
  high-fidelity Rattlesnake converged limit is $1.186446$
  ($\sim$`<!-- -->`{=html}1 pcm from the 1.186456 reference).

- **OpenPSN in this work.** The C5G7 studies of Sections 4.2--4.4
  compare two representations of the pin geometry. With the
  self-shielding-aware flux-weighted pinwise homogenisation, the best
  points of the $S\times M$ sweep are $+0.0$ pcm (UO$_2$ assembly),
  $-1.5$ pcm (MOX assembly) and $-7.7$ pcm (1/4 core) vs. the OpenMC
  homogenised references --- the two assemblies inside their own Monte
  Carlo standard errors; with the rectangular-node model that removes
  homogenisation entirely, the $A{=}3$ directional closure on the
  $2\times2$ fuel-split tiling (Section 4.4) is at $-1.7$ pcm vs. the
  OpenMC run on the identical rectangular-rod geometry ($1.1872176$,
  $\pm3.4$ pcm) at $S{=}1$, $M{=}8$ --- inside the reference standard
  error; the $S{=}1$ row then moves to $+6.2$ pcm ($M{=}12$) and
  $+9.1$ pcm ($M{=}16$), so $M{=}8$ is quoted as the point within the
  reference uncertainty, not as a converged angular limit; the completed
  $S{=}1$ row ends at $+11.0$ pcm ($M{=}24$), with only the $S{=}2$ row
  still in progress --- i.e. $+74.1$ pcm vs. $1.18646$; the pin-level
  fission maps agree to $0.055$--$0.075\%$ RMS on the assemblies and
  $0.286\%$ RMS on the core. The pinwise-homogenised point sits inside
  the MOC reference cluster ($\pm37$ pcm, plus the better-differenced SN
  and unstructured codes); the rectangular-node row is anchored to the
  rectangular-rod idealisation, whose own OpenMC reference sits
  $+75.8$ pcm above $1.18646$.

Two published outliers are worth noting for completeness: COHINT ($P_2$
interface currents, $-1125$ pcm) and HELIOS (collision probability,
$+675$ pcm), both far outside the converged cluster and attributable to
their respective current/transport approximations.

# Conclusions

We built **OpenPSN**, a clean, multi-group, sparse, vectorised
implementation of the Phase Space Nodal method, and validated it on two
levels. On the source paper's problems it reproduces all 248 published
eigenvalue data points to a few pcm (full point-by-point comparison in
Appendix A). The sparse path is additionally verified against the plain
dense path to $10^{-15}$, and extends to the $10{,}404$-node C5G7
refinement, which the dense formulation cannot address. We also carried
the rectangular-node generalisation of the PSN node operators through to
explicit closed-form coefficients for arbitrary per-node widths, giving
an implementation with explicit, un-homogenised intra-node fuel/water
structure and the node-by-node verification of Section 2.2.

On the C5G7 MOX benchmark the studies separate the two error sources
that a nodal code carries. In the *pinwise-homogenised* study, the
$S\times M$ sweep against a directly computed OpenMC reference on the
same homogenised data shows the discretisation converging: their best
points ($S{=}5$--$6$, $M{\ge}8$) sit inside the Monte Carlo standard
error ($\approx$`<!-- -->`{=html}3.3--3.5 pcm; $+0.0$ pcm UO$_2$,
$-1.5$ pcm MOX), and the best core point (with the self-shielding-aware
flux-weighted recipe) is $-7.7$ pcm versus the OpenMC run on the same
homogenised data, i.e. $+28.6$ pcm versus the high-fidelity reference
$1.18646$ --- the difference decomposes as $28.6=(-7.7)+36.3$ pcm: a
$-7.7$ pcm method-level term (PSN against the same-data OpenMC run) plus
a $+36.3$ pcm term between the homogenised OpenMC model itself and the
resolved-pin reference, which also carries the reference's own
statistical and state differences and is not uniquely attributable to
homogenisation. The best-point pin-power fields agree with the
same-recipe OpenMC maps to $0.07\%$ RMS (max $0.25\%$); a small
power-distribution RMS does not by itself exclude partial cancellation
of eigenvalue errors, so the $-7.7$ pcm term is reported as a
method-level discrepancy of unresolved origin. In the *rectangular-node*
study the homogenisation model is removed entirely. On the $3\times3$
fuel-square tiling with the original $A{=}0$ closure, with
un-homogenised 7-group material data the 1/4-core eigenvalue converges
monotonically with $M$ against the OpenMC same-geometry reference
($-184.3\rightarrow-110.9\rightarrow-61.1\rightarrow-10.2$ pcm for
$M=8\rightarrow24$, $S{=}1$): the residual $-10.2$ pcm is consistent
with a method-level bias in the angular closure; the available scans do
not separate the angular, spatial, and coupled contributions. On the
$2\times2$ fuel-split tiling --- a different, finer tiling ($4624$ vs
$2601$ rectangles per assembly at the same $S$), so this comparison
measures the new setting as a whole rather than the angular order alone
--- the $A{=}3$ directional closure of Section 4.4 brings the assemblies
to $-0.9$ pcm (MOX, $S{=}1$, $M{=}12$) and $-0.5$ pcm (UO$_2$, $S{=}2$,
$M{=}24$) --- both inside their Monte Carlo standard errors --- and the
1/4 core to $-1.7$ pcm at $S{=}1$, $M{=}8$, inside the $\pm3.4$ pcm
reference standard error (the row then moves to $+6.2$/$+9.1$ pcm at
$M{=}12/16$); the pin-averaged fission maps $\Sigma_f\phi$ agree with
OpenMC to $0.055$--$0.075\%$ RMS on the assemblies and $0.286\%$ RMS
over all $1056$ fuel pins of the core. On the published checkerboard the
same closure's coarse-grid error decays close to second order in $S$;
the sweep crosses the five-decimal reference within $0.1$ pcm (weak
absorber, $S{\approx}12$) --- a finite-range crossing, not a proven
asymptotic limit --- and the $S{=}20$ point shifts from $-3.8$ to
$-0.9$ pcm when the polar set is doubled to $I{=}60$. Placed against the
20-code NEA landscape and the recent P1/SP3/SN/nodal literature, the
pinwise-homogenised result sits inside the high-fidelity MOC reference
cluster; the rectangular-node result is anchored to the rectangular-rod
idealisation --- whose own OpenMC reference sits $+75.8$ pcm above
$1.18646$ --- and is judged against the same-geometry reference rather
than the circular-rod cluster. Selected assembly and core configurations
agree with their same-geometry OpenMC references within one reported
standard error; this agreement is point-specific --- the homogenised
core retains a $-7.7$ pcm discrepancy, and the $A{=}3$ core moves beyond
one standard error as $M$ increases from $8$ to $12$ and $16$ --- and
further controlled refinement is required to establish the limiting
accuracy. In both cases the method uses a compact nodal spatial
representation rather than a discrete-ordinates or characteristics mesh.

# Acknowledgments {#acknowledgments .unnumbered}

This work received no external funding. The implementation of OpenPSN
(sparse assembly, vectorised node operators, rectangular-node extension)
was carried out with the assistance of a large language model (Qwen3.8
27B).

::: thebibliography
11

Y.-A. Chao, Z. Li, and G. Chen, "Diffusion-based phase space nodal
method (PSN): Solving the neutron transport equation with a diffusion
code," *Annals of Nuclear Energy* **240** (2027) 112707
 [doi:10.1016/j.anucene.2026.112707](https://doi.org/10.1016/j.anucene.2026.112707).

*Benchmark on Deterministic Transport Calculations Without Spatial
Homogenisation: A 2-D/3-D MOX Fuel Assembly Benchmark*, OECD/NEA NSC
document NEA/NSC/DOC(2003)16 (2003). 2-D 20-code comparison, Tables
16--20.

M. A. Smith, E. E. Lewis, and B.-C. Na, "Benchmark on deterministic 2-D
MOX fuel assembly transport calculations without spatial
homogenization," *Progress in Nuclear Energy* **45** (2004) 107--118
 [doi:10.1016/j.pnueene.2004.09.003](https://doi.org/10.1016/j.pnueene.2004.09.003).

J. Hou, K. N. Ivanov, V. F. Boyarinov, and P. A. Fomichenko, "OECD/NEA
benchmark for time-dependent neutron transport calculations without
spatial homogenization," *Nuclear Engineering and Design* **317** (2017)
177--189
 [doi:10.1016/j.nucengdes.2017.02.008](https://doi.org/10.1016/j.nucengdes.2017.02.008).

C. N. McGraw, M. L. Adams, W. D. Hawkins, M. P. Adams, and T. Smith,
"Accuracy of the linear discontinuous Galerkin method for reactor
analyses with resolved fuel pins," in *PHYSOR 2014*, Kyoto, Japan, 2014.
(2-D C5G7 resolved-fuel-pin high-fidelity
$\kappa_{\mathrm{eff}}=1.186456$.)

Y. Wang, M. D. DeHart, D. R. Gaston, F. N. Gleicher, R. C. Martineau,
J. Ortensi, J. W. Peterson, and S. Schunert, "Convergence study of
Rattlesnake solutions for the two-dimensional C5G7 MOX benchmark,"
INL/CON-15-34115, ANS MC2015. Converged high-fidelity
$\kappa_{\mathrm{eff}}=1.186446$ ($\sim$`<!-- -->`{=html}1 pcm from the
Ref. [@mcgraw2014] reference).

P. K. Romano, N. E. Horelik, B. R. Herman, A. G. Nelson, B. Forget, and
K. Smith, "OpenMC: A state-of-the-art Monte Carlo code for research and
development," *Annals of Nuclear Energy* **82** (2015) 90--97
 [doi:10.1016/j.anucene.2014.07.048](https://doi.org/10.1016/j.anucene.2014.07.048).

C. Tang, "Development and verification of a SP3 code using semi-analytic
nodal method for pin-by-pin calculation," in *M&C 2017*, Jeju, Korea,
2017. (C5G7-2D pin-hom: STELLA-P1 $1.18426$, STELLA-SP3 $1.18561$; PEACH
MOC reference $1.18716$.)

C. Tang and S. Zhang, "Development and verification of an MOC code
employing assembly modular ray tracing and efficient acceleration
techniques," *Annals of Nuclear Energy* **36** (2009) 1013--1020
 [doi:10.1016/j.anucene.2009.06.007](https://doi.org/10.1016/j.anucene.2009.06.007).
:::

::: landscape
# Full point-by-point reproduction of the published PSN results

Table [1](#tab:repro){reference-type="ref" reference="tab:repro"}
summarised the maxima. Tables [5](#tab:appA){reference-type="ref"
reference="tab:appA"}--[8](#tab:appD){reference-type="ref"
reference="tab:appD"} give the complete 240-cell comparison for the
three printed eigenvalue grids (checkerboard weak/strong, BWR without
Gd) plus the with-Gd case, in the same "this work / paper ($\Delta$)"
cell format. Row labels: $S$ = spatial subdivision of the checkerboard
grid, $N$ = spatial grid of the BWR bundle. The remaining 8 points
(restricted $M=24$, $S=1,2,4,8$, weak/strong) agree with the printed
Figs. 4--5 within the plot-reading uncertainty; the full per-point data
are shipped with the code.

  **$S$**                                  $M$=2                   $M$=4                  $M$=8                  $M$=12                 $M$=16                 $M$=24
  ------------------------------- ----------------------- ----------------------- ---------------------- ---------------------- ---------------------- ----------------------
  1$\times$`<!-- -->`{=html}1      $+136.2/136\ (+0.2)$    $+389.0/388\ (+1.0)$    $+703.7/703\ (+0.7)$   $+793.9/793\ (+0.9)$   $+835.2/835\ (+0.2)$   $+873.5/873\ (+0.5)$
  2$\times$`<!-- -->`{=html}2      $-275.4/-276\ (+0.6)$   $-143.6/-144\ (+0.4)$   $+154.9/154\ (+0.9)$   $+217.2/217\ (+0.2)$   $+244.4/244\ (+0.4)$   $+268.2/268\ (+0.2)$
  3$\times$`<!-- -->`{=html}3      $-374.4/-375\ (+0.6)$   $-263.2/-264\ (+0.8)$    $+19.1/18\ (+1.1)$     $+79.3/79\ (+0.3)$    $+104.4/104\ (+0.4)$   $+124.1/123\ (+1.1)$
  4$\times$`<!-- -->`{=html}4      $-412.0/-413\ (+1.0)$   $-309.4/-310\ (+0.6)$   $-34.6/-35\ (+0.4)$     $+24.9/24\ (+0.9)$     $+48.2/47\ (+1.2)$     $+66.9/66\ (+0.9)$
  5$\times$`<!-- -->`{=html}5      $-430.1/-431\ (+0.9)$   $-332.1/-333\ (+0.9)$   $-61.2/-62\ (+0.8)$     $-2.7/-3\ (+0.3)$      $+20.2/19\ (+1.2)$     $+38.2/38\ (+0.2)$
  6$\times$`<!-- -->`{=html}6      $-440.3/-441\ (+0.7)$   $-344.8/-346\ (+1.2)$   $-76.3/-77\ (+0.7)$    $-18.5/-19\ (+0.5)$      $+4.1/3\ (+1.1)$      $+21.8/21\ (+0.8)$
  7$\times$`<!-- -->`{=html}7      $-446.5/-447\ (+0.5)$   $-352.7/-353\ (+0.3)$   $-85.7/-86\ (+0.3)$    $-28.4/-29\ (+0.6)$     $-6.1/-7\ (+0.9)$      $+11.4/11\ (+0.4)$
  8$\times$`<!-- -->`{=html}8      $-450.5/-451\ (+0.5)$   $-357.9/-359\ (+1.1)$   $-92.0/-93\ (+1.0)$    $-35.0/-36\ (+1.0)$    $-12.9/-14\ (+1.1)$      $+4.4/4\ (+0.4)$
  9$\times$`<!-- -->`{=html}9      $-453.4/-454\ (+0.6)$   $-361.5/-362\ (+0.5)$   $-96.3/-97\ (+0.7)$    $-39.6/-40\ (+0.4)$    $-17.7/-18\ (+0.3)$     $-0.5/-1\ (+0.5)$
  10$\times$`<!-- -->`{=html}10    $-455.4/-456\ (+0.6)$   $-364.1/-365\ (+0.9)$   $-99.5/-100\ (+0.5)$   $-43.0/-44\ (+1.0)$    $-21.2/-22\ (+0.8)$     $-4.1/-5\ (+0.9)$

  : Fig. 3-A, weak absorption, generic ($I=30$): $\Delta$ vs the printed
  values (reference $\kappa_{\mathrm{eff}}=1.12974$). $S$ = spatial
  subdivision. {#tab:appA}

*Note:* each cell is "this work / paper ($\Delta$), $\Delta$ = this work
$-$ paper, in pcm; paper values as printed in Ref. [@psn2027]; the
Fig. 10/12 references are read from the printed contour plots
($\pm1$--$2$ pcm reading uncertainty).

  **$S$**                                   $M$=2                     $M$=4                    $M$=8                    $M$=12                   $M$=16                   $M$=24
  ------------------------------- ------------------------- ------------------------- ------------------------ ------------------------ ------------------------ ------------------------
  1$\times$`<!-- -->`{=html}1       $+591.0/591\ (+0.0)$     $+1762.4/1763\ (-0.6)$    $+3199.2/3199\ (+0.2)$   $+3596.7/3597\ (-0.3)$   $+3776.6/3777\ (-0.4)$   $+3941.3/3941\ (+0.3)$
  2$\times$`<!-- -->`{=html}2      $-1534.9/-1535\ (+0.1)$    $-790.2/-790\ (-0.2)$     $+807.9/808\ (-0.1)$    $+1132.7/1133\ (-0.3)$   $+1271.3/1271\ (+0.3)$   $+1392.4/1393\ (-0.6)$
  3$\times$`<!-- -->`{=html}3      $-2068.0/-2068\ (+0.0)$   $-1425.6/-1425\ (-0.6)$    $+113.2/113\ (+0.2)$     $+431.2/431\ (+0.2)$     $+563.2/563\ (+0.2)$     $+666.4/666\ (+0.4)$
  4$\times$`<!-- -->`{=html}4      $-2272.6/-2272\ (-0.6)$   $-1676.0/-1676\ (+0.0)$   $-170.0/-170\ (+0.0)$     $+145.3/145\ (+0.3)$     $+269.2/269\ (+0.2)$     $+367.8/368\ (-0.2)$
  5$\times$`<!-- -->`{=html}5      $-2371.6/-2372\ (+0.4)$   $-1799.5/-1799\ (-0.5)$   $-311.9/-312\ (+0.1)$      $-1.4/-1\ (-0.4)$       $+120.5/121\ (-0.5)$     $+216.1/216\ (+0.1)$
  6$\times$`<!-- -->`{=html}6      $-2427.0/-2427\ (+0.0)$   $-1869.2/-1869\ (-0.2)$   $-393.2/-393\ (-0.2)$     $-86.1/-86\ (-0.1)$       $+34.1/34\ (+0.1)$      $+128.3/128\ (+0.3)$
  7$\times$`<!-- -->`{=html}7      $-2461.0/-2461\ (+0.0)$   $-1912.3/-1912\ (-0.3)$   $-444.1/-444\ (-0.1)$    $-139.3/-139\ (-0.3)$     $-20.7/-21\ (+0.3)$       $+72.6/72\ (+0.6)$
  8$\times$`<!-- -->`{=html}8      $-2483.3/-2483\ (-0.3)$   $-1940.8/-1941\ (+0.2)$   $-478.0/-478\ (+0.0)$    $-175.1/-175\ (-0.1)$     $-57.6/-58\ (+0.4)$       $+35.0/34\ (+1.0)$
  9$\times$`<!-- -->`{=html}9      $-2498.8/-2499\ (+0.2)$   $-1960.6/-1961\ (+0.4)$   $-501.8/-502\ (+0.2)$    $-200.2/-200\ (-0.2)$     $-83.6/-84\ (+0.4)$        $+8.4/7\ (+1.4)$
  10$\times$`<!-- -->`{=html}10    $-2510.0/-2510\ (-0.0)$   $-1974.9/-1975\ (+0.1)$   $-519.1/-519\ (-0.1)$    $-218.6/-219\ (+0.4)$    $-102.6/-103\ (+0.4)$     $-11.1/-12\ (+0.9)$

  : Fig. 3-B, strong absorption, generic ($I=30$): $\Delta$ vs the
  printed values (reference $\kappa_{\mathrm{eff}}=0.51673$). $S$ =
  spatial subdivision. {#tab:appB}

*Note:* each cell is "this work / paper ($\Delta$), $\Delta$ = this work
$-$ paper, in pcm; paper values as printed in Ref. [@psn2027]; the
Fig. 10/12 references are read from the printed contour plots
($\pm1$--$2$ pcm reading uncertainty).

  **$N$**           $M$=4                  $M$=8                 $M$=12                  $M$=16                  $M$=20                  $M$=24
  --------- ---------------------- --------------------- ----------------------- ----------------------- ----------------------- -----------------------
  1          $+126.4/123\ (+3.4)$   $-74.8/-78\ (+3.2)$   $-122.2/-125\ (+2.8)$   $-140.3/-144\ (+3.7)$   $-149.1/-152\ (+2.9)$   $-154.2/-157\ (+2.8)$
  2          $+206.5/203\ (+3.5)$   $+28.3/25\ (+3.3)$     $-15.6/-19\ (+3.4)$     $-31.9/-35\ (+3.1)$     $-40.1/-43\ (+2.9)$     $-44.9/-48\ (+3.1)$
  3          $+226.2/223\ (+3.2)$   $+54.8/51\ (+3.8)$      $+12.5/9\ (+3.5)$       $-3.6/-7\ (+3.4)$      $-11.6/-15\ (+3.4)$     $-16.3/-20\ (+3.7)$
  4          $+233.7/230\ (+3.7)$   $+65.2/62\ (+3.2)$     $+23.8/20\ (+3.8)$       $+7.8/5\ (+2.8)$        $-0.1/-3\ (+2.9)$       $-4.6/-8\ (+3.4)$
  5          $+237.4/234\ (+3.4)$   $+70.4/67\ (+3.4)$     $+29.4/26\ (+3.4)$      $+13.6/10\ (+3.6)$       $+5.8/2\ (+3.8)$        $+1.3/-2\ (+3.3)$
  6          $+239.4/236\ (+3.4)$   $+73.3/70\ (+3.3)$     $+32.7/29\ (+3.7)$      $+16.9/14\ (+2.9)$       $+9.1/6\ (+3.1)$        $+4.7/1\ (+3.7)$
  7          $+240.6/237\ (+3.6)$   $+75.1/72\ (+3.1)$     $+34.7/31\ (+3.7)$      $+19.0/16\ (+3.0)$       $+11.3/8\ (+3.3)$       $+6.9/4\ (+2.9)$
  8          $+241.4/238\ (+3.4)$   $+76.3/73\ (+3.3)$     $+36.0/33\ (+3.0)$      $+20.4/17\ (+3.4)$       $+12.7/9\ (+3.7)$       $+8.3/5\ (+3.3)$
  9          $+242.0/239\ (+3.0)$   $+77.2/74\ (+3.2)$     $+37.0/34\ (+3.0)$      $+21.4/18\ (+3.4)$      $+13.7/10\ (+3.7)$       $+9.3/6\ (+3.3)$
  10         $+242.4/239\ (+3.4)$   $+77.8/74\ (+3.8)$     $+37.7/34\ (+3.7)$      $+22.1/19\ (+3.1)$      $+14.5/11\ (+3.5)$       $+10.1/7\ (+3.1)$

  : Fig. 10, BWR 12-pin bundle without Gd (2-group): $\Delta$ vs the
  values read from the printed contour plot (reference
  $\kappa_{\mathrm{eff}}=1.18797$). $N$ = spatial grid. {#tab:appC}

*Note:* each cell is "this work / paper ($\Delta$), $\Delta$ = this work
$-$ paper, in pcm; paper values as printed in Ref. [@psn2027]; the
Fig. 10/12 references are read from the printed contour plots
($\pm1$--$2$ pcm reading uncertainty).

  **$N$**    $M$=4    $M$=8     $M$=12    $M$=16    $M$=20    $M$=24
  --------- ------- ---------- --------- --------- --------- --------
  1          -605    **+311**    +565      +681      +746      +788
  2          -998      -152     **+67**    +157      +203      +230
  3          -1140     -253     **-22**     +73      +123      +152
  4          -1207     -283     **-45**     +53      +103      +133
  5          -1264     -318       -72     **+28**     +79      +110
  6          -1310     -351       -98     **+4**      +57      +88
  7          -1346     -380      -122     **-18**     +36      +67
  8          -1375     -405      -144       -37     **+17**    +49
  9          -1397     -427      -162       -54     **+1**     +33
  10         -1415     -445      -178       -69     **-13**    +20

  : Fig. 12, BWR 12-pin bundle with Gd (2-group): this work only (the
  printed figure is a contour plot). $N$ = spatial grid. Bold = best $M$
  in the row (min $|\Delta|$); the value of $M$ at which $|\Delta|$ is
  smallest shifts to lower $M$ as $N$ grows, exactly the Sec. 4.4 effect
  of the source paper. Best point overall: $N=9$, $M=20$, $+1.1$ pcm.
  {#tab:appD}

*Note:* each cell is "this work / paper ($\Delta$), $\Delta$ = this work
$-$ paper, in pcm; paper values as printed in Ref. [@psn2027]; the
Fig. 10/12 references are read from the printed contour plots
($\pm1$--$2$ pcm reading uncertainty).
:::
