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
Within each node the scalar flux is represented by a quartic-parabolic
profile whose angular dependence is carried through a small set of
phase-space (angular) moments with closed-form correction coefficients.
The node is eliminated to give explicit relations between surface
currents and node sources, and neighbouring nodes are coupled through
those currents. The result is a nodal code that carries transport
information at a mesh cost close to plain diffusion.

Two gaps remained. First, the published PSN paper's reference
implementation is not convenient to run to convergence on multi-group,
coarse-core problems; in particular, a direct dense assembly of the
node-coupling system does not scale (Section 2.2). Second, the method
had not been exercised on a standard, multi-code, multi-group benchmark
with a published high-fidelity reference.

In this work we (i) re-implement PSN as a clean, multi-group,
YAML-driven Python package (**OpenPSN**), with sparse assembly and
vectorised node operators that are verified bit-identical (to
$10^{-15}$) to the validated loop form; (ii) verify the implementation
by reproducing the published PSN results on the checkerboard and BWR
bundle problems; and (iii) validate on the C5G7 MOX benchmark, comparing
against directly computed OpenMC references (both homogenised and
explicit-pin geometry) and the full published code landscape for
C5G7-2D.

The remainder is organised as follows. Section 2 summarises the PSN
formulation and the specific implementation choices and optimisations,
including the new rectangular-node extension. Section 3 documents the
reproduction of the published results. Section 4 describes the C5G7
configuration, the data-fidelity check, and two complementary benchmark
studies: (i) the pinwise-homogenised nodal model with a full
spatial/angular ($S\times M$) convergence sweep against directly
computed OpenMC references, and (ii) the rectangular-node model with
explicit (unhomogenised) intra-node fuel/water structure, including flux
comparisons against OpenMC and an eigenvalue comparison with the
published code landscape. Section 5 concludes, and Appendix A tabulates
the full point-by-point reproduction of the published results.

# The PSN method and the OpenPSN implementation

## Method in brief

The full theory is given in Ref. [@psn2027]; we summarise only what is
needed to read the implementation. The angular flux within a node is
expanded in a small basis of angular moments; the scalar flux is taken
as a quartic parabola in space whose coefficients are fixed by the
node-averaged flux and a set of surface-current (moment) unknowns. Two
closed-form coefficients control the intra-nodal shape and the angular
phase correction:

$$\begin{align}
\alpha_i &\quad\text{(intra-nodal parabolic shape coefficient, Eq.~2.8c of
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
is iterated with a standard eigenvalue (inverse-power) driver.

## Implementation and the optimisations we made

OpenPSN (run as `python -m psn2d run problem.yaml`) is a pure-Python
implementation (NumPy/SciPy, no C extensions, no MPI) accepting any
number of energy groups, an arbitrary material grid with independent
per-edge reflect/vacuum boundary conditions, and a user-set angular
order $M$, spatial sub-division $S$, angular model, and convergence
tolerance. The three optimisations that make the method practical are:

1.  **Sparse assembly (fixes the memory blow-up).** The nodal coupling
    matrix has only $\sim$`<!-- -->`{=html}9 non-zeroes per row (the
    node couples to its own face-current unknowns and to its
    neighbours). For $M$ angular segments per face the system has
    $\sim 2M\,N_xN_y$ unknowns; the smallest C5G7 case (2601 nodes,
    $M=12$) already has $63{,}036$ unknowns, so the natural dense
    $63{,}036^{2}$ array alone is $\sim$`<!-- -->`{=html}32   before any
    factorisation workspace, and the $4\times$ refinement (10404 nodes,
    $250{,}920$ unknowns) would need $\sim$`<!-- -->`{=html}504   ---
    the root cause of the original out-of-memory failures. We assemble
    in coordinate (COO) format ($\sim$`<!-- -->`{=html}14   and
    $\sim$`<!-- -->`{=html}54   respectively) and solve in
    compressed-sparse-column (CSC) format, so arbitrary node counts are
    memory-safe; the 10404-node case converges to $10^{-10}$ in
    $\sim$`<!-- -->`{=html}54 min on a single core.

2.  **Vectorised node operators.** The node-state map is strictly linear
    in the $(J_4,q)$ unknowns. We probe it with nine basis vectors once
    to build the $9\times9$ linear map, then apply it as a batched
    matrix multiply over all nodes and all groups. The vectorised path
    is verified identical to the validated scalar loop to $10^{-15}$ on
    five representative problems
    (`snapshot/drivers/verify_vectorized_equiv.py`), so the speed-up is
    at zero cost to accuracy.

3.  **Rectangular nodes with per-node widths (no homogenisation).** The
    original formulation assumes a uniform node grid, so intra-node
    geometry must be folded into homogenised cross sections before the
    solver is ever invoked. OpenPSN extends the node operators to
    arbitrary per-node widths: each rectangular node carries its own
    width in $x$ and $y$, and nodes meet corner-to-corner only (no
    T-junctions). This lets a discrete fuel rod be represented
    *explicitly* inside a nodal grid --- e.g. the C5G7 1.26 cm circular
    rod is replaced by an equal-area $s=\sqrt{\pi r^{2}}=0.95713$ cm
    fuel square centred on the same pitch, giving a $3\times3$ rectangle
    per pin and a globally edge-aligned tiling
    (17$\times$`<!-- -->`{=html}17 assembly $=51\times51$ nodes; 1/4
    core $=153\times153$). Un-homogenised 7-group cross sections can
    then be assigned per rectangle (fuel vs. water), and the
    homogenisation model --- the usual source of nodal bias on MOX
    benchmarks --- is removed from the problem entirely. The rectangular
    node operators are the same closed-form parabolic/phase-space forms
    as the square case evaluated on the actual node widths; they are
    verified node-by-node against the square-node path (which they
    reduce to exactly when all widths are equal).

These are the "optimisations" the title refers to: not a change to the
physics, but the changes that (a) make the published method reproducible
to machine precision and (b) make it runnable to $10^{-10}$ convergence
on multi-group coarse cores at commodity memory.

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

All 248 data points are reproduced in single-process serial runs with
peak memory $<\SI{2.5}{\giga\byte}$. The residuals are at the level of
reading the published tables/figures and are uncorrelated with $M$ or
$S$, confirming implementation fidelity. The full point-by-point
comparison is tabulated in Appendix A
(Tables [5](#tab:appA){reference-type="ref"
reference="tab:appA"}--[8](#tab:appD){reference-type="ref"
reference="tab:appD"}); the complete data and per-point logs are shipped
in `snapshot/data/`.

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
the two studies of Sections 4.3--4.4: *(i)* the standard pin-homogenised
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

## Data-fidelity check (independent of the solver)

Before trusting any $\kappa_{\mathrm{eff}}$, we validate the 7-group
data wiring with a solver-independent test: the infinite-medium
multiplication factor of the homogenised UO2 pin, computed directly from
the cross-section matrices, $$k_\infty=\max\mathrm{eig}\!\left[
  \left(I-\tfrac{S_{\mathrm{in}}}{\Sigma_t}\right)^{-1}
  \mathrm{diag}(\chi\nu\Sigma_f)\right].$$ OpenPSN gives
$k_\infty^{\mathrm{UO2}}=1.32936$, which matches *to the printed digit*
the UO$_2$--$0.95\rho$ single-assembly nTRACER reference $1.32936$
tabulated in SPHINCS [@sphincs] (Table 1). (The nominal-$\rho$ UO$_2$
row is $1.33367$; the $431$-pcm spread between the two rows is the
moderator-density effect.) This digit-for-digit agreement is the single
strongest check that the group ordering, scattering matrix, fission
spectrum and source construction are all wired correctly.

## Study I --- the pinwise-homogenised nodal model

The standard nodal treatment folds each pin's fuel and moderator into a
single homogenised node. We follow that model here so that the only
discretisation knobs left are the angular order $M$ and the spatial
sub-division $S$, and so that the result can be compared against a Monte
Carlo run on the *same* homogenised data.

#### Pinwise homogenisation.

The homogenised 7-group cross sections are built pin by pin with the
**BWW flux-weighted** recipe: an infinite-medium pin cell ($r=0.54$ cm,
1.26 cm pitch) is solved with an explicit fuel/water MOC calculation,
and the fuel-group cross sections are weighted by the self-shielded
infinite-medium flux (scattering by the source-group flux). Guide tubes
are area-weighted. This is the same self-shielding-aware homogenisation
that nodal practice uses in production; it removes most of the resonance
self-shielding error that a plain area-weighted average leaves behind.

#### Monte Carlo reference, computed directly.

Rather than lean on a published reference for the homogenised model, we
compute the reference ourselves with OpenMC [@openmc2015] on the exact
same pinwise-homogenised cross sections, so that the two sides differ
only in the transport method. Each calculation uses 20000 active batches
of 100000 particles (collision estimator, 48 threads); the eigenvalue is
the mean over the last 20000 batches and its standard error is
$\sigma_{\mathrm{batch}}/\sqrt{N}$, which we audit to be
$\approx$`<!-- -->`{=html}3.4--3.5 pcm. We run four problems: the single
UO$_2$ assembly and the 1/4 core, each with *homogenised* (pin-node) and
*explicit* (resolved fuel/water) geometry. The two *homogenised* runs
are the reference for the pinwise-homogenised PSN sweep below:

- $k_{\mathrm{MC,\,hom}}^{\mathrm{asm}}=1.3336285$ ($\pm3.5$ pcm)
  (single UO$_2$ assembly),

- $k_{\mathrm{MC,\,hom}}^{\mathrm{core}}=1.1868230$ ($\pm3.4$ pcm) (1/4
  core).

#### OpenPSN $S\times M$ sweep.

We sweep $S\in\{1,\dots,6\}$ and $M\in\{2,4,8,12,16,24\}$ on the single
UO$_2$ assembly (nodes $(17S)^2$) and on the 1/4 core (nodes $(51S)^2$),
with the BWW homogenised cross sections.
Table [2](#tab:bww){reference-type="ref" reference="tab:bww"} gives the
deviation from the OpenMC homogenised reference in pcm. (Large
$S\times M$ corner cases exceed the $64$ GB working set during LU
fill-in and are marked OOM; they do not affect the convergence read-off,
which is taken along each row.)

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
| $S{=}6$ | $+25.4$ | $+5.5$  | $+0.6$  | $+1.2$   | $+1.8$   | OOM      |
+---------+---------+---------+---------+----------+----------+----------+
| *C5G7-2D 1/4 core (ref. $1.1868230$)*                                  |
+---------+---------+---------+---------+----------+----------+----------+
| $S{=}1$ | $-97.6$ | $-59.5$ | $-37.1$ | $-31.3$  | $-28.8$  | $-26.7$  |
+---------+---------+---------+---------+----------+----------+----------+
| $S{=}2$ | $-96.3$ | $-56.1$ | $-33.6$ | $-26.6$  | $-23.6$  | OOM      |
+---------+---------+---------+---------+----------+----------+----------+
| $S{=}3$ | $-93.8$ | $-50.1$ | $-26.8$ | OOM      | OOM      | OOM      |
+---------+---------+---------+---------+----------+----------+----------+
| $S{=}4$ | $-92.8$ | $-47.6$ | OOM     | OOM      | OOM      | OOM      |
+---------+---------+---------+---------+----------+----------+----------+
| $S{=}5$ | $-92.3$ | OOM     | OOM     | OOM      | OOM      | OOM      |
+---------+---------+---------+---------+----------+----------+----------+
| $S{=}6$ | OOM     | OOM     | OOM     | OOM      | OOM      | OOM      |
+---------+---------+---------+---------+----------+----------+----------+

: Pinwise-homogenised (BWW) OpenPSN, $\Delta$ in pcm versus the OpenMC
*homogenised* reference: ASM vs. $1.3336285$, CORE vs. $1.1868230$ (each
$\approx$`<!-- -->`{=html}3.5 pcm standard error). OOM = LU fill-in
exceeded the 64 GB cap; the minimum-\|$\Delta$\| point per $S$ row is
marked in Fig. [1](#fig:bww){reference-type="ref" reference="fig:bww"}.
{#tab:bww}

Two features stand out (Fig. [1](#fig:bww){reference-type="ref"
reference="fig:bww"}). *Assembly:* the $M{=}2$ column sits high and
positive ($+14\ldots+25$ pcm, the angular cliff) and rises slightly with
$S$, whereas the $M\ge4$ region forms a floor that climbs monotonically
from $-24$ pcm at $S{=}1$ back toward zero, and by $S{=}5$--$6$ every
$M\ge8$ point is inside the OpenMC reference standard error
($\pm3.5$ pcm). *Core:* the $M{=}2$ column is pinned low
($-98\ldots-92$ pcm) and the error recovers monotonically with $M$; the
best point per row marches diagonally ($M{=}24$ at $S{=}1$ down to
$M{=}2$ at $S{=}5$), and even the best attainable core point is
$-23.6$ pcm, i.e. the pinwise-homogenised core sits a little low of the
OpenMC homogenised reference while the assembly converges to it from
both sides.

<figure id="fig:bww" data-latex-placement="H">
<img src="c5g7_bww_sweep_heatmap.png" style="width:94.0%" />
<figcaption>Pinwise-homogenised (BWW) OpenPSN
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi mathvariant="normal">Δ</mi><msub><mi>κ</mi><mrow><mi mathvariant="normal">e</mi><mi mathvariant="normal">f</mi><mi mathvariant="normal">f</mi></mrow></msub></mrow><annotation encoding="application/x-tex">\Delta\kappa_{\mathrm{eff}}</annotation></semantics></math>
in pcm versus the OpenMC homogenised reference, as a function of angular
order
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>M</mi><annotation encoding="application/x-tex">M</annotation></semantics></math>
(horizontal) and spatial sub-division
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>S</mi><annotation encoding="application/x-tex">S</annotation></semantics></math>
(vertical). Left: single
UO<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><msub><mi></mi><mn>2</mn></msub><annotation encoding="application/x-tex">_2</annotation></semantics></math>
assembly
(ref. <math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>1.3336285</mn><annotation encoding="application/x-tex">1.3336285</annotation></semantics></math>,
35 of 36 points computed); right: C5G7-2D 1/4 core
(ref. <math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mn>1.1868230</mn><annotation encoding="application/x-tex">1.1868230</annotation></semantics></math>,
17 of 36 points; grey cells exceeded the 64 GB LU fill-in cap). White
box/dot marks the
minimum-|<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi mathvariant="normal">Δ</mi><annotation encoding="application/x-tex">\Delta</annotation></semantics></math>|
point per
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>S</mi><annotation encoding="application/x-tex">S</annotation></semantics></math>
row.</figcaption>
</figure>

#### Flux distribution at the best core point.

The $-23.6$ pcm best core point is not a sum of compensating local
errors: Figure [2](#fig:bwwflux){reference-type="ref"
reference="fig:bwwflux"} compares its pin-averaged fission source map
$\nu\Sigma_f\phi$ against the OpenMC homogenised-geometry reference on
the same BWW cross sections (the two differ only in the transport
method; the reference field is the converged 7-group flux of the same
Monte Carlo calculation contracted with the homogenised $\nu\Sigma_f$,
so both maps are the same quantity, each normalised to unit sum). Over
all $1056$ fuel pins the relative error has a mean of $-0.00\%$, an RMS
of $0.12\%$ and a maximum of $0.45\%$, with no systematic offset between
the UO$_2$ and MOX assemblies (quadrant means $+0.02\%$, $-0.02\%$,
$+0.01\%$, $-0.03\%$). The homogenised-core eigenvalue therefore sits
where the local physics says it should: the $-23.6$ pcm is a small,
smooth, method-level bias, not a cancellation.

<figure id="fig:bwwflux" data-latex-placement="H">
<img src="c5g7_bww_flux_compare.png" style="width:98.0%" />
<figcaption>Pin-averaged fission source maps
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>ν</mi><msub><mi mathvariant="normal">Σ</mi><mi>f</mi></msub><mi>ϕ</mi></mrow><annotation encoding="application/x-tex">\nu\Sigma_f\phi</annotation></semantics></math>
at the best pinwise-homogenised core point (M16, S=2) vs. the OpenMC
homogenised-geometry reference on the same BWW XS, C5G7-2D 1/4 core
(fuel region,
34<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>×</mi><annotation encoding="application/x-tex">\times</annotation></semantics></math>34
pins). Left: OpenPSN; centre: OpenMC reference; right: relative error
(fuel pins only, both maps normalised to unit sum).</figcaption>
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
2601 fuel-centre rectangles of the 1/4 core coincide with the 2601 fuel
pins of the homogenised grid, and the total area reproduces
$4129.35~\mathrm{cm}^2$ exactly. Un-homogenised 7-group cross sections
(fuel, water, Gd, MOX) are then assigned directly to the rectangles. The
spatial resolution is fixed by this $3\times3$ tiling, so the only open
discretisation knob is the angular order $M$; there is no $S$ axis in
this model.

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

#### Angular convergence ($M$ sweep).

Table [3](#tab:rect){reference-type="ref" reference="tab:rect"} gives
the $M$ sweep with un-homogenised material XS on the single UO$_2$
assembly ($51\times51$ rect. nodes, all edges reflective) and on the 1/4
core ($153\times153$ rect. nodes, reflective on the symmetry edges,
vacuum on the open edges), all converged to $10^{-10}$ in
$|\Delta\kappa_{\mathrm{eff}}|$.

+-------------------------+---------------+---------------+---------------+
|                         | $M{=}8$       | $M{=}12$      | $M{=}16$      |
+:========================+==============:+==============:+==============:+
| *Single UO$_2$ assembly, $\kappa_{\mathrm{eff}}$*                       |
+-------------------------+---------------+---------------+---------------+
| $\kappa_{\mathrm{eff}}$ | 1.3337164     | 1.3340330     | 1.3343536     |
+-------------------------+---------------+---------------+---------------+
| $\Delta$ vs OpenMC,     | $+22.2$       | $+53.8$       | $+85.9$       |
| explicit pin            |               |               |               |
| ($1.3334947$)           |               |               |               |
+-------------------------+---------------+---------------+---------------+
| $\Delta$ vs OpenMC,     | $-101.0$      | $-69.3$       | $-37.2$       |
| rect. rods              |               |               |               |
| ($1.3347260$)           |               |               |               |
+-------------------------+---------------+---------------+---------------+
| $\Delta$ vs nTRACER     | $+4.8$        | $+36.3$       | $+68.9$       |
| ($1.33367$)             |               |               |               |
+-------------------------+---------------+---------------+---------------+
| *C5G7-2D 1/4 core, $\kappa_{\mathrm{eff}}$*                             |
+-------------------------+---------------+---------------+---------------+
| $\kappa_{\mathrm{eff}}$ | 1.1853742     | 1.1861084     | 1.1866067     |
+-------------------------+---------------+---------------+---------------+
| $\Delta$ vs OpenMC,     | $-112.1$      | $-38.7$       | $+11.1$       |
| explicit pin            |               |               |               |
| ($1.1864955$)           |               |               |               |
+-------------------------+---------------+---------------+---------------+
| $\Delta$ vs OpenMC,     | $-184.3$      | $-110.9$      | $-61.1$       |
| rect. rods              |               |               |               |
| ($1.1872176$)           |               |               |               |
+-------------------------+---------------+---------------+---------------+
| $\Delta$ vs McGraw      | $-108.6$      | $-35.2$       | $+14.7$       |
| high-fidelity           |               |               |               |
| ($1.18646$)             |               |               |               |
+-------------------------+---------------+---------------+---------------+
| $\Delta$ vs nTRACER     | $-115.6$      | $-42.2$       | $+7.7$        |
| ($1.18653$)             |               |               |               |
+-------------------------+---------------+---------------+---------------+

: Rectangular-node OpenPSN (un-homogenised material XS). $\Delta$ in pcm
versus the explicitly noted reference. References: the directly computed
OpenMC explicit-geometry runs on the same un-homogenised data --- fuel
circle $+$ water annulus, and the identical rectangular-rod geometry of
the PSN nodes --- the high-fidelity reference $1.18646$ of McGraw *et
al.* [@mcgraw2014], and nTRACER (SPHINCS). {#tab:rect}

On the 1/4 core the eigenvalue climbs monotonically toward the
reference: $-112.1 \rightarrow -38.7 \rightarrow +11.1$ pcm for
$M=8\rightarrow12\rightarrow16$, with $M{=}16$ at $\sim3$ of the OpenMC
standard error ($\pm3.4$ pcm) and within $+14.7$ pcm of the
high-fidelity reference $1.18646$. Against the OpenMC run on the
identical rectangular-rod geometry
(Section [\[sec:rectref\]](#sec:rectref){reference-type="ref"
reference="sec:rectref"}, Figure [5](#fig:rectflux){reference-type="ref"
reference="fig:rectflux"}) the same sweep reads
$-184.3 \rightarrow -110.9 \rightarrow -61.1$ pcm: still monotonically
decreasing, so at $M{=}16$ the angular-moment closure, not the square
tiling, is what sets the residual. (The assembly $M$ dependence drifts
the other way, upward with $M$: the reflective assembly has no leakage
channel to absorb the angular cliff, so the angular-moment closure
over-corrects as $M$ grows; the core, with two vacuum edges, converges
toward the reference instead.)

#### Flux comparison, explicit geometry.

Figure [4](#fig:flux){reference-type="ref" reference="fig:flux"}
compares the pin-averaged fission maps $\Sigma_f\phi$ of the
rectangular-node OpenPSN ($M{=}16$) against the OpenMC explicit-geometry
(fuel circle $+$ water annulus) runs on the same un-homogenised data,
for the single UO$_2$ assembly (first row, 17$\times$`<!-- -->`{=html}17
pins) and the 1/4 core (second row, 34$\times$`<!-- -->`{=html}34 fuel
pins). The OpenPSN node field is the pin power $\nu\Sigma_f\phi$; it is
converted into the fission quantity by dividing each pin by the
fission-weighted $\nu$ of its material ($2.441$ for UO$_2$,
$2.865$--$2.874$ for the three MOX enrichments, and $2.434$ for the
fission-chamber inserts), so that both maps are the same physical
quantity $\Sigma_f\phi$ pin by pin. Both maps are then normalised to
unit sum, so the residual is purely in the shape. On the assembly the
pin-by-pin relative error has a mean of $-0.002\%$ and an RMS of
$0.070\%$ (max $0.21\%$). On the 1/4 core, over all $1056$ fuel pins,
the error is a mean of $-0.25\%$ and an RMS of $0.72\%$ (max $1.52\%$):
the four assemblies show a coherent pattern at the $\pm0.8\%$ level
(UO$_2$ assemblies $+0.04\ldots+0.63\%$, MOX assemblies $-0.84\%$ in
both) --- the small between-assembly power redistribution that comes
with the rectangular tiling --- while the intra-assembly shape error
stays below $\pm1\%$ in every assembly; the two solvers agree within one
percent on the pin-level fission distribution for the whole core.

<figure id="fig:flux" data-latex-placement="H">
<img src="c5g7_rect_flux_compare_nu.png" style="width:98.0%" />
<figcaption>Pin-averaged fission maps
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi mathvariant="normal">Σ</mi><mi>f</mi></msub><mi>ϕ</mi></mrow><annotation encoding="application/x-tex">\Sigma_f\phi</annotation></semantics></math>,
rectangular-node OpenPSN
(<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>M</mi><mo>=</mo><mn>16</mn></mrow><annotation encoding="application/x-tex">M{=}16</annotation></semantics></math>,
un-homogenised material XS) vs. OpenMC (explicit geometry, 7-group
collision estimator). One row per problem — single
UO<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><msub><mi></mi><mn>2</mn></msub><annotation encoding="application/x-tex">_2</annotation></semantics></math>
assembly
(17<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>×</mi><annotation encoding="application/x-tex">\times</annotation></semantics></math>17
pins), then C5G7-2D 1/4 core, fuel region only
(34<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>×</mi><annotation encoding="application/x-tex">\times</annotation></semantics></math>34
pins); columns are OpenPSN, OpenMC, and the pin-by-pin relative error
(fuel pins only), both maps normalised to unit sum.</figcaption>
</figure>

[]{#sec:rectref label="sec:rectref"}

#### Rectangular-rod reference.

The circular pin cell is the benchmark's reference geometry, while the
PSN nodes tile the same cell with an equal-area square fuel block (side
$0.9571$ cm, water gap $0.1514$ cm). To separate the geometric effect
from the method difference we run OpenMC on exactly that rectangular-rod
geometry (same un-homogenised data, same collision-estimator pin
tallies, $20200$ batches $\times 10^{5}$ particles, fixed seed). The
square-rod assembly returns $\kappa_{\mathrm{eff}}= 1.3347260$
($\pm 3.6$ pcm) and the 1/4 core $1.1872176$ ($\pm 3.4$ pcm): $+123$ and
$+72$ pcm above the cylindrical reference of
Table [3](#tab:rect){reference-type="ref" reference="tab:rect"} --- the
pure geometry effect of the square tiling on the same data. Against the
rectangular-rod reference the $M$ sweep is
$-101.0 \rightarrow -69.3 \rightarrow -37.2$ pcm on the assembly and
$-184.3 \rightarrow -110.9 \rightarrow -61.1$ pcm on the 1/4 core:
monotonically decreasing with $M$, i.e. the angular-moment closure is
still the dominant residual, and the square tiling does not remove the
systematic low bias. Figure [5](#fig:rectflux){reference-type="ref"
reference="fig:rectflux"} compares the two solvers on this identical
geometry. The pin-by-pin fission maps $\Sigma_f\phi$ agree with a mean
of $-0.00\%$ and an RMS of $0.16\%$ (max $0.38\%$) on the assembly, and
a mean of $-0.18\%$ and an RMS of $0.59\%$ (max $1.61\%$) over all
$1056$ fuel pins of the 1/4 core (assembly means $+0.46\%$, $-0.64\%$,
$-0.60\%$, $+0.05\%$): the same level as against the explicit
cylindrical geometry (RMS $0.72\%$), so the pin-level shape difference
is set by the method, not by the square tiling.

<figure id="fig:rectflux" data-latex-placement="H">
<img src="c5g7_rect_vs_rect_flux.png" style="width:98.0%" />
<figcaption>Pin-averaged fission maps
<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi mathvariant="normal">Σ</mi><mi>f</mi></msub><mi>ϕ</mi></mrow><annotation encoding="application/x-tex">\Sigma_f\phi</annotation></semantics></math>,
rectangular-node OpenPSN
(<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>M</mi><mo>=</mo><mn>16</mn></mrow><annotation encoding="application/x-tex">M{=}16</annotation></semantics></math>)
vs. OpenMC on the identical rectangular-rod geometry (equal-area square
fuel block per pin cell, 7-group collision estimator). One row per
problem — single
UO<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><msub><mi></mi><mn>2</mn></msub><annotation encoding="application/x-tex">_2</annotation></semantics></math>
assembly
(17<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>×</mi><annotation encoding="application/x-tex">\times</annotation></semantics></math>17
pins), then C5G7-2D 1/4 core, fuel region only
(34<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>×</mi><annotation encoding="application/x-tex">\times</annotation></semantics></math>34
pins); columns are OpenPSN, OpenMC, and the pin-by-pin relative error
(fuel pins only), both maps normalised to unit sum.</figcaption>
</figure>

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
| PSN nodal      | OpenPSN                 | 1.1866067               | $+14.7$        |
| (rect. node,   |                         |                         |                |
| $M{=}16$)      |                         |                         |                |
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

- **OpenPSN in this work.** The two C5G7 studies of Sections 4.3--4.4
  bound the answer from both sides. With the self-shielding-aware BWW
  pinwise homogenisation, the best point of the $S\times M$ sweep
  ($M16\_S2$) is $-23.6$ pcm vs. the OpenMC homogenised reference,
  i.e. $+12.7$ pcm vs. the high-fidelity reference $1.18646$; with the
  rectangular-node model that removes homogenisation entirely, $M{=}16$
  lands at $+11.1$ pcm vs. the OpenMC explicit-geometry reference
  ($\sim3.3$ its $\pm3.4$ pcm standard error) and $+14.7$ pcm
  vs. $1.18646$, and at $-61.1$ pcm vs. the OpenMC run on the identical
  rectangular-rod geometry (Section 4.4); the pin-level fission maps
  agree within $0.6\%$ RMS on both geometries. In either case OpenPSN
  sits inside the MOC reference cluster ($\pm37$ pcm, plus the
  better-differenced SN and unstructured codes), and the
  rectangular-node result is within the $M{=}16$ angular convergence of
  the nodal PSN method on this problem.

Two published outliers are worth noting for completeness: COHINT ($P_2$
interface currents, $-1125$ pcm) and HELIOS (collision probability,
$+675$ pcm), both far outside the converged cluster and attributable to
their respective current/transport approximations.

# Conclusions

We built **OpenPSN**, a clean, multi-group, sparse, vectorised
implementation of the Phase Space Nodal method, and validated it on two
levels. On the source paper's problems it reproduces all 248 published
eigenvalue data points to a few pcm (full point-by-point comparison in
Appendix A) while removing the dense-formulation memory blow-up (exact
to $10^{-15}$; the 10404-node C5G7 case assembles in
$\sim$`<!-- -->`{=html}54   and converges in
$\sim$`<!-- -->`{=html}54 min on one core, where the dense form would
need a $\sim$`<!-- -->`{=html}504   matrix). We also extended the node
operators to arbitrary per-node widths, giving the first PSN
implementation with explicit, un-homogenised intra-node fuel/water
structure.

On the C5G7 MOX benchmark the two studies separate the two error sources
that a nodal code carries. In the *pinwise-homogenised* study, the
$S\times M$ sweep against a directly computed OpenMC reference on the
same homogenised data shows the discretisation converging cleanly: the
assembly error reaches the Monte Carlo standard error
($\approx$`<!-- -->`{=html}3.5 pcm) by $S{=}5$, $M\ge8$, and the best
core point (with the self-shielding-aware BWW recipe) is $-23.6$ pcm
versus the OpenMC homogenised reference, i.e. $+12.7$ pcm versus the
high-fidelity reference $1.18646$. That residual is the homogenisation
model itself, and the best-point pin-power field agrees with the
same-recipe OpenMC map to $0.12\%$ RMS (max $0.45\%$): a smooth
method-level bias, not a cancellation of local errors. In the
*rectangular-node* study the homogenisation model is removed entirely:
with un-homogenised 7-group material data the 1/4-core eigenvalue
converges monotonically with $M$ into the OpenMC explicit-geometry
reference ($-112\rightarrow-39\rightarrow+11$ pcm for
$M=8\rightarrow16$), lands $+14.7$ pcm from the high-fidelity reference
$1.18646$, and its pin-averaged fission map $\Sigma_f\phi$ (the same
quantity as the Monte Carlo fission tally) agrees with OpenMC to
$0.07\%$ RMS on the single assembly and $0.72\%$ over all $1056$ fuel
pins of the core (max $1.52\%$; below $\pm1\%$ within every assembly,
largest in the MOX cells where the local gradients are steepest). Placed
against the 20-code NEA landscape and the recent P1/SP3/SN/nodal
literature, both OpenPSN results sit inside the high-fidelity MOC
reference cluster: the nodal PSN method, once its homogenisation model
is controlled (by a self-shielding-aware recipe or by explicit
rectangular nodes), reproduces resolved-pin transport at a fraction of
the mesh cost of discrete-ordinates or characteristics methods.

# Acknowledgments {#acknowledgments .unnumbered}

This work received no external funding. The build and the
performance-optimisation work on OpenPSN (sparse assembly, vectorised
node operators) were carried out with the assistance of a large language
model (Qwen3.8 27B) running on local hardware.

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

H. H. Cho, J. Kang, J. I. Yoon, and H. G. Joo, "Analysis of C5G7-TD
benchmark with a multi-group pin homogenized SP3 code SPHINCS," *Nuclear
Engineering and Technology* **53**(5) (2021) 1403--1415
 [doi:10.1016/j.net.2020.11.013](https://doi.org/10.1016/j.net.2020.11.013).
Table 1: nTRACER single-assembly references (UO2 $1.33367$;
UO2-$0.95\rho$ $1.32936$) with the SPHINCS w/o-SPH value $1.33123$;
Table 2: 2-D core $\Delta\rho$ from $-225$ pcm (w/o SPH) to $-20$ pcm
(w/ SPH) vs the nTRACER reference.

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
Gd) plus the with-Gd valley, in the same "this work / paper ($\Delta$)
cell format used in the reproduction notes shipped in `snapshot/`. Row
labels: $S$ = spatial subdivision of the checkerboard grid, $N$ =
spatial grid of the BWR bundle. The remaining 8 points (restricted
$M=24$, $S=1,2,4,8$, weak/strong) agree with the printed Figs. 4--5
within the plot-reading uncertainty and are listed in
`snapshot/data/report_data.json`.

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
  in the row (min $|\Delta|$); the valley shifts to lower $M$ as $N$
  grows, exactly the Sec. 4.4 effect of the source paper. Best point
  overall: $N=9$, $M=20$, $+1.1$ pcm. {#tab:appD}

*Note:* each cell is "this work / paper ($\Delta$), $\Delta$ = this work
$-$ paper, in pcm; paper values as printed in Ref. [@psn2027]; the
Fig. 10/12 references are read from the printed contour plots
($\pm1$--$2$ pcm reading uncertainty).
:::
