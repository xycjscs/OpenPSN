::: center
:::

**Keywords:** phase-space nodal method; diffusion approximation; convergence analysis; Richardson extrapolation; verification and validation

# Introduction {#sec:intro}

Deterministic nodal methods earn their place in reactor analysis by being *predictive*: given two computations at different discretization levels, one must be able to forecast the third, bound the residual error, and decide which refinement axis to fund --- all without recourse to an external solution. This paper is about a failure mode of that property and its removal in the phase-space nodal (PSN) method [@chao2027].

PSN carries angular-moment (diffusion-moment) unknowns in phase space: each spatial node and energy band is subdivided into azimuthal sectors, and a nodal response map connects face partial currents to face partial fluxes [@chao2027]. In the reproduction of the strong/weak-absorber checkerboard benchmark of [@chao2027] (Fig. 3 therein), the error over the spatial subdivision $S$ and azimuthal order $M$ organizes along a diagonal band $S \approx M/2$: the optimum at each $M$ sits on the band, the band drifts as $M$ grows, and points off the band degrade in *both* refinement directions. We call this the *valley locking* pathology. Its practical content is not the band's existence but three consequences: (i) fixed-$S$ error sequences are non-monotonic and change sign (measured: $-37.4 \to +24.0 \to +42.1 \to +47.8$ pcm along $M=24\ldots192$ at $S{=}12$); (ii) no extrapolation of a row predicts a physically meaningful limit; (iii) convergence is certifiable only against an external reference.

The root cause is structural. The legacy PSN node solves an *isotropic* body diffusion equation whose diffusion coefficient is the full-angle-averaged one, while its face currents \[eq. (2.7c) of [@chao2027]\] carry the *sector* tensor $\bm{D}_{im}$ with the exact $\operatorname{sinc}(\pi/M)$ anisotropy. Refining $S$ converges to the solution of the body operator, refining $M$ to the face operator's; the two limits differ, the two errors oppose, and the valley is their zero contour. The paper makes this precise, then repairs the inconsistency.

The repair must not break what PSN gets right. We therefore require (i) the same node map size and closure (no new interface unknowns, no new stencil), (ii) an exact anchor: at $M=2$ the strict node equals the legacy node to machine precision, and (iii) no tuning: every coefficient is derived, none fitted. The resulting *strict phase-space diffusion closure* solves the body with the same $\bm{D}_m$ as the faces. Its mathematical core is a dispersion lemma: for every sector, every anisotropy level, and every band weight, the plane waves along the two *sector-diagonal* directions $\theta_w = \phi_m \pm \pi/4$ are exact homogeneous solutions with a single decay constant $k=\sqrt{\Sigma_t/p}$. The node basis is the four hyperbolic modes of those two directions plus the constant --- same status and conditioning as the legacy additive family --- and the paraboloidal source particular solution becomes pointwise exact rather than cell-mean exact.

Contributions:

1.  a diagnosis of the body/face operator inconsistency in PSN and the proof that the checkerboard diagonal valley is its zero-error contour (Sec. [2](#sec:legacy){reference-type="ref" reference="sec:legacy"});

2.  the strict closure: derivation, exact basis lemma, node map, $M=2$ anchor (Sec. [3](#sec:strict){reference-type="ref" reference="sec:strict"});

3.  the predictability experiment: 720 checkerboard points $S\le20$, $M\le192$, showing per-row monotone $O(M^{-2})$ convergence, Richardson prediction of row limits to sub-pcm against $M=384$ measurement, and a characterized (monotone, sign-crossing) spatial-residual sequence along $S$ (Sec. [4](#sec:exp){reference-type="ref" reference="sec:exp"});

4.  release as `angular.closure: strict_phase_space` in OpenPSN with regression gates (Sec. [5](#sec:impl){reference-type="ref" reference="sec:impl"}).

# Legacy PSN and the valley locking pathology {#sec:legacy}

## Node structure

One group, spatial node of size $h_x \times h_y$, azimuthal sectors $\phi \in [\,m\,\Delta\phi,(m{+}1)\,\Delta\phi)$, $\Delta\phi=\pi/M$, centered at $\phi_m$. The base angular function is P1-type restricted to the sector; the odd (streaming) parity is eliminated in favor of net face currents $J_f$, $f=0..3$, leaving five source moments $q=(q_0,q_{1x},q_{1y},q_{2x},q_{2y})$. The legacy node solves in the body $$\begin{equation}
-\nabla\!\cdot\!\big(p\,\nabla\phi\big) + \Sigma_t\,\phi = q(x,y),
\qquad p = \frac{\mu_i^2}{2\Sigma_t}\ \text{(TY)},\quad
p=\frac{\beta_i}{3\Sigma_t}\ \text{(generic)},
\label{eq:legacybody}
\end{equation}$$ isotropic, with additive trial family $\{1,\ \mathrm{ch}(kx),\ \mathrm{sh}(kx),\ \mathrm{ch}(ky),\
\mathrm{sh}(ky)\}$ and $k^2=\Sigma_t/p$; the face currents are then formed with the sector mixing \[2.7c\] $$\begin{equation}
J_{x\pm} = \left(1 \pm s\cos 2\phi_m\right) J_X \pm s \sin 2\phi_m
\; J_{\mathrm{avg}}, \qquad s=\operatorname{sinc}(\Delta\phi),
\label{eq:face}
\end{equation}$$ which *is* the divergence-theorem current of a tensor $$\begin{equation}
\bm{D}_m = p\begin{pmatrix} 1+s\cos 2\phi_m & s\sin 2\phi_m \\
s\sin 2\phi_m & 1-s\cos 2\phi_m \end{pmatrix}.
\label{eq:Dm}
\end{equation}$$ Identity: the unweighted sector average of the second angular moment equals the half-range-weighted face tensor of \[2.7c\] --- the same $\bm{D}_m$ in both guises. The inconsistency is that [\[eq:legacybody\]](#eq:legacybody){reference-type="eqref" reference="eq:legacybody"} replaces $\bm{D}_m$ by $p\,\bm{I}$ in the body while [\[eq:face\]](#eq:face){reference-type="eqref" reference="eq:face"} keeps $\bm{D}_m$ at the faces.

## Consequence: two operators, one valley

Let $d(S,M)$ be the $\Delta k_{\mathrm{eff}}$ against a reference. Because the discrete problem converges (as $S\to\infty$ at fixed $M$) to a fixed-$M$ *problem* whose operator mixes body-isotropic and face-tensor physics, and (as $M\to\infty$ at fixed $S$) to the consistent-tensor limit of the *face* operator on the coarse mesh, the error separates to leading order, $$\begin{equation}
d(S,M) \;=\; a(S) + b(M) + \text{h.o.t.},
\qquad a(S)\asymp c_S S^{-2},\quad
b(M)\asymp c_M M^{-2},
\label{eq:split}
\end{equation}$$ (no bound on the h.o.t. is available; [\[eq:split\]](#eq:split){reference-type="eqref" reference="eq:split"} is an empirical asymptotic separability hypothesis, and the rectangle diagnostic below is its test, not a corollary of a theorem) with $c_S c_M < 0$ for the checkerboard at moderate $S$ (measured: $a>0$ up to $S\approx10$, $b<0$, for the strong absorber; the $S^{-2}$ rate is a leading-order statement, the measured $a(S)$ crossing zero near $S\approx11$ is discussed in Sec. [4.2](#sec:pred){reference-type="ref" reference="sec:pred"}). With the error surface continuous in the interpolated arguments (discrete here, so read this as the interpolation of the measured map) a zero contour $a(S)+b(M)=0$ then exists in *every* consistent two-parameter method --- it is a consequence of the intermediate value theorem applied to the interpolated split, not a defect. The defect is when the contour *drifts with $M$* (here $S^{*}\approx M/2$, because $b(M)$ is a wrong-operator bias that decays in $M$ while $a$ does not) and the per-row limits are nonzero: the method's own asymptotics point *away* from the contour as either axis is refined past the crossing. Measured legacy optima (strong absorber): $S^{*}=5,9,15,20,\text{ out of range}$ at $M=12,24,48,96,192$ --- the band runs out of every finite grid. The split [\[eq:split\]](#eq:split){reference-type="eqref" reference="eq:split"} is directly testable without any reference: the mixed second difference of $d$ over any rectangle of the $(S,M)$ grid vanishes iff the two axis-errors are separable. Over all $6840$ rectangles ($\binom{20}{2}\binom{9}{2}$) of the $20{\times}9$ sweep, the strict map is separable to a median residual of $1.5$ pcm (strong; grid $|d|$ median $196$ pcm) and $0.3$ pcm (weak), versus $16.9$ and $3.3$ pcm for legacy --- an $11\times$ violation of separability by the inconsistent operator on both absorbers. The full distribution matters more than the median: rectangles are ranked, and the residual is dominated by coarse corners of the sweep (rows through $S{=}1$ or columns through $M{=}2$, where the leading-order split [\[eq:split\]](#eq:split){reference-type="eqref" reference="eq:split"} carries large higher-order terms by construction). Excluding those two rows and columns: strict strong median $0.9$, p95 $23.9$, max $45.5$ pcm against legacy $9.6$/$117.4$/$309.5$; strict weak $0.2$/$4.4$/$9.9$ against legacy $1.8$/$25.8$/$70.8$. The qualification budget below uses the exclusion-window maxima, not the medians. Non-separable cross coupling is the algebraic fingerprint of the valley: in a separable map, refining either axis can only shrink its own term, and the per-axis residuals $c_S S^{-2}$, $c_M M^{-2}$ can be estimated separately and added --- which is exactly the error budget a qualifier needs (Sec. [4.2](#sec:pred){reference-type="ref" reference="sec:pred"}).

# Strict phase-space diffusion closure {#sec:strict}

## Strict body equation

Replace [\[eq:legacybody\]](#eq:legacybody){reference-type="eqref" reference="eq:legacybody"} by the sector tensor body: $$\begin{equation}
-\nabla\!\cdot\!\left(\bm{D}_m \nabla\phi\right)
+ \Sigma_t\,\phi = q(x,y),
\qquad \bm{D}_m.
\label{eq:strictbody}
\end{equation}$$ where $\bm{D}_m$ is the face tensor of Eq. [\[eq:Dm\]](#eq:Dm){reference-type="ref" reference="eq:Dm"}. The face currents of [\[eq:face\]](#eq:face){reference-type="eqref" reference="eq:face"} are then exactly the divergence-theorem currents $J_f = \langle -\bm{n}\!\cdot\!\bm{D}_m
\nabla\phi\rangle_f$ of the body operator: body and face are the *same* operator (verified to $3\times10^{-15}$ against the printed mixing formula for arbitrary sectors, including non-power-of-two $M$). Equation [\[eq:strictbody\]](#eq:strictbody){reference-type="eqref" reference="eq:strictbody"} is not a heuristic blend: it is the cell-tensor projection of the same transport operator that produced [\[eq:face\]](#eq:face){reference-type="eqref" reference="eq:face"}, obtained by averaging the second angular moment over the sector before (not after) eliminating the odd parity --- the projection order that the face formula implicitly assumes.

## Exact basis lemma

::: lemma
**Lemma 1** (diagonal dispersion). *For any $p>0$, $|s|\le 1$, sector center $\phi_m$, the plane wave $e^{k\,w\cdot r}$ solves the homogeneous [\[eq:strictbody\]](#eq:strictbody){reference-type="eqref" reference="eq:strictbody"} iff $w^{\mathsf T}\bm{D}_m w = p$, i.e. iff $\theta_w = \phi_m \pm \pi/4$, with the single decay $k=\sqrt{\Sigma_t/p}$.*
:::

::: proof
*Proof.* $w^{\mathsf T}\bm{D}_m w = p\,[1 + s\cos 2(\theta_w-\phi_m)] = p$ iff $\cos 2(\theta_w - \phi_m) = 0$. The decay then follows from $\Sigma_t - k^2 p = 0$. Pointwise, no approximation. ◻
:::

The node basis is thus $\{1,\ \mathrm{ch}(k\,u),\ \mathrm{sh}(k\,u),\ \mathrm{ch}(k\,v),\
\mathrm{sh}(k\,v)\}$, $u = r\!\cdot\!w(\phi_m{+}\pi/4)$, $v = r\!\cdot\!w(\phi_m{-}\pi/4)$: four exact homogeneous modes, one constant, five source moments --- identical count to legacy, and unlike the legacy additive family it responds to cross-derivative (streaming-diagonal) structure, $\partial_{xy}\phi \ne 0$ in the body. The source particular solution $\phi_p = [q_0 + q_{1x}P_1 + q_{1y}P_1 + q_{2x}P_2 + q_{2y}P_2]/
\Sigma_t + C_p$, $C_p = 3(D_{xx}q_{2x}/a^2 + D_{yy}q_{2y}/b^2)/
\Sigma_t^2$, satisfies the tensor equation pointwise; here $P_1,P_2$ are the coordinate-linear and coordinate-quadratic polynomials in the node frame whose $\Sigma_t$-scaled moments are $q_{1x},q_{1y}$ and $q_{2x},q_{2y}$, so pointwise exactness is over the degree-$\le$`<!-- -->`{=html}2 polynomial source basis at constant $D_m$.

## Node map and the $M=2$ anchor

The closure is unchanged: four face net currents $J_4$ determine four mode coefficients through a $4\times4$ matrix of the same conditioning profile as the legacy one (measured cond $\le 5.7\times10^{2}$ up to $M{=}24$); the constant comes from the exact volume balance; face partial fluxes keep the legacy geometry-free chord factor $\alpha_f$. No interface unknowns are added anywhere.

At $M=2$ (restricted model, $\phi_m=\pi/4$, $s=2/\pi$) the diagonal directions $\phi_m \pm \pi/4$ are *exactly* the $x$ and $y$ axes: the strict basis spans the same space as the legacy basis at the same $k$, and the node map coincides with the production legacy rect node to $2.8\times10^{-15}$ relative on random $(J_4,q)$ (kernel), and the global checkerboard $k_{\mathrm{eff}}$ to $8.9\times10^{-11}$ pcm (solver noise). The strict method is therefore a conservative generalization: within the restricted polar model, the legacy method is not discarded, it is the $M{=}2$ slice of the strict family where the two operators happen to agree (this slice statement is restricted-model only --- the next paragraph shows legacy-generic is no slice of the strict family). (For the generic polar model the legacy body decay is $k=\sqrt{3}\,\Sigma_t$, i.e. it projects the *full-angle* $P_1$ second moment before the sector restriction, so its body operator differs from the face tensor at *every* $M$, including $M=2$: legacy-generic carries both error channels everywhere, and the strict family does not contain legacy-generic as any slice. The checkerboard experiments of this paper are therefore the fair test of the *repair* --- both methods on the same angular model --- and the generic results above should be read with legacy's extra channel in mind: its $M{\to}\infty$ plateau folds together both projection mismatches (the causal split between them is not separated in this paper; a TY-body legacy sweep would isolate it and is left as a regression exercise). All legacy numbers in this paper are legacy-generic.)

## In what sense "strict": the removed vs. the retained approximation {#sec:sense}

PSN rests on five modeling steps: (A1) sector-restricted P1-type base angular function; (A2) elimination of the odd parity in favor of net face currents; (A3) low-order source-moment expansion ($q_0,q_{1x},q_{1y},q_{2x},q_{2y}$); (A4) geometry-free face chord factor $\alpha_f$; (A5) *body equation projected with a different tensor than the face currents*. The strict closure removes A5 and only A5: steps A1--A4 are retained verbatim, which is why the node size, closure equations, conditioning profile and $M{=}2$ node are unchanged. In this precise sense the strict node is *internally operator-consistent relative to the retained approximations A1--A4* --- not approximation-free in any absolute sense, since A1--A4 remain: its homogeneous modes solve the body equation pointwise (Lemma 1), its source particular solution is pointwise exact, and its face currents are exactly the divergence-theorem currents of its own body operator --- nothing inside the node is fitted, blended, or projected twice. What remains wrong at finite $M$ is then *one* thing --- the angular truncation A1--A4 --- and it is $M$-dependent with an order that must be measured or validated row by row (Sec. [4.6](#sec:c5g7){reference-type="ref" reference="sec:c5g7"} finds $O(M^{-1})$, not $O(M^{-2})$, on material-interface grids). Legacy PSN has two simultaneous, opposite-sign error channels (A1--A4 truncation $\propto M^{-2}$ and the A5 operator mismatch $b_\mathrm{face}{-}b_\mathrm{body}$ with its own $M$-decay); their sum is what the valley hides. A single-channel error is what makes both axes individually estimable (Sec. [4.2](#sec:pred){reference-type="ref" reference="sec:pred"}).

# Experiments: predictability, not point accuracy {#sec:exp}

Protocol: strong ($\Sigma_a=0.70$, $k_{\mathrm{ref}}=0.51673$) and weak ($0.07$, $1.12974$) absorber checkerboards [@chao2027], generic polar $I=30$, solver tolerance $10^{-10}$, grid $S=1\ldots20 \times M=2,4,8,12,16,24,48,96,192$ (9 columns $\times$ 20 rows $=180$ points per method-material combination pair, 720 sweep points total), OpenMC references as in [@chao2027]. All numbers below are computed, none fitted. The published references carry 5--6 significant decimals (quantization floor $\pm0.5$ pcm at $k\approx0.52$; the Monte Carlo statistical uncertainty of the published values, not restated in [@chao2027], is of the same size), so sub-pcm margins are only meaningful between deterministic runs --- which is how they are used below: every prediction/hit pair compares two $k_{\mathrm{eff}}$ of this code, and $k_{\mathrm{ref}}$ cancels in the difference.

## Valley locking removed (Fig. [1](#fig:bestS){reference-type="ref" reference="fig:bestS"}) {#valley-locking-removed-fig.-figbests}

The strict optimum at fixed $M$ saturates: $S^{*}=3,5,8,10,11$ at $M=12,24,48,96,192$ (strong absorber) --- it locks onto the crossing of the spatial residual $a(S)=0$ near $S\approx11$ and stops moving. The legacy optimum tracks $S\approx M/2$ past the edge of every grid tried ($S^{*}=20$ at $M{=}96$, out of range at $M{=}192$). Both rows of Table [1](#tab:rows){reference-type="ref" reference="tab:rows"} show the same $S{=}12$ row: monotone single-signed $O(M^{-2})$ (successive ratios $3.30,2.83,2.07\to4$) versus a sign-reversal sequence.

  $M$             24        48        96       192                     384
  -------- --------- --------- --------- --------- -----------------------
  legacy     $-37.4$   $+24.0$   $+42.1$   $+47.8$   \[$+52$, Richardson\]
  strict     $-83.2$   $-25.2$    $-8.9$    $-4.3$                  $-3.0$

  : $\Delta k_{\mathrm{eff}}$ (pcm), $S{=}12$ row, strong absorber. {#tab:rows}

## Reference-free prediction {#sec:pred}

For every row with $M=24,48,96,192$ we form the Richardson limit $\hat d_\infty = (4 d_{2M} - d_M)/3$ and its *self-reported error bar* $\varepsilon = |\hat d_\infty(M_1,M_2) -
\hat d_\infty(M_2,M_3)|$: the disagreement of successive extrapolations, computable without any measurement beyond $M_3$.

- Strong, $S{=}12$: predictions from $(48,96)$ and $(96,192)$: $-3.5$ / $-2.8$ pcm, bar $0.7$ pcm; *later-measured* $M{=}384$ point: $-3.0$ pcm. Hit inside the bar.

- Weak, $S{=}12$: predictions $-1.7$ / $-1.5$, bar $0.2$ pcm; *measured* $M{=}384$: $-1.6$ pcm. Weak, $S{=}20$: predictions $-4.4$ / $-4.2$, bar $0.2$ pcm; *measured* $M{=}384$: $-4.3$ pcm.

- Strong, $S{=}20$ (the highest row, one grid doubling beyond the last sweep point): predictions $-15.9$ / $-15.3$ pcm, bar $0.6$ pcm; *measured* $M{=}384$: $-15.6$ pcm --- inside the bar, $0.3$ pcm from its centre. The prediction was issued from $(M{=}48,96,192)$ data only; the $M{=}384$ run was committed to the queue unseen.

- **Frozen blind set, 8 points.** Predictions (frozen 13:14 UTC, from $(48,96,192)$ only) versus later-measured $M{=}384$: strong $S=4$: $+148.6\,/\,+148.2$; $S=8$: $+21.6\,/\,
  +21.2$; $S=16$: $-11.4\,/\,-11.6$; $S=12$: $-2.8\,/\,-3.0$; $S=20$: $-15.3\,/\,-15.6$; weak $S=4$: $+30.8\,/\,+30.7$; $S=8$: $+3.7\,/\,+3.6$; $S=16$: $-3.4\,/\,-3.4$ pcm. All 8 inside their self-bars; $|\mathrm{pred}-\mathrm{meas}| \le 0.4$ pcm. Note the blind set spans row limits from $+148$ to $-15$ pcm: the predictor is not tuned to near-zero rows.

- All 40 rows (both absorbers, $S\le20$): $|\hat d_\infty| \le 22$ pcm with bars $\le 3$ pcm. The row-limit sequence $\hat d_\infty(S)$ --- the pure spatial residual $a(S)$, exposed by removing the angular channel --- is at least an order of magnitude smaller than the raw errors it replaces, and it is smooth and monotone (Fig. [2](#fig:limits){reference-type="ref" reference="fig:limits"}). Its measured shape is the honest one for a nodal spatial discretization on this checkerboard: $c_1 S^{-2}$-like decay at small $S$, a sign change near $S\approx11$ (the sub-grid spatial bias is positive on coarse grids and negative on fine ones), and an $O(S^{-1})$ tail beyond the crossing. This is a *characterized* residual: piecewise single-signed, extrapolable along $S$ like any discretization error, and verifiable by running one more $S$ level --- not the wandering per-point sign reversals of Table [1](#tab:rows){reference-type="ref" reference="tab:rows"}.

- Legacy, same formula: $(48,96)$ and $(96,192)$ give $+48.1$ / $+49.7$ at $S{=}12$, self-consistent to $1.6$ pcm --- and *wrong*: the limit is the body/face operator-error plateau, not the spatial residual. Self-consistency of an extrapolation is not evidence of its validity; only the operator-consistent method has limits with physical identity. Against the Monte Carlo reference the legacy limits sit at $+126.5, +49.7, +20.5, +6.2$ pcm for $S=8,12,16,20$ --- visible as error only with an external solution.

This is the operative difference for qualification, and the reason it matters practically is that *there is no reference*. In production core analysis the Monte Carlo solution used here as $k_{\mathrm{ref}}$ does not exist at design time: full-core depletion, control-rod worth, and transient scenarios are computed once, at one grid, and the analyst must state an uncertainty from the deterministic run itself. A valley-locked method cannot: its error at any point is a cancellation residual whose size is invisible without an external solution, and its own extrapolations converge to the operator plateau while looking self-consistent to $1.6$ pcm. The strict closure supplies the missing instrument: the row Richardson limit $\hat d_\infty(S)$ with its self-bar $E_M(S)$ estimates the angular-axis residual $b(M)$ at every computed point; the $S$-difference $|d(S{+}\Delta S,M)-d(S,M)|$ bounds the spatial term $a(S)$ (with $E_M$ subtracted); and separability --- itself verifiable at zero cost from the rectangle test above --- supports adding the two into a budget $|d| \lesssim |\hat d_\infty| + E_M + |a|$. Every term in that budget is produced by the method's own runs; none requires a reference. On the checkerboard, a strict computation at $M\le192$ thus knows its distance to the angular limit within $\sim1$ pcm and can allocate remaining budget to the correct axis --- a statement about the benchmarks tested here; Sec. [4.7](#sec:mox){reference-type="ref" reference="sec:mox"} exhibits an assembly (MOX) where the same self-checks correctly refuse to certify at $M\le48$, and the protocol of Sec. [4.3](#sec:protocol){reference-type="ref" reference="sec:protocol"} is written so that such refusals are an output, not a silent pass.

## Reference-free selection of $(S,M)$ {#sec:protocol}

The preceding subsection supplies the pieces; this states them as a protocol an analyst can run on a problem with no reference, given a target tolerance $\tau$ (say $5$ pcm). Every quantity below is produced by the strict runs themselves; "reference-free" here means no external solution is needed to *compute* the diagnostics --- physical validity is a separate question, and it is what the gate of Step 2 tests.

*Step 1: skeleton.* Six runs: two adjacent angular levels $S_1<S_2$ (we used $2,3$) by the triple $M_0,2M_0,4M_0$ with $M_0\ge48$. From these one estimates, per row: the row limit $\hat d_\infty$ (fit or $(4d_{2M}-d_M)/3$ on a measured law), the successive-extrapolation bar $E_M$, and the rectangle mixed second difference over the $2{\times}3$ window. $M_0$ is not chosen small on purpose: the Richardson second-order assumption behind $\hat d_\infty=(4d_{2M}-d_M)/3$ is not established below $M\approx48$ on these grids, and a premature extrapolation under-reports its own bar.

*Step 2: the separability gate --- before any acceptance.* Evaluate the rectangle mixed second difference of $d$ over the window (test of Eq. [\[eq:split\]](#eq:split){reference-type="eqref" reference="eq:split"}). This ordering is deliberate: $\hat d_\infty$ is a *diagnostic whose physical identity depends on separability*, so no row may be accepted on its bar before the gate has passed --- otherwise the protocol would walk straight into the trap this paper documents for legacy (self-consistent to $1.6$ pcm and $50$ pcm wrong). Small residual (UO2: pairwise row-limit differences $1.2$ to $5.2$ pcm with same-$M$ differences shrinking) licenses treating the axis-errors as independent. Large residual (MOX at $M\le48$: row-limit spread $38$ pcm, and Sec. [4.7](#sec:mox){reference-type="ref" reference="sec:mox"} shows the enlarged-$M$ re-test reducing it not at all --- $35.4$ pcm on $M=48/96/192$, now attributed to the $S$ axis) has two admissible responses and no third: enlarge the $M$ window and re-test, or return *unqualified --- residuals coupled at affordable cost*, which is an output of the protocol, not a failure of it.

*Step 3: set $M$ by the bar.* For each gated row accept $M$ when $E_M\le\tau/3$; otherwise slide the window one doubling up and re-extrapolate. The $\tau/3$ split is a sufficient heuristic, not a theorem: it is valid only if each term in Step 5 is independently bounded by its share; a user may distribute $\tau$ by any rule keeping every term bounded, and the reported triple makes any such repartition possible after the fact. The bar step is licensed by the strict operator only: fixed-$S$ refinement converges toward a single limit with a row-measured order (checkerboard $O(M^{-2})$, C5G7 $O(M^{-1})$), so $E_M$ measures a convergence remainder. Legacy self-consistency carries no such licence. Two honesty conditions attach to the bar. (i) When the window is slid, the window-to-window drift of $\hat d_\infty$ is part of the reported bar, not a superseded value: on MOX the $1/M$ limit moves $35$ pcm between the $M\le48$ and $M\ge48$ windows while each window fits itself to rms $\le2$ pcm. (ii) $E_M$ bounds the distance to the method's *own angular limit*, never to an external reference; $\tau$ must not be set below the spatial term's plausible size on an untested geometry, or a point several pcm from truth passes on a few-pcm angular bar (MOX M192/$S{=}2$: $-5.3$ pcm from the Monte Carlo reference, angular bar a few pcm --- the false pass this condition forbids).

*Step 4: set $S$ by the row limit.* With the angular channel extrapolated away, $\hat d_\infty(S)$ *is* the spatial residual $a(S)$ at that $S$. Its measured shape (Sec. [4.2](#sec:pred){reference-type="ref" reference="sec:pred"}: $c_1S^{-2}$ decay, a sign change near $S\approx11$, $O(S^{-1})$ tail) lets one accept the smallest $S^*$ with $|\hat d_\infty(S^*)|\le
\tau/3$ after one confirming extra level $S_3$ (or, equivalently, extrapolate along $S$ and run one confirmation point). No external solution enters at any step.

*Step 5: report the triple, not a point* $(\hat d_\infty(S),
E_M(S), a(S))$ with the budget $|d|\le|\hat d_\infty(S)|+E_M(S)+
|a(S)|$, each term at its measured value. The budget bounds the deterministic solution against whatever reference is later chosen; if that reference is itself a Monte Carlo run, its statistical uncertainty ($\pm3.6$ pcm for the C5G7 OpenMC references of this paper) adds incoherently to the bound and should be stated separately.

*What the protocol does not promise.* It does not promise that every problem enters the separable asymptotic regime by $M\le192$ --- MOX shows it need not --- it promises that when the regime has not been reached, the gate detects this and points at the axis to refine. It does not promise point accuracy at finite $(S,M)$: the correct output of a strict run is the triple $(\hat d_\infty,
E_M,a)$, not the raw point value, and quoting raw values inside the valley band is a use the framework explicitly disclaims. With those two readings, the guarantee the protocol offers is precise: at every computed point the distance to the angular limit is known to $\sim1$ pcm and the remaining budget is attributed to the correct axis, both without a reference.

## Cost to accuracy

Reaching $|\Delta k|<5$ pcm on the strong checkerboard: strict at $S{=}12, M{=}48$; legacy requires $S{=}20, M{=}96$ (an order of magnitude more unknowns) *and* degrades again at $M{=}192$ ($+4.5$ pcm, sign reversal) --- the optimum is a cancellation point, not a converged value.

## Cost parity

Table [2](#tab:cost){reference-type="ref" reference="tab:cost"}: strict and legacy wall times on the same solver (HPC node, 64 threads) agree within the scheduling noise at every matched point; the strict node evaluates four hyperbolic modes instead of four additive ones and its LU pools are identical in size. Peak memory is likewise mode-shared: the $O(M)$ LU-pool term is the same expression in both closures (both methods required the 120 GB allocation at $S{=}17$--$20$, $M{=}192$).

             $M=48$      $M=96$      $M=192$
  -------- ----------- ----------- -----------
  $S=4$       7 / 9      14 / 18     28 / 36
  $S=8$      18 / 20     37 / 41     76 / 85
  $S=12$     39 / 40     80 / 83    169 / 174
  $S=20$    117 / 108   230 / 206   418 / 407

  : Wall time, seconds, strong absorber (legacy / strict). {#tab:cost}

## Transfer to a heterogeneous assembly {#sec:c5g7}

On the C5G7 UO$_2$ rect-node assembly (51$\times$`<!-- -->`{=html}51 nodes, TY3, $I=30$, $k_\mathrm{ref}=1.334726$) legacy and strict wall times agree within 1% at every refinement (92/193/287 s vs 94/198/294 s at $M=$ 8/16/24, S=1): cost parity holds on a realistic composition. The S=1 row shows the pathology in its pure form: legacy crosses zero through the cancellation valley ($-101.0\to-37.2\to+1.2$ pcm at $M=8/16/24$); strict sits at $-264$ to $-285$ pcm, which *is* the true spatial error of an assembly model with unresolved fuel-pin rings. Rows $S=2,\dots,6$ over $M=8/16/24/48$ (thirty-two runs) give the quantitative contrast (reference $k_\mathrm{ref}=1.3347260$, OpenMC statistical uncertainty $\pm3.6$ pcm: common to all $32$ points at fixed geometry and XS, it cancels exactly in same-$M$ row differences and enters absolute limits at that floor):

::: center
+---------------------------------------------------------------------------------------------+
| $\Delta k_\mathrm{eff}$ (pcm) vs. Monte Carlo reference                                     |
+:===============+===========:+===========:+===========:+===========:+=======================:+
| row            | $M{=}8$    | 16         | 24         | 48         | $\hat d_\infty\,(1/M)$ |
+----------------+------------+------------+------------+------------+------------------------+
| legacy $S{=}2$ | $-99.4$    | $-50.1$    | $-21.0$    | $+7.5$     | $+21.9$                |
+----------------+------------+------------+------------+------------+------------------------+
| strict $S{=}2$ | $-125.3$   | $-85.0$    | $-56.8$    | $-27.9$    | $-17.3$                |
+----------------+------------+------------+------------+------------+------------------------+
| legacy $S{=}3$ | $-102.6$   | $-60.1$    | $-33.9$    | $-6.3$     | $+5.3$                 |
+----------------+------------+------------+------------+------------+------------------------+
| strict $S{=}3$ | $-113.3$   | $-75.6$    | $-52.6$    | $-29.0$    | $-18.5$                |
+----------------+------------+------------+------------+------------+------------------------+
| legacy $S{=}4$ | $-103.7$   | $-64.3$    | $-38.9$    | $-11.1$    | $-0.9$                 |
+----------------+------------+------------+------------+------------+------------------------+
| strict $S{=}4$ | $-109.0$   | $-72.4$    | $-48.8$    | $-22.7$    | $-13.3$                |
+----------------+------------+------------+------------+------------+------------------------+
| strict $S{=}5$ | $-107.4$   | $-71.2$    | $-47.8$    | $-22.0$    | $-12.7$                |
+----------------+------------+------------+------------+------------+------------------------+
| strict $S{=}6$ | $-106.4$   | $-70.2$    | $-46.9$    | $-20.8$    | $-11.6$                |
+----------------+------------+------------+------------+------------+------------------------+
:::

Measured order first, extrapolation second (same protocol as the checkerboard): unlike the checkerboard's clean $O(M^{-2})$, the C5G7 rows decay $O(M^{-1})$ --- per-refinement increments $49,29,28$ pcm barely shrink, the signature of the material-interface flux kinks that a Cartesian node grid resolves only linearly. The four-point least-squares fit to $1/M$ has rms $5$--$7$ pcm versus $11$--$14$ for a forced $1/M^2$ fit; using the measured law is what makes the row limits meaningful. With it: (1) The three *strict* row limits stay on one side of zero ($-17.3$, $-18.5$, $-13.3$ for $S=2,3,4$). Their $5.2$ pcm spread must be read honestly: the three rows share one $M$ grid, their fit residual vectors are the same shape (pairwise correlation $>0.99$), so the spread carries a large common-mode component and is not an independent-draw noise band --- three correlated fits cannot establish that any part of it is noise. The informative statistic is the same-$M$ row difference, a *common-grid cancellation* that removes whatever is row-independent under the additive split (assumed, and itself what the gate tests), leaving only $a(S)$ variation: strict $S{=}3{-}2$ shrinks from $+12.0$ pcm at $M{=}8$ to $-1.1$ at $M{=}48$, and $S{=}4{-}3$ is already at noise size throughout ($+4.3$ to $+6.3$); legacy's differences instead *grow* with $M$ ($S{=}3{-}2$: $-3.2\to-13.8$; $S{=}4{-}3$: $-1.1\to-4.8$, or $+9.6\to+17.9$ against the strict row at the same $M$). The new $S{=}5$ and $S{=}6$ rows settle the question the three-row data left open: is the $-4$ pcm rise from $S{=}2/3$ to $S{=}4$ real $a(S)$ residual or common-mode fit drift? The same-$M$ differences answer without any fit --- strict $S{=}5{-}4$ is $+1.6$ to $+0.7$ pcm across $M=8$ to $48$ and $S{=}6{-}5$ is $+1.2$ to $+0.9$, i.e. beyond $S{=}4$ the spatial term has decayed into the reference floor and fit noise, while legacy's $S{=}4$ row at the same $M$ still sits $9.6$ to $17.9$ pcm *above* strict's $S{=}3$. On this assembly the spatial axis is functionally exhausted at $S{=}4$ and the residual budget is the spatial constant $b\approx-13$ to $-18$ pcm (row limits $-17.3$, $-18.5$, $-13.3$, $-12.7$, $-11.6$ for $S=2$ to $6$: monotone, sub-$19$, converging as $a(S)$ drains out of the limit). The raw limits tell the same story once their trend is read correctly: strict limits approach zero monotonically in $S$ (residual $a(S){+}b$, angular channel removed; the confirming extra levels required by Step 4 of Sec. [4.3](#sec:protocol){reference-type="ref" reference="sec:protocol"} are exactly the $S{=}5,6$ runs, and they return the $a(S)$-settles verdict stated above), legacy limits cross zero nonmonotonically (range $23$ pcm: $+21.9$, $+5.3$, $-0.9$) --- crossing being the signature of a cancellation channel, not of convergence. (2) The valley trap in production dress: legacy $S{=}2$ at $M{=}24$ is $-21$ pcm and at $M{=}48$ has crossed zero to $+7.5$ --- $|d|$ halving toward the reference all the way, published as converged, while its own extrapolation points past zero to the operator plateau ($+21.9$). Strict stops short of the reference at $\approx$$-18$ pcm and *says so*: the row limit with its successive-fit spread is the angular budget; the measured $S{=}2{\to}3$ drift bounds the spatial term; the reference-free total at $M{=}48$, $S{=}3$ is $\sim$$50$ pcm with the dominant term (spatial, reducible only by $S$) correctly identified. (3) On the same four-point data, a practitioner without a reference can still catch legacy: its rectangle (separability) residual is $6.8$ vs $2.6$ pcm on the $(8,16)$ window, and its refinement increments grow non-monotonically toward zero-crossing rather than decaying to a same-signed limit.

## Second assembly: MOX, and where the budget degrades {#sec:mox}

The MOX assembly ($k_\mathrm{ref}=1.1851053$, OpenMC uncertainty $\pm3.3$ pcm, strong Pu peaks, $4.4\times$ leakage of UO$_2$) stress-tests the transferability claim, and the answer is split.

**What transfers.** Legacy $S{=}2$ crosses zero on the same trajectory: $-392.3$, $-125.1$, $-28.3$, $+49.8$ pcm at $M=8/16/24/48$ --- at $M{=}24$ the "error" has entered the 30 pcm band a qualification review would accept, and one doubling later it flips sign. Strict stays one-sided and monotone at every one of its twenty points, $-461$ to $-5.3$ pcm over $M=8$ to $192$: the sign guarantee holds even where the quantitative budget gets hard. Legacy's $S{=}2{\to}3$ shift grows systematically with $M$ ($-31.6$ to $-51.2$ pcm), the entangled pattern seen on UO$_2$.

::: center
+----------------------------------------------------------------------------------------------------------------------+
| $\Delta k_\mathrm{eff}$ (pcm) vs. Monte Carlo reference ($k_\mathrm{ref}=1.1851053$, $\pm3.3$ pcm)                   |
+:===============+===============:+===============:+===============:+===============:+===============:+===============:+
| row            | $M{=}8$        | 16             | 24             | 48             | 96             | 192            |
+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
| strict $S{=}2$ | $-461.3$       | $-215.9$       | $-120.0$       | $-41.0$        | $-12.9$        | $-5.3$         |
+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
| strict $S{=}3$ | $-446.6$       | $-219.6$       | $-139.7$       | $-73.6$        | $-48.1$        | $-39.4$        |
+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
| strict $S{=}4$ |                |                |                | $-54.6$        | $-25.8$        | $-16.2$        |
+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
| strict $S{=}5$ |                |                |                | $-54.6$        | $-27.3$        | $-18.5$        |
+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
| strict $S{=}6$ |                |                |                | $-51.9$        | $-23.8$        |                |
+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
:::

**What degrades, and where the fix lies.** The row Richardson limits no longer agree: with the measured $1/M$ factor over $M=8$ to $48$, strict $\hat d_\infty$ is $+43.7$ pcm ($S{=}2$) vs $+5.7$ ($S{=}3$) --- a spread of $38$ pcm against $1.2$ to $5.2$ on UO$_2$ --- and the $S$ shift no longer decays with $M$. The increments give the reason: $245$, $96$, $79$ pcm --- the second ratio is $1.2$, not the $\approx 2$ of the UO$_2$ rows, i.e. MOX pin-by-pin heterogeneity has an additional slowly-decaying component in the $M\le48$ window. The $M\ge96$ and $S\ge4$ extensions identify which axis carries it, and the answer is not the one a M-window fix would want: beyond $M{=}48$ the angular axis *recovers* clean behavior (S2 increments $28.1$, $7.6$ pcm --- ratios $2.8$, $3.7$, converging to the $2$ of a $1/M$ law; fits with rms $1.2$ to $2.0$ pcm), yet the two-row limit disagreement *survives* the extension ($+8.7$ vs $-26.7$ pcm on the shared $M=48/96/192$ window, spread $35.4$) and the $S{=}2{\to}3$ shift saturates rather than decays ($-32.6$, $-35.2$, $-34.1$ pcm at $M=48/96/192$). The $S\ge4$ runs then settle the attribution directly, measured rather than inferred: at $M{=}48$ the $S{=}4$ and $S{=}5$ points agree bit-for-bit ($k_\mathrm{eff}$ identical to seven digits, both $-54.6$ pcm) and $S{=}6$ adds $+2.7$ pcm; at $M{=}96$ the exhausted row repeats the profile within a few pcm ($-25.8$, $-27.3$, $-23.8$ pcm, spread $3.5$ against UO$_2$'s same-$M$ spread of $0.9$) --- the $a(S)$ channel is exhausted by $S{=}5$--$6$ on MOX, the same drainage profile as UO$_2$'s same-$M$ increments ($+1.6$ to $+0.7$ from $S{=}4$ to $S{=}6$) against the $+19$ pcm $S{=}4{-}3$ step left behind --- and the converged-row offset from $S{=}2$, $-13.6$ at $M{=}48$ and $-10.9$ to $-12.3$ across the three $S{\ge}4$ points at $M{=}96$, is $M$-independent to $\approx$`<!-- -->`{=html}3 pcm: an $M$-independent inter-row offset is $a(S)$ structure --- spatial, invisible to any $M$ refinement --- and its MOX size ($\approx$$13$ pcm against the exhausted $S$ row, where the checkerboard and UO$_2$ same-$M$ converged-row offsets sit at noise size) is the pin-by-pin Pu kink field saying that this assembly needs $S$, not $M$. Note also that the $S{=}3$ row is *deeper* than both its neighbours ($-73.6$ vs $-41.0$ and $-54.6$ at $M{=}48$): $\hat d_\infty(S)$ is non-monotone on MOX, the same sign-change family the checkerboard shows near $S\approx11$. The methodological point is the ordering: with only $S\le3$ the analyst sees a $35$ pcm row-limit disagreement and cannot tell an unconverged $a(S)$ from a converged spatial constant; adding one exhaustion check at $S{=}4$ resolves the question in one run pair. The strict machinery reports exactly this, gate and cross-check firing and attributing the surviving term to the spatial axis; the honest output for MOX at affordable $M$ is *unqualified --- spatial term dominant*, with the direction to refine now correctly stated as $S$. Legacy carries no flag at all --- its M48/$S{=}3$ error of $-1.4$ pcm looks superb and is pure cancellation.

A probe-integrity note in fairness: an earlier MOX strict batch was temporarily voided over a closure-name typo in the probe (silent fallback path); the API now rejects unknown closure names, and the re-run reproduced all seven overlapping points *bit-identically* --- confirming both the determinism claim and that the original numbers were computed on the correct kernel after all.

## Audits and honest limitations {#sec:audit}

Three skeptic-tests of the predictability claim were run on the sweep data; two of them legacy passes, and saying so sharpens what the claim actually is. Table [3](#tab:tests){reference-type="ref" reference="tab:tests"} ranks all candidate discriminators by measured behavior; the prose below gives each test's protocol.

+----------------------------------------+-----------------------------+-----------------------------+
| Test                                   | strict                      | legacy                      |
+:=======================================+:============================+:============================+
| self-consistent Richardson             | 17/19                       | 19/19 (no signal)           |
+----------------------------------------+-----------------------------+-----------------------------+
| monotone approach to own limit         | 19/19                       | 19/19 (no signal)           |
+----------------------------------------+-----------------------------+-----------------------------+
| refinement helps, $|d(S,2M)|<|d(S,M)|$ | 12/20, 12/20                | 3/20, 5/20 (strong)         |
+----------------------------------------+-----------------------------+-----------------------------+
| rectangle separability, full map       | 1.5 / 0.3 pcm               | 16.9 / 3.3 pcm              |
+----------------------------------------+-----------------------------+-----------------------------+
| body$\equiv$face, $M{=}2$ anchor       | algebraic: strict passes, legacy fails by construction    |
+----------------------------------------+-----------------------------------------------------------+

: Candidate convergence-certification tests, measured on the checkerboard sweeps (strong/weak absorber where two numbers). A qualifier executes rows 3--5 plus the code-level row 6. {#tab:tests}

*(i) Self-consistency is not a discriminator.* Applying the Richardson recipe with data up to $M{=}96$ and predicting the $M{=}192$ point: strict rows hit within their self-bars on 17/19 (median error $0.9$/$0.2$ pcm strong/weak; the two misses are coarse rows $S{=}2,4$ exceeding their bars by $\le$`<!-- -->`{=html}1.3 pcm). Legacy rows, *measured against their own wrong-operator limit*, are *more* self-consistent (19/19, median $0.5$/$0.1$ pcm). An extrapolation of a consistent wrong operator looks exactly as self-consistent as one of a consistent right operator. Extrapolation quality therefore cannot certify an operator --- only the analytical body$\equiv$face identity (Sec. [3](#sec:strict){reference-type="ref" reference="sec:strict"}) plus the two behavioral tests below can.

*(ii) Refinement-direction test (legacy fails).* Without any reference, ask whether refining helps: the fraction of $S$-rows where $|d(S,2M)| < |d(S,M)|$. At $M{:}48{\to}96$: strict 12/20 (both absorbers) versus legacy 3/20 and 5/20; at $96{\to}192$: strict 10--11/20 versus legacy 0/20 and 3/20. A practitioner running two adjacent grids can detect and identify the pathology from their own numbers: if refining the angular axis *often increases* the distance to the finest grid in hand, the row limits are not physical.

*(iii) Fixed-budget point accuracy (no winner --- by design).* Over the production-like sub-grid $S\le20$, $M\in\{8,16,24,48\}$ (80 points per combination), the fraction with $|\Delta k|<20$ pcm is legacy 12%/strict 8% (strong) and 40%/44% (weak); at the 50 pcm threshold 22%/24% and 62%/69%. Neither method systematically wins single-point accuracy at equal grid --- which is precisely the point: legacy's best points are valley cancellations that (a) move when the benchmark's material contrast changes and (b) are, by test (i), indistinguishable from its worst points without a reference. The strict method trades unknowable lucky points for knowable ones.

*(iv) The executable procedure, and what each test buys.* Ranking the tests by measured discrimination on the data above: self-consistency of extrapolation --- no discrimination (legacy passes); monotone approach to the method's *own* limit --- no discrimination (both 19/19; a wrong operator is still an operator); refinement-direction against the finest grid in hand --- **strong** (12/20 vs 0--5/20); rectangle separability --- **strong**: full-map medians $0.3$--$1.5$ pcm (strict) vs $3.3$--$16.9$ pcm (legacy), and over 119 sliding four-row three-column windows the $>$`<!-- -->`{=html}20 pcm non-separability flag fires on 10/238 strict windows vs 35/238 legacy --- a $3.5\times$ flag-rate difference; kernel-anchor and body$\equiv$face identity tests --- **conclusive, and free**: they are algebraic properties of the implementation, executed in the regression suite, not inferred from sweeps. A qualifier's procedure is therefore: (1) verify by code inspection or the release's anchor tests that body and face currents of the production node are divergence-theorem pairs (one run, no reference); (2) on two production grids, run the refinement-direction count --- frequent degradation under refinement is a stop condition regardless of how small the errors look; (3) measure the row's local order $p$ from adjacent-doubling increments and extrapolate with the measured factor $2^p$, never an assumed one: the checkerboard rows run $p\approx2$ but assembly rows with material-interface kinks run $p\approx1$ (Sec. [4.6](#sec:c5g7){reference-type="ref" reference="sec:c5g7"}), and a forced $1/M^2$ recipe mis-sets those row limits by $10$--$20$ pcm; (4) budget the angular axis by row Richardson with self-bars; (5) budget the spatial axis by the measured row-limit sequence $\hat d_\infty(S)$, choosing $S$ from its trend (never by the smallest-$|d|$ point, which is the valley); (6) never use point-vs-point closeness as evidence of convergence.

*(v) Convergence order is measured, not assumed.* Per-row local orders from adjacent-doubling increments, $p = \log_2|\Delta d(M)/\Delta d(2M)|$, give (strict) median $p=1.87$, 10--90% range $1.81$--$2.04$ (strong) and $1.91$, $1.74$--$1.97$ (weak) --- consistent with $O(M^{-2})$ without fitting it in. The order is a property of the *problem*, not of the closure: on the C5G7 assembly the same strict rows measure $p\approx1$ (Sec. [4.6](#sec:c5g7){reference-type="ref" reference="sec:c5g7"}), and the procedure therefore extrapolates with the locally measured factor. The Richardson formula with factor 2 is used only on doubling pairs; the sweep's non-doubling columns ($M=12,16,24$) are displayed in maps but never enter an extrapolation.

*(vi) S-axis blind extension.* Increment-ratio forecasts for $d(S,48)$, $S=21\ldots26$, issued from $S\le20$ data and frozen before the runs: strict errors are $0.3$--$0.6$ pcm (weak) and $0.3$--$5.4$ pcm (strong; the forecast slightly overshoots the drop because the measured tail beyond the $a(S)$ zero crossing decays slower than geometric --- the $O(S^{-1})$ tail again); legacy forecasts diverge from their own later behavior by up to $31.8$ pcm (strong, median 14.3), the extrapolated $S\approx M/2$ drift continuing past the forecast horizon. Spatial- axis extrapolation is thus honest-but-slow for strict and unreliable-by-construction for legacy, mirroring the $M$-axis story.

*(vii) What remains genuinely open.* A single-method, single-benchmark audit cannot distinguish "converged to a wrong operator" from "converged" without either the analytical consistency proof or a second discretization family (e.g. running the same problem at a different $P$-order basis). The strict closure supplies the first internally; the release therefore pairs every convergence claim in this paper with its kernel-level anchor test, so that a user's regression suite, not a Monte Carlo campaign, is the referee.

# Implementation {#sec:impl}

The closure ships in OpenPSN as the YAML setting `angular.closure: strict_phase_space` (module `psn2d/node_strict.py`, node-map dispatch in the solver; legacy default bit-identical). Gates in `tests/`: kernel anchor (relative $4\times10^{-11}$, machine-cancellation level), state-dict closure keys, global $M{=}2$ anchor ($8.9\times10^{-11}$ pcm), and the full example regression (7/7, $\Delta=0.000$ pcm). Production-path spot checks (YAML switch, generic $I{=}30$, rect path) reproduce the sweep-driver values to the solver's source-iteration floor: reported $\Delta k$ bit-identical ($-2540.6$/$-83.2$ pcm), keff agreeing to $1.4\times10^{-7}$, the iteration-residual floor. Every point reported in this paper --- 777 (closure, case, $M$, $S$, $k_{\mathrm{eff}}$, $\Delta k$) tuples across the checkerboard, C5G7-UO$_2$, and C5G7-MOX cases --- is released as `data/all_points_archive.csv` beside the manuscript, and all separability, refinement-direction, and self-consistency statistics in Sec. [4.8](#sec:audit){reference-type="ref" reference="sec:audit"} are recomputed from that file by a single script, so the audit table is checkable without rerunning any solver. All points --- early local runs and HPC-array runs alike --- execute the same kernel revision (method-commit lineage), the same solver tolerance $10^{-10}$ and the same reference constants; machine differences enter wall times only, and every extrapolation reported here compares keff values, not times.

# Related work and scope

Diagnoses of "angular effects polluting a spatial solver" are known under various names (ray effects in SN [@lewis1993], checkering in diffusion-synthetic accelerations); the mechanism isolated here is different: a *within-node* operator inconsistency between body and face, invisible in smooth problems, diagonal in the two-grid error map, and removable by tensor-consistent body basis construction. Phase-space nodal/characteristics hybrids whose face and body tensors are already consistent (e.g. method-of-characteristics nodes [@askew2012]) do not exhibit the pathology; the strict closure brings PSN into that class without changing its cost. Scope: fixed source and $k$-eigenvalue transport with isotropic scattering, 2D, within the accuracy class of the base PSN method; upscattering, vacuum boundaries, and 3D generalizations of the basis lemma (plane waves along the cone $\theta_w = \phi_m \pm \pi/4$ generalize sector-wise in 3D; the $S_N$-order coupling is unchanged) are open. The predictability evidence itself is scoped: $k$-eigenvalue checkerboards (1-group transport, void-free, isotropic scattering) carries the quantitative claims; the C5G7 transfer demonstrates cost parity and the zero-crossing pathology on a heterogeneous composition but not yet a closed error budget (S-levels 2--3 in progress). A fixed-source demonstration and a rod-lattice valley geometry are the next evidence items; neither should be assumed until run.

# Conclusions

A nodal method is qualified when its error is predictable without reference. For PSN the obstacle was a body operator inconsistent with its own face currents; making the node body solve the exact sector tensor equation --- at zero cost growth and with an exact $M{=}2$ anchor --- converts the checkerboard convergence map from a locked valley into 40 clean per-row $O(M^{-2})$ sequences whose extrapolated limits are the spatial residual, self-bounded to sub-pcm, and form a smooth monotone sequence under spatial refinement whose own $S$-trend is extrapolable. The diagonal zero contour remains in the error map --- it must, in any consistent method --- but it is demoted from an attractor that governs convergence to a topological fact that no longer matters. Operationally the user does four reference-free things: confirm the node's body$\equiv$face identity once (regression test); count refinement direction on two production grids and stop if refinement often hurts; budget the angular axis by row Richardson with self-bars; and choose $S$ from the trend of the measured row-limit sequence, never from the smallest-error point.

::: thebibliography
9 J. Chao, Y. Xu, et al., Phase-space nodal method for neutron transport with angular-moment phase-space unknowns, Ann. Nucl. Energy 240 (2027) 112707. E.E. Lewis, W.F. Miller, Computational Methods of Neutron Transport, American Nuclear Society, 1993. A.R. Askew, K.R. Ferreri, M.A. Lewis, Linear characteristics method for fixed-source calculations, Nucl. Sci. Eng. 170 (2012) 164--179.
:::

# Figures {#figures .unnumbered}

<figure id="fig:bestS" data-latex-placement="t">
<img src="figures/fig_bestS.png" />
<figcaption>Optimal spatial grid at fixed azimuthal order. Legacy <math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><msup><mi>S</mi><mo>*</mo></msup><annotation encoding="application/x-tex">S^{*}</annotation></semantics></math> tracks the cancellation band <math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>≈</mo><mi>M</mi><mi>/</mi><mn>2</mn></mrow><annotation encoding="application/x-tex">S\approx M/2</annotation></semantics></math> out of every finite grid; strict <math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><msup><mi>S</mi><mo>*</mo></msup><annotation encoding="application/x-tex">S^{*}</annotation></semantics></math> saturates at the zero crossing of the spatial residual.</figcaption>
</figure>

<figure id="fig:limits" data-latex-placement="t">
<img src="figures/fig_limits.png" />
<figcaption>Richardson <math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>M</mi><mo>→</mo><mi>∞</mi></mrow><annotation encoding="application/x-tex">M\to\infty</annotation></semantics></math> limit of each fixed-<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>S</mi><annotation encoding="application/x-tex">S</annotation></semantics></math> row (error bar: extrapolation self-disagreement). Strict limits form a single-signed smooth spatial-residual sequence (sign-crossing at <math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>S</mi><mo>≈</mo><mn>11</mn></mrow><annotation encoding="application/x-tex">S\approx11</annotation></semantics></math>, <math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>O</mi><mo stretchy="false" form="prefix">(</mo><msup><mi>S</mi><mrow><mi>−</mi><mn>1</mn></mrow></msup><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">O(S^{-1})</annotation></semantics></math> tail — measured, not assumed); legacy limits plateau at the body/face operator-error offset.</figcaption>
</figure>

[^1]: Corresponding author.
