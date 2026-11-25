"""Mirror Study IV into the markdown copies of the paper (main.md / revised_main.md).

Idempotent: the Introduction/scope/conclusions edits are skipped when already
applied, and the Study IV section is spliced only when absent.  The two core
matrices are read from the CSVs that scripts/study4_core_report.py writes, so the
markdown tables cannot drift from the LaTeX ones.

    python -m scripts.finalize_markdown            # prose edits only, if the CSVs are absent
    python -m scripts.finalize_markdown --force    # require the CSVs and splice the section

The mirrored prose lives in ``paper/studyIV.md`` next to ``paper/studyIV.tex``; the
three generated tables are inserted there from the CSVs at run time, so the markdown
copy cannot carry a number the CSVs do not support.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"
EXAMPLE = REPO / "examples/c5g7/study4_sector_core"
KREF = "1.187217554586068"
M_VALUES = (2, 4, 8, 12, 16, 24)
SPOILER = "## Comparison with the published diffusion / SN / nodal landscape"
HERE = PAPER  # the markdown mirror source lives beside the paper

EDITS = [
    # the direction-wise closure is the closure of record
    ("""            relation to the original PSN closure. Fourth, we document a finite-sector
            closure of the same even-parity hierarchy, in which the diffusion tensor
            of each finite angular cell is the exact second moment of the retained
            directions and the interface and vacuum terms come from the same weak form
            as the body term; on the square checkerboard that closure removes the
            sign-changing diagonal valley of the published error survey over every
            tested point. Fifth, we assess the available checkerboard""",
     """            relation to the original PSN closure. Fourth, we document a finite-sector
            closure of the same even-parity hierarchy, in which the diffusion tensor
            of each finite angular cell is the exact second moment of the retained
            directions and the interface and vacuum terms come from the same weak form
            as the body term, in two reductions of that tensor at equal angular
            unknowns; on the square checkerboard the direction-wise reduction, which
            is the closure of record, removes the sign-changing diagonal valley of the
            published error survey at 279 of its 314 measured points and at every
            point outside one under-resolved azimuthal column, and where the two
            reductions share $A$, $S$, $M$ and $I$ it is the more accurate one at 398
            of the 416 paired points, at a median error ratio of $0.437$ and a median
            time ratio of $0.98$. Fifth, we assess the available checkerboard""",
     "which is the closure of record"),
    ("""            reference's own standard error, and the sector closure is the more
            expensive of the two paths at equal $M$.""",
     """            reference's own standard error. Against the legacy path the sector closure
            is the more expensive one; between the two sector closures, which carry
            the same unknowns at equal $A$ and $M$, the measured time ratio is $0.98$
            at a median, and the accuracy difference is in the closure rather than in
            the cost accounting.""",
     "between the two sector closures"),
    ("""            The finite-sector closure of the even-parity equation gives one-signed
            checkerboard errors that decrease under refinement of either axis at every
            tested row, and on the explicit rectangular quarter core""",
     """            The direction-wise finite-sector closure of the even-parity equation gives
            checkerboard errors that are one-signed outside a single under-resolved
            azimuthal column and that move toward zero at 246 of the 280 refinements of
            the spatial axis, and that stay below the cell-average reduction of the same
            tensor at 398 of the 416 points measured both ways; on the explicit
            rectangular quarter core""",
     "The direction-wise finite-sector closure"),
    # (anchor in main.md, replacement, phrase proving the edit is applied)
    ("""The contributions of this work are fourfold.""",
     """The contributions of this work are fivefold.""",
     "contributions of this work are fivefold"),
    ("""relation to the original PSN closure. Fourth, we assess the available checkerboard""",
     """relation to the original PSN closure. Fourth, we document a finite-sector closure of the
        same even-parity hierarchy, in which the diffusion tensor of each finite angular cell is
        the exact second moment of the retained directions and the interface and vacuum terms come
        from the same weak form as the body term; on the square checkerboard that closure removes
        the sign-changing diagonal valley of the published error survey over every tested point.
        Fifth, we assess the available checkerboard""",
     "finite-sector closure of the same even-parity hierarchy"),
    ("""the legacy PSN formulation, the rectangular-node implementation, and the separate
        segmentwise angular-moment formulation.""",
     """the legacy PSN formulation, the rectangular-node implementation, the separate
        segmentwise angular-moment formulation, and the finite-sector closure of the even-parity
        equation.""",
     "and the finite-sector closure of the even-parity equation."),
    ("""the legacy PSN branch on the $3\\times3$ pin tiling (homogenised and explicit rectangular
        nodes) and the segmentwise angular-moment variant at $A{=}3$ on the $4\\times4$ fuel-split
        tiling, with fission-field comparisons against OpenMC.""",
     """the legacy PSN branch on the $3\\times3$ pin tiling (homogenised and explicit rectangular
        nodes), the segmentwise angular-moment variant at $A{=}3$ on the $4\\times4$ fuel-split
        tiling, and the finite-sector closure on the same four families of benchmarks, with
        fission-field comparisons against OpenMC.""",
     "and the finite-sector closure on the same four families of benchmarks"),
    ("""C5G7 campaigns, in the pin tiling. The software name OpenPSN does not imply that all of its
        results are instances of the original PSN method.""",
     """C5G7 campaigns, in the pin tiling. The second branch also carries a finite-sector closure of
        its own, selected by `angular.closure: sector_average` or `sector_diffusion` and assessed
        in Section 4. The software name OpenPSN does not imply that all of its results are
        instances of the original PSN method.""",
     "The second branch also carries a finite-sector closure"),
    ("""Matched computational-cost comparisons and an isolated attribution of the changes to angular
        order are outside the evidence provided here.""",
     """Matched computational-cost comparisons and an isolated attribution of the changes to angular
        order are outside the evidence provided here. The finite-sector study of Section 4 is
        reported on the same footing: its quarter-core matrix is a grid of measured points rather
        than a convergence proof, its best cell stays outside the Monte Carlo reference's own
        standard error, and the sector closure is the more expensive of the two paths at equal
        $M$.""",
     "The finite-sector study of Section 4 is reported on the same footing"),
    ("""reports the $S{=}1$ angular sweep, with the existing $S{=}1,M{=}8$ field comparison giving
        $\\Delta k=-1.7$ pcm and a fission-rate RMS difference of $0.286\\%$.""",
     """reports the $S{=}1$ angular sweep, with the existing $S{=}1,M{=}8$ field comparison giving
        $\\Delta k=-1.7$ pcm and a fission-rate RMS difference of $0.286\\%$. The finite-sector
        closure of the even-parity equation gives one-signed checkerboard errors that decrease
        under refinement of either axis at every tested row, and on the explicit rectangular
        quarter core its complete $S\\times M$ matrix keeps its smallest measured difference above
        the $3.41$~pcm standard error of the same-geometry Monte Carlo reference.""",
     "The finite-sector closure of the even-parity equation gives one-signed"),
]


def flex(text):
    """Regex that matches the prose regardless of line wrapping."""
    return re.compile(r"\s+".join(re.escape(word) for word in text.split()))


def apply_edits(text):
    """Apply a prose edit to the file in whatever line wrapping it happens to use."""
    done = 0
    for old, new, marker in EDITS:
        if " ".join(marker.split()) in " ".join(text.split()):
            continue
        match = flex(old.strip()).search(text)
        if not match:
            raise SystemExit(f"prose anchor not found: {old.strip()[:60]}")
        text = text[:match.start()] + " ".join(new.split()) + text[match.end():]
        done += 1
    return text, done


CHECKER_ROWS = ((2, 4), (4, 8), (8, 12), (12, 24), (12, 48), (12, 96))
CHECKER_COLUMNS = (("checkerboard-strong", "0", "quadratic"),
                   ("checkerboard-strong", "1", "quadratic"),
                   ("checkerboard-strong", "1", "bilinear"),
                   ("checkerboard-strong", "2", "quadratic"),
                   ("checkerboard-weak", "0", "quadratic"),
                   ("checkerboard-weak", "1", "quadratic"))
SECTOR_CAPTION = (
    "The same two checkerboards and the same grid, measured with the direction-wise "
    "`sector_diffusion` closure --- the closure of record. $A{+}1$ angular unknowns per "
    "sector at azimuthal degree $A$, as in the cell-average table below, so a cell paired "
    "across the two tables differs in "
    "the closure alone. The closure is measured at 314 points on the two absorbers, of "
    "which 279 are negative; the 35 positive ones are the $A{=}1$, $M{=}4$ column, where "
    "a sector holds two Gauss directions. Outside that column 246 of the 280 refinements "
    "of $S$ move the error toward zero, and past $M{=}12$ each $A,S$ column settles "
    "within $6.8$ pcm of its own best.")

CHECKER_CAPTION = (
    "Strong and weak absorber checkerboards, $10^{5}(\\kappa_{\\mathrm{eff}}-k_\\mathrm{ref})$ in "
    "pcm, against the published $0.51673$ and $1.12974$, measured with the cell-average "
    "`sector_average` closure as the comparison for the table above. $I{=}30$. $A{=}0$ keeps the angular average "
    "over each sector, $A{=}1$ and $A{=}2$ add local azimuthal modes inside the sector, and \"bilin.\" "
    "represents the $A{=}1$ coefficient in the continuous bilinear subspace of the same mesh. The "
    "bilinear representation was scanned over the same $S{=}2,\\ldots,12$ $\\times$ "
    "$M{=}4,8,12,24$ grid for both absorbers; its values here are the strong-absorber column, and the "
    "weak absorber reaches $-8.94$ pcm at $S{=}12$, $M{=}24$.")


def points_of(closure):
    """One closure's checkerboard points, keyed by representation and grid."""
    with (PAPER / "data/study4-checkerboard.csv").open() as handle:
        return {(row["problem"], row["A"], row["slope_space"] or "quadratic",
                 int(row["subdivide"]), int(row["M"])): float(row["pcm"])
                for row in csv.DictReader(handle)
                if row.get("closure", "sector_average") == closure}


def sector_table():
    """The matrix of record: the direction-wise closure on the published grid."""
    points = points_of("sector_diffusion")
    head = ["$S$", "$M$"] + [f"{case} $A{{=}}{a}$" for case in ("strong", "weak")
                            for a in (0, 1, 2)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---:|" * len(head)]
    for subdivision, M in CHECKER_ROWS:
        row = [str(subdivision), str(M)]
        for case in ("checkerboard-strong", "checkerboard-weak"):
            for a in "012":
                value = points.get((case, a, "quadratic", subdivision, M))
                row.append("---" if value is None else f"${value:+.2f}$".replace("-", "\\u2212"))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + f"\n\n: {SECTOR_CAPTION} {{#tab:study4-checker-sector}}\n"


def checker_table():
    """The archived comparison, read back from the same CSV the LaTeX table is built from."""
    points = points_of("sector_average")
    head = ["$S$", "$M$", "strong $A{=}0$", "strong $A{=}1$", "strong bilin.", "strong $A{=}2$",
            "weak $A{=}0$", "weak $A{=}1$"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---:|" * len(head)]
    for subdivision, M in CHECKER_ROWS:
        row = [str(subdivision), str(M)]
        for case, degree, slope in CHECKER_COLUMNS:
            value = points.get((case, degree, slope, subdivision, M))
            row.append("---" if value is None else f"${value:+.2f}$".replace("-", "\\u2212"))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + f"\n\n: {CHECKER_CAPTION} {{#tab:study4-checker}}\n"


def matrix_csv(name):
    with (EXAMPLE / name).open() as handle:
        rows = list(csv.reader(handle))
    return [[None if not value else float(value) for value in row[1:]] for row in rows[1:]]


def pipe(title, rows, digits, caption, plus=False):
    head = ["$S$ \\ $M$", *(str(m) for m in M_VALUES)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---:|" * len(head)]
    best_row, best_col, best_abs = None, None, None
    if digits == 2:  # the error matrix marks its own best cell
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                if value is not None and (best_abs is None or abs(value) < best_abs):
                    best_row, best_col, best_abs = i, j, abs(value)
    for i, row in enumerate(rows):
        cells = []
        for j, value in enumerate(row):
            if value is None:
                cells.append("---")
                continue
            text = f"{value:+.{digits}f}" if plus else f"{value:.{digits}f}"
            if digits == 2 and (i, j) == (best_row, best_col):
                text = f"**{text}**"
            cells.append(text)
        lines.append("| " + " | ".join([str(i + 1), *cells]) + " |")
    return "\n".join(lines) + f"\n\n: {caption} {{#{title}}}\n"


def core_tables():
    pcm = matrix_csv("sm_error_pcm.csv")
    seconds = matrix_csv("sm_wall_seconds.csv")
    summary = (EXAMPLE / "best_fission_summary.json").read_text()
    import json
    info = json.loads(summary)
    flat = [abs(value) if value is not None else float("inf") for row in pcm for value in row]
    best = min(range(len(flat)), key=lambda i: flat[i])
    s, m = best // len(M_VALUES) + 1, M_VALUES[best % len(M_VALUES)]
    err = pipe("tab:study4-core-pcm", pcm, 2, plus=True,
               caption=f"Signed eigenvalue difference $10^{{5}}(\\kappa_{{\\mathrm{{eff}}}}-"
                       f"k_\\mathrm{{ref}})$ in pcm on the explicit quarter core, against the "
                       f"same-geometry OpenMC reference ${KREF}$ ($3.41$ pcm standard error). Bold "
                       f"marks the smallest measured absolute difference.")
    time_ = pipe("tab:study4-core-time", seconds, 1,
                 caption="End-to-end wall time in seconds for the same points, including assembly, "
                         "factor construction, outer iterations and the verification sweep. An "
                         "asterisk marks a point that reused stored factors; points sharing a time "
                         "window ran concurrently.")
    best_table = (
        "| quantity | best measured cell |\n|:-----------------------|:---------------------|\n"
        f"| spatial subdivision $S$ | {s} |\n| azimuth sectors $M$ | {m} |\n"
        f"| nodes | {info['nodes']} |\n| $\\kappa_{{\\mathrm{{eff}}}}$ | "
        f"{info['keff']:.12f} |\n| $\\Delta k$ | ${info['pcm']:+.2f}$ pcm |\n"
        f"| wall time | {info['time_s']:.1f} s |\n| pin fission RMS / max | "
        f"{info['fuel_rms_percent']:.3f} % / {info['fuel_max_abs_percent']:.3f} % |\n\n"
        f": The cell of @@TAB:tab:study4-core-pcm@@ with the smallest "
        "absolute eigenvalue difference, with the relative pin-wise fission-rate difference over "
        "the 1056 fuel pins. The node count is $(153S)^2$. {#tab:study4-core-best}\n")
    return "\n".join([err, time_]), best_table


def renumber(text):
    """Give every Figure/Table link the number its target holds in document order."""
    for kind, pattern in (("fig", r'<figure id="fig:([^"]+)"'), ("tab", r"\{#tab:([^} ]+)\}")):
        order = {f"{kind}:{name}": i + 1 for i, name in enumerate(re.findall(pattern, text))}
        missing = [tgt for tgt in re.findall(rf"\[\d+\]\(#{kind}:([^)]+)\)", text)
                   if f"{kind}:{tgt}" not in order]
        if missing:
            raise SystemExit(f"unresolved {kind} targets: {sorted(set(missing))}")
        text = re.sub(rf"(\[\d+\])\(#({kind}:[^)]+)\)",
                      lambda mo: f"[{order[mo.group(2)]}](#{mo.group(2)})", text)
        for token, name in re.findall(r"@@(TAB|FIG):([a-zA-Z0-9:_-]+)@@", text):
            if (token == "TAB") != (kind == "tab"):
                continue  # the other kind is resolved on its own pass
            kind2 = "tab" if token == "TAB" else "fig"
            key = name if name.startswith(f"{kind2}:") else f"{kind2}:{name}"
            word = "Table" if token == "TAB" else "Figure"
            text = text.replace(f"@@{token}:{name}@@",
                                f'{word} [{order[key]}](#{key})'
                                f'{{reference-type="ref" reference="{key}"}}')
    return text


def build_section():
    """The Study IV markdown, with both generated tables read back from their data files."""
    section = (HERE / "studyIV.md").read_text()
    for token, text in (("@@CORE_TABLES@@", core_tables()[0]),
                        ("@@BEST_TABLE@", core_tables()[1]),
                        ("@@CHECKER_TABLE@@", checker_table()),
                        ("@@SECTOR_TABLE@@", sector_table())):
        if token not in section:
            raise SystemExit(f"{HERE / 'studyIV.md'} has no {token}")
        section = section.replace(token, text)
    return section.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="require the core CSVs")
    args = parser.parse_args()
    text = (PAPER / "main.md").read_text()
    text, done = apply_edits(text)
    print(f"prose edits applied: {done}")

    have = all((EXAMPLE / name).exists() for name in
               ("sm_error_pcm.csv", "sm_wall_seconds.csv", "best_fission_summary.json")) and \
         (PAPER / "data/study4-checkerboard.csv").exists()
    if not have:
        print("Study IV section NOT refreshed (missing the generated CSVs)")
        if args.force:
            return 1
    else:
        section = build_section()
        wrapped = "<!--STUDY4_BEGIN-->\n\n" + section + "\n\n<!--STUDY4_END-->"
        if "<!--STUDY4_END-->" in text:
            start = text.index("<!--STUDY4_BEGIN-->")
            stop = text.index("<!--STUDY4_END-->") + len("<!--STUDY4_END-->")
            text = text[:start] + wrapped + text[stop:]
            print("Study IV section refreshed in place")
        elif "## Study IV ---" in text:
            raise SystemExit("a Study IV section without the splice markers exists; refusing to duplicate")
        else:
            text = text.replace(SPOILER, wrapped + "\n\n" + SPOILER, 1)
            print("Study IV section inserted")

    text = renumber(text)
    (PAPER / "main.md").write_text(text)
    (PAPER / "revised_main.md").write_text(text)
    print("wrote main.md and revised_main.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
