"""Build the slide-style PDF presentation for the course submission.

Run from the repo root:
    python scripts/build_presentation.py

Output:
    reports/presentation_group_06.pdf

Each slide is one matplotlib figure laid out in 16:9, written into a
multi-page PDF via PdfPages. We deliberately rebuild the deck from
code (instead of dragging in a heavy presentation lib or pandoc) so
the file regenerates from this script + the figures already in
reports/figures/.

Edit the GROUP_NUMBER and STUDENT_NAMES constants below before
recording.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
OUT_PDF = ROOT / "reports" / "presentation_group_06.pdf"

# ----- Submission metadata (edit before recording / submitting) -----
GROUP_NUMBER = "6"
COURSE = "Scientific Programming — FS 2026"
PROJECT_TITLE = "AI-Enhanced Analysis of\nZurich-Canton Rental Listings"
STUDENT_NAMES = [
    "Drin Muslija",
    "Issa Fawaz",
    "Valdrin Dalipi",
]
RESEARCH_QUESTION = (
    "Which factors — size, room count, location, and amenity features —\n"
    "significantly influence monthly rent in Zurich canton?"
)

# 16:9 slide canvas at a print-friendly size.
SLIDE_W, SLIDE_H = 13.33, 7.5
PRIMARY = "#2E5C8A"
ACCENT = "#D97706"
DANGER = "#DC2626"
NEUTRAL = "#6B7280"
INK = "#1F2937"
PAGE_BG = "#FFFFFF"
HEADER_BAR = "#F3F4F6"


def _blank_slide() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), facecolor=PAGE_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    return fig, ax


def _slide_header(ax: plt.Axes, slide_num: int, total: int, header: str) -> None:
    """Slim top header strip + slide-counter footer."""
    ax.add_patch(FancyBboxPatch((0, 0.93), 1, 0.07, boxstyle="square,pad=0",
                                 facecolor=HEADER_BAR, edgecolor="none",
                                 transform=ax.transAxes))
    ax.text(0.04, 0.965, header, transform=ax.transAxes,
            fontsize=13, color=NEUTRAL, va="center", ha="left",
            fontweight="semibold")
    ax.text(0.96, 0.965, f"Group {GROUP_NUMBER}  ·  slide {slide_num} / {total}",
            transform=ax.transAxes, fontsize=11, color=NEUTRAL,
            va="center", ha="right")


def _bullets(ax: plt.Axes, bullets: list[str], y_start: float = 0.70,
             y_gap: float = 0.085, indent: float = 0.07,
             fontsize: int = 17, color: str = INK,
             wrap_width: int | None = None) -> None:
    """Render a vertical bullet list at the given start y.

    Bullet text is hard-wrapped with textwrap.fill so it stays inside a
    readable column instead of running edge-to-edge across the slide
    (matplotlib's wrap=True wraps to figure width, not to a column).
    Default character widths are tuned for the slide's 13.33-inch width:
    ~78 chars at fontsize 17, ~90 chars at fontsize 15.
    """
    if wrap_width is None:
        wrap_width = 78 if fontsize >= 17 else 92
    for i, b in enumerate(bullets):
        y = y_start - i * y_gap
        ax.text(indent, y, "•", transform=ax.transAxes, fontsize=fontsize + 4,
                color=PRIMARY, va="top", ha="left", fontweight="bold")
        wrapped = textwrap.fill(b, width=wrap_width)
        ax.text(indent + 0.025, y, wrapped, transform=ax.transAxes,
                fontsize=fontsize, color=color, va="top", ha="left")


def _slide_title(ax: plt.Axes, title: str, subtitle: str | None = None,
                  y_title: float = 0.83) -> None:
    ax.text(0.07, y_title, title, transform=ax.transAxes,
            fontsize=30, color=INK, fontweight="bold",
            va="bottom", ha="left")
    if subtitle:
        ax.text(0.07, y_title - 0.06, subtitle, transform=ax.transAxes,
                fontsize=15, color=NEUTRAL, style="italic",
                va="bottom", ha="left")


# ---------- Slide builders ----------

def slide_title(pdf: PdfPages, n: int, total: int) -> None:
    fig, ax = _blank_slide()
    # Coloured banner.
    ax.add_patch(FancyBboxPatch((0, 0.55), 1, 0.45, boxstyle="square,pad=0",
                                 facecolor=PRIMARY, edgecolor="none",
                                 transform=ax.transAxes))
    ax.text(0.5, 0.85, PROJECT_TITLE, transform=ax.transAxes,
            fontsize=32, color="white", fontweight="bold",
            va="center", ha="center", linespacing=1.2)
    ax.text(0.5, 0.62, RESEARCH_QUESTION, transform=ax.transAxes,
            fontsize=15, color="#DBEAFE", style="italic",
            va="center", ha="center", linespacing=1.4)
    # Student block.
    ax.text(0.5, 0.42, "Group " + GROUP_NUMBER, transform=ax.transAxes,
            fontsize=22, color=PRIMARY, fontweight="bold",
            va="center", ha="center")
    for i, name in enumerate(STUDENT_NAMES):
        ax.text(0.5, 0.32 - i * 0.06, name, transform=ax.transAxes,
                fontsize=18, color=INK, va="center", ha="center")
    ax.text(0.5, 0.08, COURSE, transform=ax.transAxes,
            fontsize=14, color=NEUTRAL, va="center", ha="center")
    pdf.savefig(fig, facecolor=PAGE_BG)
    plt.close(fig)


def slide_intro(pdf: PdfPages, n: int, total: int) -> None:
    fig, ax = _blank_slide()
    _slide_header(ax, n, total, "1. Introduction")
    _slide_title(ax, "Background, problem, objective",
                 "Why we built a Swiss-rental analytics pipeline")
    _bullets(ax, [
        "Background — Zurich canton has one of the most expensive rental "
        "markets in Europe; rent is the largest line in most household budgets.",
        "Problem — no clean, structured, canton-wide dataset exists; "
        "listing portals expose only human-readable HTML and inconsistent prices.",
        "Objective — build a reproducible end-to-end pipeline that scrapes, "
        "cleans, validates, enriches (LLM) and statistically tests Zurich rentals.",
        "Research question — which factors (size, rooms, location, "
        "features) significantly influence monthly rent in Zurich canton?",
        "Approach — hypothesis testing rather than black-box prediction; "
        "all inputs are auditable through a Streamlit dashboard.",
    ])
    pdf.savefig(fig, facecolor=PAGE_BG)
    plt.close(fig)


def slide_methods_stack(pdf: PdfPages, n: int, total: int) -> None:
    fig, ax = _blank_slide()
    _slide_header(ax, n, total, "2. Materials & methods — data + tech stack")
    _slide_title(ax, "What we used", "Public Flatfox API + Python data stack")
    _bullets(ax, [
        "Data source — flatfox.ch public JSON API; only Swiss portal not "
        "behind Cloudflare. 200 ZH apartments retained after strict filters.",
        "Filters — canton = ZH, full apartments only, monthly rent unit, "
        "20 ≤ m² and CHF 500–15 000; excludes garages, shared flats, commercial.",
        "Python stack — requests + BeautifulSoup (scrape), pandas + scipy "
        "(analysis), OpenAI gpt-4o-mini (feature extraction), SQLite (persistence).",
        "Visualisation — matplotlib for the printed report figures, "
        "Plotly inside Streamlit for the interactive multi-tab dashboard.",
        "Reproducibility — pinned requirements.txt, .env-based config, "
        "deterministic content-keyed LLM cache (no duplicate API calls).",
    ])
    pdf.savefig(fig, facecolor=PAGE_BG)
    plt.close(fig)


def slide_methods_pipeline(pdf: PdfPages, n: int, total: int) -> None:
    fig, ax = _blank_slide()
    _slide_header(ax, n, total, "3. Materials & methods — pipeline")
    _slide_title(ax, "End-to-end pipeline",
                 "Eight stages, all idempotent, full chain in ~2 min on warm cache")
    stages = [
        ("collect_zurich", "Flatfox JSON API → raw ZH listings + provenance fields"),
        ("fetch_details", "Threaded scrape of each detail page (BeautifulSoup)"),
        ("cleaning", "Regex normalisation, typed fields, balcony / parking flags"),
        ("llm_helper", "OpenAI feature extraction (gpt-4o-mini, SQLite-cached)"),
        ("data_quality", "Validate, dedupe, price QC report, outlier handling"),
        ("database", "SQLite persistence + canned SQL queries (NTILE, RANK, AVG)"),
        ("statistics", "7 tests — Pearson, Spearman, OLS, Welch, MWU, Shapiro"),
        ("insights → UI", "Auto business findings → Streamlit dashboard + PDF figures"),
    ]
    y0 = 0.74
    row_h = 0.075
    for i, (mod, desc) in enumerate(stages):
        y = y0 - i * row_h
        ax.add_patch(FancyBboxPatch((0.07, y - 0.035), 0.20, 0.060,
                                     boxstyle="round,pad=0.005",
                                     facecolor=PRIMARY, edgecolor="none",
                                     transform=ax.transAxes))
        ax.text(0.17, y - 0.005, mod, transform=ax.transAxes, fontsize=12,
                color="white", fontweight="bold", va="center", ha="center")
        ax.text(0.30, y - 0.005, "→", transform=ax.transAxes, fontsize=18,
                color=NEUTRAL, va="center", ha="center")
        ax.text(0.34, y - 0.005, desc, transform=ax.transAxes, fontsize=13,
                color=INK, va="center", ha="left")
    pdf.savefig(fig, facecolor=PAGE_BG)
    plt.close(fig)


def slide_methods_qc(pdf: PdfPages, n: int, total: int) -> None:
    fig, ax = _blank_slide()
    _slide_header(ax, n, total, "4. Materials & methods — quality control")
    _slide_title(ax, "Defensible numbers, not silent failures",
                 "Validation, suspicious-price audit, console QC report")
    _bullets(ax, [
        "Plausibility bounds — CHF 500–15 000 / month, 15–400 m², "
        "1–8 rooms, CHF/m² ∈ [10, 200]; non-numeric prices rejected.",
        "Dedup — natural key (title, price, size, rooms); also filters "
        "Tukey k = 3 statistical outliers on price.",
        "Suspicious-price audit — collector keeps raw title text and "
        "structured rent_gross; validator flags listings whose gap ≥ CHF 100.",
        "Why it matters — Flatfox occasionally serves stale public_title; "
        "without this audit, charts silently use the wrong number.",
        "QC console report — scraped 200 → valid 197 → suspicious 0 → "
        "missing 0 → removed 3 (statistical outliers).",
    ], y_start=0.70, y_gap=0.10)
    pdf.savefig(fig, facecolor=PAGE_BG)
    plt.close(fig)


def slide_figure(pdf: PdfPages, n: int, total: int, header: str,
                 title: str, subtitle: str, fig_path: Path,
                 take_away: str) -> None:
    fig, ax = _blank_slide()
    _slide_header(ax, n, total, header)
    _slide_title(ax, title, subtitle, y_title=0.85)
    # Narrowed figure leaves more horizontal room for the take-away box,
    # which previously had lines overflowing past its right border.
    if fig_path.exists():
        img = imread(str(fig_path))
        fig_ax = fig.add_axes([0.06, 0.16, 0.55, 0.62])
        fig_ax.imshow(img)
        fig_ax.set_axis_off()
    else:
        ax.text(0.4, 0.5, f"[missing: {fig_path.name}]", transform=ax.transAxes,
                fontsize=14, color=DANGER, ha="center", va="center")
    # Take-away box on the right.
    box_x, box_y, box_w, box_h = 0.65, 0.18, 0.31, 0.60
    pad_x = 0.018
    ax.add_patch(FancyBboxPatch((box_x, box_y), box_w, box_h,
                                 boxstyle="round,pad=0.01",
                                 facecolor="#F3F4F6", edgecolor=NEUTRAL,
                                 linewidth=0.8, transform=ax.transAxes))
    ax.text(box_x + box_w / 2, box_y + box_h - 0.04, "Take-away",
            transform=ax.transAxes, fontsize=14, color=PRIMARY,
            fontweight="bold", ha="center", va="top")
    # Pre-wrap paragraph by paragraph: matplotlib's wrap=True wraps to the
    # *figure* width, not the box width, so manual textwrap.fill keeps the
    # lines inside the rounded box. ~36 chars/line fits the 0.31-wide box at
    # 12pt with comfortable margins.
    wrapped = "\n\n".join(
        textwrap.fill(para, width=36)
        for para in take_away.split("\n\n")
    )
    ax.text(box_x + pad_x, box_y + box_h - 0.10, wrapped,
            transform=ax.transAxes, fontsize=12, color=INK,
            ha="left", va="top", linespacing=1.45)
    pdf.savefig(fig, facecolor=PAGE_BG)
    plt.close(fig)


def slide_dashboard_tour(pdf: PdfPages, n: int, total: int) -> None:
    """Visual title-card for the live Streamlit demo segment.

    The presenter shows this slide for ~10 seconds, then switches the
    screen-share to localhost:8501 and narrates each stop in order.
    The timestamps on the right match Speaker 2's demo script."""
    fig, ax = _blank_slide()
    _slide_header(ax, n, total, "Live demo — Streamlit dashboard")
    _slide_title(ax, "Live dashboard tour (~3 min)",
                 "What the static PDF figures cannot show")

    # Left column — context.
    ax.text(0.07, 0.66, "Why a live tour?", transform=ax.transAxes,
            fontsize=16, color=PRIMARY, fontweight="bold", va="top", ha="left")
    why_text = textwrap.fill(
        "The PDF carries the headline numbers, but the dashboard is "
        "interactive: every chart re-filters in real time, the geographic "
        "map is zoom-able, and each suspicious-price listing links straight "
        "back to Flatfox for verification.",
        width=44,
    )
    ax.text(0.07, 0.60, why_text, transform=ax.transAxes,
            fontsize=13, color=INK, va="top", ha="left", linespacing=1.45)

    # Right column — six demo stops with timestamps.
    stops = [
        ("0:00", "Overview", "KPIs + LLM status badge"),
        ("0:20", "Sidebar filters", "Every chart reacts live"),
        ("0:50", "Geography", "Interactive ZH bubble map"),
        ("1:20", "SQL Explorer", "Window-function queries (NTILE, RANK)"),
        ("1:50", "Insights", "Auto findings, confidence-tagged"),
        ("2:20", "Data Quality", "Drop audit + suspicious-price links"),
    ]
    ax.text(0.52, 0.66, "Six stops, in order",
            transform=ax.transAxes, fontsize=16, color=PRIMARY,
            fontweight="bold", va="top", ha="left")
    y0 = 0.59
    row_h = 0.065
    for i, (ts, tab, desc) in enumerate(stops):
        y = y0 - i * row_h
        ax.add_patch(FancyBboxPatch((0.52, y - 0.025), 0.07, 0.045,
                                     boxstyle="round,pad=0.005",
                                     facecolor=PRIMARY, edgecolor="none",
                                     transform=ax.transAxes))
        ax.text(0.555, y - 0.003, ts, transform=ax.transAxes, fontsize=11,
                color="white", fontweight="bold", va="center", ha="center")
        ax.text(0.61, y - 0.003, tab, transform=ax.transAxes, fontsize=13,
                color=INK, fontweight="bold", va="center", ha="left")
        ax.text(0.73, y - 0.003, "— " + desc, transform=ax.transAxes,
                fontsize=12, color=NEUTRAL, va="center", ha="left")

    # Footer cue for the presenter.
    ax.text(0.5, 0.10,
            "Presenter cue: switch screen-share to localhost:8501 now.",
            transform=ax.transAxes, fontsize=12, color=ACCENT, style="italic",
            fontweight="bold", va="center", ha="center")
    pdf.savefig(fig, facecolor=PAGE_BG)
    plt.close(fig)


def slide_discussion(pdf: PdfPages, n: int, total: int) -> None:
    fig, ax = _blank_slide()
    _slide_header(ax, n, total, "11. Discussion")
    _slide_title(ax, "What the data tells us — and what it doesn't",
                 "Interpretation + honest limitations")
    _bullets(ax, [
        "Findings — living space is the single strongest linear predictor "
        "of rent; ~CHF 12/m² per month. Rooms is a noisy proxy by comparison.",
        "Amenities — furnished is the only feature with a clear signal "
        "(+CHF 750 mean, ~25%); balcony and parking effects are within noise.",
        "Small-flat premium — size ↔ CHF/m² correlation is −0.58; "
        "micro-units consistently extract a price-per-area markup.",
        "Limitation — Flatfox-only sample of 200 listings; Homegate, "
        "ImmoScout and newhome sit behind Cloudflare.",
        "Limitation — snapshot in time; rents move with season and "
        "mortgage rates. A longitudinal study would be a natural follow-up.",
        "Limitation — LLM feature extraction is fast/cheap but imperfect; "
        "validated against a regex fallback on unambiguous cases.",
    ], y_start=0.70, y_gap=0.087, fontsize=15)
    pdf.savefig(fig, facecolor=PAGE_BG)
    plt.close(fig)


def slide_conclusions(pdf: PdfPages, n: int, total: int) -> None:
    fig, ax = _blank_slide()
    _slide_header(ax, n, total, "12. Conclusions")
    _slide_title(ax, "What we delivered", "Reproducible pipeline + defensible answers")
    _bullets(ax, [
        "Reproducible end-to-end pipeline turns Flatfox listings into a "
        "validated, statistically-tested dataset with full audit trail.",
        "Research question answered — living space dominates rent; "
        "room count is a noisy proxy; only furnishing carries a clear premium.",
        "Engineering contribution — price-QC report cross-checks scraped "
        "prices against the listing's own title text and flags mismatches.",
        "Outputs — Streamlit dashboard (interactive), six light-theme PDF "
        "figures (printed report), SQLite DB (downstream queries).",
        "Future work — multi-canton expansion, time-series snapshots, "
        "LLM short-circuit when regex is already confident, external centroid CSV.",
    ], y_start=0.70, y_gap=0.10)
    pdf.savefig(fig, facecolor=PAGE_BG)
    plt.close(fig)


def slide_thank_you(pdf: PdfPages, n: int, total: int) -> None:
    fig, ax = _blank_slide()
    ax.add_patch(FancyBboxPatch((0, 0.55), 1, 0.45, boxstyle="square,pad=0",
                                 facecolor=PRIMARY, edgecolor="none",
                                 transform=ax.transAxes))
    ax.text(0.5, 0.78, "Thank you", transform=ax.transAxes,
            fontsize=44, color="white", fontweight="bold",
            va="center", ha="center")
    ax.text(0.5, 0.66, "Questions?", transform=ax.transAxes,
            fontsize=22, color="#DBEAFE", style="italic",
            va="center", ha="center")
    ax.text(0.5, 0.40, "Group " + GROUP_NUMBER + "  ·  " + COURSE,
            transform=ax.transAxes, fontsize=16, color=INK,
            va="center", ha="center")
    ax.text(0.5, 0.30, "  ·  ".join(STUDENT_NAMES),
            transform=ax.transAxes, fontsize=14, color=NEUTRAL,
            va="center", ha="center")
    ax.text(0.5, 0.16,
            "Source code & figures: "
            "github.com/Drin06577/scientific_programming_LN",
            transform=ax.transAxes, fontsize=11, color=NEUTRAL,
            style="italic", va="center", ha="center")
    pdf.savefig(fig, facecolor=PAGE_BG)
    plt.close(fig)


# ---------- Orchestration ----------

def build_pdf() -> Path:
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    total = 16  # 15 narrated slides + 1 appendix; footer reads "N / 16"
    # Slide 6 is the headline result; slide 7 is the dashboard demo card
    # (no figure, see slide_dashboard_tour); slides 8–12 are the remaining
    # five figures. The presenter shows slide 7 for ~10 s then switches to
    # screen-share for the 3-min demo described in presentation_script.md.
    figure_slides = [
        (6, "5. Results — size vs rent",
         "Larger flats usually have higher monthly rent",
         "Pearson r = 0.39  ·  slope ≈ CHF 12 / m² / month  ·  n = 197",
         FIG_DIR / "01_size_vs_price.png",
         ("Size is the single strongest linear "
          "predictor of monthly rent.\n\n"
          "But spread is large: same-size "
          "flats can differ by CHF 2 000+ "
          "depending on location and amenities.")),
        (8, "7. Results — rent by rooms",
         "Monthly rent by number of rooms",
         "Boxplot of price per room category (≥ 5 listings each)",
         FIG_DIR / "02_rent_by_rooms.png",
         ("Room count is a much weaker price "
          "signal than m² (Spearman ρ ≈ 0.15).\n\n"
          "Same-room-count flats vary widely "
          "in m², which dilutes the signal.")),
        (9, "8. Results — rent ranges",
         "How many listings fall into each rent range?",
         "Fixed bins: 0–2 000, 2 000–3 000, … 6 000+",
         FIG_DIR / "03_rent_ranges.png",
         ("Modal bucket is CHF 2 000–3 000 "
          "(80 listings).\n\n"
          "~70% of Zurich-canton listings rent "
          "between CHF 2 000 and 4 000 / month. "
          "Only 7 cross the CHF 6 000 line.")),
        (10, "9. Results — feature impact",
         "Do specific features affect monthly rent?",
         "Mean + median rent, with sample sizes",
         FIG_DIR / "04_feature_impact.png",
         ("Furnished is the only feature with "
          "a large effect (+CHF 750, ~25%).\n\n"
          "Balcony +CHF 150 not significant. "
          "Parking shows a small negative gap, "
          "driven by location confounding.")),
        (11, "10. Results — correlation matrix",
         "Correlation between numeric housing features",
         "Pearson r — values close to ±1 mean a strong linear link",
         FIG_DIR / "05_correlation_matrix.png",
         ("Strongest correlations:\n"
          "• size ↔ price:  +0.39\n"
          "• size ↔ rooms: +0.79\n"
          "• size ↔ CHF/m²: −0.58\n\n"
          "Small flats charge a per-m² premium.")),
        (12, "11. Results — CHF/m² distribution",
         "Distribution of monthly rent per square metre",
         "Median = CHF 33.2/m²  ·  Mean = CHF 39.1/m²  ·  right-skewed",
         FIG_DIR / "06_chf_per_m2_distribution.png",
         ("Median (33.2) < Mean (39.1) → "
          "right-skewed distribution.\n\n"
          "A small tail of luxury listings "
          "pulls the mean upward; the typical "
          "listing sits around CHF 33/m².")),
    ]

    with PdfPages(OUT_PDF) as pdf:
        slide_title(pdf, 1, total)
        slide_intro(pdf, 2, total)
        slide_methods_stack(pdf, 3, total)
        slide_methods_pipeline(pdf, 4, total)
        slide_methods_qc(pdf, 5, total)
        # Slide 6 = size vs price (first figure_slides entry).
        # Slide 7 = the dashboard demo card, slotted between size-vs-price
        # and the remaining results so the live tour reads as part of
        # "Results" in the deck flow.
        for entry in figure_slides[:1]:
            slide_figure(pdf, *entry[:1], total, *entry[1:])
        slide_dashboard_tour(pdf, 7, total)
        for entry in figure_slides[1:]:
            slide_figure(pdf, *entry[:1], total, *entry[1:])
        slide_discussion(pdf, 13, total)
        slide_conclusions(pdf, 14, total)
        slide_thank_you(pdf, 15, total)
        # Slide 16 — appendix (silent reference card, not narrated).
        fig, ax = _blank_slide()
        _slide_header(ax, 16, total, "Appendix — repository pointers")
        _slide_title(ax, "Where everything lives",
                     "For graders who want to inspect the code")
        _bullets(ax, [
            "GitHub repo — github.com/Drin06577/scientific_programming_LN",
            "Pipeline entry-point — `python -m src.pipeline`  (prints QC report)",
            "Dashboard       — `streamlit run app/streamlit_app.py`",
            "Report figures  — `python -m src.report_figures`  → reports/figures/",
            "Validation rules — src/data_quality.py  (suspicious-price audit)",
            "Project context — CLAUDE.md, HOW_TO_RUN.md, README.md",
        ], y_start=0.70, y_gap=0.10, fontsize=15)
        pdf.savefig(fig, facecolor=PAGE_BG)
        plt.close(fig)

    return OUT_PDF


if __name__ == "__main__":
    out = build_pdf()
    print(f"[presentation] wrote {out}  ({out.stat().st_size / 1024:.1f} KB)")
