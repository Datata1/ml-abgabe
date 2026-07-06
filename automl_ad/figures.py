"""Slide-fertige Grafiken für den Vortrag (16:9, heller Hintergrund, selbst-erklärend).

Jede ``fig_*``-Funktion erzeugt **eine** Grafik, die als Bild direkt in die Folien passt: großer
Titel, eine Kernaussage-Zeile und (wo nötig) ein ⚠️-Caveat sind ins Bild eingebrannt. Mit ``save_as``
wird zusätzlich ein hochauflösendes PNG nach ``reports/figures/`` geschrieben (siehe
``scripts/make_figures.py`` / ``make figures``).

Die Funktionen **rechnen nicht** — sie bekommen fertige Ergebnisse (Scores/AUCs/Engine-Output) und
stellen sie dar. Rechnen: ``selection`` / ``pyod_engine``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.stats import rankdata

from . import config

# --- Stil -----------------------------------------------------------------------------
FIGSIZE = (10.0, 5.6)  # 16:9
DETECTOR_COLORS = {"knn": "#4C72B0", "iforest": "#55A868", "hdbscan": "#C44E52", "pca": "#8172B3"}
HIGHLIGHT = "#DD8452"   # gewählte/betonte Balken
MUTED = "#9AA0A6"       # neutrale Balken
OK = "#2CA02C"
WARN = "#C0392B"
_FIGDIR = config.REPORTS_DIR / "figures"


def use_slide_style() -> None:
    """Setzt matplotlib-Defaults für gut lesbare Folien-Grafiken (einmal aufrufen)."""
    plt.rcParams.update({
        "figure.figsize": FIGSIZE,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 200,
        "figure.dpi": 120,
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": False,
    })


def _color(name: str) -> str:
    return DETECTOR_COLORS.get(name, MUTED)


def _norm01(a) -> np.ndarray:
    """Min-Max-Normierung auf [0, 1] (Anomalie-Scores sind relativ; fixt riesige Roh-Magnituden)."""
    a = np.asarray(a, dtype=float)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def finish(fig, *, title: str, takeaway: str, caveat: str | None = None, save_as: str | None = None):
    """Einheitliche Beschriftung (Titel + Kernaussage + optional Caveat) und optionales Speichern."""
    fig.suptitle(title, y=0.985, fontsize=18, fontweight="bold")
    fig.text(0.5, 0.905, takeaway, ha="center", va="top", fontsize=13.5, color="#333333", wrap=True)
    if caveat:
        fig.text(0.5, 0.012, caveat, ha="center", va="bottom", fontsize=11, color=WARN,
                 style="italic", wrap=True)
    fig.tight_layout(rect=[0.02, 0.055 if caveat else 0.0, 0.98, 0.85])
    if save_as:
        _FIGDIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(_FIGDIR / save_as)
    return fig


CAVEAT_AUC = "⚠ ROC/AUC nur zur Illustration — im echten unüberwachten Betrieb NICHT verfügbar."


def _one_fault_run(split, fault: int):
    """Positionszeilen (nach sample sortiert) + sample-Array des ersten Laufs von ``fault``."""
    meta = split.meta_test
    sub = meta[meta["faultNumber"] == fault]
    if sub.empty:
        raise ValueError(f"Kein Lauf mit faultNumber={fault} gefunden.")
    run = sub["simulationRun"].iloc[0]
    rows = (
        sub[sub["simulationRun"] == run].sort_values("sample").index.to_numpy()
    )
    return rows, meta["sample"].to_numpy()[rows]


# --- 0) Modell-Steckbriefe (eine Karte je Baseline-Detektor) ----------------------------
# Inhalte bewusst knapp (Vorstellungs-Folie, nicht Deep-Dive). Links: echte Score-Landschaft
# des Detektors auf 2D-Spielzeugdaten (derselbe Code wie in der Pipeline), rechts: Stichpunkte.
_MODEL_CARDS: dict[str, dict] = {
    "knn": dict(
        display="KNN — k-nächste Nachbarn",
        idea="Score = Abstand zum k-nächsten Nachbarn: wer weit weg von allen liegt, ist anomal.",
        bullets=[
            ("⚙", "Mechanik: kein Modell im engeren Sinn — nur Distanzen\nzu den Trainingspunkten (Default k=5)."),
            ("✓", "Stärke: einfach, lokal sensitiv — Top-Performer\nin ADBench (#4)."),
            ("✗", "Schwäche: Distanzsuche kostet bei vielen Punkten;\nhohe Dimensionen verwässern Distanzen."),
        ],
    ),
    "pca": dict(
        display="PCA — Rekonstruktionsfehler",
        idea="Lernt die normale „Hauptebene“; Score = wie schlecht sich ein Punkt aus ihr rekonstruieren lässt.",
        bullets=[
            ("⚙", "Mechanik: Hauptkomponenten der Gutdaten;\nAbstand zur Ebene = Anomalie-Score."),
            ("✓", "Stärke: nutzt lineare Korrelationen zwischen\nSensoren (gekoppelte Anlagen!); sehr schnell."),
            ("✗", "Schwäche: sieht nur lineare Struktur —\nnichtlineare Muster bleiben unsichtbar."),
        ],
    ),
    "hdbscan": dict(
        display="HDBSCAN — Dichte-Clustering",
        idea="Cluster = dichte Regionen; wer in dünn besiedelten Regionen liegt, ist Ausreißer (GLOSH-Score).",
        bullets=[
            ("⚙", "Mechanik: hierarchisches Dichte-Clustering;\nOutlier-Score aus der Cluster-Hierarchie."),
            ("✓", "Stärke: findet Cluster beliebiger Form;\nbraucht keine Cluster-Anzahl vorab."),
            ("✗", "Schwäche: empfindlich auf min_cluster_size;\nneue Punkte werden nur approximativ gescort."),
        ],
    ),
    "iforest": dict(
        display="Isolation Forest",
        idea="Zufällige Schnitte isolieren Anomalien schnell: kurzer Pfad im Baum = anomal.",
        bullets=[
            ("⚙", "Mechanik: Ensemble zufälliger Bäume;\nScore = mittlere Pfadlänge bis zur Isolation."),
            ("✓", "Stärke: robuster Allrounder, skaliert linear —\nADBench #3."),
            ("✗", "Schwäche: achsenparallele Schnitte übersehen\nkorrelierte Features / lokale Anomalien."),
        ],
    ),
}


def _toy_data(seed: int = 7) -> np.ndarray:
    """2D-Spielzeugdaten für die Steckbrief-Karten: korrelierter Blob + dichter Klumpen.

    Die Mischung zeigt die Charakteristika aller vier Detektoren: lineare Kopplung (PCA),
    unterschiedliche Dichten (HDBSCAN/KNN) und klare Isolationsräume (IForest).
    """
    rng = np.random.default_rng(seed)
    blob = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.85], [0.85, 1.0]], 260)
    dense = rng.normal([2.6, -2.2], 0.28, size=(90, 2))
    return np.vstack([blob, dense])


def fig_model_card(name: str, *, save_as: str | None = None):
    """Steckbrief-Karte eines Baseline-Detektors: Score-Landschaft (links) + Stichpunkte (rechts)."""
    from .detectors import make_detector

    card = _MODEL_CARDS[name]
    X = _toy_data()
    det = make_detector(name).fit(X)

    # Score-Landschaft auf einem Gitter (echter decision_function-Aufruf, kein Schema).
    pad = 1.6
    gx = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 140)
    gy = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 140)
    GX, GY = np.meshgrid(gx, gy)
    Z = _norm01(det.decision_function(np.column_stack([GX.ravel(), GY.ravel()]))).reshape(GX.shape)

    fig, (axl, axr) = plt.subplots(1, 2, figsize=FIGSIZE, width_ratios=[1.0, 1.05])
    cf = axl.contourf(GX, GY, Z, levels=14, cmap="RdYlBu_r", alpha=0.85)
    axl.scatter(X[:, 0], X[:, 1], s=9, color="#222222", alpha=0.55, linewidths=0)
    axl.set(xticks=[], yticks=[])
    axl.set_title("Score-Landschaft auf Beispieldaten", fontsize=12, fontweight="normal", pad=6)
    cb = fig.colorbar(cf, ax=axl, fraction=0.046, pad=0.03)
    cb.set_label("Anomalie-Score (normiert)", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    axr.axis("off")
    colors = {"⚙": "#4C72B0", "✓": OK, "✗": WARN}
    for i, (sym, text) in enumerate(card["bullets"]):
        y = 0.82 - i * 0.30
        axr.text(0.02, y, sym, transform=axr.transAxes, fontsize=16, color=colors[sym],
                 fontweight="bold", va="top")
        axr.text(0.12, y, text, transform=axr.transAxes, fontsize=12.5, color="#333333",
                 va="top", linespacing=1.35)
    return finish(fig, title=card["display"], takeaway=card["idea"], save_as=save_as)


# --- 1) Detektor-Streuung -------------------------------------------------------------
def fig_detector_spread(iid_auc: dict[str, float], *, save_as: str | None = None):
    names = sorted(iid_auc, key=iid_auc.get)
    vals = [iid_auc[n] for n in names]
    fig, ax = plt.subplots()
    bars = ax.bar(names, vals, color=[_color(n) for n in names], width=0.62)
    ax.axhspan(min(vals), max(vals), color=HIGHLIGHT, alpha=0.10, zorder=0)
    ax.set(ylim=(0, 1.05), ylabel="ROC-AUC (illustrativ)")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.text(len(names) - 0.5, 0.51, "Zufall (0.5)", ha="right", va="bottom", fontsize=11, color="gray")
    for b, v in zip(bars, vals, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=13, fontweight="bold")
    xmid = (len(names) - 1) / 2  # in die Lücke zwischen den mittleren Balken (kein Label-Overlap)
    ax.annotate("", xy=(xmid, min(vals)), xytext=(xmid, max(vals)),
                arrowprops=dict(arrowstyle="<->", color=HIGHLIGHT, lw=2.2))
    ax.text(xmid + 0.12, (min(vals) + max(vals)) / 2, f"Streuung\nΔ={max(vals) - min(vals):.3f}",
            color=HIGHLIGHT, fontsize=12.5, fontweight="bold", va="center", ha="left")
    return finish(fig, title="Ohne Labels: welchen Detektor nehmen?",
                  takeaway="Die 4 PyOD-Detektoren streuen stark — blind einen zu wählen ist Glückssache.",
                  caveat=CAVEAT_AUC, save_as=save_as)


# --- 2) Score-Zeitreihe eines Fehlerlaufs ---------------------------------------------
def fig_score_timeline(split, scores: np.ndarray, *, fault: int, onset: int, detector: str = "pca",
                       save_as: str | None = None):
    rows, sample = _one_fault_run(split, fault)
    # Konsistent über den GANZEN Test normieren, damit die Schwelle aus dem Normalbetrieb
    # (Gutdaten-Läufe) auf derselben Skala liegt wie der gezeigte Fehlerlauf. Die Schwelle wird
    # — wie im echten Betrieb — nur aus label-freien Gutdaten abgeleitet, nicht aus dem Fehlerlauf.
    norm = _norm01(np.asarray(scores))
    score = norm[rows]
    good = split.meta_test["faultNumber"].to_numpy() == 0
    thr = float(np.quantile(norm[good], 0.99))
    fig, ax = plt.subplots()
    ax.axvspan(onset, sample.max(), color=WARN, alpha=0.08, zorder=0)
    ax.plot(sample, score, color=_color(detector), lw=1.7, label=f"{detector}-Score")
    ax.axhline(thr, color="gray", ls="--", lw=1.2, label="Schwelle (99%-Quantil Normalbetrieb)")
    ax.axvline(onset, color=WARN, ls=":", lw=1.6, label=f"Fehler-Onset (>{onset})")
    ax.set(xlabel="Zeit (sample)", ylabel="Anomalie-Score (normiert)")
    # Label unten in der schattierten Region (dort ist die Kurve nie; blended transform)
    ax.text((onset + sample.max()) / 2, 0.04, "Fehler aktiv (Wahrheit)", ha="center", va="bottom",
            fontsize=11, color=WARN, transform=ax.get_xaxis_transform())
    ax.legend(loc="upper right", fontsize=11.5)
    return finish(fig, title="Anlage läuft → Störung → Alarm",
                  takeaway="Nach dem Fehler-Onset bleibt der Score über der Normalbetriebs-Schwelle — das Prozess-Monitoring-Bild.",
                  save_as=save_as)


# --- 3) Konsens · Modus A -------------------------------------------------------------
def fig_consensus(tw: dict[str, np.ndarray], centrality: dict[str, float], best: str,
                  agreement_val: float, *, save_as: str | None = None):
    names = list(tw)
    ranks = np.column_stack([rankdata(tw[n]) for n in names])
    consensus = ranks.mean(axis=1)
    cols = [*names, "KONSENS"]
    corr = np.corrcoef(np.column_stack([ranks, consensus]), rowvar=False)

    fig, (axh, axb) = plt.subplots(1, 2, figsize=(10.6, 5.2), width_ratios=[1.15, 1.0])
    im = axh.imshow(corr, vmin=0, vmax=1, cmap="viridis")
    axh.set(xticks=range(len(cols)), yticks=range(len(cols)))
    axh.set_xticklabels(cols, rotation=35, ha="right")
    axh.set_yticklabels(cols)
    axh.set_title("Rang-Korrelation der Detektoren", fontsize=13)
    for i in range(len(cols)):
        for j in range(len(cols)):
            axh.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                     color="white" if corr[i, j] < 0.6 else "black", fontsize=9.5)
    fig.colorbar(im, ax=axh, fraction=0.046, pad=0.04)

    cvals = [centrality[n] for n in names]
    axb.bar(names, cvals, color=[HIGHLIGHT if n == best else MUTED for n in names], width=0.6)
    axb.set(ylim=(0, 1.05), ylabel="Centrality (Korr. zum Konsens)")
    axb.set_title(f"→ Wahl: {best}", fontsize=13)
    for i, v in enumerate(cvals):
        axb.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=12, fontweight="bold")
    return finish(fig, title="Konsens · Modus A: das zentralste Modell",
                  takeaway=f"Wähle den Detektor, der dem Rang-Konsens am nächsten ist. Schwarm-Einigkeit: agreement = {agreement_val:.2f}.",
                  save_as=save_as)


# --- 4) Konsens · Modus B -------------------------------------------------------------
def fig_ensemble(modeB_auc: dict[str, float], best_single_auc: float, *, save_as: str | None = None):
    labels = list(modeB_auc)
    vals = [modeB_auc[m] for m in labels]
    fig, ax = plt.subplots()
    bars = ax.bar(labels, vals, color=["#4C72B0", "#55A868", "#8172B3"][: len(labels)], width=0.55)
    ax.axhline(best_single_auc, color=HIGHLIGHT, ls="--", lw=1.6,
               label=f"bester Einzeldetektor = {best_single_auc:.3f}")
    ax.set(ylim=(0, 1.08), ylabel="ROC-AUC (illustrativ)", xlabel="Ensemble-Kombination")
    ax.legend(loc="lower right")
    for b, v in zip(bars, vals, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center",
                fontsize=13, fontweight="bold")
    return finish(fig, title="Konsens · Modus B: das Ensemble ist die Vorhersage",
                  takeaway="Kein Detektor gewinnt — der kombinierte Score entscheidet, robust und unabhängig von der Einzelwahl.",
                  caveat=CAVEAT_AUC, save_as=save_as)


# --- 5) Differenzierung pro Fehlerart --------------------------------------------------
def fig_per_fault(breakdown: dict[int, dict[str, float]], *, hard_faults: list[int] | None = None,
                  strategy_label: str = "Konsens B (avg)", save_as: str | None = None):
    """Pro Fehlertyp: ROC-AUC der Konsens-Vorhersage und label-freies agreement.

    Zwei **getrennte, an der x-Achse ausgerichtete** Panels (eine Größe pro Achse): AUC und
    agreement sind nicht vergleichbar, nur ihr Verlauf über die Fehlertypen soll kontrastiert
    werden — oben bricht die Erkennung bei den schweren Fehlern ein, unten bleibt die
    Schwarm-Einigkeit unauffällig im Mittelfeld (warnt also nicht).
    """
    hard = set(hard_faults or config.HARD_FAULTS)
    faults = sorted(breakdown, key=lambda f: breakdown[f]["roc_auc"], reverse=True)
    auc_vals = [breakdown[f]["roc_auc"] for f in faults]
    agr_vals = [breakdown[f]["agreement"] for f in faults]
    x = np.arange(len(faults))
    colors = [WARN if f in hard else "#4C72B0" for f in faults]

    fig, (axa, axg) = plt.subplots(2, 1, sharex=True, figsize=(11.0, 6.4), height_ratios=[1.9, 1.0])
    bars = axa.bar(x, auc_vals, width=0.66, color=colors)
    axa.axhline(0.5, color="gray", ls=":", lw=1.2)
    axa.text(len(faults) - 0.4, 0.51, "Zufall (0.5)", ha="right", va="bottom", fontsize=11,
             color="gray", bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    axa.set(ylim=(0, 1.1), ylabel="ROC-AUC")
    axa.set_title(f"Erkennungsleistung: {strategy_label}, illustrativ — bricht bei den schweren Fehlern ein",
                  fontsize=12, fontweight="normal", loc="left", pad=6)
    for b, f, v in zip(bars, faults, auc_vals, strict=True):  # nur die schweren beschriften
        if f in hard:
            axa.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center",
                     fontsize=11.5, fontweight="bold", color=WARN)

    axg.bar(x, agr_vals, width=0.66, color=colors)
    axg.set(ylim=(0, 1.1), ylabel="agreement", xlabel="Fehlertyp (sortiert nach ROC-AUC)")
    axg.set_title("Vertrauenssignal agreement (label-frei) — unterscheidet die unsichtbaren Fehler nicht",
                  fontsize=12, fontweight="normal", loc="left", pad=6)
    axg.set_xticks(x)
    axg.set_xticklabels([f"F{f}" for f in faults], fontsize=11)
    for xt, f in zip(axg.get_xticklabels(), faults, strict=True):
        if f in hard:
            xt.set_color(WARN)
            xt.set_fontweight("bold")
    return finish(fig, title="Differenzierung: nicht jeder Fehler ist detektierbar",
                  takeaway="Hohe Einigkeit schützt nicht vor Blindheit: das label-freie Signal "
                           "zeigt die schweren Fehler nicht an (Einigkeit ≠ Richtigkeit).",
                  caveat=CAVEAT_AUC, save_as=save_as)


# --- 6) ADEngine-Benchmark ------------------------------------------------------------
def fig_benchmark(top5: list[tuple[str, int]], *, save_as: str | None = None):
    top5 = sorted(top5, key=lambda t: t[1])  # compare_detectors liefert unsortiert → nach Rang
    names = [n for n, _ in top5][::-1]
    ranks = [r for _, r in top5][::-1]
    maxr = max(ranks)
    lengths = [maxr + 1 - r for r in ranks]  # kleiner Rang = längerer Balken (immer positiv)
    fig, ax = plt.subplots()
    ax.barh(names, lengths, color="#4C72B0")
    for i, (length, r) in enumerate(zip(lengths, ranks, strict=True)):
        ax.text(length + 0.1, i, f"#{r}", va="center", fontsize=13, fontweight="bold", color="#4C72B0")
    ax.set(xlim=(0, maxr + 1.2), xticks=[], xlabel="besser →")
    ax.set_title("ADBench (Han et al., NeurIPS 2022): 57 Datensätze × 30 Algorithmen",
                 fontsize=12, fontweight="normal")
    return finish(fig, title="ADEngine-Benchmark: vorberechnete Bestenliste",
                  takeaway="Kein Training auf TEP! Die Engine schlägt in Ranglisten aus vielen fremden Datensätzen nach.",
                  caveat="Plätze #1–#2 fehlen bewusst: sie brauchen Labels (semi-/supervised) — "
                         "label-frei beginnt die Liste bei #3.",
                  save_as=save_as)


# --- 7) ADEngine-Report-Card ----------------------------------------------------------
def fig_engine_report(engine_out: dict, *, save_as: str | None = None):
    val = engine_out.get("validation") or {}
    verdict = str(engine_out.get("quality_verdict", "?"))
    overall = float(engine_out.get("quality_overall", float("nan")))
    verdict_color = {"high": OK, "medium": "#E1A100", "low": WARN}.get(verdict, MUTED)

    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    ax.axis("off")
    # Kachel-Reihenfolge = Pipeline-Reihenfolge: wählen → Konsens bilden → selbst bewerten → spicken.
    cards = [
        ("① Gewählte Detektoren\n(benchmark-gestützt)", ", ".join(map(str, engine_out.get("detectors", []))), "#4C72B0"),
        ("② Agreement (Einigkeit\ndes Konsens)", f"{engine_out.get('agreement', float('nan')):.2f}", "#55A868"),
        ("③ Selbst-Bewertung:\nlabel-freies Verdikt", f"{verdict}  ({overall:.2f})", verdict_color),
        ("④ Spick-Kontrolle:\nConsensus-ROC-AUC ⚠", f"{val.get('consensus_roc_auc', float('nan')):.3f}", MUTED),
    ]
    for i, (label, value, col) in enumerate(cards):
        x = 0.03 + (i % 2) * 0.5
        y = 0.56 - (i // 2) * 0.44
        ax.add_patch(Rectangle((x, y), 0.44, 0.34, transform=ax.transAxes,
                               facecolor=col, alpha=0.12, edgecolor=col, lw=2))
        ax.text(x + 0.02, y + 0.22, label, transform=ax.transAxes, fontsize=13, color="#333")
        ax.text(x + 0.02, y + 0.06, value, transform=ax.transAxes, fontsize=17, fontweight="bold", color=col)
    return finish(fig, title="PyOD ADEngine: was liefert EIN Aufruf investigate(X)?",
                  takeaway="① benchmark-gestützt wählen → ② Multi-Detektor-Consensus → ③ Selbst-Bewertung (ohne Labels) → ④ nur wir spicken.",
                  caveat="⚠ Consensus-ROC-AUC nur zur Illustration; das Verdikt (③) ist das label-freie Signal.",
                  save_as=save_as)


# --- 8) LLM-Routing -------------------------------------------------------------------
def fig_llm_routing(llm_cache: dict | None, *, save_as: str | None = None):
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    ax.axis("off")
    ax.set(xlim=(0, 10), ylim=(0, 10))

    def box(x, y, w, h, text, fc):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, alpha=0.15, edgecolor=fc, lw=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11.5)

    box(0.3, 7.3, 3.0, 1.6, "PyOD baut Prompt\n(KB: 60+ Detektoren)", "#4C72B0")
    box(6.7, 7.3, 3.0, 1.6, "Transport →\nClaude / Ollama", "#55A868")
    box(6.7, 4.6, 3.0, 1.6, "PyOD parst +\nKB-Validierung", "#4C72B0")
    box(0.3, 4.6, 3.0, 1.6, "run_detection\n(gewählter Detektor)", "#8172B3")
    for a, b in [((3.3, 8.1), (6.7, 8.1)), ((8.2, 7.3), (8.2, 6.2)),
                 ((6.7, 5.4), (3.3, 5.4))]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="->", mutation_scale=18, color="#555", lw=1.6))
    ax.text(5.0, 8.35, "Prompt", ha="center", fontsize=10, color="#555")
    ax.text(8.45, 6.7, "JSON", ha="center", fontsize=10, color="#555")
    ax.text(5.0, 5.65, "Plan (KB-geprüft)", ha="center", fontsize=10, color="#555")

    if llm_cache and llm_cache.get("source") == "llm":
        res = f"LLM wählte: {llm_cache.get('detector')}  (ROC-AUC {llm_cache.get('roc_auc', float('nan')):.3f})"
    elif llm_cache:
        res = f"Ungültige LLM-Antwort → Regel-Fallback: {llm_cache.get('detector')}"
    else:
        res = "kein Cache — make cache ausführen"
    ax.text(5.0, 3.5, res, ha="center", fontsize=12.5, fontweight="bold", color="#333")
    ax.text(0.3, 2.2, "① Sicher: kann nicht ausbrechen / nichts Ungültiges (Form + KB-Validierung + Regel-Fallback).",
            fontsize=12, color=OK)
    ax.text(0.3, 1.4, "② Nicht-deterministisch: Wahl schwankt je Lauf → wir cachen eine gute (make cache).",
            fontsize=12, color=WARN)
    return finish(fig, title="Natives LLM-Routing: abgesichert & nicht-deterministisch",
                  takeaway="Das LLM übernimmt NUR die Auswahl — alles andere bleibt PyOD.",
                  save_as=save_as)


# --- 9) Fazit ------------------------------------------------------------------------
_STRAT_COLORS = [MUTED, "#55A868", "#4C72B0", "#8172B3", HIGHLIGHT, "#E1A100"]


def fig_final_comparison(strategies: dict[str, tuple[str, float]],
                         *, save_as: str | None = None):
    labels = list(strategies)
    picks = [strategies[s][0] for s in labels]
    vals = [strategies[s][1] for s in labels]
    colors = (_STRAT_COLORS * 2)[: len(labels)]
    fig, ax = plt.subplots()
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    ax.set(ylim=(0, 1.10), ylabel="ROC-AUC (illustrativ)")
    if len(labels) > 4:
        ax.tick_params(axis="x", labelrotation=15, labelsize=12)
    for b, pick, v in zip(bars, picks, vals, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{pick}\n{v:.3f}", ha="center",
                fontsize=11.5, fontweight="bold")
    return finish(fig, title="Fazit: label-freie Strategien im Vergleich",
                  takeaway="Konsens- und Engine-Auswahl trennen sauber — ganz ohne Labels.",
                  caveat=CAVEAT_AUC, save_as=save_as)
