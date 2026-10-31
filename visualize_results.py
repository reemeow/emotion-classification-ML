
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap


# ==========================================
# 1. File Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "final_evaluation"

METRICS_FILE = RESULTS_DIR / "final_evaluation_metrics.csv"
CONFUSION_FILE = RESULTS_DIR / "final_confusion_matrix.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

metrics = pd.read_csv(METRICS_FILE)
confusion = pd.read_csv(CONFUSION_FILE, index_col=0)

CLASSES = ["Joy", "Sadness", "Anger", "Neutral"]
PREDICTIONS = CLASSES + ["Uncertain"]


# ==========================================
# 2. Academic Style
# ==========================================

GREEN = "#426B51"
DARK_GREEN = "#234632"
TEXT = "#222222"
WHITE = "#FFFFFF"

available_fonts = {
    font.name
    for font in font_manager.fontManager.ttflist
}

if "Times New Roman" in available_fonts:
    FONT = "Times New Roman"
elif "Liberation Serif" in available_fonts:
    FONT = "Liberation Serif"
else:
    FONT = "DejaVu Serif"

plt.rcParams.update({
    "font.family": FONT,
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "savefig.dpi": 300,
})

CMAP = LinearSegmentedColormap.from_list(
    "ForestGreen",
    [
        "#FFFFFF",
        "#D6E3D9",
        "#83A18B",
        DARK_GREEN
    ]
)


# ==========================================
# 3. Class-wise F1-score
# ==========================================

def plot_f1_scores():

    class_metrics = (
        metrics.set_index("Emotion")
        .loc[CLASSES]
    )

    scores = (
        class_metrics["F1"]
        .astype(float)
        .to_numpy()
    )

    fig, ax = plt.subplots(figsize=(6.2, 3.5))

    y = np.arange(len(CLASSES))

    ax.barh(
        y,
        scores,
        color=GREEN,
        height=0.55,
        edgecolor=DARK_GREEN,
        linewidth=0.5
    )

    ax.set_yticks(y)
    ax.set_yticklabels(CLASSES)
    ax.invert_yaxis()

    ax.set_xlim(0, 1.05)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("F1-score")

    for i, score in enumerate(scores):
        ax.text(
            score + 0.015,
            i,
            f"{score:.2f}",
            va="center",
            ha="left",
            fontsize=10,
            color=TEXT
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.spines["bottom"].set_color(TEXT)

    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", direction="out")

    fig.tight_layout()

    output = RESULTS_DIR / "class_f1_scores.png"

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output.name}")


# ==========================================
# 4. Confusion Matrix
# ==========================================

def plot_confusion_matrix():

    matrix = (
        confusion.loc[CLASSES, PREDICTIONS]
        .astype(int)
        .to_numpy()
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))

    im = ax.imshow(
        matrix,
        cmap=CMAP,
        interpolation="nearest",
        aspect="auto",
        vmin=0,
        vmax=matrix.max()
    )

    ax.set_xticks(np.arange(len(PREDICTIONS)))
    ax.set_yticks(np.arange(len(CLASSES)))

    ax.set_xticklabels(PREDICTIONS)
    ax.set_yticklabels(CLASSES)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):

            value = matrix[i, j]

            color = (
                WHITE
                if im.norm(value) > 0.55
                else TEXT
            )

            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=10,
                color=color
            )

    ax.set_xticks(
        np.arange(-0.5, len(PREDICTIONS), 1),
        minor=True
    )

    ax.set_yticks(
        np.arange(-0.5, len(CLASSES), 1),
        minor=True
    )

    ax.grid(
        which="minor",
        color=WHITE,
        linewidth=1
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False
    )

    ax.tick_params(
        which="major",
        length=0
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.035,
        pad=0.04
    )

    cbar.set_label("Number of samples")
    cbar.outline.set_visible(False)

    fig.tight_layout()

    output = RESULTS_DIR / "confusion_matrix.png"

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output.name}")


# ==========================================
# 5. Overall Evaluation Table
# ==========================================

def create_metrics_table():

    accuracy = float(metrics["Accuracy"].iloc[0])
    macro_f1 = float(metrics["Macro F1"].iloc[0])
    coverage = float(metrics["Coverage"].iloc[0])

    rows = [
        ["Accuracy", f"{accuracy:.2%}"],
        ["Macro F1-score", f"{macro_f1:.4f}"],
        ["Coverage", f"{coverage:.2%}"]
    ]

    fig, ax = plt.subplots(figsize=(5, 1.9))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.65, 0.35]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():

        cell.set_facecolor(WHITE)
        cell.set_edgecolor("#B8B8B8")
        cell.set_linewidth(0.4)

        cell.get_text().set_fontfamily(FONT)
        cell.get_text().set_color(TEXT)

        if row == 0:
            cell.get_text().set_weight("bold")
            cell.set_edgecolor(DARK_GREEN)
            cell.set_linewidth(0.8)

    output = RESULTS_DIR / "evaluation_table.png"

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.15
    )

    plt.close(fig)

    print(f"Saved: {output.name}")


# ==========================================
# 6. Main
# ==========================================

if __name__ == "__main__":

    print(f"Using font: {FONT}")

    plot_f1_scores()
    plot_confusion_matrix()
    create_metrics_table()

    print("\nAll PNG figures generated successfully!")
