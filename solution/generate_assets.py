"""Повторно создать изображения «до» и «после» для отчёта."""

import argparse
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt

from data_loading import load_records
from preparation import prepare_accidents


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "raw" / "kostromskaia-oblast.geojson"
ASSET_DIR = ROOT / "solution" / "assets"

LIGHT_LABELS = {
    "В темное время суток, освещение включено": "Темно, свет включён",
    "В темное время суток, освещение не включено": "Темно, свет не включён",
    "В темное время суток, освещение отсутствует": "Темно, освещения нет",
    "Светлое время суток": "Светлое время",
    "Сумерки": "Сумерки",
}


def lighting_summary(frame):
    """Посчитать долю ДТП с погибшими для каждого условия освещения."""
    summary = (
        frame.groupby("light", dropna=False)
        .agg(total=("id", "size"), known=("has_deaths", "count"), fatal=("has_deaths", "sum"))
        .reset_index()
    )
    summary["percent"] = summary["fatal"] / summary["known"] * 100
    summary["label"] = summary["light"].map(LIGHT_LABELS).fillna("Не указано")
    return summary


def save_before(summary, path: Path) -> None:
    """Сохранить намеренно неудобную исходную версию графика."""
    order = list(LIGHT_LABELS.values())
    view = summary.set_index("label").reindex(order).dropna(subset=["percent"])
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    ax.bar(view.index, view["percent"], color="#1f77b4")
    ax.set_title("Освещение и ДТП с погибшими")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_after(summary, records_count: int, path: Path) -> None:
    """Сохранить улучшенную версию с единицами, числами групп и пояснением."""
    view = summary.sort_values("percent", ascending=False).copy()
    labels = ["\n".join(textwrap.wrap(label, 18)) for label in view["label"]]
    colors = ["#c58a0a" if label == "Темно, освещения нет" else "#397eb8" for label in view["label"]]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    bars = ax.bar(labels, view["percent"], color=colors)
    ax.set_title(
        "В темноте без освещения доля ДТП с погибшими выше\n"
        f"Костромская область · 2015–2025 · {records_count:,} ДТП".replace(",", " "),
        loc="left",
    )
    ax.set_ylabel("ДТП с погибшими, % от ДТП своей группы")
    ax.set_ylim(0, 25)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, row in zip(bars, view.itertuples(index=False)):
        fatal = f"{int(row.fatal):,}".replace(",", " ")
        known = f"{int(row.known):,}".replace(",", " ")
        label = f"{row.percent:.1f}%\n{fatal} / {known}".replace(".", ",")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.55, label, ha="center", va="bottom")

    fig.text(
        0.02,
        0.015,
        "Подпись: доля и ДТП с погибшими / ДТП группы. Связь не доказывает причинность.",
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()

    frame = prepare_accidents(load_records(args.source))
    frame = frame[frame["in_analysis_period"]]
    if frame.empty:
        raise ValueError("В основном периоде 2015–2025 нет данных")

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    summary = lighting_summary(frame)
    save_before(summary, ASSET_DIR / "lighting_before.png")
    save_after(summary, len(frame), ASSET_DIR / "lighting_after.png")
    print(f"Изображения сохранены в {ASSET_DIR}")


if __name__ == "__main__":
    main()
