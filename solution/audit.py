"""Повторить аудит и выгрузить аналитические таблицы без изменения исходных данных."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
from importlib.metadata import version

import pandas as pd

from data_loading import load_records, properties, records_to_frame
from preparation import COUNT_COLUMNS, prepare_accidents, quality_by_year


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path,
                        default=ROOT / "data/raw/kostromskaia-oblast.geojson")
    args = parser.parse_args()
    records = load_records(args.source)
    raw = records_to_frame(records)
    frame = prepare_accidents(records)
    if frame.empty:
        raise ValueError("Нет записей для аудита")
    output = ROOT / "solution/outputs"
    output.mkdir(parents=True, exist_ok=True)
    props = [properties(r) for r in records]
    fields = sorted(set().union(*(p.keys() for p in props)))
    field_rows = []
    for key in fields:
        values = [p.get(key) for p in props]
        field_rows.append({"field": key, "missing_or_null": sum(v is None for v in values),
                           "empty_string_or_list": sum(v == "" or v == [] for v in values),
                           "types": dict(Counter(type(v).__name__ for v in values))})
    dates = frame["datetime"].dropna()
    monthly = dates.groupby(dates.dt.to_period("M")).size()
    months = pd.period_range(dates.min(), dates.max(), freq="M")
    monthly = monthly.reindex(months, fill_value=0)
    canonical = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records]
    nested_counts = []
    for p in props:
        outside = p.get("participants") or []
        vehicles = p.get("vehicles") or []
        nested_counts.append(len(outside) + sum(len(v.get("participants") or []) for v in vehicles))
    metadata_path = args.source.with_suffix(".source.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    summary = {
        "records": len(frame), "sha256": digest, "metadata": metadata,
        "metadata_hash_matches": metadata.get("sha256") == digest,
        "metadata_count_matches": metadata.get("records") == len(frame),
        "date_min": str(dates.min()), "date_max": str(dates.max()),
        "missing_dates": int(frame.datetime.isna().sum()),
        "future_dates": int((dates > pd.Timestamp.now()).sum()),
        "unique_ids": int(frame.id.nunique()), "missing_ids": int(frame.id.isna().sum()),
        "repeated_nonempty_ids": int(frame.loc[frame.id.notna(), "id"].duplicated().sum()),
        "identical_record_duplicates": len(canonical) - len(set(canonical)),
        "missing_fields": raw.isna().sum().to_dict(),
        "invalid_counts": {c: int(frame[c + "_invalid"].sum()) for c in COUNT_COLUMNS},
        "casualties_exceed_participants": int(frame.casualties_exceed_participants.sum()),
        "nested_participants_mismatch": int(frame.participants_count.ne(pd.Series(nested_counts)).sum()),
        "map_invalid": int((~frame.map_valid).sum()),
        "zero_coordinates": int(((frame.latitude == 0) & (frame.longitude == 0)).sum()),
        "coordinate_review": int(frame.coordinate_review.sum()),
        "valid_coordinate_ranges": frame.loc[frame.map_valid, ["longitude", "latitude"]].agg(["min", "max"]).to_dict(),
        "by_year": {str(k): int(v) for k, v in frame.year.value_counts().sort_index().items()},
        "months_without_records": [str(m) for m in monthly[monthly == 0].index],
        "analysis_period_n": int(frame.in_analysis_period.sum()),
        "categories": {c: frame[c].value_counts(dropna=False).to_dict() for c in ("region", "category", "severity", "light")},
        "numeric_summary": frame[list(COUNT_COLUMNS)].describe().to_dict(),
        "fields": field_rows,
        "versions": {"python": platform.python_version(), **{p: version(p) for p in ("pandas", "numpy", "streamlit", "plotly", "scipy", "scikit-learn")}},
    }
    (output / "audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=int) + "\n", encoding="utf-8")
    coordinates, severity = quality_by_year(frame)
    coordinates.to_csv(output / "coordinate_quality_by_year.csv", index=False)
    severity.to_csv(output / "severity_by_year.csv", index=False)
    monthly.rename("accidents").to_csv(output / "monthly_counts.csv", index_label="month")
    frame.to_csv(output / "accidents.csv", index=False, encoding="utf-8-sig")
    frame.loc[frame.coordinate_review].to_csv(output / "coordinate_review.csv", index=False, encoding="utf-8-sig")
    # Небольшая выборка исходных записей для ручной проверки хранится только в игнорируемой папке.
    indices = list(dict.fromkeys([0, len(records)//2, len(records)-1,
        *frame.index[~frame.map_valid][:2].tolist(), int(frame.injured_count.idxmax())]))
    (output / "raw_examples.json").write_text(json.dumps([records[i] for i in indices], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("fields", "categories", "numeric_summary")}, ensure_ascii=False, indent=2))
    print(f"Результаты: {output}")


if __name__ == "__main__":
    main()
