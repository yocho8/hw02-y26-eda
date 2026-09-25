"""Аналитическая таблица: одна строка на ДТП, исходные записи не изменяются."""
import pandas as pd

from data_loading import records_to_frame, properties


COUNT_COLUMNS = ("participants_count", "injured_count", "dead_count")


def quality_by_year(frame):
    """Посчитать показатели для переданного среза, сохраняя годы с нулевыми значениями."""
    f = frame.assign(excluded=~frame.map_valid | frame.coordinate_review)
    coordinates = f.groupby("year").agg(ДТП=("id", "size"), Исключено=("excluded", "sum"))
    coordinates["Доля исключённых, %"] = coordinates.Исключено / coordinates.ДТП * 100
    coordinates.index.name = "Год"
    severity = pd.crosstab(f.year, f.severity).reindex(columns=["Легкий", "Тяжёлый", "С погибшими"], fill_value=0)
    severity.index.name = "Год"
    return coordinates.reset_index(), severity.reset_index()


def prepare_accidents(records: list[dict]) -> pd.DataFrame:
    frame = records_to_frame(records).copy()
    if frame.empty:
        return frame
    props = [properties(record) for record in records]
    frame["address_missing"] = [not p.get("address") for p in props]
    frame["parent_region_missing"] = [not p.get("parent_region") for p in props]
    frame["nested_participants_count"] = [len(p.get("participants") or []) + sum(len(v.get("participants") or []) for v in (p.get("vehicles") or [])) for p in props]
    # Сохраняем пропуски и конфликтующие дубликаты для проверки, не выбирая запись молча.
    for column in COUNT_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.notna() & values.ge(0) & values.mod(1).eq(0)
        frame[column + "_invalid"] = ~valid
        frame[column] = values.where(valid).astype("Int64")
    for part in ("year", "month", "dayofweek", "hour"):
        frame[part] = getattr(frame["datetime"].dt, part).astype("Int64")
    frame["year_month"] = frame["datetime"].dt.to_period("M").astype("string")
    frame["season"] = frame["month"].map({
        12: "Зима", 1: "Зима", 2: "Зима", 3: "Весна", 4: "Весна", 5: "Весна",
        6: "Лето", 7: "Лето", 8: "Лето", 9: "Осень", 10: "Осень", 11: "Осень",
    })
    frame["is_weekend"] = frame["dayofweek"].ge(5)
    frame["has_deaths"] = frame["dead_count"].gt(0)
    frame["casualties_count"] = frame["injured_count"] + frame["dead_count"]
    frame["casualties_exceed_participants"] = frame["casualties_count"].gt(frame["participants_count"])
    frame["participants_mismatch"] = frame["participants_count"].ne(frame["nested_participants_count"])
    # Широкий прямоугольник используется только для диагностики и не является границей области.
    frame["coordinate_review"] = frame["map_valid"] & ~(
        frame["longitude"].between(40, 48) & frame["latitude"].between(57, 60)
    )
    # Календарный период основного анализа; полнота покрытия проверяется отдельно.
    frame["in_analysis_period"] = frame["datetime"].between(
        "2015-01-01", "2025-12-31 23:59:59"
    )
    return frame
