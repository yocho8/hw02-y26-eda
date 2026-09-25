from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from data_loading import load_records
from preparation import prepare_accidents, quality_by_year
from presentation import number, table_style, configure_charts

configure_charts()
from exploration import show_exploration
from advanced import show_advanced, show_grid

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
FILTER_PREFIX = "filter_"
LABELS = {"region": "Территория", "category": "Тип ДТП", "severity": "Последствия"}

st.set_page_config(page_title="Атлас аварийности", page_icon="🗺️", layout="wide")


@st.cache_data(show_spinner="Читаем выгрузку…")
def read_data(path: str, modified_ns: int, size: int) -> tuple[list[dict], pd.DataFrame]:
    records = load_records(Path(path))
    analytical = prepare_accidents(records)
    return records, analytical


def reset_filters() -> None:
    for key in list(st.session_state):
        if str(key).startswith(FILTER_PREFIX):
            del st.session_state[key]


def main() -> None:
    st.title("Атлас аварийности")
    st.caption("ДТП Костромской области · исследование данных")
    files = sorted(DATA_DIR.glob("*.geojson")) + sorted(DATA_DIR.glob("*.json"))
    files = [p for p in files if not p.name.endswith(".source.json")]
    if not files:
        st.warning("В data/raw пока нет выгрузки. Скачайте файл своего региона.")
        st.code(
            "python data/assign_region.py YOUR_GITHUB_USERNAME\n"
            "python data/import_data.py /path/to/region.geojson.zip",
            language="bash",
        )
        st.link_button("Открыть данные «Карты ДТП»", "https://dtp-stat.ru/opendata/")
        st.markdown("Инструкция находится в `data/README.md` репозитория.")
        return

    selected_path = st.sidebar.selectbox(
        "Выгрузка", files, format_func=lambda p: p.name,
        key="source_file", on_change=reset_filters,
    )
    try:
        stat = selected_path.stat()
        records, frame = read_data(str(selected_path), stat.st_mtime_ns, stat.st_size)
    except (ValueError, OSError, TypeError) as error:
        st.error(f"Не удалось прочитать выгрузку: {error}")
        return
    if frame.empty:
        st.warning("Файл содержит пустой список записей.")
        return

    invalid_dates = int(frame["datetime"].isna().sum())
    duplicate_ids = int(frame.loc[frame["id"].notna(), "id"].duplicated().sum())
    st.sidebar.caption(f"Исходных записей: {number(len(frame))}")
    st.sidebar.caption(
        f"Без распознанной даты: {number(invalid_dates)}. "
        f"Повторных непустых ID: {number(duplicate_ids)}. Дубликаты не удалены."
    )
    dated = frame[frame["datetime"].notna()]
    if dated.empty:
        st.error("Нет распознанных дат. Проверьте исходный формат и загрузчик.")
        st.dataframe(frame.head(20), hide_index=True)
        return

    first, last = dated["datetime"].min().date(), dated["datetime"].max().date()
    st.sidebar.button("Сбросить фильтры", on_click=reset_filters)
    period = st.sidebar.date_input(
        "Период", value=(max(first, min(last, pd.Timestamp("2015-01-01").date())), min(last, max(first, pd.Timestamp("2025-12-31").date())) ), min_value=first, max_value=last,
        key="filter_period",
    )
    if len(period) != 2:
        st.info("Выберите начало и конец периода.")
        return
    start, end = period
    current = dated[
        (dated["datetime"] >= pd.Timestamp(start))
        & (dated["datetime"] < pd.Timestamp(end) + pd.Timedelta(days=1))
    ].copy()
    for column, title in list(LABELS.items())[:2]:
        include_all = st.sidebar.checkbox(
            "Все территории" if column == "region" else "Все типы ДТП",
            value=True, key=f"filter_all_{column}",
        )
        if include_all:
            continue
        values = frame[column]
        options = sorted(values.dropna().astype(str).unique().tolist())
        if values.isna().any():
            options.append(None)
        selected = st.sidebar.multiselect(
            title, options, default=[],
            format_func=lambda value: "〈значение отсутствует〉" if value is None else value,
            key=f"filter_{column}",
        )
        mask = current[column].astype("string").isin([v for v in selected if v is not None])
        if None in selected:
            mask |= current[column].isna()
        current = current[mask].copy()

    st.caption(
        f"Период: {start:%d.%m.%Y} - {end:%d.%m.%Y}. "
        "Ниже показаны данные с учётом фильтров. Исходные записи посчитаны "
        "без автоматической очистки."
    )
    if invalid_dates:
        st.warning(f"Из фильтра по времени исключено записей без даты: {number(invalid_dates)}.")
    if current.empty:
        st.warning("В выбранном срезе нет записей. Измените или сбросьте фильтры.")
        return
    a, b, c = st.columns(3)
    a.metric("Записей в срезе", f"{number(len(current))}")
    b.metric("Уникальных непустых ID", f"{number(current['id'].nunique())}")
    c.metric("Исключено из карт по координатам", f"{number(int((~current.map_valid | current.coordinate_review).sum()))}")
    if len(current) < 30:
        st.warning(
            "В выборке меньше 30 записей. Посмотрите, как отдельные ДТП влияют на результат. "
            "Число 30 здесь выбрано для напоминания, а не как критерий надёжности."
        )

    overview, exploration, geography, advanced, source, quality = st.tabs(["Динамика", "Исследование", "Карта", "Статистика и проекции", "Исходные записи", "Качество данных"])
    with advanced:
        show_advanced(frame)
        st.subheader("График до и после")
        for name, caption in [("lighting_before.png", "До: наклонные подписи, недостаточно пояснений"), ("lighting_after.png", "После: сортировка, единицы и знаменатели")]:
            asset = ROOT / "solution" / "assets" / name
            if asset.exists():
                st.image(str(asset), caption=caption, width="stretch")
        st.caption("Сохранённые иллюстрации относятся к исходной выгрузке Костромской области, 2015–2025; фильтры их не меняют.")
    with exploration:
        show_exploration(current)
    with quality:
        st.subheader("Проверки текущего среза")
        st.write("Единица анализа — одно ДТП. Исходная выгрузка сохранена без изменений; дубликаты автоматически не удаляются.")
        st.dataframe(pd.DataFrame({
            "Проверка": ["Повторные непустые ID", "Непригодные координаты", "Координаты требуют проверки", "Пострадавших больше, чем участников"],
            "Записей": [int(current.loc[current.id.notna(), "id"].duplicated().sum()), int((~current.map_valid).sum()), int(current.coordinate_review.sum()), int(current.casualties_exceed_participants.sum())],
        }), hide_index=True)
        st.caption("Дополнительный экран координат: долгота 40–48°, широта 57–60°. Это грубая проверка, а не граница области. Точки за прямоугольником не исправляются автоматически.")
        st.subheader("Что требует внимания")
        issues = pd.DataFrame({
            "Проверка": ["Нет адреса", "Число вложенных участников не совпадает со счётчиком", "Не указан родительский регион"],
            "ДТП": [int(current.address_missing.sum()), int(current.participants_mismatch.sum()), int(current.parent_region_missing.sum())],
        })
        issues["Доля среза, %"] = issues["ДТП"] / len(current) * 100
        st.table(issues.style.format({"ДТП": lambda x: number(x), "Доля среза, %": lambda x: number(x, 1)}))
        st.caption("Проблемы могут пересекаться в одном ДТП, поэтому строки не складываются. Отсутствующий родительский регион может быть особенностью разметки, а не ошибкой. Вложенные списки не заменяют общий счётчик участников.")
        field_names = {"id": "ID происшествия", "datetime": "Дата и время", "region": "Территория", "category": "Тип ДТП", "severity": "Тяжесть последствий", "light": "Освещение", "participants_count": "Число участников", "injured_count": "Число раненых", "dead_count": "Число погибших"}
        missing = current[list(field_names)].isna().sum().rename_axis("Код поля").reset_index(name="Пропусков")
        missing.insert(0, "Поле", missing["Код поля"].map(field_names))
        missing["Доля, %"] = missing["Пропусков"] / len(current) * 100
        if missing["Пропусков"].sum() == 0:
            st.success(f"В девяти основных полях текущего среза ({number(len(current))} ДТП) пропусков нет: ID, дата, территория, тип ДТП, тяжесть, освещение, число участников, раненых и погибших.")
        else:
            st.write("Обнаруженные пропуски в основных полях")
            st.table(missing.loc[missing["Пропусков"].gt(0), ["Поле", "Пропусков", "Доля, %"]])
        st.subheader("Сопоставимость данных по годам")
        coordinates, severity = quality_by_year(current)
        st.caption("Число записей каждой исходной категории тяжести по годам текущего среза.")
        st.dataframe(severity, hide_index=True, width="stretch")
        st.warning("В полной выгрузке за 2020–2024 годы нет категории «Легкий». Причина неизвестна: возможно изменение регистрации или обработки. Сравнивать лёгкие и тяжёлые ДТП между годами напрямую нельзя. Для исследования используем наличие погибших по счётчику; оно совпадает с категорией «С погибшими» в этой выгрузке.")
        st.caption("Исключение координат по годам текущего среза: знаменатель — все ДТП данного года в срезе.")
        st.dataframe(coordinates.style.format({"Доля исключённых, %": lambda x: number(x, 2)}), hide_index=True, width="stretch")
        st.write("Ошибки координат сосредоточены в ранних годах. Пространственные сравнения по годам могут искажаться из-за разной полноты карт.")
        st.info("Во всех записях исходной выгрузки есть хотя бы один раненый или погибший. Результаты описывают ДТП с пострадавшими; переносить их на аварии только с материальным ущербом нельзя.")
        st.info("Для основных сравнений выбран период 2015–2025. Январь 2026 остаётся доступен в фильтре. Наличие записей во всех месяцах не доказывает полноту регистрации ДТП.")
    with overview:
        counts = current.set_index("datetime").resample("MS").size().rename("records")
        months = pd.date_range(pd.Timestamp(start).to_period("M").start_time,
                               pd.Timestamp(end).to_period("M").start_time, freq="MS")
        counts = counts.reindex(months, fill_value=0).rename_axis("month").reset_index()
        fig = px.line(counts, x="month", y="records", markers=True,
                      labels={"month": "Месяц", "records": "Число исходных записей"})
        st.subheader("Число записей по месяцам")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Ноль на графике означает, что за этот месяц в выбранных данных нет записей. "
            "Первый и последний месяцы могут быть неполными. "
            "Перед сравнением проверьте, за все ли месяцы данные собраны полностью."
        )
        peak = counts.loc[counts.records.idxmax()]
        st.write(f"Максимум в выбранном срезе: {peak['month']:%m.%Y}, {number(peak['records'])} записей. Изменение числа записей может отражать как происшествия, так и полноту регистрации; риск поездки без транспортного потока не оцениваем.")
    with geography:
        st.caption(f"Из обеих карт исключено: {number(int((~current.map_valid).sum()))} непригодных координат и {number(int(current.coordinate_review.sum()))} точек вне диагностического прямоугольника. Остальные точки не проверены по административной границе.")
        points = current[current.map_valid & ~current.coordinate_review].copy()
        if points.empty:
            st.info("В срезе нет пригодных для карты координат.")
        else:
            if len(points) > 5000:
                points = points.sample(5000, random_state=42)
                st.warning(
                    "На точечной карте показана случайная подвыборка 5 000 точек. Сетка использует все проверенные точки. "
                    "Остальные показатели и CSV используют полный срез."
                )
            backdrop = st.checkbox("Показывать подложку карты", value=True)
            show_grid(current, backdrop)
            points["severity_display"] = points["has_deaths"].map({True: "С погибшими", False: "Без погибших"}).fillna("Неизвестно")
            fig = px.scatter_map(
                points, lat="latitude", lon="longitude", color="severity_display",
                color_discrete_map={
                    "Без погибших": "#3579B8",
                    "С погибшими": "#B33440", "〈значение отсутствует〉": "#8D939C",
                },
                category_orders={"severity_display": ["Без погибших", "С погибшими"]},
                hover_data=["source_row", "id", "datetime", "category", "region"],
                labels={"severity_display": "Последствия", **LABELS},
                opacity=0.6, zoom=5,
                center={"lat": float(points["latitude"].median()),
                        "lon": float(points["longitude"].median())},
                map_style="open-street-map" if backdrop else "white-bg", height=540,
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Подложка: © OpenStreetMap contributors, https://www.openstreetmap.org/copyright. "
                "Координаты взяты из выгрузки; принадлежность региону здесь не проверена."
            )
    with source:
        columns = ["source_row", "id", "datetime", "region", "category", "severity",
                   "participants_count", "injured_count", "dead_count", "latitude", "longitude"]
        st.dataframe(current[columns].head(500).rename(columns={"source_row": "Строка источника", "id": "ID", "datetime": "Дата и время", "region": "Территория", "category": "Тип ДТП", "severity": "Категория тяжести", "participants_count": "Участников", "injured_count": "Раненых", "dead_count": "Погибших", "latitude": "Широта", "longitude": "Долгота"}), hide_index=True, width="stretch")
        st.caption("Первые 500 записей текущего среза; CSV содержит весь срез.")
        st.download_button(
            "Скачать текущий срез CSV", current[columns].to_csv(index=False).encode("utf-8-sig"),
            file_name="selected_records.csv", mime="text/csv",
        )
        lookup = current.set_index("source_row")
        chosen = st.selectbox(
            "Открыть исходную запись", current["source_row"].tolist(),
            format_func=lambda index: f"Строка {index} · ID {lookup.at[index, 'id']}",
        )
        st.json(records[int(chosen)], expanded=False)


if __name__ == "__main__":
    main()
