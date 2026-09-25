import numpy as np
import h3
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from presentation import number, table_style

ON = "В темное время суток, освещение включено"
OFF = "В темное время суток, освещение отсутствует"
SEED = 42


def format_p(value, label="p"):
    if value < 0.000001:
        return f"{label} < 0,000001"
    decimal = f"{value:.8f}".rstrip("0").rstrip(".").replace(".", ",")
    return f"{label} = {decimal}"


def inference(frame):
    f = frame[frame.in_analysis_period]
    definitions = [
        (
            "Освещение и наличие погибших",
            f[f.light.eq(OFF)].has_deaths,
            f[f.light.eq(ON)].has_deaths,
            "Доля ДТП с погибшими без освещения минус доля при включённом освещении",
            "χ² Пирсона",
            100,
        ),
        (
            "Наезды на пешеходов: освещение и наличие погибших",
            f[f.light.eq(OFF) & f.category.eq("Наезд на пешехода")].has_deaths,
            f[f.light.eq(ON) & f.category.eq("Наезд на пешехода")].has_deaths,
            "Доля ДТП с погибшими без освещения минус доля при включённом освещении, только наезды на пешеходов",
            "χ² Пирсона",
            100,
        ),
        (
            "Выходные и будни: число пострадавших",
            f[f.is_weekend].casualties_count,
            f[~f.is_weekend].casualties_count,
            "Среднее число пострадавших на ДТП в выходные минус среднее в будни",
            "t-тест Уэлча",
            1,
        ),
    ]
    results = []
    rng = np.random.default_rng(SEED)
    for title, a, b, meaning, test, scale in definitions:
        x, y = a.dropna().to_numpy(dtype=float), b.dropna().to_numpy(dtype=float)
        if min(len(x), len(y)) < 2:
            raise ValueError(
                "Недостаточно данных для фиксированных статистических проверок"
            )
        expected_min = None
        if scale == 100:
            table = [[x.sum(), len(x) - x.sum()], [y.sum(), len(y) - y.sum()]]
            statistic, p, _, expected = stats.chi2_contingency(table, correction=False)
            expected_min = float(expected.min())
        else:
            statistic, p = stats.ttest_ind(x, y, equal_var=False)
        replicates = []
        for _ in range(50):
            bx = rng.choice(x, size=(100, len(x)), replace=True).mean(axis=1)
            by = rng.choice(y, size=(100, len(y)), replace=True).mean(axis=1)
            replicates.extend((bx - by) * scale)
        lo, hi = np.quantile(replicates, [0.025, 0.975])
        results.append(
            dict(
                title=title,
                meaning=meaning,
                test=test,
                n_a=len(x),
                n_b=len(y),
                mean_a=float(x.mean() * scale),
                mean_b=float(y.mean() * scale),
                effect=float((x.mean() - y.mean()) * scale),
                low=float(lo),
                high=float(hi),
                p=float(p),
                statistic=float(statistic),
                expected_min=expected_min,
                units="п.п." if scale == 100 else "чел./ДТП",
            )
        )
    order = np.argsort([r["p"] for r in results])
    running = 0
    for rank, idx in enumerate(order):
        running = max(running, min(1, (len(results) - rank) * results[idx]["p"]))
        results[idx]["p_holm"] = running
    return results


def projections(frame):
    f = frame[frame.in_analysis_period].copy()
    features = pd.DataFrame(
        {
            "Доля столкновений": f.category.eq("Столкновение").astype(float),
            "Доля наездов на пешеходов": f.category.eq("Наезд на пешехода").astype(
                float
            ),
            "Доля съездов с дороги": f.category.eq("Съезд с дороги").astype(float),
            "Доля ДТП без освещения": f.light.eq(OFF).astype(float),
            "Доля ДТП со включённым освещением": f.light.eq(ON).astype(float),
            "Доля зимних ДТП": f.season.eq("Зима").astype(float),
            "Доля ДТП в выходные": f.is_weekend.astype(float),
            "Среднее число участников": f.participants_count.astype(float),
        }
    )
    features["region"], features["year"] = f.region, f.year
    grouped = features.groupby(["region", "year"])
    means = grouped.mean()
    sizes = grouped.size()
    means = means[sizes >= 20].dropna()
    if len(means) <= 30:
        raise ValueError("Для проекций требуется больше 30 профилей с n ≥ 20")
    scaler = StandardScaler()
    x = scaler.fit_transform(means)
    model = PCA(n_components=2, svd_solver="full")
    pc = model.fit_transform(x)
    view = means.reset_index()
    view["n"] = sizes.loc[means.index].to_numpy()
    view["PC1"], view["PC2"] = pc[:, 0], pc[:, 1]
    qualities, neighbors = {}, []
    for perplexity in (10, 30):
        z = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=SEED,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
        ).fit_transform(x)
        view[f"t{perplexity}_1"], view[f"t{perplexity}_2"] = z[:, 0], z[:, 1]
        qualities[str(perplexity)] = float(trustworthiness(x, z, n_neighbors=10))
        neighbors.append(
            NearestNeighbors(n_neighbors=11)
            .fit(z)
            .kneighbors(z, return_distance=False)[:, 1:]
        )
    overlap = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(*neighbors)])
    return view, dict(
        variance=model.explained_variance_ratio_.tolist(),
        loadings=pd.DataFrame(
            model.components_.T, index=means.columns, columns=["PC1", "PC2"]
        ),
        trustworthiness=qualities,
        neighbor_overlap=float(overlap),
    )


def sensitivity(frame):
    f = frame[frame.in_analysis_period].copy()
    months = pd.period_range("2015-01", "2025-12", freq="M").astype(str)
    rng = np.random.default_rng(43)
    draws = rng.integers(0, len(months), size=(5000, len(months)))
    masks = [
        (f.light.eq(OFF), f.light.eq(ON), "has_deaths", 100),
        (
            f.light.eq(OFF) & f.category.eq("Наезд на пешехода"),
            f.light.eq(ON) & f.category.eq("Наезд на пешехода"),
            "has_deaths",
            100,
        ),
        (f.is_weekend, ~f.is_weekend, "casualties_count", 1),
    ]
    intervals = []
    for a, b, column, scale in masks:
        means = []
        for mask in (a, b):
            monthly = (
                f.loc[mask]
                .groupby("year_month")[column]
                .agg(["sum", "count"])
                .reindex(months, fill_value=0)
            )
            sums = monthly["sum"].to_numpy(dtype=float)[draws].sum(axis=1)
            counts = monthly["count"].to_numpy(dtype=float)[draws].sum(axis=1)
            means.append(
                np.divide(
                    sums, counts, out=np.full_like(sums, np.nan), where=counts > 0
                )
            )
        effect = (means[0] - means[1]) * scale
        if not np.isfinite(effect).all():
            raise ValueError(
                "Для месячного bootstrap недостаточно наблюдений в группах"
            )
        low, high = np.quantile(effect, [0.025, 0.975])
        intervals.append(dict(low=float(low), high=float(high)))
    yearly = (
        f[f.light.isin([ON, OFF])]
        .groupby(["year", "light"])
        .has_deaths.mean()
        .unstack()
        * 100
    )
    yearly = yearly.reindex(columns=[OFF, ON]).rename(
        columns={OFF: "Освещения нет, %", ON: "Свет включён, %"}
    )
    yearly["Разница, п.п."] = yearly.iloc[:, 0] - yearly.iloc[:, 1]
    yearly.index.name = "Год"
    trimmed = f[f.casualties_count.lt(f.casualties_count.max())]
    delta = (
        trimmed.loc[trimmed.is_weekend, "casualties_count"].mean()
        - trimmed.loc[~trimmed.is_weekend, "casualties_count"].mean()
    )
    return dict(
        month_intervals=intervals, yearly=yearly.reset_index(), without_max=float(delta)
    )


def grid_counts(frame):
    points = frame[frame.map_valid & ~frame.coordinate_review].copy()
    points["h3"] = [
        h3.latlng_to_cell(lat, lon, 7)
        for lat, lon in zip(points.latitude, points.longitude)
    ]
    grid = points.groupby("h3").size().reset_index(name="ДТП")
    centers = [h3.cell_to_latlng(cell) for cell in grid.h3]
    grid["latitude"] = [c[0] for c in centers]
    grid["longitude"] = [c[1] for c in centers]
    grid["Площадь, км²"] = [h3.cell_area(cell, unit="km^2") for cell in grid.h3]
    return grid


def show_grid(current, backdrop):
    grid = grid_counts(current)
    if grid.empty:
        st.info("Нет точек для агрегации после проверки координат.")
        return
    features = []
    for cell in grid.h3:
        boundary = [[lon, lat] for lat, lon in h3.cell_to_boundary(cell)]
        boundary.append(boundary[0])
        features.append(
            dict(
                type="Feature",
                id=cell,
                properties={},
                geometry=dict(type="Polygon", coordinates=[boundary]),
            )
        )
    fig = px.choropleth_map(
        grid,
        geojson=dict(type="FeatureCollection", features=features),
        locations="h3",
        color="ДТП",
        color_continuous_scale="YlOrRd",
        hover_data={"Площадь, км²": ":.2f"},
        center=dict(
            lat=float(grid.latitude.median()), lon=float(grid.longitude.median())
        ),
        zoom=5,
        map_style="open-street-map" if backdrop else "white-bg",
        opacity=0.7,
        height=540,
    )
    fig.update_traces(marker_line_width=0.3, marker_line_color="rgba(80,80,80,0.4)")
    st.subheader("Число ДТП по гексагонам")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Гексагоны H3, разрешение 7. Площадь показанных ячеек {number(grid['Площадь, км²'].min(), 2)}–{number(grid['Площадь, км²'].max(), 2)} км². Цвет — число ДТП, не риск. Учтено {number(int(grid['ДТП'].sum()))} точек без подвыборки. Пустые ячейки не закрашены. Подложка: © OpenStreetMap contributors."
    )
    top_share = grid.nlargest(3, "ДТП")["ДТП"].sum() / grid["ДТП"].sum() * 100
    st.write(
        f"Три наиболее заполненные ячейки содержат {number(top_share, 1)}% пригодных для карты ДТП текущего среза. "
        "Это концентрация зарегистрированных событий, а не оценка риска поездки."
    )


@st.cache_data(show_spinner="Считаем гипотезы и проекции за 2015–2025 годы…")
def fixed_analysis(frame):
    return inference(frame), projections(frame), sensitivity(frame)


def show_advanced(frame):
    st.caption(
        "Фиксированная выборка: все территории и типы ДТП, 2015–2025 годы. Фильтры слева на этот раздел не влияют. Seed=42."
    )
    if frame.loc[frame.in_analysis_period, "id"].duplicated().any():
        st.error("Повторные ID: перед статистикой нужно разрешить дубликаты.")
        return
    try:
        tests, (view, info), robust = fixed_analysis(frame)
    except ValueError as error:
        st.warning(str(error))
        return
    st.subheader("Три статистические проверки")
    st.write(
        "Все H1 двусторонние; H0 — равенство соответствующих долей или средних. α=0,05. Интервалы: 95%, percentile bootstrap, 5 000 повторов с пересэмплированием ДТП внутри групп."
    )
    for i, r in enumerate(tests):
        st.markdown(f"**{r['title']}**")
        a, b = ("Без освещения", "Свет включён") if i < 2 else ("Выходные", "Будни")
        unit = "%" if i < 2 else "чел./ДТП"
        st.write(
            f"{a}: {number(r['mean_a'], 3)} {unit}, n = {number(r['n_a'])}; {b.lower()}: {number(r['mean_b'], 3)} {unit}, n = {number(r['n_b'])}."
        )
        st.write(
            f"{r['meaning']}. {r['test']}: эффект {number(r['effect'], 3)} {r['units']}, 95% ДИ [{number(r['low'], 3)}; {number(r['high'], 3)}]. {format_p(r['p'])}; {format_p(r['p_holm'], 'p с поправкой Холма')}."
        )
        if i < 2:
            st.caption(
                f"Таблица 2×2; минимальная ожидаемая частота {number(r['expected_min'], 2)} > 5. χ² без поправки Йейтса."
            )
        else:
            st.caption(
                "Уэлч сравнивает средние без требования равенства дисперсий. Группы большие, но распределение асимметрично; экстремальные значения сохранены."
            )
        st.write(
            "H0 отвергается после поправки Холма."
            if r["p_holm"] < 0.05
            else "Оснований отвергнуть H0 после поправки Холма нет."
        )
    st.info(
        "Как читать p: если H0 верна и предпосылки теста выполнены, p — вероятность получить значение статистики не менее экстремальное, чем наблюдаемое. Сравниваем p после поправки Холма с 0,05. Это не вероятность истинности H0 и не размер эффекта. Значения меньше 0,000001 показаны порогом, а не нулём."
    )
    st.caption(
        "Проверки разведочные: вопросы выбраны после просмотра данных. Предполагается независимость разных ДТП; временная и пространственная зависимость может сужать интервалы и занижать p-value. Доверительные интервалы индивидуальные, без поправки на множественность. Причинный эффект не установлен."
    )
    with st.expander("Устойчивость: годы, месяцы и максимальное значение"):
        st.write(
            "95% интервалы при пересэмплировании 132 календарных месяцев целиком: 5 000 повторов, seed=43. Одни и те же месяцы выбираются для обеих групп, сохраняются все ДТП внутри месяца. Это проверка чувствительности; зависимость между месяцами и пространственное смешение остаются."
        )
        for r, interval in zip(tests, robust["month_intervals"]):
            st.write(
                f"{r['title']}: [{number(interval['low'], 3)}; {number(interval['high'], 3)}] {r['units']}."
            )
        st.caption(
            "Таблица: доли ДТП с погибшими внутри каждой группы освещения и календарного года. Описательное сравнение, не скорректированный причинный эффект."
        )
        st.dataframe(table_style(robust["yearly"].set_index("Год")), width="stretch")
        st.write(
            f"Без ДТП с максимальным числом пострадавших разница выходных и будней: {number(robust['without_max'], 3)} чел./ДТП. Основной расчёт сохраняет это ДТП."
        )
    st.subheader("PCA: профили территории за год")
    st.write(
        f"Объект — метка территории × год с минимум 20 ДТП: {len(view)} профилей. Восемь признаков стандартизованы. Две компоненты сохраняют {number((sum(info['variance']))*100, 1)}% дисперсии."
    )
    color = "Доля ДТП без освещения"
    view = view.copy()
    for column in view.columns:
        if column.startswith("Доля"):
            view[column] *= 100
    display_labels = {
        "region": "Территория",
        "year": "Год",
        "n": "ДТП в профиле",
        **{c: c + ", %" for c in view if c.startswith("Доля")},
    }
    hover = [
        "region",
        "year",
        "n",
        "Доля наездов на пешеходов",
        "Среднее число участников",
    ]
    st.plotly_chart(
        px.scatter(
            view,
            x="PC1",
            y="PC2",
            color=color,
            hover_data=hover,
            labels=display_labels,
            color_continuous_scale="Viridis",
        ),
        width="stretch",
    )
    st.caption(
        "PC1 противопоставляет освещённые дороги и наезды на пешеходов съездам с дороги и отсутствующему освещению. PC2 прежде всего связана со столкновениями и средним числом участников; это описание сочетаний признаков."
    )
    with st.expander("Коэффициенты компонент и исходные признаки"):
        st.caption(
            "Коэффициенты стандартизованных признаков в компонентах PCA; знак оси условен."
        )
        st.dataframe(table_style(info["loadings"], 3))
        st.caption(
            "Исходные профили: доли показаны в процентах; в расчёте они стандартизованы."
        )
        st.dataframe(
            table_style(
                view.drop(columns=["PC1", "PC2", "t10_1", "t10_2", "t30_1", "t30_2"])
                .rename(columns=display_labels)
                .set_index(["Территория", "Год", "ДТП в профиле"])
            )
        )
    st.subheader("t-SNE: сравнение двух perplexity")
    for p in (10, 30):
        st.plotly_chart(
            px.scatter(
                view,
                x=f"t{p}_1",
                y=f"t{p}_2",
                color=color,
                hover_data=hover,
                title=f"perplexity={p}",
                labels={
                    **display_labels,
                    f"t{p}_1": "Координата 1",
                    f"t{p}_2": "Координата 2",
                },
                color_continuous_scale="Viridis",
            ),
            width="stretch",
        )
    st.write(
        f"Среднее совпадение десяти ближайших соседей между запусками: {number((info['neighbor_overlap'])*100, 1)}%. Trustworthiness: {number(info['trustworthiness']['10'], 3)} и {number(info['trustworthiness']['30'], 3)}."
    )
    st.caption(
        "Одинаковые объекты, признаки и seed=42; init=pca, learning_rate=auto, 1 000 итераций. Положение и расстояния между далёкими группами t-SNE не интерпретируются. Цвет — исходная доля ДТП без освещения, а не найденный класс. Метка «Костромская область» не равнозначна отдельному району."
    )
    branch = view[view.t10_2.lt(-20)]
    if not branch.empty:
        st.write(
            f"В нижней ветви первого запуска (координата 2 < −20) — {len(branch)} профилей. Их доля ДТП без освещения: {number(branch[color].min(), 1)}–{number(branch[color].max(), 1)}%; наездов на пешеходов: {number(branch['Доля наездов на пешеходов'].min(), 1)}–{number(branch['Доля наездов на пешеходов'].max(), 1)}%. Сходство подтверждается исходными признаками. Ветвь выделена после просмотра; это не установленный класс, и её граница не переносится на второй запуск."
        )
    st.subheader("Три главных вывода")
    st.write(
        f"1. Разница долей ДТП с погибшими без освещения и при включённом освещении — {number(tests[0]['effect'], 2)} п.п.; среди наездов на пешеходов — {number(tests[1]['effect'], 2)} п.п. Это связь в зарегистрированных ДТП, не доказательство причинности."
    )
    st.write(
        f"2. Разница среднего числа пострадавших между выходными и буднями — {number(tests[2]['effect'], 3)} чел./ДТП. Практический размер эффекта нужно отличать от статистической значимости."
    )
    grid = grid_counts(frame[frame.in_analysis_period])
    if not grid.empty:
        share = grid.nlargest(3, "ДТП")["ДТП"].sum() / grid["ДТП"].sum()
        st.write(
            f"3. Три наиболее заполненные ячейки содержат {number((share)*100, 1)}% пригодных для карты ДТП. Профили различаются по освещению и сценариям ДТП; две компоненты PCA сохраняют {number((sum(info['variance']))*100, 1)}% дисперсии. Концентрация событий не равна риску поездки."
        )
    st.write(
        "Следующий вопрос: сохраняется ли связь освещения с тяжестью при учёте типа дороги, скорости, территории и года? Нужны дорожные характеристики, транспортный поток и проверенные координаты."
    )
