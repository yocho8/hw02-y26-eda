"""Описательные графики, рассчитанные по текущему срезу дашборда."""
import pandas as pd
import plotly.express as px
import streamlit as st
import textwrap
from presentation import number, table_style


def wrap_label(value, width=18):
    return "<br>".join(textwrap.wrap(str(value), width, break_long_words=False, break_on_hyphens=False))


LIGHT_NAMES = {
    "Светлое время суток": "Светлое время",
    "Сумерки": "Сумерки",
    "В темное время суток, освещение включено": "Темно, свет включён",
    "В темное время суток, освещение не включено": "Темно, свет не включён",
    "В темное время суток, освещение отсутствует": "Темно, освещения нет",
}


def fatal_summary(frame, groups):
    """В знаменатель входят только ДТП с известным числом погибших."""
    result = frame.groupby(groups, dropna=False, observed=True).agg(
        n=("id", "size"), known=("has_deaths", "count"), fatal=("has_deaths", "sum")
    ).reset_index()
    result["percent"] = result["fatal"].div(result["known"].replace(0, float("nan"))) * 100
    return result


def show_exploration(current):
    st.caption(f"Все графики используют текущие фильтры: {number(len(current))} ДТП. Доли считаются среди зарегистрированных ДТП, а не среди поездок.")
    st.subheader("Какие типы ДТП встречаются чаще")
    counts = current.category.fillna("Не указано").value_counts().rename_axis("Тип ДТП").reset_index(name="ДТП")
    display = counts.head(8).copy()
    if len(counts) > 8:
        display.loc[len(display)] = ["Остальные типы", counts.iloc[8:]["ДТП"].sum()]
    display["Подпись"] = display["ДТП"].map(number)
    fig = px.bar(display, x="Тип ДТП", y="ДТП", text="Подпись")
    fig.update_xaxes(tickmode="array", tickvals=display["Тип ДТП"], ticktext=[wrap_label(t, 13) for t in display["Тип ДТП"]], tickangle=0, tickfont=dict(size=11))
    fig.update_yaxes(tickformat=",d", rangemode="tozero")
    fig.update_layout(height=480, margin=dict(l=10, r=25, t=35, b=110))
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("Восемь самых частых типов; остальные объединены, чтобы подписи читались. Сумма столбцов равна числу ДТП в срезе.")
    with st.expander("Все типы ДТП — точные числа"):
        st.dataframe(counts, hide_index=True)
    top = counts.iloc[0]
    st.write(f"Самый частый тип в срезе — «{top['Тип ДТП']}»: {number(top['ДТП'])} из {number(len(current))} ДТП ({number((top['ДТП']/len(current))*100, 1)}%).")

    st.subheader("Сколько людей пострадало в одном ДТП")
    values = current.casualties_count.dropna().astype(int)
    if values.empty:
        st.info("Нет известных значений числа пострадавших.")
    else:
        distribution = values.value_counts().sort_index().rename_axis("Пострадавших, человек").reset_index(name="ДТП")
        distribution["Накопленная доля ДТП, %"] = distribution["ДТП"].cumsum()/len(values)*100
        distribution = pd.concat([pd.DataFrame({"Пострадавших, человек": [0], "ДТП": [0], "Накопленная доля ДТП, %": [0.0]}), distribution], ignore_index=True) if values.min() > 0 else distribution
        fig = px.line(distribution, x="Пострадавших, человек", y="Накопленная доля ДТП, %", markers=True, line_shape="hv", hover_data=["ДТП"])
        fig.update_yaxes(range=[0, 102])
        st.plotly_chart(fig, width="stretch")
        st.write(f"Не более одного пострадавшего: {number(values.le(1).sum())} из {number(len(values))} ДТП ({number(values.le(1).mean()*100, 2)}%).")
        st.caption(f"ECDF: доля ДТП, в которых число раненых плюс погибших не превышает значение по оси X. Известных значений: {number(len(values))}; пропусков: {number(len(current)-len(values))}. Максимум {values.max()} человек сохранён.")

    st.subheader("Когда происходят ДТП: день недели и час")
    grid = pd.crosstab(current.dayofweek, current.hour).reindex(index=range(7), columns=range(24), fill_value=0)
    fig = px.imshow(grid.to_numpy(), x=list(range(24)), y=["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"],
                    labels=dict(x="Час", y="День недели", color="ДТП"), aspect="auto", color_continuous_scale="Blues")
    st.plotly_chart(fig, width="stretch")
    st.caption("Цвет — число ДТП за весь выбранный период; показатель не нормирован на транспортный поток или число соответствующих дней недели.")
    peak = grid.stack().idxmax()
    st.write(f"Наибольшее число записей в ячейке: {['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'][peak[0]]}, {peak[1]}:00–{peak[1]}:59 — {number(grid.loc[peak])} ДТП. Это мотивирует сравнение последствий в будни и выходные.")
    st.subheader("Пострадавшие в будни и выходные")
    groups = current.assign(Группа=current.is_weekend.map({False: "Будни", True: "Выходные"})).groupby("Группа").casualties_count.agg(
        **{"ДТП с известным числом пострадавших": "count", "Среднее, чел./ДТП": "mean", "Медиана, человек": "median"}).reset_index()
    fig = px.bar(groups, x="Группа", y="Среднее, чел./ДТП", hover_data=["ДТП с известным числом пострадавших", "Медиана, человек"], text_auto=".3f")
    st.plotly_chart(fig, width="stretch")
    st.caption("Среднее число раненых плюс погибших на ДТП. Будни — понедельник–пятница, выходные — суббота и воскресенье; праздники отдельно не выделены. Здесь описание текущего среза; тест Уэлча и интервалы для 2015–2025 — во вкладке статистики.")
    st.dataframe(table_style(groups.set_index("Группа"), 3), width="stretch")

    st.subheader("Освещение и доля ДТП с погибшими")
    lighting = fatal_summary(current, ["light"])
    lighting["Освещение"] = lighting.light.map(LIGHT_NAMES).fillna("Не указано")
    lighting["Подпись"] = lighting.apply(lambda r: f"{number(r['percent'], 1)}%<br>{number(r['fatal'])}/{number(r['known'])}" if pd.notna(r['percent']) else "Нет данных", axis=1)
    labels = dict(percent="ДТП с погибшими, %", n="Всего ДТП", known="С известным числом погибших", fatal="ДТП с погибшими")
    lighting = lighting.sort_values("percent", ascending=False)
    fig = px.bar(lighting, x="Освещение", y="percent", text="Подпись", hover_data=["n", "known", "fatal"], labels=labels)
    fig.update_traces(textposition="outside", cliponaxis=False)
    maximum = lighting.percent.max()
    fig.update_yaxes(range=[0, min(100, max(5, maximum*1.4))] if pd.notna(maximum) else [0, 100])
    fig.update_xaxes(tickmode="array", tickvals=lighting["Освещение"], ticktext=[wrap_label(t, 16) for t in lighting["Освещение"]], tickangle=0)
    fig.update_layout(margin=dict(l=10, r=25, t=35, b=80), height=440)
    st.plotly_chart(fig, width="stretch")
    st.caption("Подпись: процент · ДТП с погибшими / ДТП с известным числом погибших в группе. Маленькие группы дают нестабильные доли. Это описание текущего среза; статистические проверки фиксированного периода показаны отдельно. Причинный эффект не установлен.")

    st.subheader("Один тип ДТП — два условия освещения")
    st.write("Сравниваем только ДТП в темноте. Для каждого типа рядом стоят два столбца: со включённым освещением и без освещения. Высота — процент ДТП с погибшими среди ДТП этого типа в соответствующей группе. Это помогает понять, связано ли общее различие лишь с разным составом ДТП.")
    comparison = current[current.light.isin([
        "В темное время суток, освещение включено", "В темное время суток, освещение отсутствует"
    ])].copy()
    if comparison.empty:
        st.info("В срезе нет ДТП для сравнения включённого и отсутствующего освещения.")
        return
    # Выбираем пять наиболее частых типов сразу по обеим сравниваемым группам.
    top_types = comparison.category.value_counts().head(5).index
    grouped = fatal_summary(comparison[comparison.category.isin(top_types)], ["category", "light"])
    grouped["Освещение"] = grouped.light.map(LIGHT_NAMES)
    grouped["Подпись"] = grouped.percent.map(lambda v: f"{number(v, 1)}%" if pd.notna(v) else "Нет данных")
    fig = px.bar(grouped, x="category", y="percent", color="Освещение", barmode="group", text="Подпись",
                     hover_data=["n", "known", "fatal"], labels={**labels, "category": "Тип ДТП"},
                     color_discrete_map={"Темно, свет включён": "#3579B8", "Темно, освещения нет": "#C18A16"})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_xaxes(tickmode="array", tickvals=list(top_types), ticktext=[wrap_label(t) for t in top_types], tickangle=0)
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(height=450, legend=dict(orientation="h", y=1.2), margin=dict(l=10, r=20, t=65, b=70))
    st.plotly_chart(fig, width="stretch")
    st.caption("Пять наиболее частых типов в двух сравниваемых группах текущего среза. Отсутствующая группа не заменяется нулём. Числитель и знаменатель доступны при наведении и в таблице ниже.")
    with st.expander("Числа для сравнения типов ДТП"):
        st.dataframe(grouped[["category", "Освещение", "known", "fatal", "percent"]].rename(columns={"category": "Тип ДТП", "known": "ДТП с известным числом погибших", "fatal": "ДТП с погибшими", "percent": "Доля, %"}).style.format({"Доля, %": lambda x: number(x, 2)}), hide_index=True)
    example = grouped[grouped.category.eq("Наезд на пешехода")]
    if len(example) == 2:
        parts = [f"{r['Освещение'].lower()}: {number(r['fatal'])}/{number(r['known'])} = {number(r['percent'], 1)}%" for _, r in example.iterrows()]
        st.info("Пример чтения для наездов на пешеходов: " + "; ".join(parts) + ". Сравниваются доли, а не общее число происшествий. Это не доказывает причинный эффект освещения.")
