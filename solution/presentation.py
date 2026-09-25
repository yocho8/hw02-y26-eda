"""Общее форматирование чисел и настройки Plotly для русскоязычного дашборда."""
import math
import plotly.io as pio
import plotly.graph_objects as go


def number(value, digits=0):
    if value is None or not math.isfinite(float(value)):
        return "нет данных"
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def configure_charts():
    template = go.layout.Template(layout=dict(
        separators=", ",
        xaxis=dict(exponentformat="none", showexponent="none"),
        yaxis=dict(exponentformat="none", showexponent="none"),
    ))
    pio.templates["readable_ru"] = template
    pio.templates.default = "plotly+readable_ru"


def table_style(frame, digits=2):
    numeric = frame.select_dtypes(include="number").columns
    return frame.style.format({c: lambda x: number(x, 0 if float(x).is_integer() else digits) for c in numeric}, na_rep="нет данных")
