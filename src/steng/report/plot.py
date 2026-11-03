"""Functions built on altair to simplify plotting in pynb

Functions:
"""

import math

import altair as alt
import polars as pl

WIDTH_MAX = 1200
HEIGHT_MAX = 400

def plot_plan_circle(df, color, scheme, tooltip, title, text_max=None, text_min=None, domain="unaggregated"):
    """Plot on plan given the value"""

    width, height = _calculate_dimensions(df)

    chart = (
        alt.Chart(df).mark_circle().encode(
            alt.X("x:Q").title("x-coordinate [m]"),
            alt.Y("y:Q").title("y-coordinate [m]"),
            alt.Color(color).scale(scheme=scheme, domain=domain),
            alt.Tooltip(tooltip)
        )
    )

    if text_max is not None:
        chart += (
            alt.Chart(df.select(pl.all().sort_by(text_max[:-2]).last())).mark_text().encode(
                alt.X("x:Q").title("x-coordinate [m]"),
                alt.Y("y:Q").title("y-coordinate [m]"),
                text=alt.Text(text_max, format=".3f"),
            )
        )

    if text_min is not None:
        chart += (
            alt.Chart(df.select(pl.all().sort_by(text_min[:-2]).first())).mark_text().encode(
                alt.X("x:Q").title("x-coordinate [m]"),
                alt.Y("y:Q").title("y-coordinate [m]"),
                text=alt.Text(text_min, format=".3f"),
            )
        )

    return (
        chart
        .properties(width=width, height=height, title=title)
        .configure_scale(zero=False)
        .configure_axisX(tickMinStep=1)
    )

def plot_force_moment_curve(df_curve, color, scheme, tooltip, title):
    """Plot PM curve"""

    chart = (
        alt.Chart(df_curve).mark_line().encode(
            alt.X("moment:Q",sort=None).title("Moment [kNm]"),
            alt.Y("force:Q",sort=None).scale(reverse=True).title("Force [kN]"),
            alt.Color(color).scale(scheme=scheme),
            alt.Tooltip(tooltip),
            alt.Order("force:Q", sort="ascending")
        )
    )

    chart += (
        alt.Chart(df_curve).mark_circle().encode(
            alt.X("moment:Q",sort=None).title("Moment [kNm]"),
            alt.Y("force:Q",sort=None).scale(reverse=True).title("Force [kN]"),
            alt.Color(color).scale(scheme=scheme),
            alt.Tooltip(tooltip),
        )
    )

    return (
        chart
        .properties(width=WIDTH_MAX, title=title)
        .configure_scale(zero=False)
        .configure_axisX(tickMinStep=1)
    )

def plot_force_moment_curve_with_point(
        df_curve, color_curve, scheme_curve, tooltip_curve, 
        df_reaction, color_reaction, scheme_reaction, tooltip_reaction, 
        text,
        title
    ):

    chart = (
        alt.Chart(df_curve).mark_line().encode(
            alt.X("moment:Q",sort=None).title("Moment [kNm]"),
            alt.Y("force:Q",sort=None).scale(reverse=True).title("Force [kN]"),
            alt.Color(color_curve).scale(scheme=scheme_curve),
            alt.Tooltip(tooltip_curve),
            alt.Order("force:Q", sort="ascending")
        )
    )

    if df_reaction.shape[0] > 3000:
        n_unique_diameter = len(
            df_reaction.select("dia").unique().to_series().to_list()
        )
        df_reaction = (
            df_reaction
            .with_columns(
                length = (pl.col("force").pow(2) + pl.col("moment").pow(2)).sqrt()
            )
            .select(
                pl.all()
                .top_k_by("length", k=math.floor(1000 / n_unique_diameter))
                .over("dia", mapping_strategy="explode")
            )
        )
    
    chart += (
        alt.Chart(df_reaction).mark_circle().encode(
            alt.X("moment:Q").title("Moment [kNm]"),
            alt.Y("force:Q").scale(reverse=True).title("Force [kN]"),
            alt.Color(color_reaction).scale(scheme=scheme_reaction),
            alt.Tooltip(tooltip_reaction)
        )
    )

    chart += (
        alt.Chart(df_reaction.select(pl.all().sort_by(text[:-2]).last())).mark_text().encode(
            alt.X("moment:Q",sort=None).title("Moment [kNm]"),
            alt.Y("force:Q",sort=None).scale(reverse=True).title("Force [kN]"),
            text=alt.Text(text, format=".2f"),
        )
    )

    return (
        chart
        .properties(width=WIDTH_MAX, title=title)
        .configure_scale(zero=False)
        .configure_axisX(tickMinStep=1)
    )

def _calculate_height(df):
    xmin = df.select("x").min().to_series().to_list()[0]
    xmax = df.select("x").max().to_series().to_list()[0]
    ymin = df.select("y").min().to_series().to_list()[0]
    ymax = df.select("y").max().to_series().to_list()[0]

    return WIDTH_MAX / (xmax - xmin) * (ymax - ymin)

def _calculate_dimensions(df):
    xmin = df.select("x").min().to_series().to_list()[0]
    xmax = df.select("x").max().to_series().to_list()[0]
    ymin = df.select("y").min().to_series().to_list()[0]
    ymax = df.select("y").max().to_series().to_list()[0]

    if (WIDTH_MAX / (xmax - xmin) * (ymax - ymin)) > HEIGHT_MAX:
        return (
            HEIGHT_MAX / (ymax - ymin) * (xmax - xmin),
            HEIGHT_MAX
        )

    return (
        WIDTH_MAX,
        WIDTH_MAX / (xmax - xmin) * (ymax - ymin)
    )