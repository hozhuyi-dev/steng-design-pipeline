"""Altair charts of design results: values on plan, force-moment curves and wall designs
"""

import math

import altair as alt
import polars as pl

WIDTH_MAX = 1200
HEIGHT_MAX = 400
SERIES = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"

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

def plot_wall_design(df_wall: pl.DataFrame, pier: str, ur_limit: float, title: str) -> alt.HConcatChart:
    """Plot a pier's vertical reinforcement and its utilisation up the building, one bar per storey

    The reinforcement is labelled where it changes.

    Args:
        df_wall: The designed piers, as get_rc_wall returns them.
        pier: The pier to plot.
        ur_limit: The utilisation limit the design used, drawn as a dashed line.
        title: The chart title.
    """

    df = (
        df_wall
        .filter(pl.col("pier") == pier)
        .sort("elev")
        .with_columns(
            elev_mid = (pl.col("elev") + pl.col("elev_below")) / 2,
            bars = (
                pl.concat_list(pl.col(f"dia_vert{k}") for k in range(6))
                .list.eval(pl.format("H{}", pl.element().filter(pl.element() > 0)))
                .list.join("+")
            ),
        )
        .with_columns(
            label = pl.format("{} @ {}", "bars", "s_vert0"),
        )
        .with_columns(
            label = pl.when(pl.col("label").ne_missing(pl.col("label").shift(1))).then(pl.col("label")),
        )
        .select("story", "elev", "elev_below", "elev_mid", "label", "reo_vert", "perc_vert", "ur_vert")
    )

    y = alt.Y("elev_below:Q").title("Elevation [m]").scale(domain=[0, df["elev"].max()], nice=False)
    tooltip = [
        alt.Tooltip("story:N"), alt.Tooltip("reo_vert:N"),
        alt.Tooltip("perc_vert:Q", format=".2f"), alt.Tooltip("ur_vert:Q", format=".2f"),
    ]
    bar = {"color": SERIES, "stroke": "white", "strokeWidth": 2, "cornerRadiusEnd": 4}

    x = alt.X("perc_vert:Q").title("Vertical reinforcement [%]").scale(domain=[0, 4.5], nice=False)
    reinforcement = (
        alt.Chart(df).mark_bar(**bar).encode(x, alt.X2(datum=0), y, alt.Y2("elev:Q"), tooltip=tooltip)
        + alt.Chart(df.filter(pl.col("label").is_not_null())).mark_text(
            align="left", dx=6, color=TEXT_PRIMARY, fontSize=11,
        ).encode(x, alt.Y("elev_mid:Q"), alt.Text("label:N"))
    ).properties(width=420, height=400)

    x = alt.X("ur_vert:Q").title("Utilisation").scale(domain=[0, 1.2], nice=False)
    utilisation = (
        alt.Chart(df).mark_bar(**bar).encode(
            x, alt.X2(datum=0), y.title(None), alt.Y2("elev:Q"), tooltip=tooltip,
        )
        + alt.Chart(pl.DataFrame({"ur_limit": [ur_limit]})).mark_rule(
            color=TEXT_SECONDARY, strokeDash=[4, 4], strokeWidth=1.5,
        ).encode(alt.X("ur_limit:Q"))
        + alt.Chart(pl.DataFrame({"ur_limit": [ur_limit], "elev": [df["elev"].max()]})).mark_text(
            align="left", baseline="top", dx=4, dy=2, color=TEXT_SECONDARY, fontSize=11,
        ).encode(alt.X("ur_limit:Q"), alt.Y("elev:Q"), text=alt.value(f"limit {ur_limit:g}"))
    ).properties(width=300, height=400)

    return (
        alt.hconcat(reinforcement, utilisation)
        .resolve_scale(y="shared")
        .properties(title=title)
        .configure_axis(gridColor="#e8e7e3", labelColor=TEXT_SECONDARY, titleColor=TEXT_SECONDARY)
        .configure_view(stroke=None)
    )

def plot_wall_overview(df_wall: pl.DataFrame, ur_limit: float, title: str) -> alt.HConcatChart:
    """Plot every pier's vertical reinforcement and utilisation by storey, one dot per pier and storey

    Args:
        df_wall: The designed piers, as get_rc_wall returns them.
        ur_limit: The utilisation limit the design used, the top of the utilisation scale's range.
        title: The chart title.
    """

    df = (
        df_wall
        .with_columns(
            elev_mid = (pl.col("elev") + pl.col("elev_below")) / 2,
            label = pl.format("{} · {} m", "pier", (pl.col("length") / 1000).round(1)),
        )
        .select("pier", "label", "story", "elev_mid", "reo_vert", "perc_vert", "ur_vert", "outputcase_vert")
    )
    x = (
        alt.X("label:N").title("Pier and length").sort(alt.EncodingSortField("pier"))
        .axis(labelAngle=0, labelExpr="split(datum.label, ' · ')", labelPadding=6)
    )
    y = alt.Y("elev_mid:Q").title("Elevation [m]").scale(domain=[0, df_wall["elev"].max()], nice=False)
    tooltip = [
        alt.Tooltip("pier:N"), alt.Tooltip("story:N"), alt.Tooltip("reo_vert:N"),
        alt.Tooltip("perc_vert:Q", format=".2f"), alt.Tooltip("ur_vert:Q", format=".2f"),
        alt.Tooltip("outputcase_vert:N"),
    ]
    dot = {"size": 260, "opacity": 1, "stroke": "#d9d8d4", "strokeWidth": 0.75}

    reinforcement = alt.Chart(df).mark_circle(**dot).encode(
        x, y,
        alt.Color("perc_vert:Q").title("Vertical [%]").scale(domain=[0, 4], range=["#cde2fb", "#0d366b"]),
        tooltip=tooltip,
    ).properties(width=300, height=400, title="Vertical reinforcement")

    utilisation = alt.Chart(df).mark_circle(**dot).encode(
        x, y.title(None),
        alt.Color("ur_vert:Q").title("Utilisation").scale(domain=[0, ur_limit], scheme="oranges"),
        tooltip=tooltip,
    ).properties(width=300, height=400, title=f"Utilisation, limit {ur_limit:g}")

    return (
        alt.hconcat(reinforcement, utilisation)
        .resolve_scale(color="independent")
        .properties(title=title)
        .configure_axis(gridColor="#e8e7e3", labelColor=TEXT_SECONDARY, titleColor=TEXT_SECONDARY)
        .configure_title(anchor="start")
        .configure_view(stroke=None)
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