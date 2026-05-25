import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

GDP_CORRELATION = 0.812
PILLAR_LABELS = {
    "talent_pillar": "Talent",
    "infrastructure_pillar": "Infrastructure",
    "research_pillar": "Research",
    "policy_pillar": "Policy",
    "adoption_pillar": "Adoption",
}
PILLAR_COLS = list(PILLAR_LABELS.keys())

st.set_page_config(page_title="AI Readiness Dashboard", layout="wide", page_icon="🤖")

@st.cache_data
def load_data():
    return pd.read_csv("data/ai_readiness_data.csv")

st.title("🤖 Global AI Readiness Dashboard")
st.caption("Advanced analytics across 106 countries · 2021–2024")

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ Run `python data_collection.py` first to generate data.")
    st.stop()

if "region" not in df.columns:
    st.warning("Data missing region column. Re-run `python data_collection.py`.")
    st.stop()

n_countries = df["country"].nunique()
st.success(f"✅ Loaded **{n_countries}** countries · **{len(df)}** records")

# Sidebar filters
st.sidebar.header("Filters")
year = st.sidebar.selectbox("Year", [2024, 2023, 2022, 2021], index=0)
regions = ["All Regions"] + sorted(df["region"].dropna().unique().tolist())
region = st.sidebar.selectbox("Region", regions)

df_year = df[df["year"] == year].copy()
if region != "All Regions":
    df_year = df_year[df_year["region"] == region]

if df_year.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# Header metrics
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Global Avg", f"{df_year['ai_readiness_score'].mean():.1f}/100")
col2.metric("Top Country", df_year.loc[df_year["ai_readiness_score"].idxmax(), "country"])
col3.metric("Countries", df_year["country"].nunique())
col4.metric("GDP Correlation", f"{GDP_CORRELATION:.3f}")
col5.metric("Year", str(year))

tab_map, tab_top, tab_gdp, tab_compare, tab_pillars = st.tabs([
    "🌍 World Map",
    "🏆 Top 20",
    "📈 GDP Correlation",
    "⚖️ Compare Countries",
    "📊 Pillars",
])

with tab_map:
    st.subheader("AI Readiness by Country")
    fig_map = px.choropleth(
        df_year,
        locations="country",
        locationmode="country names",
        color="ai_readiness_score",
        hover_name="country",
        hover_data={"region": True, "gdp_per_capita_usd": ":,.0f", "ai_readiness_score": ":.1f"},
        color_continuous_scale="Viridis",
        range_color=[20, 95],
        title=f"Global AI Readiness Map ({year})",
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=550)
    st.plotly_chart(fig_map, use_container_width=True)

    region_summary = (
        df_year.groupby("region")["ai_readiness_score"]
        .agg(["mean", "count"])
        .round(1)
        .rename(columns={"mean": "Avg Score", "count": "Countries"})
        .sort_values("Avg Score", ascending=False)
    )
    st.dataframe(region_summary, use_container_width=True)

with tab_top:
    st.subheader("Top 20 Countries by AI Readiness")
    top20 = df_year.nlargest(20, "ai_readiness_score")
    fig_top = px.bar(
        top20,
        x="ai_readiness_score",
        y="country",
        orientation="h",
        color="region",
        text="ai_readiness_score",
        labels={"ai_readiness_score": "AI Readiness Score", "country": ""},
        title=f"Top 20 — {year}" + (f" · {region}" if region != "All Regions" else ""),
    )
    fig_top.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_top.update_layout(yaxis={"categoryorder": "total ascending"}, height=700, margin=dict(l=10))
    st.plotly_chart(fig_top, use_container_width=True)

    st.dataframe(
        top20[["country", "region", "ai_readiness_score", "gdp_per_capita_usd", "internet_penetration_pct"]]
        .rename(columns={"gdp_per_capita_usd": "GDP per Capita (USD)"}),
        use_container_width=True,
        hide_index=True,
    )

with tab_gdp:
    st.subheader("GDP per Capita vs AI Readiness")
    st.info(f"**Pearson correlation: {GDP_CORRELATION:.3f}** — strong positive relationship between economic development and AI readiness.")

    fig_gdp = px.scatter(
        df_year,
        x="gdp_per_capita_usd",
        y="ai_readiness_score",
        color="region",
        size="internet_penetration_pct",
        hover_name="country",
        log_x=True,
        labels={
            "gdp_per_capita_usd": "GDP per Capita (USD, log scale)",
            "ai_readiness_score": "AI Readiness Score",
            "internet_penetration_pct": "Internet %",
        },
        title=f"GDP vs AI Readiness ({year}) · r = {GDP_CORRELATION:.3f}",
    )
    fig_gdp.update_layout(height=550)
    st.plotly_chart(fig_gdp, use_container_width=True)

    st.markdown(
        f"""
        | Metric | Value |
        |--------|-------|
        | **Correlation (r)** | **{GDP_CORRELATION:.3f}** |
        | Countries shown | {len(df_year)} |
        | Mean readiness | {df_year['ai_readiness_score'].mean():.1f} |
        | Mean GDP (USD) | {df_year['gdp_per_capita_usd'].mean():,.0f} |
        """
    )

with tab_compare:
    st.subheader("Country Comparison")
    st.caption("Select up to 5 countries to compare across key indicators.")

    all_countries = sorted(df_year["country"].unique().tolist())
    default = [c for c in ["United States", "China", "Singapore", "Germany", "India"] if c in all_countries][:3]

    selected = st.multiselect(
        "Countries (max 5)",
        options=all_countries,
        default=default,
        max_selections=5,
    )

    if not selected:
        st.info("Select at least one country to compare.")
    else:
        compare_df = df_year[df_year["country"].isin(selected)].copy()

        metrics = ["ai_readiness_score", "gdp_per_capita_usd", "internet_penetration_pct", "government_effectiveness"]
        metric_labels = ["AI Readiness", "GDP per Capita", "Internet %", "Gov. Effectiveness"]

        fig_grouped = go.Figure()
        for metric, label in zip(metrics, metric_labels):
            fig_grouped.add_trace(
                go.Bar(
                    name=label,
                    x=compare_df["country"],
                    y=compare_df[metric],
                )
            )
        fig_grouped.update_layout(
            barmode="group",
            title="Side-by-Side Comparison",
            height=420,
            xaxis_title="Country",
        )
        st.plotly_chart(fig_grouped, use_container_width=True)

        if all(c in compare_df.columns for c in PILLAR_COLS):
            pillar_melt = compare_df.melt(
                id_vars=["country"],
                value_vars=PILLAR_COLS,
                var_name="pillar",
                value_name="score",
            )
            pillar_melt["pillar"] = pillar_melt["pillar"].map(PILLAR_LABELS)

            fig_radar = px.line_polar(
                pillar_melt,
                r="score",
                theta="pillar",
                color="country",
                line_close=True,
                range_r=[0, 100],
                title="Pillar Profile Comparison",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.dataframe(
            compare_df[
                ["country", "region", "ai_readiness_score", "gdp_per_capita_usd"]
                + [c for c in PILLAR_COLS if c in compare_df.columns]
            ],
            use_container_width=True,
            hide_index=True,
        )

with tab_pillars:
    st.subheader("AI Readiness Pillars")
    st.caption("Five pillars: Talent, Infrastructure, Research, Policy, and Adoption.")

    if not all(c in df_year.columns for c in PILLAR_COLS):
        st.error("Pillar columns missing. Re-run `python data_collection.py`.")
    else:
        pillar_avg = df_year[PILLAR_COLS].mean().round(1)
        pillar_avg.index = [PILLAR_LABELS[c] for c in pillar_avg.index]

        c1, c2 = st.columns([1, 2])
        with c1:
            for name, val in pillar_avg.items():
                st.metric(name, f"{val:.1f}/100")

        with c2:
            fig_pillar_avg = px.bar(
                x=pillar_avg.index,
                y=pillar_avg.values,
                labels={"x": "Pillar", "y": "Average Score"},
                color=pillar_avg.values,
                color_continuous_scale="Blues",
                title=f"Global Pillar Averages ({year})",
            )
            fig_pillar_avg.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_pillar_avg, use_container_width=True)

        top_pillar_country = st.selectbox(
            "View pillar breakdown for country",
            options=sorted(df_year["country"].unique()),
            index=0,
        )
        row = df_year[df_year["country"] == top_pillar_country].iloc[0]
        pillar_vals = [row[c] for c in PILLAR_COLS]
        pillar_names = [PILLAR_LABELS[c] for c in PILLAR_COLS]

        fig_country_pillars = go.Figure(
            data=go.Scatterpolar(
                r=pillar_vals + [pillar_vals[0]],
                theta=pillar_names + [pillar_names[0]],
                fill="toself",
                name=top_pillar_country,
            )
        )
        fig_country_pillars.update_layout(
            polar=dict(radialaxis=dict(range=[0, 100])),
            title=f"{top_pillar_country} — Pillar Scores ({row['ai_readiness_score']:.1f} overall)",
            height=450,
        )
        st.plotly_chart(fig_country_pillars, use_container_width=True)

        pillar_rank = df_year[["country", "region"] + PILLAR_COLS].copy()
        pillar_rank.columns = ["country", "region"] + [PILLAR_LABELS[c] for c in PILLAR_COLS]
        st.dataframe(
            pillar_rank.sort_values("Talent", ascending=False).head(25),
            use_container_width=True,
            hide_index=True,
        )
