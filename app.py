"""Tender Radar: open South African public-sector tenders, ranked for your business."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from radar import brief as brief_mod
from radar.match import Profile, score
from radar.pipeline import MIN_SCORE, now_sa, open_tenders
from radar.store import BRIEFS, PROFILES, STATUS, load_json, load_tenders
from radar.taxonomy import OTHER, SECTORS
from radar.ui import ACCENT, CSS, MUTED_BAR, bar_layout, esc, fmt_date, kpis

st.set_page_config(page_title="Tender Radar · South Africa", page_icon=":material/radar:",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

PROVINCES = ["Eastern Cape", "Free State", "Gauteng", "KwaZulu-Natal", "Limpopo",
             "Mpumalanga", "National", "North West", "Northern Cape", "Western Cape"]


@st.cache_data(ttl=3600, show_spinner=False)
def load():
    tenders = load_tenders()
    at = now_sa()
    live = open_tenders(tenders, at).copy()
    live["days_left"] = ((live["closes"] - at).dt.total_seconds() / 86400).round(1)
    return tenders, live, at, load_json(STATUS, {}), load_json(BRIEFS, {})


@st.cache_resource(show_spinner=False)
def similarity_index(ids: tuple):
    tenders = load_tenders().set_index("id").loc[list(ids)]
    text = (tenders["description"] + " " + tenders["category"]).str.lower()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                          stop_words="english")
    return vec, vec.fit_transform(text)


tenders, live, at, status, briefs = load()
if "briefs" not in st.session_state:
    st.session_state.briefs = dict(briefs)

# --- Sidebar: business profile ---------------------------------------------------------
presets = load_json(PROFILES, [])
with st.sidebar:
    st.markdown("## Your business")
    names = ["Custom"] + [p["name"] for p in presets]
    choice = st.selectbox("Start from a sample profile", names, index=1 if presets else 0)
    base = next((p for p in presets if p["name"] == choice), {})
    sectors = st.multiselect("Sectors you work in", SECTORS,
                             default=[s for s in base.get("sectors", []) if s in SECTORS],
                             key=f"sec_{choice}")
    provinces = st.multiselect("Provinces you deliver to", PROVINCES,
                               default=base.get("provinces", []), key=f"prov_{choice}",
                               help="National tenders still score well when you pick provinces.")
    keywords = st.text_area("What you do, one phrase per line",
                            value="\n".join(base.get("keywords", [])), height=110,
                            key=f"kw_{choice}",
                            help="Short phrases a buyer would use, e.g. 'office cleaning'.")
    exclude = st.text_input("Exclude tenders mentioning", value=", ".join(base.get("exclude", [])),
                            key=f"ex_{choice}", help="Comma-separated.")
    min_days = st.slider("Hide tenders closing within (days)", 0, 14, 2,
                         help="Leaves out tenders you would not have time to prepare for.")
    st.markdown('<div class="small">Profiles used for the daily digest live in '
                '<code>profiles.json</code>.</div>', unsafe_allow_html=True)

profile = Profile(name=choice, sectors=sectors, provinces=provinces,
                  keywords=[k for k in keywords.splitlines() if k.strip()],
                  exclude=[e for e in exclude.split(",") if e.strip()])
has_profile = bool(sectors or provinces or profile.keywords)
pool = live[live["days_left"] >= min_days]
ranked = score(pool, profile) if has_profile else pool.assign(score=np.nan, reason="")
matches = ranked[ranked["score"] >= MIN_SCORE] if has_profile else ranked.iloc[0:0]

# --- Header -----------------------------------------------------------------------------
st.markdown(
    '<div class="brand"><div class="mark">Tender<span>Radar</span></div>'
    '<div class="tag">South African public-sector tenders</div></div>'
    '<div class="lede">Every open tender on the National Treasury eTenders portal, sorted '
    'into sectors by a machine-learning classifier and ranked against your business '
    f'profile. Data updated {esc(status.get("updated", "recently"))} (SAST).</div>',
    unsafe_allow_html=True)

week_ago = at.normalize() - pd.Timedelta(days=7)
st.markdown(kpis([
    ("Open tenders", f"{len(live):,}", "across all buyers"),
    ("Match your profile", f"{len(matches):,}" if has_profile else "–",
     f"score {MIN_SCORE}+ out of 100" if has_profile else "set a profile on the left"),
    ("New in the last 7 days", f"{int((live['first_seen'] >= week_ago).sum()):,}",
     f"{int((matches['first_seen'] >= week_ago).sum())} of them match" if has_profile else ""),
    ("Closing in 7 days", f"{int((live['days_left'] <= 7).sum()):,}",
     f"{int((matches['days_left'] <= 7).sum())} of them match" if has_profile else ""),
]), unsafe_allow_html=True)


# --- Tender detail ----------------------------------------------------------------------
def render_brief(b: dict) -> None:
    parts = [f"<p>{esc(b.get('summary'))}</p>"] if b.get("summary") else []
    facts = [("CIDB grading", b.get("cidb_grading")), ("Contract period", b.get("contract_period")),
             ("Preference points", b.get("preference")), ("Evaluation", b.get("evaluation")),
             ("Estimated value", b.get("estimated_value"))]
    facts = [(k, v) for k, v in facts if v]
    if facts:
        parts.append('<div class="meta">' + "".join(
            f"<div><b>{esc(k)}</b>{esc(v)}</div>" for k, v in facts) + "</div>")
    for title, key in (("Scope", "scope"), ("To qualify", "requirements"),
                       ("Watch out for", "watch_out")):
        if b.get(key):
            parts.append(f'<div class="section-label">{title}</div><ul>'
                         + "".join(f"<li>{esc(x)}</li>" for x in b[key]) + "</ul>")
    parts.append(f'<div class="small">Generated {esc(b.get("generated"))} by '
                 f'{esc(b.get("model"))} from {esc(b.get("source"))}. Check the official '
                 'document before bidding.</div>')
    st.markdown('<div class="brief">' + "".join(parts) + "</div>", unsafe_allow_html=True)


def render_detail(row: pd.Series) -> None:
    pills = [f'<span class="pill accent">{esc(row["sector"])}</span>']
    if row.get("sector_source") == "model":
        pills.append(f'<span class="pill">classifier {row["sector_confidence"]:.0%}</span>')
    if pd.notna(row.get("score")):
        pills.append(f'<span class="pill">fit {row["score"]:.0f}/100</span>')
    if row["days_left"] <= 7:
        pills.append(f'<span class="pill warn">{row["days_left"]:.0f} days left</span>')
    if row["esubmission"]:
        pills.append('<span class="pill">e-submission</span>')

    briefing = ""
    if row["briefing_compulsory"]:
        when = fmt_date(row["briefing_date"]) if pd.notna(row["briefing_date"]) else "date in document"
        missed = pd.notna(row["briefing_date"]) and row["briefing_date"] < at
        briefing = (f'<div class="notice"><b>Compulsory briefing session</b>: {esc(when)}'
                    f'{" (already held, so only attendees can bid)" if missed else ""}'
                    f'{"<br>" + esc(row["briefing_venue"]) if row["briefing_venue"] else ""}</div>')

    docs = json.loads(row["documents"] or "[]")
    doc_html = "".join(f'<a href="{esc(d["url"])}" target="_blank">{esc(d["name"])}</a>'
                       for d in docs) or '<span class="small">No documents attached</span>'
    st.markdown(f"""
<div class="panel">
  <div class="small">{esc(row["reference"])} · {esc(row["type"])}</div>
  <h3>{esc(row["description"])}</h3>
  <div style="margin-top:8px">{"".join(pills)}</div>
  <div class="meta">
    <div><b>Buyer</b>{esc(row["buyer"])}</div>
    <div><b>Province</b>{esc(row["province"])}</div>
    <div><b>Closes</b>{esc(fmt_date(row["closes"]))}</div>
    <div><b>Published</b>{esc(fmt_date(row["published"], with_time=False))}</div>
    <div><b>Contact</b>{esc(row["contact"]) or "Not stated"}<br>{esc(row["email"])}<br>{esc(row["phone"])}</div>
    <div><b>Delivery / site</b>{esc(row["delivery"]) or "Not stated"}</div>
  </div>
  {briefing}
  {f'<div class="section-label">Conditions</div><div class="small">{esc(row["conditions"])}</div>' if row["conditions"] else ""}
  <div class="section-label">Documents</div><div class="doclinks">{doc_html}</div>
</div>""", unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown('<div class="section-label">AI brief</div>', unsafe_allow_html=True)
        key = str(row["id"])
        if key in st.session_state.briefs:
            render_brief(st.session_state.briefs[key])
        elif brief_mod.available_provider():
            if st.button("Read the tender document and summarise it", key=f"brief_{key}"):
                with st.spinner("Reading the tender document…"):
                    try:
                        st.session_state.briefs[key] = brief_mod.generate(row.to_dict())
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not generate a brief: {exc}")
        else:
            st.markdown('<div class="small">Briefs are generated each morning for new '
                        'tenders that match a saved profile. Add a free GEMINI_API_KEY or '
                        'GROQ_API_KEY secret to generate them on demand.</div>',
                        unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-label">Similar tenders</div>', unsafe_allow_html=True)
        ids = tuple(tenders["id"])
        vec, matrix = similarity_index(ids)
        pos = ids.index(row["id"])
        sims = linear_kernel(matrix[pos], matrix).ravel()
        sims[pos] = 0
        top = np.argsort(-sims)[:5]
        similar = tenders.iloc[top].assign(similarity=sims[top])
        similar = similar[similar["similarity"] > 0.15]
        if similar.empty:
            st.markdown('<div class="small">Nothing similar in the archive yet.</div>',
                        unsafe_allow_html=True)
        for s in similar.itertuples():
            state = "open" if pd.notna(s.closes) and s.closes > at else "closed"
            st.markdown(f'<div style="margin-bottom:10px"><div style="font-size:.85rem">'
                        f'{esc(s.description[:140])}</div><div class="small">{esc(s.buyer)} · '
                        f'{esc(s.province)} · {state} {esc(fmt_date(s.closes, False))}</div></div>',
                        unsafe_allow_html=True)


def tender_table(df: pd.DataFrame, key: str, show_score: bool):
    view = pd.DataFrame({
        "Fit": df["score"],
        "Tender": df["description"].str.slice(0, 200),
        "Buyer": df["buyer"],
        "Province": df["province"],
        "Sector": df["sector"],
        "Closes": df["closes"],
        "Days left": df["days_left"],
        "Briefing": np.where(df["briefing_compulsory"], "Compulsory",
                             np.where(df["briefing"], "Optional", "")),
    })
    if not show_score:
        view = view.drop(columns="Fit")
    event = st.dataframe(
        view, key=key, hide_index=True, width="stretch", height=430,
        on_select="rerun", selection_mode="single-row",
        column_config={
            "Fit": st.column_config.ProgressColumn("Fit", min_value=0, max_value=100,
                                                   format="%d", width="small"),
            "Tender": st.column_config.TextColumn(width="large"),
            "Buyer": st.column_config.TextColumn(width="medium"),
            "Province": st.column_config.TextColumn(width="small"),
            "Sector": st.column_config.TextColumn(width="medium"),
            "Briefing": st.column_config.TextColumn(width="small"),
            "Closes": st.column_config.DatetimeColumn(format="D MMM YYYY, HH:mm"),
            "Days left": st.column_config.NumberColumn(format="%.0f", width="small"),
        },
    )
    rows = event.selection.rows if event and event.selection else []
    return df.iloc[rows[0]] if rows else None


def show_selected(selected, fallback_id=None):
    if selected is not None:
        render_detail(selected)
    elif fallback_id is not None and fallback_id in set(ranked["id"]):
        render_detail(ranked[ranked["id"] == fallback_id].iloc[0])
    else:
        st.markdown('<div class="small">Select a row to see the full tender, contact '
                    'details, documents and AI brief.</div>', unsafe_allow_html=True)


linked = st.query_params.get("tender")
linked_id = int(linked) if linked and linked.isdigit() else None

tab_matches, tab_all, tab_market, tab_method = st.tabs(
    ["Your matches", "All open tenders", "Market overview", "How it works"])

with tab_matches:
    if not has_profile:
        st.info("Choose sectors, provinces or keywords in the sidebar to rank tenders for your business.")
    elif matches.empty:
        st.info("No open tenders reach a fit score of "
                f"{MIN_SCORE}. Try adding sectors or broader keywords.")
    else:
        sort = st.radio("Sort by", ["Best fit", "Closing soonest", "Newest"], horizontal=True,
                        label_visibility="collapsed")
        view = {"Best fit": matches,
                "Closing soonest": matches.sort_values("closes"),
                "Newest": matches.sort_values("first_seen", ascending=False)}[sort]
        picked = tender_table(view, "matches", show_score=True)
        show_selected(picked, linked_id)

with tab_all:
    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    query = c1.text_input("Search", placeholder="e.g. borehole, laptops, Polokwane")
    f_sector = c2.multiselect("Sector", SECTORS + [OTHER])
    f_province = c3.multiselect("Province", PROVINCES)
    f_type = c4.multiselect("Type", sorted(live["type"].dropna().unique()))
    df = ranked.sort_values("closes")
    if query:
        hay = (df["description"] + " " + df["buyer"] + " " + df["reference"] + " "
               + df["delivery"]).str.lower()
        for term in query.lower().split():
            df = df[hay.loc[df.index].str.contains(term, regex=False)]
    if f_sector:
        df = df[df["sector"].isin(f_sector)]
    if f_province:
        df = df[df["province"].isin(f_province)]
    if f_type:
        df = df[df["type"].isin(f_type)]
    st.markdown(f'<div class="small">{len(df):,} tenders</div>', unsafe_allow_html=True)
    picked = tender_table(df, "all", show_score=has_profile)
    show_selected(picked)

with tab_market:
    st.markdown('<div class="small">Open tenders right now. Bars show counts; hover for exact values.</div>',
                unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown('<div class="chart-title">By sector</div>', unsafe_allow_html=True)
        # Plotly draws the first row at the bottom. "Other" is not a sector, so it sits
        # below the ranked sectors, in grey.
        counts = live["sector"].value_counts()
        counts = pd.concat([counts.reindex([OTHER]).dropna(),
                            counts.drop(OTHER, errors="ignore").sort_values()])
        fig = px.bar(x=counts.values, y=counts.index, orientation="h")
        fig.update_traces(hovertemplate="%{y}: %{x} open tenders<extra></extra>")
        fig = bar_layout(fig, 470)
        fig.update_traces(marker_color=[MUTED_BAR if s == OTHER else ACCENT for s in counts.index])
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with b:
        st.markdown('<div class="chart-title">By province</div>', unsafe_allow_html=True)
        counts = live["province"].value_counts().sort_values()
        fig = px.bar(x=counts.values, y=counts.index, orientation="h")
        fig.update_traces(hovertemplate="%{y}: %{x} open tenders<extra></extra>")
        st.plotly_chart(bar_layout(fig, 470), width="stretch", config={"displayModeBar": False})

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown('<div class="chart-title">Closing dates, next 30 days</div>', unsafe_allow_html=True)
        days = live[live["days_left"] <= 30]["closes"].dt.normalize().value_counts().sort_index()
        days = days.reindex(pd.date_range(at.normalize(), at.normalize() + pd.Timedelta(days=30)),
                            fill_value=0)
        fig = px.bar(x=days.index, y=days.values)
        fig.update_traces(hovertemplate="%{x|%a %d %b}: %{y} tenders close<extra></extra>")
        fig = bar_layout(fig, 360)
        fig.update_layout(bargap=0.15)
        fig.update_xaxes(showgrid=False, tickformat="%d %b")
        fig.update_yaxes(showgrid=True, gridcolor="#f0f1f3", tickfont=dict(color="#6b7280"))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with b:
        st.markdown('<div class="chart-title">Most active buyers</div>', unsafe_allow_html=True)
        counts = live["buyer"].value_counts().head(12).sort_values()
        fig = px.bar(x=counts.values, y=[n if len(n) < 40 else n[:38] + "…" for n in counts.index],
                     orientation="h")
        fig.update_traces(hovertemplate="%{y}: %{x} open tenders<extra></extra>")
        fig = bar_layout(fig, 360)
        fig.update_xaxes(tickangle=0, nticks=5)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    briefing_share = live["briefing_compulsory"].mean()
    rfq_share = live["type"].eq("Request for Quotation").mean()
    st.markdown(f'<div class="small">{briefing_share:.0%} of open tenders require attendance '
                f'at a compulsory briefing session; {rfq_share:.0%} are requests for quotation, '
                'which usually have shorter deadlines and lower values.</div>',
                unsafe_allow_html=True)

with tab_method:
    clf = status.get("classifier", {})
    src = status.get("sector_sources", {})
    total = sum(src.values()) or 1
    st.markdown(f"""
#### Data
Tender listings come from the public [eTenders portal](https://www.etenders.gov.za) run by
National Treasury. A scheduled job collects every advertised tender each weekday morning
and keeps closed tenders for six months, so the archive grows over time.

#### Sector classifier
eTenders does not give tenders a usable sector: the buyer-selected category mixes two
coding systems and is often just "Services: General". Tender Radar assigns one of
{len(SECTORS)} sectors in two steps:

1. **Keyword rules** label tenders whose description clearly names the work
   ({src.get("rules", 0) / total:.0%} of tenders).
2. **A TF-IDF + logistic regression model**, trained on
   {clf.get("training_examples", 0):,} rule-labelled examples drawn from
   {clf.get("corpus_size", 0):,} current and historical listings, labels the rest
   ({src.get("model", 0) / total:.0%}). The rest, where the model is not confident, stay
   in "Other".

During training the rule keywords are removed from the text, so the model has to learn
from context such as the buyer, the category and the surrounding words. That matches
the tenders it is actually used on, which contain none of the keywords. On 5-fold
cross-validation it reaches **{clf.get("cv_accuracy", 0):.0%} accuracy** and a
**macro F1 of {clf.get("cv_macro_f1", 0):.2f}**.

#### Fit score
Each open tender gets a score from 0 to 100 for your profile: 40% sector match (half
credit when it is the model's second choice), 30% similarity between your phrases and the
tender text, and 30% province (national tenders get partial credit). Tenders scoring
{MIN_SCORE} or more count as matches.

#### AI briefs and alerts
For new matches, the job downloads the tender document and asks a free-tier LLM
(Google Gemini or Groq) to pull out the scope, CIDB grading, preference points,
qualification requirements and anything easy to miss. A digest of new matches and
upcoming deadlines is sent by Telegram or email. Everything runs on free services:
GitHub Actions, Streamlit Community Cloud and free LLM tiers.
""")
    if clf.get("per_sector_f1"):
        perf = (pd.Series(clf["per_sector_f1"], name="F1 score").sort_values(ascending=False)
                .rename_axis("Sector").reset_index())
        st.dataframe(perf, hide_index=True, width="content", height=36 * (len(perf) + 1) + 4,
                     column_config={"F1 score": st.column_config.NumberColumn(format="%.2f")})
    st.markdown('<div class="small">Not affiliated with National Treasury. Always read the '
                'official tender document before bidding.</div>', unsafe_allow_html=True)
