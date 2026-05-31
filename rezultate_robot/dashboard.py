"""
Dashboard Streamlit pentru monitorizarea conversatiilor robotului telefonic.

Instalare:
    pip install streamlit pandas plotly

Rulare:
    streamlit run dashboard.py
"""

import os
import json
import glob
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path

# ─── CONFIGURARE ─────────────────────────────────────────────────────────────

RESULTS_DIR = "/Users/antoniadumitru/Desktop/facultate/Disertatie/rezultate_robot"
st.set_page_config(
    page_title="Robot Telefonic — Dashboard",
    page_icon="📞",
    layout="wide"
)

# ─── INCARCARE DATE ───────────────────────────────────────────────────────────

@st.cache_data(ttl=30)  # refresh la 30 secunde
def incarca_conversatii(results_dir):
    fisiere = glob.glob(os.path.join(results_dir, "conversatie_*.json"))
    conversatii = []
    for f in sorted(fisiere, reverse=True):
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
            analiza = data.get("analiza", {})
            conversatii.append({
                "id":           data.get("id", Path(f).stem),
                "domeniu":      data.get("domeniu", "necunoscut"),
                "dificultate":  data.get("dificultate", "necunoscut"),
                "timestamp":    data.get("timestamp", ""),
                "intentie":     analiza.get("intentie", "necunoscut"),
                "satisfactie":  analiza.get("satisfactie", "necunoscut"),
                "rezumat":      analiza.get("rezumat", ""),
                "nr_replici":   len(data.get("conversatie", [])),
                "conversatie":  data.get("conversatie", []),
                "fisier":       f,
            })
        except Exception:
            pass
    return conversatii

# ─── HEADER ───────────────────────────────────────────────────────────────────

st.title("📞 Robot Telefonic — Dashboard Monitorizare")
st.markdown("---")

# ─── INCARCARE ────────────────────────────────────────────────────────────────

if not os.path.exists(RESULTS_DIR):
    st.warning(f"Folderul `{RESULTS_DIR}` nu există. Asigură-te că robotul a salvat conversații.")
    st.stop()

conversatii = incarca_conversatii(RESULTS_DIR)

if not conversatii:
    st.info("Nu există conversații salvate încă. Pornește robotul și efectuează câteva apeluri.")
    st.stop()

df = pd.DataFrame(conversatii)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["data"] = df["timestamp"].dt.date

# ─── SIDEBAR FILTRE ───────────────────────────────────────────────────────────

st.sidebar.header("🔍 Filtre")

domenii_disponibile = ["Toate"] + sorted(df["domeniu"].unique().tolist())
domeniu_sel = st.sidebar.selectbox("Domeniu", domenii_disponibile)

satisfactii_disponibile = ["Toate"] + sorted(df["satisfactie"].dropna().unique().tolist())
satisfactie_sel = st.sidebar.selectbox("Satisfacție", satisfactii_disponibile)

dificultati_disponibile = ["Toate"] + sorted(df["dificultate"].unique().tolist())
dificultate_sel = st.sidebar.selectbox("Dificultate", dificultati_disponibile)

if df["data"].notna().any():
    data_min = df["data"].min()
    data_max = df["data"].max()
    interval_date = st.sidebar.date_input(
        "Interval date",
        value=(data_min, data_max),
        min_value=data_min,
        max_value=data_max
    )
else:
    interval_date = None

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reîncarcă date"):
    st.cache_data.clear()
    st.rerun()

# ─── APLICARE FILTRE ──────────────────────────────────────────────────────────

df_filtrat = df.copy()
if domeniu_sel != "Toate":
    df_filtrat = df_filtrat[df_filtrat["domeniu"] == domeniu_sel]
if satisfactie_sel != "Toate":
    df_filtrat = df_filtrat[df_filtrat["satisfactie"] == satisfactie_sel]
if dificultate_sel != "Toate":
    df_filtrat = df_filtrat[df_filtrat["dificultate"] == dificultate_sel]
if interval_date and len(interval_date) == 2:
    df_filtrat = df_filtrat[
        (df_filtrat["data"] >= interval_date[0]) &
        (df_filtrat["data"] <= interval_date[1])
    ]

# ─── METRICI GLOBALE ──────────────────────────────────────────────────────────

st.subheader("📊 Statistici generale")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total conversații", len(df_filtrat))
with col2:
    n_poz = (df_filtrat["satisfactie"] == "pozitiv").sum()
    st.metric("😊 Pozitive", n_poz)
with col3:
    n_neu = (df_filtrat["satisfactie"] == "neutru").sum()
    st.metric("😐 Neutre", n_neu)
with col4:
    n_neg = (df_filtrat["satisfactie"] == "negativ").sum()
    st.metric("😠 Negative", n_neg)

st.markdown("---")

# ─── GRAFICE ──────────────────────────────────────────────────────────────────

col_g1, col_g2, col_g3 = st.columns(3)

with col_g1:
    st.subheader("Distribuție satisfacție")
    if not df_filtrat.empty:
        count_sat = df_filtrat["satisfactie"].value_counts().reset_index()
        count_sat.columns = ["satisfactie", "count"]
        culori = {"pozitiv": "#2ecc71", "neutru": "#f39c12", "negativ": "#e74c3c", "necunoscut": "#95a5a6"}
        fig = px.pie(
            count_sat, values="count", names="satisfactie",
            color="satisfactie", color_discrete_map=culori,
            hole=0.4
        )
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250)
        st.plotly_chart(fig, use_container_width=True)

with col_g2:
    st.subheader("Conversații per domeniu")
    if not df_filtrat.empty:
        count_dom = df_filtrat["domeniu"].value_counts().reset_index()
        count_dom.columns = ["domeniu", "count"]
        fig2 = px.bar(count_dom, x="domeniu", y="count", color="domeniu")
        fig2.update_layout(
            margin=dict(t=0, b=0, l=0, r=0), height=250,
            showlegend=False, xaxis_title="", yaxis_title=""
        )
        st.plotly_chart(fig2, use_container_width=True)

with col_g3:
    st.subheader("Distribuție dificultate")
    if not df_filtrat.empty:
        count_dif = df_filtrat["dificultate"].value_counts().reset_index()
        count_dif.columns = ["dificultate", "count"]
        culori_dif = {"simpla": "#3498db", "medie": "#f39c12", "complexa": "#e74c3c"}
        fig3 = px.pie(
            count_dif, values="count", names="dificultate",
            color="dificultate", color_discrete_map=culori_dif,
            hole=0.4
        )
        fig3.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250)
        st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ─── TABEL CONVERSATII ────────────────────────────────────────────────────────

st.subheader(f"📋 Conversații ({len(df_filtrat)} rezultate)")

EMOTII = {"pozitiv": "😊", "neutru": "😐", "negativ": "😠", "necunoscut": "❓"}

df_afisare = df_filtrat[["id", "domeniu", "dificultate", "satisfactie", "intentie", "nr_replici", "timestamp"]].copy()
df_afisare["satisfactie"] = df_afisare["satisfactie"].apply(
    lambda x: f"{EMOTII.get(x, '')} {x}" if x else "—"
)
df_afisare["timestamp"] = df_afisare["timestamp"].dt.strftime("%d.%m.%Y %H:%M")
df_afisare.columns = ["ID", "Domeniu", "Dificultate", "Satisfacție", "Intenție", "Nr. replici", "Data"]

st.dataframe(df_afisare, use_container_width=True, hide_index=True)

st.markdown("---")

# ─── DETALII CONVERSATIE ──────────────────────────────────────────────────────

st.subheader("🔎 Vizualizare conversație")

if df_filtrat.empty:
    st.info("Nicio conversație nu corespunde filtrelor selectate.")
else:
    optiuni = {
        f"{row['id']} — {row['domeniu']} — {EMOTII.get(row['satisfactie'], '')} {row['satisfactie']}": idx
        for idx, row in df_filtrat.iterrows()
    }
    selectie = st.selectbox("Selectează conversația", list(optiuni.keys()))
    idx_sel = optiuni[selectie]
    conv = df_filtrat.loc[idx_sel]

    # Info conversatie
    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    with col_i1:
        st.info(f"**Domeniu:** {conv['domeniu']}")
    with col_i2:
        st.info(f"**Dificultate:** {conv['dificultate']}")
    with col_i3:
        emoji = EMOTII.get(conv['satisfactie'], '')
        st.info(f"**Satisfacție:** {emoji} {conv['satisfactie']}")
    with col_i4:
        st.info(f"**Intenție:** {conv['intentie']}")

    # Rezumat
    if conv["rezumat"]:
        st.markdown("**📝 Rezumat:**")
        st.success(conv["rezumat"])

    # Dialog
    st.markdown("**💬 Dialog:**")
    for replica in conv["conversatie"]:
        rol = replica.get("rol", "")
        text = replica.get("text", "")
        if rol == "operator":
            st.markdown(
                f"<div style='background:#e8f4f8;padding:8px 12px;border-radius:8px;"
                f"margin:4px 0;border-left:4px solid #3498db'>"
                f"<b>🤖 Operator:</b> {text}</div>",
                unsafe_allow_html=True
            )
        elif rol == "client":
            st.markdown(
                f"<div style='background:#fef9e7;padding:8px 12px;border-radius:8px;"
                f"margin:4px 0;border-left:4px solid #f39c12'>"
                f"<b>👤 Client:</b> {text}</div>",
                unsafe_allow_html=True
            )