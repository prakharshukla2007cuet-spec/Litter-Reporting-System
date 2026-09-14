import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from streamlit_folium import st_folium

from utils import db
from ml.hotspot_analysis import find_hotspots
from maps.map_view import build_map

st.title(" Litter Reports Map")

db.init_db()
df = db.get_all_reports()

if df.empty:
    st.warning("⚠ No reports available yet. Go to 'Report Litter' to add one.")
else:
    st.markdown("""
    **Marker colours:** 🔴 High priority &nbsp;&nbsp; 🟠 Medium priority &nbsp;&nbsp;
    🟢 Low priority &nbsp;&nbsp; ⚪ Resolved
    """)

    show_hotspots = st.checkbox("Show garbage hotspots", value=True)

    # ---- Hotspot Analysis (DBSCAN) ----
    clustered_df, hotspot_df = find_hotspots(df)

    if show_hotspots and hotspot_df is not None:
        st.info(f"Detected **{len(hotspot_df)}** hotspot area(s) with repeated litter reports.")

    # ---- Interactive Map ----
    m = build_map(df, hotspot_df=hotspot_df, show_hotspots=show_hotspots)
    if m:
        st_folium(m, width=900, height=550)

    # ---- Report list ----
    st.subheader(" Report List")
    display_cols = ["id", "location", "garbage_type", "priority", "status",
                     "duplicate_count", "date", "description"]
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True)
