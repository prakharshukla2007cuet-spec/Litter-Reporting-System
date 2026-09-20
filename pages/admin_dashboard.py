import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from utils import db
from ml.hotspot_analysis import find_hotspots

st.title("📊 Admin Dashboard")

db.init_db()
df = db.get_all_reports()

if df.empty:
    st.warning("⚠ No reports available yet.")
    st.stop()

# ------------------------------------------------------------------
# TOP-LEVEL STATS
# ------------------------------------------------------------------
total_reports = len(df)
resolved = (df["status"] == "Resolved").sum()
unresolved = (df["status"] == "Unresolved").sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reports", total_reports)
col2.metric("Resolved", resolved)
col3.metric("Unresolved", unresolved)

_, hotspot_df = find_hotspots(df)
col4.metric("Active Hotspots", 0 if hotspot_df is None else len(hotspot_df))

st.markdown("---")

# ------------------------------------------------------------------
# CHARTS
# ------------------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("🗑 Garbage Type Distribution")
    type_counts = df["garbage_type"].value_counts()
    st.bar_chart(type_counts)

with chart_col2:
    st.subheader("🚦 Priority Distribution")
    priority_counts = df["priority"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
    st.bar_chart(priority_counts)

st.markdown("---")

# ------------------------------------------------------------------
# HOTSPOT SUMMARY TABLE
# ------------------------------------------------------------------
st.subheader("🔥 Hotspot Areas (DBSCAN)")
if hotspot_df is not None and not hotspot_df.empty:
    st.dataframe(hotspot_df, use_container_width=True)
else:
    st.write("No hotspots detected yet (need at least 3 nearby reports in one area).")

st.markdown("---")

# ------------------------------------------------------------------
# MANAGE REPORTS  (mark resolved / unresolved)
# ------------------------------------------------------------------
st.subheader("🛠 Manage Reports")

for _, row in df.sort_values("id", ascending=False).iterrows():
    with st.expander(f"#{row['id']} - {row['location']} | {row['garbage_type']} | "
                      f"{row['priority']} priority | {row['status']}"):
        st.write(f"**Description:** {row['description']}")
        st.write(f"**Reported:** {row['date']}  |  **Times reported:** {row['duplicate_count']}")
        st.write(f"**Coordinates:** {row['latitude']:.5f}, {row['longitude']:.5f}")

        if row["image"] and os.path.exists(row["image"]):
            st.image(row["image"], width=250)

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if row["status"] != "Resolved":
                if st.button("✅ Mark Resolved", key=f"resolve_{row['id']}"):
                    db.mark_resolved(row["id"])
                    st.rerun()
        with btn_col2:
            if row["status"] == "Resolved":
                if st.button("↩️ Mark Unresolved", key=f"unresolve_{row['id']}"):
                    db.mark_unresolved(row["id"])
                    st.rerun()
