import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from utils import db

# Page configuration
st.set_page_config(
    page_title="Litter Reporting App",
    
    layout="wide"
)

# make sure the database table exists on first run
db.init_db()

# Title
st.title("Litter Reporting System")

# Description
st.write("""
This platform allows citizens to report littered or dirty places in their city.
Users upload a photo, the app automatically captures GPS location, a CNN model
classifies the type of garbage, and a Random Forest model predicts how
urgently the spot needs to be cleaned. Repeated reports of the same spot are
detected as duplicates and automatically merged, and DBSCAN clustering is
used to surface city-wide garbage hotspots for the admin team.
""")

st.markdown("---")

# Navigation section
st.header("Choose an Option")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader(" Report Litter")
    st.write("Upload a photo and report a dirty location.")
    if st.button("Go to Report Page"):
        st.switch_page("pages/report_litter.py")

with col2:
    st.subheader(" View Reports")
    st.write("See all reported litter locations on the map, colour-coded by priority.")
    if st.button("View Map Reports"):
        st.switch_page("pages/view_reports.py")

with col3:
    st.subheader("Admin Dashboard")
    st.write("View stats, garbage-type charts, hotspots, and resolve reports.")
    if st.button("Open Dashboard"):
        st.switch_page("pages/admin_dashboard.py")

st.markdown("---")

# About section
st.header("About the Project")

st.write("""
**Pipeline:**

Upload Image → Capture GPS → CNN Classifies Garbage Type → ML Predicts Priority
→ Duplicate Detection → Save to Database → Interactive Map → DBSCAN Hotspot
Analysis → Admin Dashboard

**Main Features:**
-  Automatic GPS location capture with reported accuracy
-  CNN-based garbage type classification (plastic, paper, organic, metal, glass, other)
-  Priority prediction using a Random Forest model
-  Duplicate report detection using GPS distance + time
-  Garbage hotspot visualization using DBSCAN clustering
-  Interactive map with colour-coded priority markers
-  Admin dashboard with report stats and garbage-type distribution
""")

st.markdown("---")

# Footer
st.write("Developed using Streamlit, Python, scikit-learn, and Folium Maps")
