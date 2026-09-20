# ============================================================
#  INTERACTIVE MAP BUILDER
# ============================================================
# Builds a folium map with:
#   - one marker per report, colour-coded by priority
#         red    = High priority
#         orange = Medium priority   (shown as "yellow" marker icon)
#         green  = Low priority
#   - optional shaded circles showing DBSCAN hotspots
# ============================================================

import folium
from folium.plugins import MarkerCluster

PRIORITY_COLOR = {
    "High": "red",
    "Medium": "orange",   # folium has no built-in "yellow" icon; orange reads as yellow/amber
    "Low": "green",
}


def build_map(df, hotspot_df=None, show_hotspots=True):
    """
    df : DataFrame of reports, must contain latitude, longitude, priority,
         garbage_type, status, description, location, date, duplicate_count
    hotspot_df : optional DataFrame from ml/hotspot_analysis.py with
                 center_lat, center_lon, report_count
    """
    if df.empty:
        return None

    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Satellite",
    )

    marker_cluster = MarkerCluster().add_to(m)

    for _, row in df.iterrows():
        color = PRIORITY_COLOR.get(row.get("priority", "Low"), "green")
        resolved = row.get("status") == "Resolved"

        popup_html = f"""
        <b>Type:</b> {row.get('garbage_type', 'N/A')}<br>
        <b>Priority:</b> {row.get('priority', 'N/A')}<br>
        <b>Status:</b> {row.get('status', 'Unresolved')}<br>
        <b>Location:</b> {row.get('location', '')}<br>
        <b>Description:</b> {row.get('description', '')}<br>
        <b>Reported:</b> {row.get('date', '')}<br>
        <b>Times reported:</b> {row.get('duplicate_count', 1)}
        """

        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row.get('garbage_type','')} - {row.get('priority','')}",
            icon=folium.Icon(
                color="gray" if resolved else color,
                icon="ok" if resolved else "trash",
                prefix="glyphicon",
            ),
        ).add_to(marker_cluster)

    # draw hotspot circles
    if show_hotspots and hotspot_df is not None and not hotspot_df.empty:
        for _, h in hotspot_df.iterrows():
            folium.Circle(
                location=[h["center_lat"], h["center_lon"]],
                radius=150,
                color="purple",
                fill=True,
                fill_opacity=0.15,
                popup=f"🔥 Hotspot - {h['report_count']} reports nearby",
            ).add_to(m)

    folium.LayerControl().add_to(m)
    return m
