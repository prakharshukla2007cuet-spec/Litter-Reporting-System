# ============================================================
#  GARBAGE HOTSPOT DETECTION using DBSCAN
# ============================================================
# Groups nearby litter reports into "hotspots" - areas that get
# dumped in repeatedly and therefore need priority attention.
#
# DBSCAN (Density-Based Spatial Clustering) is used because, unlike
# k-means, it does NOT need us to guess the number of clusters ahead
# of time, and it naturally marks isolated one-off reports as noise
# (cluster = -1) instead of forcing them into a group.
#
# We use the haversine distance metric so "eps" (the neighbourhood
# radius) can be specified in real-world meters instead of raw
# lat/lon degrees.
# ============================================================

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

EPS_METERS = 150     # reports within 150m of each other are "close"
MIN_SAMPLES = 3      # need at least 3 reports nearby to call it a hotspot
EARTH_RADIUS_M = 6371000


def find_hotspots(df):
    """
    df: pandas DataFrame with 'latitude' and 'longitude' columns.

    Returns:
        df_with_clusters : same DataFrame + a 'cluster' column
                            (-1 means "not part of any hotspot")
        hotspot_summary  : DataFrame summarising each hotspot
                            (center location + how many reports in it)
                            or None if no hotspots were found
    """
    df = df.copy()

    if df.empty or len(df) < MIN_SAMPLES:
        df["cluster"] = -1
        return df, None

    coords = np.radians(df[["latitude", "longitude"]].values)
    eps_radians = EPS_METERS / EARTH_RADIUS_M

    dbscan = DBSCAN(eps=eps_radians, min_samples=MIN_SAMPLES, metric="haversine")
    df["cluster"] = dbscan.fit_predict(coords)

    hotspot_rows = []
    for cluster_id in sorted(set(df["cluster"])):
        if cluster_id == -1:
            continue
        cluster_points = df[df["cluster"] == cluster_id]
        hotspot_rows.append({
            "cluster_id": int(cluster_id),
            "center_lat": cluster_points["latitude"].mean(),
            "center_lon": cluster_points["longitude"].mean(),
            "report_count": len(cluster_points),
        })

    hotspot_summary = pd.DataFrame(hotspot_rows) if hotspot_rows else None
    return df, hotspot_summary
