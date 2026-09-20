# ============================================================
#  DUPLICATE REPORT DETECTION
# ============================================================
# Two reports are treated as the SAME litter spot if they are:
#   1) close together in DISTANCE   (within DISTANCE_THRESHOLD_METERS)
#   2) close together in TIME       (within TIME_THRESHOLD_HOURS)
#
# Distance between two GPS points is calculated using the
# Haversine formula (accounts for the curve of the Earth).
# ============================================================

from math import radians, sin, cos, sqrt, atan2
from datetime import datetime

DISTANCE_THRESHOLD_METERS = 30   # spots within 30 meters = same spot
TIME_THRESHOLD_HOURS = 72        # reported again within 3 days = same problem


def haversine_distance(lat1, lon1, lat2, lon2):
    """Returns distance in METERS between two (lat, lon) points."""
    R = 6371000  # radius of Earth in meters
    p1, p2 = radians(lat1), radians(lat2)
    d_phi = radians(lat2 - lat1)
    d_lambda = radians(lon2 - lon1)

    a = sin(d_phi / 2) ** 2 + cos(p1) * cos(p2) * sin(d_lambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def find_duplicate(new_lat, new_lon, new_time, existing_reports):
    """
    Checks a new report against all existing reports.

    existing_reports : list of dicts, each with keys
                        'id', 'latitude', 'longitude', 'date' (string "YYYY-MM-DD HH:MM")

    Returns the id of the matching existing report if a duplicate is found,
    otherwise returns None.
    """
    for r in existing_reports:
        distance = haversine_distance(new_lat, new_lon, r["latitude"], r["longitude"])

        try:
            old_time = datetime.strptime(r["date"], "%Y-%m-%d %H:%M")
            hours_diff = abs((new_time - old_time).total_seconds()) / 3600
        except Exception:
            hours_diff = 999999  # if date is unreadable, treat as "not recent"

        if distance <= DISTANCE_THRESHOLD_METERS and hours_diff <= TIME_THRESHOLD_HOURS:
            return r["id"]

    return None
