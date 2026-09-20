import sys
import os

# make sure the project root is importable regardless of working directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from datetime import datetime
from geopy.geocoders import Nominatim
from streamlit_js_eval import get_geolocation

from utils import db
from ml.cnn_classifier import classify_garbage
from ml.priority_model import predict_priority
from ml.duplicate_detection import find_duplicate

st.title("📤 Report Litter Spot")

# make sure database + uploads folder exist
db.init_db()
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# ------------------------------------------------------------------
# STEP 1: User inputs
# ------------------------------------------------------------------
location = st.text_input("Enter Location / Address")
description = st.text_area("Describe the litter problem")
image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# ------------------------------------------------------------------
# STEP 2: Automatic GPS capture (with accuracy)
# ------------------------------------------------------------------
st.subheader("📍 Get GPS Location")

gps = get_geolocation()

latitude, longitude, accuracy = None, None, None

if gps:
    latitude = gps["coords"]["latitude"]
    longitude = gps["coords"]["longitude"]
    accuracy = gps["coords"].get("accuracy")
    st.success(f"📡 Detected Location: {latitude:.6f}, {longitude:.6f}  (± {accuracy:.0f} m accuracy)")

# fallback: convert typed address -> coordinates if GPS unavailable / overridden
if location:
    geolocator = Nominatim(user_agent="litter_app")
    try:
        loc = geolocator.geocode(location)
        if loc:
            latitude = loc.latitude
            longitude = loc.longitude
            accuracy = accuracy or 100  # address-based lookup, assume ~100m accuracy
            st.info(f"📌 Coordinates from address: {latitude:.6f}, {longitude:.6f}")
    except Exception:
        st.warning("Geocoding failed — using GPS coordinates if available.")

# ------------------------------------------------------------------
# STEP 3: Show a live preview of what the ML pipeline WOULD do
#         (garbage type + priority) before final submission
# ------------------------------------------------------------------
predicted_type, classification_mode, confidence = None, None, None

if image is not None:
    # save a temp copy so PIL / the CNN can read it
    temp_path = os.path.join("uploads", "_preview_" + image.name)
    with open(temp_path, "wb") as f:
        f.write(image.getbuffer())

    predicted_type, classification_mode, confidence = classify_garbage(temp_path)

    st.subheader("🧠 CNN Garbage Classification")
    if classification_mode == "CNN":
        st.info(f"Predicted type: **{predicted_type.upper()}**  "
                f"(trained CNN model, confidence: {confidence*100:.1f}%)")
    else:
        st.info(f"Predicted type: **{predicted_type.upper()}**  "
                f"(fallback rule-based classifier)")

# ------------------------------------------------------------------
# STEP 4: Submit
# ------------------------------------------------------------------
if st.button("Submit Report"):

    if location and description and image and latitude and longitude:

        # save the final image file
        image_path = f"uploads/{image.name}"
        with open(image_path, "wb") as f:
            f.write(image.getbuffer())

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d %H:%M")

        # -------- Duplicate Detection --------
        existing_reports = db.get_reports_as_dicts()
        duplicate_id = find_duplicate(latitude, longitude, now, existing_reports)

        # -------- Priority Prediction --------
        # a fresh report starts with duplicate_count = 1
        priority = predict_priority(predicted_type, duplicate_count=1, accuracy=accuracy or 50)
        classification_mode = classification_mode or "fallback"

        if duplicate_id:
            # This spot has already been reported recently -> don't insert a
            # new row, just strengthen the existing one (bumps its duplicate
            # count, which in turn can raise its predicted priority).
            new_priority = predict_priority(predicted_type, duplicate_count=2, accuracy=accuracy or 50)
            db.increment_duplicate(duplicate_id, new_priority=new_priority)
            st.warning(
                f"⚠️ This looks like a spot that was already reported "
                f"(report #{duplicate_id}). We've marked it as reported again "
                f"instead of creating a duplicate — this raises its priority."
            )
        else:
            db.insert_report({
                "location": location,
                "description": description,
                "latitude": latitude,
                "longitude": longitude,
                "accuracy": accuracy,
                "image": image_path,
                "garbage_type": predicted_type,
                "classification_mode": classification_mode,
                "priority": priority,
                "status": "Unresolved",
                "duplicate_count": 1,
                "date": date_str,
            })
            st.success(f"✅ Report submitted! Predicted type: {predicted_type}, Priority: {priority}")

    else:
        st.warning("Please fill all fields, upload an image, and allow GPS/enter a location.")
