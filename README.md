# 🧹 Crowdsourced Litter Reporting System

A Streamlit web app where citizens report littered spots with a photo. The
app automatically captures GPS location, classifies the garbage type with a
CNN, predicts cleanup priority with a Random Forest model, detects duplicate
reports of the same spot, and gives admins a dashboard with hotspot analysis.

## Pipeline

```
User Uploads Garbage Image
          │
          ▼
Capture GPS Coordinates (+ accuracy)
          │
          ▼
CNN Classifies Garbage Type
          │
          ▼
ML Predicts Priority (Random Forest)
          │
          ▼
Duplicate Detection (distance + time)
          │
          ▼
Save to Database
          │
          ▼
Display on Interactive Map (colour-coded by priority)
          │
          ▼
Hotspot Analysis (DBSCAN)
          │
          ▼
Admin Dashboard
```

## Project Structure

```
litter_reporting_app/
├── app.py                     # Home page
├── seed_demo_data.py          # Loads sample data so the demo isn't empty
├── requirements.txt
├── pages/
│   ├── report_litter.py       # Report form: GPS + CNN + priority + duplicate check
│   ├── view_reports.py        # Interactive map + hotspots
│   └── admin_dashboard.py     # Stats, charts, resolve reports
├── ml/
│   ├── cnn_classifier.py      # CNN garbage classifier (+ simple fallback mode)
│   ├── train_cnn.py           # Optional: train the CNN on your own dataset
│   ├── priority_model.py      # Random Forest priority prediction
│   ├── train_priority_model.py# Trains + saves the Random Forest model
│   ├── duplicate_detection.py # Haversine distance + time-window duplicate check
│   └── hotspot_analysis.py    # DBSCAN clustering
├── maps/
│   └── map_view.py            # Builds the folium map with colour-coded markers
├── utils/
│   └── db.py                  # All SQLite operations
├── models/
│   └── priority_rf_model.pkl  # Pre-trained Random Forest (ready to use)
├── database/database.db       # SQLite database (auto-created)
└── uploads/                   # Uploaded report images
```

## Setup

```bash
pip install -r requirements.txt
python seed_demo_data.py     # optional: adds a few sample reports to demo with
streamlit run app.py
```

## About the CNN classifier (important to know before your demo)

Training a real CNN needs a labelled image dataset (e.g. thousands of photos
sorted into plastic/paper/organic/etc. folders), which this project doesn't
ship with. So `ml/cnn_classifier.py` works in **two modes**:

- **CNN mode**: if you train a model (`python ml/train_cnn.py` after adding
  images to `data/train/<class>/`), it's saved to
  `models/cnn_garbage_classifier.h5` and used automatically.
- **Fallback mode** (default, no setup needed): a simple, easy-to-explain
  colour/brightness rule-based classifier, so the whole app works out of the
  box for a live demo. The UI clearly labels which mode produced each
  prediction, so you can explain this honestly to an evaluator.

The Random Forest **priority model**, on the other hand, is fully trained
and shipped ready-to-use (`models/priority_rf_model.pkl`), since it only
needs small structured features (garbage type, repeat-report count, GPS
accuracy) rather than a large image dataset.

## Features

- ✅ Automatic GPS location capture with reported accuracy
- ✅ CNN-based garbage type classification (plastic, paper, organic, metal, glass, other)
- ✅ Duplicate report detection using GPS distance + time window
- ✅ Priority prediction using a Random Forest model
- ✅ Garbage hotspot visualization using DBSCAN clustering
- ✅ Interactive map with colour-coded markers (red = high, orange/yellow = medium, green = low)
- ✅ Admin dashboard: total / resolved / unresolved reports, garbage-type distribution, hotspot list, resolve controls

## How each ML piece works (for your presentation)

| Feature | Technique | File |
|---|---|---|
| Garbage classification | CNN (Conv2D x2 + Dense), with fallback color-rule classifier | `ml/cnn_classifier.py` |
| Priority prediction | Random Forest Classifier (scikit-learn) | `ml/priority_model.py` |
| Duplicate detection | Haversine distance formula + time-window check | `ml/duplicate_detection.py` |
| Hotspot analysis | DBSCAN clustering (haversine metric) | `ml/hotspot_analysis.py` |

Every file has comments at the top explaining what it does and why, so you
can walk an evaluator through the logic file-by-file.
