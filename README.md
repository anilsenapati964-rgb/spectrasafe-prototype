# SpectraSafe

**Inline Hyperspectral-AI Food Contamination Screening** is a runnable software prototype for demonstrating a proposed screening workflow. The application uses ordinary RGB images, derives 12 simulated spectral-like bands, extracts image features, classifies against procedural demo data, and visualizes a virtual PASS/FLAG routing decision.

> **Simulation notice:** This project does not connect to an HSI camera, NIR illumination, conveyor, PLC, or actuator. Pseudo-HSI bands and risk scores are software simulations. Outputs are not laboratory results, validated probabilities, pathogen identifications, or suitable for food-safety decisions.

## Features

- Industrial-style Streamlit inspection console with sample, upload, and optional webcam input.
- Procedurally generated grain imagery: five clean and five flagged samples are created locally on first use when sample files are absent.
- Image preprocessing, foreground ROI approximation, and 12-band RGB-derived pseudo-HSI cube and false-color view.
- Spectral-like intensity signature and spectral-spatial features.
- Random Forest automatically trained on generated demo features; model is cached at `models/classifier.pkl`.
- Stable scripted scenarios: clean sample = 18% (PASS at default threshold), flagged sample = 84% (FLAG at default threshold).
- Prototype risk display, virtual conveyor routing, simulated air-jet/servo status, sample IDs, history, and risk trend chart.
- Future deployment architecture and limitations pages.
- A separate pseudo-isometric 3D Inspection Line Simulation with timed stages, pause/resume/restart, auto demo, scanning/illumination effects, and PASS/FLAG diverter routing.

## Project structure

```text
SIH26233/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/config.toml      # Enables local project static assets
├── data/samples/              # Generated on first app launch
├── models/classifier.pkl      # Trained/cached automatically
├── static/industrial_conveyor.jpg    # Industrial line visual used in the digital twin
├── static/spectrasafe_conveyor_3d.png  # Original concept render
├── src/
│   ├── classifier.py
│   ├── conveyor_simulation.py
│   ├── feature_extraction.py
│   ├── preprocessing.py
│   ├── risk_scoring.py
│   └── spectral_simulation.py
└── outputs/                   # Available for future exports
```

## Installation

Python 3.10 or newer is recommended.

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS activation:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run the application

```bash
python -m streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`).

## Demo mode

1. Open **Inspection Console**.
2. Select **CLEAN SAMPLE**, then **DEMO MODE · RUN PIPELINE**. The deterministic demonstration reports 18% risk, PASS, and continue on main line.
3. Select **FLAGGED SAMPLE**, run again. It reports 84% risk, FLAG, simulated air-jet/servo activation, and quarantine routing.
4. Expand **Pseudo-HSI data cube** to show the simulated bands and signature. The history and graph update with every run.
5. Turn on **Presentation / Demo Mode** in the sidebar for a reduced-control presentation layout. Use browser/app full-screen mode for a clean 16:9 recording.

For the digital-twin view, open **3D Inspection Line Simulation**, choose CLEAN or FLAGGED, then use **START INSPECTION**, **PAUSE / RESUME**, **RESTART**, or **AUTO DEMO MODE**. The animated line shows each active station, illumination/camera scan, camera-to-edge-AI data flow, decision, and virtual diverter destination.

For a 45–90 second recording: show the header and clean run, expand the pseudo-HSI section, run the flagged case to show quarantine, then finish with history/graph and the Deployment Architecture page. Capture the app window at 1920×1080 if available.

## Dataset format

The demo does not require an external dataset. If images are placed under `data/samples/`, the console can use them as canned examples; otherwise it creates ten small PNG examples automatically. Uploaded images accept PNG, JPG/JPEG, and WebP. Generated flagged imagery is illustrative only and is not real contamination evidence.

## Model training

On first run, `src/classifier.py` generates two procedural feature classes, performs a stratified train/test split, fits a small Random Forest, computes demo metrics, and saves `models/classifier.pkl`. A missing or unreadable model is rebuilt automatically. Metrics are explicitly demo-dataset performance and must not be read as real-world validation. Clean/flagged canned scenarios use deterministic presentation scores to ensure the requested cases are repeatable.

## How pseudo-HSI works

The input RGB image is resized, denoised, contrast adjusted, and converted to a foreground mask. Twelve smooth mixtures of its RGB channels form a cube shaped `height × width × 12`. A selected-band composite provides false color and mean band values provide a signature. The displayed band labels are illustrative only; these are not calibrated wavelengths or measured reflectance.

## Limitations and future hardware integration

This concept prototype contains no physical camera, NIR source, line-scan acquisition, edge accelerator, PLC connection, conveyor, object-tracking sensor, or reject actuator. The classifier is trained on procedural synthetic feature vectors rather than food imagery or laboratory-confirmed samples. Risk outputs are for demonstrating UI and control flow, not diagnosing pathogens or clearing food lots. Production integration would require validated sensors, representative ground-truth datasets, calibration, safety engineering, latency tests, and controller/actuator interfaces. The **Deployment Architecture** page depicts that future architecture and labels it as future hardware.
