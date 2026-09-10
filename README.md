# Real-Time Micro-Expression & Cognitive Load Analyzer

A placement-portfolio project: detects subtle facial Action Units (AUs) via
webcam, tracks blink rate and micro-movements, and (eventually) scores
stress/confidence over time using a temporal deep learning model.

This repo is structured so you can build it **incrementally**, phase by phase,
instead of trying to write everything at once.

```
micro-expression-detector/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app (Phase 3)
│   │   ├── au_extraction.py     # MediaPipe landmark -> Action Unit logic
│   │   ├── blink_detector.py    # Eye Aspect Ratio (EAR) blink counting
│   │   └── schemas.py           # Pydantic models for API payloads
│   └── models/
│       └── lstm_classifier.py   # Phase 2: temporal model definition
├── data/                        # put CASME II / SAMM datasets here (gitignored)
├── notebooks/
│   └── phase2_train_model.ipynb # placeholder for training experiments
├── frontend/
│   └── (Streamlit app or React app goes here, Phase 3)
├── phase1_blink_and_au_poc.py   # RUN THIS FIRST — standalone webcam PoC
├── requirements.txt
└── README.md
```

## Quickstart (Phase 1 — Proof of Concept)

This is the very first thing you should run. It needs no model training,
no backend, no frontend — just your webcam. It proves out the core CV
pipeline: face landmarks → Eye Aspect Ratio → blink counting → a couple of
basic Action Units (brow lowerer, lip corner pull) printed live on screen.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python phase1_blink_and_au_poc.py
```

Press `q` to quit the webcam window.

What you should see:
- A live video feed with your face mesh landmarks drawn on top.
- Blink count incrementing when you blink.
- Two live AU proxy values (brow lowerer distance, mouth corner pull)
  printed as an overlay — these are the raw signals your future LSTM
  model will consume as a time series.

## Roadmap

### Phase 1 — Proof of Concept (this repo gives you a working version)
- [x] Open webcam with OpenCV
- [x] Extract 468 3D face landmarks with MediaPipe Face Mesh
- [x] Compute Eye Aspect Ratio (EAR) → detect blinks → track blink rate
- [x] Compute 2 basic AU proxy distances (AU4 brow lowerer, AU12 lip pull)
- [ ] Log these signals to a CSV so you have your own baseline dataset

### Phase 2 — Model Training
- [ ] Download CASME II or SAMM (needs a signed data-use agreement — apply early)
- [ ] Preprocess: crop face, normalize landmarks, build sliding windows
      (15–30 frames per window)
- [ ] Train `backend/models/lstm_classifier.py` to classify
      micro-expression vs. neutral, and/or regress a cognitive-load score
- [ ] Evaluate: accuracy/F1 on held-out subjects (subject-independent split
      — this matters a lot for micro-expression datasets)

### Phase 3 — Integration
- [ ] Wrap Module 1 (AU extraction) + Module 2 (LSTM) in FastAPI,
      streaming frames over WebSockets
- [ ] Build a Streamlit or React dashboard with a live stress/confidence
      timeline (Plotly/Chart.js)
- [ ] Generate a post-interview PDF/HTML report with timestamped flags

### Stretch goals (great for standing out in interviews)
- [ ] Port the MediaPipe + inference pipeline to TensorFlow.js so video
      never leaves the browser (privacy-first pitch)
- [ ] Add a calibration step per-user (everyone's neutral face is different)
- [ ] Add confidence intervals / uncertainty estimates on the stress score
      rather than a single number

## Notes on datasets

CASME II and SAMM both require you to fill out an academic data-use
request form from the respective university labs — this can take a few
days to be approved, so start that process now, in parallel with building
Phase 1.

## Ethical / scope note

This tool is designed as a **self-practice aid** (mock interviews you run
on yourself), not for scoring real candidates without consent — be upfront
about that framing if you present this for placements, since automated
"confidence/stress" scoring of real interviewees raises real
fairness/consent questions that are worth being able to speak to.
