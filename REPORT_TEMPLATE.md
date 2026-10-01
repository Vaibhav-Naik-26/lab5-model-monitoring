# Lab Assignment 5 — Model Monitoring for Drift and Latency using Cloud Monitoring Tools

## 1. Model, Data and Deployment Setup

### 1.1 Model
- Dataset: scikit-learn Wine Recognition dataset
- Model: Random Forest classifier
- Number of estimators: 200
- Random state: 42
- Task: 3-class wine classification
- Training/validation split: 80/20 stratified split

### 1.2 API
The model is served through FastAPI.
- `/health`
- `/features`
- `/predict`

The `/predict` endpoint accepts 13 numerical input features and returns the predicted class and measured model inference time.

### 1.3 Container and Google Cloud deployment
- Docker image:
- Artifact Registry repository:
- Cloud Run service:
- Region:
- Project ID:

**Screenshot:** GCP project ID

**Screenshot:** Cloud Run service

---

## 2. Drift Introduction and Observation

### 2.1 Type of drift
The experiment demonstrates **data drift**: the distribution of incoming numerical features changes between a reference window and a current window.

### 2.2 Reference and current windows
- Reference window size: 300 rows
- Current window size: 300 rows
- Statistical test: two-sample Kolmogorov-Smirnov (KS) test
- Significance level: alpha = 0.05

For demonstration, the current window includes controlled shifts in `alcohol` and `color_intensity`.

### 2.3 Statistical test
The KS test compares the empirical distributions of the reference and current windows. A p-value below 0.05 is treated as evidence that the two samples are unlikely to come from the same distribution.

---

## 3. Drift Results

Insert the generated `drift/drift_results.csv` table here.

| Feature | Test | KS statistic | p-value | Interpretation |
|---|---|---:|---:|---|
| alcohol | KS two-sample | | | |
| color_intensity | KS two-sample | | | |
| | | | | |

### Interpretation
The strongest drift should appear in the deliberately shifted features. Features with p-value < 0.05 are interpreted as having statistically significant distribution change at the chosen significance level.

---

## 4. Latency Results

### 4.1 Load generation
- Total requests:
- Concurrency:
- Warm-up requests:
- Cold-start handling:

### 4.2 Client-observed latency
- p50:
- p95:
- Minimum:
- Maximum:

### 4.3 Google Cloud Monitoring latency
- p50:
- p95:

**Screenshot:** Cloud Run Metrics / Cloud Monitoring latency chart

Note: Cloud Run's built-in request latency metric is automatically captured by Cloud Monitoring. The metric is distribution-valued, so percentile alignment can be used for p50 and p95.

---

## 5. Interpretation and Action

### What changed?
The current input distribution was intentionally shifted relative to the reference window. The KS test provides statistical evidence of distribution changes for the affected features.

### What does the evidence show?
Drift means the input distribution changed; it does not by itself prove that model quality became worse. In this lab there are no live ground-truth labels for the deployed traffic, so the result is interpreted as input data drift rather than performance degradation.

### What action would I take?
A practical next step would be to investigate why the affected feature distributions changed, collect ground-truth outcomes, measure model quality on the new traffic, and retrain or recalibrate the model only if the performance evidence supports doing so.

---

## 6. Assignment Checklist

- [ ] Model trained and served through own API
- [ ] Deployed on Google Cloud Run
- [ ] Monitoring shown in Google Cloud Monitoring
- [ ] p50 and p95 latency measured under stated load
- [ ] Reference/current drift windows compared
- [ ] KS statistical test applied
- [ ] Drift table and interpretation included
- [ ] GCP screenshot showing project ID
- [ ] GCP screenshot showing metrics
- [ ] Budget alert configured
- [ ] No secrets/keys in submitted code
- [ ] Cloud resources cleaned up after testing

---

## 7. Screenshots

1. Project ID
2. Cloud Run service
3. Cloud Run metrics
4. Cloud Monitoring p50
5. Cloud Monitoring p95
6. Optional: successful API response / Swagger UI
