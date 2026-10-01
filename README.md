# Lab Assignment 5 — Model Monitoring for Drift and Latency

## Subject
Cloud Based AI Deployment

## Objective
Serve a trained machine-learning model through an API, deploy it to Google Cloud Run, monitor request latency using Google Cloud Monitoring, and demonstrate numerical data drift using the two-sample Kolmogorov-Smirnov (KS) test.

## Model and data
This project uses the scikit-learn Wine Recognition dataset (`load_wine`). A Random Forest classifier with 200 trees is trained by `train.py`.

## API
- `GET /health` — health check
- `GET /features` — feature names and required order
- `POST /predict` — returns predicted class and model inference latency

## Drift experiment
The drift script creates:
- a reference window of 300 observations
- a current window of 300 observations
- controlled numerical shifts in `alcohol` and `color_intensity`

Each feature is compared with the two-sample KS test. Significance level: alpha = 0.05.

Run:

```bash
python drift/drift_detection.py
```

## Local setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python train.py
pytest
uvicorn app.main:app --reload --port 8080
```

Open `http://127.0.0.1:8080/docs` for the interactive API.

## Docker

```bash
docker build -t wine-monitoring:1.0 .
docker run --rm -p 8080:8080 wine-monitoring:1.0
```

## Google Cloud deployment

Use a Google Cloud project with billing enabled. Create an Artifact Registry Docker repository, build/push the image, then deploy the image to Cloud Run.

Example region used for the lab: `asia-south1`.

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com monitoring.googleapis.com

gcloud artifacts repositories create lab5-repo --repository-format=docker --location=asia-south1

gcloud builds submit --tag asia-south1-docker.pkg.dev/YOUR_PROJECT_ID/lab5-repo/wine-monitoring:1.0

gcloud run deploy wine-monitoring --image asia-south1-docker.pkg.dev/YOUR_PROJECT_ID/lab5-repo/wine-monitoring:1.0 --region asia-south1 --platform managed --allow-unauthenticated
```

## GCP monitoring
Cloud Run automatically sends built-in metrics to Cloud Monitoring. In Cloud Run, open the service and use the Metrics tab. In Metrics Explorer, use the Cloud Run Revision request latency metric and configure percentile alignment for p50 and p95.

## Load test
After deployment:

```bash
python load_test/load_test.py --url "YOUR_CLOUD_RUN_URL/predict" --requests 500 --concurrency 20 --warmup 20
```

Record the request count, concurrency, warm-up policy, p50 and p95 values in the report.

## Security
Do not commit passwords, API keys, service-account keys, or `.env` files.

## Cleanup
Delete the Cloud Run service and Artifact Registry repository after screenshots and testing if they are no longer needed. If the project is only for the lab and contains nothing else, shutting down the project is another option.
