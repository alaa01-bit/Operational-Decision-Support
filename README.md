# Enterprise Operational Decision Support & SLA Risk Engine

An enterprise AI microservice developed to address operational bottlenecks, SLA breach risks, and automated triage for high-volume enterprise workflows.

## Business Context & Impact
- **Target Audience:** Enterprises, Corporate Operations, and Government Digital Platforms.
- **Problem:** Unanticipated SLA breaches result in financial penalties, client churn, and operational latency.
- **Solution:** Hybrid ML engine combining TF-IDF NLP semantic representation with Random Forest Classifier (Breach Probability) and Gradient Boosting Regressor (Resolution Hours), paired with prescriptive business action triggers.

## Key Features
- **Dual-Model Inference:** 
  - Predicts probability of SLA failure ($Acc \approx 89\%$).
  - Forecasts resolution duration with RMSE $\approx 8.1$ hrs.
- **Operational Data Telemetry:** Automatically logs inference parameters, latency, and drift signals into an SQLite operational store.
- **Continuous Monitoring:** `/metrics` endpoint exposes health, average inference latency, and high-risk alerts.
- **Enterprise-Ready:** Fully containerized with Docker, RESTful OpenAPI docs, and strict Pydantic schemas.

## Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models and generate synthetic business datasets
python train_pipeline.py

# 3. Launch the API service
uvicorn app:app --port 8001

# 4. Run automated tests
python test_app.py
```
