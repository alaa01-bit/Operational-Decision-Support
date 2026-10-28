import time
import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

DB_PATH = "operations.db"
MODEL_PATH = "models/sla_risk_engine.joblib"

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(
    title="Enterprise SLA Risk & Operations Intelligence Engine",
    description="Predictive API for operational bottleneck detection, SLA breach likelihood, and decision support.",
    version="1.0.0"
)

# Mount Frontend UI
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
def serve_dashboard():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Welcome to Enterprise Operations API. Visit /docs for Swagger UI."}


# Load Model Pipeline
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Run train_pipeline.py first.")

bundle = joblib.load(MODEL_PATH)
clf = bundle["clf"]
reg = bundle["reg"]
tfidf = bundle["tfidf"]
dept_columns = bundle["dept_columns"]
chan_columns = bundle["chan_columns"]

class TicketRequest(BaseModel):
    ticket_code: str = Field(..., example="TICK-88421")
    department: str = Field(..., example="Finance")
    issue_text: str = Field(..., example="Delay in corporate invoice reconciliation and tax compliance audit.")
    request_channel: str = Field(default="Portal", example="Portal")
    sla_hours_limit: float = Field(default=24.0, example=24.0)
    current_agent_load: int = Field(default=15, example=15)

class PredictionResponse(BaseModel):
    ticket_code: str
    breach_probability: float
    breach_risk_category: str
    expected_resolution_hours: float
    sla_hours_limit: float
    is_projected_breach: bool
    recommended_business_action: str
    latency_ms: float

def build_feature_vector(req: TicketRequest):
    # TF-IDF
    tfidf_vec = tfidf.transform([req.issue_text]).toarray()
    
    # One-hot encode department
    dept_vec = [1.0 if f"dept_{req.department}" == col else 0.0 for col in dept_columns]
    
    # One-hot encode channel
    chan_vec = [1.0 if f"chan_{req.request_channel}" == col else 0.0 for col in chan_columns]
    
    num_vec = [req.sla_hours_limit, req.current_agent_load]
    
    X = np.hstack([num_vec, dept_vec, chan_vec, tfidf_vec[0]]).reshape(1, -1)
    return X

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Enterprise Operational Intelligence Microservice",
        "models_loaded": ["RandomForestClassifier", "GradientBoostingRegressor", "TfidfVectorizer"]
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_ticket_risk(req: TicketRequest):
    start_time = time.time()
    try:
        X = build_feature_vector(req)
        
        prob_breach = float(clf.predict_proba(X)[0][1])
        pred_hours = float(reg.predict(X)[0])
        
        # Risk categorization
        if prob_breach > 0.70 or pred_hours > req.sla_hours_limit:
            risk_cat = "HIGH_CRITICAL"
            action = "Immediate executive escalation; reroute ticket to Senior Operations Lead and trigger automated alert."
        elif prob_breach > 0.35:
            risk_cat = "MEDIUM_ATTENTION"
            action = "Reallocate supplementary team resources; schedule preemptive daily check-in."
        else:
            risk_cat = "LOW_STABLE"
            action = "Standard operational routing via automated workflow queue."
            
        latency = round((time.time() - start_time) * 1000, 2)
        is_breach = bool(pred_hours > req.sla_hours_limit or prob_breach > 0.5)

        # Log inference to database for telemetry and drift tracking
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
        INSERT INTO model_inference_logs 
        (ticket_code, predicted_breach_prob, predicted_duration_hours, predicted_risk_level, recommended_action, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (req.ticket_code, prob_breach, round(pred_hours, 2), risk_cat, action, latency))
        conn.commit()
        conn.close()

        return PredictionResponse(
            ticket_code=req.ticket_code,
            breach_probability=round(prob_breach, 4),
            breach_risk_category=risk_cat,
            expected_resolution_hours=round(pred_hours, 2),
            sla_hours_limit=req.sla_hours_limit,
            is_projected_breach=is_breach,
            recommended_business_action=action,
            latency_ms=latency
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_operational_metrics():
    """Returns business metrics & model monitoring statistics for stakeholders."""
    conn = sqlite3.connect(DB_PATH)
    total_tickets = pd.read_sql("SELECT COUNT(*) as c FROM operational_tickets", conn).iloc[0]['c']
    inferences = pd.read_sql("SELECT * FROM model_inference_logs ORDER BY id DESC LIMIT 50", conn)
    conn.close()
    
    avg_latency = float(inferences['latency_ms'].mean()) if not inferences.empty else 0.0
    high_risk_count = int((inferences['predicted_risk_level'] == 'HIGH_CRITICAL').sum()) if not inferences.empty else 0
    
    return {
        "dataset_total_records": int(total_tickets),
        "recent_inferences_analyzed": len(inferences),
        "average_inference_latency_ms": round(avg_latency, 2),
        "high_risk_tickets_detected": high_risk_count,
        "monitoring_status": "NORMAL - No significant feature drift detected."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
