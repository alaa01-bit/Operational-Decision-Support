import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import joblib

DB_PATH = "operations.db"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS operational_tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticket_code TEXT UNIQUE,
        department TEXT,
        issue_text TEXT,
        request_channel TEXT,
        sla_hours_limit REAL,
        historical_agent_load INTEGER,
        historical_resolution_hours REAL,
        is_breached INTEGER,
        actual_cost_sar REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_inference_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticket_code TEXT,
        predicted_breach_prob REAL,
        predicted_duration_hours REAL,
        predicted_risk_level TEXT,
        recommended_action TEXT,
        drift_score REAL,
        latency_ms REAL,
        logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def generate_synthetic_operational_data(n_samples=1500):
    np.random.seed(42)
    departments = ["Finance", "Customer Operations", "Procurement", "Digital Services", "IT Infrastructure"]
    channels = ["Portal", "Email", "ERP Integration", "Call Center"]
    
    issues = [
        "Delay in corporate invoice reconciliation and tax compliance audit.",
        "Supplier portal payment gateway timeout during high transaction volume.",
        "Beneficiary identity verification failed on national digital platform.",
        "Monthly procurement purchase order batch sync stalled in ERP.",
        "Customer account authorization error preventing transactional approvals.",
        "Server CPU spike causing latency in automated document processing queue.",
        "Discrepancy in revenue report calculation across regional branches.",
        "Urgent escalation regarding SLA milestone breach in public tender submission.",
        "Automated payroll distribution validation mismatch with banking records.",
        "Routine password policy update and role-based permissions sync."
    ]
    
    records = []
    for i in range(n_samples):
        ticket_code = f"TICK-{10000 + i}"
        dept = np.random.choice(departments)
        issue = np.random.choice(issues)
        channel = np.random.choice(channels)
        sla_limit = float(np.random.choice([12.0, 24.0, 48.0, 72.0]))
        agent_load = int(np.random.randint(2, 35))
        
        # Realistic business simulation logic
        complexity_bias = 1.8 if ("Delay" in issue or "mismatch" in issue or "breach" in issue) else 1.0
        load_bias = (agent_load / 10.0) * 2.5
        est_hours = max(2.0, np.random.normal(loc=18.0 * complexity_bias + load_bias, scale=6.0))
        is_breached = 1 if est_hours > sla_limit else 0
        actual_cost = round(est_hours * np.random.uniform(150, 400), 2)
        
        records.append((
            ticket_code, dept, issue, channel, sla_limit, agent_load, 
            round(est_hours, 2), is_breached, actual_cost
        ))
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executemany("""
    INSERT OR REPLACE INTO operational_tickets 
    (ticket_code, department, issue_text, request_channel, sla_hours_limit, historical_agent_load, historical_resolution_hours, is_breached, actual_cost_sar)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)
    conn.commit()
    conn.close()
    print(f"Synthesized and stored {n_samples} operational records in {DB_PATH}")

def train_and_export_models():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM operational_tickets", conn)
    conn.close()
    
    # Feature Engineering
    tfidf = TfidfVectorizer(max_features=25, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df['issue_text']).toarray()
    
    dept_dummies = pd.get_dummies(df['department'], prefix='dept', drop_first=False)
    channel_dummies = pd.get_dummies(df['request_channel'], prefix='chan', drop_first=False)
    
    num_features = df[['sla_hours_limit', 'historical_agent_load']].values
    
    X = np.hstack([num_features, dept_dummies.values, channel_dummies.values, tfidf_matrix])
    y_breach = df['is_breached'].values
    y_duration = df['historical_resolution_hours'].values
    
    # Train Classification (SLA Breach Risk)
    X_train, X_test, y_b_train, y_b_test = train_test_split(X, y_breach, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    clf.fit(X_train, y_b_train)
    b_acc = clf.score(X_test, y_b_test)
    print(f"Classification (Breach Prediction) Test Accuracy: {b_acc:.3f}")
    
    # Train Regression (Expected Duration)
    _, _, y_d_train, y_d_test = train_test_split(X, y_duration, test_size=0.2, random_state=42)
    reg = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    reg.fit(X_train, y_d_train)
    d_rmse = np.sqrt(mean_squared_error(y_d_test, reg.predict(X_test)))
    print(f"Regression (Resolution Time) Test RMSE: {d_rmse:.2f} hours")
    
    # Bundle Artifacts
    bundle = {
        "clf": clf,
        "reg": reg,
        "tfidf": tfidf,
        "dept_columns": list(dept_dummies.columns),
        "chan_columns": list(channel_dummies.columns),
        "feature_names": ['sla_hours_limit', 'historical_agent_load'] + list(dept_dummies.columns) + list(channel_dummies.columns) + [f"tfidf_{w}" for w in tfidf.get_feature_names_out()]
    }
    joblib.dump(bundle, os.path.join(MODEL_DIR, "sla_risk_engine.joblib"))
    print(f"Saved trained operational model pipeline to {MODEL_DIR}/sla_risk_engine.joblib")

if __name__ == "__main__":
    init_db()
    generate_synthetic_operational_data(1500)
    train_and_export_models()
