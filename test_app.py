import unittest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

class TestOperationalEngine(unittest.TestCase):
    def test_health_check(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_high_risk_prediction(self):
        payload = {
            "ticket_code": "TICK-TEST-001",
            "department": "Finance",
            "issue_text": "Delay in corporate invoice reconciliation and tax compliance audit milestone breach.",
            "request_channel": "ERP Integration",
            "sla_hours_limit": 12.0,
            "current_agent_load": 30
        }
        response = client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("breach_probability", data)
        self.assertIn("expected_resolution_hours", data)
        self.assertTrue(data["breach_probability"] >= 0.0)
        self.assertIn(data["breach_risk_category"], ["HIGH_CRITICAL", "MEDIUM_ATTENTION", "LOW_STABLE"])
        self.assertIn("recommended_business_action", data)

    def test_low_risk_prediction(self):
        payload = {
            "ticket_code": "TICK-TEST-002",
            "department": "IT Infrastructure",
            "issue_text": "Routine password policy update and role-based permissions sync.",
            "request_channel": "Portal",
            "sla_hours_limit": 72.0,
            "current_agent_load": 3
        }
        response = client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("breach_probability", data)
        self.assertEqual(data["breach_risk_category"], "LOW_STABLE")

    def test_metrics_endpoint(self):
        response = client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("dataset_total_records", data)
        self.assertIn("recent_inferences_analyzed", data)

if __name__ == "__main__":
    unittest.main()
