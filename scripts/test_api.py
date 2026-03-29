"""Quick test script for ClinicalTrialEnv API."""
import requests
import json

BASE = "http://localhost:7860"

# Test health
print("=== HEALTH ===")
r = requests.get(f"{BASE}/health")
print(json.dumps(r.json(), indent=2))
print()

# Test tasks
print("=== TASKS ===")
r = requests.get(f"{BASE}/tasks")
print(json.dumps(r.json(), indent=2))
print()

# Test reset
print("=== RESET (eligibility_screening) ===")
r = requests.post(f"{BASE}/reset", json={"task": "eligibility_screening"})
data = r.json()
print(f"Case ID: {data['observation']['case_id']}")
print(f"Total cases: {data['observation']['total_cases']}")
print()

# Test step — correct decision for P001 (eligible)
print("=== STEP 1 ===")
action = {
    "decision": "eligible",
    "reasoning": "Patient meets all inclusion criteria. Age 67 is 18-80, NYHA III, LVEF 28 percent lte 40, stable GDMT 8 weeks, NT-proBNP 1250.",
    "criteria_cited": ["INC-01", "INC-02", "INC-03", "INC-04", "INC-05"],
    "confidence": 0.95
}
r = requests.post(f"{BASE}/step", json=action)
step = r.json()
print(f"Reward: {step['reward']}")
print(f"Done: {step['done']}")
print(f"Feedback: {step['info'].get('feedback', '')}")
print()

# Test state
print("=== STATE ===")
r = requests.get(f"{BASE}/state")
print(json.dumps(r.json(), indent=2))
print()

# Test adverse event triage
print("=== RESET (adverse_event_triage) ===")
r = requests.post(f"{BASE}/reset", json={"task": "adverse_event_triage"})
data = r.json()
print(f"Case ID: {data['observation']['case_id']}")
print(f"Total cases: {data['observation']['total_cases']}")
print()

# Step on AE001 — 7_day_report
print("=== STEP AE ===")
ae_action = {
    "urgency_classification": "7_day_report",
    "reporting_timeline": "7 calendar days IND Safety Report per 21 CFR 312.32(c)(1)",
    "rationale": "Serious unexpected AE with probable causality. Requires IND Safety Report within 7 calendar days. 21 CFR 312.32.",
    "confidence": 0.85
}
r = requests.post(f"{BASE}/step", json=ae_action)
step = r.json()
print(f"Reward: {step['reward']}")
print(f"Feedback: {step['info'].get('feedback', '')}")
print()

# Test deviation assessment
print("=== RESET (deviation_assessment) ===")
r = requests.post(f"{BASE}/reset", json={"task": "deviation_assessment"})
data = r.json()
print(f"Case ID: {data['observation']['case_id']}")
print(f"Total cases: {data['observation']['total_cases']}")
print()

# Step on DEV001 — major consent deviation
print("=== STEP DEV ===")
dev_action = {
    "classification": "major",
    "corrective_action": "report_irb_sponsor_immediately",
    "rationale": "ICH E6 4.8.1 requires consent before trial procedures. Subject safety compromised. Major GCP violation.",
    "confidence": 0.9
}
r = requests.post(f"{BASE}/step", json=dev_action)
step = r.json()
print(f"Reward: {step['reward']}")
print(f"Feedback: {step['info'].get('feedback', '')}")

print()
print("ALL TESTS PASSED!")
