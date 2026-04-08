"""
scripts/generate_all.py — Generate clinical trial data using Google Gemini 2.0 Flash.

Free API key from: https://aistudio.google.com → Get API Key (no card required)

Generates:
  - data/patients.json       (50 patient records)
  - data/adverse_events.json (40 adverse event reports)
  - data/deviations.json     (30 protocol deviations)
  - data/criteria.json       (trial criteria)

Usage:
    export GEMINI_API_KEY=your_key_here
    python scripts/generate_all.py

Note: The data/ directory already contains pre-baked datasets.
      This script is provided for regenerating or expanding data.
"""

import os
import sys
import json
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

api_key = os.environ.get("GEMINI_API_KEY")
client = OpenAI(
    api_key=api_key if api_key else "missing_key",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


import time

def generate(prompt: str) -> dict | list:
    """Generate JSON data from Gemini via OpenAI SDK."""
    try:
        # Give a small 4-second delay before making requests since Gemini Free Tier is 15 RPM
        time.sleep(4)
        response = client.chat.completions.create(
            model="gemini-2.0-flash",  # Reverted to 2.0-flash (1.5-flash returned 404 on OpenAI layer)
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  [ERROR] Failed to generate: {e}")
        if "429" in str(e) and "limit: 0" in str(e):
            print("\n  [!] CRITICAL QUOTA ERROR: Your Google account has exactly 0 free limit.")
            print("  This typically happens if you reside in the UK/EU/Switzerland where Google disables the Free Tier.")
            print("  To fix: You must set up a GCP Billing Account with a credit card, OR use a VPN/US account,")
            print("          OR switch back to a local model like Ollama.")
        raise



def generate_patients(count: int = 50):
    """Generate patient records for eligibility screening."""
    print(f"[GEN] Generating {count} patient records...")
    prompt = f"""Generate exactly {count} realistic patient records for a heart failure clinical trial (CARDIO-SHIELD Phase III).

Each patient must have these fields:
- id: "P001" through "P{count:03d}"
- name: realistic full name
- age: integer 18-90
- sex: "M" or "F"
- nyha_class: "I", "II", "III", or "IV"
- lvef_percent: integer 10-60
- stable_gdmt_weeks: integer 0-30
- nt_probnp: integer 100-6000
- egfr: integer 10-120
- potassium: float 3.0-6.5
- days_since_cardiac_event: integer 30-1000
- active_malignancy: boolean
- pregnancy_risk: boolean
- drug_allergy: boolean
- systolic_bp: integer 70-160
- hemoglobin: float 7.0-17.0
- comorbidities: list of strings
- medications: list of strings
- eligible: boolean (true if meets ALL inclusion and NO exclusion criteria)
- notes: string explaining eligibility decision with criteria IDs

Inclusion: Age 18-80, NYHA II-IV, LVEF<=40, stable GDMT>=4wk, NT-proBNP>=400
Exclusion: eGFR<20, K>5.5, cardiac event<90d, malignancy, pregnancy, drug allergy, SBP<90, Hgb<9.0

Make roughly 50% eligible and 50% ineligible with varied exclusion reasons.
Return as a JSON array."""

    patients = generate(prompt)
    with open(DATA_DIR / "patients.json", "w", encoding="utf-8") as f:
        json.dump(patients, f, indent=2)
    print(f"  [OK] Saved {len(patients)} patients")


def generate_adverse_events(count: int = 40):
    """Generate adverse event reports."""
    print(f"[GEN] Generating {count} adverse event reports...")
    prompt = f"""Generate exactly {count} realistic adverse event reports for a clinical trial.

Each AE must have these fields:
- id: "AE001" through "AE{count:03d}"
- subject_id: "S101" through "S{100+count}"
- event: description of the adverse event
- onset_date: ISO date string in 2024
- severity: "Grade 1" through "Grade 5" (CTCAE)
- causality: "Definite", "Probable", "Possible", "Unlikely", "Unrelated"
- seriousness: "serious" or "non-serious"
- requires_hospitalization: boolean
- life_threatening: boolean
- expected: boolean (was this AE listed in the IB)
- urgency: the agent's field (leave empty string)
- category: organ system category
- outcome: "resolved", "recovering", "ongoing", "fatal", "unknown"
- reporter: name with title
- ground_truth_urgency: one of "immediate_7_day", "immediate_15_day", "7_day_report", "15_day_report", "routine_monitoring", "annual_report"
- ground_truth_rationale: explanation citing FDA 21 CFR 312.32

Rules for ground_truth_urgency:
- immediate_7_day: fatal or life-threatening unexpected SAE
- 15_day_report or 7_day_report: serious unexpected SAE
- routine_monitoring: serious expected or non-serious unexpected with moderate severity
- annual_report: non-serious expected AE

Mix of all urgency levels. Return as a JSON array."""

    aes = generate(prompt)
    with open(DATA_DIR / "adverse_events.json", "w", encoding="utf-8") as f:
        json.dump(aes, f, indent=2)
    print(f"  [OK] Saved {len(aes)} adverse events")


def generate_deviations(count: int = 30):
    """Generate protocol deviation reports."""
    print(f"[GEN] Generating {count} protocol deviations...")
    prompt = f"""Generate exactly {count} realistic protocol deviation reports for a clinical trial.

Each deviation must have these fields:
- id: "DEV001" through "DEV{count:03d}"
- site_id: "SITE-01" through "SITE-10"
- description: detailed deviation description
- category: one of "consent", "procedure", "ip_management", "concomitant_medication", "assessment", "visit_compliance", "blinding", "delegation", "documentation", "dosing", "laboratory", "safety_reporting"
- severity: "minor", "major", or "critical"
- detection_date: ISO date in 2024
- protocol_section: relevant protocol section reference
- ground_truth_classification: "minor", "major", or "critical"
- ground_truth_action: one of the valid corrective actions
- ground_truth_rationale: explanation citing ICH E6 GCP sections

Valid corrective actions:
minor: document_log_only, document_recontact_subject, document_recollect_sample
major: report_irb_sponsor_immediately, report_sponsor_assess_continuation, repeat_assessment_report_sponsor, report_irb_sponsor_retrain, report_sponsor_audit_required, report_sponsor_assess_safety
critical: report_irb_sponsor_immediately, quarantine_ip_notify_sponsor_irb_immediately

Mix of minor/major/critical. Return as a JSON array."""

    devs = generate(prompt)
    with open(DATA_DIR / "deviations.json", "w", encoding="utf-8") as f:
        json.dump(devs, f, indent=2)
    print(f"  [OK] Saved {len(devs)} deviations")


def main():
    print("[APP] ClinicalTrialEnv Data Generator")
    print("=" * 40)
    print(f"Using Gemini via OpenAI API (gemini-2.0-flash)")
    print(f"Output: {DATA_DIR}")
    print()

    if not os.environ.get("GEMINI_API_KEY"):
        print("[ERROR] GEMINI_API_KEY environment variable is missing.")
        print("Please provide a valid API key in .env")
        return

    generate_patients(20)
    generate_adverse_events(20)
    generate_deviations(20)

    print()
    print("[OK] All data generated successfully!")
    print("   You can now start the server with: uvicorn server.app:app --port 7860")


if __name__ == "__main__":
    main()
