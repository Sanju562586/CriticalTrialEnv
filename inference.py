"""
Root-level inference script for the ClinicalTrialEnv agent.
Uses the HuggingFace Inference Router via OpenAI-compatible client.
"""

import os
import sys
import json
import textwrap
from typing import Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from clinical_trial_env.client import ClinicalTrialEnv

# Load .env file if present
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

TEMPERATURE = 0.1
MAX_TOKENS = 800
FALLBACK_ACTION = {"decision": "unknown", "reasoning": "Empty response", "confidence": 0.0}

TASKS = ["eligibility_screening", "adverse_event_triage", "deviation_assessment"]

SYSTEM_PROMPTS = {
    "eligibility_screening": textwrap.dedent(
        """
        You are a clinical research coordinator screening patients for the CARDIO-SHIELD Phase III trial.

        Given a patient record and trial criteria, determine if the patient is eligible.

        Respond in JSON only with this exact schema:
        {
          "decision": "eligible" or "ineligible",
          "reasoning": "detailed explanation of your decision",
          "criteria_cited": ["INC-01", "EXC-03"],
          "confidence": 0.95
        }

        Rules:
        - A patient must meet ALL inclusion criteria to be eligible
        - A patient is ineligible if they meet ANY exclusion criterion
        - Cite specific criteria IDs (INC-XX or EXC-XX) in your reasoning
        - Set confidence between 0 and 1
        """
    ).strip(),
    "adverse_event_triage": textwrap.dedent(
        """
        You are a clinical research coordinator triaging adverse events per FDA 21 CFR 312.32.

        Given an adverse event report, classify its urgency and determine the reporting timeline.

        Respond in JSON only with this exact schema:
        {
          "urgency_classification": "immediate_7_day" | "immediate_15_day" | "7_day_report" | "15_day_report" | "routine_monitoring" | "annual_report",
          "reporting_timeline": "description of reporting timeline",
          "rationale": "regulatory justification citing 21 CFR 312.32",
          "confidence": 0.85
        }

        Rules:
        - Fatal/life-threatening unexpected SAEs → immediate_7_day (phone/fax within 7 days)
        - Other serious unexpected SAEs → 15_day_report
        - Serious expected SAEs → routine_monitoring
        - Non-serious expected AEs → annual_report
        - Always cite regulatory basis
        """
    ).strip(),
    "deviation_assessment": textwrap.dedent(
        """
        You are a clinical research coordinator assessing protocol deviations per ICH E6(R2) GCP.

        Given a protocol deviation report, classify severity and recommend corrective action.

        Respond in JSON only with this exact schema:
        {
          "classification": "minor" | "major" | "critical",
          "corrective_action": "specific corrective action identifier",
          "rationale": "ICH E6 GCP justification",
          "confidence": 0.75
        }

        Valid corrective actions:
        - minor: document_log_only, document_recontact_subject, document_recollect_sample
        - major: report_irb_sponsor_immediately, report_sponsor_assess_continuation, repeat_assessment_report_sponsor, report_irb_sponsor_retrain, report_sponsor_audit_required, report_sponsor_assess_safety
        - critical: report_irb_sponsor_immediately, quarantine_ip_notify_sponsor_irb_immediately

        Rules:
        - Critical: affects subject safety or trial integrity immediately
        - Major: protocol violation requiring sponsor/IRB notification
        - Minor: documentation-level deviation with no safety impact
        - Cite specific ICH E6 sections (e.g., 4.8.1, 5.14, 4.2.5)
        """
    ).strip(),
}


def build_user_prompt(observation_dict: Dict[str, Any]) -> str:
    """Format the observation as a JSON string for the prompt."""
    return json.dumps(observation_dict, indent=2)


def parse_model_action(response_text: str) -> Dict[str, Any]:
    """Parse the JSON action from the model output safely."""
    if not response_text:
        return FALLBACK_ACTION

    content = response_text.strip()
    
    # Handle markdown code blocks
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass
        
        print(f"  ⚠ Failed to parse LLM response: {content[:100]}...")
        return {"decision": "unknown", "reasoning": content, "confidence": 0.0}


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print("🏥 ClinicalTrialEnv Agent — Inference Run")
    print("=" * 50)
    print(f"Model: {MODEL_NAME}")
    print(f"API:   {API_BASE_URL}")
    print()

    scores = {}

    for task in TASKS:
        print(f"📋 Task: {task}")
        print("-" * 40)

        with ClinicalTrialEnv(base_url="http://localhost:7860").sync() as env:
            obs_result = env.reset(task=task)
            total_reward = 0.0
            steps = 0

            while True:
                obs = obs_result.observation if hasattr(obs_result, 'observation') else obs_result
                
                obs_dict = {
                    "task": getattr(obs, "task", ""),
                    "case_id": getattr(obs, "case_id", ""),
                    "case_data": getattr(obs, "case_data", {}),
                    "criteria": getattr(obs, "criteria", {}),
                    "step_number": getattr(obs, "step_number", 0),
                    "total_cases": getattr(obs, "total_cases", 0),
                    "previous_reward": getattr(obs, "previous_reward", 0.0),
                    "previous_feedback": getattr(obs, "previous_feedback", ""),
                }

                user_prompt = build_user_prompt(obs_dict)
                system_prompt = SYSTEM_PROMPTS.get(task, f"You are a clinical research coordinator. Task: {task}. Respond in JSON only.")

                messages = [
                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
                ]

                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        response_format={"type": "json_object"},
                        stream=False,
                    )
                    response_text = completion.choices[0].message.content or ""
                except Exception as exc:
                    failure_msg = f"\n  ❌ LLM API Error: {str(exc)}"
                    print(failure_msg)
                    
                    if 'exceeded your free tier' in str(exc).lower() or 'subscribe to pro' in str(exc).lower():
                        print("  ⚠️ You have hit the Hugging Face Free Tier quota limits for serverless API.")
                        print("  ⚠️ Consider switching to a smaller model in .env or upgrading to PRO.")
                        sys.exit(1)
                    
                    response_text = ""

                action_dict = parse_model_action(response_text)
                result = env.step(action_dict)

                reward = getattr(result, "reward", 0.0)
                info = getattr(result, "info", {})
                done = getattr(result, "done", False)

                total_reward += reward
                steps += 1
                
                feedback = info.get('feedback', '')[:100] if isinstance(info, dict) else ""
                print(f"  Step {steps}: reward={reward:+.3f} | {feedback}")

                if done:
                    print(f"Episode complete for task: {task}")
                    break

                # Prepare for next iteration
                obs_result = result

        avg = round(total_reward / max(steps, 1), 3)
        scores[task] = avg
        print(f"  ✅ Average score: {avg}")
        print()

    print("=" * 50)
    print("📊 Final Scores:")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
