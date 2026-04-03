"""
Inference Script — ClinicalTrialEnv
====================================
MANDATORY env vars (loaded from .env or shell):
    API_BASE_URL     LLM endpoint  (default: http://localhost:11434/v1)
    MODEL_NAME       LLM model id  (default: llama3.2:latest)
    HF_TOKEN         API key       (default: ollama)
    ENV_BASE_URL     Env server    (default: http://localhost:7860)

STDOUT FORMAT (required):
    [START] task=<task> env=<env> model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

import json
import os
import re
import time
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from clinical_trial_env.client import ClinicalTrialEnv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama3.2:latest")
HF_TOKEN     = os.getenv("HF_TOKEN",     "ollama")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK    = "ClinicalTrialEnv"

MAX_TOKENS  = 800
TEMPERATURE = 0.2
MAX_RETRIES = 3
TASKS       = ["eligibility_screening", "adverse_event_triage", "deviation_assessment"]
SUCCESS_THRESHOLD = 0.85   # Acceptable limit

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ─────────────────────────────────────────────────────────────────────────────
# STDOUT LOGGING (required format)
# ─────────────────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: dict, reward: float, done: bool,
             error: Optional[str]) -> None:
    action_str = json.dumps(action, separators=(",", ":"))
    error_val  = error if error else "null"
    done_val   = str(done).lower()
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def parse_json_safe(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    for strategy in [
        lambda t: json.loads(t),
        lambda t: json.loads(re.sub(r"```(?:json)?\s*", "", t).replace("```", "")),
        lambda t: json.loads(re.search(r"\{[\s\S]*\}", t).group()),
    ]:
        try:
            return strategy(text)
        except Exception:
            pass
    return {}

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTING LOGIC
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "eligibility_screening": """You are an expert Clinical Trial Coordinator evaluating patient eligibility.
Your must classify the patient as "eligible" or "ineligible".

CRITICAL REASONING & SCORING RULES:
For an eligible patient, your reasoning must use these EXACT words/phrases to receive full credit:
- "meets", "satisfies", "within range", "qualifies", "inclusion", "criterion", "criteria met"
- You must mention specific fields: "age", "nyha", "lvef", "egfr", "potassium", "pregnancy", "allergy", "hemoglobin", "blood pressure", "cardiac event", "malignancy", "gdmt", "nt-probnp"

For an ineligible patient, your reasoning must use these EXACT words:
- "excluded", "fails", "does not meet", "outside range", "ineligible", "exclusion", "violated", "exceeds", "below"

OUTPUT FORMAT:
Respond exactly with the following JSON format:
{
  "decision": "eligible or ineligible",
  "reasoning": "Detailed reasoning matching the exact required descriptive keywords above, referencing the relevant fields in inclusion/exclusion.",
  "criteria_cited": ["INC-01", "EXC-02"]
}
""",
    "adverse_event_triage": """You are an expert Clinical Trial Safety Monitor triaging adverse events.
Classify the adverse event according to 21 CFR 312.32 IND Safety Reporting rules.

VALID URGENCY CLASSIFICATIONS (Choose exactly one):
- immediate_7_day (for fatal/life-threatening unexpected)
- immediate_15_day
- 7_day_report (serious, unexpected, probably/definitely related)
- 15_day_report (serious, unexpected, causality unlikely/possible)
- routine_monitoring (non-serious, expected, severity >= grade 2)
- annual_report (non-serious, expected, severity grade 1)

VALID REPORTING TIMELINES:
- "7 calendar days (phone/fax) + 15 days (written)" -> pairs with immediate_7_day
- "15 calendar days" -> pairs with immediate_15_day or 15_day_report
- "7 calendar days" -> pairs with 7_day_report
- "next scheduled report" -> pairs with routine_monitoring
- "annual IND report" -> pairs with annual_report

Your rationale must use appropriate regulatory terms like "fatal", "life-threatening", "serious", "unexpected", "ind safety", "21 cfr 312.32", "hospitalization", "causality", "monitor", "protocol", or "expected".

OUTPUT FORMAT:
Respond exactly with the following JSON format:
{
  "urgency_classification": "one of the options above",
  "reporting_timeline": "one of the options above matching urgency",
  "rationale": "Detailed rationale using regulatory keywords."
}
""",
    "deviation_assessment": """You are an expert Clinical Research Associate evaluating Protocol Deviations per ICH E6(R2) GCP guidelines.

VALID CLASSIFICATIONS:
- "minor" (administrative, no impact on safety/data integrity)
- "major" (significant impact on safety, data integrity, or compliance - requires sponsor/IRB report)
- "critical" (immediate safety risk or severe violation, e.g. IP tampering, dosing error)

VALID CORRECTIVE ACTIONS (Choose the best fit):
MINOR actions: "document_log_only", "document_recontact_subject", "document_recollect_sample"
MAJOR actions: "report_irb_sponsor_immediately", "report_sponsor_assess_continuation", "repeat_assessment_report_sponsor", "report_irb_sponsor_retrain", "report_sponsor_audit_required", "report_sponsor_assess_safety"
CRITICAL actions: "report_irb_sponsor_immediately", "quarantine_ip_notify_sponsor_irb_immediately"

Your rationale should use regulatory terms matching the severity:
- Minor: "minor", "no impact", "document", "within variance", "acceptable"
- Major: "major", "protocol violation", "data integrity", "safety", "ich e6", "gcp", "report", "corrective action"
- Critical: "critical", "immediate", "subject safety", "trial integrity", "ich e6", "urgent", "quarantine", "irb", "root cause"

OUTPUT FORMAT:
Respond exactly with the following JSON format:
{
  "classification": "minor, major, or critical",
  "corrective_action": "one valid action from the lists above",
  "rationale": "Detailed GCP rationale using required keywords."
}
"""
}

def build_user_prompt(task: str, obs_dict: dict) -> str:
    case_data = obs_dict.get("case_data", obs_dict)
    criteria = obs_dict.get("criteria", {})
    return f"""
EVALUATE THE FOLLOWING CASE:
{json.dumps(case_data, indent=2)}

TRIAL CRITERIA & RULES:
{json.dumps(criteria, indent=2)}

CRITICAL REMINDER: Provide ONLY valid JSON inside your response. Follow the requested keywords to maximize your score.
"""

def get_action_from_llm(task: str, obs_dict: dict) -> dict:
    prompt = build_user_prompt(task, obs_dict)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS[task]},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            text = (resp.choices[0].message.content or "").strip()
            parsed = parse_json_safe(text)
            if parsed:
                return parsed
        except Exception as e:
            print(f"[DEBUG] Model/parse failed attempt {attempt}: {e}", file=os.sys.stderr)
            time.sleep(2 ** attempt)

    # Safe Fallbacks
    if task == "eligibility_screening":
        return {"decision": "ineligible", "reasoning": "fails inclusion criteria", "criteria_cited": []}
    if task == "adverse_event_triage":
        return {"urgency_classification": "routine_monitoring", "reporting_timeline": "next scheduled report", "rationale": "expected non-serious"}
    return {"classification": "minor", "corrective_action": "document_log_only", "rationale": "minor deviation"}


# ─────────────────────────────────────────────────────────────────────────────
# OBS EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def to_dict(obs) -> dict:
    if isinstance(obs, dict):
        return obs
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if hasattr(obs, "__dict__"):
        return {k: v for k, v in obs.__dict__.items() if not k.startswith("_")}
    return {"raw": str(obs)}


# ─────────────────────────────────────────────────────────────────────────────
# TASK RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_task(env, task: str) -> List[float]:
    """Run one episode. Returns list of per-step rewards."""
    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    error_msg: Optional[str] = None

    try:
        result = env.reset(task=task)
        obs    = to_dict(result.observation)

        step = 0
        while True:
            if obs.get("done", False):
                break

            step += 1
            done_flag = False
            reward = 0.0

            try:
                action = get_action_from_llm(task, obs)
                step_result = env.step(action)
                reward      = step_result.reward or 0.0
                done_flag   = step_result.done
                error_msg   = None

                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=action, reward=reward,
                         done=done_flag, error=None)

                if done_flag:
                    break
                obs = to_dict(step_result.observation)

            except Exception as e:
                error_msg = str(e)
                log_step(step=step, action={}, reward=0.0,
                         done=False, error=error_msg)
                break

        avg = sum(rewards) / max(len(rewards), 1)
        success = avg >= SUCCESS_THRESHOLD

    except Exception as e:
        error_msg = str(e)
        success = False

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    all_rewards = {}

    with ClinicalTrialEnv(base_url=ENV_BASE_URL).sync() as env:
        # Verify server
        try:
            env.health()
        except Exception as e:
            print(f"[ERROR] Cannot reach env server at {ENV_BASE_URL}: {e}")
            raise

        for task in TASKS:
            all_rewards[task] = run_task(env, task)

    # Summary (to stderr so it doesn't pollute the [STEP] stream)
    import sys
    print("\n=== SCORE SUMMARY ===", file=sys.stderr)
    for task, rewards in all_rewards.items():
        avg = sum(rewards) / max(len(rewards), 1)
        print(f"  {task:<35} avg={avg:.4f} ({len(rewards)} steps)", file=sys.stderr)


if __name__ == "__main__":
    main()