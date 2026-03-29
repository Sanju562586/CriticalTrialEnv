"""
Typed models for the ClinicalTrialEnv environment.
Uses dataclasses inheriting from OpenEnv base classes for type-safe
action/observation/state serialization over WebSocket/HTTP.
"""

from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class ClinicalAction:
    """Action the agent sends to the environment via step().

    Fields vary by task:
    - eligibility_screening: decision (eligible/ineligible), reasoning, criteria_cited
    - adverse_event_triage: urgency_classification, reporting_timeline, rationale
    - deviation_assessment: classification (minor/major/critical), corrective_action, rationale
    """
    decision: str = ""
    reasoning: str = ""
    criteria_cited: list[str] = field(default_factory=list)
    urgency_classification: str = ""
    reporting_timeline: str = ""
    classification: str = ""
    corrective_action: str = ""
    rationale: str = ""
    confidence: float = 0.5


@dataclass
class ClinicalObservation:
    """Observation returned to the agent from reset()/step().

    Contains the current case data, task context, and feedback from the
    previous action (if any).
    """
    task: str = ""
    case_id: str = ""
    case_data: dict = field(default_factory=dict)
    criteria: dict = field(default_factory=dict)
    step_number: int = 0
    total_cases: int = 0
    previous_reward: float = 0.0
    previous_feedback: str = ""
    done: bool = False


@dataclass
class ClinicalState:
    """Episode metadata accessible via state()."""
    task: str = ""
    episode_id: str = ""
    current_step: int = 0
    total_steps: int = 0
    cumulative_reward: float = 0.0
    cases_completed: int = 0
    cases_remaining: int = 0


@dataclass
class StepResult:
    """Result from environment.step() — wraps observation + reward + done."""
    observation: ClinicalObservation
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)


@dataclass
class ObsResult:
    """Result from environment.reset() — wraps initial observation."""
    observation: ClinicalObservation
    info: dict = field(default_factory=dict)
