"""
ClinicalTrialEnvironment — Core environment implementing reset/step/state.
Routes actions to task-specific graders and manages episode lifecycle.
"""

import json
import os
import uuid
from pathlib import Path

from clinical_trial_env.models import (
    ClinicalAction,
    ClinicalObservation,
    ClinicalState,
    StepResult,
    ObsResult,
)
from .graders import EligibilityGrader, AdverseEventGrader, DeviationGrader


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class ClinicalTrialEnvironment:
    """OpenEnv-compatible environment for clinical trial coordination.

    Supports 3 tasks:
    - eligibility_screening (easy): screen patients against trial criteria
    - adverse_event_triage (medium): classify AE urgency per FDA rules
    - deviation_assessment (hard): assess protocol deviations per ICH E6 GCP

    Each episode iterates over all cases for a given task.
    """

    VALID_TASKS = ["eligibility_screening", "adverse_event_triage", "deviation_assessment"]

    def __init__(self):
        self._load_data()
        self._graders = {
            "eligibility_screening": EligibilityGrader(),
            "adverse_event_triage": AdverseEventGrader(),
            "deviation_assessment": DeviationGrader(),
        }

        # Episode state
        self._task = ""
        self._episode_id = ""
        self._cases = []
        self._criteria = {}
        self._current_step = 0
        self._cumulative_reward = 0.0
        self._previous_reward = 0.0
        self._previous_feedback = ""
        self._done = True

    def _load_data(self):
        """Load all JSON datasets from the data directory."""
        with open(DATA_DIR / "patients.json", "r", encoding="utf-8") as f:
            self._patients = json.load(f)

        with open(DATA_DIR / "adverse_events.json", "r", encoding="utf-8") as f:
            self._adverse_events = json.load(f)

        with open(DATA_DIR / "deviations.json", "r", encoding="utf-8") as f:
            self._deviations = json.load(f)

        with open(DATA_DIR / "criteria.json", "r", encoding="utf-8") as f:
            self._criteria_data = json.load(f)

    def reset(self, task: str = "eligibility_screening") -> ObsResult:
        """Reset the environment for a new episode.

        Args:
            task: One of the VALID_TASKS

        Returns:
            ObsResult with the initial observation
        """
        if task not in self.VALID_TASKS:
            raise ValueError(
                f"Invalid task '{task}'. Must be one of: {self.VALID_TASKS}"
            )

        self._task = task
        self._episode_id = str(uuid.uuid4())[:8]
        self._current_step = 0
        self._cumulative_reward = 0.0
        self._previous_reward = 0.0
        self._previous_feedback = "Episode started. Evaluate the first case."
        self._done = False

        # Select cases based on task
        if task == "eligibility_screening":
            self._cases = self._patients.copy()
            self._criteria = {
                "inclusion": self._criteria_data.get("inclusion_criteria", []),
                "exclusion": self._criteria_data.get("exclusion_criteria", []),
                "trial_name": self._criteria_data.get("trial_name", ""),
                "indication": self._criteria_data.get("indication", ""),
            }
        elif task == "adverse_event_triage":
            self._cases = self._adverse_events.copy()
            self._criteria = {
                "framework": "FDA 21 CFR 312.32 — IND Safety Reporting",
                "urgency_levels": [
                    "immediate_7_day", "immediate_15_day", "7_day_report",
                    "15_day_report", "routine_monitoring", "annual_report"
                ],
            }
        elif task == "deviation_assessment":
            self._cases = self._deviations.copy()
            self._criteria = {
                "framework": "ICH E6(R2) Good Clinical Practice",
                "severity_levels": ["minor", "major", "critical"],
                "guidelines": [
                    "Deviations must be documented and reported per severity",
                    "Critical deviations require immediate IRB/sponsor notification",
                    "Major deviations require sponsor notification and corrective action",
                    "Minor deviations require documentation in deviation log",
                ],
            }

        # Build the initial observation (no ground truth exposed)
        observation = self._build_observation()
        return ObsResult(observation=observation, info={"episode_id": self._episode_id})

    def step(self, action: dict) -> StepResult:
        """Execute an action and advance to the next case.

        Args:
            action: Dictionary matching ClinicalAction fields

        Returns:
            StepResult with observation, reward, and done flag
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        if self._current_step >= len(self._cases):
            self._done = True
            obs = ClinicalObservation(
                task=self._task, done=True,
                step_number=self._current_step,
                total_cases=len(self._cases),
            )
            return StepResult(observation=obs, reward=0.0, done=True)

        # Grade the current case
        current_case = self._cases[self._current_step]
        grader = self._graders[self._task]
        reward, feedback = grader.grade(action, current_case)

        self._cumulative_reward += reward
        self._previous_reward = reward
        self._previous_feedback = feedback
        self._current_step += 1

        # Check if episode is done
        self._done = self._current_step >= len(self._cases)

        # Build next observation
        observation = self._build_observation()

        return StepResult(
            observation=observation,
            reward=reward,
            done=self._done,
            info={
                "feedback": feedback,
                "case_id": current_case.get("id", ""),
                "step": self._current_step,
            },
        )

    @property
    def state(self) -> ClinicalState:
        """Return current episode metadata."""
        return ClinicalState(
            task=self._task,
            episode_id=self._episode_id,
            current_step=self._current_step,
            total_steps=len(self._cases),
            cumulative_reward=round(self._cumulative_reward, 4),
            cases_completed=self._current_step,
            cases_remaining=max(0, len(self._cases) - self._current_step),
        )

    def _build_observation(self) -> ClinicalObservation:
        """Build an observation for the current step.

        IMPORTANT: Ground truth fields are stripped from the case data
        before sending to the agent.
        """
        if self._done or self._current_step >= len(self._cases):
            return ClinicalObservation(
                task=self._task,
                done=True,
                step_number=self._current_step,
                total_cases=len(self._cases),
                previous_reward=self._previous_reward,
                previous_feedback=self._previous_feedback,
            )

        current_case = self._cases[self._current_step]
        sanitized_case = self._strip_ground_truth(current_case)

        return ClinicalObservation(
            task=self._task,
            case_id=current_case.get("id", f"case_{self._current_step}"),
            case_data=sanitized_case,
            criteria=self._criteria,
            step_number=self._current_step,
            total_cases=len(self._cases),
            previous_reward=self._previous_reward,
            previous_feedback=self._previous_feedback,
            done=False,
        )

    def _strip_ground_truth(self, case: dict) -> dict:
        """Remove ground truth fields from case data before sending to agent."""
        gt_keys = {
            "eligible", "notes",                                    # eligibility
            "ground_truth_urgency", "ground_truth_rationale",       # adverse event
            "ground_truth_classification", "ground_truth_action",   # deviation
            "ground_truth_rationale",
        }
        return {k: v for k, v in case.items() if k not in gt_keys}

    def get_tasks(self) -> list[dict]:
        """Return available tasks metadata."""
        return [
            {
                "id": "eligibility_screening",
                "description": "Screen patient records against trial inclusion/exclusion criteria",
                "difficulty": "easy",
                "num_cases": len(self._patients),
            },
            {
                "id": "adverse_event_triage",
                "description": "Rank adverse event reports by FDA reporting urgency",
                "difficulty": "medium",
                "num_cases": len(self._adverse_events),
            },
            {
                "id": "deviation_assessment",
                "description": "Classify and respond to protocol deviations per ICH E6 GCP",
                "difficulty": "hard",
                "num_cases": len(self._deviations),
            },
        ]
