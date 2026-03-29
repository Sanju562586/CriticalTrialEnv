"""
ClinicalTrialEnv Client — HTTP client for interacting with the environment server.
Wraps REST endpoints into a clean Python API matching OpenEnv patterns.
"""

import json
import requests
from dataclasses import dataclass
from typing import Optional

from .models import ClinicalObservation, ClinicalState, StepResult, ObsResult


class ClinicalTrialEnv:
    """HTTP client for the ClinicalTrialEnv server.

    Usage:
        with ClinicalTrialEnv(base_url="http://localhost:7860").sync() as env:
            obs = env.reset(task="eligibility_screening")
            while True:
                action = agent.act(obs.observation)
                result = env.step(action)
                if result.done:
                    break
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._session: Optional[requests.Session] = None

    def sync(self):
        """Return self for context manager usage (sync mode)."""
        return self

    def __enter__(self):
        self._session = requests.Session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            self._session.close()
            self._session = None

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def health(self) -> dict:
        """Check server health."""
        resp = self._get_session().get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def get_tasks(self) -> list[dict]:
        """List available tasks."""
        resp = self._get_session().get(f"{self.base_url}/tasks")
        resp.raise_for_status()
        return resp.json().get("tasks", [])

    def reset(self, task: str = "eligibility_screening") -> ObsResult:
        """Reset environment for a new episode.

        Args:
            task: Task ID (eligibility_screening, adverse_event_triage, deviation_assessment)

        Returns:
            ObsResult with initial observation
        """
        resp = self._get_session().post(
            f"{self.base_url}/reset",
            json={"task": task},
        )
        resp.raise_for_status()
        data = resp.json()

        observation = ClinicalObservation(**data["observation"])
        return ObsResult(observation=observation, info=data.get("info", {}))

    def step(self, action: dict) -> StepResult:
        """Submit an action and receive reward.

        Args:
            action: Action dictionary with task-appropriate fields

        Returns:
            StepResult with observation, reward, and done flag
        """
        resp = self._get_session().post(
            f"{self.base_url}/step",
            json=action,
        )
        resp.raise_for_status()
        data = resp.json()

        observation = ClinicalObservation(**data["observation"])
        return StepResult(
            observation=observation,
            reward=data.get("reward", 0.0),
            done=data.get("done", False),
            info=data.get("info", {}),
        )

    def get_state(self) -> ClinicalState:
        """Get current episode metadata."""
        resp = self._get_session().get(f"{self.base_url}/state")
        resp.raise_for_status()
        data = resp.json()
        return ClinicalState(**data)
