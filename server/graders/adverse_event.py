"""
Adverse Event Triage Grader (Medium Task)
Scores agent decisions on AE urgency classification per FDA 21 CFR 312.32.
Returns scores in [0.0, 1.0] with partial credit.
"""


class AdverseEventGrader:
    """Grades adverse event triage decisions.

    Scoring breakdown (total = 1.0):
    - Correct urgency classification:  0.40
    - Correct reporting timeline:      0.30
    - Rationale quality:               0.30
    """

    URGENCY_LEVELS = {
        "immediate_7_day": 4,
        "immediate_15_day": 3,
        "7_day_report": 3,
        "15_day_report": 2,
        "routine_monitoring": 1,
        "annual_report": 0,
    }

    TIMELINE_MAP = {
        "immediate_7_day": "7 calendar days (phone/fax) + 15 days (written)",
        "immediate_15_day": "15 calendar days",
        "7_day_report": "7 calendar days",
        "15_day_report": "15 calendar days",
        "routine_monitoring": "next scheduled report",
        "annual_report": "annual IND report",
    }

    RATIONALE_KEYWORDS = {
        "immediate_7_day": ["fatal", "life-threatening", "death", "21 cfr 312.32",
                            "ind safety", "telephone", "unexpected", "serious"],
        "immediate_15_day": ["life-threatening", "serious", "unexpected",
                             "ind safety", "hospitalization"],
        "7_day_report": ["serious", "unexpected", "ind safety", "7 day",
                         "probable", "causality"],
        "15_day_report": ["serious", "unexpected", "15 day", "sae"],
        "routine_monitoring": ["expected", "non-serious", "monitor", "protocol"],
        "annual_report": ["expected", "non-serious", "mild", "annual"],
    }

    def grade(self, action: dict, ground_truth: dict) -> tuple[float, str]:
        """Grade an adverse event triage decision.

        Args:
            action: Agent's triage dict with keys:
                urgency_classification, reporting_timeline, rationale
            ground_truth: AE record with ground_truth_urgency and ground_truth_rationale

        Returns:
            (score, feedback) tuple
        """
        score = 0.0
        feedback_parts = []

        gt_urgency = ground_truth.get("ground_truth_urgency", "")
        gt_rationale = ground_truth.get("ground_truth_rationale", "")

        # --- 1. Urgency classification (0.40) ---
        agent_urgency = self._normalize_urgency(
            str(action.get("urgency_classification", "")).strip().lower()
        )
        urgency_score = self._score_urgency(agent_urgency, gt_urgency)
        score += urgency_score
        if urgency_score >= 0.35:
            feedback_parts.append(f"Urgency: CORRECT (+{urgency_score:.2f})")
        else:
            feedback_parts.append(
                f"Urgency: {'PARTIAL' if urgency_score > 0 else 'INCORRECT'} "
                f"(+{urgency_score:.2f}). Expected '{gt_urgency}'"
            )

        # --- 2. Reporting timeline (0.30) ---
        agent_timeline = str(action.get("reporting_timeline", "")).strip().lower()
        timeline_score = self._score_timeline(agent_timeline, gt_urgency)
        score += timeline_score
        feedback_parts.append(f"Timeline: +{timeline_score:.2f}/0.30")

        # --- 3. Rationale quality (0.30) ---
        rationale = str(action.get("rationale", "")).strip().lower()
        rationale_score = self._score_rationale(rationale, gt_urgency, gt_rationale)
        score += rationale_score
        feedback_parts.append(f"Rationale: +{rationale_score:.2f}/0.30")

        score = max(0.0, min(1.0, score))
        return score, " | ".join(feedback_parts)

    def _normalize_urgency(self, urgency: str) -> str:
        """Map various urgency descriptions to canonical values."""
        urgency = urgency.replace(" ", "_").replace("-", "_")
        if "immediate" in urgency and "7" in urgency:
            return "immediate_7_day"
        if "immediate" in urgency and "15" in urgency:
            return "immediate_15_day"
        if "7_day" in urgency or "7day" in urgency:
            return "7_day_report"
        if "15_day" in urgency or "15day" in urgency:
            return "15_day_report"
        if "routine" in urgency or "monitor" in urgency:
            return "routine_monitoring"
        if "annual" in urgency:
            return "annual_report"
        return urgency

    def _score_urgency(self, agent: str, ground_truth: str) -> float:
        """Score urgency with partial credit for near-misses."""
        if agent == ground_truth:
            return 0.40

        agent_level = self.URGENCY_LEVELS.get(agent, -1)
        gt_level = self.URGENCY_LEVELS.get(ground_truth, -1)

        if agent_level < 0 or gt_level < 0:
            return 0.0

        diff = abs(agent_level - gt_level)
        if diff == 1:
            return 0.20  # One level off
        if diff == 2:
            return 0.08  # Two levels off

        # Penalize under-triaging more than over-triaging
        if agent_level < gt_level:
            return 0.0  # Under-triaging is dangerous
        return 0.05

    def _score_timeline(self, timeline: str, gt_urgency: str) -> float:
        """Score reporting timeline answer."""
        if not timeline:
            return 0.0

        expected = self.TIMELINE_MAP.get(gt_urgency, "")
        expected_lower = expected.lower()

        # Check for key time periods
        if gt_urgency in ("immediate_7_day",):
            if "7" in timeline and ("day" in timeline or "calendar" in timeline):
                return 0.30
            if "15" in timeline:
                return 0.10
        elif gt_urgency in ("immediate_15_day", "15_day_report"):
            if "15" in timeline and ("day" in timeline or "calendar" in timeline):
                return 0.30
            if "7" in timeline:
                return 0.15
        elif gt_urgency == "7_day_report":
            if "7" in timeline:
                return 0.30
        elif gt_urgency == "routine_monitoring":
            if "routine" in timeline or "monitor" in timeline or "next" in timeline:
                return 0.30
        elif gt_urgency == "annual_report":
            if "annual" in timeline or "yearly" in timeline:
                return 0.30

        # Generic partial credit
        if any(word in timeline for word in expected_lower.split()):
            return 0.10

        return 0.0

    def _score_rationale(self, rationale: str, gt_urgency: str, gt_rationale: str) -> float:
        """Score rationale quality based on regulatory keyword coverage."""
        if not rationale or len(rationale) < 10:
            return 0.0

        keywords = self.RATIONALE_KEYWORDS.get(gt_urgency, [])
        if not keywords:
            return 0.10  # Some credit for attempting

        hits = sum(1 for kw in keywords if kw in rationale)
        coverage = hits / len(keywords)

        # Base score from keyword coverage
        base_score = coverage * 0.20

        # Bonus for citing regulatory references
        reg_bonus = 0.0
        if "21 cfr" in rationale or "312.32" in rationale:
            reg_bonus += 0.05
        if "ich" in rationale or "e6" in rationale:
            reg_bonus += 0.03
        if "ind" in rationale:
            reg_bonus += 0.02

        return min(base_score + reg_bonus, 0.30)
