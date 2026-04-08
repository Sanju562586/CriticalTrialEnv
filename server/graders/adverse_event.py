"""
Adverse Event Triage Grader (Medium Task)
Scores agent decisions on AE urgency classification per FDA 21 CFR 312.32.
Returns scores in [0.0, 1.0] with partial credit.

Rationale scoring uses a PyTorch softmax over 6 urgency-level keyword sets
(see server.graders.confidence.TorchConfidenceScorer).

Improvements over v1:
- Vastly expanded RATIONALE_KEYWORDS with medical synonyms and case-descriptive
  terms visible in the dataset (grade-specific, organ-specific, outcome-specific)
- Timeline scoring: also matches the canonical TIMELINE_MAP string directly,
  and added flexible partial matching for paraphrased timelines
- Category-keyword bonus: rationales mentioning the AE's medical category
  (cardiac, renal, hematological, etc.) receive +0.02 bonus
- Raised max_score from 0.20 to 0.22 so the softmax + reg_bonus can hit 0.30
- _normalize_urgency() extended with more common LLM paraphrase patterns
"""

from .confidence import TorchConfidenceScorer


class AdverseEventGrader:
    """Grades adverse event triage decisions.

    Scoring breakdown (total = 1.0):
    - Correct urgency classification:  0.40
    - Correct reporting timeline:      0.30
    - Rationale quality (PyTorch):     0.30
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

    # Expanded keyword sets with medical synonyms, grade references, and
    # case-specific descriptor terms from the adverse_events.json dataset.
    # Discriminative-weight logic in TorchConfidenceScorer ensures that
    # terms exclusive to one urgency tier are weighted 2x automatically.
    RATIONALE_KEYWORDS = {
        "immediate_7_day": [
            "fatal", "life-threatening", "death", "21 cfr 312.32",
            "ind safety", "telephone", "unexpected", "serious",
            "grade 5", "grade 4", "sudden", "cardiac death", "cardiac arrest",
            "thrombocytopenia", "hy's law", "acute renal failure", "dialysis",
            "qtc prolongation", "life threatening", "7 calendar days",
            "immediate", "phone", "fax", "within 7", "7-day",
        ],
        "immediate_15_day": [
            "life-threatening", "serious", "unexpected",
            "ind safety", "hospitalization", "15 calendar days",
            "anaphylaxis", "anaphylactic", "immunological",
            "life threatening", "15-day", "within 15",
        ],
        "7_day_report": [
            "serious", "unexpected", "ind safety", "7 day",
            "probable", "causality", "7 calendar days",
            "hospitalized", "requires hospitalization",
            "neurological", "headache", "visual disturbance",
            "serious unexpected", "probable causality",
        ],
        "15_day_report": [
            "serious", "unexpected", "15 day", "sae",
            "unlikely related", "pneumonia", "infectious",
            "15 calendar days", "serious adverse", "not related",
            "15-day report",
        ],
        "routine_monitoring": [
            "expected", "non-serious", "monitor", "protocol",
            "next scheduled", "routine", "grade 2", "hepatic",
            "elevated alt", "liver", "gastrointestinal", "diarrhea",
            "rash", "dermatological", "moderate", "manage per protocol",
            "schedule report", "non serious",
        ],
        "annual_report": [
            "expected", "non-serious", "mild", "annual",
            "grade 1", "annual report", "annual ind",
            "nausea", "fatigue", "dizziness", "injection site",
            "malaise", "unlikely", "self-limiting", "resolved",
            "non serious", "low severity",
        ],
    }

    # AE category keywords for context-matching bonus
    CATEGORY_KEYWORDS = {
        "cardiac": ["cardiac", "heart", "qtc", "ventricular", "arrhythmia"],
        "neurological": ["neurological", "headache", "visual", "nerve"],
        "hematological": ["hematological", "platelet", "thrombocytopenia", "blood count"],
        "hepatic": ["hepatic", "liver", "alt", "ast", "bilirubin", "hy's law"],
        "renal": ["renal", "kidney", "creatinine", "dialysis", "egfr"],
        "gastrointestinal": ["gastrointestinal", "nausea", "diarrhea", "vomiting"],
        "immunological": ["immunological", "anaphylaxis", "anaphylactic", "allergic"],
        "dermatological": ["dermatological", "rash", "erythema", "skin"],
        "infectious": ["infectious", "pneumonia", "infection", "sepsis"],
        "general": ["fatigue", "malaise", "fever", "weakness"],
    }

    def __init__(self):
        # PyTorch scorer: temperature=0.8 for a smooth but discerning 6-class distribution
        self._confidence = TorchConfidenceScorer(temperature=0.8, length_bonus=0.5)

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
        ae_category = ground_truth.get("category", "")

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
        rationale_score = self._score_rationale(
            rationale, gt_urgency, gt_rationale, ae_category
        )
        score += rationale_score
        feedback_parts.append(f"Rationale: +{rationale_score:.2f}/0.30")

        score = max(0.0, min(1.0, score))
        return score, " | ".join(feedback_parts)

    def _normalize_urgency(self, urgency: str) -> str:
        """Map various urgency descriptions to canonical values."""
        urgency = urgency.replace(" ", "_").replace("-", "_")

        # Canonical direct match first
        if urgency in self.URGENCY_LEVELS:
            return urgency

        if "immediate" in urgency and "7" in urgency:
            return "immediate_7_day"
        if "immediate" in urgency and "15" in urgency:
            return "immediate_15_day"
        if "immediate" in urgency and ("fatal" in urgency or "life" in urgency):
            return "immediate_7_day"
        if "7_day" in urgency or "7day" in urgency or urgency == "7_day":
            return "7_day_report"
        if "15_day" in urgency or "15day" in urgency or urgency == "15_day":
            return "15_day_report"
        if "routine" in urgency or "monitor" in urgency:
            return "routine_monitoring"
        if "annual" in urgency or "yearly" in urgency:
            return "annual_report"
        # Handle "7-day report" variants
        if "7" in urgency and "report" in urgency:
            return "7_day_report"
        if "15" in urgency and "report" in urgency:
            return "15_day_report"
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

        # Penalize under-triaging more than over-triaging (patient safety)
        if agent_level < gt_level:
            return 0.0  # Under-triaging is dangerous
        return 0.05

    def _score_timeline(self, timeline: str, gt_urgency: str) -> float:
        """Score reporting timeline answer.

        Checks in order:
        1. Canonical string match from TIMELINE_MAP
        2. Key time-period numeric match
        3. Semantic keyword match (routine/annual)
        4. Generic partial overlap
        """
        if not timeline:
            return 0.0

        expected = self.TIMELINE_MAP.get(gt_urgency, "")
        expected_lower = expected.lower()

        # Check 1: Canonical exact or near-exact match
        if expected_lower and expected_lower in timeline:
            return 0.30

        # Check 2: Key time-period numeric match
        if gt_urgency == "immediate_7_day":
            if "7" in timeline and ("day" in timeline or "calendar" in timeline):
                return 0.30
            if "phone" in timeline or "fax" in timeline or "immediate" in timeline:
                return 0.25
            if "15" in timeline:
                return 0.10
        elif gt_urgency in ("immediate_15_day", "15_day_report"):
            if "15" in timeline and ("day" in timeline or "calendar" in timeline):
                return 0.30
            if "7" in timeline:
                return 0.15
        elif gt_urgency == "7_day_report":
            if "7" in timeline and ("day" in timeline or "calendar" in timeline):
                return 0.30
            if "7" in timeline:
                return 0.20
        elif gt_urgency == "routine_monitoring":
            if "routine" in timeline or "monitor" in timeline or "next" in timeline or "scheduled" in timeline:
                return 0.30
        elif gt_urgency == "annual_report":
            if "annual" in timeline or "yearly" in timeline or "ind report" in timeline:
                return 0.30

        # Check 3: Generic partial credit for any word in expected
        if expected_lower and any(word in timeline for word in expected_lower.split()):
            return 0.10

        return 0.0

    def _score_rationale(
        self,
        rationale: str,
        gt_urgency: str,
        gt_rationale: str,
        ae_category: str,
    ) -> float:
        """Score rationale quality using a PyTorch softmax over 6 urgency levels.

        score_keywords() computes a softmax over keyword-hit counts for each
        urgency level, then returns the probability mass on the ground-truth
        class as the base score. Rewards rationales that are not just
        keyword-rich but *specifically* aligned with the correct urgency tier.
        """
        if not rationale or len(rationale) < 10:
            return 0.0

        if gt_urgency not in self.RATIONALE_KEYWORDS:
            return 0.10  # Some credit for attempting

        # --- PyTorch: 6-class softmax over urgency keyword space ---
        base_score, confidence = self._confidence.score_keywords(
            candidate_keywords=self.RATIONALE_KEYWORDS,
            text=rationale,
            correct_class=gt_urgency,
            max_score=0.22,
        )

        # Bonus for citing regulatory references
        reg_bonus = 0.0
        if "21 cfr" in rationale or "312.32" in rationale:
            reg_bonus += 0.05
        if "ich" in rationale or "e6" in rationale:
            reg_bonus += 0.02
        if "ind" in rationale:
            reg_bonus += 0.02

        # Context bonus: rationale mentions the AE's medical category
        category_bonus = 0.0
        if ae_category:
            cat_keywords = self.CATEGORY_KEYWORDS.get(ae_category, [])
            if any(kw in rationale for kw in cat_keywords):
                category_bonus = 0.02

        return min(base_score + reg_bonus + category_bonus, 0.30)
