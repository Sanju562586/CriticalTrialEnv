"""
Eligibility Screening Grader (Easy Task)
Scores agent decisions on patient eligibility against trial criteria.
Returns scores in [0.0, 1.0] with partial credit.
"""


class EligibilityGrader:
    """Grades eligibility screening decisions.

    Scoring breakdown (total = 1.0):
    - Correct decision (eligible/ineligible): 0.50
    - Correct reasoning quality:              0.30
    - Correct criteria citation:              0.20
    """

    # Keywords that indicate good reasoning for eligible patients
    INCLUSION_KEYWORDS = [
        "meets", "satisfies", "within range", "qualifies",
        "inclusion", "criterion", "criteria met"
    ]

    # Keywords that indicate good reasoning for ineligible patients
    EXCLUSION_KEYWORDS = [
        "excluded", "fails", "does not meet", "outside range",
        "ineligible", "exclusion", "violated", "exceeds", "below"
    ]

    def grade(self, action: dict, ground_truth: dict) -> tuple[float, str]:
        """Grade an eligibility screening decision.

        Args:
            action: Agent's decision dict with keys: decision, reasoning, criteria_cited
            ground_truth: Patient record with 'eligible' and 'notes' fields

        Returns:
            (score, feedback) tuple where score is in [0.0, 1.0]
        """
        score = 0.0
        feedback_parts = []

        gt_eligible = ground_truth.get("eligible", False)
        gt_notes = ground_truth.get("notes", "")

        # --- 1. Decision correctness (0.50) ---
        agent_decision = str(action.get("decision", "")).strip().lower()
        correct_decision = self._normalize_decision(agent_decision) == gt_eligible

        if correct_decision:
            score += 0.50
            feedback_parts.append("Decision: CORRECT (+0.50)")
        else:
            feedback_parts.append(
                f"Decision: INCORRECT (+0.00). "
                f"Expected {'eligible' if gt_eligible else 'ineligible'}, "
                f"got '{agent_decision}'"
            )

        # --- 2. Reasoning quality (0.30) ---
        reasoning = str(action.get("reasoning", "")).lower()
        reasoning_score = self._score_reasoning(reasoning, gt_eligible, gt_notes)
        score += reasoning_score
        feedback_parts.append(f"Reasoning: +{reasoning_score:.2f}/0.30")

        # --- 3. Criteria citation (0.20) ---
        cited = action.get("criteria_cited", [])
        citation_score = self._score_citations(cited, gt_notes, gt_eligible)
        score += citation_score
        feedback_parts.append(f"Citations: +{citation_score:.2f}/0.20")

        # Clamp to [0.0, 1.0]
        score = max(0.0, min(1.0, score))
        feedback = " | ".join(feedback_parts)

        return score, feedback

    def _normalize_decision(self, decision: str) -> bool:
        """Convert agent decision string to boolean."""
        positive = {"eligible", "yes", "true", "include", "included", "pass", "accept"}
        return decision in positive

    def _score_reasoning(self, reasoning: str, gt_eligible: bool, gt_notes: str) -> float:
        """Score reasoning quality based on keyword matching and length."""
        if not reasoning or len(reasoning) < 10:
            return 0.0

        keywords = self.INCLUSION_KEYWORDS if gt_eligible else self.EXCLUSION_KEYWORDS
        keyword_hits = sum(1 for kw in keywords if kw in reasoning)
        keyword_score = min(keyword_hits / 3, 1.0) * 0.15

        # Check if reasoning references specific criteria from ground truth
        gt_lower = gt_notes.lower()
        specificity_score = 0.0
        # Look for criterion IDs like INC-01, EXC-02
        for token in gt_lower.split():
            if token.startswith(("inc-", "exc-")) and token in reasoning:
                specificity_score = 0.15
                break

        # Partial credit for mentioning the right category
        if specificity_score == 0.0:
            field_keywords = ["age", "nyha", "lvef", "egfr", "potassium",
                              "pregnancy", "allergy", "hemoglobin", "blood pressure",
                              "cardiac event", "malignancy", "gdmt", "nt-probnp"]
            for fk in field_keywords:
                if fk in reasoning and fk in gt_lower:
                    specificity_score = 0.10
                    break

        return keyword_score + specificity_score

    def _score_citations(self, cited: list, gt_notes: str, gt_eligible: bool) -> float:
        """Score criteria citation accuracy."""
        if not cited:
            return 0.0

        if isinstance(cited, str):
            cited = [cited]

        gt_lower = gt_notes.lower()
        correct_citations = 0
        total_cited = len(cited)

        for c in cited:
            c_lower = str(c).lower().strip()
            if c_lower in gt_lower:
                correct_citations += 1
            elif gt_eligible and c_lower.startswith("inc-"):
                correct_citations += 0.5
            elif not gt_eligible and c_lower.startswith("exc-"):
                correct_citations += 0.5

        if total_cited > 0:
            precision = correct_citations / total_cited
        else:
            precision = 0.0

        # Penalize citing too many criteria (overconfident)
        if total_cited > 5:
            precision *= 0.8

        return min(precision * 0.20, 0.20)
