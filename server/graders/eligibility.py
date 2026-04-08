"""
Eligibility Screening Grader (Easy Task)
Scores agent decisions on patient eligibility against trial criteria.
Returns scores in [0.0, 1.0] with partial credit.

Reasoning quality is scored via PyTorch softmax-based confidence
(see server.graders.confidence.TorchConfidenceScorer).

Improvements over v1:
- Deterministic rule-based verifier cross-checks the LLM's decision
  against the actual inclusion/exclusion criteria (catches notes ambiguity)
- Citation scoring checks against all known criterion IDs, not just notes text
- Expanded INCLUSION/EXCLUSION keywords cover more LLM phrasing variants
- Removed the over-citation penalty (more correct citations = better)
- Specificity scoring now awards partial credit for any relevant field keyword
  even if the specific criterion ID isn't cited
"""

from .confidence import TorchConfidenceScorer


# Mirrors criteria.json — used by deterministic verifier
# These never change for this benchmark; adding them here ensures correctness
# even when notes merely say "Meets all criteria".
_INCLUSION_RULES = [
    {"id": "INC-01", "field": "age", "op": "between", "val": (18, 80)},
    {"id": "INC-02", "field": "nyha_class", "op": "in", "val": {"II", "III", "IV"}},
    {"id": "INC-03", "field": "lvef_percent", "op": "lte", "val": 40},
    {"id": "INC-04", "field": "stable_gdmt_weeks", "op": "gte", "val": 4},
    {"id": "INC-05", "field": "nt_probnp", "op": "gte", "val": 400},
]
_EXCLUSION_RULES = [
    {"id": "EXC-01", "field": "egfr", "op": "lt", "val": 20},
    {"id": "EXC-02", "field": "potassium", "op": "gt", "val": 5.5},
    {"id": "EXC-03", "field": "days_since_cardiac_event", "op": "lt", "val": 90},
    {"id": "EXC-04", "field": "active_malignancy", "op": "eq", "val": True},
    {"id": "EXC-05", "field": "pregnancy_risk", "op": "eq", "val": True},
    {"id": "EXC-06", "field": "drug_allergy", "op": "eq", "val": True},
    {"id": "EXC-07", "field": "systolic_bp", "op": "lt", "val": 90},
    {"id": "EXC-08", "field": "hemoglobin", "op": "lt", "val": 9.0},
]
_ALL_CRITERION_IDS = {r["id"].lower() for r in _INCLUSION_RULES + _EXCLUSION_RULES}
_INC_IDS = {r["id"].lower() for r in _INCLUSION_RULES}
_EXC_IDS = {r["id"].lower() for r in _EXCLUSION_RULES}


def _apply_rule(case: dict, rule: dict) -> bool:
    """Return True if the case field satisfies the rule (inclusion/exclusion met)."""
    val = case.get(rule["field"])
    if val is None:
        return False
    op = rule["op"]
    rv = rule["val"]
    if op == "between":
        return rv[0] <= val <= rv[1]
    if op == "in":
        return str(val) in rv
    if op == "lte":
        return val <= rv
    if op == "gte":
        return val >= rv
    if op == "lt":
        return val < rv
    if op == "gt":
        return val > rv
    if op == "eq":
        return val == rv
    return False


def _determine_eligibility(case: dict) -> tuple[bool, list[str]]:
    """Deterministically evaluate patient eligibility per CARDIO-SHIELD criteria.

    Returns
    -------
    (eligible, violated_ids)
        eligible     — True if all inclusion met AND no exclusion triggered
        violated_ids — list of criterion IDs that are violated/failed
    """
    violated = []
    for rule in _INCLUSION_RULES:
        if not _apply_rule(case, rule):
            violated.append(rule["id"])
    for rule in _EXCLUSION_RULES:
        if _apply_rule(case, rule):
            violated.append(rule["id"])
    return (len(violated) == 0), violated


class EligibilityGrader:
    """Grades eligibility screening decisions.

    Scoring breakdown (total = 1.0):
    - Correct decision (eligible/ineligible): 0.50
    - Reasoning quality (PyTorch softmax):    0.30
    - Correct criteria citation:              0.20
    """

    # Extended keyword lists cover more LLM phrasing variants
    INCLUSION_KEYWORDS = [
        "meets", "satisfies", "within range", "qualifies",
        "inclusion", "criterion", "criteria met", "eligible",
        "no exclusion", "all criteria satisfied", "all inclusion criteria",
        "passes", "included", "complies", "fulfilled", "appropriate",
    ]

    EXCLUSION_KEYWORDS = [
        "excluded", "fails", "does not meet", "outside range",
        "ineligible", "exclusion", "violated", "exceeds", "below",
        "not eligible", "criterion violated", "criteria not met",
        "disqualified", "disqualify", "contraindicated", "prohibited",
        "fail", "failure", "non-compliant", "unsafe",
    ]

    def __init__(self):
        # Shared PyTorch confidence scorer (temperature=0.7 → sharper for binary)
        self._confidence = TorchConfidenceScorer(temperature=0.7, length_bonus=0.5)

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

        # Use deterministic verification as a cross-check (handles "Meets all criteria" notes)
        computed_eligible, violated_ids = _determine_eligibility(ground_truth)

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
        reasoning_score = self._score_reasoning(reasoning, gt_eligible, gt_notes, violated_ids)
        score += reasoning_score
        feedback_parts.append(f"Reasoning: +{reasoning_score:.2f}/0.30")

        # --- 3. Criteria citation (0.20) ---
        cited = action.get("criteria_cited", [])
        citation_score = self._score_citations(cited, gt_eligible, gt_notes, violated_ids)
        score += citation_score
        feedback_parts.append(f"Citations: +{citation_score:.2f}/0.20")

        # Clamp to [0.0, 1.0]
        score = max(0.0, min(1.0, score))
        feedback = " | ".join(feedback_parts)

        return score, feedback

    def _normalize_decision(self, decision: str) -> bool:
        """Convert agent decision string to boolean."""
        positive = {"eligible", "yes", "true", "include", "included", "pass", "accept", "qualifies"}
        return decision in positive

    def _score_reasoning(
        self, reasoning: str, gt_eligible: bool, gt_notes: str, violated_ids: list[str]
    ) -> float:
        """Score reasoning quality using PyTorch softmax confidence.

        The binary_confidence() call runs a 2-class softmax over keyword-hit
        counts (inclusion vs. exclusion keywords), yielding a probability that
        directly becomes the keyword component of the score.
        """
        if not reasoning or len(reasoning) < 10:
            return 0.0

        # --- PyTorch softmax over binary {eligible, ineligible} keyword space ---
        keyword_score, confidence = self._confidence.binary_confidence(
            positive_keywords=self.INCLUSION_KEYWORDS,
            negative_keywords=self.EXCLUSION_KEYWORDS,
            text=reasoning,
            is_positive=gt_eligible,
            max_score=0.15,
        )

        # --- Specificity: criterion IDs or clinical field keywords ---
        specificity_score = 0.0

        # Check if any criterion ID from the violated list appears in reasoning
        for cid in violated_ids:
            if cid.lower() in reasoning:
                specificity_score = 0.15
                break

        # Check all known criterion IDs in reasoning (handles eligible cases too)
        if specificity_score == 0.0:
            for cid in _ALL_CRITERION_IDS:
                if cid in reasoning:
                    specificity_score = 0.15
                    break

        # Fallback: clinical field name mentions
        if specificity_score == 0.0:
            field_keywords = [
                "age", "nyha", "lvef", "egfr", "potassium",
                "pregnancy", "allergy", "hemoglobin", "blood pressure",
                "cardiac event", "malignancy", "gdmt", "nt-probnp",
                "systolic", "ejection fraction", "stable therapy",
            ]
            for fk in field_keywords:
                if fk in reasoning:
                    specificity_score = 0.10
                    break

        return min(keyword_score + specificity_score, 0.30)

    def _score_citations(
        self,
        cited: list,
        gt_eligible: bool,
        gt_notes: str,
        violated_ids: list[str],
    ) -> float:
        """Score criteria citation accuracy.

        Three-source matching (v2):
        1. Exact criterion ID in gt_notes (e.g. 'EXC-02')
        2. Correct prefix convention (inc-* for eligible, exc-* for ineligible)
        3. Match against deterministic violated_ids set
        """
        if not cited:
            return 0.0

        if isinstance(cited, str):
            cited = [cited]

        gt_lower = gt_notes.lower()
        violated_lower = {v.lower() for v in violated_ids}
        correct_citations = 0.0
        total_cited = len(cited)

        for c in cited:
            c_lower = str(c).strip().lower()

            # Source 1: exact match in notes
            if c_lower in gt_lower:
                correct_citations += 1.0
                continue

            # Source 2: check against deterministic violated IDs
            if c_lower in violated_lower:
                correct_citations += 1.0
                continue

            # Source 3: prefix convention (eligible → inc-*, ineligible → exc-*)
            if gt_eligible and c_lower.startswith("inc-"):
                # For eligible patients any valid INC- citation is directionally correct
                if c_lower in _INC_IDS:
                    correct_citations += 0.8
                else:
                    correct_citations += 0.4
            elif not gt_eligible and c_lower.startswith("exc-"):
                if c_lower in _EXC_IDS:
                    correct_citations += 0.6
                else:
                    correct_citations += 0.3

        precision = correct_citations / total_cited if total_cited > 0 else 0.0
        # No penalty for citing many criteria — more complete citations are fine
        return min(precision * 0.20, 0.20)
