"""
Protocol Deviation Assessment Grader (Hard Task)
Scores agent decisions on deviation classification and corrective action
per ICH E6 GCP guidelines. Returns scores in [0.0, 1.0] with partial credit.

Rationale scoring uses a PyTorch softmax over 3 severity-level keyword sets
(see server.graders.confidence.TorchConfidenceScorer).

Improvements over v1:
- CRITICAL BUG FIX: _normalize_action() now does keyword matching BEFORE
  replacing spaces with underscores, fixing the "follow up" / "follow_up"
  mismatch that caused 0 action scores for all keyword-matched actions
- Expanded ACTION_KEYWORDS with natural-language synonyms
- Expanded RATIONALE_KEYWORDS with deviation-category-specific vocabulary
  (consent, blinding, IP, storage, documentation, delegation, dosing, lab)
- Raised max_score from 0.18 to 0.22 to close the score ceiling gap
- Added ICH section-aware bonus: cites specific section numbers from
  the deviation protocol_section field get extra reward (+0.03)
- _normalize_classification() handles more LLM variants
"""

from .confidence import TorchConfidenceScorer


class DeviationGrader:
    """Grades protocol deviation assessment decisions.

    Scoring breakdown (total = 1.0):
    - Correct severity classification:  0.35
    - Appropriate corrective action:    0.35
    - Rationale quality (PyTorch):      0.30
    """

    SEVERITY_LEVELS = {
        "minor": 0,
        "major": 1,
        "critical": 2,
    }

    VALID_ACTIONS = {
        "minor": [
            "document_log_only",
            "document_recontact_subject",
            "document_recollect_sample",
        ],
        "major": [
            "report_irb_sponsor_immediately",
            "report_sponsor_assess_continuation",
            "repeat_assessment_report_sponsor",
            "report_irb_sponsor_retrain",
            "report_sponsor_audit_required",
            "report_sponsor_assess_safety",
        ],
        "critical": [
            "report_irb_sponsor_immediately",
            "quarantine_ip_notify_sponsor_irb_immediately",
        ],
    }

    # Expanded with natural-language synonyms for robust matching.
    # IMPORTANT: keyword matching is done BEFORE underscore replacement —
    # so keep keywords in their natural space-separated form.
    ACTION_KEYWORDS = {
        "document_log_only": [
            "document", "log", "record", "deviation log", "note",
            "log only", "document only",
        ],
        "document_recontact_subject": [
            "recontact", "follow up", "follow-up", "reschedule", "contact subject",
            "reach out", "contact participant",
        ],
        "document_recollect_sample": [
            "recollect", "resample", "repeat sample", "collect again",
            "new sample", "re-collect",
        ],
        "report_irb_sponsor_immediately": [
            "report", "irb", "sponsor", "immediate", "notify irb",
            "notify sponsor", "irb notification", "report immediately",
            "urgent report",
        ],
        "report_sponsor_assess_continuation": [
            "sponsor", "assess", "continue", "continuation", "assess continuation",
            "evaluate continuation", "subject continuation",
        ],
        "repeat_assessment_report_sponsor": [
            "repeat", "assessment", "sponsor", "redo", "repeat assessment",
            "re-assess", "perform again",
        ],
        "report_irb_sponsor_retrain": [
            "retrain", "training", "delegation", "re-train", "staff training",
            "education", "competency",
        ],
        "report_sponsor_audit_required": [
            "audit", "data integrity", "sponsor", "for-cause", "cause audit",
            "audit required", "inspect",
        ],
        "report_sponsor_assess_safety": [
            "safety", "dose", "sponsor", "assess", "subject safety",
            "assess safety", "safety assessment",
        ],
        "quarantine_ip_notify_sponsor_irb_immediately": [
            "quarantine", "drug", "ip", "sponsor", "irb",
            "investigational product", "quarantine drug", "quarantine ip",
            "affected batch", "storage",
        ],
    }

    # Expanded keyword sets with deviation-category-specific vocabulary.
    # Words here directly mirror the categories and descriptions in deviations.json.
    RATIONALE_KEYWORDS = {
        "minor": [
            "minor", "no impact", "document", "within variance", "acceptable",
            "timing", "window", "schedule", "missed window", "lab error",
            "hemolyzed", "sample", "visit", "recontact", "minor deviation",
            "administrative", "no safety", "within acceptable", "tolerance",
        ],
        "major": [
            "major", "protocol violation", "data integrity", "safety",
            "ich e6", "gcp", "report", "corrective action",
            "concomitant medication", "prohibited medication", "pk interaction",
            "drug interaction", "non-validated", "equipment", "delegation",
            "qualified", "unqualified", "source document", "retrospective",
            "contemporaneous", "dose modification", "unauthorized dose",
            "blinding", "unblinded", "sponsor notification", "significant",
        ],
        "critical": [
            "critical", "immediate", "subject safety", "trial integrity",
            "ich e6", "urgent", "quarantine", "irb", "root cause",
            "informed consent", "consent", "before procedure", "rights",
            "ip storage", "storage temperature", "drug stability",
            "blinding integrity", "unblinding", "sae reporting", "late reporting",
            "timely", "immediate notification", "within 24 hours",
        ],
    }

    # ICH E6 section numbers present in the deviation dataset
    # Awarding a bonus for specific section citations (e.g., "4.8.1", "5.14")
    ICH_SECTIONS = [
        "4.8", "4.8.1", "5.14", "5.2", "4.2", "4.2.5",
        "4.9", "4.9.1", "4.11", "5.4", "4.11.1",
    ]

    def __init__(self):
        # PyTorch scorer: temperature=0.7 for a sharper 3-class severity distribution
        self._confidence = TorchConfidenceScorer(temperature=0.7, length_bonus=0.5)

    def grade(self, action: dict, ground_truth: dict) -> tuple[float, str]:
        """Grade a protocol deviation assessment.

        Args:
            action: Agent's assessment dict with keys:
                classification, corrective_action, rationale
            ground_truth: Deviation record with ground_truth_classification,
                ground_truth_action, ground_truth_rationale

        Returns:
            (score, feedback) tuple
        """
        score = 0.0
        feedback_parts = []

        gt_class = ground_truth.get("ground_truth_classification", "")
        gt_action = ground_truth.get("ground_truth_action", "")
        gt_rationale = ground_truth.get("ground_truth_rationale", "")
        dev_category = ground_truth.get("category", "")
        protocol_section = ground_truth.get("protocol_section", "")

        # --- 1. Severity classification (0.35) ---
        agent_class = self._normalize_classification(
            str(action.get("classification", "")).strip().lower()
        )
        class_score = self._score_classification(agent_class, gt_class)
        score += class_score
        if class_score >= 0.30:
            feedback_parts.append(f"Classification: CORRECT (+{class_score:.2f})")
        else:
            feedback_parts.append(
                f"Classification: {'PARTIAL' if class_score > 0 else 'INCORRECT'} "
                f"(+{class_score:.2f}). Expected '{gt_class}'"
            )

        # --- 2. Corrective action (0.35) ---
        raw_action = str(action.get("corrective_action", "")).strip().lower()
        agent_action = self._normalize_action(raw_action)
        action_score = self._score_action(agent_action, gt_action, gt_class)
        score += action_score
        feedback_parts.append(f"Action: +{action_score:.2f}/0.35")

        # --- 3. Rationale quality (0.30) ---
        rationale = str(action.get("rationale", "")).strip().lower()
        rationale_score = self._score_rationale(
            rationale, gt_class, gt_rationale, dev_category, protocol_section
        )
        score += rationale_score
        feedback_parts.append(f"Rationale: +{rationale_score:.2f}/0.30")

        score = max(0.0, min(1.0, score))
        return score, " | ".join(feedback_parts)

    def _normalize_classification(self, classification: str) -> str:
        """Normalize classification string with broader variant coverage."""
        classification = classification.strip().lower()
        if "critical" in classification or "severe" in classification or "extreme" in classification:
            return "critical"
        if "major" in classification or "significant" in classification or "serious" in classification:
            return "major"
        if "minor" in classification or "trivial" in classification or "minimal" in classification or "low" in classification:
            return "minor"
        return classification

    def _normalize_action(self, action: str) -> str:
        """Try to match free-text corrective action to canonical action.

        FIX (v2): Keyword matching is now performed on the ORIGINAL space-based
        string BEFORE replacing spaces with underscores. This fixes the v1 bug
        where "follow up" in ACTION_KEYWORDS never matched "follow_up" after
        the replacement, causing 0 action scores for keyword-matched cases.
        """
        # Step 1: Try direct canonical match (underscore-normalized)
        canonical_action = action.replace(" ", "_").replace("-", "_").lower()
        all_actions = []
        for actions in self.VALID_ACTIONS.values():
            all_actions.extend(actions)
        if canonical_action in all_actions:
            return canonical_action

        # Step 2: Keyword matching on the ORIGINAL string (before underscore replacement)
        # This is the critical fix — keywords like "follow up" now match correctly
        best_match = ""
        best_score = 0
        for canonical, keywords in self.ACTION_KEYWORDS.items():
            # Match against original action string (preserves spaces)
            hits = sum(1 for kw in keywords if kw in action)
            if hits > best_score:
                best_score = hits
                best_match = canonical

        if best_score > 0:
            return best_match

        # Step 3: Try underscore-normalized keyword matching as fallback
        for canonical, keywords in self.ACTION_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw.replace(" ", "_") in canonical_action)
            if hits > best_score:
                best_score = hits
                best_match = canonical

        return best_match if best_score > 0 else canonical_action

    def _score_classification(self, agent: str, gt: str) -> float:
        """Score classification with partial credit."""
        if agent == gt:
            return 0.35

        agent_level = self.SEVERITY_LEVELS.get(agent, -1)
        gt_level = self.SEVERITY_LEVELS.get(gt, -1)

        if agent_level < 0 or gt_level < 0:
            return 0.0

        diff = abs(agent_level - gt_level)
        if diff == 1:
            # Under-classifying critical deviations is worse
            if agent_level < gt_level and gt == "critical":
                return 0.05
            return 0.15
        return 0.0

    def _score_action(self, agent_action: str, gt_action: str, gt_class: str) -> float:
        """Score corrective action with partial credit."""
        if agent_action == gt_action:
            return 0.35

        # Check if action is at least valid for the severity level
        valid_for_class = self.VALID_ACTIONS.get(gt_class, [])
        if agent_action in valid_for_class:
            return 0.20  # Right category, wrong specific action

        # Check if action keywords overlap with ground truth
        gt_keywords = set(self.ACTION_KEYWORDS.get(gt_action, []))
        agent_keywords = set(self.ACTION_KEYWORDS.get(agent_action, []))
        if gt_keywords and agent_keywords:
            overlap = len(gt_keywords & agent_keywords)
            if overlap >= 2:
                return 0.15
            if overlap == 1:
                return 0.10

        # Any valid action from any category gets minimal credit
        all_actions = []
        for actions in self.VALID_ACTIONS.values():
            all_actions.extend(actions)
        if agent_action in all_actions:
            return 0.05

        return 0.0

    def _score_rationale(
        self,
        rationale: str,
        gt_class: str,
        gt_rationale: str,
        dev_category: str,
        protocol_section: str,
    ) -> float:
        """Score rationale quality using a PyTorch softmax over 3 severity levels.

        score_keywords() computes a 3-class (minor / major / critical) softmax
        over keyword-hit counts, so the probability mass flows entirely to the
        correct severity when the agent uses the right regulatory language.
        Temperature=0.7 makes this a sharp test: ambiguous rationales score low.
        """
        if not rationale or len(rationale) < 10:
            return 0.0

        if gt_class not in self.RATIONALE_KEYWORDS:
            return 0.05

        # --- PyTorch: 3-class softmax over severity keyword space ---
        base_score, confidence = self._confidence.score_keywords(
            candidate_keywords=self.RATIONALE_KEYWORDS,
            text=rationale,
            correct_class=gt_class,
            max_score=0.22,
        )

        # Bonus for regulatory references
        reg_bonus = 0.0
        if "ich e6" in rationale or "ich-e6" in rationale:
            reg_bonus += 0.04
        if "gcp" in rationale:
            reg_bonus += 0.02
        if "good clinical practice" in rationale:
            reg_bonus += 0.01

        # ICH section-aware bonus: mention of specific section numbers
        section_bonus = 0.0
        for sec in self.ICH_SECTIONS:
            if sec in rationale:
                section_bonus = 0.03
                break

        # Category-keyword bonus: rationale mentions the deviation category
        category_bonus = 0.0
        if dev_category and dev_category.replace("_", " ") in rationale:
            category_bonus = 0.01

        return min(base_score + reg_bonus + section_bonus + category_bonus, 0.30)
