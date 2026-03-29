"""
Protocol Deviation Assessment Grader (Hard Task)
Scores agent decisions on deviation classification and corrective action
per ICH E6 GCP guidelines. Returns scores in [0.0, 1.0] with partial credit.
"""


class DeviationGrader:
    """Grades protocol deviation assessment decisions.

    Scoring breakdown (total = 1.0):
    - Correct severity classification:  0.35
    - Appropriate corrective action:    0.35
    - Rationale quality:                0.30
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

    ACTION_KEYWORDS = {
        "document_log_only": ["document", "log", "record"],
        "document_recontact_subject": ["recontact", "follow up", "reschedule"],
        "document_recollect_sample": ["recollect", "resample", "repeat"],
        "report_irb_sponsor_immediately": ["report", "irb", "sponsor", "immediate"],
        "report_sponsor_assess_continuation": ["sponsor", "assess", "continue", "continuation"],
        "repeat_assessment_report_sponsor": ["repeat", "assessment", "sponsor"],
        "report_irb_sponsor_retrain": ["retrain", "training", "delegation"],
        "report_sponsor_audit_required": ["audit", "data integrity", "sponsor"],
        "report_sponsor_assess_safety": ["safety", "dose", "sponsor", "assess"],
        "quarantine_ip_notify_sponsor_irb_immediately": [
            "quarantine", "drug", "ip", "sponsor", "irb"
        ],
    }

    RATIONALE_KEYWORDS = {
        "minor": ["minor", "no impact", "document", "within variance", "acceptable"],
        "major": ["major", "protocol violation", "data integrity", "safety",
                  "ich e6", "gcp", "report", "corrective action"],
        "critical": ["critical", "immediate", "subject safety", "trial integrity",
                     "ich e6", "urgent", "quarantine", "irb", "root cause"],
    }

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
        agent_action = self._normalize_action(
            str(action.get("corrective_action", "")).strip().lower()
        )
        action_score = self._score_action(agent_action, gt_action, gt_class)
        score += action_score
        feedback_parts.append(f"Action: +{action_score:.2f}/0.35")

        # --- 3. Rationale quality (0.30) ---
        rationale = str(action.get("rationale", "")).strip().lower()
        rationale_score = self._score_rationale(rationale, gt_class, gt_rationale)
        score += rationale_score
        feedback_parts.append(f"Rationale: +{rationale_score:.2f}/0.30")

        score = max(0.0, min(1.0, score))
        return score, " | ".join(feedback_parts)

    def _normalize_classification(self, classification: str) -> str:
        """Normalize classification string."""
        classification = classification.strip().lower()
        if "critical" in classification or "severe" in classification:
            return "critical"
        if "major" in classification or "significant" in classification:
            return "major"
        if "minor" in classification or "trivial" in classification:
            return "minor"
        return classification

    def _normalize_action(self, action: str) -> str:
        """Try to match free-text corrective action to canonical action."""
        action = action.replace(" ", "_").replace("-", "_").lower()

        # Direct match
        all_actions = []
        for actions in self.VALID_ACTIONS.values():
            all_actions.extend(actions)
        if action in all_actions:
            return action

        # Keyword matching
        best_match = ""
        best_score = 0
        for canonical, keywords in self.ACTION_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in action)
            if hits > best_score:
                best_score = hits
                best_match = canonical
        return best_match if best_score > 0 else action

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
        gt_keywords = self.ACTION_KEYWORDS.get(gt_action, [])
        agent_keywords = self.ACTION_KEYWORDS.get(agent_action, [])
        if gt_keywords and agent_keywords:
            overlap = len(set(gt_keywords) & set(agent_keywords))
            if overlap > 0:
                return 0.10

        # Any valid action gets minimal credit
        all_actions = []
        for actions in self.VALID_ACTIONS.values():
            all_actions.extend(actions)
        if agent_action in all_actions:
            return 0.05

        return 0.0

    def _score_rationale(self, rationale: str, gt_class: str, gt_rationale: str) -> float:
        """Score rationale quality."""
        if not rationale or len(rationale) < 10:
            return 0.0

        keywords = self.RATIONALE_KEYWORDS.get(gt_class, [])
        if not keywords:
            return 0.05

        hits = sum(1 for kw in keywords if kw in rationale)
        coverage = hits / len(keywords)
        base_score = coverage * 0.18

        # Bonus for regulatory references
        reg_bonus = 0.0
        if "ich e6" in rationale or "ich-e6" in rationale:
            reg_bonus += 0.05
        if "gcp" in rationale:
            reg_bonus += 0.03
        if any(s in rationale for s in ["4.8", "5.14", "4.2", "4.9", "4.11", "5.2"]):
            reg_bonus += 0.04

        return min(base_score + reg_bonus, 0.30)
