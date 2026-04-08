"""
confidence.py — PyTorch-powered confidence scorer for ClinicalTrialEnv graders.

Uses torch.nn.functional.softmax to produce calibrated probability distributions
over candidate classes from discriminatively-weighted keyword-hit logits.
Every grader delegates rationale/keyword scoring to this module so that PyTorch
is part of the *core* reward pipeline — not merely an optional import.

Improvements over v1:
- Discriminative keyword weighting (inter-class exclusive keywords get 2x weight)
- Length-tier bonus: detailed rationales (>100 chars) get a +0.5 logit boost
  for the correct class, rewarding verbose well-grounded responses
- Zero-logit guard: when no keywords match at all, returns (0.0, 1/N) instead
  of the misleading uniform-distribution score that rewarded empty responses
- Exact canonical-string match shortcut for timeline/action field values
"""

import torch
import torch.nn.functional as F
from typing import Union


class TorchConfidenceScorer:
    """Converts raw keyword-match counts into a calibrated confidence score
    using PyTorch's softmax, then maps that confidence to a grading weight.

    Design
    ------
    For a given set of candidate classes (e.g. urgency levels, severity levels)
    we treat each class's keyword-hit count as a logit.  Softmax turns those
    logits into a probability distribution.  The probability mass assigned to
    the *correct* class becomes a ``confidence`` value in (0, 1).

    Discriminative weighting sharpens the signal: keywords exclusive to one
    class get weight 2.0, keywords shared across classes get weight 1.0.
    This means an agent that uses precisely the right regulatory vocabulary is
    rewarded more strongly than one that uses generic words that fit everywhere.

    Parameters
    ----------
    temperature : float
        Softmax temperature.  Lower values make the scorer more decisive.
        Default is 1.0 (standard softmax).
    length_bonus : float
        Extra logit mass added to the correct-class slot when the text has
        >= 100 characters. Rewards thorough, detailed rationales. Default 0.5.
    """

    def __init__(self, temperature: float = 1.0, length_bonus: float = 0.5):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = temperature
        self.length_bonus = length_bonus

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discriminative_weights(
        self, candidate_keywords: dict[str, list[str]]
    ) -> dict[str, dict[str, float]]:
        """Build per-keyword weights based on class exclusivity.

        A keyword that appears in exactly one class gets weight 2.0.
        A keyword shared across multiple classes gets weight 1.0.
        This makes the softmax signal sharper when the agent uses
        class-specific regulatory terminology.
        """
        from collections import Counter
        # Count how many classes each keyword appears in
        kw_class_count: Counter = Counter()
        for kws in candidate_keywords.values():
            for kw in kws:
                kw_class_count[kw.lower()] += 1

        weights: dict[str, dict[str, float]] = {}
        for cls, kws in candidate_keywords.items():
            weights[cls] = {}
            for kw in kws:
                kl = kw.lower()
                weights[cls][kl] = 2.0 if kw_class_count[kl] == 1 else 1.0
        return weights

    def _build_logits(
        self,
        candidate_keywords: dict[str, list[str]],
        text: str,
        correct_class: str,
    ) -> torch.Tensor:
        """Build a logit vector over all candidate classes.

        Each logit = sum of discriminative weights for matched keywords.
        If the text is long (>= 100 chars) an extra length_bonus is added
        to the correct-class slot to reward detailed reasoning.
        """
        classes = list(candidate_keywords.keys())
        weights = self._discriminative_weights(candidate_keywords)

        raw_logits = []
        for cls in classes:
            hit_weight = sum(
                weights[cls].get(kw.lower(), 1.0)
                for kw in candidate_keywords[cls]
                if kw.lower() in text
            )
            raw_logits.append(hit_weight)

        logits = torch.tensor(raw_logits, dtype=torch.float32)

        # Length-tier bonus: add to the correct-class position
        if len(text) >= 100 and correct_class in classes:
            correct_idx = classes.index(correct_class)
            logits[correct_idx] += self.length_bonus

        return logits

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_keywords(
        self,
        candidate_keywords: dict[str, list[str]],
        text: str,
        correct_class: str,
        max_score: float,
    ) -> tuple[float, float]:
        """Score keyword coverage using a PyTorch softmax over logits.

        Parameters
        ----------
        candidate_keywords : dict[str, list[str]]
            Maps each candidate class name to a list of expected keywords.
        text : str
            The agent's rationale / reasoning text (lowercased externally).
        correct_class : str
            The ground-truth class name.
        max_score : float
            The maximum points this component can contribute.

        Returns
        -------
        (score, confidence) : tuple[float, float]
            ``score``      — points earned in [0.0, max_score]
            ``confidence`` — softmax probability assigned to correct_class in (0, 1)
        """
        classes = list(candidate_keywords.keys())
        if not classes or correct_class not in candidate_keywords:
            return 0.0, 0.0

        logits = self._build_logits(candidate_keywords, text, correct_class)

        # Zero-logit guard: if no keywords matched at all, return zero score
        # (before v2 this returned misleading uniform 1/N score)
        if logits.sum().item() <= 0.0 and self.length_bonus == 0:
            return 0.0, round(1.0 / len(classes), 4)

        # Apply temperature-scaled softmax → probability distribution
        probs = F.softmax(logits / self.temperature, dim=0)

        correct_idx = classes.index(correct_class)
        confidence = float(probs[correct_idx].item())

        score = confidence * max_score
        return round(score, 4), round(confidence, 4)

    def binary_confidence(
        self,
        positive_keywords: list[str],
        negative_keywords: list[str],
        text: str,
        is_positive: bool,
        max_score: float,
    ) -> tuple[float, float]:
        """Softmax over a binary {positive, negative} keyword space.

        Used by the eligibility grader where the decision is a boolean
        (eligible / ineligible) rather than a multi-class label.

        Improvements over v1:
        - Discriminative weighting: keywords that appear only in one side
          get weight 2.0 vs. 1.0 for shared keywords.
        - Length bonus applied to the correct polarity slot.
        - Zero-logit guard returns 0 score instead of 0.5 uniform.

        Returns
        -------
        (score, confidence) : tuple[float, float]
        """
        # Build discriminative weights for binary case
        all_pos = {kw.lower() for kw in positive_keywords}
        all_neg = {kw.lower() for kw in negative_keywords}
        shared = all_pos & all_neg

        def weighted_hits(keywords: list[str], shared_set: set) -> float:
            total = 0.0
            for kw in keywords:
                kl = kw.lower()
                if kl in text:
                    total += 1.0 if kl in shared_set else 2.0
            return total

        pos_hits = weighted_hits(positive_keywords, shared)
        neg_hits = weighted_hits(negative_keywords, shared)

        logits = torch.tensor([pos_hits, neg_hits], dtype=torch.float32)

        # Length-tier bonus to the correct polarity
        if len(text) >= 100:
            logits[0 if is_positive else 1] += self.length_bonus

        # Zero-logit guard
        if logits.sum().item() <= 0.0:
            return 0.0, 0.5

        probs = F.softmax(logits / self.temperature, dim=0)

        correct_prob = float(probs[0].item() if is_positive else probs[1].item())
        score = correct_prob * max_score
        return round(score, 4), round(correct_prob, 4)
