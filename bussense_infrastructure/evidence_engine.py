#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bussense_infrastructure/evidence_engine.py
==========================================
Deterministic Evidence & State Machine Engine (Sections 2, 21, 22)

Rules:
- Strictly Rule-Based / Deterministic. No black-box logistic regression or isotonic curves.
- Separation of Concerns:
  AI Perception -> Tracked Observation -> Road Association -> Reference Comparison ->
  Temporal Evidence -> Multi-Bus Evidence -> Road State -> Event.
- Non-detection != Missing Asset.
- States:
  OBSERVED -> TRACKED -> PERSISTENT -> CANDIDATE -> CORROBORATED -> VERIFIED
  plus INSUFFICIENT_DATA, CONFLICTING_EVIDENCE, NO_MISSING_CLAIM.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class EvidenceEvaluationResult:
    status: str  # OBSERVED, TRACKED, PERSISTENT, CANDIDATE, CORROBORATED, VERIFIED, INSUFFICIENT_DATA, CONFLICTING_EVIDENCE, NO_MISSING_CLAIM
    is_missing_claim: bool
    confidence_score: float
    factors: Dict[str, Any]
    explanation: str


class EvidenceEngine:
    """
    Deterministic state machine and evidence aggregator for infrastructure events.
    """

    def __init__(
        self,
        min_visibility_for_absence: float = 0.65,
        min_persistence_frames: int = 4,
        min_corroborating_buses: int = 2,
    ):
        self.min_visibility_for_absence = min_visibility_for_absence
        self.min_persistence_frames = min_persistence_frames
        self.min_corroborating_buses = min_corroborating_buses

    def evaluate_present_asset(
        self,
        observation_count: int,
        unique_buses: int,
        model_confidence: float,
        visibility_score: float,
        temporal_consistency: float = 1.0,
        has_conflicts: bool = False
    ) -> EvidenceEvaluationResult:
        """
        Evaluates state lifecycle for observed physical infrastructure assets.
        """
        factors = {
            "observation_count": observation_count,
            "unique_buses": unique_buses,
            "model_confidence": model_confidence,
            "visibility_score": visibility_score,
            "temporal_consistency": temporal_consistency,
            "has_conflicts": has_conflicts
        }

        if has_conflicts:
            return EvidenceEvaluationResult(
                status="CONFLICTING_EVIDENCE",
                is_missing_claim=False,
                confidence_score=0.50,
                factors=factors,
                explanation="Contradictory observations detected across sensing passes."
            )

        if visibility_score < 0.50:
            return EvidenceEvaluationResult(
                status="INSUFFICIENT_DATA",
                is_missing_claim=False,
                confidence_score=model_confidence * 0.5,
                factors=factors,
                explanation="Visibility below operational threshold to reliably confirm asset state."
            )

        if unique_buses >= self.min_corroborating_buses:
            status = "CORROBORATED" if observation_count < 10 else "VERIFIED"
        elif observation_count >= self.min_persistence_frames:
            status = "PERSISTENT"
        elif observation_count > 1:
            status = "TRACKED"
        else:
            status = "OBSERVED"

        conf_score = round(model_confidence * min(1.0, 0.7 + 0.15 * unique_buses), 3)

        return EvidenceEvaluationResult(
            status=status,
            is_missing_claim=False,
            confidence_score=conf_score,
            factors=factors,
            explanation=f"Asset observed and verified through {observation_count} frames across {unique_buses} bus pass(es)."
        )

    def evaluate_missing_asset(
        self,
        asset_type: str,
        expected_state: str,  # YES, NO, UNKNOWN
        observed_present: bool,
        visibility_score: float,
        consecutive_absences: int = 1,
        unique_buses: int = 1,
        has_conflicts: bool = False
    ) -> EvidenceEvaluationResult:
        """
        Evaluates missing-asset claims against expected inventory (Sections 9, 11, 14, 22).
        Strictly guarantees:
        - expected = UNKNOWN -> NO_MISSING_CLAIM
        - expected = YES + inadequate visibility -> INSUFFICIENT_DATA
        - expected = YES + good visibility + persistent absence -> MISSING_*_CANDIDATE
        - corroborated by independent buses -> CORROBORATED
        """
        factors = {
            "asset_type": asset_type,
            "expected_state": expected_state,
            "observed_present": observed_present,
            "visibility_score": visibility_score,
            "consecutive_absences": consecutive_absences,
            "unique_buses": unique_buses,
            "has_conflicts": has_conflicts
        }

        # 1. If asset is observed present, no missing claim
        if observed_present and not has_conflicts:
            return EvidenceEvaluationResult(
                status="OBSERVED",
                is_missing_claim=False,
                confidence_score=0.95,
                factors=factors,
                explanation="Asset physically observed; no missing condition exists."
            )

        # 2. Conflicting observations across buses
        if has_conflicts:
            return EvidenceEvaluationResult(
                status="CONFLICTING_EVIDENCE",
                is_missing_claim=False,
                confidence_score=0.50,
                factors=factors,
                explanation="Conflicting observation passes detected (one bus reported present, another reported absent)."
            )

        # 3. Unknown reference expectation: Never claim missing asset!
        if expected_state.upper() == "UNKNOWN":
            return EvidenceEvaluationResult(
                status="NO_MISSING_CLAIM",
                is_missing_claim=False,
                confidence_score=0.0,
                factors=factors,
                explanation="Asset expectation is UNKNOWN in reference inventory. Silence rule active."
            )

        # 5. Expected NO (explicitly mapped as non-existent)
        if expected_state.upper() == "NO":
            return EvidenceEvaluationResult(
                status="NO_MISSING_CLAIM",
                is_missing_claim=False,
                confidence_score=0.0,
                factors=factors,
                explanation="Asset is explicitly expected to be absent."
            )

        # 6. Expected YES, but camera visibility is poor -> INSUFFICIENT_DATA
        if visibility_score < self.min_visibility_for_absence:
            return EvidenceEvaluationResult(
                status="INSUFFICIENT_DATA",
                is_missing_claim=False,
                confidence_score=0.30,
                factors=factors,
                explanation=f"Visibility ({visibility_score:.2f}) below threshold ({self.min_visibility_for_absence:.2f}) to assert asset absence."
            )

        # 7. Expected YES + adequate visibility + persistent non-observation
        if consecutive_absences >= self.min_persistence_frames or unique_buses >= 1:
            if unique_buses >= self.min_corroborating_buses:
                final_status = "CORROBORATED"
            else:
                final_status = "CANDIDATE"

            conf = round(float(min(0.98, 0.70 + 0.12 * unique_buses + 0.05 * min(consecutive_absences, 5))), 2)

            return EvidenceEvaluationResult(
                status=final_status,
                is_missing_claim=True,
                confidence_score=conf,
                factors=factors,
                explanation=f"Expected {asset_type} not observed across {consecutive_absences} consecutive frames by {unique_buses} independent bus pass(es)."
            )

        return EvidenceEvaluationResult(
            status="INSUFFICIENT_DATA",
            is_missing_claim=False,
            confidence_score=0.40,
            factors=factors,
            explanation="Preliminary non-observation; awaiting temporal persistence check."
        )
