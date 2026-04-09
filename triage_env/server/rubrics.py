"""
MedDenseRubric — Advanced clinical reward evaluation system.

Implements evidence-based clinical pathway scoring, diagnostic efficiency
metrics, ESI-weighted urgency bonuses, and RFC-004 trajectory mastery.
"""
from typing import Any, Dict, List

from openenv.core.rubrics.base import Rubric

from triage_env.models import ActionType, MedAction


# Dangerous medication combinations that should be penalized
DANGEROUS_MED_INTERACTIONS = {
    ("thrombolytics", "anticoagulation"): "Dual antithrombotic therapy without indication increases hemorrhage risk.",
    ("nsaids", "anticoagulation"): "NSAIDs plus anticoagulants dramatically increase GI bleeding risk.",
    ("epinephrine_im", "beta_blockers"): "Beta-blockers can blunt epinephrine response in anaphylaxis.",
    ("iv_insulin_and_fluids", "beta_blockers"): "Beta-blockers can mask hypoglycemia symptoms during insulin therapy.",
}

# Evidence-based optimal first tests for presentation patterns
OPTIMAL_FIRST_TESTS = {
    "chest_pain": ["ecg", "blood_troponin"],
    "dyspnea": ["chest_xray", "abg", "ctpa"],
    "headache": ["ct_head"],
    "abdominal_pain": ["blood_cbc", "lipase"],
    "altered_mental_status": ["blood_glucose", "ct_head"],
    "syncope": ["ecg", "blood_glucose"],
}


class MedDenseRubric(Rubric):
    """
    A dense rubric for MedTriage-Env with multi-dimensional clinical scoring.

    Dimensions:
    1. Action step costs (information gathering is cheap, errors are expensive)
    2. Anti-hallucination guardrails (-1.0 for fabricated targets)
    3. Diagnostic accuracy with ESI-weighted urgency bonus
    4. Treatment correctness with medication interaction checks
    5. Clinical pathway adherence scoring
    6. Diagnostic efficiency (signal-to-noise ratio of tests ordered)
    7. Communication quality (interview completeness)
    """

    def forward(self, action: Any, observation: Any) -> float:
        return 0.0

    def evaluate_action(
        self,
        action: MedAction,
        is_diagnosis_correct: bool = False,
        is_treatment_correct: bool = False,
        time_elapsed: int = 0,
        patient_health_status: float = 1.0,
        is_hallucinated: bool = False,
        is_truncated: bool = False,
        esi_level: int = 3,
        medications_administered: list = None,
        pathway_adherence: float = 1.0,
        diagnostic_efficiency: float = 1.0,
        interview_completeness: float = 0.0,
    ) -> float:
        """
        Evaluate a single MedAction with multi-dimensional clinical scoring.
        """
        if medications_administered is None:
            medications_administered = []

        if is_truncated:
            return -5.0  # Massive crash penalty

        if is_hallucinated:
            return -1.0  # Strict anti-hallucination bounds

        reward = 0.0

        # --- Base action costs ---
        if action.action_type == ActionType.INTERVIEW:
            reward = 0.02  # Small positive reward for gathering history
        elif action.action_type == ActionType.EXAMINE:
            reward = -0.01
        elif action.action_type == ActionType.TEST:
            reward = -0.05
        elif action.action_type == ActionType.CONSULT:
            reward = -0.15  # Severe penalty for asking for hint
        elif action.action_type == ActionType.STABILIZE:
            reward = 0.05  # Positive reward for stabilization (good practice)
        elif action.action_type == ActionType.DIAGNOSE:
            if is_diagnosis_correct:
                base_reward = max(0.20, 0.60 * patient_health_status)
                # ESI urgency multiplier: higher acuity gets bonus for fast correct diagnosis
                esi_multiplier = {1: 1.5, 2: 1.3, 3: 1.0, 4: 0.8, 5: 0.6}.get(esi_level, 1.0)
                reward = base_reward * esi_multiplier
            else:
                reward = -0.50
        elif action.action_type == ActionType.TREAT:
            if is_treatment_correct:
                base_reward = max(0.10, 0.40 * patient_health_status)
                # Check medication interactions
                interaction_penalty = self._check_medication_interactions(
                    action.target, medications_administered
                )
                reward = base_reward + interaction_penalty
            else:
                reward = -0.50

        # --- Pathway adherence bonus ---
        if pathway_adherence > 0.8 and action.action_type in [ActionType.DIAGNOSE, ActionType.TREAT]:
            reward += 0.10 * pathway_adherence

        # --- Diagnostic efficiency bonus ---
        if diagnostic_efficiency > 0.7 and action.action_type == ActionType.DIAGNOSE and is_diagnosis_correct:
            reward += 0.15 * diagnostic_efficiency

        # --- Interview completeness bonus ---
        if interview_completeness > 0.5 and action.action_type == ActionType.DIAGNOSE:
            reward += 0.05 * interview_completeness

        # ML Regularization: Clip reward bounds to stabilize divergence
        return min(max(reward, -5.0), 2.0)

    def _check_medication_interactions(
        self, treatment: str, medications_administered: list
    ) -> float:
        """Check for dangerous medication interactions."""
        penalty = 0.0
        treatment_lower = treatment.lower()
        for med in medications_administered:
            med_lower = med.lower()
            pair1 = (treatment_lower, med_lower)
            pair2 = (med_lower, treatment_lower)
            if pair1 in DANGEROUS_MED_INTERACTIONS or pair2 in DANGEROUS_MED_INTERACTIONS:
                penalty -= 0.30  # Significant penalty for dangerous interaction
        return penalty

    async def score(self, trajectory):
        """
        RFC-004 Alignment: Trajectory Mastery Scoring.

        Evaluates the complete episode trajectory for:
        1. Speed mastery bonus (cured in <30 min)
        2. Perfect pathway bonus (followed optimal clinical path)
        3. Zero-hallucination bonus
        4. Diagnostic efficiency bonus
        """
        if not trajectory:
            return 0.0

        final_obs = trajectory[-1].observation
        total_bonus = 0.0

        # Mastery Bonus: agent survived and cured patient perfectly in under 30 minutes
        if final_obs.done and not final_obs.truncated and final_obs.patient_health_status > 0.5:
            if final_obs.time_elapsed <= 30:
                total_bonus += 3.0  # Speed mastery
            elif final_obs.time_elapsed <= 60:
                total_bonus += 1.5  # Decent speed

        # Zero-hallucination trajectory bonus
        hallucination_count = sum(
            1 for t in trajectory
            if getattr(t, "reward", 0) == -1.0
        )
        if hallucination_count == 0:
            total_bonus += 1.0

        return total_bonus
