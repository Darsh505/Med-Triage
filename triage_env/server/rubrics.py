from typing import Any

from openenv.core.rubrics.base import Rubric

from triage_env.models import ActionType, MedAction


class MedDenseRubric(Rubric):
    """
    A dense rubric for MedTriage-Env.
    Assigns rewards/penalties based on the actions taken.
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
    ) -> float:
        """
        Evaluate a single MedAction.
        Reward scales with patient_health_status.
        """
        if is_truncated:
            return -5.0  # Massive crash penalty

        if is_hallucinated:
            return -1.0  # Strict anti-hallucination bounds

        if action.action_type == ActionType.INTERVIEW:
            reward = 0.0  # Free to ask questions mostly!
        elif action.action_type == ActionType.EXAMINE:
            reward = -0.01
        elif action.action_type == ActionType.TEST:
            reward = -0.05
        elif action.action_type == ActionType.CONSULT:
            reward = -0.15  # Severe penalty for hint
        elif action.action_type == ActionType.DIAGNOSE:
            if is_diagnosis_correct:
                reward = max(0.20, 0.60 * patient_health_status)
            else:
                reward = -0.50
        elif action.action_type == ActionType.TREAT:
            if is_treatment_correct:
                reward = max(0.10, 0.40 * patient_health_status)
            else:
                reward = -0.50
        else:
            reward = 0.0

        # ML Regularization: Clip reward bounds to stabilize divergence
        return min(max(reward, -2.0), 1.0)

    async def score(self, trajectory):
        """
        RFC-004 Alignment: Trajectory Mastery Scoring.
        Reward high-efficiency end-of-episode outcomes.
        """
        if not trajectory:
            return 0.0

        final_obs = trajectory[-1].observation

        # Mastery Bonus: if agent survived and cured patient perfectly in under 30 minutes!
        if final_obs.done and not final_obs.truncated and final_obs.patient_health_status > 0.5:
            if final_obs.time_elapsed <= 30:
                return 3.0

        return 0.0
