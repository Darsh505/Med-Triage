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
    ) -> float:
        """
        Evaluate a single MedAction.
        Note: The actual correct/incorrect check is done by the environment and passed here.
        """
        if action.action_type == ActionType.EXAMINE:
            return -0.01
        elif action.action_type == ActionType.TEST:
            return -0.05
        elif action.action_type == ActionType.DIAGNOSE:
            if is_diagnosis_correct:
                return 0.60
            else:
                return -0.50
        elif action.action_type == ActionType.TREAT:
            if is_treatment_correct:
                return 0.40
            else:
                return -0.50

        # Default zero reward
        return 0.0

    async def score(self, trajectory):
        """
        Required by some versions of openenv Rubric if they use trajectory scoring natively.
        We primarily rely on explicit reward shaping per step.
        """
        pass
