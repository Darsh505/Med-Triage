from typing import Type

from openenv.core.env_client import EnvClient

from triage_env.models import MedAction, MedObservation, MedState


class TriageEnvClient(EnvClient[MedAction, MedObservation, MedState]):
    """
    Client for interacting with the MedTriage-Env.
    """

    @property
    def action_type(self) -> Type[MedAction]:
        return MedAction

    @property
    def observation_type(self) -> Type[MedObservation]:
        return MedObservation

    @property
    def state_type(self) -> Type[MedState]:
        return MedState

    def get_action_examples(self) -> list[MedAction]:
        """Provide some example actions for testing/agents."""
        from triage_env.models import ActionType

        return [
            MedAction(action_type=ActionType.EXAMINE, target="abdomen"),
            MedAction(action_type=ActionType.TEST, target="blood_cbc"),
            MedAction(action_type=ActionType.DIAGNOSE, target="appendicitis"),
        ]
