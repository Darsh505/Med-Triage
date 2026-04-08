from enum import Enum
from typing import Any, Dict

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class ActionType(str, Enum):
    EXAMINE = "EXAMINE"
    TEST = "TEST"
    DIAGNOSE = "DIAGNOSE"
    TREAT = "TREAT"
    CONSULT = "CONSULT"
    INTERVIEW = "INTERVIEW"


class MedAction(Action):
    """
    Action format the Agent uses to interact with the MedTriage-Env.
    """

    action_type: ActionType = Field(
        ...,
        description="The type of action to perform: EXAMINE, TEST, DIAGNOSE, TREAT",
    )
    target: str = Field(
        ...,
        description="The specific target for the action, e.g., 'chest', 'blood_cbc', 'appendicitis', etc.",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional parameters required for the action.",
    )


class MedObservation(Observation):
    """
    Observation format returned by MedTriage-Env.
    """

    terminal_output: str = Field(
        ...,
        description="The main text feedback from the environment, e.g., initial symptoms or test results.",
    )
    patient_vitals: Dict[str, Any] = Field(
        default_factory=dict,
        description="Known patient vitals.",
    )
    test_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Results of tests previously ordered.",
    )
    done: bool = Field(
        False,
        description="Whether the episode has terminated ordinarily.",
    )
    truncated: bool = Field(
        False,
        description="Whether the episode was truncated due to max steps boundary limit.",
    )
    time_elapsed: int = Field(
        0,
        description="Time elapsed in the episode in minutes.",
    )
    patient_health_status: float = Field(
        1.0,
        description="Health status of the patient (1.0 healthy, 0.0 critical).",
    )
    reward: float = Field(
        0.0,
        description="The reward accumulated from the last step.",
    )
    available_actions: Dict[str, list[str]] = Field(
        default_factory=dict,
        description="A list of allowed target strings dynamically enumerated for each ActionType.",
    )


class MedState(State):
    """
    State to track the total diagnostic process.
    """

    step_count: int = Field(0, description="Number of actions taken so far.")
    total_reward: float = Field(0.0, description="Total reward accumulated so far.")
    diagnosis_submitted: bool = Field(
        False, description="Whether a diagnosis was successfully submitted."
    )
    audit_trail: list[dict] = Field(
        default_factory=list,
        description="A trace logger of action targets and rewards.",
    )
