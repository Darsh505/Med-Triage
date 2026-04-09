from enum import Enum
from typing import Any, Dict, List

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class ActionType(str, Enum):
    EXAMINE = "EXAMINE"
    TEST = "TEST"
    DIAGNOSE = "DIAGNOSE"
    TREAT = "TREAT"
    CONSULT = "CONSULT"
    INTERVIEW = "INTERVIEW"
    STABILIZE = "STABILIZE"  # Manage vitals before definitive treatment


class MedAction(Action):
    """
    Action format the Agent uses to interact with the MedTriage-Env.
    """

    action_type: ActionType = Field(
        ...,
        description="The type of action to perform: EXAMINE, TEST, DIAGNOSE, TREAT, STABILIZE",
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
        description="The main text feedback from the environment.",
    )
    patient_vitals: Dict[str, Any] = Field(
        default_factory=dict,
        description="Known patient vitals (HR, BP, Temp, RR, SpO2).",
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
        description="Whether the episode was truncated due to max steps or patient crash.",
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
        description="The reward from the last step.",
    )
    available_actions: Dict[str, list[str]] = Field(
        default_factory=dict,
        description="Allowed target strings for each ActionType.",
    )
    # --- Advanced observation fields ---
    medications_administered: List[str] = Field(
        default_factory=list,
        description="Medications given during this episode (for interaction checking).",
    )
    esi_level: int = Field(
        3,
        description="Emergency Severity Index (1=resuscitation, 5=non-urgent).",
    )
    clinical_notes: List[str] = Field(
        default_factory=list,
        description="Running log of significant clinical events this episode.",
    )
    sepsis_risk: float = Field(
        0.0,
        description="Computed sepsis risk score (0.0 = none, 1.0 = septic shock).",
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
    # --- Advanced state fields ---
    differential_diagnoses: List[str] = Field(
        default_factory=list,
        description="Running list of differential diagnoses considered by the agent.",
    )
    evidence_gathered: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of evidence type to finding for clinical reasoning.",
    )
    pathway_score: float = Field(
        0.0, description="Score for following optimal clinical pathway."
    )
    diagnostic_efficiency: float = Field(
        0.0, description="Ratio of useful tests ordered vs total tests."
    )
    interviews_conducted: List[str] = Field(
        default_factory=list,
        description="Track which interview topics have been covered.",
    )
