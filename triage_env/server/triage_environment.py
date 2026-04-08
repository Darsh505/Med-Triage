import random

from openenv.core.env_server import Environment

from triage_env.models import ActionType, MedAction, MedObservation, MedState
from triage_env.patients import PATIENT_POOL
from triage_env.server.rubrics import MedDenseRubric


class TriageEnvironment(Environment):
    """
    MedTriage-Env core environment.
    """

    def __init__(self):
        super().__init__()
        self.rubric = MedDenseRubric()
        self._current_patient = None
        self._state = MedState()
        self._observation = None

    def reset(self) -> MedObservation:
        """
        Reset the environment, pick a new patient scenario,
        and return the initial observation.
        """
        self._current_patient = random.choice(PATIENT_POOL)
        self._state = MedState()

        self._observation = MedObservation(
            terminal_output=self._current_patient.initial_symptoms,
            patient_vitals=self._current_patient.vitals.copy(),
            test_results={},
            done=False,
            reward=0.0,
        )
        return self._observation

    def step(self, action: MedAction) -> MedObservation:
        """
        Process the action based on the dense reward rubric
        and return the new observation.
        """
        if getattr(self._observation, "done", False):
            # Already done, no further actions allowed
            return self._observation

        if self._state.diagnosis_submitted and action.action_type != ActionType.TREAT:
            # Only treatment permitted after diagnosis
            return self._observation

        reward = 0.0
        done = False
        terminal_output = ""

        # Flags to pass to rubric
        is_diag_correct = False
        is_treat_correct = False

        # Make a copy of old test results
        new_test_results = self._observation.test_results.copy()

        if action.action_type == ActionType.EXAMINE:
            target = action.target.lower()
            if target in self._current_patient.exam_findings:
                terminal_output = self._current_patient.exam_findings[target]
            else:
                terminal_output = f"No significant findings in {target}."

        elif action.action_type == ActionType.TEST:
            target = action.target.lower()
            if target in self._current_patient.test_results:
                result_text = self._current_patient.test_results[target]
                new_test_results[target] = result_text
                terminal_output = f"Test result for {target}: {result_text}"
            else:
                terminal_output = f"Test {target} returned normal results."

        elif action.action_type == ActionType.DIAGNOSE:
            target = action.target.lower()
            if target == self._current_patient.correct_diagnosis:
                is_diag_correct = True
                terminal_output = "Correct diagnosis submitted."
            else:
                is_diag_correct = False
                terminal_output = f"Incorrect diagnosis: {target}"

            self._state.diagnosis_submitted = True
            if not is_diag_correct:
                done = True  # Fail condition

        elif action.action_type == ActionType.TREAT:
            target = action.target.lower()
            if target == self._current_patient.correct_treatment:
                is_treat_correct = True
                terminal_output = "Correct treatment applied. Patient condition improving."
            else:
                is_treat_correct = False
                terminal_output = f"Incorrect treatment: {target}. Patient condition worsening."
            done = True  # Any treatment ends the episode

        # Calculate reward
        reward = self.rubric.evaluate_action(
            action=action,
            is_diagnosis_correct=is_diag_correct,
            is_treatment_correct=is_treat_correct,
        )

        self._state.step_count += 1
        self._state.total_reward += reward

        self._observation = MedObservation(
            terminal_output=terminal_output,
            patient_vitals=self._observation.patient_vitals.copy(),
            test_results=new_test_results,
            done=done,
            reward=reward,
        )

        return self._observation

    def state(self) -> MedState:
        return self._state
