import logging
import os
import random
import time

from openenv.core.env_server import Environment

from triage_env.models import ActionType, MedAction, MedObservation, MedState
from triage_env.patients import PATIENT_POOL
from triage_env.server.rubrics import MedDenseRubric

logger = logging.getLogger(__name__)


class TriageEnvironment(Environment):
    """
    MedTriage-Env core environment.
    """

    MAX_STEPS = 20

    def __init__(self, difficulty: str = "all"):
        super().__init__()
        self.rubric = MedDenseRubric()

        # Difficulty Filtering Support
        self.difficulty = difficulty.lower()
        if self.difficulty in ["easy", "hard"]:
            self.active_patient_pool = [
                p for p in PATIENT_POOL if getattr(p, "difficulty", "easy") == self.difficulty
            ]
        elif self.difficulty == "infinite":
            self.active_patient_pool = []  # Will procedurally generate in reset()
        else:
            self.active_patient_pool = PATIENT_POOL

        self._current_patient = None
        self._state = MedState()
        self._observation = None
        self._np_random = random.Random()

    def seed(self, seed: int):
        """
        Seed the environment's PRNG for deterministic reproducibility.
        """
        self._np_random.seed(seed)
        logger.info(f"MedTriage-Env seeded with {seed}")

    def _build_available_actions(self) -> dict[str, list[str]]:
        if not self._current_patient:
            return {}
        return {
            ActionType.INTERVIEW.value: list(self._current_patient.interview_responses.keys()),
            ActionType.EXAMINE.value: list(self._current_patient.exam_findings.keys()),
            ActionType.TEST.value: list(self._current_patient.test_results.keys()),
            ActionType.DIAGNOSE.value: [
                self._current_patient.correct_diagnosis,
                "flu",
                "pulmonary_embolism",
                "viral_gastroenteritis",
            ],
            ActionType.TREAT.value: [
                self._current_patient.correct_treatment,
                "surgery",
                "iv_fluids",
                "nsaids",
            ],
        }

    def reset(self) -> MedObservation:
        """
        Reset the environment, pick a new patient scenario,
        and return the initial observation.
        """
        if self.difficulty == "infinite":
            self._current_patient = self._generate_procedural_patient()
        else:
            self._current_patient = self._np_random.choice(self.active_patient_pool)

        self._state = MedState()

        self._observation = MedObservation(
            terminal_output=self._current_patient.initial_symptoms,
            patient_vitals=self._current_patient.vitals.copy(),
            test_results={},
            done=False,
            truncated=False,
            time_elapsed=0,
            patient_health_status=1.0,
            reward=0.0,
            available_actions=self._build_available_actions(),
        )
        logger.info(f"Environment reset. Patient ID: {self._current_patient.id}")
        return self._observation

    def step(self, action: MedAction) -> MedObservation:
        """
        Process the action based on the dense reward rubric
        and return the new observation.
        """
        if getattr(self._observation, "done", False) or getattr(
            self._observation, "truncated", False
        ):
            # Already done or truncated, no further actions allowed
            return self._observation

        # Hard Step Bounds Prevention
        if self._state.step_count >= self.MAX_STEPS:
            self._observation.truncated = True
            self._observation.terminal_output = "Episode Truncated: Max step limit reached."
            # Apply severe negative regularization penalty for looping!
            self._state.total_reward -= 2.0
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
        is_hallucinated = False

        # Make a copy of old test results
        new_test_results = self._observation.test_results.copy()
        time_spent = 0
        current_health = self._observation.patient_health_status

        if action.action_type == ActionType.INTERVIEW:
            target = action.target.lower()
            valid_targets = self._observation.available_actions.get(action.action_type.value, [])
            if target not in valid_targets:
                is_hallucinated = True
                time_spent = 5
                terminal_output = f"Invalid {action.action_type.value} target: {target}."
            else:
                time_spent = 2  # Asking questions is extremely fast
                if (
                    hasattr(self._current_patient, "interview_responses")
                    and target in self._current_patient.interview_responses
                ):
                    terminal_output = self._current_patient.interview_responses[target]
                else:
                    terminal_output = f"Patient has no comment on {target}."

        elif action.action_type == ActionType.EXAMINE:
            target = action.target.lower()
            valid_targets = self._observation.available_actions.get(action.action_type.value, [])
            if target not in valid_targets:
                is_hallucinated = True
                time_spent = 5
                terminal_output = f"Invalid {action.action_type.value} target: {target}."
            else:
                time_spent = 5  # Exams take 5 mins
                if target in self._current_patient.exam_findings:
                    terminal_output = self._current_patient.exam_findings[target]
                else:
                    terminal_output = f"No significant findings in {target}."

        elif action.action_type == ActionType.TEST:
            target = action.target.lower()
            valid_targets = self._observation.available_actions.get(action.action_type.value, [])
            if target not in valid_targets:
                is_hallucinated = True
                time_spent = 15
                terminal_output = f"Invalid {action.action_type.value} target: {target}."
            else:
                time_spent = 45  # Tests take 45 mins
                if target in self._current_patient.test_results:
                    result_text = self._current_patient.test_results[target]
                    new_test_results[target] = result_text
                    terminal_output = f"Test result for {target}: {result_text}"
                else:
                    terminal_output = f"Test {target} returned normal results."

        elif action.action_type == ActionType.CONSULT:
            time_spent = 15
            terminal_output = f"Attending physician notes patient might have {self._current_patient.correct_diagnosis}."

        elif action.action_type == ActionType.DIAGNOSE:
            time_spent = 10
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
            time_spent = 5
            target = action.target.lower()
            if target == self._current_patient.correct_treatment:
                is_treat_correct = True
                terminal_output = "Correct treatment applied. Patient condition improving."
            else:
                is_treat_correct = False
                terminal_output = f"Incorrect treatment: {target}. Patient condition worsening."
            done = True  # Any treatment ends the episode

        # Update Time & Deterioration
        # Calculate Time & Deterioration
        new_time_elapsed = self._observation.time_elapsed + time_spent
        is_truncated = False

        if new_time_elapsed >= 180:
            # Episode crash truncation!
            is_truncated = True
            current_health = 0.0
            terminal_output += " FATAL: Patient crashed due to massive treatment delay."
        elif new_time_elapsed > 60:
            current_health -= 0.15
            terminal_output += (
                " Warning: Patient condition critically worsening due to time elapsed."
            )
        elif new_time_elapsed > 30:
            current_health -= 0.05

        current_health = max(0.0, current_health)

        # 1. Chaos Engine: Dynamic Vitals Drift
        new_vitals = self._observation.patient_vitals.copy()
        try:
            if current_health < 0.70 and "HR" in new_vitals:
                # Add deterministic physical decay matching health status
                base_hr = (
                    int(new_vitals["HR"].split(" ")[0])
                    if new_vitals["HR"].split(" ")[0].isdigit()
                    else 100
                )
                drift = int((0.70 - current_health) * 100)
                new_vitals["HR"] = f"{base_hr + drift} (Drifting upward due to treatment delay)"
                if "BP" in new_vitals:
                    new_vitals["BP"] = "Dropping rapidly (Orthostatic deterioration)"
        except Exception:
            pass  # Fallback if JSON format was irregular

        # 2. Clinical After-Action Audit Report & Discharge Markdown Dump
        if done or is_truncated:
            valid_tests = self._observation.available_actions.get(ActionType.TEST.value, [])
            missed_tests = len(valid_tests) - len(new_test_results)
            report_body = f"\n\n--- CLINICAL AUDIT REPORT ---\nTime Elapsed: {new_time_elapsed} mins\nCritical Tests Missed: {missed_tests}\nFinal Patient Health: {current_health * 100:.0f}%\n-----------------------------"
            terminal_output += report_body

            try:
                os.makedirs("reports", exist_ok=True)
                report_file = (
                    f"reports/Discharge_Report_{self._current_patient.id}_{int(time.time())}.md"
                )
                with open(report_file, "w") as f:
                    f.write(
                        f"# Medical Discharge Report\n\n**Patient ID:** {self._current_patient.id}\n"
                    )
                    f.write(
                        f"**Presentation:** {self._current_patient.initial_symptoms}\n\n## Timeline Log\n"
                    )
                    for audit in self._state.audit_trail:
                        f.write(
                            f"- Step {audit['step']} | [ {audit['action_type'].upper()} ] -> {audit['target']} | Costs: {audit['time_spent']}m | Reward: {audit['reward']:.2f}\n"
                        )
                    f.write(f"\n## Clinical Conclusion\n{terminal_output}\n")
            except Exception as e:
                logger.error(f"Failed to generate medical discharge report: {e}")

        # Calculate reward
        reward = self.rubric.evaluate_action(
            action=action,
            is_diagnosis_correct=is_diag_correct,
            is_treatment_correct=is_treat_correct,
            time_elapsed=new_time_elapsed,
            patient_health_status=current_health,
            is_hallucinated=is_hallucinated,
            is_truncated=is_truncated,
        )

        self._state.step_count += 1
        self._state.total_reward += reward
        self._state.audit_trail.append(
            {
                "step": self._state.step_count,
                "action_type": action.action_type.value,
                "target": action.target,
                "time_spent": time_spent,
                "reward": reward,
            }
        )
        logger.info(
            f"Step {self._state.step_count} | Action: {action.action_type.value} | Target: {action.target} | Reward: {reward:.2f}"
        )

        self._observation = MedObservation(
            terminal_output=terminal_output,
            patient_vitals=new_vitals,
            test_results=new_test_results,
            done=done,
            truncated=is_truncated,  # Crashing bound natively returns early Truncation
            time_elapsed=new_time_elapsed,
            patient_health_status=current_health,
            reward=reward,
            available_actions=self._build_available_actions(),
        )

        return self._observation

    def _generate_procedural_patient(self):
        """
        Dynamically synthesize a new patient scenario for infinite RL horizons.
        """
        import copy

        # Randomly mutate patients to act as an infinite offline generation sandbox
        # In a generic environment, you hook ChatGPT here directly via API.
        base_patient = copy.deepcopy(self._np_random.choice(PATIENT_POOL))
        base_patient.id = f"P_SYNTH_{random.randint(1000, 9999)}"
        base_patient.initial_symptoms += (
            f" (Patient arrived at {random.randint(1, 12)}:00 PM displaying atypical distress)."
        )
        # Perturb vitals slightly to prevent static overfitting
        if "HR" in base_patient.vitals:
            try:
                hr = int(base_patient.vitals["HR"].split(" ")[0])
                base_patient.vitals["HR"] = f"{hr + random.randint(-15, 15)} bpm"
            except Exception:
                pass
        return base_patient

    def state(self) -> MedState:
        return self._state
