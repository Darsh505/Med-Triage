import math

import pytest

from triage_env.models import ActionType, MedAction
from triage_env.patients import PATIENT_POOL
from triage_env.server.triage_environment import TriageEnvironment


@pytest.fixture
def env():
    return TriageEnvironment()


def test_initial_reset(env):
    obs = env.reset()
    assert obs.terminal_output is not None
    assert obs.done is False
    assert obs.reward == 0.0
    state = env.state()
    assert state.step_count == 0
    assert state.total_reward == 0.0


def test_examine_action_cost(env):
    env.reset()
    env._current_patient = PATIENT_POOL[0]
    env._observation.available_actions = env._build_available_actions()
    action = MedAction(action_type=ActionType.EXAMINE, target="abdomen")
    obs = env.step(action)
    assert obs.reward == -0.01
    assert env.state().step_count == 1
    assert env.state().total_reward == -0.01
    assert obs.done is False


def test_seed_determinism(env):
    env.seed(42)
    obs1 = env.reset()
    env2 = TriageEnvironment()
    env2.seed(42)
    obs2 = env2.reset()
    assert obs1.terminal_output == obs2.terminal_output
    assert env._current_patient.id == env2._current_patient.id


def test_test_action_cost(env):
    env.reset()
    env._current_patient = PATIENT_POOL[0]
    env._observation.available_actions = env._build_available_actions()
    action = MedAction(action_type=ActionType.TEST, target="blood_cbc")
    obs = env.step(action)
    assert obs.reward == -0.05
    assert env.state().step_count == 1
    assert env.state().total_reward == -0.05
    assert obs.done is False


def test_successful_episode(env):
    """Test a complete successful episode: examine, test, diagnose, treat."""
    env.reset()
    env._current_patient = PATIENT_POOL[0]  # Appendicitis (ESI-3)
    env._observation.available_actions = env._build_available_actions()

    obs = env.step(MedAction(action_type=ActionType.EXAMINE, target="abdomen"))
    assert obs.reward == -0.01

    obs = env.step(MedAction(action_type=ActionType.TEST, target="blood_cbc"))
    assert obs.reward == -0.05
    assert "blood_cbc" in obs.test_results

    obs = env.step(MedAction(action_type=ActionType.DIAGNOSE, target="appendicitis"))
    # ESI-3 multiplier = 1.0, health ~0.94 after 60min, reward = max(0.20, 0.60*0.94)*1.0 + pathway bonus
    assert obs.reward > 0.4  # Positive reward for correct diagnosis
    assert env.state().diagnosis_submitted is True
    assert obs.done is False

    obs = env.step(MedAction(action_type=ActionType.TREAT, target="surgery"))
    assert obs.reward > 0.1  # Positive reward for correct treatment
    assert obs.done is True

    assert env.state().total_reward > 0  # Overall positive outcome


def test_failed_diagnosis_ends_episode(env):
    env.reset()
    env._current_patient = PATIENT_POOL[0]
    env._observation.available_actions = env._build_available_actions()

    obs = env.step(MedAction(action_type=ActionType.DIAGNOSE, target="flu"))
    assert obs.reward == -0.50
    assert obs.done is True
    assert env.state().total_reward == -0.50


def test_correct_diagnosis_incorrect_treatment(env):
    env.reset()
    env._current_patient = PATIENT_POOL[0]
    env._observation.available_actions = env._build_available_actions()

    obs = env.step(MedAction(action_type=ActionType.DIAGNOSE, target="appendicitis"))
    diag_reward = obs.reward
    assert diag_reward > 0  # Correct diagnosis gives positive reward

    obs = env.step(MedAction(action_type=ActionType.TREAT, target="antibiotics"))
    assert obs.reward < 0  # Wrong treatment gives negative reward
    assert obs.done is True


def test_repeated_actions_after_done(env):
    env.reset()
    env._current_patient = PATIENT_POOL[0]
    env._observation.available_actions = env._build_available_actions()

    obs = env.step(MedAction(action_type=ActionType.DIAGNOSE, target="flu"))
    assert obs.done is True

    obs2 = env.step(MedAction(action_type=ActionType.EXAMINE, target="chest"))
    assert obs2.reward == obs.reward
    assert env.state().step_count == 1


def test_examine_non_existent_target(env):
    env.reset()
    obs = env.step(MedAction(action_type=ActionType.EXAMINE, target="non_existent"))
    assert obs.reward == -1.0
    assert "Invalid EXAMINE" in obs.terminal_output


def test_test_non_existent_target(env):
    env.reset()
    obs = env.step(MedAction(action_type=ActionType.TEST, target="magic_scan"))
    assert obs.reward == -1.0
    assert "Invalid TEST" in obs.terminal_output
    assert "magic_scan" not in obs.test_results


def test_time_crashing_truncation(env):
    env.reset()
    env._current_patient = PATIENT_POOL[0]
    env._observation.available_actions = env._build_available_actions()
    env.step(MedAction(action_type=ActionType.TEST, target="blood_cbc"))
    env.step(MedAction(action_type=ActionType.TEST, target="ultrasound_abdomen"))
    env.step(MedAction(action_type=ActionType.TEST, target="blood_cbc"))
    obs = env.step(MedAction(action_type=ActionType.TEST, target="ultrasound_abdomen"))

    assert obs.truncated is True
    assert obs.patient_health_status == 0.0
    assert obs.reward == -5.0
    assert "FATAL: Patient crashed" in obs.terminal_output


# Dynamic patient profile tests for all original patients
def _create_patient_test(patient_index):
    def test(env):
        env.reset()
        patient = PATIENT_POOL[patient_index]
        env._current_patient = patient
        env._observation.available_actions = env._build_available_actions()

        if patient.exam_findings:
            target = list(patient.exam_findings.keys())[0]
            obs = env.step(MedAction(action_type=ActionType.EXAMINE, target=target))
            assert patient.exam_findings[target] in obs.terminal_output

        if patient.test_results:
            target = list(patient.test_results.keys())[0]
            obs = env.step(MedAction(action_type=ActionType.TEST, target=target))
            assert target in obs.test_results

        obs = env.step(MedAction(action_type=ActionType.DIAGNOSE, target=patient.correct_diagnosis))
        assert obs.reward > 0.1

        obs = env.step(MedAction(action_type=ActionType.TREAT, target=patient.correct_treatment))
        assert obs.reward > 0.05
        assert obs.done is True

    return test


# Test first 10 patients (original set)
for i in range(min(10, len(PATIENT_POOL))):
    globals()[f"test_patient_profile_{i}"] = _create_patient_test(i)


def test_invalid_action_type(env):
    env.reset()
    pass


def test_multiple_tests_accumulate(env):
    env.reset()
    env._current_patient = PATIENT_POOL[0]
    env._observation.available_actions = env._build_available_actions()

    env.step(MedAction(action_type=ActionType.TEST, target="blood_cbc"))
    obs = env.step(MedAction(action_type=ActionType.TEST, target="ultrasound_abdomen"))

    assert "blood_cbc" in obs.test_results
    assert "ultrasound_abdomen" in obs.test_results
    assert len(obs.test_results) == 2


def test_multiple_examine_does_not_accumulate_in_obs(env):
    env.reset()
    env._current_patient = PATIENT_POOL[0]
    env._observation.available_actions = env._build_available_actions()

    obs1 = env.step(MedAction(action_type=ActionType.EXAMINE, target="abdomen"))
    obs2 = env.step(MedAction(action_type=ActionType.EXAMINE, target="general"))

    assert obs1.terminal_output != obs2.terminal_output


def test_patient_vitals_available(env):
    obs = env.reset()
    assert "HR" in obs.patient_vitals
    assert "BP" in obs.patient_vitals
    assert "Temp" in obs.patient_vitals


def test_reward_state_matches_step(env):
    env.reset()
    env._current_patient = PATIENT_POOL[1]
    env._observation.available_actions = env._build_available_actions()

    assert env.state().total_reward == 0.0
    env.step(MedAction(action_type=ActionType.EXAMINE, target="neurological"))
    assert env.state().total_reward == -0.01


def test_diagnosis_without_treatment_not_done(env):
    env.reset()
    env._current_patient = PATIENT_POOL[1]
    env._observation.available_actions = env._build_available_actions()
    obs = env.step(
        MedAction(action_type=ActionType.DIAGNOSE, target=PATIENT_POOL[1].correct_diagnosis)
    )
    assert obs.done is False
    assert env.state().diagnosis_submitted is True


def test_multiple_diagnosis_attempts(env):
    env.reset()
    env._current_patient = PATIENT_POOL[1]
    env._observation.available_actions = env._build_available_actions()
    obs1 = env.step(
        MedAction(action_type=ActionType.DIAGNOSE, target=PATIENT_POOL[1].correct_diagnosis)
    )
    obs2 = env.step(MedAction(action_type=ActionType.DIAGNOSE, target="flu"))

    assert obs1.reward > 0  # Correct diagnosis
    assert obs2.reward == obs1.reward  # Second diagnosis blocked, returns same state


def test_treatment_without_diagnosis_allowed(env):
    env.reset()
    env._current_patient = PATIENT_POOL[2]
    env._observation.available_actions = env._build_available_actions()
    obs = env.step(
        MedAction(action_type=ActionType.TREAT, target=PATIENT_POOL[2].correct_treatment)
    )
    assert obs.reward > 0
    assert obs.done is True


def test_wrong_treatment_without_diagnosis_allowed(env):
    env.reset()
    env._current_patient = PATIENT_POOL[2]
    env._observation.available_actions = env._build_available_actions()
    obs = env.step(MedAction(action_type=ActionType.TREAT, target="surgery"))
    assert obs.reward < 0
    assert obs.done is True


def test_no_extra_unnecessary_rewards(env):
    env.reset()
    env._current_patient = PATIENT_POOL[3]
    env._observation.available_actions = env._build_available_actions()
    for _ in range(5):
        env.step(
            MedAction(
                action_type=ActionType.EXAMINE, target=list(PATIENT_POOL[3].exam_findings.keys())[0]
            )
        )
    assert round(env.state().total_reward, 2) == -0.05


def test_truncation_max_steps():
    env = TriageEnvironment()
    obs = env.reset()
    assert obs.truncated is False

    for _ in range(20):
        obs = env.step(MedAction(action_type=ActionType.EXAMINE, target="wall"))
        assert obs.truncated is False

    obs = env.step(MedAction(action_type=ActionType.EXAMINE, target="wall"))
    assert obs.truncated is True
    assert "Truncated" in obs.terminal_output
    assert env.state().step_count == 20


def test_curriculum_difficulty_initialization():
    env_easy = TriageEnvironment(difficulty="easy")
    env_easy.seed(1337)
    env_easy.reset()
    assert env_easy._current_patient.difficulty == "easy"

    env_hard = TriageEnvironment(difficulty="hard")
    env_hard.seed(1337)
    env_hard.reset()
    assert env_hard._current_patient.difficulty == "hard"

    env_expert = TriageEnvironment(difficulty="expert")
    env_expert.seed(1337)
    env_expert.reset()
    assert env_expert._current_patient.difficulty == "expert"


# --- NEW TESTS for advanced mechanics ---

def test_stabilize_action():
    env = TriageEnvironment()
    env.reset()
    obs = env.step(MedAction(action_type=ActionType.STABILIZE, target="iv_access"))
    assert obs.reward > 0  # Stabilization should give positive reward
    assert "IV access" in obs.terminal_output


def test_interview_action():
    env = TriageEnvironment()
    env.reset()
    env._current_patient = PATIENT_POOL[0]  # Appendicitis
    env._observation.available_actions = env._build_available_actions()
    obs = env.step(MedAction(action_type=ActionType.INTERVIEW, target="onset"))
    assert obs.reward > 0  # Interview gives small positive reward
    assert "pain" in obs.terminal_output.lower() or "belly" in obs.terminal_output.lower()


def test_esi_level_in_observation():
    env = TriageEnvironment()
    obs = env.reset()
    assert obs.esi_level in [1, 2, 3, 4, 5]


def test_clinical_notes_accumulate():
    env = TriageEnvironment()
    env.reset()
    env._current_patient = PATIENT_POOL[0]
    env._observation.available_actions = env._build_available_actions()
    initial_notes = len(env._observation.clinical_notes)
    env.step(MedAction(action_type=ActionType.EXAMINE, target="abdomen"))
    assert len(env._observation.clinical_notes) > initial_notes


def test_sepsis_risk_computed():
    env = TriageEnvironment()
    obs = env.reset()
    assert isinstance(obs.sepsis_risk, float)
    assert 0.0 <= obs.sepsis_risk <= 1.0


def test_medications_tracked():
    env = TriageEnvironment()
    env.reset()
    env._current_patient = PATIENT_POOL[0]
    env._observation.available_actions = env._build_available_actions()
    env.step(MedAction(action_type=ActionType.STABILIZE, target="fluid_bolus"))
    assert len(env._medications_given) > 0


def test_30_patients_loaded():
    assert len(PATIENT_POOL) >= 30


def test_all_patients_have_required_fields():
    for p in PATIENT_POOL:
        assert p.id
        assert p.initial_symptoms
        assert p.vitals
        assert p.correct_diagnosis
        assert p.correct_treatment
        assert p.difficulty in ["easy", "hard", "expert"]
        assert p.esi_level in [1, 2, 3, 4, 5]


def test_expert_difficulty_pool():
    env = TriageEnvironment(difficulty="expert")
    env.seed(42)
    env.reset()
    assert env._current_patient.difficulty == "expert"


def test_evidence_gathered_tracking():
    env = TriageEnvironment()
    env.reset()
    env._current_patient = PATIENT_POOL[0]
    env._observation.available_actions = env._build_available_actions()
    env.step(MedAction(action_type=ActionType.EXAMINE, target="abdomen"))
    assert "exam_abdomen" in env.state().evidence_gathered
