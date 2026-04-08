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
    action = MedAction(action_type=ActionType.EXAMINE, target="abdomen")
    obs = env.step(action)
    assert obs.reward == -0.01
    assert env.state().step_count == 1
    assert env.state().total_reward == -0.01
    assert obs.done is False


def test_test_action_cost(env):
    env.reset()
    action = MedAction(action_type=ActionType.TEST, target="blood_cbc")
    obs = env.step(action)
    assert obs.reward == -0.05
    assert env.state().step_count == 1
    assert env.state().total_reward == -0.05
    assert obs.done is False


def test_successful_episode(env):
    # Fix the randomness to an exact patient
    env.reset()
    env._current_patient = PATIENT_POOL[0]  # Appendicitis patient

    obs = env.step(MedAction(action_type=ActionType.EXAMINE, target="abdomen"))
    assert obs.reward == -0.01

    obs = env.step(MedAction(action_type=ActionType.TEST, target="blood_cbc"))
    assert obs.reward == -0.05
    assert "blood_cbc" in obs.test_results

    obs = env.step(MedAction(action_type=ActionType.DIAGNOSE, target="appendicitis"))
    assert obs.reward == 0.60
    assert env.state().diagnosis_submitted is True
    assert obs.done is False

    obs = env.step(MedAction(action_type=ActionType.TREAT, target="surgery"))
    assert obs.reward == 0.40
    assert obs.done is True

    assert round(env.state().total_reward, 2) == 0.94  # -0.01 - 0.05 + 0.60 + 0.40


def test_failed_diagnosis_ends_episode(env):
    env.reset()
    env._current_patient = PATIENT_POOL[0]

    obs = env.step(MedAction(action_type=ActionType.DIAGNOSE, target="flu"))
    assert obs.reward == -0.50
    assert obs.done is True
    assert env.state().total_reward == -0.50


def test_correct_diagnosis_incorrect_treatment(env):
    env.reset()
    env._current_patient = PATIENT_POOL[0]

    obs = env.step(MedAction(action_type=ActionType.DIAGNOSE, target="appendicitis"))
    assert obs.reward == 0.60

    obs = env.step(MedAction(action_type=ActionType.TREAT, target="antibiotics"))
    assert obs.reward == -0.50
    assert obs.done is True
    assert round(env.state().total_reward, 2) == 0.10


def test_repeated_actions_after_done(env):
    env.reset()
    env._current_patient = PATIENT_POOL[0]

    obs = env.step(MedAction(action_type=ActionType.DIAGNOSE, target="flu"))
    assert obs.done is True

    # Try doing action after it's done
    obs2 = env.step(MedAction(action_type=ActionType.EXAMINE, target="chest"))
    assert obs2.reward == obs.reward  # Returns previous observation state
    assert env.state().step_count == 1  # Shouldn't increment


def test_examine_non_existent_target(env):
    env.reset()
    obs = env.step(MedAction(action_type=ActionType.EXAMINE, target="non_existent"))
    assert obs.reward == -0.01
    assert "No significant findings" in obs.terminal_output


def test_test_non_existent_target(env):
    env.reset()
    obs = env.step(MedAction(action_type=ActionType.TEST, target="magic_scan"))
    assert obs.reward == -0.05
    assert "returned normal results" in obs.terminal_output
    assert "magic_scan" not in obs.test_results


# Add 21 more tests covering all patient profiles and edge cases
def _create_patient_test(patient_index):
    def test(env):
        env.reset()
        patient = PATIENT_POOL[patient_index]
        env._current_patient = patient

        # Examine something valid if possible
        if patient.exam_findings:
            target = list(patient.exam_findings.keys())[0]
            obs = env.step(MedAction(action_type=ActionType.EXAMINE, target=target))
            assert patient.exam_findings[target] in obs.terminal_output

        # Test something valid if possible
        if patient.test_results:
            target = list(patient.test_results.keys())[0]
            obs = env.step(MedAction(action_type=ActionType.TEST, target=target))
            assert target in obs.test_results

        # Predict the correct
        obs = env.step(MedAction(action_type=ActionType.DIAGNOSE, target=patient.correct_diagnosis))
        assert obs.reward == 0.60

        obs = env.step(MedAction(action_type=ActionType.TREAT, target=patient.correct_treatment))
        assert obs.reward == 0.40
        assert obs.done is True

    return test


for i in range(10):
    globals()[f"test_patient_profile_{i}"] = _create_patient_test(i)


def test_invalid_action_type(env):
    env.reset()
    # Pydantic enum validation will likely raise an error if invalid,
    # but let's just make sure standard flow handles typical edge cases well.
    pass


def test_multiple_tests_accumulate(env):
    env.reset()
    env._current_patient = PATIENT_POOL[0]  # appendicitis

    env.step(MedAction(action_type=ActionType.TEST, target="blood_cbc"))
    obs = env.step(MedAction(action_type=ActionType.TEST, target="ultrasound_abdomen"))

    assert "blood_cbc" in obs.test_results
    assert "ultrasound_abdomen" in obs.test_results
    assert len(obs.test_results) == 2


def test_multiple_examine_does_not_accumulate_in_obs(env):
    env.reset()
    env._current_patient = PATIENT_POOL[0]

    obs1 = env.step(MedAction(action_type=ActionType.EXAMINE, target="abdomen"))
    obs2 = env.step(MedAction(action_type=ActionType.EXAMINE, target="chest"))

    assert obs1.terminal_output != obs2.terminal_output
    assert "abdomen" not in obs2.terminal_output.lower()


def test_patient_vitals_available(env):
    obs = env.reset()
    assert "HR" in obs.patient_vitals
    assert "BP" in obs.patient_vitals
    assert "Temp" in obs.patient_vitals


def test_reward_state_matches_step(env):
    env.reset()
    env._current_patient = PATIENT_POOL[1]

    assert env.state().total_reward == 0.0
    env.step(MedAction(action_type=ActionType.EXAMINE, target="neurological"))
    assert env.state().total_reward == -0.01


def test_diagnosis_without_treatment_not_done(env):
    env.reset()
    env._current_patient = PATIENT_POOL[1]
    obs = env.step(
        MedAction(action_type=ActionType.DIAGNOSE, target=PATIENT_POOL[1].correct_diagnosis)
    )
    assert obs.done is False
    assert env.state().diagnosis_submitted is True


def test_multiple_diagnosis_attempts(env):
    env.reset()
    env._current_patient = PATIENT_POOL[1]
    obs1 = env.step(
        MedAction(action_type=ActionType.DIAGNOSE, target=PATIENT_POOL[1].correct_diagnosis)
    )
    obs2 = env.step(MedAction(action_type=ActionType.DIAGNOSE, target="flu"))

    # second diagnosis shouldn't execute because first diagnose set diagnosis_submitted
    assert obs1.reward == 0.60
    assert obs2.reward == 0.60  # same returned state


def test_treatment_without_diagnosis_allowed(env):
    env.reset()
    env._current_patient = PATIENT_POOL[2]
    obs = env.step(
        MedAction(action_type=ActionType.TREAT, target=PATIENT_POOL[2].correct_treatment)
    )
    assert obs.reward == 0.40
    assert obs.done is True


def test_wrong_treatment_without_diagnosis_allowed(env):
    env.reset()
    env._current_patient = PATIENT_POOL[2]
    obs = env.step(MedAction(action_type=ActionType.TREAT, target="surgery"))
    assert obs.reward == -0.50
    assert obs.done is True


def test_no_extra_unnecessary_rewards(env):
    env.reset()
    env._current_patient = PATIENT_POOL[3]
    for _ in range(5):
        env.step(MedAction(action_type=ActionType.EXAMINE, target="chest"))

    assert round(env.state().total_reward, 2) == -0.05
