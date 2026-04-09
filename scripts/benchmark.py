#!/usr/bin/env python3
"""
benchmark.py — Comprehensive benchmarking suite for MedTriage-Env.

Runs all 30 patients through multiple agent strategies and produces
formatted results showing diagnostic accuracy, efficiency, and scoring.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triage_env.server.triage_environment import TriageEnvironment
from triage_env.models import MedAction, ActionType
from triage_env.patients import PATIENT_POOL


def run_optimal_agent(patient_index: int) -> dict:
    """Agent that knows the correct answer — represents theoretical maximum."""
    env = TriageEnvironment()
    env.reset()
    patient = PATIENT_POOL[patient_index]
    env._current_patient = patient
    env._observation.available_actions = env._build_available_actions()

    # Optimal path: interview, examine, test, diagnose, treat
    if patient.interview_responses:
        key = list(patient.interview_responses.keys())[0]
        env.step(MedAction(action_type=ActionType.INTERVIEW, target=key))

    if patient.exam_findings:
        key = list(patient.exam_findings.keys())[0]
        env.step(MedAction(action_type=ActionType.EXAMINE, target=key))

    if patient.test_results:
        key = list(patient.test_results.keys())[0]
        env.step(MedAction(action_type=ActionType.TEST, target=key))

    env.step(MedAction(action_type=ActionType.DIAGNOSE, target=patient.correct_diagnosis))
    obs = env.step(MedAction(action_type=ActionType.TREAT, target=patient.correct_treatment))

    return {
        "patient": patient.id,
        "diagnosis": patient.correct_diagnosis,
        "esi": patient.esi_level,
        "difficulty": patient.difficulty,
        "reward": env.state().total_reward,
        "steps": env.state().step_count,
        "time": obs.time_elapsed,
        "health": obs.patient_health_status,
        "correct_diag": True,
        "correct_treat": True,
    }


def run_heuristic_agent(patient_index: int) -> dict:
    """A simple heuristic agent — examine, test, diagnose, treat."""
    env = TriageEnvironment()
    env.reset()
    patient = PATIENT_POOL[patient_index]
    env._current_patient = patient
    env._observation.available_actions = env._build_available_actions()

    available = env._observation.available_actions

    # Step 1: Examine first target
    exam_targets = available.get("EXAMINE", [])
    if exam_targets:
        env.step(MedAction(action_type=ActionType.EXAMINE, target=exam_targets[0]))

    # Step 2: Run first test
    test_targets = available.get("TEST", [])
    if test_targets:
        env.step(MedAction(action_type=ActionType.TEST, target=test_targets[0]))

    # Step 3: Diagnose (pick first option — may be wrong)
    diag_targets = available.get("DIAGNOSE", [])
    diag_target = diag_targets[0] if diag_targets else "flu"
    obs = env.step(MedAction(action_type=ActionType.DIAGNOSE, target=diag_target))

    correct_diag = (diag_target == patient.correct_diagnosis)

    # Step 4: Treat (if not already done)
    if not obs.done:
        treat_targets = available.get("TREAT", [])
        treat_target = treat_targets[0] if treat_targets else "iv_fluids"
        obs = env.step(MedAction(action_type=ActionType.TREAT, target=treat_target))
        correct_treat = (treat_target == patient.correct_treatment)
    else:
        correct_treat = False

    return {
        "patient": patient.id,
        "diagnosis": patient.correct_diagnosis,
        "esi": patient.esi_level,
        "difficulty": patient.difficulty,
        "reward": env.state().total_reward,
        "steps": env.state().step_count,
        "time": obs.time_elapsed,
        "health": obs.patient_health_status,
        "correct_diag": correct_diag,
        "correct_treat": correct_treat,
    }


def run_random_agent(patient_index: int) -> dict:
    """Random agent — picks actions randomly. Baseline for comparison."""
    import random
    random.seed(42 + patient_index)

    env = TriageEnvironment()
    env.reset()
    patient = PATIENT_POOL[patient_index]
    env._current_patient = patient
    env._observation.available_actions = env._build_available_actions()

    obs = env._observation
    steps = 0
    while not obs.done and not obs.truncated and steps < 10:
        action_types = [ActionType.EXAMINE, ActionType.TEST, ActionType.DIAGNOSE, ActionType.TREAT]
        act_type = random.choice(action_types)
        targets = obs.available_actions.get(act_type.value, ["abdomen"])
        target = random.choice(targets) if targets else "abdomen"
        obs = env.step(MedAction(action_type=act_type, target=target))
        steps += 1

    return {
        "patient": patient.id,
        "diagnosis": patient.correct_diagnosis,
        "esi": patient.esi_level,
        "difficulty": patient.difficulty,
        "reward": env.state().total_reward,
        "steps": env.state().step_count,
        "time": obs.time_elapsed,
        "health": obs.patient_health_status,
        "correct_diag": env.state().diagnosis_submitted,
        "correct_treat": obs.done and env.state().total_reward > 0,
    }


def format_table(results: list[dict], strategy: str) -> str:
    """Format results as markdown table."""
    lines = [
        f"\n### {strategy}\n",
        "| Patient | Diagnosis | ESI | Diff | Reward | Steps | Time | Health | Diag | Treat |",
        "|---------|-----------|-----|------|--------|-------|------|--------|------|-------|",
    ]
    total_reward = 0
    correct_diag_count = 0
    correct_treat_count = 0

    for r in results:
        diag_icon = "Y" if r["correct_diag"] else "N"
        treat_icon = "Y" if r["correct_treat"] else "N"
        lines.append(
            f"| {r['patient']} | {r['diagnosis'][:20]} | {r['esi']} | {r['difficulty'][:4]} | "
            f"{r['reward']:+.2f} | {r['steps']} | {r['time']}m | {r['health']:.0%} | "
            f"{diag_icon} | {treat_icon} |"
        )
        total_reward += r["reward"]
        if r["correct_diag"]:
            correct_diag_count += 1
        if r["correct_treat"]:
            correct_treat_count += 1

    n = len(results)
    lines.append(f"\n**{strategy} Summary**: Avg Reward: {total_reward/n:+.2f} | "
                 f"Diag Accuracy: {correct_diag_count}/{n} ({correct_diag_count/n:.0%}) | "
                 f"Treat Accuracy: {correct_treat_count}/{n} ({correct_treat_count/n:.0%})")
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("  MedTriage-Env: Comprehensive Benchmarking Suite")
    print(f"  Total Patients: {len(PATIENT_POOL)}")
    print("=" * 70)

    strategies = {
        "Optimal Agent (Oracle)": run_optimal_agent,
        "Heuristic Agent (Rule-Based)": run_heuristic_agent,
        "Random Agent (Baseline)": run_random_agent,
    }

    all_results = {}
    for name, agent_fn in strategies.items():
        print(f"\nRunning {name}...")
        results = []
        for i in range(len(PATIENT_POOL)):
            result = agent_fn(i)
            results.append(result)
        all_results[name] = results
        print(format_table(results, name))

    # Summary comparison
    print("\n" + "=" * 70)
    print("  STRATEGY COMPARISON")
    print("=" * 70)
    print("\n| Strategy | Avg Reward | Diag Acc | Treat Acc |")
    print("|----------|-----------|----------|-----------|")
    for name, results in all_results.items():
        n = len(results)
        avg_r = sum(r["reward"] for r in results) / n
        diag_acc = sum(1 for r in results if r["correct_diag"]) / n
        treat_acc = sum(1 for r in results if r["correct_treat"]) / n
        print(f"| {name[:30]:30} | {avg_r:+.2f} | {diag_acc:.0%} | {treat_acc:.0%} |")


if __name__ == "__main__":
    main()
