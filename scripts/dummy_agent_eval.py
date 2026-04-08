import asyncio
import os
import sys

# Adds root to path so we can run directly if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triage_env.models import ActionType, MedAction


async def evaluate_agent_strategy(strategy_name: str, verbose: bool = False):
    """
    Evaluates different agent benchmarking strategies.
    Since we are not interacting with a deployed server here for simplicity,
    we can test directly using the native environment, but this script represents
    the interface logic for the OpenEnv.
    """
    from triage_env.server.triage_environment import TriageEnvironment

    env = TriageEnvironment()
    obs = env.reset()

    if verbose:
        print(f"=== Starting Eval: {strategy_name} ===")
        print(f"Initial Symptoms: {obs.terminal_output[:100]}...")

    actions_taken = 0
    total_time = 0

    # Strategy 1: Random Guesser (Bad Strategy)
    if strategy_name == "random":
        import random

        while not obs.done and actions_taken < 10:
            target = random.choice(["blood_cbc", "abdomen", "flu", "surgery"])
            act_type = random.choice(list(ActionType))
            obs = env.step(MedAction(action_type=act_type, target=target))
            actions_taken += 1

    # Strategy 2: The Optimal Doctor (Knows everything immediately)
    elif strategy_name == "optimal":
        correct_diag = env._current_patient.correct_diagnosis
        correct_treat = env._current_patient.correct_treatment
        obs = env.step(MedAction(action_type=ActionType.DIAGNOSE, target=correct_diag))
        actions_taken += 1
        obs = env.step(MedAction(action_type=ActionType.TREAT, target=correct_treat))
        actions_taken += 1

    # Strategy 3: The Over-Tester (Loses health due to time)
    elif strategy_name == "overtester":
        for test in ["blood_cbc", "mri", "ct_scan"]:
            obs = env.step(MedAction(action_type=ActionType.TEST, target=test))
            actions_taken += 1

        correct_diag = env._current_patient.correct_diagnosis
        correct_treat = env._current_patient.correct_treatment
        obs = env.step(MedAction(action_type=ActionType.DIAGNOSE, target=correct_diag))
        actions_taken += 1
        obs = env.step(MedAction(action_type=ActionType.TREAT, target=correct_treat))
        actions_taken += 1

    state = env.state()
    total_time = obs.time_elapsed
    patient_health = obs.patient_health_status

    if verbose:
        print(f"Total Steps: {actions_taken}")
        print(f"Time Elapsed: {total_time} mins")
        print(f"Patient Health: {patient_health:.2f}")
        print(f"Total Reward: {state.total_reward:.2f}\n")

    return state.total_reward


async def main():
    print("--- MedTriage-Env: Evaluation Benchmark ---")
    score_random = await evaluate_agent_strategy("random")
    score_overtester = await evaluate_agent_strategy("overtester")
    score_optimal = await evaluate_agent_strategy("optimal")

    print("Results Average Metric:")
    print(f"Random Guesser:  {score_random:.2f}")
    print(f"Over-Tester:     {score_overtester:.2f} (Penalized for Temporal Degradation)")
    print(f"Optimal Doctor:  {score_optimal:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
