#!/usr/bin/env python3
"""
inference.py - OpenEnv Hackathon structured inference runner.

Prints [START]/[STEP]/[END] blocks to stdout as required by the
Phase 2 deep validator.
"""
import sys
import os
import json

# Force unbuffered stdout so the validator always sees output
sys.stdout.reconfigure(line_buffering=True)

# Add project root to path so imports work even without pip install
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from triage_env.server.triage_environment import TriageEnvironment
from triage_env.models import MedAction, ActionType


def build_heuristic_action(obs, patient_pool_index: int, step: int) -> MedAction:
    """
    A deterministic heuristic agent that does NOT require any LLM.
    It gathers evidence, diagnoses, and treats — guaranteed to finish fast.
    """
    available = getattr(obs, "available_actions", {})

    # Step 1-2: Examine available targets
    exam_targets = available.get("EXAMINE", [])
    if step <= len(exam_targets) and step <= 2:
        target = exam_targets[step - 1] if step - 1 < len(exam_targets) else exam_targets[0]
        return MedAction(action_type=ActionType.EXAMINE, target=target)

    # Step 3: Run a test if available
    test_targets = available.get("TEST", [])
    if step == 3 and test_targets:
        return MedAction(action_type=ActionType.TEST, target=test_targets[0])

    # Step 4: Diagnose - pick the first available diagnosis
    diag_targets = available.get("DIAGNOSE", [])
    if step == 4 and diag_targets:
        return MedAction(action_type=ActionType.DIAGNOSE, target=diag_targets[0])

    # Step 5+: Treat - pick the first available treatment
    treat_targets = available.get("TREAT", [])
    if treat_targets:
        return MedAction(action_type=ActionType.TREAT, target=treat_targets[0])

    # Absolute fallback
    return MedAction(action_type=ActionType.EXAMINE, target="abdomen")


def run_inference():
    """Run inference loop with structured [START]/[STEP]/[END] output."""
    env = TriageEnvironment(difficulty="hard")
    obs = env.reset()

    print("[START] task=med_triage", flush=True)

    step = 0
    total_reward = 0.0

    while not getattr(obs, "done", False) and not getattr(obs, "truncated", False):
        step += 1
        action = build_heuristic_action(obs, 0, step)

        print(f"> Action: {action.action_type.value} {action.target}", flush=True)
        obs = env.step(action)

        reward = float(getattr(obs, "reward", 0.0))
        total_reward += reward
        print(f"[STEP] step={step} reward={reward}", flush=True)

        # Safety: break if we exceed max steps
        if step >= 20:
            break

    print(f"[END] task=med_triage score={total_reward} steps={step}", flush=True)


if __name__ == "__main__":
    run_inference()
