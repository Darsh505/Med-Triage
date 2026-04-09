#!/usr/bin/env python3
"""
inference.py - OpenEnv Hackathon structured inference runner.

Uses the hackathon-provided LiteLLM proxy (API_BASE_URL + MODEL_NAME)
to make LLM calls, and prints [START]/[STEP]/[END] blocks to stdout.
"""
import sys
import os
import json
import urllib.request

# Force unbuffered stdout so the validator always sees output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
else:
    os.environ["PYTHONUNBUFFERED"] = "1"

# Add project root to path so imports work even without pip install
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from triage_env.server.triage_environment import TriageEnvironment
from triage_env.models import MedAction, ActionType


def query_llm(prompt: str, api_base_url: str, model_name: str, api_key: str) -> str:
    """
    Call the hackathon's LiteLLM proxy using OpenAI-compatible API format.
    Handles multiple URL formats the proxy might use.
    """
    base = api_base_url.rstrip("/")

    # Try multiple URL patterns to handle different proxy configurations
    urls_to_try = []
    if "/chat/completions" in base:
        urls_to_try.append(base)
    elif base.endswith("/v1"):
        urls_to_try.append(base + "/chat/completions")
    else:
        urls_to_try.append(base + "/v1/chat/completions")
        urls_to_try.append(base + "/chat/completions")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a medical triage AI agent. Given patient information, "
                    "decide the next clinical action. Respond ONLY with valid JSON: "
                    '{"action_type": "<TYPE>", "target": "<target>"}. '
                    "Valid action_types: EXAMINE, TEST, DIAGNOSE, TREAT, INTERVIEW. "
                    "Pick targets from the available_actions provided. "
                    "Pick the most medically relevant action based on symptoms."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 150,
    }
    payload = json.dumps(data).encode("utf-8")

    for url in urls_to_try:
        try:
            req = urllib.request.Request(
                url, data=payload, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"> LLM proxy error ({url}): {e}", flush=True)
            continue

    return ""


def parse_llm_action(response_text: str, available_actions: dict, step: int) -> MedAction:
    """Parse LLM response into a MedAction, with intelligent fallback."""
    if response_text:
        try:
            clean = response_text
            # Strip markdown code fences
            for marker in ["```json", "```JSON", "```"]:
                clean = clean.replace(marker, "")
            clean = clean.strip()
            # Handle cases where LLM wraps in extra text
            start = clean.find("{")
            end = clean.rfind("}") + 1
            if start >= 0 and end > start:
                clean = clean[start:end]
            data = json.loads(clean)
            action_type_str = data.get("action_type", "EXAMINE").upper()
            target_str = data.get("target", "abdomen").lower()
            action_type = ActionType(action_type_str)
            return MedAction(action_type=action_type, target=target_str)
        except Exception:
            pass

    # Intelligent fallback based on step number
    exam_targets = available_actions.get("EXAMINE", [])
    test_targets = available_actions.get("TEST", [])
    diag_targets = available_actions.get("DIAGNOSE", [])
    treat_targets = available_actions.get("TREAT", [])

    if step <= 2 and exam_targets:
        idx = min(step - 1, len(exam_targets) - 1)
        return MedAction(action_type=ActionType.EXAMINE, target=exam_targets[idx])
    if step == 3 and test_targets:
        return MedAction(action_type=ActionType.TEST, target=test_targets[0])
    if step == 4 and diag_targets:
        return MedAction(action_type=ActionType.DIAGNOSE, target=diag_targets[0])
    if treat_targets:
        return MedAction(action_type=ActionType.TREAT, target=treat_targets[0])

    return MedAction(action_type=ActionType.EXAMINE, target="abdomen")


def run_inference():
    """Run inference loop with structured [START]/[STEP]/[END] output."""
    # Read hackathon-provided environment variables
    api_base_url = os.environ.get("API_BASE_URL", "")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    # Check all common API key env var names
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("API_KEY")
        or os.environ.get("HF_TOKEN")
        or "no-key"
    )

    env = TriageEnvironment(difficulty="hard")
    obs = env.reset()

    print("[START] task=med_triage", flush=True)

    step = 0
    total_reward = 0.0

    while not getattr(obs, "done", False) and not getattr(obs, "truncated", False):
        step += 1
        available = getattr(obs, "available_actions", {})

        prompt = (
            f"Patient vitals: {getattr(obs, 'patient_vitals', {})}\n"
            f"Current observation: {getattr(obs, 'terminal_output', '')}\n"
            f"Test results so far: {getattr(obs, 'test_results', {})}\n"
            f"Time elapsed: {getattr(obs, 'time_elapsed', 0)} mins\n"
            f"Health status: {getattr(obs, 'patient_health_status', 1.0)}\n"
            f"Available actions: {json.dumps(available)}\n"
            f"Step {step} of max 20. Choose the best next action."
        )

        # Make LLM call through the hackathon proxy
        llm_response = query_llm(prompt, api_base_url, model_name, api_key)
        action = parse_llm_action(llm_response, available, step)

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
