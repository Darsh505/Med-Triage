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
sys.stdout.reconfigure(line_buffering=True)

# Add project root to path so imports work even without pip install
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from triage_env.server.triage_environment import TriageEnvironment
from triage_env.models import MedAction, ActionType


def query_llm(prompt: str, api_base_url: str, model_name: str, api_key: str) -> str:
    """
    Call the hackathon's LiteLLM proxy using OpenAI-compatible API format.
    """
    # Normalize base URL - ensure it ends with /v1/chat/completions
    base = api_base_url.rstrip("/")
    if not base.endswith("/v1"):
        if "/v1" not in base:
            base = base + "/v1"
    url = base + "/chat/completions"

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
                    "Pick targets from the available_actions provided."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 150,
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"> LLM proxy error: {e}", flush=True)
        return ""


def parse_llm_action(response_text: str, available_actions: dict) -> MedAction:
    """Parse LLM response into a MedAction, with robust fallback."""
    try:
        clean = response_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        action_type = ActionType(data["action_type"].upper())
        target = data["target"].lower()
        return MedAction(action_type=action_type, target=target)
    except Exception:
        pass

    # Fallback: pick first available treatment to end episode
    treat_targets = available_actions.get("TREAT", [])
    if treat_targets:
        return MedAction(action_type=ActionType.TREAT, target=treat_targets[0])
    return MedAction(action_type=ActionType.EXAMINE, target="abdomen")


def run_inference():
    """Run inference loop with structured [START]/[STEP]/[END] output."""
    # Read hackathon-provided environment variables
    api_base_url = os.environ.get("API_BASE_URL", "")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY", os.environ.get("HF_TOKEN", "no-key"))

    if not api_base_url:
        print("> WARNING: API_BASE_URL not set, LLM calls will fail", flush=True)

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
            f"Observation: {getattr(obs, 'terminal_output', '')}\n"
            f"Time elapsed: {getattr(obs, 'time_elapsed', 0)} mins\n"
            f"Health status: {getattr(obs, 'patient_health_status', 1.0)}\n"
            f"Available actions: {json.dumps(available)}\n"
            f"Step {step} of max 20. Choose wisely."
        )

        # Make LLM call through the hackathon proxy
        llm_response = query_llm(prompt, api_base_url, model_name, api_key)
        action = parse_llm_action(llm_response, available)

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
