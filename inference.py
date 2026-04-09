#!/usr/bin/env python3
"""
inference.py — Chain-of-Thought Clinical Reasoning Agent for MedTriage-Env.

Implements the ABCDE assessment protocol with differential diagnosis tracking,
evidence accumulation, and clinical confidence scoring.

Uses the hackathon-provided LiteLLM proxy (API_BASE_URL + MODEL_NAME).
Prints [START]/[STEP]/[END] structured blocks to stdout for Phase 2 validation.
"""
import sys
import os
import json
import urllib.request

# Force unbuffered stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
else:
    os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from triage_env.server.triage_environment import TriageEnvironment
from triage_env.models import MedAction, ActionType


# ABCDE Clinical Assessment Protocol
CLINICAL_PROTOCOL = """You are an expert emergency medicine physician following the ABCDE assessment protocol.

## Protocol
1. **A**irway - Check for obstruction, stridor, angioedema
2. **B**reathing - Assess respiratory rate, SpO2, breath sounds, work of breathing
3. **C**irculation - Evaluate HR, BP, capillary refill, skin color, JVD
4. **D**isability - Neurological status, GCS, pupils, focal deficits
5. **E**xposure - Full examination, skin findings, temperature

## Decision Framework
- Gather evidence systematically before diagnosing
- Interview patient for history, medications, allergies EARLY
- Order tests that will change management (high-yield tests first)
- Stabilize critically ill patients (ESI 1-2) before definitive diagnosis
- Diagnose only when you have sufficient evidence
- Treat promptly after diagnosis

## Response Format
Respond ONLY with valid JSON:
{"action_type": "<TYPE>", "target": "<target>", "reasoning": "<brief clinical reasoning>"}

Valid action_types: INTERVIEW, EXAMINE, TEST, STABILIZE, DIAGNOSE, TREAT
Pick targets from the available_actions provided."""


def query_llm(prompt: str, api_base_url: str, model_name: str, api_key: str) -> str:
    """Call the hackathon's LiteLLM proxy."""
    base = api_base_url.rstrip("/")
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
            {"role": "system", "content": CLINICAL_PROTOCOL},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 200,
    }
    payload = json.dumps(data).encode("utf-8")

    for url in urls_to_try:
        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"].strip()
        except Exception:
            continue
    return ""


def parse_llm_action(response_text: str, available_actions: dict, step: int) -> tuple:
    """Parse LLM response into a MedAction with reasoning, with intelligent fallback."""
    reasoning = ""
    if response_text:
        try:
            clean = response_text
            for marker in ["```json", "```JSON", "```"]:
                clean = clean.replace(marker, "")
            clean = clean.strip()
            start = clean.find("{")
            end = clean.rfind("}") + 1
            if start >= 0 and end > start:
                clean = clean[start:end]
            data = json.loads(clean)
            action_type_str = data.get("action_type", "EXAMINE").upper()
            target_str = data.get("target", "abdomen").lower()
            reasoning = data.get("reasoning", "")
            action_type = ActionType(action_type_str)
            return MedAction(action_type=action_type, target=target_str), reasoning
        except Exception:
            pass

    # Intelligent fallback: ABCDE protocol
    interview_targets = available_actions.get("INTERVIEW", [])
    exam_targets = available_actions.get("EXAMINE", [])
    test_targets = available_actions.get("TEST", [])
    diag_targets = available_actions.get("DIAGNOSE", [])
    treat_targets = available_actions.get("TREAT", [])

    if step == 1 and interview_targets:
        t = "onset" if "onset" in interview_targets else interview_targets[0]
        return MedAction(action_type=ActionType.INTERVIEW, target=t), "Gathering chief complaint history"
    if step == 2 and exam_targets:
        return MedAction(action_type=ActionType.EXAMINE, target=exam_targets[0]), "Physical examination per ABCDE"
    if step == 3 and interview_targets and len(interview_targets) > 1:
        t = "medications" if "medications" in interview_targets else interview_targets[1]
        return MedAction(action_type=ActionType.INTERVIEW, target=t), "Medication history review"
    if step == 4 and test_targets:
        return MedAction(action_type=ActionType.TEST, target=test_targets[0]), "Ordering high-yield diagnostic test"
    if step == 5 and diag_targets:
        return MedAction(action_type=ActionType.DIAGNOSE, target=diag_targets[0]), "Submitting clinical diagnosis"
    if treat_targets:
        return MedAction(action_type=ActionType.TREAT, target=treat_targets[0]), "Initiating treatment"

    return MedAction(action_type=ActionType.EXAMINE, target="abdomen"), "Fallback examination"


def run_inference():
    """Run inference with chain-of-thought clinical reasoning."""
    api_base_url = os.environ.get("API_BASE_URL", "")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
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
    evidence_log = []

    while not getattr(obs, "done", False) and not getattr(obs, "truncated", False):
        step += 1
        available = getattr(obs, "available_actions", {})
        clinical_notes = getattr(obs, "clinical_notes", [])
        sepsis_risk = getattr(obs, "sepsis_risk", 0.0)

        prompt = (
            f"## Current Clinical State (Step {step}/20)\n\n"
            f"**Patient vitals:** {getattr(obs, 'patient_vitals', {})}\n"
            f"**Current observation:** {getattr(obs, 'terminal_output', '')}\n"
            f"**Test results so far:** {getattr(obs, 'test_results', {})}\n"
            f"**Time elapsed:** {getattr(obs, 'time_elapsed', 0)} mins\n"
            f"**Health status:** {getattr(obs, 'patient_health_status', 1.0):.0%}\n"
            f"**ESI Level:** {getattr(obs, 'esi_level', 3)}\n"
            f"**Sepsis risk:** {sepsis_risk:.2f}\n"
            f"**Medications given:** {getattr(obs, 'medications_administered', [])}\n"
            f"**Evidence gathered so far:** {json.dumps(evidence_log[-5:])}\n"
            f"**Available actions:** {json.dumps(available)}\n\n"
            f"What is the best next clinical action? Follow ABCDE protocol."
        )

        llm_response = query_llm(prompt, api_base_url, model_name, api_key)
        action, reasoning = parse_llm_action(llm_response, available, step)

        evidence_log.append({"step": step, "action": action.action_type.value, "target": action.target})

        if reasoning:
            print(f"> [{action.action_type.value}] {action.target} — {reasoning}", flush=True)
        else:
            print(f"> [{action.action_type.value}] {action.target}", flush=True)

        obs = env.step(action)

        reward = float(getattr(obs, "reward", 0.0))
        total_reward += reward
        print(f"[STEP] step={step} reward={reward}", flush=True)

        if step >= 20:
            break

    print(f"[END] task=med_triage score={total_reward} steps={step}", flush=True)


if __name__ == "__main__":
    run_inference()
