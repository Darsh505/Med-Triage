#!/usr/bin/env python3
"""
inference.py — Chain-of-Thought Clinical Reasoning Agent for MedTriage-Env.

Uses the hackathon-provided LiteLLM proxy via the OpenAI Python client.
Reads API_BASE_URL and API_KEY from environment variables.
Prints [START]/[STEP]/[END] structured blocks to stdout for Phase 2 validation.
"""
import sys
import os
import json

# Force unbuffered stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
else:
    os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

from triage_env.server.triage_environment import TriageEnvironment
from triage_env.models import MedAction, ActionType


# ABCDE Clinical Assessment Protocol
CLINICAL_PROTOCOL = """You are an expert emergency medicine physician following the ABCDE assessment protocol.

## Protocol
1. **A**irway - Check for obstruction, stridor, angioedema
2. **B**reathing - Assess respiratory rate, SpO2, breath sounds
3. **C**irculation - Evaluate HR, BP, capillary refill, skin color
4. **D**isability - Neurological status, GCS, pupils
5. **E**xposure - Full examination, skin findings, temperature

## Decision Framework
- Gather evidence systematically before diagnosing
- Interview patient for history, medications, allergies EARLY
- Order high-yield tests first
- Stabilize critically ill patients before definitive diagnosis
- Diagnose only when you have sufficient evidence
- Treat promptly after diagnosis

## Response Format
Respond ONLY with valid JSON:
{"action_type": "<TYPE>", "target": "<target>", "reasoning": "<brief clinical reasoning>"}

Valid action_types: INTERVIEW, EXAMINE, TEST, STABILIZE, DIAGNOSE, TREAT
Pick targets from the available_actions provided."""


def query_llm(client: OpenAI, prompt: str, model_name: str) -> str:
    """Call the hackathon's LiteLLM proxy via the OpenAI client."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": CLINICAL_PROTOCOL},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"> LLM proxy error: {e}", flush=True)
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
        return MedAction(action_type=ActionType.INTERVIEW, target=t), "Gathering chief complaint"
    if step == 2 and exam_targets:
        return MedAction(action_type=ActionType.EXAMINE, target=exam_targets[0]), "Physical exam per ABCDE"
    if step == 3 and interview_targets and len(interview_targets) > 1:
        t = "medications" if "medications" in interview_targets else interview_targets[1]
        return MedAction(action_type=ActionType.INTERVIEW, target=t), "Medication history"
    if step == 4 and test_targets:
        return MedAction(action_type=ActionType.TEST, target=test_targets[0]), "High-yield diagnostic test"
    if step == 5 and diag_targets:
        return MedAction(action_type=ActionType.DIAGNOSE, target=diag_targets[0]), "Clinical diagnosis"
    if treat_targets:
        return MedAction(action_type=ActionType.TREAT, target=treat_targets[0]), "Initiating treatment"

    return MedAction(action_type=ActionType.EXAMINE, target="abdomen"), "Fallback examination"


def clamp_score(raw_reward: float) -> float:
    """Clamp raw reward to strictly between 0 and 1 (exclusive) as validator requires."""
    # Normalize: our rewards typically range from -5 to +3
    # Map to (0, 1) using a sigmoid-like transform
    normalized = (raw_reward + 5.0) / 10.0  # maps [-5, 5] -> [0, 1]
    return max(0.01, min(0.99, normalized))


def run_single_task(client: OpenAI, model_name: str, task_name: str, difficulty: str):
    """Run a single task (patient episode) with structured output."""
    env = TriageEnvironment(difficulty=difficulty)
    obs = env.reset()

    print(f"[START] task={task_name}", flush=True)

    step = 0
    total_reward = 0.0
    evidence_log = []

    while not getattr(obs, "done", False) and not getattr(obs, "truncated", False):
        step += 1
        available = getattr(obs, "available_actions", {})

        prompt = (
            f"## Clinical State (Step {step}/20)\n\n"
            f"**Vitals:** {getattr(obs, 'patient_vitals', {})}\n"
            f"**Observation:** {getattr(obs, 'terminal_output', '')}\n"
            f"**Tests done:** {getattr(obs, 'test_results', {})}\n"
            f"**Time:** {getattr(obs, 'time_elapsed', 0)} mins\n"
            f"**Health:** {getattr(obs, 'patient_health_status', 1.0):.0%}\n"
            f"**ESI:** {getattr(obs, 'esi_level', 3)}\n"
            f"**Sepsis risk:** {getattr(obs, 'sepsis_risk', 0.0):.2f}\n"
            f"**Meds given:** {getattr(obs, 'medications_administered', [])}\n"
            f"**Evidence so far:** {json.dumps(evidence_log[-5:])}\n"
            f"**Available actions:** {json.dumps(available)}\n\n"
            f"What is the best next clinical action?"
        )

        llm_response = query_llm(client, prompt, model_name)
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

    # Clamp score to strictly (0, 1) as validator requires
    final_score = clamp_score(total_reward)
    print(f"[END] task={task_name} score={final_score} steps={step}", flush=True)


def run_inference():
    """Run 3 tasks across different difficulties to satisfy validator requirements."""
    api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("API_KEY", os.environ.get("OPENAI_API_KEY", "no-key"))
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    client = OpenAI(
        base_url=api_base_url,
        api_key=api_key,
    )

    # Run 3 tasks with different difficulty levels
    tasks = [
        ("med_triage_easy", "easy"),
        ("med_triage_hard", "hard"),
        ("med_triage_expert", "expert"),
    ]

    for task_name, difficulty in tasks:
        run_single_task(client, model_name, task_name, difficulty)


if __name__ == "__main__":
    run_inference()

