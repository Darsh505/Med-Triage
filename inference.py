import sys
import os
import json
import urllib.request
import time

from triage_env.client import TriageEnvClient
from triage_env.server.triage_environment import TriageEnvironment
from triage_env.models import MedAction, ActionType

def query_llm(prompt: str, api_key: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an RL Agent solving the MedTriage-Env task. Respond ONLY with the next action in JSON format: {'action_type': 'test', 'target': 'blood_cbc'}. Available action types: interview, examine, test, consult, diagnose, treat."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return json.dumps({"action_type": "examine", "target": "abdomen"})

def run_inference():
    api_key = os.environ.get("OPENAI_API_KEY", "dummy_key")
    base_url = os.environ.get("API_BASE_URL", "")
    
    if base_url:
        print(f"Connecting to remote environment at {base_url}")
        env = TriageEnvClient(base_url=base_url)
    else:
        print("Using local environment")
        env = TriageEnvironment(difficulty="hard")
        
    obs = env.reset()
    print("[START] task=med_triage", flush=True)
    step = 0
    total_reward = 0.0
    
    while getattr(obs, "done", False) is False and getattr(obs, "truncated", False) is False:
        step += 1
        prompt = f"Time Elapsed: {getattr(obs, 'time_elapsed', 0)}. Health: {getattr(obs, 'patient_health_status', 1)}.\nVitals: {getattr(obs, 'patient_vitals', {})}\nObservation: {getattr(obs, 'terminal_output', '')}\nAvailable: {getattr(obs, 'available_actions', {})}"
        print("\nThinking...")
        response_text = query_llm(prompt, api_key)
        
        try:
            clean_json = response_text.replace("```json", "").replace("```", "")
            action_data = json.loads(clean_json)
            action = MedAction(
                action_type=ActionType(action_data["action_type"].lower()), 
                target=action_data["target"].lower()
            )
            print(f"> LLM Chose: {action.action_type.value.upper()} {action.target}")
            obs = env.step(action)
        except Exception:
            obs = env.step(MedAction(action_type=ActionType.EXAMINE, target="hallucinated_invalid"))
            
        reward = getattr(obs, "reward", 0.0)
        total_reward += float(reward)
        print(f"[STEP] step={step} reward={reward}", flush=True)
        time.sleep(1)
            
    print("\n=== INFERENCE EPISODE FINISHED ===")
    print(getattr(obs, "terminal_output", ""))
    print(f"[END] task=med_triage score={total_reward} steps={step}", flush=True)

if __name__ == "__main__":
    run_inference()
