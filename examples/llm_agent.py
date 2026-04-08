import sys
import os
import json
import urllib.request
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triage_env.server.triage_environment import TriageEnvironment
from triage_env.models import MedAction, ActionType

def query_llm(prompt: str, api_key: str) -> str:
    """Zero-shot query against a generic LLM (OpenAI style endpoint)."""
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
    
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        # Mock fallback for benchmark testing if API key fails
        return json.dumps({"action_type": "examine", "target": "abdomen"})

def run_agent_benchmark():
    api_key = os.environ.get("OPENAI_API_KEY", "dummy_key")
    env = TriageEnvironment(difficulty="hard")
    obs = env.reset()
    
    print("=== STARTING ZERO-SHOT LLM BENCHMARK ===")
    
    while not obs.done and not obs.truncated:
        prompt = f"Time Elapsed: {obs.time_elapsed}. Health: {obs.patient_health_status}.\nVitals: {obs.patient_vitals}\nObservation: {obs.terminal_output}\nAvailable: {obs.available_actions}"
        
        # Query Model
        print("\nThinking...")
        response_text = query_llm(prompt, api_key)
        
        try:
            # Strip potential markdown blocks
            clean_json = response_text.replace("```json", "").replace("```", "")
            action_data = json.loads(clean_json)
            
            action = MedAction(
                action_type=ActionType(action_data["action_type"].lower()), 
                target=action_data["target"].lower()
            )
            print(f"> LLM Chose: {action.action_type.value.upper()} {action.target}")
            
            obs = env.step(action)
            time.sleep(1) # Visual pacing
            
        except Exception as e:
            print(f"Agent generated invalid schema: {response_text}. Applying severe penalties!")
            obs = env.step(MedAction(action_type=ActionType.EXAMINE, target="hallucinated_invalid"))
            
    # Final Results
    state = env.state()
    print("\n=== EPISODE FINISHED ===")
    print(obs.terminal_output)
    print(f"Final Total Reward: {state.total_reward:.2f}")
    if obs.truncated:
        print("RESULT: FAILURE (Agent crashed patient)")
    elif state.total_reward > 0:
        print("RESULT: SUCCESS (Agent solved patient)")
    else:
        print("RESULT: FAILED (Agent incorrectly diagnosed/treated)")

if __name__ == "__main__":
    run_agent_benchmark()
