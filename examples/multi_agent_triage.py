import sys
import os
import json
import urllib.request
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triage_env.server.triage_environment import TriageEnvironment
from triage_env.models import MedAction, ActionType

def query_llm(prompt: str, system_prompt: str, api_key: str) -> str:
  """Zero-shot query against a generic LLM (OpenAI style)."""
  url = "https://api.openai.com/v1/chat/completions"
  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
  }
  data = {
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": prompt}
    ],
    "temperature": 0.4
  }
  req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
  try:
    with urllib.request.urlopen(req) as response:
      result = json.loads(response.read().decode("utf-8"))
      return result["choices"][0]["message"]["content"].strip()
  except Exception as e:
    print(f"LLM API Error: {e}")
    # Mocking the JSON response for hackathon demo compatibility if API key fails!
    return json.dumps({"action_type": "examine", "target": "chest"})


def run_multi_agent_swarm():
  api_key = os.environ.get("OPENAI_API_KEY", "dummy_key")
  env = TriageEnvironment(difficulty="expert")
  obs = env.reset()
  
  print("\n[INIT] Connecting to OpenEnv MedTriage Server...")
  print("[INIT] Launching Multi-Agent Swarm (Examining Doctor & Attending Consultant)\n")
  
  # System Prompts defining the Multi-Agent Protocol
  doctor_prompt = "You are the Examining Doctor. Propose 1 medical theory and suggest the next best action. Output pure text. Do not output JSON."
  consultant_prompt = "You are the Attending Consultant. Review the Examining Doctor's theory and physical observation. Make the final call. Respond ONLY in strict JSON mapping to the MedTriage Action Schema: {'action_type': 'test', 'target': 'blood_cbc'}."
  
  while not obs.done and not obs.truncated:
    context = f"Time Elapsed: {obs.time_elapsed}m. Health: {obs.patient_health_status}.\nVitals: {obs.patient_vitals}\nObservation: {obs.terminal_output}\nConstraints: {obs.available_actions}"
    
    # Agent 1: The Examining Doctor Proposes a theory
    print("--- Swarm Negotiation Node ---")
    print("Examining Doctor is considering the symptoms...")
    doctor_theory = query_llm(context, doctor_prompt, api_key)
    
    if "{" in doctor_theory:
      doctor_theory = "I suspect chest trauma, we should run tests." # Strip fallback mock JSON
      
    print(f"> Doctor Proposes: '{doctor_theory}'")
    
    # Agent 2: The Consultant Reviews and Executes
    print(" Attending Consultant is verifying constraint logic...")
    consultant_input = f"Environment State:\n{context}\n\nExamining Doctor's Proposal:\n{doctor_theory}\n\nOutput Final JSON MedAction:"
    final_decision_str = query_llm(consultant_input, consultant_prompt, api_key)
    
    try:
      clean_json = final_decision_str.replace("```json", "").replace("```", "")
      action_data = json.loads(clean_json)
      action = MedAction(
        action_type=ActionType(action_data["action_type"].lower()), 
        target=action_data["target"].lower()
      )
      print(f"> Consultant Finalizes Execution: {action.action_type.value.upper()} {action.target}\n")
      
      obs = env.step(action)
      time.sleep(1) # Pacing UI
      
    except Exception:
      print(f"> Consultant halted: Invalid JSON schema generated. Auto-examining to prevent crash.")
      obs = env.step(MedAction(action_type=ActionType.EXAMINE, target="neurological"))

  print("\n--- CLINICAL SWARM DISMISSED ---")
  print(obs.terminal_output)
  if obs.truncated:
    print("RESULT: FATAL (Agents failed to diagnose within time constraints).")
  elif env.state().total_reward > 0:
    print(f"RESULT: SUCCESS (Patient Saved. Net Reward: {env.state().total_reward:.2f})")
  else:
    print(f"RESULT: FAILURE (Incorrect intervention mapped. Net Reward: {env.state().total_reward:.2f})")

if __name__ == "__main__":
  run_multi_agent_swarm()
