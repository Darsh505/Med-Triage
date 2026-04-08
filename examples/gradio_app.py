import sys
import os
import gradio as gr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triage_env.server.triage_environment import TriageEnvironment
from triage_env.models import MedAction, ActionType

# Global state hold for Gradio Session
class EnvSession:
  def __init__(self):
    self.env = TriageEnvironment(difficulty="infinite")
    self.obs = self.env.reset()
    self.chat_history = []

session = EnvSession()

def reset_env():
  global session
  session = EnvSession()
  initial_msg = f"**Patient Arrives:**\n{session.obs.terminal_output}\n\n**Vitals:**\n{session.obs.patient_vitals}\n\n*You are the Doctor. What is your first action?*"
  return [(None, initial_msg)], "Time Elapsed: 0 mins", "Health: 100%", "Reward: 0.0"

def parse_input_to_action(user_text):
  parts = user_text.strip().split(" ", 1)
  if len(parts) == 0:
    raise ValueError("Empty")
  a_type = parts[0].strip().lower()
  target = parts[1].strip().lower() if len(parts) > 1 else ""
  return MedAction(action_type=ActionType(a_type), target=target)

def step_env(user_message, history):
  global session
  
  if session.obs.done or session.obs.truncated:
    return history + [(user_message, "Episode is medically finished. Please click 'Reset Patient'.")], get_stats()[0], get_stats()[1], get_stats()[2]
    
  try:
    action = parse_input_to_action(user_message)
    session.obs = session.env.step(action)
    
    reply = session.obs.terminal_output
    if session.obs.patient_vitals:
      reply += f"\n\n**Vitals Trace:** {session.obs.patient_vitals}"
      
    history.append((user_message, reply))
    
  except ValueError:
    history.append((user_message, " Invalid Command Format. Please use: `interview <symptom>`, `examine <body_part>`, `test <blood_cbc>`, or `treat <medicine>`"))
    
  return history, f"Time Elapsed: {session.obs.time_elapsed} mins", f"Health: {session.obs.patient_health_status * 100:.0f}%", f"Reward: {session.env.state().total_reward:.2f}"

def get_stats():
  global session
  return f"Time Elapsed: {session.obs.time_elapsed} mins", f"Health: {session.obs.patient_health_status * 100:.0f}%", f"Reward: {session.env.state().total_reward:.2f}"

with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal"), title="MedTriage-Env Simulator") as demo:
  gr.Markdown("# MedTriage-Env: Interactive Simulator\nTest the OpenEnv Reinforcement Learning mechanics natively.")
  
  with gr.Row():
    with gr.Column(scale=3):
      chatbot = gr.Chatbot(height=500, label="Clinical Output")
      msg = gr.Textbox(placeholder="e.g. 'examine chest' or 'test blood_cbc'", label="Command Input (type 'action target')")
      
      with gr.Row():
        submit_btn = gr.Button("Execute Action", variant="primary")
        reset_btn = gr.Button("Reset Patient (Infinite Horizon)")
        
    with gr.Column(scale=1):
      gr.Markdown("### Live Telemetry")
      time_txt = gr.Textbox(label="Time Elapsed", value="0 mins", interactive=False)
      health_txt = gr.Textbox(label="Patient Health", value="100%", interactive=False)
      reward_txt = gr.Textbox(label="Total Reward Matrix", value="0.0", interactive=False)
      
      gr.Markdown("### Action Schema\n- `interview <question>`\n- `examine <body_part>`\n- `test <test_name>`\n- `diagnose <disease>`\n- `treat <medicine>`")

  # Wire up initial load
  demo.load(reset_env, inputs=None, outputs=[chatbot, time_txt, health_txt, reward_txt])
  
  msg.submit(step_env, inputs=[msg, chatbot], outputs=[chatbot, time_txt, health_txt, reward_txt])
  submit_btn.click(step_env, inputs=[msg, chatbot], outputs=[chatbot, time_txt, health_txt, reward_txt])
  reset_btn.click(reset_env, inputs=None, outputs=[chatbot, time_txt, health_txt, reward_txt])

if __name__ == "__main__":
  demo.launch()
