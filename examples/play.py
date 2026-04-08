import sys
import os

# Add parent directory to path to allow `python examples/play.py`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich import box
from rich.text import Text

from triage_env.server.triage_environment import TriageEnvironment
from triage_env.models import MedAction, ActionType

console = Console()

def print_header():
    console.clear()
    title = Text("MedTriage-Env Interactive RL Demo", justify="center", style="bold cyan")
    console.print(Panel(title, box=box.DOUBLE, expand=False, border_style="cyan"))
    console.print("[italic]You are the RL Agent. Survive. Diagnose. Treat.[/italic]\n", justify="center")

def render_state(obs, state):
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Time Elapsed (mins)")
    table.add_column("Patient Health")
    table.add_column("Step Count")
    table.add_column("Total Reward")
    
    health_style = "bold green" if obs.patient_health_status > 0.5 else "bold red blink"
    table.add_row(
        str(obs.time_elapsed),
        f"[{health_style}]{obs.patient_health_status * 100:.0f}%[/{health_style}]",
        str(state.step_count),
        f"{state.total_reward:.2f}"
    )
    console.print(table)
    
    # Terminal Output
    console.print(Panel(f"[yellow]{obs.terminal_output}[/yellow]", title="Observation", border_style="yellow"))

def main():
    print_header()
    difficulty = Prompt.ask("Select Difficulty", choices=["easy", "hard", "expert"], default="easy")
    
    env = TriageEnvironment(difficulty=difficulty)
    obs = env.reset()
    
    while not obs.done and not obs.truncated:
        console.rule("[bold cyan]Agent Input")
        state = env.state()
        render_state(obs, state)
        
        console.print("\n[bold]Available Action Types:[/bold] [cyan]INTERVIEW, EXAMINE, TEST, CONSULT, DIAGNOSE, TREAT[/cyan]")
        raw_action = Prompt.ask("\n[bold white]Enter Action (Format: TYPE target)[/bold white]")
        
        try:
            parts = raw_action.split(" ", 1)
            action_type_str = parts[0].strip().upper()
            target = parts[1].strip() if len(parts) > 1 else ""
            
            action_type = ActionType(action_type_str.lower())
            
            action = MedAction(action_type=action_type, target=target)
            obs = env.step(action)
        except ValueError:
            console.print("[bold red]Invalid Action Type or Format. Please use standard schema. e.g. 'TEST blood_cbc'[/bold red]")
            continue

    print_header()
    console.rule("[bold red]EPISODE TERMINATED")
    render_state(obs, env.state())
    
    if obs.truncated:
        console.print(f"\n[bold red blink]CRASH: Episode Forcibly Truncated! Final Reward: {env.state().total_reward:.2f}[/bold red blink]")
    else:
        console.print(f"\n[bold green]EPISODE COMPLETE! Final Reward: {env.state().total_reward:.2f}[/bold green]")
        
    console.print("\n[italic dim]This simulates exactly what the RL network 'sees' during training epochs.[/italic dim]")

if __name__ == "__main__":
    main()
