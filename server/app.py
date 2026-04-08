import uvicorn
from openenv.core.env_server import create_web_interface_app

from triage_env.models import MedAction, MedObservation
from triage_env.server.triage_environment import TriageEnvironment

env_class = TriageEnvironment

app = create_web_interface_app(env_class, MedAction, MedObservation)

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
