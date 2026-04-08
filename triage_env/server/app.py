from openenv.core.env_server import create_web_interface_app

from triage_env.models import MedAction, MedObservation
from triage_env.server.triage_environment import TriageEnvironment

env_class = TriageEnvironment

app = create_web_interface_app(env_class, MedAction, MedObservation)
