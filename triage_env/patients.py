import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class PatientScenario:
    id: str
    initial_symptoms: str
    vitals: Dict[str, str]
    exam_findings: Dict[str, str]
    test_results: Dict[str, str]
    correct_diagnosis: str
    correct_treatment: str
    difficulty: str = "easy"
    interview_responses: Dict[str, str] = field(default_factory=dict)


# Compute absolute path to data file
DATA_DIR = Path(__file__).parent / "data"
PATIENT_JSON_PATH = DATA_DIR / "patients.json"


def _load_patients() -> list[PatientScenario]:
    if not PATIENT_JSON_PATH.exists():
        return []
    with open(PATIENT_JSON_PATH, "r") as f:
        data = json.load(f)
    return [PatientScenario(**p) for p in data]


PATIENT_POOL = _load_patients()
