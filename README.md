# MedTriage-Env

A medical diagnostic simulator designed for post-training LLMs via Reinforcement Learning (RL), competing in the Meta PyTorch OpenEnv Hackathon. 

## Overview
The agent acts as an emergency room doctor. It is given a patient's initial symptoms. The agent interacts with the environment sequentially to gather information through tests and examinations to reach a correct diagnosis and final treatment. 

The environment assigns a continuous dense reward/penalty based on resource expenditure (ordering too many medical tests) versus diagnostic accuracy.

## Features
- **Deterministic Patient Pool:** Features 10 distinct, highly complex medical diagnostic paths (e.g. Appendicitis, Stroke, Gout, etc).
- **Fully Typed:** Pydantic schema validation through OpenEnv's base models.
- **Strict CI Hooks Conformant:** Uses `usort`, `ruff`, and `mypy` formatting.
- **HuggingFace Space Deployment Ready:** `Dockerfile` and `app.py` expose the environment across Meta's OpenEnv generic FastAPI. Validated via `EnvClient`.

## Installation & Testing

```bash
# Set up env
python3 -m venv venv
source venv/bin/activate

# Install with dependencies
pip install -e .[dev]

# Run tests
pytest tests/envs/test_triage_environment.py -v
```
