FROM python:3.10-slim

WORKDIR /app
ENV PORT=7860

COPY triage_env/server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire package since it relies on triage_env/__init__.py and modules
COPY triage_env /app/triage_env

CMD ["uvicorn", "triage_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
