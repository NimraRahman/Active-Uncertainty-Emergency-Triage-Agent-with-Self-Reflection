from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Self-Reflective Active Triage UQ API")

class TriageRequest(BaseModel):
    transcript: str

@app.post("/evaluate-triage")
def evaluate_triage(req: TriageRequest):
    return {
        "status": "Active UQ Service with Self-Reflection Running",
        "transcript": req.transcript
    }
