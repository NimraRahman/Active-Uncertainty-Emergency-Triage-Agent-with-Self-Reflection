# 4
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional
from config import ENTROPY_THRESHOLD, T_SAMPLES, DEVICE
from uq_engine import load_uq_artifacts, evaluate_dialogue_uncertainty

app = FastAPI(
    title="QUEST: Quantifying Uncertainty in Emergency System Triage",
    description="REST Microservice for Epistemic Uncertainty Quantification (MC Dropout) & Safe-Fail Gating",
    version="1.1.0"
)

@app.on_event("startup")
def startup_event():
    load_uq_artifacts()

class TriageRequest(BaseModel):
    dialogue_history: str = Field(..., description="Multi-turn emergency dialogue transcript.")

class TriageResponse(BaseModel):
    triage_prediction: str
    predicted_class_id: int
    mean_probabilities: Dict[str, float]
    confidence: float
    predictive_entropy: float
    epistemic_risk: str
    action_taken: str
    human_in_the_loop_prompt: Optional[str] = None

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "QUEST Triage Epistemic UQ Engine",
        "mc_dropout_samples": T_SAMPLES,
        "entropy_threshold": ENTROPY_THRESHOLD,
        "device": str(DEVICE)
    }

@app.post("/evaluate-triage", response_model=TriageResponse)
def evaluate_triage(payload: TriageRequest):
    if not payload.dialogue_history.strip():
        raise HTTPException(status_code=400, detail="Dialogue history cannot be empty.")
    
    result = evaluate_dialogue_uncertainty(payload.dialogue_history)
    return TriageResponse(**result)
