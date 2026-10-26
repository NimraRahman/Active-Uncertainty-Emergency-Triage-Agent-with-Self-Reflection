!pip install -q fastapi "uvicorn[standard]" pydantic torch transformers nest_asyncio
---------
%%writefile app.py
import math
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME: str = "distilbert-base-uncased"
NUM_CLASSES: int = 3
T_SAMPLES: int = 20
ENTROPY_THRESHOLD: float = 0.5

LABEL_MAP: Dict[int, str] = {
    0: "Low (Category 3)",
    1: "Medium (Category 2)",
    2: "Critical (Category 1)"
}

app = FastAPI(
    title="QUEST: Quantifying Uncertainty in Emergency System Triage",
    description="REST Microservice for Epistemic Uncertainty Quantification (MC Dropout) & Safe-Fail Gating",
    version="1.0.0"
)

tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
def load_artifacts():
    global tokenizer, model
    print(f"Loading tokenizer and model ({MODEL_NAME}) onto device: {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES
    ).to(device)
    print("QUEST Pipeline loaded successfully.")

def enable_mc_dropout(model_instance: torch.nn.Module):
    model_instance.eval()
    for module in model_instance.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()

def calculate_predictive_entropy(mean_probs: torch.Tensor) -> float:
    entropy = 0.0
    for p in mean_probs:
        p_val = p.item()
        if p_val > 0.0:
            entropy -= p_val * math.log(p_val)
    return float(entropy)

def generate_clarifying_prompt(dialogue: str, predicted_class: str) -> str:
    if "chest" in dialogue.lower() or "breath" in dialogue.lower():
        return "Does the patient experience chest pain or pressure radiating to the arm, neck, or jaw?"
    elif "head" in dialogue.lower() or "fall" in dialogue.lower():
        return "Did the patient lose consciousness or experience sudden numbness/weakness?"
    else:
        return f"Clarification required: Please specify the primary symptom onset and severity for class '{predicted_class}'."

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
        "device": str(device)
    }

@app.post("/evaluate-triage", response_model=TriageResponse)
def evaluate_triage(payload: TriageRequest):
    if not payload.dialogue_history.strip():
        raise HTTPException(status_code=400, detail="Dialogue history cannot be empty.")

    inputs = tokenizer(
        payload.dialogue_history,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    ).to(device)

    enable_mc_dropout(model)

    monte_carlo_logits: List[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(T_SAMPLES):
            outputs = model(**inputs)
            monte_carlo_logits.append(outputs.logits)

    stacked_logits = torch.cat(monte_carlo_logits, dim=0)
    probs_distribution = F.softmax(stacked_logits, dim=-1)
    mean_probabilities = torch.mean(probs_distribution, dim=0)

    predicted_class_id = int(torch.argmax(mean_probabilities).item())
    triage_label = LABEL_MAP.get(predicted_class_id, "Unknown")
    confidence = float(mean_probabilities[predicted_class_id].item())
    predictive_entropy = calculate_predictive_entropy(mean_probabilities)

    prob_dict = {
        LABEL_MAP[i]: round(float(mean_probabilities[i].item()), 4)
        for i in range(NUM_CLASSES)
    }

    if predictive_entropy >= ENTROPY_THRESHOLD:
        epistemic_risk = "HIGH_EPISTEMIC_RISK"
        action_taken = "HALT_AUTONOMOUS_DISPATCH"
        hitl_prompt = generate_clarifying_prompt(payload.dialogue_history, triage_label)
    else:
        epistemic_risk = "LOW_EPISTEMIC_RISK"
        action_taken = "AUTONOMOUS_DISPATCH_PERMITTED"
        hitl_prompt = None

    return TriageResponse(
        triage_prediction=triage_label,
        predicted_class_id=predicted_class_id,
        mean_probabilities=prob_dict,
        confidence=round(confidence, 4),
        predictive_entropy=round(predictive_entropy, 4),
        epistemic_risk=epistemic_risk,
        action_taken=action_taken,
        human_in_the_loop_prompt=hitl_prompt
    )
------------
# Launch the local Server & Expose via Public Tunnel

import threading
import uvicorn
import nest_asyncio
import time

# Apply async loop patch for Colab notebooks
nest_asyncio.apply()

# Use a new port to avoid the "address already in use" conflict
PORT = 8090

def run_server():
    uvicorn.run("app:app", host="0.0.0.0", port=PORT)

# Start FastAPI in a background daemon thread
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

print(f"FastAPI server starting on port {PORT}. Waiting 10 seconds for model to load...")
time.sleep(10) # Gives PyTorch time to load DistilBERT into memory

# Install localtunnel and generate a public URL pointing to your new port
!npm install -g localtunnel &> /dev/null
print(f"\nYour new public tunnel URL for port {PORT} is:")
!npx localtunnel --port {PORT}
