# 3
import math
from typing import Dict, List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODEL_NAME, NUM_CLASSES, T_SAMPLES, ENTROPY_THRESHOLD, LABEL_MAP, DEVICE

tokenizer = None
model = None

def load_uq_artifacts():
    global tokenizer, model
    print(f"Loading tokenizer and model ({MODEL_NAME}) onto device: {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES
    ).to(DEVICE)
    print("UQ Engine pipeline loaded successfully.")

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
    dialogue_lower = dialogue.lower()
    if "chest" in dialogue_lower or "breath" in dialogue_lower:
        return "Does the patient experience chest pain or pressure radiating to the arm, neck, or jaw?"
    elif "head" in dialogue_lower or "fall" in dialogue_lower:
        return "Did the patient lose consciousness or experience sudden numbness/weakness?"
    else:
        return f"Clarification required: Please specify the primary symptom onset and severity for class '{predicted_class}'."

def evaluate_dialogue_uncertainty(dialogue_history: str) -> dict:
    inputs = tokenizer(
        dialogue_history,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    ).to(DEVICE)

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
        hitl_prompt = generate_clarifying_prompt(dialogue_history, triage_label)
    else:
        epistemic_risk = "LOW_EPISTEMIC_RISK"
        action_taken = "AUTONOMOUS_DISPATCH_PERMITTED"
        hitl_prompt = None

    return {
        "triage_prediction": triage_label,
        "predicted_class_id": predicted_class_id,
        "mean_probabilities": prob_dict,
        "confidence": round(confidence, 4),
        "predictive_entropy": round(predictive_entropy, 4),
        "epistemic_risk": epistemic_risk,
        "action_taken": action_taken,
        "human_in_the_loop_prompt": hitl_prompt
    }
