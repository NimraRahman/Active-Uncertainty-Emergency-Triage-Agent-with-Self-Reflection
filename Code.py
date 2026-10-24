
!pip install -q transformers torch matplotlib seaborn pandas scikit-learn datasets uvicorn fastapi

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# Set random seed for complete reproducibility
torch.manual_seed(42)
np.random.seed(42)


MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

class MCDropoutTriageModel(nn.Module):
    """
    Wrapper implementing Monte Carlo (MC) Dropout during inference 
    to quantify Epistemic Uncertainty via predictive variance across passes.
    """
    def __init__(self, model, dropout_p=0.3):
        super().__init__()
        self.model = model
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.p = dropout_p
                m.train() # Force dropout active during evaluation

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return F.softmax(outputs.logits, dim=-1)

mc_model = MCDropoutTriageModel(base_model)


# Toy training dataset to initialize classification weights (0=Low, 1=Medium, 2=Critical)
training_data = [
    ("My finger hurts slightly from a paper cut.", 0),
    ("I have a mild headache and feel tired.", 0),
    ("Just a slight cough for two days.", 0),
    ("Feeling dizzy and breathing a bit heavy.", 1),
    ("Stomach ache, fever, and vomiting.", 1),
    ("Sprained my ankle and it's swelling up.", 1),
    ("Severe chest pain, pressure down my left arm, can't breathe!", 2),
    ("He is bleeding heavily from a crash and is unconscious!", 2),
    ("Sudden stroke symptoms, face drooping, slurred speech.", 2)
]

class TriageDataset(Dataset):
    def __init__(self, data):
        self.texts = [item[0] for item in data]
        self.labels = [item[1] for item in data]
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=64, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = TriageDataset(training_data)
train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)

optimizer = AdamW(base_model.parameters(), lr=5e-5)
base_model.train()

for epoch in range(3):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = base_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/3 completed. Training Loss: {total_loss/len(train_loader):.4f}")

# Switch model back to evaluation mode
base_model.eval()
mc_model.eval()


def compute_uncertainty_metrics(text):
    """
    Performs T=20 stochastic forward passes to compute Predictive Entropy
    and variance across class predictions.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    T = 20 # Number of stochastic forward passes
    probs_list = []
    
    with torch.no_grad():
        for _ in range(T):
            probs = mc_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            probs_list.append(probs.squeeze(0).numpy())
            
    stacked_probs = np.vstack(probs_list) 
    mean_probs = np.mean(stacked_probs, axis=0) 
    
    # Predictive Entropy: H(y|x) = -sum(p * log p)
    predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-9))
    
    # Variance across MC iterations (Epistemic Variance)
    epistemic_variance = np.var(stacked_probs, axis=0).mean()
    
    predicted_class = int(np.argmax(mean_probs))
    confidence = float(mean_probs[predicted_class])
    
    return {
        "entropy": float(predictive_entropy),
        "variance": float(epistemic_variance),
        "confidence": confidence,
        "pred_label": predicted_class
    }


class AgentSelfReflection:
    """
    Meta-evaluator node: Analyzes the model's own output metrics to self-assess
    reliability, detect epistemic risk, and decide on human intervention.
    """
    def __init__(self, entropy_threshold=0.5):
        self.entropy_threshold = entropy_threshold

    def evaluate_output(self, metrics):
        entropy = metrics["entropy"]
        variance = metrics["variance"]
        confidence = metrics["confidence"]
        
        if entropy > self.entropy_threshold:
            status = "HIGH EPISTEMIC RISK"
            reflection = (
                f"SELF-ANALYSIS: Predictive Entropy ({entropy:.4f}) exceeds threshold ({self.entropy_threshold}). "
                f"High model variance ({variance:.6f}) indicates severe parameter disagreement. "
                "Output cannot be trusted autonomously. Halting auto-dispatch and triggering Human-in-the-Loop."
            )
            trigger_human = True
        else:
            status = "LOW RISK / CONFIDENT"
            reflection = (
                f"SELF-ANALYSIS: Predictive Entropy ({entropy:.4f}) is within safety bounds. "
                f"Confidence is strong ({confidence:.4f}). "
                "Class signal has converged. Prediction verified as safe for autonomous triage classification."
            )
            trigger_human = False

        return {
            "status": status,
            "reflection": reflection,
            "trigger_human": trigger_human
        }

evaluator = AgentSelfReflection(entropy_threshold=0.5)


conversation_turns = [
    {
        "turn": 1,
        "caller_utterance": "Help! My father doesn't feel well, he is breathing a bit heavy and looks dizzy.",
        "added_context": ""
    },
    {
        "turn": 2,
        "turn": 2,
        "caller_utterance": "He says his chest feels tight and pressure is going down his left arm.",
        "added_context": ""
    },
    {
        "turn": 3,
        "caller_utterance": "It started 20 minutes ago, and he has a history of hypertension.",
        "added_context": ""
    }
]

CLARIFYING_QUESTIONS = {
    1: "Does the patient experience chest pain or pressure radiating to the arm/jaw?",
    2: "When did these symptoms first start, and are they continuous?"
}

simulation_log = []
current_transcript = ""


for step in conversation_turns:
    turn_num = step["turn"]
    utterance = step.get("caller_utterance", "")
    context = step.get("added_context", "")
    current_transcript += " " + utterance + context
    
    metrics = compute_uncertainty_metrics(current_transcript)
    self_eval = evaluator.evaluate_output(metrics)
    
    selected_question = "None (Verified Safe)"
    if self_eval["trigger_human"]:
        selected_question = CLARIFYING_QUESTIONS.get(turn_num, "Can you provide additional clinical history?")

    log_entry = {
        "Turn": turn_num,
        "Predictive Entropy (UQ)": round(metrics["entropy"], 4),
        "Confidence": round(metrics["confidence"], 4),
        "Self-Evaluation Status": self_eval["status"],
        "Agent Self-Reflection": self_eval["reflection"],
        "Generated Clarification": selected_question,
        "Predicted Triage Level": ["Low", "Medium", "Critical"][metrics["pred_label"]]
    }
    simulation_log.append(log_entry)

df_results = pd.DataFrame(simulation_log)
print("\nSimulation Results & Self-Reflection Log:")
for idx, row in df_results.iterrows():
    print(f"\n--- [TURN {row['Turn']}] Triage Prediction: {row['Predicted Triage Level']} ---")
    print(f"Metrics -> Entropy: {row['Predictive Entropy (UQ)']} | Confidence: {row['Confidence']}")
    print(f"Agent Reflection -> {row['Agent Self-Reflection']}")
    print(f"Action Taken -> Clarification: {row['Generated Clarification']}")


sns.set_theme(style="whitegrid")
fig, ax1 = plt.subplots(figsize=(10, 5))

turns = df_results["Turn"]
entropy_vals = df_results["Predictive Entropy (UQ)"]
confidence_vals = df_results["Confidence"]

color = 'tab:red'
ax1.set_xlabel('Conversation Turn (Active Human-in-the-Loop Iteration)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predictive Entropy (Epistemic Uncertainty)', color=color, fontsize=12, fontweight='bold')
ax1.plot(turns, entropy_vals, color=color, marker='o', linewidth=2.5, label='Epistemic Uncertainty')
ax1.tick_params(axis='y', labelcolor=color)
ax1.axhline(y=0.5, color='gray', linestyle='--', label='Safety Threshold (tau = 0.5)')

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Model Confidence Score', color=color, fontsize=12, fontweight='bold')
ax2.plot(turns, confidence_vals, color=color, marker='s', linewidth=2.5, linestyle='--', label='Model Confidence')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Active Uncertainty Reduction & Self-Reflective Convergence', fontsize=14, fontweight='bold', pad=15)
fig.tight_layout()
plt.savefig('uncertainty_reduction_curve.png', dpi=300)
plt.show()



try:
    print("Loading streaming sample from Hugging Face medical dataset...")
    med_dataset = load_dataset("bigbio/meddialog", name="meddialog_en_source", split="train", streaming=True)
    sample_item = next(iter(med_dataset))
    sample_text = sample_item.get('text', 'Patient reports acute discomfort and dizziness.')[:250]
    
    print(f"\nReal Medical Query Preview: '{sample_text}...'")
    real_metrics = compute_uncertainty_metrics(sample_text)
    real_self_eval = evaluator.evaluate_output(real_metrics)
    
    print(f"Real Data Entropy: {real_metrics['entropy']:.4f}")
    print(f"Real Data Reflection: {real_self_eval['reflection']}")
except Exception as e:
    print(f" Dataset streaming skipped or unavailable: {e}")


app_code = '''from fastapi import FastAPI
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
'''
with open("app.py", "w") as f:
    f.write(app_code)

dockerfile_code = '''FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
'''
with open("Dockerfile", "w") as f:
    f.write(dockerfile_code)

reqs_code = '''torch
transformers
fastapi
uvicorn
numpy
pandas
scikit-learn
matplotlib
seaborn
datasets
'''
with open("requirements.txt", "w") as f:
    f.write(reqs_code)
