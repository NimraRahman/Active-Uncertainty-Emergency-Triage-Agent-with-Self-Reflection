import matplotlib.pyplot as plt
from config import ENTROPY_THRESHOLD
from uq_engine import load_uq_artifacts, evaluate_dialogue_uncertainty

def run_simulation():
    print("Initializing UQ Artifacts for Simulation...")
    load_uq_artifacts()

    dialogue_turns = [
        "Caller: My head hurts a bit.",
        "Caller: My head hurts a bit and she fell down the kitchen stairs.",
        "Caller: She fell down the kitchen stairs, hit her head hard, and was unresponsive for a few minutes."
    ]

    entropies = []
    turn_labels = []

    for idx, dialogue in enumerate(dialogue_turns, 1):
        res = evaluate_dialogue_uncertainty(dialogue)
        entropy = res["predictive_entropy"]
        entropies.append(entropy)
        turn_labels.append(f"Turn {idx}")
        print(f"[Turn {idx}] Entropy: {entropy:.4f} | Action: {res['action_taken']}")

    plt.figure(figsize=(8, 5))
    plt.plot(turn_labels, entropies, marker='o', color='crimson', linewidth=2, label='Predictive Entropy (H)')
    plt.axhline(y=ENTROPY_THRESHOLD, color='black', linestyle='--', label=f'Safety Threshold (τ={ENTROPY_THRESHOLD})')
    plt.title("Epistemic Uncertainty Reduction Across Active Triage Turns")
    plt.xlabel("Dialogue Turn / Information Gain")
    plt.ylabel("Predictive Entropy")
    plt.ylim(0, max(1.2, max(entropies) + 0.1))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("uncertainty_reduction_curve.png", dpi=300)
    plt.close()
    print("[INFO] Simulation complete. Chart saved as 'uncertainty_reduction_curve.png'.")

if __name__ == "__main__":
    run_simulation()
