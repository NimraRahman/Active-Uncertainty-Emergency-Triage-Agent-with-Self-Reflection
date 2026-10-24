# Active Uncertainty Emergency Triage Agent with Self-Reflective Gating

An advanced, practice-oriented emergency triage prototype designed to address epistemic uncertainty and fragmented human-machine coordination in emergency dispatch workflows.

## Architecture Overview
* **Epistemic Uncertainty Quantification (UQ):** Implements Monte Carlo Dropout ($T=20$ stochastic forward passes) in PyTorch to compute Predictive Entropy $H(y|x)$ and variance across class predictions.
* **Self-Reflection Node:** A meta-analytic evaluation layer that checks model reliability against a safety threshold ($\tau = 0.5$), halting autonomous dispatch and triggering Human-in-the-Loop (HITL) clarifying questions when epistemic risk is high.
* **Deployment Stack:** FastAPI backend containerized via Docker and hosted via cloud deployment.

## Repository Structure
* `app.py`: FastAPI production backend service.
* `Dockerfile`: Container build configuration for cloud hosting.
* `requirements.txt`: Python package dependencies.
* `uncertainty_reduction_curve.png`: Visualization of uncertainty convergence across dialogue turns.
