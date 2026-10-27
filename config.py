# 2
import torch

MODEL_NAME: str = "distilbert-base-uncased"
NUM_CLASSES: int = 3
T_SAMPLES: int = 20
ENTROPY_THRESHOLD: float = 0.5
PORT: int = 8090

LABEL_MAP = {
    0: "Low (Category 3)",
    1: "Medium (Category 2)",
    2: "Critical (Category 1)"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
