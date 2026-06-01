"""Shared utilities: paths, MLflow setup, dataset, transforms."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent          # sketch_clf/
DATA = ROOT / "data"
MLDB = ROOT / "mlflow.db"                               # sqlite tracking backend (in repo)
MLARTIFACTS = ROOT / "mlartifacts"                      # artifact store (in repo)
TRACKING_URI = f"sqlite:///{MLDB}"


def setup_mlflow(experiment="sketch-classifier"):
    import mlflow
    MLARTIFACTS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=MLARTIFACTS.as_uri())
    mlflow.set_experiment(experiment)
    return mlflow


def load_meta():
    return json.loads((DATA / "meta.json").read_text())


def load_split(split):
    X = np.load(DATA / f"{split}_images.npy")           # (N,224,224) uint8
    y = np.load(DATA / f"{split}_labels.npy")           # (N,) int64
    return X, y


class SketchDataset(Dataset):
    """Grayscale uint8 sketches -> 3-channel normalized tensors."""

    def __init__(self, X, y, mean, std, train=False, aug=None):
        self.X = X
        self.y = y
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.train = train
        self.aug = aug

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.X[i]                                  # (224,224) uint8
        from PIL import Image
        pil = Image.fromarray(img).convert("RGB")
        if self.train and self.aug is not None:
            pil = self.aug(pil)
        t = torch.from_numpy(np.array(pil, dtype=np.float32) / 255.0).permute(2, 0, 1)
        t = (t - self.mean) / self.std
        return t, int(self.y[i])
