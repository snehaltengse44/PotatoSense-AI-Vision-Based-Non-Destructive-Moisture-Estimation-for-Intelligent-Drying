import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50

from preprocessing import preprocess_image_array


"""BASE_DIR = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "final_model" / "resnet50_svr_linear"
WEIGHTS_PATH = BASE_DIR / "final_model" / "resnet50_weights.pth"
REFERENCE_IMAGE_PATH = BASE_DIR / "final_model" / "reference_image.jpg"
"""

BASE_DIR = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "tuned_model" / "resnet50_svr_linear_tuned"
WEIGHTS_PATH = BASE_DIR / "tuned_model" / "resnet50_weights.pth"
REFERENCE_IMAGE_PATH = BASE_DIR / "tuned_model" / "reference_image.jpg"


class MoisturePredictor:
    def __init__(self, model_dir: Path = MODEL_DIR, weights_path: Path = WEIGHTS_PATH,
                 reference_image_path: Path = REFERENCE_IMAGE_PATH):
        model_path = model_dir / "model.joblib"
        metadata_path = model_dir / "metadata.json"

        for p, label in [(model_path, "model.joblib"), (metadata_path, "metadata.json"),
                          (weights_path, "resnet50_weights.pth")]:
            if not p.exists():
                raise FileNotFoundError(f"{label} not found at {p}")

        with open(metadata_path, "r") as f:
            self.meta = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.regressor = joblib.load(model_path)
        self.feature_extractor = self._build_resnet50(weights_path)

        self.target_size = self.meta.get("target_size", 224)
        self.mean_bgr = np.array(self.meta.get("mean_bgr", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std_bgr = np.array(self.meta.get("std_bgr", [0.229, 0.224, 0.225]), dtype=np.float32)
        self.use_clahe = bool(self.meta.get("use_clahe", False))

        # Reference image for histogram matching -- must match training exactly.
        if reference_image_path.exists():
            self.reference_img = cv2.imread(str(reference_image_path))
        else:
            self.reference_img = None
            print(f"[warn] No reference image found at {reference_image_path} — "
                  f"histogram matching will be SKIPPED. This creates a train/serve "
                  f"mismatch if the training pipeline used it.")

    def _build_resnet50(self, weights_path: Path):
        model = resnet50(weights=None)
        model.fc = nn.Identity()  # matches training script exactly

        state_dict = torch.load(weights_path, map_location=self.device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = { k.replace("module.", ""): v for k, v in state_dict.items() }

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[warn] load_state_dict mismatches — missing: {missing}, unexpected: {unexpected}")

        model.eval().to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        return model

    def _decode(self, image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image")
        return img_bgr

    def _extract_features(self, processed_bgr: np.ndarray) -> np.ndarray:
        """Mirrors training's extract_features(): convert to RGB, ToTensor,
        normalize with the BGR-order mean/std reversed to match RGB order."""
        rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        tensor = T.ToTensor()(rgb)  # CHW, scaled to [0,1]

        mean_rgb = torch.tensor(self.mean_bgr[::-1].copy()).view(3, 1, 1)
        std_rgb = torch.tensor(self.std_bgr[::-1].copy()).view(3, 1, 1)
        tensor = (tensor - mean_rgb) / (std_rgb + 1e-6)

        tensor = tensor.unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            feats = self.feature_extractor(tensor)  # (1, 2048)
        return feats.cpu().numpy()

    def extract_features_raw(self, image_bytes: bytes) -> np.ndarray:
        """Returns the raw (2048,) ResNet50 feature vector. Used for both
        the potato gate and the moisture prediction, so the CNN forward
        pass only runs once per upload."""
        raw_bgr = self._decode(image_bytes)
        processed = preprocess_image_array(
            raw_bgr,
            reference_img=self.reference_img,
            use_clahe=self.use_clahe,
            target_size=self.target_size,
        )
        feats = self._extract_features(processed)  # (1, 2048)
        return feats[0]

    def predict_from_features(self, feats: np.ndarray) -> float:
        pred = self.regressor.predict(feats.reshape(1, -1))[0]
        return round(float(pred), 2)
    
    def predict(self, image_bytes: bytes) -> float:
        feats = self.extract_features_raw(image_bytes)
        return self.predict_from_features(feats)