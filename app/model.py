"""Model 层：预处理与推理，不关心 HTTP 和展示。"""

from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn

from app.config import BOX, INPUT_RESOLUTION, THRESHOLD


def _binarize(value: int) -> int:
    return 255 if value > THRESHOLD else 0

def to_tensor(image_bytes: bytes) -> torch.Tensor | None:

    # read image bytes and convert it to grayscale
    try:
        image = Image.open(BytesIO(image_bytes)).convert("L")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("invalid image") from exc

    # find pixel points that are above (brighter than) THRESHOLD (check in _binarize)
    mask = image.point(_binarize) 
    box = mask.getbbox() # minimum bounding rectangle (to involve whole mask)
    if box is None:
        return None

    # crop the area of box out
    digit = image.crop(box)
    scaling_factor = BOX / max(digit.size)
    scaled_width = max(1, round(digit.width * scaling_factor))
    scaled_height = max(1, round(digit.height * scaling_factor))
    # with resampling filter：LanczosWindow (algorithm)
    digit = digit.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS) 

    # paste cropped digit (rectangle) on a new canvas (square)
    canvas = Image.new("L", INPUT_RESOLUTION, 0)
    canvas.paste(digit, ((INPUT_RESOLUTION[0] - scaled_width) // 2, (INPUT_RESOLUTION[1] - scaled_height) // 2)) # center alignment

    # normalize the data range of pixels
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(array).reshape(1, 1, *INPUT_RESOLUTION)

class RecognizerModel:

    def __init__(
        self, 
        dl_model : nn.Module, 
        device: str | torch.device | None = None,
    ):
        
        self.dl_model = dl_model

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def load_checkpoint(
        self,
        checkpoint_path : Path
    ):
        self.dl_model.to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.dl_model.load_state_dict(checkpoint["model"])

        self.dl_model.eval()

       
    def recognize(self, image: bytes) -> tuple[int, list[float]] | None:
        
        tensor = to_tensor(image)
        if tensor is None:
            return None

        tensor = tensor.to(self.device)

        with torch.no_grad():

            logits = self.dl_model(tensor)
            # softmax on the dim 1 of logits in shape of (Batches, Catogaries) 
            # and get the result of sample 0 (the only sample of this batch)
            probabilities = logits.softmax(dim=1)[0] 

        return int(probabilities.argmax()), probabilities.tolist() # the number, probabilities

