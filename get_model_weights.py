import os
from pathlib import Path

import gdown

URL = "https://drive.google.com/uc?id=1mQGdOYGBBpcTWa32YK-NR4F2iW7AV6Xt"
root_dir = Path(__file__).absolute().resolve().parent
model_dir = root_dir / "models"
model_dir.mkdir(exist_ok=True, parents=True)

output_model = "models/final_model_weights.pth"

gdown.download(URL, output_model, quiet=False)