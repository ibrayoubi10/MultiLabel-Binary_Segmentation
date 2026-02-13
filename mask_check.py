import numpy as np
from PIL import Image
import os

mask_dir = "toy_multiclass_dataset/masks"
sample = sorted(os.listdir(mask_dir))[0]

mask = np.array(Image.open(os.path.join(mask_dir, sample)).convert("L"))
print("Unique values in mask:", np.unique(mask))