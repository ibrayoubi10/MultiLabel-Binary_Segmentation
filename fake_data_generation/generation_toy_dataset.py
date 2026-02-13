import os
import random
from PIL import Image, ImageDraw

def make_toy_binary_seg_dataset(root="toy_binary_dataset", n=200, size=224, seed=0):
    random.seed(seed)

    img_dir = os.path.join(root, "images")
    msk_dir = os.path.join(root, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    for i in range(n):
        # fond noir
        img = Image.new("RGB", (size, size), (0, 0, 0))
        msk = Image.new("L", (size, size), 0)

        draw_img = ImageDraw.Draw(img)
        draw_msk = ImageDraw.Draw(msk)

        # cercle aléatoire
        r = random.randint(size // 10, size // 4)
        cx = random.randint(r, size - r)
        cy = random.randint(r, size - r)

        bbox = (cx - r, cy - r, cx + r, cy + r)

        # objet blanc dans l'image
        draw_img.ellipse(bbox, fill=(255, 255, 255))
        # mask binaire: 255 = foreground
        draw_msk.ellipse(bbox, fill=255)

        name = f"{i:04d}.png"
        img.save(os.path.join(img_dir, name))
        msk.save(os.path.join(msk_dir, name))

    print(f"Done: {root} (n={n})")

if __name__ == "__main__":
    make_toy_binary_seg_dataset()