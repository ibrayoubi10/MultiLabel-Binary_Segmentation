import os
import random
from PIL import Image, ImageDraw

def _draw_triangle(draw, bbox, fill):
    x0, y0, x1, y1 = bbox
    # triangle simple (haut-centre, bas-gauche, bas-droite)
    pts = [((x0 + x1) // 2, y0), (x0, y1), (x1, y1)]
    draw.polygon(pts, fill=fill)

def make_toy_multiclass_seg_dataset(
    root="toy_multiclass_dataset",
    n=300,
    size=224,
    seed=0,
    max_shapes=3
):
    """
    Multi-classe (single-label per pixel):
      0 background
      1 circle
      2 rectangle
      3 triangle
    Les objets peuvent se chevaucher: le dernier dessiné "gagne".
    """
    random.seed(seed)

    img_dir = os.path.join(root, "images")
    msk_dir = os.path.join(root, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    for i in range(n):
        img = Image.new("RGB", (size, size), (0, 0, 0))
        mask = Image.new("L", (size, size), 0)  # labels 0..3

        d_img = ImageDraw.Draw(img)
        d_msk = ImageDraw.Draw(mask)

        num_shapes = random.randint(1, max_shapes)

        for _ in range(num_shapes):
            shape_type = random.choice(["circle", "rect", "tri"])

            w = random.randint(size // 8, size // 3)
            h = random.randint(size // 8, size // 3)
            x0 = random.randint(0, size - w)
            y0 = random.randint(0, size - h)
            x1 = x0 + w
            y1 = y0 + h
            bbox = (x0, y0, x1, y1)

            if shape_type == "circle":
                cls = 1
                color = (255, 255, 255)     # blanc
                d_img.ellipse(bbox, fill=color)
                d_msk.ellipse(bbox, fill=cls)

            elif shape_type == "rect":
                cls = 2
                color = (255, 255, 255)     # blanc
                d_img.rectangle(bbox, fill=color)
                d_msk.rectangle(bbox, fill=cls)

            else:  # triangle
                cls = 3
                color = (255, 255, 255)     # blanc
                _draw_triangle(d_img, bbox, fill=color)
                _draw_triangle(d_msk, bbox, fill=cls)

        name = f"{i:04d}.png"
        img.save(os.path.join(img_dir, name))
        mask.save(os.path.join(msk_dir, name))

    print(f"Done: {root} (n={n}, classes=4 with background)")

if __name__ == "__main__":
    make_toy_multiclass_seg_dataset()