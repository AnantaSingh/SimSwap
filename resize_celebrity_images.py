import os
from PIL import Image

root_dir = "crop_224/celebrities"

for celeb in os.listdir(root_dir):
    celeb_dir = os.path.join(root_dir, celeb)
    if not os.path.isdir(celeb_dir):
        continue
    for fname in os.listdir(celeb_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            fpath = os.path.join(celeb_dir, fname)
            try:
                img = Image.open(fpath)
                img = img.resize((224, 224), Image.LANCZOS)
                img.save(fpath)
                print(f"Resized: {fpath}")
            except Exception as e:
                print(f"Failed to process {fpath}: {e}") 