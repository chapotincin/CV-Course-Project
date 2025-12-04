import os
from pathlib import Path
import shutil
import pandas as pd

# ==== EDIT THESE ====
excel_path = "arms.xlsx"      # your Excel file
image_id_col = "image_id"       # column name with image IDs
label_col = "arms"             # column name with 'e' or 's'
src_image_dir = Path(r"C:\Users\udayk\OneDrive\Desktop\Computer_Vision\Project\images_gz2\images_filtered\images_filtered")   # folder where all images currently are

out_root = Path("sorted_arms")     # root folder for sorted images
out_1_dir = out_root / "1"
out_2_dir = out_root / "2"
out_3_dir = out_root / "3"
out_4_dir = out_root / "4"

# Whether to copy or move images:
USE_COPY = True   # True = copy, False = move
# =====================

# Create output folders
out_1_dir.mkdir(parents=True, exist_ok=True)
out_2_dir.mkdir(parents=True, exist_ok=True)
out_3_dir.mkdir(parents=True, exist_ok=True)
out_4_dir.mkdir(parents=True, exist_ok=True)
# Read Excel
df = pd.read_excel(excel_path)

# Helper: find image file by ID (with or without extension)
def find_image_file(img_id):
    """
    img_id can be '123.jpg' or just '123'.
    We try to resolve to an existing file in src_image_dir.
    """
    img_id_str = str(img_id)

    # If Excel already has extension (e.g., '123.jpg')
    direct_path = src_image_dir / img_id_str
    if direct_path.exists():
        return direct_path

    # If Excel only has bare ID (e.g., '123'), try common extensions
    base = img_id_str
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        candidate = src_image_dir / f"{base}{ext}"
        if candidate.exists():
            return candidate

    return None

num_moved = 0
missing = []

for _, row in df.iterrows():
    img_id = row[image_id_col]
    label = str(row[label_col]).strip().lower()  # 'e' or 's'

    src_path = find_image_file(img_id)
    if src_path is None:
        missing.append(img_id)
        continue

    if label == "1":
        dst_dir = out_1_dir
    elif label == "2":
        dst_dir = out_2_dir
    elif label == "3":
        dst_dir = out_3_dir
    elif label == "4":
        dst_dir = out_4_dir
    else:
        print(f"⚠️ Unknown label '{label}' for image_id={img_id}, skipping.")
        continue

    dst_path = dst_dir / src_path.name

    if USE_COPY:
        shutil.copy2(src_path, dst_path)
    else:
        shutil.move(src_path, dst_path)

    num_moved += 1

print(f"Done. {'Copied' if USE_COPY else 'Moved'} {num_moved} images.")
if missing:
    print("Could not find files for these image IDs:")
    for m in missing:
        print("  ", m)
