import os
import glob
from PIL import Image, ImageFilter

# ----------------------------
# USER PATHS
# ----------------------------
HR_DIR = r"E:\ML\Testing\test_2x_HR"# your existing 100 HR images
LR_DIR = r"E:\ML\Testing\test_2x_LR"      # output LR folder

SCALE = 2
BLUR_RADIUS = 1.5

# ----------------------------
# Create LR folder
# ----------------------------
os.makedirs(LR_DIR, exist_ok=True)

# ----------------------------
# Collect HR images
# ----------------------------
hr_images = []
for ext in ("*.png", "*.jpg", "*.jpeg"):
    hr_images.extend(glob.glob(os.path.join(HR_DIR, ext)))

print(f"Found {len(hr_images)} HR images.")

# ----------------------------
# MAIN PROCESS
# ----------------------------
count = 0
for hr_path in hr_images:
    try:
        hr = Image.open(hr_path).convert("RGB")
        filename = os.path.basename(hr_path)

        # Blur HR
        blurred = hr.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))

        # Downscale by 2× using bicubic
        w, h = hr.size
        lr = blurred.resize((w // SCALE, h // SCALE), Image.BICUBIC)

        # Save LR with same filename
        lr.save(os.path.join(LR_DIR, filename))

        count += 1

    except Exception as e:
        print(f"Error processing {hr_path}: {e}")

print(f"\nGenerated {count} LR images.")
print("LR images saved to:", LR_DIR)
