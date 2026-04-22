# --- Imports ---
import glob
from PIL import Image, ImageFilter
import os
from shutil import copyfile, copytree
import zipfile

# --- CONFIGURE YOUR LOCAL PATHS ---
zip_path = r"E:\ML\Testing\Testing_data\Test_HR\P1310.zip"
extract_path = r"E:\ML\Testing\Testing_data\Test_LR\test_2x"

# --- Ensure output folders exist ---
os.makedirs(extract_path, exist_ok=True)

# --- EXTRACT IMAGES ---
extracted_count = 0
print(f"Extracting images from {zip_path} ...")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    for member in zip_ref.infolist():
        if member.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_name = os.path.basename(member.filename)
            if not img_name:
                continue
            target_path = os.path.join(extract_path, img_name)

            # Avoid overwriting duplicates
            if os.path.exists(target_path):
                base, ext = os.path.splitext(img_name)
                target_path = os.path.join(extract_path, f"{base}_{extracted_count}{ext}")

            with zip_ref.open(member) as src, open(target_path, 'wb') as out:
                out.write(src.read())
            extracted_count += 1

print(f"Total images extracted: {extracted_count}")

# --- MAIN PIPELINE ---

# Step 1: Setup stage folders
stage1_dir = os.path.join(os.path.dirname(extract_path), 'images_stage1')
stage2_dir = os.path.join(os.path.dirname(extract_path), 'images_stage2')
stage3_dir = os.path.join(os.path.dirname(extract_path), 'images_stage3')

os.makedirs(stage1_dir, exist_ok=True)
os.makedirs(stage2_dir, exist_ok=True)
os.makedirs(stage3_dir, exist_ok=True)

# Copy extracted images to stage1
dota_images = glob.glob(os.path.join(extract_path, "*.png")) + \
              glob.glob(os.path.join(extract_path, "*.jpg")) + \
              glob.glob(os.path.join(extract_path, "*.jpeg"))

for img in dota_images:
    copyfile(img, os.path.join(stage1_dir, os.path.basename(img)))

print(f"Copied {len(dota_images)} images to stage1.")


# --- Crop function ---
def crop(output_dir, img_path, crop_w, crop_h):
    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(img_path)
    w, h = image.size
    basename = os.path.splitext(os.path.basename(img_path))[0]

    for i in range(0, h, crop_h):
        for j in range(0, w, crop_w):
            box = (j, i, j + crop_w, i + crop_h)
            patch = image.crop(box)
            patch.save(os.path.join(output_dir, f"{basename}_{i}_{j}.png"))


# --- Crop to 1024x1024 patches (stage2) ---
images_st1 = glob.glob(os.path.join(stage1_dir, "*.png")) + \
             glob.glob(os.path.join(stage1_dir, "*.jpg"))

for img in images_st1:
    crop(stage2_dir, img, 1024, 1024)

print("Stage2 cropping done.")

# Copy stage2 to stage3
copytree(stage2_dir, stage3_dir, dirs_exist_ok=True)

final_st3_count = len(glob.glob(os.path.join(stage3_dir, "*.png")))
print(f"Total patches in stage3: {final_st3_count}")


# --- Step 2: Create HR–LR image pairs with SAME FILENAMES ---

SCALE_FACTOR = 2
BLUR_RADIUS = 2
BASE_SIZE = 1024  # HR size

HR_DIR = os.path.join(os.path.dirname(extract_path), 'HR')
LR_DIR = os.path.join(os.path.dirname(extract_path), f'LR_x{SCALE_FACTOR}')

os.makedirs(HR_DIR, exist_ok=True)
os.makedirs(LR_DIR, exist_ok=True)

images_hr_source = glob.glob(os.path.join(stage3_dir, "*.png"))
pair_count = 0

for hr_path in images_hr_source:
    try:
        hr_img = Image.open(hr_path)
        filename = os.path.basename(hr_path)  # SAME NAME FOR HR & LR

        # --- Save HR image ---
        hr_img.save(os.path.join(HR_DIR, filename))

        # --- Blur HR ---
        blurred = hr_img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))

        # --- Downscale to LR ---
        lr_size = (BASE_SIZE // SCALE_FACTOR, BASE_SIZE // SCALE_FACTOR)
        lr_img = blurred.resize(lr_size, Image.BICUBIC)

        # --- Save LR image with SAME NAME ---
        lr_img.save(os.path.join(LR_DIR, filename))

        pair_count += 1

    except Exception as e:
        print(f"Error: {hr_path} -> {e}")

print(f"\nGenerated {pair_count} HR–LR pairs.")
print(f"HR directory: {HR_DIR}")
print(f"LR directory: {LR_DIR}")


# --- (OPTIONAL) ZIP OUTPUT ---
output_zip = os.path.join(os.path.dirname(extract_path), "SISRimage_pairs.zip")
print("\nZipping HR and LR folders...")

with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for folder in [HR_DIR, LR_DIR]:
        for root, _, files in os.walk(folder):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, os.path.dirname(HR_DIR))
                zipf.write(full_path, arcname)

print(f"Zipped dataset saved to: {output_zip}")
