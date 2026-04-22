import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# -----------------------------
#  USER: SET THESE TWO PATHS
# -----------------------------
SR_SAVE_DIR = r"E:\ML\Testing\Results\final"       # e.g., r"E:\ML\Testing\SR_output"
PLOT_SAVE_DIR = r"E:\ML\Testing\Results\final\plots"     # e.g., r"E:\ML\Testing\plots"

# -----------------------------
# OTHER REQUIRED PATHS
# -----------------------------
VAL_LR = r"E:\ML\Training\x2_final\val_LR"
VAL_HR = r"E:\ML\Training\x2_final\val_HR"
MODEL_PATH = r"best_edsr_dota.pth"
SCALE = 2

os.makedirs(SR_SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

# -----------------------------
# VALIDATION PSNR LIST
# -----------------------------
validation_psnr = [
    28.60252890715728, 29.818993040033288, 30.426258602657832, 30.97869054046837,
    31.3907112173132, 31.784801496041787, 32.01196064175786, 32.013665134842334,
    32.328007034353305, 32.40412482699833, 31.851482681325965, 32.57174880440171,
    32.64964471636592, 32.69246207056819, 32.70978832889248, 32.625014015146206,
    32.82674555520754, 32.8160873361536, 32.8235604054219, 32.575432223242686,
    32.8730558060311, 32.97683568258543, 32.93049938614304, 33.008845503265796,
    32.98531615411913, 32.82772709872272, 33.154051664713265, 33.19024062801052,
    33.22834038734436, 33.225603342056274, 33.20449461163701, 33.118975272049774,
    33.30797219920803, 33.06367642815049, 33.360339564246104, 33.366985159951284,
    33.40068333213394, 33.366095620232656, 32.55992405479019, 33.421561028506304,
    33.42672360909952, 33.482500398481214, 33.45727762660465, 33.48968782296052,
    33.46719054273657, 33.47373403729619, 33.52518838160747, 33.54173265921103,
    33.36477297061199, 33.54914379764248, 33.43165355115323, 33.52210713077236,
    33.44287497288472, 33.515729221137796, 33.4857449789305, 33.563217807460475,
    33.568695261671735, 33.5980812730016, 33.645283525054516, 33.51410203366666,
    33.56038528519708, 33.35544200201292, 33.40898356566558, 33.66742777180027,
    33.64999764030044, 33.684722552428376, 33.61965263856424, 33.71920495419889,
    33.69260401983519, 33.69256485475076, 33.547025506560864, 33.71362578546679,
    33.50770976736739, 33.68023882041106, 33.74223810917622, 33.50152782491735,
    33.72181932346241, 33.640512260230814, 33.67290685627911, 33.71542005925565,
    33.70670218725462, 33.73293482290732, 33.73195792533256, 33.7623377490688,
    33.788976611317814, 33.77772770701228, 33.82013300947241, 33.739920235968924,
    33.674437355350804, 33.79538057301495, 33.775896310806274, 33.77241525778899,
    33.747273760872915, 33.83844210006095, 33.836034620130384, 33.67621064186096,
    33.72787064474982, 33.691643592473625, 33.763378304404185, 33.7559211834057
]

# -----------------------------
# UTIL FUNCTIONS
# -----------------------------
def rgb_to_y(img):
    r, g, b = img[0], img[1], img[2]
    return (0.299*r + 0.587*g + 0.114*b).unsqueeze(0)

def psnr_y(sr, hr, shave=2):
    sr_y = rgb_to_y(sr)
    hr_y = rgb_to_y(hr)

    if shave > 0:
        sr_y = sr_y[..., shave:-shave, shave:-shave]
        hr_y = hr_y[..., shave:-shave, shave:-shave]

    mse = torch.mean((sr_y - hr_y)**2)
    return 10 * torch.log10(1.0 / mse).item()

# -----------------------------
# MAIN
# -----------------------------
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your model
    from EDSR_final import EDSR
    model = EDSR(scale=SCALE).to(device)

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # Get images
    lr_list = sorted(glob.glob(os.path.join(VAL_LR, "*")))
    hr_list = sorted(glob.glob(os.path.join(VAL_HR, "*")))

    assert len(lr_list) == len(hr_list), "Mismatch between LR and HR count!"

    psnr_test = []
    ssim_test = []

    for lr_path, hr_path in zip(lr_list, hr_list):

        lr = TF.to_tensor(Image.open(lr_path).convert("RGB")).unsqueeze(0).to(device)
        hr = TF.to_tensor(Image.open(hr_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            sr = model(lr).clamp(0,1)

        # Save SR image
        sr_img = (sr[0].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
        out_name = os.path.basename(lr_path)
        Image.fromarray(sr_img).save(os.path.join(SR_SAVE_DIR, out_name))

        # Metrics
        p = psnr_y(sr[0], hr[0], shave=SCALE)
        psnr_test.append(p)

        sr_np = sr[0].cpu().numpy().transpose(1,2,0)
        hr_np = hr[0].cpu().numpy().transpose(1,2,0)
        s = ssim(sr_np, hr_np, data_range=1.0, channel_axis=2)
        ssim_test.append(s)

        print(f"{out_name} | PSNR: {p:.4f} | SSIM: {s:.4f}")

    # -----------------------------
    # PLOTS
    # -----------------------------

    # 1) Validation PSNR + Testing PSNR in ONE plot
    plt.figure(figsize=(10,6))
    plt.plot(validation_psnr, label="Validation PSNR", linewidth=2)
    plt.plot(psnr_test, label="Testing PSNR", linewidth=2)
    plt.xlabel("Image Index")
    plt.ylabel("PSNR (dB)")
    plt.title("Validation vs Testing PSNR")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_SAVE_DIR, "psnr_comparison.png"))
    plt.close()

    # 2) Testing SSIM curve
    plt.figure(figsize=(10,6))
    plt.plot(ssim_test, label="Testing SSIM", linewidth=2, color="green")
    plt.xlabel("Image Index")
    plt.ylabel("SSIM")
    plt.title("Testing SSIM Curve")
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_SAVE_DIR, "ssim_testing.png"))
    plt.close()

    # Print summary
    print("\n========== FINAL RESULTS ==========")
    print(f"Average Testing PSNR: {np.mean(psnr_test):.4f} dB")
    print(f"Average Testing SSIM: {np.mean(ssim_test):.4f}")
    print("Saved SR Images To:", SR_SAVE_DIR)
    print("Saved Plots To:", PLOT_SAVE_DIR)
    print("===================================\n")


if __name__ == "__main__":
    main()
