import torch
import clip
import numpy as np
import os
from PIL import Image
import cv2
import time

from unet import UNet


CHECKPOINT_DIR = "runs"
WEIGHTS = "runs/diffusion_epoch_039.pth"
TIMESTEPS = 30
OUTDIR = "outputs"

#
# Set seed and device 
#
RANDOM_SEED    = time.time()
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
	device = "cuda:1"
	torch.cuda.manual_seed(RANDOM_SEED)
	torch.cuda.manual_seed_all(RANDOM_SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
else:
	device = "cpu"

#device="cpu"

# Define model and load state dict
print("DEVICE: ", device)
model = UNet()
model.to(device)


checkpoint = torch.load(WEIGHTS, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.eval()
print("U-Net Model and Weights Loaded...", flush=True)

img = torch.rand(1, 3, 64, 64).to(device)

for i in range(TIMESTEPS, -1, -1):
	print("Timestep: %d" %i)

	timestep_encoding   = torch.full((1, 1, 64, 64), i).to(device)
	concatenated_tensor = torch.cat((img, timestep_encoding), dim=1)

	img = model(concatenated_tensor)

output_filename = "output.png"
output_path = os.path.join(OUTDIR, output_filename)
img = img.squeeze(dim=0)
img = img.mul(255).byte().permute(1, 2, 0).cpu().numpy()
pil_image = Image.fromarray(img)
pil_image.save(output_path)
	
