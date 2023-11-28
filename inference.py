import torch
import clip
import numpy as np
import os
from PIL import Image
import cv2
import time

from unet import UNet


CHECKPOINT_DIR = "runs"
#WEIGHTS = "runs/diffusion_epoch_039.pth"
WEIGHTS = "runs/diffusion_epoch_036.pth"
#WEIGHTS = "model.pth"
TIMESTEPS = 500
OUTDIR = "outputs"
INFDEBUG = "infdebug"

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


# Define model and load state dict
print("DEVICE: ", device)
model = UNet()

checkpoint = torch.load(WEIGHTS, map_location=torch.device('cpu'))
print(len(checkpoint))
model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)

#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print("U-Net Model and Weights Loaded...", flush=True)

img = torch.rand(1, 3, 64, 64).to(device)
#print(img)


def add_noise_to_image(image, variance):
	std_dev = torch.sqrt(torch.tensor(variance))
	noise = torch.randn_like(image) * std_dev
	noised_image = image + noise
	noised_image = torch.clamp(noised_image, 0.0, 1.0)
	return noised_image

for i in range(TIMESTEPS, -1, -1):
	print("Timestep: %d" %i)

	timestep_encoding   = torch.full((1, 1, 64, 64), i).to(device)
	concatenated_tensor = torch.cat((img, timestep_encoding), dim=1)

	img = model(concatenated_tensor)
	#img = add_noise_to_image(img, 0.01)

	if(False):
		debug_filename = "infdebug_%s.png" %(str(i).zfill(3)) 
		debug_filepath = os.path.join(INFDEBUG, debug_filename)
		
		img_array = img.squeeze(dim=0)
		img_array = img_array.mul(255).byte().permute(1, 2, 0).cpu().numpy()

		pil_image = Image.fromarray(img_array)
		pil_image.save(debug_filepath)
		pil_image.close()


'''
for i in range(10, -1, -1):
	print("Timestep: %d" %i)
	timestep_encoding   = torch.full((1, 1, 64, 64), i).to(device)
	concatenated_tensor = torch.cat((img, timestep_encoding), dim=1)

	img = model(concatenated_tensor)


timestep_encoding   = torch.full((1, 1, 64, 64), 0).to(device)
for i in range(10):
	concatenated_tensor = torch.cat((img, timestep_encoding), dim=1)
	img = model(concatenated_tensor)
'''




output_filename = "output.png"
output_path = os.path.join(OUTDIR, output_filename)
img = img.squeeze(dim=0)
img = img.mul(255).byte().permute(1, 2, 0).cpu().numpy()
pil_image = Image.fromarray(img)
pil_image.save(output_path)
	