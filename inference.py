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
WEIGHTS = "runs/diffusion_epoch_033.pth"
#WEIGHTS = "model.pth"
TIMESTEPS = 1000
SAMPLE_DIR = "samples1"

# params for the noise schedule
BETA_MIN = 0.00001
BETA_MAX = 0.75




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


def cosine_noise_schedule(T, beta_min, beta_max):
	t = np.linspace(0, T-1, T)
	beta_t = beta_min + (beta_max - beta_min) * (1 - np.cos(t / T * np.pi)) / 2
	return beta_t


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

def add_noise_to_image(image, variance):
	std_dev = torch.sqrt(torch.tensor(variance))
	noise = torch.randn_like(image) * std_dev
	noised_image = image + noise
	noised_image = torch.clamp(noised_image, 0.0, 1.0)
	return noised_image


schedule = cosine_noise_schedule(TIMESTEPS, BETA_MIN, BETA_MAX)
def inference(model, iterations):

	acc_noise_delta = torch.zeros(1,3,64,64).to(device)

	for i in range(iterations):

		noisy_image = torch.rand(1, 3, 64, 64).to(device)   # make a purely random image

		for t in range(TIMESTEPS, 1, -1):

			timestep_encoding   = torch.full((1, 1, 64, 64), t).to(device)
			noisy_image_with_timestep = torch.cat((noisy_image, timestep_encoding), dim=1)

			beta_delta = schedule[t-1] - schedule[t-2]
			print(beta_delta)

			predicted_noise = model(noisy_image_with_timestep)
			noisy_image -= (predicted_noise * beta_delta)
			noisy_image = torch.clamp(noisy_image, 0.0, 1.0)


			filename = "sample%s_step%s.png" %(str(i).zfill(3), str(t).zfill(3))
			filepath = os.path.join(SAMPLE_DIR, filename)

			img_array = noisy_image.squeeze(dim=0)
			img_array = img_array.mul(255).byte().permute(1, 2, 0).cpu().numpy()

			pil_image = Image.fromarray(img_array)
			pil_image.save(filepath)
			pil_image.close()

	print(acc_noise_delta)

inference(model, 3)

