import torch
import clip
import pickle
import numpy as np
import os
from PIL import Image
import cv2

from unet import UNet


IMG	   = "CLIP.png"
TRAIN_DATA = "COCO_with_CLIP_embeddings.pickle"
TIMESTEPS  = 30
OUTDIR     = "outputs"



#LABELS = ["dog", "cat", "bear", "person"]
LABELS = ["mean", "funny", "cute", "ugly"]

def linear_beta_schedule(num_steps):
	return [i / num_steps for i in range(1, num_steps + 1)]

def cosine_beta_schedule(num_steps, s=0.008):
	steps = np.arange(num_steps, dtype=np.float64)
	return 1.0-np.cos(((steps / num_steps) + s) / (1 + s) * np.pi / 2) ** 2

def cosine_noise_schedule(T, beta_min, beta_max):
    t = np.linspace(0, T-1, T)
    beta_t = beta_min + (beta_max - beta_min) * (1 - np.cos(t / T * np.pi)) / 2
    return beta_t


def apply_gaussian_noise(img, beta):
	noise = np.random.normal(size=img.shape)
	print(beta, noise.shape)
	#return np.sqrt(1 - beta) * img + np.sqrt(beta) * noise
	return (1-beta)*img + beta*noise


'''
model.train()
for epoch in range(num_epochs):
	for images in data_loader:
		for timestep in range(num_timesteps):
			optimizer.zero_grad()

			# Calculate beta for the current timestep
			beta = linear_scheduler(timestep, num_timesteps, beta_start, beta_end)

			# Add noise to the images based on the current beta
			noisy_images = apply_gaussian_noise(images, beta)

			# Forward pass with noisy images
			output = model(noisy_images)

			# Compute loss (e.g., mean squared error between output and original images)
			loss = nn.MSELoss()(output, images)

			# Backward pass and optimize
			loss.backward()
			optimizer.step()

			# Logging
			if timestep % 100 == 0:
				print(f"Epoch [{epoch}/{num_epochs}], Timestep [{timestep}/{num_timesteps}], Loss: {loss.item()}")
'''



device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet()
print(model)

print("Loading training data ", TRAIN_DATA)
# read our training data
with open(TRAIN_DATA, 'rb') as traindata_file:
	train_data = pickle.load(traindata_file)

print("Loaded %d training records" %len(train_data), flush=True)

img = train_data[1000][1]
print("First Image: ", img.shape)

beta_min = 0.0001
beta_max = 0.5
#schedule = linear_beta_schedule(TIMESTEPS)
#schedule = cosine_beta_schedule(TIMESTEPS)
schedule = cosine_noise_schedule(TIMESTEPS, beta_min, beta_max)

for i, beta in enumerate(schedule):
	new_image = apply_gaussian_noise(img/255, beta)
	
	filename = IMG+"_forward_"+str(i).zfill(3)+".png"
	filepath = os.path.join(OUTDIR, filename)

	new_image = (new_image*255).astype(np.uint8)
	new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
	cv2.imwrite(filepath, new_image)

print("Created %d forward diffusion images in %s"  %(len(schedule), OUTDIR))

image_array = np.random.rand(64,64,3).astype(np.float32) 
image_array_transposed = np.transpose(image_array,(2,0,1))
image_tensor = torch.from_numpy(image_array_transposed).unsqueeze(0)
reverse_diff_images = []
for i,beta in enumerate(reversed(schedule)):
	print("Calling model() with: ", image_tensor.shape)
	result = model(image_tensor)
	print("FROM model(): ", type(result))
	print("FROM model(): ", result.shape)

	result = result.squeeze(0)
	print("after squeeze(): ", type(result))
	print("after squeeze(): ", result.shape)


	reverse_diff_images.append(result.squeeze(0).detach().numpy())

for i, new_image in enumerate(reverse_diff_images):
	filename = IMG+"_reverse_"+str(i).zfill(3)+".png"
	filepath = os.path.join(OUTDIR, filename)

	new_image = (new_image*255).astype(np.uint8)

	print("L: ", new_image.shape)
	new_image = np.transpose(new_image,(1,2,0))
	print("L: ", new_image.shape)

	new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

	print("TO FILE: ", type(new_image))
	print("TO FILE: ", new_image.shape)
    
	cv2.imwrite(filepath, new_image)

print("Done")

