import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import clip
import pickle
import numpy as np
import os
from PIL import Image
import cv2

from unet import UNet


IMG	       = "CLIP.png"
TRAIN_DATA     = "COCO_with_CLIP_embeddings.pickle"
TIMESTEPS      = 30
OUTDIR         = "outputs"
MAX_EPOCHS     = 100
LEARNING_RATE  = 0.001
CHECKPOINT_DIR = "runs"

RANDOM_SEED    = 42


#
# Set seed and device 
#
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
	device = "cuda"
	torch.cuda.manual_seed(RANDOM_SEED)
	torch.cuda.manual_seed_all(RANDOM_SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
else:
	device = "cpu"


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
	#print(beta, noise.shape)
	#return np.sqrt(1 - beta) * img + np.sqrt(beta) * noise
	return (1-beta)*img + beta*noise


# Define a custom RMSE loss function
class NormRRMSELoss(nn.Module):
	def __init__(self):
		super(NormRRMSELoss, self).__init__()

	def forward(self, predicted, target):
		mse_loss = F.mse_loss(predicted, target, reduction='mean')
		rrmse_loss = torch.sqrt(torch.sqrt(mse_loss))
		norm_rrmse_loss = (rrmse_loss/(64*64*3))*100.0   # normalized to the pixel level

		return norm_rrmse_loss



model = UNet()
print(model)

model.to(device)

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

#loss_function = nn.MSELoss()
loss_function = NormRRMSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# For each Epoch
for epoch in range(MAX_EPOCHS):
	print("Epoch: ", epoch)
    
	model.train()
	epoch_running_loss = 0.0

	running_loss = 0.0

	# For all images in training data
	for i,td in enumerate(train_data):
		(filename, image, image_embeddings, caption_embeddings) = td
		#print("Training on Image %d/%d" %(i,len(train_data)))

		image = image.astype(np.float32) 
	
		# For all timesteps in our noise schedule	
		for ti, beta in enumerate(schedule): 
			#print(".", end="", flush=True)
    
			optimizer.zero_grad()

			noisy_image = apply_gaussian_noise(image/255, beta).astype(np.float32)
    
			# timestep encoding
			timestep_encoding = np.full((64,64,1), float(ti)).astype(np.float32)
			noisy_image = np.concatenate((noisy_image, timestep_encoding), axis=2)	

			image_array_transposed = np.transpose(noisy_image,(2,0,1))
			image_tensor = torch.from_numpy(image_array_transposed).unsqueeze(0)
			image_tensor = image_tensor.to(device)
		
			outputs = model(image_tensor)

			# get what the image should look like in the prior timestep
			if(ti==0):
			    prior_step = ti
			else:
			    prior_step = ti-1

			noisy_target  = apply_gaussian_noise(image/255, schedule[prior_step]).astype(np.float32)
			timestep_encoding = np.full((64,64,1), float(prior_step)).astype(np.float32)
			#noisy_target = np.concatenate((noisy_target, timestep_encoding), axis=2)	
			image_array_transposed = np.transpose(noisy_target,(2,0,1))
			target_tensor = torch.from_numpy(image_array_transposed).unsqueeze(0)
			target_tensor = target_tensor.to(device)

			#print("outputs: ", outputs.shape)
			#print("target_tensor: ", target_tensor.shape)
			
			loss = loss_function(outputs, target_tensor)

			loss.backward()
			optimizer.step()	
	
			#print(type(loss.item()), loss.item())

			epoch_running_loss += loss.item()
			running_loss += loss.item()

		# Every 100 images, show running loss
		if(i>0 and i%100==0):
			print("%d/%d:  RUNNING LOSS = %.06f" %(i,len(train_data),running_loss/100))
			running_loss = 0.0

	# Print epoch statistics
	epoch_loss = epoch_running_loss / len(train_data)
	print(f"Epoch [{epoch+1}/{MAX_EPOCHS}], Loss: {epoch_running_loss:.4f}")

	checkpoint = {
	    'epoch': epoch,
	    'model_state_dict': model.state_dict(),
	    'optimizer_state_dict': optimizer.state_dict(),
	} 
	checkpoint_name = "diffusion_epoch_%s.pth" %(str(epoch).zfill(3))
	checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
	torch.save(checkpoint, checkpoint_path)

	


exit(0)	    


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

