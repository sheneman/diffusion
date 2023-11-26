import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

# params for the noise schedule
BETA_MIN = 0.0001
BETA_MAX = 0.5

RANDOM_SEED    = 42

BATCH_SIZE = 1250
SHUFFLE = True


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


def add_noise_to_images(images, variance):

	std_dev = torch.sqrt(torch.tensor(variance))

	# Generate Gaussian noise
	noise = torch.randn_like(images) * std_dev

	# Add noise to the images
	noised_images = images + noise

	return noised_images


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
		rrmse_loss = torch.sqrt(mse_loss)
		norm_rrmse_loss = (rrmse_loss/(64*64*3))*1000.0   # normalized to the pixel level

		return norm_rrmse_loss


class CustomDataset(Dataset):
	def __init__(self, filename, imgs, image_embeddings, caption_embeddings):
		self.filename = filename
		self.imgs = imgs
		self.image_embeddings = image_embeddings
		self.caption_embeddings = caption_embeddings

	def __len__(self):
		return len(self.imgs)

	def __getitem__(self, idx):

		# convert to PyTorch tensors
		image1		    = np.transpose(self.imgs[idx],(2,0,1))/255
		image2		    = torch.tensor(image1, dtype=torch.float32)

		image_embedding	    = torch.tensor(self.image_embeddings[idx], dtype=torch.float32)
		caption_embedding   = torch.tensor(self.image_embeddings[idx], dtype=torch.float32)

		return self.filename[idx], image2, image_embedding, caption_embedding



model = UNet()
#print(model)
model.to(device)


print("Loading training data ", TRAIN_DATA)
# read our training data
with open(TRAIN_DATA, 'rb') as traindata_file:
	train_data = pickle.load(traindata_file)

print("Loaded %d training records" %len(train_data), flush=True)

img = train_data[1000][1]
print("First Image: ", img.shape)

a0,a1,a2,a3 = zip(*train_data)
filenames	    = list(a0)
images		    = list(a1)
image_embeddings    = list(a2)
caption_embeddings  = list(a3)

del train_data

dataset = CustomDataset(filenames, images, image_embeddings, caption_embeddings)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, drop_last=True)
    

#schedule = linear_beta_schedule(TIMESTEPS)
#schedule = cosine_beta_schedule(TIMESTEPS)
schedule = cosine_noise_schedule(TIMESTEPS, BETA_MIN, BETA_MAX)

loss_function = nn.MSELoss()
#loss_function = NormRRMSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)



# Train across all EPOCHS
for epoch in range(MAX_EPOCHS):
	print("Epoch: ", epoch)
    
	model.train()
	epoch_running_loss = 0.0
	batch_running_loss = 0.0

	# For all images in training data
	for i, batch in enumerate(data_loader):
    
		#print("BATCH: ", i, flush=True)

		optimizer.zero_grad()

		filenames, images, image_embeddings, caption_embeddings = batch

		images		    = images.to(device) 
		image_embeddings    = image_embeddings.to(device)
		caption_embeddings  = caption_embeddings.to(device)

		#print("Training batch %d/%d" %(i,len(data_loader)))

		#image = image.astype(np.float32) 
	
		# For all timesteps in our noise schedule	
		for ti, beta in enumerate(schedule): 

			noisy_batch = add_noise_to_images(images, beta)

			# Create a tensor of zeros with the same batch size, height, and width
			timestep_encoding = torch.full((BATCH_SIZE, 1, 64, 64), ti).to(device)
			#print(timestep_encoding.shape)

			#print("HERE: ", noisy_batch.shape, timestep_encoding.shape)

			# Concatenate the timestep encoding tensor to the batch along the channel dimension
			noisy_batch_with_timestep = torch.cat((noisy_batch, timestep_encoding), dim=1)
    
			# timestep encoding
			#timestep_encoding = np.full((64,64,1), float(ti)).astype(np.float32)
			#noisy_image = np.concatenate((noisy_image, timestep_encoding), axis=2)	

			#image_array_transposed = np.transpose(noisy_image,(2,0,1))
			#image_tensor = torch.from_numpy(image_array_transposed).unsqueeze(0)
			#image_tensor = image_tensor.to(device)
		
			outputs = model(noisy_batch_with_timestep)

			# get what the image should look like in the prior timestep
			if(ti==0):
			    prior_step = ti
			else:
			    prior_step = ti-1

			
			#target_tensor = add_noise_to_images(images, schedule[prior_step]) 
			target_tensor = images

			#print("outputs: ", outputs.shape)
			#print("target_tensor: ", target_tensor.shape)

			#print("OUTPUTS: ") 
			#print(outputs)
			#print("\n")
#
#			print("TARGETS: ") 
#			print(target_tensor)
#			print("***************************")
			
			loss = loss_function(outputs, target_tensor)

			loss.backward()
			optimizer.step()	
	
			#print(type(loss.item()), loss.item())

			epoch_running_loss += loss.item()
			batch_running_loss += loss.item()

		print("%d/%d:  BATCHLOSS = %.06f" %(i,len(data_loader),batch_running_loss/BATCH_SIZE))
		batch_running_loss = 0.0

	# Print epoch statistics
	epoch_loss = epoch_running_loss / (BATCH_SIZE*ti) 
	print(f"Epoch [{epoch+1}/{MAX_EPOCHS}], Loss: {epoch_running_loss:.4f}")

	checkpoint = {
	    'epoch': epoch,
	    'model_state_dict': model.state_dict(),
	    'optimizer_state_dict': optimizer.state_dict(),
	} 
	checkpoint_name = "diffusion_epoch_%s.pth" %(str(epoch).zfill(3))
	checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
	torch.save(checkpoint, checkpoint_path)

	train_cnt = 0

	
