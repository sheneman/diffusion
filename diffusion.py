import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import clip
import pickle
import time
import numpy as np
import os
from PIL import Image
import cv2

from unet import UNet


IMG	       = "CLIP.png"
#TRAIN_DATA     = "COCO_with_CLIP_embeddings.pickle"
TRAIN_DATA     = "CELEBA_with_CLIP_embeddings.pickle"
TIMESTEPS      = 100
OUTDIR         = "outputs"
MAX_EPOCHS     = 100
LEARNING_RATE  = 0.001
CHECKPOINT_DIR = "/tmp"
DEBUG_DIR      = "debug"
SAMPLE_DIR     = "samples"

# params for the noise schedule
BETA_MIN = 0.00001
BETA_MAX = 4.00000

RANDOM_SEED    = time.time()

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

	noised_images = torch.clamp(noised_images, 0.0, 1.0)

	return noised_images


#
# This function is useful for grabbing inference samples as we are training
#
def sample_model(epoch, model, device):

	print("Sampling the model")
	img = torch.rand(1, 3, 64, 64).to(device)   # make a purely random image

	for i in range(TIMESTEPS, -1, -1):
		print("Timestep: %d" %i)

		timestep_encoding   = torch.full((1, 1, 64, 64), i).to(device)
		concatenated_tensor = torch.cat((img, timestep_encoding), dim=1)

		img = model(concatenated_tensor)

		sample_filename = "sample_epoch%s_step%s.png" %(str(epoch).zfill(3), str(i).zfill(3))
		sample_filepath = os.path.join(SAMPLE_DIR, sample_filename)

		img_array = img.squeeze(dim=0)
		img_array = img_array.mul(255).byte().permute(1, 2, 0).cpu().numpy()

		pil_image = Image.fromarray(img_array)
		pil_image.save(sample_filepath)
		pil_image.close()




# Define a custom MSE loss function
class CustomMSELoss(nn.Module):
	def __init__(self):
		super(CustomMSELoss, self).__init__()

	def forward(self, predicted, target):
		mse_loss = F.mse_loss(predicted, target, reduction='mean')
		custom_loss = mse_loss * 1e4

		return custom_loss


# Define a custom dataset for our data loader
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


#
# A useful function for debuggings stuff as we train
#
def debug_batch(name, batch, timestep):
	print("In debug_batch():")
	print("Batch Dimensions: ", batch.shape)

	example = batch[10]
	print("Example Dimensions: ", example.shape)

	if(example.shape[0]==4):
		print("This batch includes a timestep channel!")

		# lets get the timestep encoding value first
		timestep_encoding = int(example[3][0][0].cpu().numpy())
		print("Timestep Encoding: ", timestep_encoding)

		img = example[:3, :, :]
	else:
		img = example


	print("Tensor Image: ", img.shape)
		
	img = img.cpu().detach().numpy()
	img = np.transpose(img,(1,2,0))

	print("Transposed Image: ", img.shape)
	#print(img)

	filename = "%s_%s.png" %(name, str(timestep).zfill(3))
	filepath = os.path.join(DEBUG_DIR, filename)

	pil_image = Image.fromarray(img)
	pil_image.save(filepath)	
	print("Saved file to %s", filepath)

	



# print instantiate our model and send it to the GPU
model = UNet()
model.to(device)


# read our training data
print("Loading training data ", TRAIN_DATA)
with open(TRAIN_DATA, 'rb') as traindata_file:
	train_data = pickle.load(traindata_file)
print("Loaded %d training records" %len(train_data), flush=True)



a0,a1,a2,a3 = zip(*train_data)
filenames	    = list(a0)
images		    = list(a1)
image_embeddings    = list(a2)
caption_embeddings  = list(a3)
del train_data

# Build our custom dataset and data loader
dataset     = CustomDataset(filenames, images, image_embeddings, caption_embeddings)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, drop_last=True)
    

#schedule = linear_beta_schedule(TIMESTEPS)
schedule  = cosine_noise_schedule(TIMESTEPS, BETA_MIN, BETA_MAX)

# define our loss function and optimizer
loss_function = CustomMSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Train across all EPOCHS
for epoch in range(MAX_EPOCHS):
	print("Epoch: ", epoch+1)
    
	model.train()
	epoch_running_loss = 0.0
	batch_running_loss = 0.0

	# For all images in training data
	for batch_i, batch in enumerate(data_loader):
    
		optimizer.zero_grad()

		filenames, images, image_embeddings, caption_embeddings = batch

		images		    = images.to(device) 
		image_embeddings    = image_embeddings.to(device)
		caption_embeddings  = caption_embeddings.to(device)

		# For all timesteps in our noise schedule	
		for ti, beta in enumerate(schedule): 

			#debug_batch(images, ti)

			noisy_batch = add_noise_to_images(images, beta)
			#debug_batch(noisy_batch, ti)

			# Create a tensor of zeros with the same batch size, height, and width
			timestep_encoding = torch.full((BATCH_SIZE, 1, 64, 64), ti).to(device)
			#print(timestep_encoding.shape)

			#print("HERE: ", noisy_batch.shape, timestep_encoding.shape)

			# Concatenate the timestep encoding tensor to the batch along the channel dimension
			noisy_batch_with_timestep = torch.cat((noisy_batch, timestep_encoding), dim=1)

			#print("SHAPE: ", noisy_batch_with_timestep.shape)

			#debug_batch(noisy_batch_with_timestep, ti)
    
			outputs = model(noisy_batch_with_timestep)

			# get what the image should look like in the prior timestep
			if(ti==0):
			    prior_step = ti
			else:
			    prior_step = ti-1

			
			#target_tensor = add_noise_to_images(images, schedule[prior_step]) 
			target_tensor = images

			
			loss = loss_function(outputs, target_tensor)
			loss.backward()
			optimizer.step()	
	
			epoch_running_loss += loss.item()
			batch_running_loss += loss.item()

		print("%d/%d:  BATCHLOSS = %.06f" %(batch_i,len(data_loader),batch_running_loss/(BATCH_SIZE*TIMESTEPS)))
		batch_running_loss = 0.0




	# Print epoch statistics
	epoch_loss = epoch_running_loss / (BATCH_SIZE*(batch_i+1)*TIMESTEPS) 
	print(f"Epoch [{epoch+1}/{MAX_EPOCHS}], Loss: {epoch_loss:.4f}")
	epoch_running_loss = 0.0

	sample_model(epoch, model, device)	

	checkpoint = {
	    'epoch': epoch,
	    'model_state_dict': model.state_dict(),
	    'optimizer_state_dict': optimizer.state_dict(),
	} 
	checkpoint_name = "diffusion_epoch_%s.pth" %(str(epoch).zfill(3))
	checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
	torch.save(checkpoint, checkpoint_path)

	checkpoint_name = "FULL_diffusion_epoch_%s.pt" %(str(epoch).zfill(3))
	checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
	torch.save(model, checkpoint_path)
	
