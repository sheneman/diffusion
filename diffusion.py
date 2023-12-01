####################################################################
#
# diffusion.py
#
# Luke Sheneman
# sheneman@uidaho.edu
# November 2023
#
# Generative AI diffusion model for University of Idaho Tech Talk
#
####################################################################

import argparse
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


#TRAIN_DATA		= "COCO_with_CLIP_embeddings.pickle"
DEFAULT_INPUT		= "CELEBA_with_CLIP_embeddings.pickle"
DEFAULT_TIMESTEPS	= 1000
DEFAULT_OUTPUT_DIR      = "outputs"
DEFAULT_EPOCHS		= 500
DEFAULT_LEARNING_RATE	= 0.0001
DEFAULT_CHECKPOINT_DIR	= "runs"
DEFAULT_DEBUG_DIR	= "debug"
DEFAULT_SAMPLE		= False
DEFAULT_SAMPLE_DIR	= "samples"
DEFAULT_RANDOM_SEED	= int(time.time())
DEFAULT_BATCH_SIZE	= 700
DEFAULT_INPUT_SHUFFLE	= True
DEFAULT_ITERATIONS	= 3	# Number of inferences to do

# params for the noise schedule
DEFAULT_BETA_MIN	= 0.00
DEFAULT_BETA_MAX	= 0.75




parser = argparse.ArgumentParser(prog='diffusion', description='Train and inference with a generative AI Diffusion model.')

parser.add_argument('-m', '--model',	     type=str,   help='Path to the PyTorch model weights file for inference or continued training')
parser.add_argument('-t', '--timesteps',     type=int,   default=DEFAULT_TIMESTEPS, help='Number of diffusion timesteps for training of inference.  (DEFAULT: '+str(DEFAULT_TIMESTEPS)+')')
parser.add_argument('-s', '--seed',	     type=int,   default=DEFAULT_RANDOM_SEED, help='Random seed to use for inference and training (DEFAULT: '+str(DEFAULT_RANDOM_SEED)+')')

group = parser.add_mutually_exclusive_group()
group.add_argument('-T', '--train',	     action='store_true', default=True,  help='Train mode (DEFAULT)')
group.add_argument('-j', '--inference',	     action='store_true', default=False, help='Inference mode')

train_group = parser.add_argument_group('train_group')
parser.add_argument('-i',  '--input',	     type=str,   default=DEFAULT_INPUT,		 help='Dataset file for training. Specific pickle format expected.  (DEFAULT: '+str(DEFAULT_INPUT)+')')
parser.add_argument('-e',  '--epochs',	     type=int,   default=DEFAULT_EPOCHS,	 help='Number of training epochs. (DEFAULT: '+str(DEFAULT_EPOCHS)+')')
parser.add_argument('-b',  '--batchsize',    type=str,   default=DEFAULT_BATCH_SIZE,	 help='The batch size for training. (DEFAULT: '+str(DEFAULT_BATCH_SIZE)+')')
parser.add_argument('-lr', '--learningrate', type=str,   default=DEFAULT_LEARNING_RATE,  help='The learning rate (DEFAULT: '+str(DEFAULT_LEARNING_RATE)+')')
parser.add_argument('-x',  '--shuffle',	     type=str,   default=DEFAULT_INPUT_SHUFFLE,	 help='Shuffle inputs during training.(DEFAULT: '+str(DEFAULT_INPUT_SHUFFLE)+')')
parser.add_argument('-S',  '--sample',	     type=str,   default=DEFAULT_SAMPLE,	 help='Sample from our model every epoch. (DEFAULT: '+str(DEFAULT_SAMPLE)+')')
parser.add_argument('-d',  '--sampledir',    type=str,   default=DEFAULT_SAMPLE_DIR,	 help='Directory to put samples during training. (DEFAULT: '+str(DEFAULT_SAMPLE_DIR)+')')
parser.add_argument('-c',  '--checkpointdir',type=str,   default=DEFAULT_CHECKPOINT_DIR, help='Directory to save checkpoints from model training. (DEFAULT: '+str(DEFAULT_CHECKPOINT_DIR)+')')
parser.add_argument('-bmin',  '--betamin',   type=str,   default=DEFAULT_BETA_MIN,	 help='The variance (beta) minimum for applying noise. (DEFAULT: '+str(DEFAULT_BETA_MIN)+')')
parser.add_argument('-bmax',  '--betamax',   type=str,   default=DEFAULT_BETA_MAX,	 help='The variance (beta) maximum for applying noise. (DEFAULT: '+str(DEFAULT_BETA_MAX)+')')

inference_group = parser.add_argument_group("Inference")
inference_group.add_argument('-o', '--output',	    type=str,   default=DEFAULT_OUTPUT_DIR, help='The directory for generated images (DEFAULT: '+str(DEFAULT_OUTPUT_DIR)+')')

args = parser.parse_args()


if(args.inference):
	args.train = False
	if(not 'model' in args or ('model' in args and not args.model)):
		print("Error: You must specify a model for inference.")
		parser.print_usage()
		exit(-1)
	if(not os.path.isdir(args.output)):
		print("Error: Could not find inference output directory: ", args.output)
		parser.print_usage()
		exit(-1)

if 'model' in args:
	if(args.model):
		if(not os.path.exists(args.model)):
			print("Error: Could not find model: ", args.model)
			parser.print_usage()
			exit(-1)

if(args.train):
	if(args.sample):
		if(not os.path.exists(args.sampledir)):
			print("Error:  Could not find sample directory: ", args.sampledir)
			parse.print_usage()
			exit(-1)
	if(not os.path.isdir(args.checkpointdir)):
			print("Error:  Could not find checkpoint directory: ". args.checkpointdir)
			parse.print_usage()
			exit(-1)



print("\n\n")
print("*******************************")
print("*                             *")
print("*         DIFFUSION           *") 
print("*                             *")
print("*******************************")
print("\n")

if(args.train == True):
	print("MODE: TRAINING")
	print("--------------")
	print("Training Data: ", args.input)
	print("Random Seed:   ", args.seed)
	print("Learning Rate: ", args.learningrate)
	print("Sample:        ", args.sample)
	if(args.sample):
		print("Sample Dir:    ", args.sampledir)
	print("Shuffle Data:  ", args.shuffle)
	print("Epochs:        ", args.epochs)
	print("Batch Size:    ", args.batchsize)
	print("Timesteps:     ", args.timesteps)
    
else:
	print("MODE: INFERENCE")
	print("---------------")
	print("Model:     ", args.model)
	print("Output:    ", args.output)
	print("Timesteps: ", args.timesteps)
	print("   Seed:   ", args.seed)

print("\n\n*******************************")


#
# Set seed and device 
#
torch.manual_seed(args.seed)
if torch.cuda.is_available():
	device = "cuda"
	torch.cuda.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
else:
	device = "cpu"


print("DEVICE = ", device)
print("\n")


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
		
	img = (img.cpu().detach().numpy()*255).astype('uint8')
	img = np.transpose(img,(1,2,0))

	print("Transposed Image: ", img.shape)
	#print(img)

	filename = "%s_%s.png" %(name, str(timestep).zfill(3))
	filepath = os.path.join(DEFAULT_DEBUG_DIR, filename)

	pil_image = Image.fromarray(img)
	pil_image.save(filepath)	
	print("Saved file to %s", filepath)


def train():	
	global args
	global device

	# instantiate our model and send it to the device (GPU, etc.)
	model = UNet()

	if('model' in args and args.model):
		print("Loading checkpoint for additional training...")
		checkpoint = torch.load(args.model, map_location=torch.device('cpu'))
		model.load_state_dict(checkpoint['model_state_dict'])
	else:
		print("Training a model from scratch...")

	model.to(device)

	# read our training data
	print("Loading training data ", args.input)
	with open(args.input, 'rb') as traindata_file:
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
	data_loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=args.shuffle, drop_last=True)
	    

	#schedule = linear_beta_schedule(args.timesteps)
	schedule  = cosine_noise_schedule(args.timesteps, args.betamin, args.betamax)

	# define our loss function and optimizer
	loss_function = CustomMSELoss()
	optimizer = optim.Adam(model.parameters(), lr=args.learningrate)


	# Train across all epochs...
	for epoch in range(args.epochs):
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
			ti = 0
			noisy_batch = images
			for ti in range(args.timesteps):
				if(ti == 0):
					beta_delta = schedule[ti]
				else:
					beta_delta = schedule[ti]-schedule[ti-1]

				noisy_batch = add_noise_to_images(noisy_batch, beta_delta)

				#debug_batch("noise", noisy_batch, ti)

				timestep_encoding = torch.full((args.batchsize, 1, 64, 64), ti).to(device)
				noisy_batch_with_timestep = torch.cat((noisy_batch, timestep_encoding), dim=1)
	    
				predicted_noise = model(noisy_batch_with_timestep)
				actual_noise	= noisy_batch - images	

				loss = loss_function(predicted_noise, actual_noise)
				loss.backward()
				optimizer.step()	
		
				epoch_running_loss += loss.item()
				batch_running_loss += loss.item()

			print("%d/%d:  BATCHLOSS = %.06f" %(batch_i+1,len(data_loader),batch_running_loss/(args.batchsize*args.timesteps)), flush=True)
			batch_running_loss = 0.0
		    


		# Print epoch statistics
		epoch_loss = epoch_running_loss / (args.batchsize*(batch_i+1)*args.timesteps) 
		print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss:.4f}", flush=True)
		epoch_running_loss = 0.0

		#if(epoch%10 == 0):
		#	#sample_model(epoch, model, device)	
		#	# FIX THIS TO SAMPLE VIA THE INFERENCE FUNCTION
		#	inference()

		checkpoint = {
		    'epoch': epoch,
		    'model_state_dict': model.state_dict(),
		    'optimizer_state_dict': optimizer.state_dict(),
		} 
		checkpoint_name = "diffusion_epoch_%s.pth" %(str(epoch).zfill(3))
		checkpoint_path = os.path.join(args.checkpointdir, checkpoint_name)
		torch.save(checkpoint, checkpoint_path)

		# Save full model instead of just the state dict
		#checkpoint_name = "FULL_diffusion_epoch_%s.pt" %(str(epoch).zfill(3))
		#checkpoint_path = os.path.join(args.checkpointdir, checkpoint_name)
		#torch.save(model, checkpoint_path)




def inference(iterations):

	global args
	global device

	print("INFERENCE:")

	if(not os.path.exists(args.model)):
		print("Error.  Could not find model file %s for inference.  Aborting." %args.model)
		exit(0)

	model = UNet()
	checkpoint = torch.load(args.model, map_location=torch.device('cpu'))
	model.load_state_dict(checkpoint['model_state_dict'])
	model.to(device)

	#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	print("U-Net Model and Weights Loaded...", flush=True)

	schedule = cosine_noise_schedule(args.timesteps, args.betamin, args.betamax)

	for i in range(iterations):

		noisy_image = torch.rand(1, 3, 64, 64).to(device)   # make a purely random image

		for t in range(args.timesteps, 1, -1):

			timestep_encoding   = torch.full((1, 1, 64, 64), t).to(device)
			noisy_image_with_timestep = torch.cat((noisy_image, timestep_encoding), dim=1)

			beta_delta = schedule[t-1] - schedule[t-2]

			predicted_noise = model(noisy_image_with_timestep)
			noisy_image -= (predicted_noise * beta_delta)
			noisy_image = torch.clamp(noisy_image, 0.0, 1.0)

			filename = "sample%s_step%s.png" %(str(i).zfill(3), str(t).zfill(3))
			filepath = os.path.join(args.output, filename)

			img_array = noisy_image.squeeze(dim=0)
			img_array = img_array.mul(255).byte().permute(1, 2, 0).cpu().numpy()

			pil_image = Image.fromarray(img_array)
			pil_image.save(filepath)
			pil_image.close()
	


if(args.train):
	train()
else:
	inference(DEFAULT_ITERATIONS)	


