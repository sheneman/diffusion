import cv2
import pandas as pd
import numpy as np
import torch
import clip
import sys, os
from PIL import Image
import pickle

MAX_RECORDS = 5000
IMAGE_DIR = "img_align_celeba"

# Path to the JSON file
annotations_file = 'list_attr_celeba.txt'

df = pd.read_csv(annotations_file, sep="\s+", skiprows=1, header=0)


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)



def crop_largest_square(img):
	# Get image dimensions
	height, width = img.shape[:2]

	# Determine the size of the square
	square_size = min(height, width)

	# Calculate the top left point of the square
	top_left_x = (width - square_size) // 2
	top_left_y = (height - square_size) // 2

	# Crop the image
	cropped_img = img[top_left_y:top_left_y + square_size, top_left_x:top_left_x + square_size]

	return cropped_img



# Iterate through each row
cnt = 0
data = []
for filename, row in df.iterrows():

	if(cnt >= MAX_RECORDS):
		break
	cnt += 1

	matching_columns = row.index[row == 1].tolist()

	prompt = "Photo of person "
	for d in matching_columns:
		term = d.replace("_", " ")
		term = term.lower()
		prompt += term + ", "
	prompt = prompt[:-2]
	prompt += "."


	filepath = os.path.join(IMAGE_DIR, filename)
	image = Image.open(filepath)
	image_array = np.array(image)
	image.close()

	if(len(image_array.shape)>2):
		height,width,channels = image_array.shape
	else:
		height,width = image_array.shape
		channels = 1

	print("Height = %d, Width = %d, Channels = %d" %(height, width, channels))
	
	if(channels < 3):
		print("***APPARENT GRAYSCALE IMAGE.  CONVERTING TO COLOR")
		image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)

	crop_img = crop_largest_square(image_array)
	img = cv2.resize(crop_img, (64,64))
	print("Resized: ", img.shape)

	print(filename, prompt)	
	print("\n")

	try:
		tokenized_prompt = clip.tokenize(prompt).to(device)
	except:
		print("WARNING:  Prompt too long for tokenizer.  Skipping.")
		continue

	with torch.no_grad():
		text_embeddings  = model.encode_text(tokenized_prompt)

		pil_image = Image.fromarray(img)

		preprocess_image = preprocess(pil_image).unsqueeze(0).to(device)
		image_embeddings = model.encode_image(preprocess_image)

	data.append((filename, img, image_embeddings, text_embeddings))
    
# Pickle the tensor
print("Saving %d images to pickle file" %(len(data)))
with open("celeba.pickle", 'wb') as f:
	pickle.dump(data, f)
