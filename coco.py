import json
import cv2
import torch
import clip
import sys, os
from PIL import Image
import pickle

IMAGE_DIR = "train2014"

# Path to the JSON file
coco_captions_file = 'annotations/captions_train2014.json'

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# Function to read and parse the JSON file
def read_coco_captions(filename):
	with open(filename, 'r') as file:
		data = json.load(file)

	# Extracting captions and image ids
	images = data['images']
	annotations = data['annotations']

	# Mapping image IDs to captions
	image_captions = {img['id']: [] for img in images}
	for ann in annotations:
		image_captions[ann['image_id']].append(ann['caption'])

	return images, image_captions

images, coco_captions = read_coco_captions(coco_captions_file)

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


data = []
for i in images:
	id = i['id']
	fn = i['file_name']
	fp = os.path.join(IMAGE_DIR, fn)
	annotation = coco_captions[id]

	print("ID: ", id)
	print("File Path: ", fp)
	print("Caption: ")
	full_caption = ""
	for caption in coco_captions[id]:
		full_caption += " " + caption
	print(full_caption)

	if(len(full_caption) > 77):  # should be tokens, not characters!
		full_caption = full_caption[0:77]


	img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
	#print(img.shape)
	if(len(img.shape)>2):
		height,width,channels = img.shape
	else:
		height,width = img.shape
		channels = 1

	print("Height = %d, Width = %d, Channels = %d" %(height, width, channels))
	
	if(channels < 3):
		print("***APPARENT GRAYSCALE IMAGE.  CONVERTING TO COLOR")
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

	crop_img = crop_largest_square(img)
	#print(crop_img.shape)
	
	img = cv2.resize(crop_img, (64,64))
	#print(img.shape)

	# OpenCV uses BGR ordering for color bands, much change for PyTorch
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	
	text = clip.tokenize(full_caption).to(device)
	with torch.no_grad():
		text_embeddings  = model.encode_text(text)

		pil_image = Image.fromarray(img)

		preprocess_image = preprocess(pil_image).unsqueeze(0).to(device)
		image_embeddings = model.encode_image(preprocess_image)

	#print(text_embeddings)
	#print(image_embeddings)

	data.append((fn, img, image_embeddings, text_embeddings))

	print("\n\n")


# Pickle the tensor
with open("data.pickle", 'wb') as f:
	pickle.dump(data, f)
