import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import Compose, Lambda, RandomApply, GaussianBlur, RandomRotation, Grayscale, Resize, ToTensor
from torch.utils.data import Dataset
from PIL import Image

import os

class ContrastiveLearningDataset(Dataset):
	def __init__(self, img_dir, img_size=(32, 384)):
		super().__init__()
		self.img_size = img_size
		assert os.path.isdir(img_dir), f"{img_dir} is not a valid directory"
		self.filepaths = [
			os.path.join(img_dir, filename) for filename in os.listdir(img_dir) if filename.endswith((".jpg", ".jpeg", ".png"))
		]
		if not self.filepaths:
			raise FileNotFoundError(f"No images found in {img_dir}")
		self.transform = Compose([
			Lambda(lambda img: img.convert("RGB")),
			RandomApply([GaussianBlur(kernel_size=3)], p=0.3),
			RandomRotation(degrees=3.5, fill=255),
			Grayscale(num_output_channels=3),
			Resize(img_size),
			ToTensor()
		])
	
	def __len__(self):
		return len(self.filepaths)
	
	def __getitem__(self, idx):
		try:
			img = Image.open(self.filepaths[idx])
		except IOError:
			raise FileNotFoundError(f"Could not read image at {self.filepaths[idx]}")
		img = self.transform(img)
		return img


class DecoderDataset(Dataset):
	def __init__(self, csv_path, img_dir, token_dict, img_size=(32, 384), max_length=20):
		super().__init__()
		self.img_dir = img_dir
		self.annotation = pd.read_csv(csv_path)
		self.token_dict = token_dict
		self.max_length = max_length
		self.transform = Compose([
			Lambda(lambda img: img.convert("RGB")),
			RandomApply([GaussianBlur(kernel_size=3)], p=0.3),
			RandomRotation(degrees=3.5, fill=255),
			Grayscale(num_output_channels=3),
			Resize(img_size),
			ToTensor()
		])
		assert len(self.annotation) == len(os.listdir(img_dir)), "Number of images and annotations do not match"
	
	def __len__(self):
		return len(self.annotation)
	
	def __getitem__(self, idx):
		img_path = os.path.join(self.img_dir, self.annotation.iloc[idx, 1])
		try:
			img = Image.open(img_path)
		except IOError:
			raise FileNotFoundError(f"Could not read image at {img_path}")
		img = self.transform(img)
		label = self.annotation.iloc[idx, 0]
		label_tokenized = [
			self.token_dict[char.lower()] 
			if char.lower() in self.token_dict 
			else self.token_dict["<UNK>"] 
			for char in label
		]
		label_tokenized = label_tokenized[:self.max_length]
		label_tokenized.append(self.token_dict["<EOS>"])
		label_length = len(label_tokenized)
		for _ in range(label_length, self.max_length+1):
			label_tokenized.append(self.token_dict["<PAD>"])
		return img, torch.tensor(label_tokenized, dtype=torch.long)
