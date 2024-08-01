import torch
import torch.nn as nn

from ViT import Vit
from context_aggregator import ContextAggregator

class PerSec(nn.Module):
	def __init__(self, img_size, in_channels):
		super(PerSec, self).__init__()
		self.img_size = img_size
		self.stage1 = Vit(img_size=img_size, in_channels=in_channels, patch_size=7, emb_size=64, stride=(4, 4), padding=3, num_heads=1, mlp_ratio=4, reduction=4, qkv_dim=64, num_layers=2, dropout=0)
		self.replacement_low = nn.Parameter(torch.randn(64))
		self.stroke_context_aggregator = ContextAggregator(in_channel=64, num_heads=8, mlp_ratio=4, qkv_dim=128, num_layer=2, dropout=0, window=0.5)
		self.stage2 = Vit(img_size=(img_size[0] // 4, img_size[1] // 4), in_channels=64, patch_size=3, emb_size=128, stride=(2, 1), padding=1, num_heads=2, mlp_ratio=4, reduction=4, qkv_dim=128, num_layers=2, dropout=0)
		self.stage3 = Vit(img_size=(img_size[0] // 8, img_size[1] // 4), in_channels=128, patch_size=3, emb_size=320, stride=(2, 1), padding=1, num_heads=4, mlp_ratio=4, reduction=2, qkv_dim=320, num_layers=2, dropout=0)
		self.stage4 = Vit(img_size=(img_size[0] // 16, img_size[1] // 4), in_channels=320, patch_size=3, emb_size=512, stride=(2, 1), padding=1, num_heads=8, mlp_ratio=4, reduction=1, qkv_dim=512, num_layers=2, dropout=0)
		self.replacement_high = nn.Parameter(torch.randn(512))
		self.semantic_context_aggregator = ContextAggregator(in_channel=512, num_heads=8, mlp_ratio=4, qkv_dim=512, num_layer=2, dropout=0, window=10)
		self.lstm = nn.LSTM(512, 512, num_layers=2, bidirectional=True, batch_first=True)
		
	def _replace_mask(self, x, mask, new_value):
		if mask is None:
			return x
		B, M = mask.shape
		C, H, W = x.shape[1], x.shape[2], x.shape[3]

		h = mask // W
		w = mask % W

		b = torch.arange(B).view(B, 1).expand(-1, M).reshape(-1)

		h_flat = h.reshape(-1)
		w_flat = w.reshape(-1)

		x[b, :, h_flat, w_flat] = new_value.unsqueeze(0).expand(B*M, -1)
		return x

	def forward(self, x, mask_low=None, mask_high=None):
		x = self.stage1(x)
		# print("stage1", x.shape)
		x = self._replace_mask(x, mask_low, self.replacement_low)
		x = self.stroke_context_aggregator(x)
		# print("stroke_context_aggregator",x.shape)
		x = x.transpose(1, 2).reshape(x.shape[0], -1, self.img_size[0] // 4, self.img_size[1] // 4)
		x = self.stage2(x)
		x = self.stage3(x)
		x = self.stage4(x)
		# print("stage4", x.shape)
		x = self._replace_mask(x, mask_high, self.replacement_high)
		x = self.semantic_context_aggregator(x)
		# print("semantic_context_aggregator", x.shape)
		x, hidden = self.lstm(x)
		return x, hidden

class PerSecCNN(nn.Module):
	def __init__(self, img_size, in_channels):
		super(PerSecCNN, self).__init__()
		self.img_size = img_size
		self.relu = nn.ReLU()
		self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1)
		self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
		self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
		self.replacement_low = nn.Parameter(torch.randn(256))
		self.stroke_context_aggregator = ContextAggregator(in_channel=256, num_heads=8, mlp_ratio=4, qkv_dim=512, num_layer=2, dropout=0, window=0.5)
		self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
		self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
		self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
		self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
		self.conv6 = nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0)
		self.replacement_high = nn.Parameter(torch.randn(512))
		self.semantic_context_aggregator = ContextAggregator(in_channel=512, num_heads=8, mlp_ratio=4, qkv_dim=512, num_layer=2, dropout=0, window=10)
		self.lstm = nn.LSTM(512, 512, num_layers=2, bidirectional=True, batch_first=True)
	
	def _replacement_mask(self, x, mask, new_value):
		if mask is None:
			return x
		B, M = mask.shape
		C, H, W = x.shape[1], x.shape[2], x.shape[3]

		h = mask // W
		w = mask % W

		b = torch.arange(B).view(B, 1).expand(-1, M).reshape(-1)

		h_flat = h.reshape(-1)
		w_flat = w.reshape(-1)

		x[b, :, h_flat, w_flat] = new_value.unsqueeze(0).expand(B*M, -1)
		return x
	
	def forward(self, x, mask_low=None, mask_high=None):
		x = self.relu(self.conv1(x))
		x = self.maxpool1(x)
		x = self.relu(self.conv2(x))
		x = self.maxpool2(x)
		x = self.relu(self.conv3(x))
		x = self._replacement_mask(x, mask_low, self.replacement_low)
		x = self.stroke_context_aggregator(x)
		x = x.transpose(1, 2).reshape(x.shape[0], -1, self.img_size[0] // 4, self.img_size[1] // 4)
		x = self.relu(self.conv4(x))
		x = self.maxpool3(x)
		x = self.relu(self.conv5(x))
		x = self.maxpool4(x)
		x = self.relu(self.conv6(x))
		x = self._replacement_mask(x, mask_high, self.replacement_high)
		x = self.semantic_context_aggregator(x)
		x, hidden = self.lstm(x)
		return x, hidden