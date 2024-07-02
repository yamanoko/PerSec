import torch
import torch.nn as nn

from ViT import Vit
from context_aggregator import ContextAggregator

class PerSec(nn.Module):
	def __init__(self, img_size, in_channels):
		super(PerSec, self).__init__()
		self.stage1 = Vit(img_size=img_size, in_channels=in_channels, patch_size=7, emb_size=64, stride=(4, 4), padding=3, num_heads=1, mlp_ratio=4, reduction=4, qkv_dim=64, num_layers=2, dropout=0)
		self.stroke_context_aggregator = ContextAggregator(in_channel=64, emb_size=128, num_heads=8, mlp_ratio=4, qkv_dim=128, num_layer=2, dropout=0, range=0.5)
		self.stage2 = Vit(img_size=(img_size[0] // 4, img_size[1] // 4), in_channels=128, patch_size=3, emb_size=128, stride=(2, 1), padding=1, num_heads=2, mlp_ratio=4, reduction=4, qkv_dim=128, num_layers=2, dropout=0)
		self.stage3 = Vit(img_size=(img_size[0] // 8, img_size[1] // 4), in_channels=128, patch_size=3, emb_size=320, stride=(2, 1), padding=1, num_heads=4, mlp_ratio=4, reduction=2, qkv_dim=320, num_layers=2, dropout=0)
		self.stage4 = Vit(img_size=(img_size[0] // 16, img_size[1] // 4), in_channels=320, patch_size=3, emb_size=512, stride=(2, 1), padding=1, num_heads=8, mlp_ratio=4, reduction=1, qkv_dim=512, num_layers=2, dropout=0)
		self.semantic_context_aggregator = ContextAggregator(in_channel=512, emb_size=512, num_heads=8, mlp_ratio=4, qkv_dim=512, num_layer=2, dropout=0, range=10)

	def forward(self, x):
		x = self.stage1(x)
		x = self.stroke_context_aggregator(x)
		x = x.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.img_size[0] // 4, self.img_size[1] // 4)
		x = self.stage2(x)
		x = self.stage3(x)
		x = self.stage4(x)
		x = self.semantic_context_aggregator(x)
		return x