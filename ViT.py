import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
	def __init__(self, in_channels, patch_size, emb_size, stride, padding):
		super(PatchEmbedding, self).__init__()
		self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=stride, padding=padding)

	def forward(self, x):
		# x: [B, C, H, W]
		x = self.projection(x)
		# x: [B, E, N, N], N = H // patch_size = W // patch_size
		x = x.flatten(2)
		# x: [B, E, N * N]
		x = x.transpose(1, 2)
		# x: [B, N * N, E]
		return x

class VitInput(nn.Module):
	def __init__(self, img_size, in_channels, patch_size, emb_size, stride, padding):
		super(VitInput, self).__init__()
		h, w = img_size
		out_h = (h - patch_size + 2 * padding) // stride[0] + 1
		out_w = (w - patch_size + 2 * padding) // stride[1] + 1
		self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, stride, padding)
		self.pos_embedding = nn.Parameter(torch.randn(1, out_h*out_w, emb_size))

	def forward(self, x):
		# x: [B, C, H, W]
		x = self.patch_embedding(x)
		x = x + self.pos_embedding
		# x: [B, N * N, E]
		return x

class SpatialReductionAttention(nn.Module):
	def __init__(self, emb_size, num_heads, reduction, qkv_dim, drop_out):
		super(SpatialReductionAttention, self).__init__()
		self.num_head = num_heads
		self.reduction = reduction
		self.q_linear = nn.Linear(emb_size, qkv_dim)
		self.k_linear = nn.Linear(emb_size * (reduction ** 2), qkv_dim)
		self.v_linear = nn.Linear(emb_size * (reduction ** 2), qkv_dim)
		self.ln = nn.LayerNorm(qkv_dim // num_heads)
		self.dropout = nn.Dropout(drop_out)
		self.fc = nn.Linear(qkv_dim, emb_size)

	def forward(self, x):
		B, N, E = x.shape
		q = self.q_linear(x).reshape(B, N, self.num_head, -1).permute(0, 2, 1, 3)
		# q: [B, H, N, D], D=qkv_dim//num_heads
		k = x.reshape(B, N // (self.reduction ** 2), -1)
		# k: [B, N // (r^2), E*r^2]
		k = self.k_linear(k).reshape(B, N // (self.reduction ** 2), self.num_head, -1).permute(0, 2, 1, 3)
		# k: [B, H, N // (r^2), D]
		k = self.ln(k)
		# k: [B, H, N // (r^2), D]
		v = x.reshape(B, N // (self.reduction ** 2), -1)
		v = self.v_linear(v).reshape(B, N // (self.reduction ** 2), self.num_head, -1).permute(0, 2, 1, 3)
		v = self.ln(v)
		# v: [B, H, N // (r^2), D]
		attn = (q @ k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
		# attn: [B, H, N, N // (r^2)]
		attn = attn.softmax(dim=-1)
		x = attn @ v
		# x: [B, H, N, D]
		x = x.transpose(1, 2).reshape(B, N, -1)
		x = self.fc(x)
		x = self.dropout(x)
		# x: [B, N, E]
		return x

class VitEncoder(nn.Module):
	def __init__(self, emb_size, num_heads, mlp_ratio, reduction, qkv_dim, dropout):
		super(VitEncoder, self).__init__()
		self.ln1 = nn.LayerNorm(emb_size)
		self.attn = SpatialReductionAttention(emb_size, num_heads, reduction, qkv_dim, dropout)
		self.ln2 = nn.LayerNorm(emb_size)
		hidden_dim = int(emb_size * mlp_ratio)
		self.mlp = nn.Sequential(
			nn.Linear(emb_size, hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, emb_size),
			nn.Dropout(dropout)
		)

	def forward(self, x):
		# x: [B, N, E]
		x = x + self.attn(self.ln1(x))
		x = x + self.mlp(self.ln2(x))
		# x: [B, N, D]
		return x

class Vit(nn.Module):
	def __init__(self, img_size, in_channels, patch_size, emb_size, stride, padding, num_heads, mlp_ratio, reduction, qkv_dim, dropout, num_layers):
		super(Vit, self).__init__()
		self.h = img_size[0] // stride[0]
		self.vit_input = VitInput(img_size, in_channels, patch_size, emb_size, stride, padding)
		self.layers = nn.Sequential(*[VitEncoder(emb_size, num_heads, mlp_ratio, reduction, qkv_dim, dropout) for _ in range(num_layers)])
	
	def forward(self, x):
		# x: [B, C, H, W]
		x = self.vit_input(x)
		x = self.layers(x)
		# x: [B, N * N, D]
		x = x.transpose(1, 2)
		# x: [B, D, N * N]
		x = x.reshape(x.shape[0], x.shape[1], self.h, -1)
		# x: [B, D, H, W]
		return x

