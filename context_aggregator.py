import torch
import torch.nn as nn

class WMultiHeadAttention(nn.Module):
	def __init__(self, emb_size, num_heads, qkv_dim, dropout, range=1.0):
		super(WMultiHeadAttention, self).__init__()
		self.num_heads = num_heads
		self.q_linear = nn.Linear(emb_size, qkv_dim)
		self.k_linear = nn.Linear(emb_size, qkv_dim)
		self.v_linear = nn.Linear(emb_size, qkv_dim)
		self.fc = nn.Linear(qkv_dim, emb_size)
		self.dropout = nn.Dropout(dropout)
		self.range = range

	def forward(self, x):
		# x: [B, N, E]
		B, N, E = x.shape
		if self.range <= 1.0:
			W = self.range * N
		else:
			W = self.range
		M = torch.full((N, N), float('-inf'), device=x.device)
		window_indices = torch.abs(torch.arange(N).unsqueeze(0) - torch.arange(N).unsqueeze(1)) <= W // 2
		window_indices.to(x.device)
		M[window_indices] = 0

		q = self.q_linear(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
		# q: [B, H, N, D]
		k = self.k_linear(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
		# k: [B, H, N, D]
		v = self.v_linear(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
		# v: [B, H, N, D]
		attn = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5) + M
		attn = attn.softmax(dim=-1)
		x = attn @ v
		# x: [B, H, N, D]
		x = x.transpose(1, 2).reshape(B, N, -1)
		x = self.fc(x)
		x = self.dropout(x)
		# x: [B, N, E]
		return x

class VitEncoderWithWMHSA(nn.Module):
	def __init__(self, emb_size, num_heads, mlp_ratio, qkv_dim, dropout, range=1.0):
		super(VitEncoderWithWMHSA, self).__init__()
		self.ln1 = nn.LayerNorm(emb_size)
		self.ln2 = nn.LayerNorm(emb_size)
		self.mh_attention = WMultiHeadAttention(emb_size, num_heads, qkv_dim, dropout, range)
		self.mlp = nn.Sequential(
			nn.Linear(emb_size, int(emb_size * mlp_ratio)),
			nn.GELU(),
			nn.Linear(int(emb_size * mlp_ratio), emb_size),
			nn.Dropout(dropout)
		)

	def forward(self, x):
		# x: [B, N, E]
		x = x + self.mh_attention(self.ln1(x))
		x = x + self.mlp(self.ln2(x))
		# x: [B, N, E]
		return x


class VitInputWith2DPositionEncoding(nn.Module):
	def __init__(self, in_channels, emb_size):
		super(VitInputWith2DPositionEncoding, self).__init__()
		self.dw_conv = nn.Conv2d(in_channels, emb_size, kernel_size=3, stride=1, padding=1, groups=in_channels)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		# x: [B, C, H, W]
		x = x * self.sigmoid(self.dw_conv(x))
		x = x.flatten(2)
		x = x.transpose(1, 2)
		# x: [B, N * N, E]
		return x

class ContextAggregator(nn.Module):
	def __init__(self, in_channel, emb_size, num_heads, mlp_ratio, qkv_dim, num_layer, dropout, range=1.0):
		super(ContextAggregator, self).__init__()
		self.vit_input = VitInputWith2DPositionEncoding(in_channel, emb_size)
		self.layers = nn.Sequential(*[VitEncoderWithWMHSA(emb_size, num_heads, mlp_ratio, qkv_dim, dropout, range) for _ in range(num_layer)])

	def forward(self, x):
		# x: [B, C, H, W]
		x = self.vit_input(x)
		x = self.layers(x)
		# x: [B, N * N, E]
		return x
