import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import normalize

class ContrastiveLoss(nn.Module):
	def __init__(self, temperature=0.07):
		super(ContrastiveLoss, self).__init__()
		self.temperature = temperature

	def forward(self, x, y, mask_index):
		# x: (B, N, D), y: (B, N, D), mask_index: (B, M)
		normalized_x = normalize(x, dim=-1)
		normalized_y = normalize(y, dim=-1)
		mask = mask_index.unsqueeze(-1).expand(-1, -1, x.size(-1))
		# mask: (B, M, D)
		normalized_x = torch.gather(normalized_x, 1, mask)
		# normalized_x: (B, M, D), normalized_y: (B, N, D)
		cos_similarities = torch.matmul(normalized_y, normalized_x.transpose(1, 2)) / self.temperature
		# cos_similarities: (B, N, M)
		loss = CrossEntropyLoss()(cos_similarities, mask_index)
		return loss
