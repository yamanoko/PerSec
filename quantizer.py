import torch
import torch.nn as nn

class GumbelVectorQuantizer(nn.Module):
	def __init__(self, extracted_feature_size, num_groups, num_vectors, temperature):
		super(GumbelVectorQuantizer, self).__init__()
		self.num_groups = num_groups
		self.num_vectors = num_vectors
		self.linear = nn.Linear(extracted_feature_size, num_groups * num_vectors)
		self.codebook = nn.Parameter(torch.randn(1, num_groups, num_vectors, extracted_feature_size // num_groups))
		self.temperature = temperature
	
	def forward(self, x):
		# x: [B, N, E]
		B, N, E = x.shape
		hidden_states = self.linear(x)
		hidden_states = hidden_states.view(B, N, self.num_groups, self.num_vectors)
		# hidden_states: [B, N, G, V]
		code_vector_prob = nn.functional.gumbel_softmax(hidden_states, tau=self.temperature, hard=True)
		# code_vector_prob: [B, N, G, V]
		code_vector_prob_copy = code_vector_prob.clone()
		code_vector_prob = code_vector_prob.view(B*N, self.num_groups, self.num_vectors)
		# code_vector_prob_compute: [B*N, G, V]
		code_vector_prob = code_vector_prob.unsqueeze(-1).expand(-1, -1, -1, self.codebook.shape[-1])
		# code_vector_prob_compute: [B*N, G, V, D]
		batched_codebook = self.codebook.clone().expand(B*N, -1, -1, -1)
		# batched_codebook: [B*N, G, V, D]
		code_vectors = code_vector_prob * batched_codebook
		# code_vectors: [B*N, G, V, D]
		code_vectors = code_vectors.sum(dim=2)
		# code_vectors: [B*N, G, D]
		code_vector = code_vectors.view(B, N, -1)
		# code_vector: [B, N, E]

		code_vector_prob_copy = code_vector_prob_copy.transpose(1, 2)
		# code_vector_prob: [B, G, N, V]
		code_vector_prob_copy = code_vector_prob_copy.sum(dim=2)
		# code_vector_prob: [B, G, V]
		code_vector_prob_copy = torch.softmax(code_vector_prob_copy, dim=-1)
		entropy = - (code_vector_prob_copy * torch.log(code_vector_prob_copy + 1e-8)).sum() / (self.num_groups * self.num_vectors)
		# entropy: []

		return code_vector, entropy
