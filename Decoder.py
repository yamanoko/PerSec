import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
	def __init__(self, hidden_size):
		super(Attention, self).__init__()
		self.hidden_size = hidden_size
	
	def forward(self, hidden, encoder_outputs):
		# hidden: (Num_layers * Num_directions, Batch_size, Hidden_size)
		# encoder_outputs: (Batch_size, Seq_len, Hidden_size)
		hidden = hidden.mean(dim=0).unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
		# hidden: (Batch_size, Seq_len, Hidden_size)
		attn_weights = F.softmax(torch.sum(hidden * encoder_outputs, dim=2), dim=1).unsqueeze(1)
		# attn_weights: (Batch_size, 1, Seq_len)
		attn_weighted = torch.bmm(attn_weights, encoder_outputs)
		# context: (Batch_size, 1, Hidden_size)
		return attn_weighted

class LSTMAttnDecoder(nn.Module):
	def __init__(self, hidden_size, output_size, dropout=0.1):
		super(LSTMAttnDecoder, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size

		self.embedding = nn.Embedding(output_size, hidden_size)
		self.dropout = nn.Dropout(dropout)
		self.attention = Attention(hidden_size)
		self.lstm = nn.LSTM(hidden_size * 2, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
		self.out = nn.Linear(hidden_size * 2, output_size)

	def forward(self, input, last_hidden, encoder_outputs):
		# input: (Batch_size, 1)
		embedded = self.embedding(input)
		embedded = self.dropout(embedded)
		# embedded: (Batch_size, 1, Hidden_size)
		attn_weighted = self.attention(last_hidden, encoder_outputs)
		# attn_weighted: (Batch_size, 1, Hidden_size)
		rnn_input = torch.cat((embedded, attn_weighted), dim=2)
		output, hidden = self.lstm(rnn_input, last_hidden)
		# output: (Batch_size, 1, Hidden_size * 2)
		# hidden: (Num_layers * Num_directions, Batch_size, Hidden_size)
		output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
		output = output.squeeze(1)
		# output: (Batch_size, Hidden_size)
		context = attn_weighted.squeeze(1)
		# context: (Batch_size, Hidden_size)
		output = self.out(torch.cat((output, context), dim=1))
		# output: (Batch_size, Output_size)
		return output, hidden
