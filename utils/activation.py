import torch.nn as nn
import torch

class Swish(nn.Module):
	def __init__(self):
		super(Swish, self).__init__()

	def forward(self, x):
		return x * torch.sigmoid(x)