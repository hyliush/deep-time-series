import torch.nn as nn
import torch
import torch.nn.functional as F

class Swish(nn.Module):
	def __init__(self):
		super(Swish, self).__init__()

	def forward(self, x):
		return x * torch.sigmoid(x)
		
class SoftRelu(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return torch.log(torch.tensor(1.0) + torch.exp(x))

class Relu(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return torch.relu(x)

class Gelu(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return F.gelu(x)