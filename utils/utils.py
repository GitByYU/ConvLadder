import torch
from torch.autograd import Variable
import torch.nn as nn


'''
make some noise for the encode imput data
'''


class Noise(nn.Module):
	def __init__(self,std=0.05):
		super(Noise, self).__init__()
		self.std = std
	def forward(self, x):
		shape = x.shape
		noise = Variable(torch.zeros(shape).cuda())
		return x + noise.data.normal_(0, std=self.std)

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
