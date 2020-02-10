import torch
from src.layers.pmaputils import sampleprob
from torch import Tensor as Tensor
from torch.utils.data.dataset import Dataset
import pandas as pd
import typing


class SyntheticData(Dataset):
	def __init__(self,data,*args,**kwargs):
		'''

		:param inputdim: number of input states
		:param outputdim: number of target states
		:param numsamples: self evident
		:param args:
		:param kwargs:
		'''
		super(SyntheticData,self).__init__(*args,**kwargs)
		self.data = data
		return

	def __getitem__(self, item):
		x = self.data[0][item,0:,0:,0:]
		y = self.data[1][item]
		return x.log(),y

	def __len__(self):
		return self.data[0].shape[0]


class Joint:

	def __init__(self,inputdim,outputdim):
		self.inputdim = inputdim
		self.outputdim = outputdim
		self.jointdist = self._create_oracle()

	def _create_oracle(self):

		joint = -torch.rand(self.inputdim, self.outputdim).log()
		joint = joint / joint.sum()
		px  = (joint*0+1).sum(dim=1,keepdim=True)
		px = px/px.sum()
		pygx =  joint / joint.sum(dim=1,keepdim=True)
		pygx = sampleprob(pygx,1)[0]
		joint = pygx * px
		return joint.detach()

	def create_dataset(self, numsamples ) -> SyntheticData:
		joint = self.jointdist
		sh = joint.shape
		jointvec = joint.reshape((1, joint.shape[0] * joint.shape[1], 1,1))  # type: Tensor
		samples= jointvec.repeat([numsamples, 1, 1, 1])
		samples = sampleprob(samples, 1)[0].view([numsamples, sh[0], sh[1], 1,1])
		xsamples = samples.sum(dim=2, keepdim=True)
		ysamples = samples.sum(dim=1, keepdim=True)
		ysamples = ysamples.max(dim=2)[1].squeeze()
		return SyntheticData((xsamples.clone(), ysamples.clone()))
