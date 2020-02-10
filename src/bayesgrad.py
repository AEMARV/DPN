import torch
from src.layers.pmaputils import sample
from src.layers.Initializers import *
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
sns.set(color_codes=True)
# PARAMS
def samplers(fnum,inpsz,batchsz,trials,option,layer):
	input = torch.rand(batchsz,inpsz,1,1).to(device='cuda:0')
	input = input.log()
	dzdout = torch.rand(batchsz,fnum,1,1).log().to(device='cuda:0')
	dzdout = sample(dzdout,1,1)
	input.requires_grad = True
	if layer is None:
		init = LogParameter(isstoch=False,isuniform=False,isdirichlet=False)
		layer = BayesFunc(fnum=fnum,
				               kersize=1,
				               inp_chan_sz=inpsz,
				               isbiased=True,
				               isrelu=False,
				               biasinit=init,
				               padding='valid',
				               stride=1,
				               paraminit=init,
				               isstoch=False,
				               coefinit=1,
							   islast=False,
				               blockidx=0)
	layer.to('cuda:0')
	with torch.enable_grad():
		layer.exact = True
		output = layer(input)
		dzdxexact, = torch.autograd.grad(output, input, grad_outputs=dzdout, only_inputs=True)
	dzdxstoch= 0
	layer.exact = False
	layer.option = option

	for i in range(trials):
		with torch.enable_grad():

			output = layer(input)
			dzdxstochtemp, = torch.autograd.grad(output, input, grad_outputs=dzdout, only_inputs=True)
		dzdxstoch += dzdxstochtemp

	dzdxstoch /= trials
	return dzdxexact, dzdxstoch,layer

if __name__ == '__main__':
	trials = 1000
	inputdim = 10
	outputdim =5
	batchnum=1
	samples = 100
	probs_reject = []
	probs_mile = []
	layer = None
	for i in range(samples):
		print(str((i + 1) / samples * 100) + '\%', end='\r')
		dzdxexact, dzdxstoch,layer = samplers(outputdim,inputdim,batchnum,trials,0,layer)
		probability =(dzdxexact / dzdxstoch).min(dim=1, keepdim=True)[0].log().sum().exp().item()
		probs_reject = probs_reject + [probability]

	for i in range(samples):
		print(str((i + 1) / samples * 100) + '\%', end='\r')
		dzdxexact, dzdxstoch,layer = samplers(outputdim, inputdim, batchnum, trials, 1,layer)
		probability =(dzdxexact / dzdxstoch).min(dim=1, keepdim=True)[0].log().sum().exp().item()
		probs_mile = probs_mile + [probability]
	sns.distplot(probs_reject,color='r')
	sns.distplot(probs_mile,color='b')
	print(np.mean(probs_reject))
	print(np.mean(probs_mile))
	plt.show()
	input()
