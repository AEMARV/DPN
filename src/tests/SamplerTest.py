import torch
from src.layers.klmodules import Sampler


s = Sampler(blockidx=-1)

lp = torch.rand(1,10,1,1,1).log().to('cuda:0')
lp= lp.log_softmax(dim=1)

lp = torch.rand(1,10,1,1,1).log().to('cuda:0')
lp= lp.log_softmax(dim=1)

empirical = 0
count = 0
iters= 1000000
for i in range(iters):

	samp , logprob, modelprob = s(lp,0,None)
	count+= samp

	empirical = count/(i+1)
	deviation = (empirical - lp.exp()).abs().sum()
	l2norm = ((empirical - lp.exp())**2).sum().sqrt()
	maxprob = (lp - empirical.log()).min(dim=1)[0]


	print("Deviation= {} |".format(deviation.item())+
	      "L2Norm = {}  |".format(l2norm.item())+
	      "MaxLogProb = {}\r".format(maxprob.item()))




