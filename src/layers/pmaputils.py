import torch
import torch.functional
from torch.autograd import Function,Variable
import torch.nn
from torch.distributions import Multinomial
from torch import Tensor
import math


class LogSumExp(Function):
	''' Takes log sum exp along axis 1'''
	@staticmethod
	def forward(ctx, input:torch.Tensor,axis,*args):
		input = input.detach()
		maxval = input.max(dim=axis, keepdim=True)[0]
		logsumexp = (input - maxval).exp().sum(dim=axis, keepdim=True).log() + maxval
		ctx.save_for_backward(input, logsumexp, Variable(input.new_tensor(axis, dtype=torch.int32)))
		return logsumexp
	@staticmethod
	def backward(ctx, grad_output:torch.Tensor):
		grad_output = grad_output.detach()
		input,lognorm,axis = ctx.saved_tensors
		prob = (input - lognorm).exp()
		grad_input = grad_output * prob
		return grad_input,None,None



def sample(lp:Tensor,axis=1,numsamples=1,MAP=False):
	lastaxis = lp.ndimension() -1
	lpt = lp.transpose(lastaxis,axis)
	M = Multinomial(total_count=numsamples,logits=lpt)
	#D = Dirichlet((lp.exp())*(numsamples.float())/(lp.size(lastaxis)))
	samps = M.sample().detach()
	samps = samps.transpose(lastaxis,axis)/numsamples
	logprob = (lp-(samps.detach()).log())
	logprob[logprob!=logprob] = float('Inf')
	logprob = logprob.min(dim=axis,keepdim=True)[0]

	return None,None

def sample_manual( lp: Tensor, axis=1, manualrand=None):
	lp = lp.transpose(0, axis)
	p = lp.exp() #type:Tensor
	cumprob = p.cumsum(dim=0)

	if manualrand is not None:
		rand = manualrand
		rand = rand.transpose(axis, 0)
	else:
		rand = torch.rand_like(p[0:1])
	samps = cumprob >= rand
	samps[1:] = samps[1:] ^ samps[0:-1]
	samps = samps.type_as(p).detach()
	logprob = samps * lp
	logprob[logprob != logprob] = 0
	logprob = logprob.sum(dim=0, keepdim=True)
	samps = samps.transpose(0, axis)
	logprob = logprob.transpose(0, axis)
	return samps.detach(), logprob
def logminusexp_deprecated(l1:Tensor,l2:Tensor):
	flag = ((l1-l2)>=0).all()
	# if not flag:
	# 	print("SHIT")
	# 	print(l1-l2)
	# 	print(l1)
	# 	print(l2)
	# 	input("press to continue")
	# 	pass
	#assert (l1>=l2).all()
	out = (1.0- (l2-l1).exp()).log() + l1
	out[(1-(l2-l1).exp())<1e-4]= -100000
	return out


def max_lprob(mother,model,dim):
	lrate = (mother - model)
	max_lrate= lrate.min(dim=dim , keepdim=True)[0]
	max_logical_ind = (lrate == max_lrate).float()

	max_logical_ind = max_logical_ind / max_logical_ind.sum(dim=dim,keepdim=True)
	max_lprob = (max_logical_ind * lrate).sum(dim=dim,keepdim=True)
	return max_lprob
def max_correction(tensor,dim):
	max = tensor.max(dim=dim,keepdim=True)[0]
	max_inds = (tensor == max).float()
	max_inds = max_inds/max_inds.sum(dim=dim,keepdim=True)
	max = (tensor*max_inds)
	max[max != max] = float(0)
	max = max.sum(dim=dim, keepdim=True)
	return max
def softplus(tensor:Tensor)->Tensor:
	''' ln(1+ e^t)'''
	max = tensor.clamp_min(0)
	ret = ((-max).exp()+ (tensor-max).exp()).log()
	ret = ret + max
	return ret
def logsigmoid(inputs:Tensor)->Tensor:
	return -softplus(-inputs)
def min_correction(tensor,dim):
	min = -max_correction(-tensor,dim)
	return min
def sample_liklihood(lp,axis=1,numsamples=1):
	lastaxis = lp.ndimension() - 1
	lporig = lp
	lpunif = torch.zeros_like(lp)
	lpunif = lp.exp() * 0 - (lp.exp() * 0).logsumexp(dim=1, keepdim=True)
	samplinglp = lpunif
	lpt = samplinglp.transpose(lastaxis, axis)
	M = Multinomial(total_count=numsamples, logits=lpt)
	samps = M.sample().detach()
	samps = samps.transpose(lastaxis, axis) / numsamples
	logprob = (lporig - (samps.detach()).log())
	logprob[logprob != logprob] = float('Inf')
	logprob = logprob.min(dim=axis, keepdim=True)[0]

	lpmodel = min_correction(lpunif - lporig, axis)

	return samps.detach(), logprob, lpmodel

def softmin(lp,axis,coef=1):
	min = -(-lp*coef).logsumexp(dim=axis,keepdim=True)/coef
	return min
def softmin_pair(lp1,lp2):
	return -LSE_pair(-lp1,-lp2)
def softmax_pair(lp1,lp2):
	return LSE_pair(lp1,lp2)

def LSE_pair(lp1,lp2):
	m = torch.max(lp1,lp2)
	lp1 = lp1- m
	lp2 = lp2 - m
	ret = (lp1.exp() + lp2.exp()).log()
	ret += m
	return ret
def renyi_prob(lq,lp, alpha):
	divg = (alpha*lp - (alpha-1)*lq).logsumexp(dim=1,keepdim=True)/(alpha-1)
	return -divg
def sampleunif(lp:Tensor,axis=1,numsamples=1):
	''' Samples from the random variables uniformly
	A model is given in the probability space with logit vector lp
	The probability that the sample is in the model is calculated.

	'''
	lastaxis = lp.ndimension() -1
	lporig = lp
	lpunif = torch.zeros_like(lp)
	lpunif = lpunif - (lpunif).logsumexp(dim=1,keepdim=True)
	lpt = lpunif.transpose(lastaxis,axis)
	M = Multinomial(total_count=numsamples,logits=lpt)
	samps = M.sample().detach()
	samps = samps.transpose(lastaxis,axis)/numsamples
	logprob = (lporig-(samps.detach()).log())
	logprob[logprob!=logprob] = float('Inf')
	logprob = logprob.min(dim=axis,keepdim=True)[0]
	# lpmodel = (lpunif-lporig).min(dim=axis,keepdim=True)[0]
	# TODO min
	lpmodel = softmin(lpunif-lporig,axis)
	# lpmodel= (lpunif-lporig).min(dim=1,keepdim=True)[0]# -  float(lporig.shape[1])
	# lpmodel = renyi_prob(lpunif,lporig,1)
	inmodel_lprobs = logprob + lpmodel - lpunif.mean(dim=1, keepdim=True)  # - max_correction(-lporig, axis)
	return None, None, None

def sample_mine(lp: Tensor, axis=1,numsamples=1):
	p = lp.exp()
	cumprob = p.cumsum(axis)
	sh = p.size()
	if axis == 1:
		rand = p.new_empty(sh[0], 1, sh[2], sh[3]).uniform_(0, 1)
	# np_rand = np.random.uniform(low=0.0,high=1.0,size=(sh[0],1,sh[2],sh[3]))
	else:
		rand = p.new_empty(sh[0], sh[1], sh[2], sh[3], 1).uniform_(0, 1)
	# np_rand = np.random.uniform(low=0.0, high=1.0, size=(sh[0], sh[1], sh[2], sh[3],1))
	samps = cumprob > rand
	if axis == 1:
		samps[0:, 1:, 0:, 0:] = samps[0:, 1:, 0:, 0:] ^ samps[0:, 0:-1, 0:, 0:]
	elif axis == 4:
		samps[0:, 0:, 0:, 0:, 1:] = samps[0:, 0:, 0:, 0:, 1:] ^ samps[0:, 0:, 0:, 0:, 0:-1]
	else:
		raise (Exception(
			'Only axis=1 and axis=4(for binary distributions) is acceptable ' + 'axis=%d' % axis + ' was given'))
	samps = samps.type_as(p)
	return samps.detach(),None
def sampleprob_mine(p: Tensor, axis=1,numsamples=1):
	cumprob = p.cumsum(axis)
	sh = p.size()
	if axis == 1:
		rand = p.new_empty(sh[0], 1, sh[2], sh[3]).uniform_(0, 1)
	# np_rand = np.random.uniform(low=0.0,high=1.0,size=(sh[0],1,sh[2],sh[3]))
	else:
		rand = p.new_empty(sh[0], sh[1], sh[2], sh[3], 1).uniform_(0, 1)
	# np_rand = np.random.uniform(low=0.0, high=1.0, size=(sh[0], sh[1], sh[2], sh[3],1))
	samps = cumprob > rand
	if axis == 1:
		samps[0:, 1:, 0:, 0:] = samps[0:, 1:, 0:, 0:] ^ samps[0:, 0:-1, 0:, 0:]
	elif axis == 4:
		samps[0:, 0:, 0:, 0:, 1:] = samps[0:, 0:, 0:, 0:, 1:] ^ samps[0:, 0:, 0:, 0:, 0:-1]
	else:
		raise (Exception(
			'Only axis=1 and axis=4(for binary distributions) is acceptable ' + 'axis=%d' % axis + ' was given'))
	samps = samps.type_as(p)
	return samps.detach(),None

def sampleprob(p:Tensor,axis:int):
	samps, logprob = sample_manual(p.log(),axis=axis)
	return samps,logprob


def sample_exp(p:Tensor):
	statenum = float(p.shape[1])
	bitnum = -(torch.rand_like(p[0:,0:1,0:1,0:1]).log()/math.log(2)).floor()

	torch.rand_like(p).log()/math.log(float(p.shape[1]))

	return None


def sample_maxprob(p,dum1,dum2):
	statenum = float(p.shape[1])
	print(p.sum(dim=1).mean().item())
	sample = (((p/p.sum(dim=1,keepdim=True))*statenum) >=1).float()
	#print(sample.sum(dim=1).mean().item())
	return sample,None
def sample_map(lp,axis,numsamps):
	maxlp,maxind = lp.max(dim=1,keepdim=True)
	boolind = (lp == maxlp).float()
	newlprob = (boolind/boolind.sum(dim=axis,keepdim=True)).log()
	samp,_ = sample(newlprob,axis=1,numsamples=numsamps)
	logprob = samp*lp
	logprob[logprob!=logprob] = 0
	return samp,logprob.sum(dim=axis,keepdim=True)

