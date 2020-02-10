import numpy as np
import torch
import torch.functional
import torch.nn.functional as F
from torch.autograd import Function,Variable
import torch.nn
from typing import List,Tuple,Dict,Any
from torch.distributions import Multinomial
from torch.distributions import Dirichlet
from torch import Tensor
from src.layers.pmaputils import *
import argparse



class Alpha_Lnorm(Function):
	@staticmethod
	def forward(ctx: Any,input: Any,dim,alpha, **kwargs: Any):
		norm_input = input.log_softmax(dim=dim)
		ctx.save_for_backward(norm_input)
		ctx.constant = (dim,alpha)
		return norm_input
	@staticmethod
	def backward(ctx: Any, grad_outputs: Any):
		ninput, = ctx.saved_tensors

		dim,alpha = ctx.constant
		g = grad_outputs - ((grad_outputs.sum(dim=dim,keepdim=True))*((ninput*alpha).softmax(dim=dim)))
		return g,None,None


class Alpha_LRnorm(Function):
	@staticmethod
	def forward(ctx: Any,input: Any,dim,alpha, **kwargs: Any):
		norm_input = input - softplus(input.logsumexp(dim=dim,keepdim=True))
		ctx.save_for_backward(norm_input)
		ctx.constant = (dim,alpha)
		return norm_input
	@staticmethod
	def backward(ctx: Any, grad_outputs: Any):
		ninput, = ctx.saved_tensors

		dim,alpha = ctx.constant
		g = grad_outputs - ((grad_outputs.sum(dim=dim,keepdim=True))*((ninput*alpha).softmax(dim=dim)))
		return g,None,None


class IdentityReg(Function):
	''' Input, dim , alpha'''
	@staticmethod
	def forward(ctx: Any,input: Any,dim,alpha, **kwargs: Any):
		ctx.save_for_backward(input)
		ctx.constant = (dim,alpha)
		return input
	@staticmethod
	def backward(ctx: Any, grad_outputs: Any):
		input, = ctx.saved_tensors

		dim,alpha = ctx.constant
		prob1 = input.sigmoid()
		# isone = (torch.rand_like(prob1)<prob1).float()
		alpha_prob1 = (alpha*input).sigmoid()
		reg_grad = alpha_prob1 - prob1
		reg_grad = reg_grad/(input.shape[0])
		g = grad_outputs + reg_grad
		return g,None,None


class Alpha_Lnorm_stoch(Function):
	@staticmethod
	def forward(ctx: Any,input: Any,dim,alpha, **kwargs: Any):
		norm_input = input.log_softmax(dim=dim)
		ctx.save_for_backward(norm_input)
		ctx.constant = (dim,alpha)
		return norm_input
	@staticmethod
	def backward(ctx: Any, grad_outputs: Any):
		ninput, = ctx.saved_tensors
		dim,alpha = ctx.constant
		alphaprob = (ninput*alpha).log_softmax(dim=dim)
		samp,_= sample_manual(alphaprob,dim)
		g = grad_outputs - ((grad_outputs.sum(dim=dim,keepdim=True))*(samp))
		return g,None,None

class LogSumExpStoch(Function):
	@staticmethod
	def forward(ctx: Any,input: Any,dim, **kwargs: Any):
		output = input.logsumexp(dim=dim,keepdim=True)
		ctx.save_for_backward(input)
		ctx.constant = (dim)
		return output
	@staticmethod
	def backward(ctx: Any, grad_outputs: Any):
		input, = ctx.saved_tensors
		dim = ctx.constant
		alphaprob = (input).log_softmax(dim=dim)
		samp,_= sample_manual(alphaprob,dim)
		g = ((grad_outputs.sum(dim=dim,keepdim=True))*(samp))
		return g,None,None

class LogMinusExp(Function):
	@staticmethod
	def forward(ctx: Any, l1, l2, **kwargs: Any):
		y = l1 + ((1 - (l2-l1).exp()).relu()).log()
		ctx.save_for_backward(l1,l2)
		return y
	@staticmethod
	def backward(ctx: Any, grad_out):
		l1,l2 = ctx.saved_tensors
		denom = (1-(l2-l1).exp()).relu()
		dl1 = 1/denom
		dl2= (l2-l1).exp()/denom
		dl1 = grad_out*dl1
		dl2 = grad_out*dl2

		dl1[dl1!=dl1] = 1
		dl2[dl2!=dl2] = -1
		return dl1, dl2

class LogSumExpPair(Function):
	@staticmethod
	def forward(ctx: Any, l1, l2, **kwargs: Any):
		l1.detach_()
		l2.detach_()
		m = torch.max(l1,l2)
		y = m + ((l1-m).exp() + (l2-m).exp()).log()
		y[m == -float("inf")] = -float('inf')
		if (y != y).sum() > 0:
			print("FOrward NAN")
			print("L1:")
			print(l1)
			print("L2:")
			print(l2)
			print(y)
			input("PRESS TO CONITNUE")
		ctx.save_for_backward(l1, l2,y,m)
		return y

	@staticmethod
	def backward(ctx: Any, grad_out):
		l1, l2,y,m = ctx.saved_tensors
		dl1 = (l1-y).exp()
		dl2 = (l2 - y).exp()

		dl1[dl1 != dl1]= 0.5
		dl2[dl2 !=dl2] =0.5

		dl1 = grad_out * dl1
		dl2 = grad_out * dl2
		if (dl1!=dl1).sum()>0  or (dl2!=dl2).sum()>0:
			print("HASNAN")
			print(dl1)
			print(dl2)
			print(l1)
			print(l2)
			print(y)
			print(m)
		return dl1, dl2
''' Aliases'''
lsepair= LogSumExpPair.apply
idreg= IdentityReg.apply
alpha_lnorm = Alpha_Lnorm.apply
alpha_lnorm_s = Alpha_Lnorm_stoch.apply
logminusexp = LogMinusExp.apply
logsumexpstoch = LogSumExpStoch.apply
'''Probabilistic Loss'''


def label_to_onehot( output: Tensor, label):
	onehot = output.new_zeros(output.size())
	onehot.scatter_(1, label,1)
	return onehot.float()


def posterior(lprob, label,alpha):
	assert lprob.ndimension() == 2
	label = label.unsqueeze(1)
	prob_label = label_to_onehot(lprob, label)
	prob_label += 1
	lprob_label = (prob_label / prob_label.sum(dim=1, keepdim=True)).log()

	lorgc = lprob_label - (lprob_label * alpha).logsumexp(dim=1, keepdim=True) / alpha
	# print(lprob_label)
	lmgc = lprob - (lprob * alpha).logsumexp(dim=1, keepdim=True) / alpha

	ldeltac = ((lmgc.exp() * (2 * lorgc.exp() - 1) + (1 - lorgc.exp())).mean(dim=1, keepdim=True)).log()
	loss = ldeltac.logsumexp(dim=1, keepdim=True)
	return loss


def sdc( lprob:Tensor, label,alpha):
		assert lprob.ndimension() == 2
		label = label.unsqueeze(1)
		prob_label = label_to_onehot(lprob,label)
		prob_label += 1
		lprob_label = (prob_label/prob_label.sum(dim=1,keepdim=True)).log()

		lorgc = lprob_label - (lprob_label*alpha).logsumexp(dim=1,keepdim=True)/alpha
		# print(lprob_label)
		lmgc = lprob - (lprob*alpha).logsumexp(dim=1,keepdim=True)/alpha

		ldeltac = ((lmgc.exp()*(2*lorgc.exp() -1)+ (1- lorgc.exp())).mean(dim=1,keepdim=True)).log()
		loss = ldeltac.logsumexp(dim=1,keepdim=True)
		return -loss



'''PMAPS'''

class glavgpool(Function):
	@staticmethod
	def forward(ctx, x):
		out = x.exp().mean(dim=(2,3),keepdim=True)
		#out = out.mean(dim=3, keepdim=True)
		out = out.clamp(definition.epsilon,None)
		out = out.log()
		return out,torch.zeros(1,1,dtype=out.dtype)