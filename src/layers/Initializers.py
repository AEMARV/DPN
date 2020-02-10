from torch import Tensor
from src.layers.klfunctions import *
from definition import epsilon
import torch
from typing import Tuple

''' Interface Class'''
class Parameterizer(object):
	def __init__(self,isstoch=False,isbinary=False,dtype=torch.float32, coef=1):
		self.isstoch = isstoch
		self.isbinary = isbinary
		self.coef = coef
		if self.isbinary:
			self.normaxis = 4
		else:
			self.normaxis = 1
		self.dtype = dtype
	def __call__(self, shape, dtype='float32', isbinary=False):
		raise("Outputs a ndarray with 5-th dim being the indepenent components")
	def get_log_kernel(self,k:Tensor):
		raise(Exception("This class is Abstract"))
	def get_prob_kernel(self,k:Tensor):
		raise (Exception("This class is Abstract"))
	def get_log_norm(self,k:Tensor):
		raise (Exception("This class is Abstract"))
	def get_prob_norm(self,k:Tensor):
		raise (Exception("This class is Abstract"))
	def projectKernel(self,k):
		return k

''' Dirichlet Inits'''
class LogParameter(Parameterizer):
	def __init__(self,
	             isstoch=True,
	             isuniform=True,
	             isdirichlet=True,
	             isbinary=False,
	             **kwargs):
		''' isuniform sets all the distributions to uniform
		and is equivalent to initializing with zeros in RealCNNS
		Obviously it is advised to turn on isstoch switch when uniform is on
		'''
		super(LogParameter,self).__init__(**kwargs)
		if isstoch:
			self.normfunc = LogSumExpStoch.apply
		else:
			self.normfunc = LogSumExp.apply
		self.isuniform = isuniform
		self.isbinary = isbinary
		self.isdirichlet= isdirichlet
		self.isuniform = isuniform

	def __call__(self, shape:Tuple,isbias=False)->Tensor:
		if self.isbinary:
			self.normaxis=4
			if not isbias:
				shape = shape + (2,)
			else:
				self.normaxis=1
		out = torch.empty(shape)
		out = -out.exponential_()
		if self.isdirichlet:
			out = out / out.sum(dim=self.normaxis,keepdim=True)
			out = out.clamp(epsilon, 1)
			out = out.log()#/math.log(out.shape[self.normaxis])
		if self.isuniform or isbias:
			out = out * 0

		out = out * self.coef
		return out.detach()

	def get_log_norm(self,k:Tensor)->Tensor:
		try:
			lognorm = self.normfunc(k,self.normaxis,1)
		except :

			print(k,end=" ")
		return lognorm
	def get_log_prior(self,k):
		return -(k.abs()).sum()
	def get_log_kernel(self,k:Tensor)->Tuple[Tensor,Tensor]:
		norm = k.logsumexp(dim=self.normaxis,keepdim=True)
		k =  k- k.logsumexp(dim=self.normaxis,keepdim=True)
		return k,norm
	def get_prob_kernel(self,k:Tensor)->Tensor:
		k = self.get_log_kernel(k)
		k = k.exp()
		return k
class PsuedoCount(Parameterizer):
	def __call__(self, shape:Tuple):
		if self.isbinary:
			self.normaxis=4
			shape = shape + (2,)
		out = torch.empty(shape)
		out = -out.exponential_()
		if self.isdirichlet:
			out = out / out.sum(dim=self.normaxis,keepdim=True)
			out = out.clamp(epsilon, 1)
			out = out.log()
		if self.isuniform:
			out = out * 0
		out = out * self.coef
		out = out.exp()
		return out
	def get_prob_kernel(self,k:Tensor):
		return k
	def projectKernel(self,k):
		k = k.detach()
		k = k.relu()
		k = k/k.sum(dim=self.normaxis,keepdim=True)
		return k
class LogParameterProjector(LogParameter):
	def __init__(self,*args,**kwargs):
		super(LogParameterProjector,self).__init__(*args,**kwargs)
	def get_log_kernel(self,k:Tensor):
		k = k - self.get_log_norm(k).detach()
		return k
class NormalParameter(Parameterizer):
	def __init__(self, **kwargs):
		super(NormalParameter, self).__init__(**kwargs)
	def __call__(self, shape,coef):
		out = torch.empty(shape) # type:Tensor
		out.normal_(0,1) # type: Tensor
		out =  out / ((out ** 2).sum(dim=self.normaxis, keepdim=True)).sqrt_()
		out = out * coef
		return out.detach_()
	def get_kernel(self,k:Tensor):
		return k / ((k ** 2).sum(dim=self.normaxis, keepdim=True)).sqrt_()


class SphereParameter(Parameterizer):
	def __init__(self,
	             isstoch=True,
	             isuniform=True,
	             isdirichlet=True,
	             isbinary=False,
	             **kwargs):
		''' isuniform sets all the distributions to uniform
		and is equivalent to initializing with zeros in RealCNNS
		Obviously it is advised to turn on isstoch switch when uniform is on
		'''
		super(SphereParameter,self).__init__(**kwargs)
		if isstoch:
			self.normfunc = LogSumExpStoch.apply
		else:
			self.normfunc = LogSumExp.apply
		self.isuniform = isuniform
		self.isbinary = isbinary
		self.isdirichlet= isdirichlet
		#if isuniform and not(isstoch):
		#	raise(Warning('Uniform Parameterizaiton, while being not stochastic'))
		self.isuniform = isuniform

	def __call__(self, shape:Tuple,isbias=False)->Tensor:
		if self.isbinary:
			self.normaxis=4
			shape = shape + (2,)
		out = torch.empty(shape)
		out = out.normal_(0,1)
		if self.isuniform:
			out = (out * 0) +1
		#out = out / ((out ** 2).sum(dim=self.normaxis, keepdim=True)).sqrt_()
		#out = out.clamp(epsilon, 1)
		if isbias:
			out = out
		out = out * self.coef
		# out = out/(out**2).sum(dim=self.normaxis,keepdim=True).sqrt()
		return out.detach()
	def get_log_prior(self,k):
		return -(k**2).sum()
	def get_log_norm(self,k:Tensor)->Tensor:
		prob = k**2 + definition.epsilon
		lognorm = prob.sum(dim=self.normaxis,keepdim=True).log()

		return lognorm
	#TODO NORMALIZE GETLOGKER
	def get_log_kernel(self,k:Tensor)->Tuple[Tensor]:
		prob = k**2 + definition.epsilon
		prob = prob/ prob.sum(dim=self.normaxis,keepdim=True)
		lprob = prob.log()
		#k = 2*((k.abs().clamp(epsilon,None)).log()) - self.get_log_norm(k)
		#return k[0:,0:,0:,0:,0], k[0:,0:,0:,0:,1]
		return lprob
	def get_prob_kernel(self,k:Tensor)->Tensor:
		#k = self.get_log_kernel(k)
		#k = k.exp()
		p = k**2

		return p/(p.sum(dim=self.normaxis,keepdim=True))






