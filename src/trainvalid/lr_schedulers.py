import math
from typing import List
class lr_sched_from_list(object):
	def __init__(self,lr_list:List):
		self.lr_list = lr_list
		self.lr_size= lr_list.__len__()
	def __call__(self, epoch):
		epoch +=1
		return self.lr_list[epoch]

class constant_lr(object):
	''' every $step epochs the lr decreases in the exponent by $exp_decay_perstep'''
	def __init__(self,init_lr=1.0,**kwargs):
		self.init_lr= init_lr
	def __call__(self, epoch):

		return 1

class discrete_exp_decay_lr(object):
	''' every $step epochs the lr decreases in the exponent by $exp_decay_perstep'''
	def __init__(self,init_lr=1.0,step=10,exp_decay_perstep=1):
		self.init_lr= init_lr
		self.step= step
		self.exp_decay_perstep= exp_decay_perstep
	def __call__(self, epoch):
		epoch += 2
		num_steps = epoch // self.step
		coef = self.init_lr * math.exp(-abs(self.exp_decay_perstep)*num_steps)
		return coef

class exp_decay_lr(object):
	''' every $step epochs the lr decreases in the exponent by $exp_decay_perstep'''
	def __init__(self,init_lr=1,step=10,exp_decay_perstep=1):
		self.init_lr= init_lr
		self.step= step
		self.exp_decay_perstep= exp_decay_perstep
	def __call__(self, epoch):
		epoch += 1
		num_steps = epoch / self.step
		coef = self.init_lr * math.exp(-abs(self.exp_decay_perstep)*num_steps)
		return coef

def nin_caffe_lr( epoch):
	lr = [.1] +[2e-3] + [1e-2] + [2e-2] + 80 * [4e-2] + 10 * [4e-3] + 100 * [4e-4]
	epoch = epoch
	if epoch >= len(lr):
		epoch = len(lr)-1
	return lr[epoch]
def vgg_lr(epoch):
	# lr = .1*(.5**(epoch//25))
	lr = .1*(.5**(epoch//60))
	return lr
