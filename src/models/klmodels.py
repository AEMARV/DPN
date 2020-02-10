from src.optstructs import *
from typing import Tuple
from torch.nn.modules import NLLLoss
from src.trainvalid.lr_schedulers import *

''' Available Options for KLCONV('klconv'):
	r: receptive field,
	f: number of filters,
	pad: padding amount either 'same' or 'valid'
	bias: does it have bias (Bias not implemented yet)

	param: selects the parameterization methods including->

	 1. 'logstochu': log parameterization, stochastic gradient
	 and all dists are initialized to uniform
	 2. 'logstoch' : log parameterizaiton, stochastic gradient
	 3. 'log' : log parametrizaiton, exponential dists initialization  



	'''

def booltostr(tf):
	if tf:
		return '1'
	else:
		return '0'

def finite_quick_cifar(data_opts: DataOpts, isrelu=True,isnormstoch=False) -> Tuple[NetOpts, OptimOpts]:
	model_string = ''
	isrelu = booltostr(isrelu)
	isnormstoch = booltostr(isnormstoch)
	nl = 'lnorm|s:{}'.format(isnormstoch)
	convparam = 'param:logdirich,stoch:0,isrelu:{},coef:1'.format(isrelu)
	d = '->'
	finish = 'fin'
	model_string += 'klconvb|r:5,f:32,pad:same,bias:1,{}'.format(convparam) + d + nl + d
	model_string += 'klavgpool|r:3,f:32,pad:same,stride:2,bias:1' + d
	model_string += 'klconv|r:5,f:64,pad:same,bias:1,{}'.format(convparam) + d + nl + d
	model_string += 'klavgpool|r:3,f:32,pad:same,stride:2,bias:1' + d
	model_string += 'klconv|r:4,f:64,pad:same,bias:1,{}'.format(convparam) + d + nl + d
	model_string += 'klconv|r:7,f:' + str(data_opts.classnum) + ',pad:valid,bias:1,{}'.format(
		convparam) + d + nl + d
	model_string += finish

	'''Data OPTs'''
	data_transforms = []

	'''LR SCHED'''
	initlr = 1
	lr_sched = discrete_exp_decay_lr(init_lr=initlr, step=20, exp_decay_perstep=1)

	''' Net Options'''
	opts_net = NetOpts(model_string,
	                   inputspatszvalidator=lambda x: x == 32,
	                   data_transforms=data_transforms,
	                   classicNet=False,
	                   weightinit=lambda x: x.normal_(0, 0.05),
	                   biasinit=lambda x: x.zero_(),
	                   )

	'''Optimizer Options'''
	opts_optim = OptimOpts(lr=initlr,
	                       lr_sched_lambda=lr_sched,
	                       type='SGD',
	                       momentum=0.9,
	                       weight_decay=0,
	                       dampening=0,
	                       nestrov=False,
	                       loss=NLLLoss(reduce=False))

	return opts_net, opts_optim


def finite_nin_caffe(data_opts: DataOpts, isrelu=True,isnormstoch=False) -> Tuple[NetOpts, OptimOpts]:
	model_string = ''
	isrelu = booltostr(isrelu)
	isnormstoch = booltostr(isnormstoch)
	nl = 'lnorm|s:{}'.format(isnormstoch)
	convparam = 'param:logdirich,stoch:0,isrelu:{},coef:4'.format(isrelu)
	d = '->'
	finish = 'fin'
	model_string += 'klconvb|r:5,f:192,pad:same,bias:1,{}'.format(convparam) + d + nl + d
	model_string += 'klconv|r:1,f:160,pad:same,bias:1,{}'.format(convparam) + d + nl + d
	model_string += 'klconv|r:1,f:96,pad:same,bias:1,{}'.format(convparam) + d + nl + d
	model_string += 'klavgpool|r:3,f:32,pad:valid,stride:2,bias:1' + d
	# model_string += 'dropout|p:0.5' + d
	model_string += 'klconv|r:5,f:192,pad:same,bias:1,{}'.format(convparam) + d + nl + d
	model_string += 'klconv|r:1,f:192,pad:same,bias:1,{}'.format(convparam) + d + nl + d
	model_string += 'klconv|r:1,f:192,pad:same,bias:1,{}'.format(convparam) + d + nl + d
	model_string += 'klavgpool|r:3,f:32,pad:valid,stride:2,bias:1' + d
	# model_string += 'dropout|p:0.5' + d
	model_string += 'klconv|r:3,f:192,pad:same,bias:1,{}'.format(convparam) + d + nl + d
	model_string += 'klconv|r:1,f:192,pad:same,bias:1,{}'.format(convparam) + d + nl + d
	model_string += 'klconv|r:1,f:' + str(data_opts.classnum) + ',pad:valid,bias:0,{}'.format(convparam) + d + nl + d
	model_string += 'klavgpool|r:7,f:32,pad:valid,stride:2,bias:1' + d + nl + d

	model_string += finish

	'''Data OPTs'''
	data_transforms = []
	'''LR SCHED'''
	lr_sched = discrete_exp_decay_lr(init_lr=1, step=30, exp_decay_perstep=1)
	''' Net Options'''
	opts_net = NetOpts(model_string,
	                   inputspatszvalidator=lambda x: x == 32,
	                   data_transforms=data_transforms,
	                   classicNet=False,
	                   weightinit=lambda x: x.normal_(0, 0.05),
	                   biasinit=lambda x: x.zero_(),
	                   )
	opts_optim = OptimOpts(lr=1,
	                       lr_sched_lambda=lr_sched,
	                       type='SGD',
	                       momentum=0.9,
	                       weight_decay=0,
	                       dampening=0,
	                       nestrov=False,
	                       loss=NLLLoss(reduce=False)
	                       )
	'''Optimizer Options'''

	return opts_net, opts_optim


def finite_vgg(data_opts: DataOpts, isrelu=True,isnormstoch=False) -> Tuple[NetOpts, OptimOpts]:
	model_string = ''
	isrelu = booltostr(isrelu)
	isnormstoch = booltostr(isnormstoch)
	nl = 'lnorm|s:{}'.format(isnormstoch)
	convparam = 'param:logdirich,stoch:0,isrelu:{},coef:4'.format(isrelu)
	d = '->'
	finish = 'fin'
	model_string += 'klconvb|r:3,f:64,pad:same,bias:1,{}'.format(convparam) + d + nl + d  # + 'bn' + d + 'dropout|p:0.3' + d
	model_string += 'klconv|r:3,f:64,pad:same,bias:1,{}'.format(convparam) + d + nl + d
	model_string += 'klavgpool|r:2,pad:valid,stride:2,bias:1' + d
	model_string += 'klconv|r:3,f:128,pad:same,bias:1,{}'.format(convparam) + d + nl + d  # + 'bn' + d + 'dropout|p:0.4' + d
	model_string += 'klconv|r:3,f:128,pad:same,bias:1,{}'.format(convparam) + d + nl + d
	model_string += 'klavgpool|r:2,pad:valid,stride:2,bias:1' + d
	model_string += 'klconv|r:3,f:256,pad:same,bias:1,{}'.format(convparam) + d + nl + d  # + 'bn' + d + 'dropout|p:0.4' + d
	model_string += 'klconv|r:3,f:256,pad:same,bias:1,{}'.format(convparam) + d + nl + d  # + 'bn' + d + 'dropout|p:0.4' + d
	model_string += 'klconv|r:3,f:256,pad:same,bias:1,{}'.format(convparam) + d + nl + d
	model_string += 'klavgpool|r:2,pad:valid,stride:2,bias:1' + d
	model_string += 'klconv|r:3,f:512,pad:same,bias:1,{}'.format(convparam) + d + nl + d  # + 'bn' + d + 'dropout|p:0.4' + d
	model_string += 'klconv|r:3,f:512,pad:same,bias:1,{}'.format(convparam) + d + nl + d  # + 'bn' + d + 'dropout|p:0.4' + d
	model_string += 'klconv|r:3,f:512,pad:same,bias:1,{}'.format(convparam) + d + nl + d
	model_string += 'klavgpool|r:2,pad:valid,stride:2,bias:1' + d
	model_string += 'klconv|r:3,f:512,pad:same,bias:1,{}'.format(convparam) + d + nl + d  # + 'bn' + d + 'dropout|p:0.4' + d
	model_string += 'klconv|r:3,f:512,pad:same,bias:1,{}'.format(convparam) + d + nl + d  # + 'bn' + d + 'dropout|p:0.4' + d
	model_string += 'klconv|r:3,f:512,pad:same,bias:1,{}'.format(convparam) + d + nl + d  # + 'bn' + d
	model_string += 'klavgpool|r:2,pad:valid,stride:2,bias:1' + d
	model_string += 'klconv|r:1,f:512,pad:same,bias:1,{}'.format(convparam) + d + nl + d  # + 'dropout|p:0.5' + d
	model_string += 'klconv|r:1,f:' + str(data_opts.classnum) + ',pad:valid,bias:0,{}'.format(convparam) + d
	model_string += nl + d
	model_string += finish

	'''Data OPTs'''
	data_transforms = []
	'''LR SCHED'''
	lr_sched = discrete_exp_decay_lr(init_lr=1, step=30, exp_decay_perstep=1)

	''' Net Options'''
	opts_net = NetOpts(model_string,
	                   inputspatszvalidator=lambda x: x == 32,
	                   data_transforms=data_transforms,
	                   classicNet=False,
	                   weightinit=None,
	                   biasinit=None)
	'''Optimizer Options'''
	opts_optim = OptimOpts(lr=1,
	                       lr_sched_lambda=lr_sched,
	                       type='SGD',
	                       momentum=0.9,
	                       weight_decay=0,
	                       dampening=0,
	                       nestrov=False,
	                       loss=NLLLoss(reduce=False))
	return opts_net, opts_optim

